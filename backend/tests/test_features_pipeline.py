import json

import numpy as np

from tidescout.config import load_fishery
from tidescout.pipeline.features import build_features, load_features

from . import synth


def _fake_bathy(tmp_path, monkeypatch, z):
    from tidescout import paths
    from tidescout.pipeline.derivatives import _write

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    t = synth.TRANSFORM
    meta = {
        "crs": "EPSG:26917",
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "width": z.shape[1], "height": z.shape[0],
        "stats": {"min": float(np.nanmin(z)), "max": float(np.nanmax(z)), "pct_nodata": 0.0},
    }
    (d / "bathy_meta.json").write_text(json.dumps(meta))
    _write(d / "bathy_utm.tif", np.nan_to_num(z, nan=-9999.0), meta, "float32", -9999.0)


def test_build_features_on_synthetic(tmp_path, monkeypatch):
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    out = build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    assert out.name == "features.geojson"
    types = {feat["properties"]["type"] for feat in fc["features"]}
    assert "creek_mouth" in types
    assert "jetty" in types  # seeds always present
    for feat in fc["features"]:
        gtype = feat["geometry"]["type"]
        coords = feat["geometry"]["coordinates"]
        flat = (
            [coords] if gtype == "Point"
            else coords if gtype == "LineString"
            else coords[0]
        )
        for lon, lat in flat:
            assert -180 <= lon <= 180 and -90 <= lat <= 90


def test_feature_ids_unique_and_prefixed_by_type(tmp_path, monkeypatch):
    """Interface contract from the brief: id is "<type>-<n>", 1-indexed per type."""
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    ids = [feat["id"] for feat in fc["features"]]
    assert len(ids) == len(set(ids))
    for feat in fc["features"]:
        assert feat["id"].rsplit("-", 1)[0] == feat["properties"]["type"]


def test_build_features_rounds_float_attrs_to_2dp(tmp_path, monkeypatch):
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    for feat in fc["features"]:
        for v in feat["properties"].values():
            if isinstance(v, float):
                assert v == round(v, 2)


def test_load_features_reads_back_what_build_wrote(tmp_path, monkeypatch):
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    out = build_features("winyah-bay", f)
    on_disk = json.loads(out.read_text())
    assert load_features("winyah-bay") == on_disk
