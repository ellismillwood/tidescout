import json

import numpy as np
import pytest
from rasterio.warp import transform as warp_transform
from shapely.geometry import Polygon, shape

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


def _independent_to_utm(poly4326, epsg):
    """Reproject a lon/lat Polygon (with interiors) back to UTM, computed
    from scratch in the test rather than by reusing features.py's `_to4326`
    or cli.py's `to_utm_geom` -- so this check doesn't just confirm the fix
    agrees with itself."""

    def tx(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        us, vs = warp_transform("EPSG:4326", f"EPSG:{epsg}", xs, ys)
        return list(zip(us, vs, strict=True))

    return Polygon(
        tx(list(poly4326.exterior.coords)),
        [tx(list(r.coords)) for r in poly4326.interiors],
    )


def test_donut_bar_polygon_interior_ring_survives_export(tmp_path, monkeypatch):
    """Regression: a bar that fully encircles a deep pocket rasterizes to a
    polygon with an interior ring (a donut). If `_to4326` drops interior
    rings on export (as it used to), the written GeoJSON polygon becomes a
    filled blob that claims the hole's area as part of the bar -- exactly
    the bar-78 bug found on real Winyah Bay data (attrs.area_m2 1,324,200 m2
    vs an exported ring enclosing ~3,116,957 m2 with the hole filled in).

    Assert the exported polygon keeps both rings, and that its ring-enclosed
    area (reconstructed back to the UTM analysis CRS, exterior minus hole)
    reconciles with the area_m2 attribute computed upstream in engine.detect
    from the original hole-bearing UTM geometry.
    """
    z = synth.donut_bar_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    bars = [feat for feat in fc["features"] if feat["properties"]["type"] == "bar"]
    assert bars, "synthetic donut bar must be detected"
    donut = max(bars, key=lambda ft: len(ft["geometry"]["coordinates"]))
    assert donut["geometry"]["type"] == "Polygon"
    rings = donut["geometry"]["coordinates"]
    assert len(rings) == 2, "exported polygon must keep exterior + interior ring"

    poly4326 = shape(donut["geometry"])
    assert len(poly4326.interiors) == 1

    poly_utm = _independent_to_utm(poly4326, f.bathymetry.epsg)
    assert poly_utm.area == pytest.approx(donut["properties"]["area_m2"], rel=0.01)
