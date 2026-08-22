import json

import numpy as np
import pytest
from rasterio.warp import transform as warp_transform
from shapely.geometry import Polygon, mapping, shape

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


def test_build_features_defaults_oyster_attrs_and_writes_json_safe_null_for_missing_reef_layer(
    tmp_path, monkeypatch
):
    """No oyster_reefs.geojson on disk is the documented "SCDNR layer
    unavailable" contingency, not an error. Every feature must get
    oyster_area_m2 = 0.0 (a real "no reef nearby" answer, not NaN) and
    oyster_nearest_m = null -- NOT the raw `Infinity` token nearest_reef_m
    returns internally, which is invalid JSON per RFC 8259 and would blow up
    the frontend's JSON.parse. This is the reachable worst case: with no reef
    layer at all, *every* feature hits this path at once."""
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    out = build_features("winyah-bay", f)
    raw = out.read_text()
    assert "Infinity" not in raw
    fc = json.loads(raw)
    assert fc["features"], "must have detected some synthetic features"
    for feat in fc["features"]:
        assert feat["properties"]["oyster_area_m2"] == 0.0
        assert feat["properties"]["oyster_nearest_m"] is None


def test_build_features_attaches_positive_oyster_area_when_reef_layer_present(
    tmp_path, monkeypatch
):
    """Every other oyster test either has no reef layer at all, or checks
    load_reefs_utm/reef_area_m2_within/nearest_reef_m directly -- nothing
    exercises build_features end-to-end with a reef layer actually present,
    so a wrong epsg threaded into load_reefs_utm, or the oyster block
    drifting to after _to4326 (reef and feature no longer sharing a CRS
    when distances are computed), would be caught by nothing in CI. Only a
    manual `tidescout features winyah-bay --rebuild` over gitignored data/
    would ever surface it.

    Places two small reefs, in the reef layer's real EPSG:4326, directly on
    top of the two seeded jetty vertices. Jetty geometry is seeded straight
    from fisheries/winyah-bay.yaml's real lon/lat (independent of the
    synthetic DEM), so this is a deterministic, hand-placed check, not a
    hopeful one: get the CRS handling wrong in either direction and the
    reef lands nowhere near the jetty, and oyster_area_m2 stays 0.0."""
    from tidescout import paths

    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")

    def _tiny_square(clon, clat, side_deg=0.0005):
        h = side_deg / 2.0
        return Polygon(
            [
                (clon - h, clat - h),
                (clon + h, clat - h),
                (clon + h, clat + h),
                (clon - h, clat + h),
            ]
        )

    reef_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(_tiny_square(*jetty.coords[0])),
            }
            for jetty in f.jetties
        ],
    }
    (paths.fishery_data_dir("winyah-bay") / "oyster_reefs.geojson").write_text(
        json.dumps(reef_fc)
    )

    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    jetties = [feat for feat in fc["features"] if feat["properties"]["type"] == "jetty"]
    assert len(jetties) == len(f.jetties)
    for feat in jetties:
        assert feat["properties"]["oyster_area_m2"] > 0
        assert feat["properties"]["oyster_nearest_m"] == pytest.approx(0.0)


def test_a_small_oyster_density_survives_the_props_rounding(tmp_path, monkeypatch):
    """The regression this pins: `oyster_density` used to go through the same
    blanket 2 dp rounding as every other float property, and 25 of the 146
    reef-carrying features in the real inventory (17%) landed on exactly 0.0 --
    "no reef at all" -- while the whole 2,162-feature inventory collapsed to 13
    distinct values. The field exists only so Phase 3 can divide out the extent
    confound in `oyster_area_m2`; a value that reads 0.0 for a feature with reef
    on it cannot do that.

    A ~2 m reef square on each jetty vertex gives a density far below 0.005, so
    `round(density, 2)` is exactly 0.0 and the assertion below fails outright
    if the exemption is removed."""
    from tidescout import paths

    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")

    def _tiny_square(clon, clat, side_deg=0.00002):
        h = side_deg / 2.0
        return Polygon(
            [
                (clon - h, clat - h),
                (clon + h, clat - h),
                (clon + h, clat + h),
                (clon - h, clat + h),
            ]
        )

    (paths.fishery_data_dir("winyah-bay") / "oyster_reefs.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": mapping(_tiny_square(*jetty.coords[0]))}
                for jetty in f.jetties
            ],
        })
    )

    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    jetties = [feat for feat in fc["features"] if feat["properties"]["type"] == "jetty"]
    assert jetties
    for feat in jetties:
        density = feat["properties"]["oyster_density"]
        assert feat["properties"]["oyster_area_m2"] > 0.0
        assert round(density, 2) == 0.0, "fixture must sit below the 2 dp floor"
        assert density > 0.0, "2 dp rounding would have erased this reef entirely"


def test_feature_ids_unique_and_prefixed_by_type(tmp_path, monkeypatch):
    """Interface contract: id is "<type>-<12 hex chars>", a hash of the type
    plus quantised centroid, stable across rebuilds."""
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    ids = [feat["id"] for feat in fc["features"]]
    assert len(ids) == len(set(ids))
    for feat in fc["features"]:
        assert feat["id"].rsplit("-", 1)[0] == feat["properties"]["type"]


def test_build_features_rounds_float_attrs_to_2dp_except_oyster_density(tmp_path, monkeypatch):
    """Metre and square-metre properties round to 2 dp; `oyster_density` is
    exempt because it is a dimensionless fraction whose whole useful range
    (~0.001-0.14 on the real inventory) sits inside the first two decimals --
    see `features._PROP_DECIMALS`. The exemption is asserted here rather than
    contradicted: the previous version of this test required *every* float to
    equal round(v, 2), which pinned the destruction of `oyster_density` as
    intended behaviour."""
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    for feat in fc["features"]:
        for k, v in feat["properties"].items():
            if not isinstance(v, float):
                continue
            if k == "oyster_density":
                assert v == round(v, 6)
            else:
                assert v == round(v, 2)


def test_load_features_reads_back_what_build_wrote(tmp_path, monkeypatch):
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    out = build_features("winyah-bay", f)
    on_disk = json.loads(out.read_text())
    assert load_features("winyah-bay") == on_disk


def test_rebuilt_features_keep_their_ids(tmp_path, monkeypatch):
    """The carryover's trap (c): `bar-78` renumbered on every rebuild."""
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")

    first = json.loads(build_features("winyah-bay", f).read_text())
    second = json.loads(build_features("winyah-bay", f).read_text())

    ids_first = sorted(feat["id"] for feat in first["features"])
    ids_second = sorted(feat["id"] for feat in second["features"])
    assert ids_first == ids_second
    assert len(set(ids_first)) == len(ids_first), "feature ids must be unique"


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
