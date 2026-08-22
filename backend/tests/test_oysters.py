"""Reef proximity on hand-placed geometry."""

import json
import logging

import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping

from tidescout import paths
from tidescout.pipeline.oysters import (
    load_reefs_utm,
    nearest_reef_m,
    reef_area_m2_within,
    reef_to_utm,
)


def _square(cx, cy, side):
    h = side / 2.0
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def test_reef_area_counts_only_reefs_inside_the_radius():
    reefs = [_square(0.0, 0.0, 10.0), _square(500.0, 0.0, 10.0)]
    got = reef_area_m2_within([Point(0.0, 0.0)], reefs, radius_m=100.0)
    assert got[0] == pytest.approx(100.0)


def test_reef_area_sums_multiple_nearby_reefs():
    reefs = [_square(10.0, 0.0, 10.0), _square(-10.0, 0.0, 10.0), _square(0.0, 20.0, 10.0)]
    got = reef_area_m2_within([Point(0.0, 0.0)], reefs, radius_m=100.0)
    assert got[0] == pytest.approx(300.0)


def test_a_feature_with_no_reefs_nearby_gets_zero_not_nan():
    """Zero reef is a real, common, meaningful answer -- most of the bay. NaN
    would make Phase 3 exclude the factor and renormalise, turning 'no oysters
    here' into 'we do not know', which are opposite statements."""
    got = reef_area_m2_within([Point(0.0, 0.0)], [_square(9999.0, 9999.0, 10.0)], 100.0)
    assert got[0] == 0.0


def test_nearest_reef_distance_is_zero_when_the_feature_sits_on_one():
    reefs = [_square(0.0, 0.0, 40.0)]
    assert nearest_reef_m([Point(5.0, 5.0)], reefs)[0] == pytest.approx(0.0)


def test_nearest_reef_distance_is_inf_when_there_are_no_reefs():
    import math

    assert math.isinf(nearest_reef_m([Point(0.0, 0.0)], [])[0])


def test_reef_lookup_is_correct_at_the_full_layer_size():
    """8,451 reefs x thousands of features approximates the real inventory's
    scale. This does not assert performance -- a wall-clock/timing assertion
    here would be flaky, and the STRtree requirement is enforced by code
    review, not by this test. What it asserts is that the *result* stays
    correct once the reef count is large: each of the 500 probe points sees
    exactly its 3 neighbouring 8x8 m (64 m^2) reefs, at offsets -30 m, 0 m and
    +30 m, all inside the 50 m buffer radius, for a total of 3 x 64 = 192.0
    m^2 -- except point 0, whose -30 m neighbour does not exist (reef index
    -1 is out of range), leaving only 2 neighbours and 128.0 m^2."""
    reefs = [_square(float(i) * 30.0, 0.0, 8.0) for i in range(3000)]
    pts = [Point(float(i) * 30.0, 0.0) for i in range(500)]
    got = reef_area_m2_within(pts, reefs, radius_m=50.0)
    assert len(got) == 500
    assert got[0] == pytest.approx(128.0)
    assert all(g == pytest.approx(192.0) for g in got[1:])


def test_reef_area_and_nearest_for_a_linestring_feature():
    """The real inventory is dominated by Polygon (2,026) and has 2
    LineString features -- Point (134) is the minority every other test here
    uses. A line 10 m from a reef's nearest edge (perpendicular distance from
    y=15 to the reef's edge at y=5) must report that exact distance, and a
    radius wide enough to fully reach the reef must count its whole area."""
    line = LineString([(-50.0, 15.0), (50.0, 15.0)])
    reef = _square(0.0, 0.0, 10.0)
    assert nearest_reef_m([line], [reef])[0] == pytest.approx(10.0)
    got = reef_area_m2_within([line], [reef], radius_m=25.0)
    assert got[0] == pytest.approx(100.0)


def test_reef_area_and_nearest_for_a_polygon_feature_containing_a_reef():
    """2,026 of the real inventory's features are Polygon -- a drop-off or bar
    polygon that already contains a reef must count that reef's full area
    (not just a sliver near the boundary) and report distance zero. This pins
    two semantics for Polygon features specifically: buffering a Polygon by
    the radius includes reef that is already inside it, and a feature
    containing a reef has nearest_reef_m == 0.0, same as the Point case
    above."""
    feature_poly = _square(0.0, 0.0, 40.0)
    reef = _square(0.0, 0.0, 10.0)
    assert nearest_reef_m([feature_poly], [reef])[0] == 0.0
    got = reef_area_m2_within([feature_poly], [reef], radius_m=5.0)
    assert got[0] == pytest.approx(100.0)


def _lonlat_square(clon, clat, side_deg):
    h = side_deg / 2.0
    return Polygon(
        [
            (clon - h, clat - h),
            (clon + h, clat - h),
            (clon + h, clat + h),
            (clon - h, clat + h),
        ]
    )


def test_reef_to_utm_reprojects_a_polygon_reef():
    poly = _lonlat_square(-79.06, 33.52, 0.001)
    utm = reef_to_utm(poly, 26917)
    assert isinstance(utm, Polygon)
    # UTM 17N eastings/northings run ~1e5-1e6 / ~1e6-1e7; if reprojection
    # silently no-opped this would still look like lon/lat-scale numbers.
    minx, miny, maxx, maxy = utm.bounds
    assert 1e5 < minx < maxx < 1e6
    assert 1e6 < miny < maxy < 1e7
    assert utm.area > 0


def test_reef_to_utm_reprojects_a_multipolygon_reef():
    """42 of the 8,451 real SCDNR reefs are MultiPolygon -- one reef id traced
    as disjoint patches. features.py's `_to4326` raises TypeError on
    MultiPolygon (and runs the opposite direction besides), so this reef
    loader has its own reprojection with explicit MultiPolygon coverage."""
    multi = MultiPolygon(
        [
            _lonlat_square(-79.06, 33.52, 0.001),
            _lonlat_square(-79.05, 33.53, 0.001),
        ]
    )
    utm = reef_to_utm(multi, 26917)
    assert isinstance(utm, MultiPolygon)
    assert len(utm.geoms) == 2
    assert all(g.area > 0 for g in utm.geoms)
    minx, miny, maxx, maxy = utm.bounds
    assert 1e5 < minx < maxx < 1e6
    assert 1e6 < miny < maxy < 1e7


def test_load_reefs_utm_returns_empty_list_and_logs_when_the_layer_is_missing(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    with caplog.at_level(logging.WARNING):
        reefs = load_reefs_utm("winyah-bay", 26917)
    assert reefs == []
    assert "oyster" in caplog.text.lower()


def test_load_reefs_utm_reads_and_reprojects_mixed_polygon_and_multipolygon(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(_lonlat_square(-79.06, 33.52, 0.001)),
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(
                    MultiPolygon(
                        [
                            _lonlat_square(-79.05, 33.53, 0.001),
                            _lonlat_square(-79.04, 33.54, 0.001),
                        ]
                    )
                ),
            },
        ],
    }
    (d / "oyster_reefs.geojson").write_text(json.dumps(fc))
    reefs = load_reefs_utm("winyah-bay", 26917)
    assert len(reefs) == 2
    assert isinstance(reefs[0], Polygon)
    assert isinstance(reefs[1], MultiPolygon)
    for g in reefs:
        minx, miny, maxx, maxy = g.bounds
        assert 1e5 < minx < 1e6
        assert 1e6 < miny < 1e7
