import numpy as np
from shapely.geometry import Point

from tidescout.config import load_fishery
from tidescout.engine import detect
from tidescout.engine.terrain import slope_deg

from . import synth


def _thresholds():
    return load_fishery("winyah-bay").features


def test_gate_creek_mouth_found():
    z = synth.creek_mouth_dem()
    feats = detect.detect_creek_mouths(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    mouths = [f for f in feats if f.type == "creek_mouth"]
    assert mouths, "spec gate: idealized creek mouth must be detected"
    true_mouth = Point(synth.TRANSFORM * (99.5, 100.5))
    assert min(m.geometry.distance(true_mouth) for m in mouths) < 100.0  # within 100 m


def test_gate_point_bar_found():
    z = synth.point_bar_dem()
    feats = detect.detect_bars(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    assert feats, "spec gate: idealized bar must be detected"
    ridge_center = Point(synth.TRANSFORM * (100.5, 100.5))
    assert any(f.geometry.buffer(0).contains(ridge_center) for f in feats)
    assert all(60 <= f.attrs["orientation_deg"] <= 120 for f in feats)  # E-W ridge


def test_gate_dropoff_found_and_typed():
    z = synth.dropoff_dem()
    s = slope_deg(z, synth.CELL)
    feats = detect.detect_dropoffs(z, s, _thresholds(), synth.TRANSFORM)
    assert feats
    assert {f.type for f in feats} <= {"dropoff", "wall"}
    assert any(f.type == "wall" for f in feats)  # 9 m over 10 m cell = 24 deg


def test_gate_hole_found():
    z = synth.hole_dem()
    feats = detect.detect_holes(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    assert feats
    f = max(feats, key=lambda x: x.attrs["area_m2"])
    assert f.attrs["depth_below_rim_m"] >= 3.0
    assert f.attrs["min_z"] <= -9.0


def test_flat_detected_only_in_band():
    z = np.full((100, 100), -0.5, dtype="float32")  # in flat band
    s = slope_deg(z, synth.CELL)
    feats = detect.detect_flats(z, s, _thresholds(), synth.TRANSFORM)
    assert feats and all(f.type == "flat" for f in feats)
    z2 = np.full((100, 100), -6.0, dtype="float32")  # too deep
    assert detect.detect_flats(z2, slope_deg(z2, synth.CELL), _thresholds(), synth.TRANSFORM) == []


def test_no_features_on_empty_basin():
    z = synth.open_basin()
    s = slope_deg(z, synth.CELL)
    t = _thresholds()
    assert detect.detect_dropoffs(z, s, t, synth.TRANSFORM) == []
    assert detect.detect_holes(z, t, synth.CELL, synth.TRANSFORM) == []
    assert detect.detect_creek_mouths(z, t, synth.CELL, synth.TRANSFORM) == []
    assert detect.detect_bars(z, t, synth.CELL, synth.TRANSFORM) == []


def test_orientation_of_ew_line():
    from shapely.geometry import LineString

    assert abs(detect.orientation_deg(LineString([(0, 0), (100, 0)])) - 90.0) < 1.0


def test_creek_mouth_border_artifact_suppressed():
    """Regression (fix round 1, critical finding): a NaN border 'like the
    real raster' wrapped around an otherwise flat, fully-wet basin used to
    produce 4 spurious creek_mouth features -- one per corner rounded off
    by opening()'s disk footprint. No real feature exists anywhere in this
    DEM, bordered or not."""
    z = synth.border_nan(synth.open_basin())
    feats = detect.detect_creek_mouths(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    assert feats == []


def test_short_real_creek_found_despite_nan_border():
    """Regression (fix round 1, critical finding): the original fix for the
    border-corner artifact above filtered candidate mouths by skeleton
    *length* (>= open_radius px), which also silently dropped genuine short
    creeks -- a 60 m marsh feeder creek skeletonizes to only 2 px, the same
    ballpark as a 1-2 px corner-rounding stub. This creek sits in the grid
    interior, away from the NaN border, alongside the border's own corner
    artifacts elsewhere in the grid; the mouth must still be found."""
    z = synth.border_nan(synth.short_creek_mouth_dem(60.0))
    feats = detect.detect_creek_mouths(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    mouths = [f for f in feats if f.type == "creek_mouth"]
    assert len(mouths) == 1
    true_mouth = Point(synth.TRANSFORM * (99.5, 100.5))
    assert mouths[0].geometry.distance(true_mouth) < 100.0


def test_bar_touching_array_edge_not_boundary_biased():
    """Regression (fix round 1, important finding): binary_erosion's default
    border_value=0 treats space beyond the array edge as background, so a
    region pixel sitting on the array's own boundary always eroded into
    "boundary" regardless of whether real deep water was anywhere nearby --
    deflating pct_deep_boundary (verified: 0.9557 instead of the true 1.0
    for this exact ridge, an 18-pixel dilution from the west-edge column
    alone). With border_value=1, only real interior-to-region edges count."""
    z = synth.edge_touching_bar_dem()
    feats = detect.detect_bars(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    assert feats, "ridge touching the array edge must still be detected as a bar"
    assert all(f.attrs["pct_deep_boundary"] >= 0.99 for f in feats)
