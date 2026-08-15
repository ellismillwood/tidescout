import numpy as np
import pytest

from tidescout.config import load_fishery
from tidescout.pipeline import mesh

from . import synth
from .test_features_pipeline import _fake_bathy


def test_domain_mask_is_single_connected_component():
    z = np.full((200, 200), -5.0, dtype="float32")
    z[0:20, :] = 5.0                     # land strip
    z[150:160, 150:160] = -5.0           # isolated pond, must be dropped
    z[140:170, 60:70] = 5.0
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []  # synthetic raster has no overlap with real Winyah km
    m = mesh.domain_mask(z, synth.TRANSFORM, f, polygon=None)
    from scipy import ndimage
    _, n = ndimage.label(m)
    assert n == 1, "mesh domain must be exactly one connected water body"


def test_clean_mask_fills_small_hole_but_preserves_large_one():
    mask = np.ones((200, 200), dtype=bool)
    mask[20:25, 20:25] = False      # 25 cells = 2,500 m2 = 0.0025 km2, below threshold
    mask[100:130, 100:130] = False  # 900 cells = 90,000 m2 = 0.09 km2, above threshold
    # cells=1 makes the closing/opening structuring element a single pixel, a
    # no-op, so only the hole-fill-threshold logic under test can move a pixel.
    cleaned = mesh.clean_mask(mask, cells=1, min_island_hole_km2=0.05, pixel_area_m2=100.0)
    assert cleaned[22, 22], "small island must be filled as sub-mesh-scale noise"
    assert not cleaned[115, 115], "large island must be preserved as a mesh hole"


def test_domain_polygon_simplifies_hard_but_keeps_area():
    z = np.full((200, 200), -5.0, dtype="float32")
    # rough up the shoreline so simplification has something to do
    z[0:30, :] = 5.0
    z[30, ::2] = 5.0
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []  # synthetic raster has no overlap with real Winyah km
    m = mesh.domain_mask(z, synth.TRANSFORM, f, polygon=None)
    poly = mesh.domain_polygon(m, synth.TRANSFORM, simplify_m=25.0)
    assert poly.is_valid
    assert len(poly.exterior.coords) < 400
    assert poly.area > 0.9 * m.sum() * 100.0   # 10 m cells = 100 m2 each


def test_build_mesh_sets_elevation_on_every_centroid(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[0:40, :] = 5.0
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []      # empty => use the whole water mask
    # Synthetic raster lives at unrelated coordinates from the real Winyah
    # ocean_boundary_utm_km polygon, so clear it too -- otherwise no deep
    # segment would ever fall inside it and build_mesh would raise.
    f.model_domain.ocean_boundary_utm_km = []
    d = mesh.build_mesh("winyah-bay", f)
    elev = d.get_quantity("elevation").get_values(location="centroids")
    assert len(elev) == len(d.triangles)
    assert np.isfinite(elev).all(), "no NaN may reach the solver"
    assert elev.min() < 0.0


def test_build_mesh_carves_interior_holes_for_large_islands(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[120:180, 120:180] = 5.0  # 60x60 cells = 600x600 m = 0.36 km2, well interior
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    f.model_domain.ocean_boundary_utm_km = []  # see rationale in the elevation test above
    d = mesh.build_mesh("winyah-bay", f)

    cx, cy = d.get_centroid_coordinates(absolute=True).T
    cols, rows = ~synth.TRANSFORM * (cx, cy)
    # Margin in from the raw 120..180 island bounds so ring-simplification at
    # domain_polygon's 25 m simplify_m can't produce a false failure right at
    # the edge -- only the deep interior needs to stay uncovered.
    inside = (rows > 130) & (rows < 170) & (cols > 130) & (cols < 170)
    assert not inside.any(), "no triangle centroid may fall inside a large island hole"

    # Same raster, but the hole threshold is set high enough that this island
    # gets filled instead of carved out -- the hole variant must mesh fewer
    # triangles than the filled one.
    f_filled = load_fishery("winyah-bay")
    f_filled.model_domain.polygon_utm_km = []
    f_filled.model_domain.ocean_boundary_utm_km = []
    f_filled.model_domain.min_island_hole_km2 = 1000.0
    d_filled = mesh.build_mesh("winyah-bay", f_filled)
    assert len(d.triangles) < len(d_filled.triangles)


def test_classify_boundary_splits_ocean_from_wall():
    z = np.full((50, 50), -5.0, dtype="float32")
    z[0:5, :] = 2.0  # land/shallow strip along the north edge
    ring = [
        [500000.0, 3699510.0],  # SW
        [500490.0, 3699510.0],  # SE
        [500490.0, 3699995.0],  # NE, over the shallow strip
        [500000.0, 3699995.0],  # NW, over the shallow strip
    ]
    ocean_idx, wall_idx, open_idx = mesh.classify_boundary(
        ring, z, synth.TRANSFORM, ocean_max_z_m=-2.0
    )
    assert 0 in ocean_idx, "south segment sits over deep water, must be ocean"
    assert 2 in wall_idx, "north segment sits over the shallow strip, must be wall"
    assert open_idx == [], "empty ocean_boundary_utm_km must reproduce the old two-class split"


def test_classify_boundary_with_ocean_polygon_splits_three_ways():
    """A deep segment inside the authored ocean polygon is `ocean`; a deep
    segment outside it is a severed inland channel (`open`), not ocean or
    wall; a shallow segment is `wall` regardless of the polygon."""
    z = np.full((50, 50), -5.0, dtype="float32")
    z[0:5, :] = 2.0  # land/shallow strip along the north edge
    ring = [
        [500000.0, 3699510.0],  # SW -- deep, midpoint (500.245, 3699.510) km
        [500490.0, 3699510.0],  # SE -- east edge, deep but off the polygon
        [500490.0, 3699995.0],  # NE -- over the shallow strip
        [500000.0, 3699995.0],  # NW -- west edge, deep but off the polygon
    ]
    # Encloses only the south segment's midpoint.
    ocean_poly_km = [
        (499.9, 3699.4), (500.6, 3699.4), (500.6, 3699.6), (499.9, 3699.6),
    ]
    ocean_idx, wall_idx, open_idx = mesh.classify_boundary(
        ring, z, synth.TRANSFORM, ocean_max_z_m=-2.0, ocean_boundary_utm_km=ocean_poly_km
    )
    assert ocean_idx == [0], "south segment is deep and inside the ocean polygon"
    assert set(open_idx) == {1, 3}, (
        "east/west segments are deep but outside the ocean polygon -- a severed "
        "inland channel, not the seaward opening"
    )
    assert wall_idx == [2], "north segment is shallow, must stay wall regardless of the polygon"


def test_build_mesh_boundary_tags_are_ocean_and_wall_minority(tmp_path, monkeypatch):
    z = np.full((300, 300), -1.0, dtype="float32")  # shallow basin -> wall by default
    z[290:300, :] = -5.0                             # deep water along the south edge only
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    f.model_domain.ocean_boundary_utm_km = []  # see rationale in the elevation test above
    d = mesh.build_mesh("winyah-bay", f)
    tags = set(d.boundary.values())
    assert tags == {"ocean", "wall"}
    n_ocean = sum(1 for t in d.boundary.values() if t == "ocean")
    n_wall = sum(1 for t in d.boundary.values() if t == "wall")
    assert 0 < n_ocean < n_wall, "ocean must be present but a minority of the boundary"


def test_build_mesh_raises_when_no_ocean_segment_qualifies(tmp_path, monkeypatch):
    """Nothing is deep enough anywhere -- distinct from the polygon-excludes-
    deep-segments case below. The error must say so: 0 segments are deep
    enough (regardless of the polygon), not that the polygon is the problem."""
    z = np.full((300, 300), -1.0, dtype="float32")  # nowhere deep enough for the tide
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    f.model_domain.ocean_boundary_utm_km = []
    f.model_domain.ocean_max_z_m = -999.0
    with pytest.raises(ValueError, match="no boundary segment") as excinfo:
        mesh.build_mesh("winyah-bay", f)
    assert "0 segment(s) are deep enough" in str(excinfo.value), (
        "message must name the 'nothing is deep enough' cause, not the polygon"
    )


def test_build_mesh_raises_when_ocean_boundary_polygon_excludes_all_deep_segments(
    tmp_path, monkeypatch
):
    """Even with a genuinely deep segment, an ocean_boundary_utm_km polygon
    that doesn't cover it must still raise -- the segment becomes `open`, not
    `ocean`, and the tide would have nowhere authored to enter. Distinct from
    the no-deep-segment-anywhere case above: the error must report a nonzero
    count of segments that ARE deep enough but fall outside the polygon, so
    it points at the polygon rather than the bathymetry."""
    z = np.full((300, 300), -1.0, dtype="float32")  # shallow basin
    z[290:300, :] = -5.0                             # deep water along the south edge only
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    # Real Winyah polygon -- nowhere near the synthetic raster's coordinates,
    # so it excludes every deep segment this raster can produce.
    f.model_domain.ocean_boundary_utm_km = [(665.0, 3669.0), (676.0, 3669.0), (676.0, 3695.0)]
    with pytest.raises(ValueError, match="no boundary segment") as excinfo:
        mesh.build_mesh("winyah-bay", f)
    msg = str(excinfo.value)
    assert "segment(s) are deep enough" in msg
    assert "0 segment(s) are deep enough" not in msg, (
        "the deep south segment(s) exist and were excluded by the polygon, "
        "not absent -- the message must not claim the zero-case"
    )


def test_friction_field_has_no_zero_values(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[0:40, :] = 5.0
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    f.model_domain.ocean_boundary_utm_km = []  # see rationale in the elevation test above
    from tidescout.pipeline.derivatives import build_derivatives
    build_derivatives("winyah-bay", f)
    d = mesh.build_mesh("winyah-bay", f)
    n = mesh.friction_field(d, "winyah-bay", f)
    assert len(n) == len(d.triangles)
    assert (n > 0).all(), "a zero Manning n is frictionless, not a default"
    assert n.max() <= 0.1
