"""Geodesic distance on hand-built channel geometry.

Straight-line distance is the wrong answer everywhere in an estuary, so these
fixtures are built so the two answers differ measurably.
"""

import numpy as np
import pytest
from affine import Affine

from tidescout.engine.structure import to_grid
from tidescout.pipeline import estuary


class _Spec:
    """A 20x20 grid of 100 m cells, with an explicit in-domain mask."""

    def __init__(self, mask):
        self.shape = mask.shape
        self.cell_m = 100.0
        self.transform = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 2000.0)
        rows, cols = np.nonzero(mask)
        self.flat_index = np.ravel_multi_index((rows, cols), mask.shape)
        self.xs, self.ys = self.transform * (cols + 0.5, rows + 0.5)


def test_distance_grows_along_a_straight_channel():
    mask = np.zeros((20, 20), bool)
    mask[10, :] = True                       # one east-west channel
    spec = _Spec(mask)
    seeds = spec.xs <= 100.0                 # the westernmost cell is the sea

    d = estuary.along_estuary_km(spec, seed_mask=seeds)

    order = np.argsort(spec.xs)
    assert d[order][0] == pytest.approx(0.0)
    assert np.all(np.diff(d[order]) > 0), "distance must increase away from the sea"
    assert d[order][-1] == pytest.approx(1.9, abs=0.05)  # 19 cells x 100 m


def test_distance_follows_water_around_a_barrier_not_through_it():
    """The whole point: a U-shaped channel puts the far end 100 m away in a
    straight line and ~2 km away through water."""
    mask = np.zeros((20, 20), bool)
    mask[5, 2:18] = True     # north leg
    mask[5:15, 17] = True    # east connector
    mask[14, 2:18] = True    # south leg, ending beside the start
    spec = _Spec(mask)
    seeds = (spec.ys > 1400.0) & (spec.xs < 300.0)   # west end of the north leg

    d = estuary.along_estuary_km(spec, seed_mask=seeds)

    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    far = (rows == 14) & (cols == 2)          # 900 m south in a straight line
    assert d[far][0] > 3.0, "must route the long way around, not across land"


def test_unreachable_water_is_nan_not_zero():
    """An isolated pond has no route to the sea. Zero would read as 'at the
    mouth', which is maximally salty -- the most wrong answer available."""
    mask = np.zeros((20, 20), bool)
    mask[10, 0:5] = True
    mask[2, 15:19] = True       # disconnected
    spec = _Spec(mask)
    seeds = spec.xs <= 100.0

    d = estuary.along_estuary_km(spec, seed_mask=seeds)
    rows, _ = np.unravel_index(spec.flat_index, spec.shape)
    assert np.all(np.isnan(d[rows == 2]))
    assert np.all(np.isfinite(d[rows == 10]))


def test_diagonal_steps_cost_more_than_orthogonal_ones():
    """8-connectivity with equal weights would make a diagonal channel read
    30% shorter than it is."""
    mask = np.zeros((20, 20), bool)
    for i in range(10):
        mask[i, i] = True
    spec = _Spec(mask)
    seeds = (spec.xs < 100.0) & (spec.ys > 1900.0)

    d = estuary.along_estuary_km(spec, seed_mask=seeds)
    assert np.nanmax(d) == pytest.approx(9 * 100.0 * np.sqrt(2) / 1000.0, rel=0.02)


# A polygon covering the whole 20x20 grid (spans 0-2 km on each axis in the
# _Spec fixture's transform), so `ocean_seed_mask`'s polygon-containment test
# always passes -- these two tests isolate the edge and depth criteria on
# top of it.
_WHOLE_GRID_POLYGON_KM = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]


def test_ocean_seed_mask_excludes_interior_water_inside_the_polygon():
    """Polygon-containment alone is not enough: a block of open, deep water
    entirely inside the authored polygon is not itself 'the sea' unless it
    also touches the domain's outer edge. This is the defect that made
    Georgetown Lighthouse read 0.00 km -- containment alone swallowed
    interior estuary, not just the true mouth."""
    mask = np.zeros((20, 20), bool)
    mask[5:10, 5:10] = True  # a solid 5x5 block of open water
    spec = _Spec(mask)
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    bed_elev_m = np.full(spec.flat_index.size, -5.0)  # uniformly deep

    seeds = estuary.ocean_seed_mask(
        spec, _WHOLE_GRID_POLYGON_KM, bed_elev_m, ocean_max_z_m=-2.0
    )

    interior = (rows == 7) & (cols == 7)  # centre cell, all 4 neighbours wet
    edge = (rows == 5) & (cols == 7)  # top row of the block, has a dry neighbour
    assert not seeds[interior][0], "interior water must not seed even inside the polygon"
    assert seeds[edge][0], "a deep edge cell inside the polygon must seed"


def test_ocean_seed_mask_excludes_shallow_edge_cells_inside_the_polygon():
    """Depth is also necessary, not just edge-and-polygon: a shallow edge
    cell is a shoreline touch point, not open sea, even where the domain
    boundary and the polygon agree."""
    mask = np.zeros((20, 20), bool)
    mask[10, 5:10] = True  # a channel one cell wide -- every cell is an edge cell
    spec = _Spec(mask)
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    bed_elev_m = np.where(cols <= 6, -5.0, -1.0)  # west half deep, east half shallow

    seeds = estuary.ocean_seed_mask(
        spec, _WHOLE_GRID_POLYGON_KM, bed_elev_m, ocean_max_z_m=-2.0
    )

    deep_edge = cols == 5
    shallow_edge = cols == 9
    assert seeds[deep_edge][0], "a deep edge cell inside the polygon must seed"
    assert not seeds[shallow_edge][0], "a shallow edge cell must not seed even on the edge"


def test_ocean_seed_mask_keeps_only_the_largest_contiguous_run():
    """Edge-and-deep-and-inside-the-polygon still admits scattered shoreline
    fragments alongside the true mouth -- a small detached run must not
    seed even though it individually passes every other test, because the
    real seaward opening is one contiguous stretch of coast, not several."""
    mask = np.zeros((20, 20), bool)
    mask[2, 0:15] = True  # a long 15-cell run: the true mouth
    mask[15, 0:4] = True  # a separate 4-cell run, far away: a shoreline fragment
    spec = _Spec(mask)
    rows, _ = np.unravel_index(spec.flat_index, spec.shape)
    bed_elev_m = np.full(spec.flat_index.size, -5.0)  # uniformly deep

    seeds = estuary.ocean_seed_mask(
        spec, _WHOLE_GRID_POLYGON_KM, bed_elev_m, ocean_max_z_m=-2.0
    )

    assert seeds[rows == 2].all(), "the large run is the true mouth and must all seed"
    assert not seeds[rows == 15].any(), "the small detached run must not seed"


def test_ocean_seed_mask_excludes_edge_and_deep_cells_outside_the_polygon():
    """The polygon leg matters on its own, not just edge and depth: an edge
    cell that is deep enough and would otherwise qualify must not seed if it
    sits outside the authored ocean polygon. Every other test here uses a
    whole-grid polygon, so this is the one regression check that would catch
    `inside` being dropped from the seed calculation entirely."""
    mask = np.zeros((20, 20), bool)
    mask[10, 0:10] = True  # a channel one cell wide -- every cell is an edge cell
    spec = _Spec(mask)
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    bed_elev_m = np.full(spec.flat_index.size, -5.0)  # uniformly deep
    # Covers only the west half of the grid (x up to 0.5 km); cols 0-4 sit at
    # x <= 0.45 km (inside), cols 5-9 at x >= 0.55 km (outside).
    polygon_km = [(0.0, 0.0), (0.0, 2.0), (0.5, 2.0), (0.5, 0.0)]

    seeds = estuary.ocean_seed_mask(spec, polygon_km, bed_elev_m, ocean_max_z_m=-2.0)

    assert seeds[cols <= 4].all(), "edge, deep, and inside the polygon must seed"
    assert not seeds[cols >= 5].any(), "edge and deep but outside the polygon must not seed"


def test_load_distance_field_round_trips_a_written_array(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    np.save(d / "estuary_km.npy", np.array([0.0, 1.5, np.nan], dtype="float32"))

    d_out = estuary.load_distance_field("winyah-bay")

    assert np.allclose(d_out[:2], [0.0, 1.5])
    assert np.isnan(d_out[2])


def test_load_distance_field_names_the_missing_file(tmp_path, monkeypatch):
    """A missing field must point at the command that builds it, not raise a
    bare FileNotFoundError on a path the caller has to decode."""
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    with pytest.raises(FileNotFoundError, match="tidescout salinity field winyah-bay"):
        estuary.load_distance_field("winyah-bay")



# -- Two seaward openings in one polygon ------------------------------------
# `_largest_component` documents "ASSUMES exactly one seaward opening" and
# guards against a DROPPED second mouth. The failure these tests pin is the
# other one: two genuine openings CONTIGUOUS in the same coastal run, so the
# filter keeps both and Dijkstra silently hands every cell whichever is
# nearer. Measured on Winyah 2026-08-23: the mid/upper bay routed east
# through Mud Bay and out North Inlet instead of down the bay past the
# jetties, understating WYSS1 by 4.00 km (15.03 vs 19.03) and Thousand Acre
# by 5.01 km (11.67 vs 16.68) -- 27% and 43%.


def _two_mouth_spec():
    """A head that can reach the sea two ways: 12 cells south down the bay's
    own channel, or 4 cells east through a short side passage.

    Both routes are real water and both ends are genuine openings, so no
    edge/depth/polygon test tells them apart -- only which one the salt
    front actually advances from, which is authored, not inferred.
    """
    mask = np.zeros((20, 20), bool)
    mask[2:15, 3] = True   # the bay's own channel, north-south; south end is the mouth
    mask[2, 3:8] = True    # a side passage east from the head to a second opening
    return _Spec(mask)


def test_a_second_opening_in_the_same_polygon_captures_the_upper_estuary():
    """Seed both openings and the head measures to the NEARER one -- which
    is not the mouth the salt front advances from. This is the defect."""
    spec = _two_mouth_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)

    both = ((rows == 14) & (cols == 3)) | ((rows == 2) & (cols == 7))
    d = estuary.along_estuary_km(spec, both)

    head = (rows == 2) & (cols == 3)
    assert d[head][0] == pytest.approx(0.4)  # 4 cells east, not 12 south


def test_seeding_only_the_main_mouth_measures_the_route_the_salt_takes():
    """Same geometry, same head cell, three times the distance -- because
    the coordinate now means 'distance up THIS estuary'."""
    spec = _two_mouth_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)

    main_only = (rows == 14) & (cols == 3)
    d = estuary.along_estuary_km(spec, main_only)

    head = (rows == 2) & (cols == 3)
    assert d[head][0] == pytest.approx(1.2)  # 12 cells south down the channel


def test_a_narrower_polygon_excludes_the_second_opening():
    """The polygon leg is what separates them. `_Spec` puts row 0 at
    y = 1.95 km and row 19 at y = 0.05 km, so a polygon over y < 1.0 km
    contains the southern mouth and excludes the northern side passage."""
    spec = _two_mouth_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    bed_elev_m = np.full(spec.flat_index.size, -5.0)

    south_only_km = [(0.0, 0.0), (0.0, 1.0), (2.0, 1.0), (2.0, 0.0)]
    seeds = estuary.ocean_seed_mask(spec, south_only_km, bed_elev_m, ocean_max_z_m=-2.0)

    assert seeds[(rows == 14) & (cols == 3)][0], "the main mouth must seed"
    assert not seeds[rows == 2].any(), "the second opening must not seed"


def test_salt_source_boundary_defaults_to_the_ocean_boundary():
    """A fishery that has not authored a salt source keeps the old
    behaviour exactly, so this is opt-in per fishery rather than a silent
    re-interpretation of every existing config."""
    from tidescout.models import ModelDomain

    md = ModelDomain(
        polygon_utm_km=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        ocean_boundary_utm_km=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
    )

    assert md.salt_source_boundary_utm_km == []
    assert md.salt_source_polygon_utm_km == md.ocean_boundary_utm_km


def test_an_authored_salt_source_wins_over_the_ocean_boundary():
    """The two fields answer different questions and must not be conflated:
    `ocean_boundary_utm_km` says which mesh boundary segments take the TIDE
    (every genuine opening belongs, and `mesh.classify_boundary` reads it,
    so narrowing it would change the hydrodynamics and invalidate the
    library); `salt_source_boundary_utm_km` says which opening the salt
    front advances FROM."""
    from tidescout.models import ModelDomain

    md = ModelDomain(
        polygon_utm_km=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        ocean_boundary_utm_km=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        salt_source_boundary_utm_km=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
    )

    assert md.salt_source_polygon_utm_km == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert md.salt_source_polygon_utm_km != md.ocean_boundary_utm_km


def test_build_distance_field_seeds_from_the_salt_source(tmp_path, monkeypatch):
    """`build_distance_field` must read the salt source, not the ocean
    boundary -- the wiring is the whole point, and a field built from the
    wrong polygon is silently wrong rather than an error."""
    seen = {}

    def fake_seed(spec, polygon_km, bed, ocean_max_z_m):
        seen["polygon"] = polygon_km
        rows, cols = np.unravel_index(spec.flat_index, spec.shape)
        return (rows == 14) & (cols == 3)

    spec = _two_mouth_spec()
    monkeypatch.setattr(estuary, "ocean_seed_mask", fake_seed)
    monkeypatch.setattr(estuary, "_bed_elevation_m", lambda *a: np.full(spec.flat_index.size, -5.0))
    monkeypatch.setattr("tidescout.pipeline.flowlib.grid_spec", lambda *a: spec)
    monkeypatch.setattr(estuary, "fishery_data_dir", lambda slug: tmp_path)

    from tidescout.models import ModelDomain

    class _F:
        model_domain = ModelDomain(
            polygon_utm_km=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            ocean_boundary_utm_km=[(9.0, 9.0), (9.0, 8.0), (8.0, 8.0)],
            salt_source_boundary_utm_km=[(0.0, 0.0), (0.0, 1.0), (2.0, 1.0), (2.0, 0.0)],
        )

    estuary.build_distance_field("winyah-bay", _F())

    assert seen["polygon"] == _F.model_domain.salt_source_boundary_utm_km
    assert seen["polygon"] != _F.model_domain.ocean_boundary_utm_km


# -- Main stem and branch membership ----------------------------------------


def _branch_spec():
    """A main channel with one long side creek joining it near the mouth."""
    mask = np.zeros((20, 20), bool)
    mask[2:15, 3] = True   # main channel, north-south, mouth at the south end
    mask[13, 3:12] = True  # a side creek joining low down
    return _Spec(mask)


def test_descent_path_runs_downhill_to_a_seed():
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    seeds = (rows == 14) & (cols == 3)
    d = estuary.along_estuary_km(spec, seeds)
    grid = to_grid(d, spec.flat_index, spec.shape, fill=np.nan)

    path = estuary.descent_path(grid, (2, 3))

    assert path[0] == (2, 3)
    assert path[-1] == (14, 3), "must terminate at the seed"
    assert len(path) == 13


def test_distance_to_stem_separates_a_side_creek_from_the_channel():
    """The head of a side creek is far from the stem THROUGH WATER even
    when it is close in a straight line."""
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    stem = (cols == 3) & (rows >= 2) & (rows <= 14)

    to_stem = estuary.along_estuary_km(spec, stem)

    on_channel = (rows == 8) & (cols == 3)
    creek_head = (rows == 13) & (cols == 11)
    assert to_stem[on_channel][0] == pytest.approx(0.0)
    assert to_stem[creek_head][0] == pytest.approx(0.8)  # 8 cells of 100 m


@pytest.mark.parametrize(
    ("station", "expect_off_axis"),
    [
        ("NIWTAWQ", False), ("WYSS1", False), ("NIWWBWQ", False),
        ("NIWCBWQ", True), ("NIWOLWQ", True), ("NIWDCWQ", True),
    ],
)
def test_the_six_nerrs_stations_land_on_the_right_side_of_the_screen(
    station, expect_off_axis
):
    """Regression against the real built field, so a future threshold change
    cannot silently re-admit North Inlet to a Winyah Bay fit.

    Skips rather than fails when the field has not been built -- a fresh
    clone has no data/ directory, and this asserts about real geometry.
    """
    import numpy as np
    from rasterio.warp import transform as warp_transform

    from tidescout.config import load_fishery
    from tidescout.pipeline.flowlib import grid_spec
    from tidescout.sources import cdmo

    fishery = load_fishery("winyah-bay")
    try:
        stem = estuary.load_stem_distance_field("winyah-bay")
    except FileNotFoundError:
        pytest.skip("stem field not built -- run `tidescout salinity stem winyah-bay`")

    spec = grid_spec("winyah-bay", fishery)
    lon, lat = cdmo.NIW_STATION_COORDS_LONLAT[station]
    x, y = (v[0] for v in warp_transform(
        "EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", [lon], [lat]))
    i = int(np.argmin((spec.xs - x) ** 2 + (spec.ys - y) ** 2))

    assert (float(stem[i]) > estuary.ON_AXIS_MAX_KM) is expect_off_axis


def test_on_axis_threshold_sits_in_the_measured_gap():
    """Measured on the real field 2026-08-24: on-axis stations span
    0.048-1.604 km to the stem and off-axis ones 7.798-11.918, with only
    Jones Creek / Mud Bay (2.170) between. The threshold must sit in that
    gap, not on either shoulder."""
    assert 1.604 < estuary.ON_AXIS_MAX_KM < 2.170


# -- Task 4 review fixes: main_stem_mask guards + isolated unit coverage ----
# Review found the six-station real-data test above is the ONLY coverage for
# `main_stem_mask`, `build_stem_distance_field`, and `load_stem_distance_field`
# -- and it skips outright on a fresh clone with no data/ directory, so the
# whole river-union / nearest-cell / write-then-read path could execute zero
# times. It also found `main_stem_mask` had no guard on `inflow_lonlat` being
# `None` (RiverGauge's own documented "not attached" state) or on a snapped
# cell having no water route to the sea -- exactly the hazard
# `regimes.py::_attach_river_inflows` already guards for the same
# `fishery.rivers` list feeding the ANUGA mesh (see that function's own
# "most dangerous silent failure" comment). These tests close both gaps
# without touching ON_AXIS_MAX_KM, the stem definition, or the six-station
# gate.


class _Bathy26917:
    epsg = 26917


def _identity_warp(monkeypatch):
    """`main_stem_mask` locally imports `rasterio.warp.transform` at call
    time, so patching the real attribute (not a name in `estuary`'s
    namespace) reaches it. Returning the lon/lat pair unchanged makes the
    `_Spec` fixture's own xs/ys the "utm" coordinates too, keeping these
    tests decoupled from real CRS math, which is not what is under test."""
    monkeypatch.setattr(
        "rasterio.warp.transform", lambda from_crs, to_crs, lons, lats: (lons, lats)
    )


def test_main_stem_mask_returns_the_union_of_descent_paths(monkeypatch):
    """Synthetic-fixture coverage for the union logic itself: one river
    inflow point at the head of `_branch_spec`'s main channel must pull in
    the whole channel and none of the side creek."""
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    mouth = (rows == 14) & (cols == 3)
    field = estuary.along_estuary_km(spec, mouth)
    _identity_warp(monkeypatch)
    head = (rows == 2) & (cols == 3)
    head_x, head_y = spec.xs[head][0], spec.ys[head][0]

    class _River:
        name = "Test River"
        inflow_lonlat = (head_x, head_y)

    class _Fishery:
        rivers = [_River()]
        bathymetry = _Bathy26917()

    stem = estuary.main_stem_mask(_Fishery(), spec, field)

    on_main_channel = (cols == 3) & (rows >= 2) & (rows <= 14)
    on_creek = (rows == 13) & (cols > 3)
    assert stem[on_main_channel].all(), "the whole descent path must be marked"
    assert not stem[on_creek].any(), "the side creek must not be pulled in"


def test_main_stem_mask_skips_a_river_with_no_inflow_point(monkeypatch):
    """Mirrors `regimes.py::_attach_river_inflows`'s
    `seed = getattr(river, "inflow_lonlat", None); if seed is None: continue`
    -- `inflow_lonlat=None` means 'not attached', not an error."""
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    mouth = (rows == 14) & (cols == 3)
    field = estuary.along_estuary_km(spec, mouth)
    _identity_warp(monkeypatch)
    head = (rows == 2) & (cols == 3)
    head_x, head_y = spec.xs[head][0], spec.ys[head][0]

    class _UnattachedRiver:
        name = "Unattached River"
        inflow_lonlat = None

    class _River:
        name = "Test River"
        inflow_lonlat = (head_x, head_y)

    class _Fishery:
        rivers = [_UnattachedRiver(), _River()]
        bathymetry = _Bathy26917()

    stem = estuary.main_stem_mask(_Fishery(), spec, field)  # must not raise

    on_main_channel = (cols == 3) & (rows >= 2) & (rows <= 14)
    assert stem[on_main_channel].all()


def test_main_stem_mask_raises_when_a_river_lands_off_connected_water(monkeypatch):
    """A coordinate that snaps to a cell with no water route to the sea
    (NaN in the along-estuary field) must fail loudly and name the river --
    not silently contribute an orphan cell or send `descent_path` wandering
    through undefined NaN comparisons."""
    mask = np.zeros((20, 20), bool)
    mask[2:15, 3] = True  # the estuary's main channel
    mask[0, 10] = True    # an isolated pond -- no route to the sea
    spec = _Spec(mask)
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    mouth = (rows == 14) & (cols == 3)
    field = estuary.along_estuary_km(spec, mouth)
    _identity_warp(monkeypatch)
    pond = (rows == 0) & (cols == 10)
    pond_x, pond_y = spec.xs[pond][0], spec.ys[pond][0]

    class _River:
        name = "Misplaced River"
        inflow_lonlat = (pond_x, pond_y)

    class _Fishery:
        rivers = [_River()]
        bathymetry = _Bathy26917()

    with pytest.raises(RuntimeError, match="Misplaced River"):
        estuary.main_stem_mask(_Fishery(), spec, field)


def test_main_stem_mask_raises_when_no_river_has_an_inflow_point():
    """Every river skipped (all `inflow_lonlat=None`) must not silently
    return an empty stem -- an empty seed set would make the caller's
    `along_estuary_km` produce an all-NaN distance field with no error
    anywhere upstream of it."""
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    mouth = (rows == 14) & (cols == 3)
    field = estuary.along_estuary_km(spec, mouth)

    class _UnattachedRiver:
        name = "Unattached River"
        inflow_lonlat = None

    class _Fishery:
        rivers = [_UnattachedRiver()]
        bathymetry = _Bathy26917()

    with pytest.raises(RuntimeError, match="no river inflow point"):
        estuary.main_stem_mask(_Fishery(), spec, field)


def test_build_stem_distance_field_writes_distance_from_the_stem(tmp_path, monkeypatch):
    """The write half of the round trip `load_stem_distance_field`'s tests
    below check the read half of -- wires `main_stem_mask` into the SAME
    `along_estuary_km` Dijkstra, seeded from the stem instead of the sea."""
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    stem_mask = (cols == 3) & (rows >= 2) & (rows <= 14)

    monkeypatch.setattr("tidescout.pipeline.flowlib.grid_spec", lambda *a: spec)
    monkeypatch.setattr(
        estuary, "load_distance_field", lambda slug: np.zeros(spec.flat_index.size)
    )
    monkeypatch.setattr(estuary, "main_stem_mask", lambda fishery, s, field: stem_mask)
    monkeypatch.setattr(estuary, "fishery_data_dir", lambda slug: tmp_path)

    path = estuary.build_stem_distance_field("winyah-bay", object())

    assert path == tmp_path / "stem_km.npy"
    d = np.load(path)
    on_stem = (rows == 8) & (cols == 3)
    creek_head = (rows == 13) & (cols == 11)
    assert d[on_stem][0] == pytest.approx(0.0)
    assert d[creek_head][0] == pytest.approx(0.8)  # 8 cells of 100 m


def test_load_stem_distance_field_round_trips_a_written_array(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    np.save(d / "stem_km.npy", np.array([0.0, 1.5, np.nan], dtype="float32"))

    d_out = estuary.load_stem_distance_field("winyah-bay")

    assert np.allclose(d_out[:2], [0.0, 1.5])
    assert np.isnan(d_out[2])


def test_load_stem_distance_field_names_the_missing_file(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    with pytest.raises(FileNotFoundError, match="tidescout salinity stem winyah-bay"):
        estuary.load_stem_distance_field("winyah-bay")
