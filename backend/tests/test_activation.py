"""Feature sampling on hand-built fields.

Each fixture puts a known value under a known feature so a wrong answer points
at the sampling, not at the physics.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine
from rasterio.warp import transform as warp_transform
from shapely.geometry import Polygon

from tidescout.engine import activation, detect


class _Spec:
    """Minimal stand-in for flowlib.GridSpec: an 8x8 grid of 20 m cells.

    `origin` is the grid's north-west corner in `crs`. It defaults to the
    arbitrary (0, 160) most of these tests use, which is fine while everything
    stays in grid coordinates; the reprojection test moves it to a real
    EPSG:26917 easting/northing so a lon/lat round trip means something.
    """

    def __init__(self, origin=(0.0, 160.0), crs="EPSG:26917"):
        self.shape = (8, 8)
        self.cell_m = 20.0
        self.crs = crs
        self.transform = Affine(20.0, 0.0, origin[0], 0.0, -20.0, origin[1])
        self.flat_index = np.arange(64)
        cols, rows = np.meshgrid(np.arange(8), np.arange(8))
        self.xs, self.ys = self.transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)


def _feature(key, ftype, lonlat_coords):
    return {
        "id": key,
        "properties": {"type": ftype},
        "geometry": {"type": "Point", "coordinates": lonlat_coords},
    }


def test_sample_features_averages_the_field_within_the_radius():
    """radius_m=25.0 on a 20 m grid selects five cells around index 27: the
    centre itself (distance 0) plus its four orthogonal neighbours (distance
    exactly 20 m, inside 25). The diagonals sit at 28.28 m and fall outside.
    So the true mean is (2.5 + 4*0.5) / 5 == 0.9, not the centre's own 2.5 --
    recording the arithmetic here so the next reader doesn't have to
    rediscover it. This is also the only test in the file that exercises
    `nanmean` pooling more than one cell; a tighter radius that selects only
    the centre would degenerate to "mean of one value equals that value" and
    pass regardless of whether the pooling picked the right cells."""
    spec = _Spec()
    speed = np.zeros(64)
    speed[spec.flat_index] = 0.5
    # One cell much faster, inside the sample radius of the feature below.
    speed[27] = 2.5

    feats = [_feature("hole-abc", "hole", (spec.xs[27], spec.ys[27]))]
    out = activation.sample_features(
        feats, spec, {"speed": speed}, radius_m=25.0, already_projected=True
    )
    assert len(out) == 1
    assert out[0].key == "hole-abc"
    assert out[0].speed == pytest.approx(0.9)


def test_sample_features_reports_the_max_for_ambush_not_the_mean():
    """An ambush point is defined by its best cell. Averaging a 150 m disc over
    a 20 m grid would dilute a real pocket into the channel around it."""
    spec = _Spec()
    ambush = np.zeros(64)
    ambush[27] = 1.0
    feats = [_feature("bar-def", "bar", (spec.xs[27], spec.ys[27]))]
    out = activation.sample_features(
        feats, spec, {"ambush": ambush}, radius_m=60.0, already_projected=True
    )
    assert out[0].ambush == pytest.approx(1.0)


def test_features_with_no_cells_in_the_domain_are_returned_with_nan_not_dropped():
    """Dropping them would make a feature vanish from the map with no
    explanation. NaN plus n_cells=0 says 'outside the model domain'."""
    spec = _Spec()
    feats = [_feature("hole-far", "hole", (999999.0, 999999.0))]
    out = activation.sample_features(
        feats, spec, {"speed": np.zeros(64)}, radius_m=25.0, already_projected=True
    )
    assert len(out) == 1
    assert out[0].n_cells == 0
    assert np.isnan(out[0].speed)


def _polygon_feature(key, ftype, ring):
    return {
        "id": key,
        "properties": {"type": ftype},
        "geometry": {"type": "Polygon", "coordinates": [[list(pt) for pt in ring]]},
    }


def test_the_sampling_anchor_is_the_centroid_the_feature_id_is_hashed_from():
    """A feature's id and its metrics have to describe the same place.
    `detect.feature_key` hashes `geometry.centroid`; Phase 3 keys scoring off
    that id and reads the `FeatureMetrics` beside it, so sampling anywhere else
    means the id says "this place" while the numbers describe another.

    The anchor used to be an unweighted mean of the exterior ring's vertices,
    counting the duplicated closing vertex twice -- which is not any named
    point of a polygon. On the real 2,162-feature inventory that sat a median
    7.6 m and a maximum 726 m from the centroid.

    This triangle separates the two candidates by 31.9 m, more than a cell:
    its centroid is (63.3, 56.7) and the old ring-mean was (85.0, 80.0). The
    12 m sample radius selects exactly one cell, so the assertion below reads
    5.0 under the centroid and 1.0 under the ring-mean -- it cannot pass by
    accident."""
    spec = _Spec()
    ring = [(150.0, 150.0), (10.0, 10.0), (30.0, 10.0), (150.0, 150.0)]
    poly = Polygon(ring)
    key = detect.feature_key(detect.Feature("bar", poly))

    speed = np.zeros(64)
    speed[43] = 5.0                 # the cell holding the true centroid
    speed[36] = speed[28] = 1.0     # the cells the old ring-mean would sample

    feats = [_polygon_feature(key, "bar", ring)]
    xs, ys = activation._sampling_anchors(feats, spec, already_projected=True)
    assert (xs[0], ys[0]) == pytest.approx((poly.centroid.x, poly.centroid.y))

    out = activation.sample_features(
        feats, spec, {"speed": speed}, radius_m=12.0, already_projected=True
    )
    assert out[0].key == key
    assert out[0].n_cells == 1
    assert out[0].speed == pytest.approx(5.0)


def test_sample_features_reprojects_lonlat_geojson_onto_the_grid():
    """The `already_projected=False` branch is what production actually runs --
    `build_features` writes EPSG:4326 GeoJSON and the library grid is in UTM --
    and it had no test anywhere; every other test in this file short-circuits
    it. The grid sits at real EPSG:26917 coordinates here so the round trip is
    a genuine reprojection, and the 5 m radius selects a single cell, so
    landing one 20 m cell away already fails."""
    spec = _Spec(origin=(662000.0, 3690000.0))
    target = 27
    speed = np.zeros(64)
    speed[target] = 3.0
    lons, lats = warp_transform(
        spec.crs, "EPSG:4326", [float(spec.xs[target])], [float(spec.ys[target])]
    )

    feats = [_feature("hole-utm", "hole", (lons[0], lats[0]))]
    out = activation.sample_features(feats, spec, {"speed": speed}, radius_m=5.0)
    assert out[0].n_cells == 1
    assert out[0].speed == pytest.approx(3.0)


def test_structure_fields_returns_masked_1d_arrays_on_the_library_layout():
    """The round trip must give back exactly the cells it was handed."""
    spec = _Spec()
    n = spec.flat_index.size
    u = np.full(n, 0.4)
    v = np.zeros(n)
    depth = np.full(n, 2.0)
    fields = activation.structure_fields(u, v, depth, spec)
    for name in activation._SAMPLED_FIELDS:
        assert fields[name].shape == (n,), name
    assert np.allclose(fields["speed"], 0.4)
    assert np.allclose(fields["ambush"], 0.0)  # uniform flow: no contrast


def _schedule(flood_by_cell, n=64):
    """Minimal stand-in for schedule.CellSchedule over the 8x8 _Spec grid."""
    flood = np.full(n, np.nan)
    for idx, phase in flood_by_cell.items():
        flood[idx] = phase
    return SimpleNamespace(wet_fraction=np.ones(n), flood_phase=flood)


def test_flood_phase_is_averaged_on_the_circle_not_on_the_line():
    """Phase wraps. 0.95 and 0.05 are 0.1 of a cycle apart and their centre is
    0.0 -- LOW water, the start of the flood half -- while an ordinary median
    of that pair returns 0.5, which is high water, the other half of the tide.
    `pipeline/schedule.py`'s module docstring establishes that an ordinary
    median of a phase "lands on whichever side of 0.5 that cluster happens to
    fall, which is an artifact of the cut point, not of the physics";
    `sample_features` reduced `flood_phase` with `np.nanmedian` and undid it.

    Cells 27 and 28 are the two cells within 15 m of (80, 90) on the _Spec
    grid, so the disc holds exactly this straddling pair."""
    spec = _Spec()
    feats = [_feature("flat-wrap", "flat", (80.0, 90.0))]
    out = activation.sample_features(
        feats, spec, {}, _schedule({27: 0.95, 28: 0.05}),
        radius_m=15.0, already_projected=True,
    )
    assert out[0].n_cells == 2
    assert np.nanmedian([0.95, 0.05]) == 0.5, "what the old reduction returned"
    assert out[0].flood_phase == pytest.approx(0.0, abs=1e-9)


def test_flood_phase_of_a_cluster_that_does_not_wrap_is_its_ordinary_centre():
    """The circular mean must not move an ordinary cluster: away from the wrap
    it has to agree with the obvious answer, or it would trade one artifact
    for another."""
    spec = _Spec()
    feats = [_feature("flat-plain", "flat", (80.0, 90.0))]
    out = activation.sample_features(
        feats, spec, {}, _schedule({27: 0.28, 28: 0.32}),
        radius_m=15.0, already_projected=True,
    )
    assert out[0].flood_phase == pytest.approx(0.3, abs=1e-3)


def _rotation_and_strain_grid(n=32, cell=20.0, strain=0.01, omega=0.004):
    """A pure-strain background with a solid-body rotation patch cut into it.

    Analytic: strain gives W = +4*strain^2 = +4e-4 everywhere outside the patch,
    rotation gives W = -4*omega^2 = -6.4e-5 inside it, and the splice between
    the two carries a much larger positive W than either. So the patch is a
    genuine eddy and the disc's strongest cell is a seam -- which is exactly
    the situation a max reducer cannot describe.
    """
    c = (np.arange(n) - n / 2) * cell
    x = np.tile(c, (n, 1))
    y = np.tile(-c[:, None], (1, n))  # rows run south
    u = strain * x
    v = -strain * y
    patch = (slice(8, 20), slice(8, 20))
    u[patch] = -omega * y[patch]
    v[patch] = omega * x[patch]
    xs, ys = np.meshgrid(c, -c)
    spec = SimpleNamespace(
        shape=(n, n), cell_m=cell, flat_index=np.arange(n * n),
        xs=xs.ravel(), ys=ys.ravel(), crs="EPSG:26917",
    )
    return u, v, spec, (float(x[14, 14]), float(y[14, 14]))


def test_eddy_share_reports_the_rotation_that_max_reduced_okubo_w_hides():
    """W > 0 is a seam and W < 0 is an eddy, so a MAX over the disc returns the
    most seam-like cell in it and structurally cannot report an eddy. That is
    not a hypothetical: over the shipped winyah-bay `mean_med` library, all 26
    phases, exactly 2 of 13,588 finite per-feature `okubo_w` samples were
    negative (-8.2e-7 and -4.5e-9), both more than ten times inside the
    `quiet_w = 1e-5` dead band -- none ever crossed -quiet_w. The feature-level
    eddy channel was dead by construction.

    `eddy_share` is that channel. Here the disc is two-thirds rotation, and
    `okubo_w` still reads strongly positive off the splice between the patch
    and the strained background -- both numbers are right, and only one of
    them can find the eddy."""
    u, v, spec, centre = _rotation_and_strain_grid()
    depth = np.full(u.size, 2.0)
    fields = activation.structure_fields(u.ravel(), v.ravel(), depth, spec)

    feats = [_feature("hole-eddy", "hole", centre)]
    m = activation.sample_features(
        feats, spec, fields, radius_m=140.0, already_projected=True
    )[0]
    assert m.okubo_w > 0.0, "the max reducer keeps reporting the strongest seam"
    assert m.eddy_share == pytest.approx(0.664, abs=0.01)


def test_eddy_share_excludes_dry_cells_from_its_denominator():
    """The share is eddy-over-WET, not eddy-over-disc. Counting dry cells as
    "not an eddy" would be the same mistake `structure_fields` masks twice to
    avoid -- ANUGA writes u = v = 0.0 on dry ground, and that is not still
    water, it is not water. Here every wet cell rotates and half the disc is
    dry: the honest answer is 1.0, and a disc denominator would say ~0.5."""
    n, cell, omega = 32, 20.0, 0.004
    c = (np.arange(n) - n / 2) * cell
    x = np.tile(c, (n, 1))
    y = np.tile(-c[:, None], (1, n))
    u = -omega * y
    v = omega * x
    depth = np.full((n, n), 2.0)
    depth[n // 2 :, :] = 0.0
    xs, ys = np.meshgrid(c, -c)
    spec = SimpleNamespace(
        shape=(n, n), cell_m=cell, flat_index=np.arange(n * n),
        xs=xs.ravel(), ys=ys.ravel(), crs="EPSG:26917",
    )
    fields = activation.structure_fields(u.ravel(), v.ravel(), depth.ravel(), spec)
    feats = [_feature("flat-halfdry", "flat", (float(x[15, 15]), float(y[15, 15])))]
    m = activation.sample_features(
        feats, spec, fields, radius_m=140.0, already_projected=True
    )[0]
    assert m.n_cells > 100, "the disc must straddle the waterline for this to bite"
    assert m.eddy_share == pytest.approx(1.0)


def test_structure_fields_masks_dry_cells_out_of_every_field():
    """ANUGA reports u = v = 0.0 on a dry cell, not NaN (regimes.py's
    _centroid_speed zeroes momentum where depth <= 0.01 m rather than masking
    it). Left alone, a dry marsh cell sitting beside a fast channel is
    indistinguishable from genuine slack water -- a slow cell next to a fast
    one is exactly the shape ambush_contrast hunts for, so it would score as
    a perfect ambush pocket. You cannot fish dry marsh: structure_fields must
    use `depth` to mask those cells to NaN in every field it returns, not
    just leave them at their true-but-misleading 0.0 value.

    The dry cell here is a single ISOLATED cell -- wet on all four sides --
    not a thick block, and not even a full dry row or column. That distinction
    is load-bearing: a dry ROW would still read NaN in strain/okubo_w/
    convergence even without an explicit output mask, because the along-row
    derivative (du_dx here) differentiates using OTHER cells in that same dry
    row, which are also NaN, so the existing input-side masking already
    poisons it -- that fixture would pass even with the output-masking fix
    removed, silently testing nothing. A single isolated cell is the minimal
    case that actually needs the output mask: np.gradient's central
    difference at index i is (a[i+1] - a[i-1]) / 2h -- it never reads a[i]
    itself -- so BOTH the row-direction and column-direction derivatives at
    the isolated cell are computed entirely from its four wet orthogonal
    neighbours and bridge straight across it, landing on a finite,
    plausible-looking number instead of NaN (confirmed against a real run: an
    isolated dry cell on a shear line returned strain=0.03, okubo_w=0.0009,
    convergence=-0.03, all finite, while speed and ambush -- masked directly,
    not via the gradient -- correctly read NaN there). Since those three
    fields are MAX-reduced per feature, one such cell inside a 150 m disc
    could set an entire feature's reported seam or convergence score from
    unfishable dry land."""
    n = 64
    shape = (n, n)
    flat_index = np.arange(n * n)
    cell_m = 20.0

    # Pure shear (du/dy = a, dv = 0), the same fixture shape test_structure.py
    # uses for a known strain answer -- a nonzero background so a gradient
    # "bridged" across the dry cell lands on a finite, non-trivial value
    # rather than an accidental 0.0 that could be mistaken for NaN handling.
    a = 0.004
    row_y_m = (np.arange(n) * cell_m)[:, None]
    u = np.broadcast_to(a * row_y_m, shape).astype("float64").copy()
    v = np.zeros(shape)
    depth = np.full(shape, 2.0)

    # A single dry cell, wet on all four orthogonal sides -- one cell wide
    # along BOTH axes np.gradient differentiates across.
    dry_row, dry_col = 30, 30
    u[dry_row, dry_col] = 0.0
    v[dry_row, dry_col] = 0.0
    depth[dry_row, dry_col] = 0.0

    spec = SimpleNamespace(shape=shape, cell_m=cell_m, flat_index=flat_index)
    fields = activation.structure_fields(u.ravel(), v.ravel(), depth.ravel(), spec)

    dry_pos = dry_row * n + dry_col
    for name in ("speed", "ambush", "strain", "okubo_w", "convergence"):
        assert np.isnan(fields[name][dry_pos]), name
