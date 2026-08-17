"""Feature sampling on hand-built fields.

Each fixture puts a known value under a known feature so a wrong answer points
at the sampling, not at the physics.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

from tidescout.engine import activation


class _Spec:
    """Minimal stand-in for flowlib.GridSpec: an 8x8 grid of 20 m cells."""

    def __init__(self):
        self.shape = (8, 8)
        self.cell_m = 20.0
        self.transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 160.0)
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


def test_structure_fields_returns_masked_1d_arrays_on_the_library_layout():
    """The round trip must give back exactly the cells it was handed."""
    spec = _Spec()
    n = spec.flat_index.size
    u = np.full(n, 0.4)
    v = np.zeros(n)
    depth = np.full(n, 2.0)
    fields = activation.structure_fields(u, v, depth, spec)
    for name in ("speed", "ambush", "strain", "okubo_w", "convergence"):
        assert fields[name].shape == (n,), name
    assert np.allclose(fields["speed"], 0.4)
    assert np.allclose(fields["ambush"], 0.0)  # uniform flow: no contrast


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
