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
    spec = _Spec()
    speed = np.zeros(64)
    speed[spec.flat_index] = 0.5
    # One cell much faster, inside the sample radius of the feature below.
    # radius_m=15.0 is deliberately BELOW cell_m=20.0: orthogonal neighbours
    # sit exactly 20 m away, so a radius of 25.0 here would pull four of them
    # into the disc too (distance 20 <= 25) and the mean over all five cells
    # is 0.9, not 2.5 -- silently testing something other than what the
    # comment above claims. Below one cell spacing, only the feature's own
    # cell can ever be "inside," so the mean of one value trivially equals
    # that value and the assertion matches the stated intent exactly.
    speed[27] = 2.5

    feats = [_feature("hole-abc", "hole", (spec.xs[27], spec.ys[27]))]
    out = activation.sample_features(
        feats, spec, {"speed": speed}, radius_m=15.0, already_projected=True
    )
    assert len(out) == 1
    assert out[0].key == "hole-abc"
    assert out[0].speed == pytest.approx(2.5)


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
    use `depth` to mask those cells to NaN in speed and ambush alike, not
    just leave them at their true-but-misleading 0.0 value."""
    n = 64
    shape = (n, n)
    flat_index = np.arange(n * n)

    u = np.full(shape, 1.0)  # a fast conveyor filling the whole grid
    v = np.zeros(shape)
    depth = np.full(shape, 2.0)

    # A block of dry marsh hard against the fast water -- u=v=0, depth below
    # the wet threshold, exactly what the ANUGA pipeline reports there.
    dry = np.zeros(shape, dtype=bool)
    dry[28:36, 24:32] = True
    u[dry] = 0.0
    v[dry] = 0.0
    depth[dry] = 0.0

    spec = SimpleNamespace(shape=shape, cell_m=20.0, flat_index=flat_index)
    fields = activation.structure_fields(u.ravel(), v.ravel(), depth.ravel(), spec)

    dry_flat = dry.ravel()[flat_index]
    assert np.all(np.isnan(fields["speed"][dry_flat]))
    assert np.all(np.isnan(fields["ambush"][dry_flat]))
