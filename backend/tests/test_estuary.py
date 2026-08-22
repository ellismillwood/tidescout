"""Geodesic distance on hand-built channel geometry.

Straight-line distance is the wrong answer everywhere in an estuary, so these
fixtures are built so the two answers differ measurably.
"""

import numpy as np
import pytest
from affine import Affine

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
