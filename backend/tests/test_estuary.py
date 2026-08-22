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
