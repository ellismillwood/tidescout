"""Along-estuary distance: how far each cell is from the sea THROUGH WATER.

Salinity depends on channel distance, not straight-line distance -- a cell 2 km
from the ocean across a barrier island is 30 km from it up the channel, and the
two answers differ by an order of magnitude over most of Winyah Bay. This walks
the domain mask as a graph, so the branching up the Pee Dee, Waccamaw, Black and
Sampit needs no special handling: each branch simply gets longer.

Built once per fishery. The result is static -- geometry, not state.
"""

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Point, Polygon

from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir

# 8-connectivity. Orthogonal steps cost one cell, diagonals sqrt(2) -- with
# equal weights a diagonal channel would measure ~30% shorter than it is.
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
    (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
]


def along_estuary_km(spec, seed_mask: np.ndarray) -> np.ndarray:
    """Geodesic distance in km from the seeded cells, over in-domain cells only.

    `seed_mask` is a boolean over the same 1-D layout as the library arrays.
    Cells with no water route to a seed come back NaN, never 0.0: zero means
    "at the mouth", which is the saltiest place in the model and so the most
    damaging possible default for an isolated pond.
    """
    n = spec.flat_index.size
    if not seed_mask.any():
        raise ValueError(
            "no seed cells -- the ocean polygon selects nothing inside the "
            "model domain, so there is no sea to measure distance from"
        )

    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    # Position -> compact node id, for O(1) neighbour lookup.
    lookup = np.full(int(spec.shape[0]) * int(spec.shape[1]), -1, dtype="int64")
    lookup[spec.flat_index] = np.arange(n)

    src, dst, weight = [], [], []
    for dr, dc, cost in _NEIGHBOURS:
        nr, nc = rows + dr, cols + dc
        ok = (nr >= 0) & (nr < spec.shape[0]) & (nc >= 0) & (nc < spec.shape[1])
        nid = np.full(n, -1, dtype="int64")
        nid[ok] = lookup[np.ravel_multi_index((nr[ok], nc[ok]), spec.shape)]
        joined = nid >= 0
        src.append(np.nonzero(joined)[0])
        dst.append(nid[joined])
        weight.append(np.full(int(joined.sum()), cost * spec.cell_m))

    graph = coo_matrix(
        (np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))),
        shape=(n, n),
    ).tocsr()

    d = dijkstra(graph, directed=False, indices=np.nonzero(seed_mask)[0], min_only=True)
    d = np.asarray(d, dtype="float64") / 1000.0
    d[np.isinf(d)] = np.nan
    return d


def ocean_seed_mask(spec, ocean_boundary_utm_km: list) -> np.ndarray:
    """In-domain cells lying inside the authored ocean polygon: the sea itself.

    Reuses `model_domain.ocean_boundary_utm_km` rather than inferring the mouth
    from depth. Plan 3 established twice over that depth cannot classify
    geography -- it put the ocean tide 40 km up the Pee Dee -- and the seaward
    opening is already authored, so there is nothing to infer.
    """
    if not ocean_boundary_utm_km:
        raise ValueError(
            "model_domain.ocean_boundary_utm_km is empty -- the along-estuary "
            "distance field has no sea to measure from"
        )
    poly = Polygon([(x * 1000.0, y * 1000.0) for x, y in ocean_boundary_utm_km])
    if not poly.is_valid:
        raise ValueError("ocean_boundary_utm_km is not a valid polygon")
    return np.fromiter(
        (poly.contains(Point(x, y)) for x, y in zip(spec.xs, spec.ys, strict=True)),
        dtype=bool,
        count=spec.xs.size,
    )


def build_distance_field(slug: str, fishery: Fishery) -> Path:
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    seeds = ocean_seed_mask(spec, fishery.model_domain.ocean_boundary_utm_km)
    d = along_estuary_km(spec, seeds)
    path = fishery_data_dir(slug) / "estuary_km.npy"
    np.save(path, d.astype("float32"))
    return path


def load_distance_field(slug: str) -> np.ndarray:
    path = fishery_data_dir(slug) / "estuary_km.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"no along-estuary distance field at {path} -- run "
            f"`tidescout salinity field {slug}` first"
        )
    return np.load(path)
