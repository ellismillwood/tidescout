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

from tidescout.engine.structure import from_grid, to_grid
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


def _domain_edge_mask(spec) -> np.ndarray:
    """In-domain cells with at least one 4-neighbour outside the domain.

    The domain's outer boundary, one cell wide. Scatters the flat mask to 2-D
    and back via `engine.structure.to_grid`/`from_grid` -- the established
    bridge for any neighbour computation over the library's masked layout
    (see that module's docstring). Needed by `ocean_seed_mask`: an authored
    polygon says which AREA the sea may occupy, not which cells are actually
    the boundary between that area and open water.
    """
    domain = to_grid(np.ones(spec.flat_index.size), spec.flat_index, spec.shape, fill=0.0) > 0.5
    padded = np.pad(domain, 1, constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge = domain & ~interior
    return from_grid(edge, spec.flat_index)


def ocean_seed_mask(
    spec,
    ocean_boundary_utm_km: list,
    bed_elev_m: np.ndarray,
    ocean_max_z_m: float,
) -> np.ndarray:
    """In-domain cells that are the sea itself: on the domain's outer edge,
    below `ocean_max_z_m`, and inside the authored ocean polygon.

    All three, matching `mesh.classify_boundary`'s own criterion exactly --
    "a segment whose midpoint is deep AND inside that polygon is the true
    seaward opening" -- carried over from ring segments to library cells.
    `ocean_boundary_utm_km` alone is NOT a "this area is the sea" test: it is
    `classify_boundary`'s ring-segment filter, authored generously (a box
    spanning much of the bay's mouth) so a HANDFUL of boundary-ring midpoints
    near the true coast fall inside it regardless of exactly how the
    shoreline was traced -- 10 of Winyah's 605 ring segments pass it. Testing
    polygon-containment alone against every in-domain CELL instead of every
    boundary SEGMENT made 40.5% of this fishery's domain (238,106 of 587,325
    cells) read as "at the sea": Georgetown Lighthouse, a few km up-estuary
    by every other measure, landed at 0.00 km, tied with the jetty itself.
    Restricting to edge cells that are also deep reproduces
    `classify_boundary`'s real criterion and independently matches Phase 1's
    measurement of the same opening: 1,323 such cells at 20 m is 26.5 km of
    coastline, against the 31.20 km Phase 1's ring-segment classifier
    measured for Winyah's seaward boundary by a completely different method.

    `bed_elev_m` is bed elevation in metres, one value per `spec.flat_index`
    cell, sampled the same way `flowlib.grid_spec` decimates the domain mask
    (see `_bed_elevation_m`) so "in domain" and "how deep" agree pixel for
    pixel.
    """
    if not ocean_boundary_utm_km:
        raise ValueError(
            "model_domain.ocean_boundary_utm_km is empty -- the along-estuary "
            "distance field has no sea to measure from"
        )
    poly = Polygon([(x * 1000.0, y * 1000.0) for x, y in ocean_boundary_utm_km])
    if not poly.is_valid:
        raise ValueError("ocean_boundary_utm_km is not a valid polygon")
    inside = np.fromiter(
        (poly.contains(Point(x, y)) for x, y in zip(spec.xs, spec.ys, strict=True)),
        dtype=bool,
        count=spec.xs.size,
    )
    deep = np.isfinite(bed_elev_m) & (bed_elev_m < ocean_max_z_m)
    return inside & _domain_edge_mask(spec) & deep


def _bed_elevation_m(slug: str, fishery: Fishery, spec) -> np.ndarray:
    """Bed elevation at each library cell, one value per `spec.flat_index`.

    Decimated exactly the way `flowlib.grid_spec` decimates the domain mask
    -- same `z[::step, ::step]`, same `step` -- so a cell's elevation here is
    read from the identical source pixel that decided whether the mask put
    that cell in-domain. A nearest-pixel-centre lookup via the inverted
    affine transform was tried and rejected: near a decimation boundary it
    can pick a different source pixel than the mask used, silently
    misaligning "in domain" from "how deep" by up to half a library cell.
    """
    from tidescout.pipeline.bathy import read_bathy

    z, _, _ = read_bathy(slug)
    step = int(round(fishery.anuga.library_cell_m / fishery.bathymetry.cell_m))
    z_lib = z[::step, ::step]
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    return z_lib[rows, cols].astype("float64")


def build_distance_field(slug: str, fishery: Fishery) -> Path:
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    md = fishery.model_domain
    bed_elev_m = _bed_elevation_m(slug, fishery, spec)
    seeds = ocean_seed_mask(spec, md.ocean_boundary_utm_km, bed_elev_m, md.ocean_max_z_m)
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
