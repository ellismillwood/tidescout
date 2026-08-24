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
from scipy import ndimage
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
            "no seed cells -- the seed mask is empty, so there is no sea to "
            "measure distance from"
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


def _largest_component(mask: np.ndarray, spec) -> np.ndarray:
    """Keep only the largest 8-connected component of a 1-D masked boolean
    array; drop everything else.

    The true seaward opening is contiguous by construction -- one stretch of
    coastline. Edge-and-deep-and-inside-the-polygon does not by itself
    guarantee that: a shoreline fragment deep inside the bay, or an isolated
    deep pocket that happens to touch the domain edge inside the polygon,
    passes all three tests without being part of the mouth. On Winyah at
    `ocean_max_z_m = -2.0` the raw edge-and-deep-and-inside set fragments
    into 9 separate 8-connected components; the largest holds 950 of 1,317
    candidate cells, and the other ~370 are scattered noise -- including the
    single near-threshold cell 1.14 km from Georgetown Lighthouse that made
    its distance swing by 5 km across a physically-arbitrary choice of
    `ocean_max_z_m` (0.50 to 5.51 km over -1.6 to -3.0 m). Restricted to the
    largest component alone, that swing disappears: Georgetown holds at
    5.52 km and North Jetty at 2.58 km across the same sweep.

    ASSUMES exactly one seaward opening. Winyah Bay has one. A fishery with
    two genuinely disconnected true mouths would silently lose the smaller
    one to this filter -- worth checking explicitly the first time this runs
    against a new fishery, since the Phase 2 stamp-out list (Charleston,
    Awendaw, Murrells Inlet) means new fisheries arrive here unauthored, and
    nothing else in this module would catch a dropped second mouth.
    """
    grid = to_grid(mask.astype("float64"), spec.flat_index, spec.shape, fill=0.0) > 0.5
    labels, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background is never the "largest component"
    largest = int(np.argmax(counts))
    return from_grid(labels == largest, spec.flat_index)


def ocean_seed_mask(
    spec,
    ocean_boundary_utm_km: list,
    bed_elev_m: np.ndarray,
    ocean_max_z_m: float,
) -> np.ndarray:
    """In-domain cells that are the sea itself: the largest contiguous run of
    cells that are on the domain's outer edge, below `ocean_max_z_m`, AND
    inside the authored ocean polygon.

    The edge/depth/polygon triple matches `mesh.classify_boundary`'s own
    criterion exactly -- "a segment whose midpoint is deep AND inside that
    polygon is the true seaward opening" -- carried over from ring segments
    to library cells. `ocean_boundary_utm_km` alone is NOT a "this area is
    the sea" test: it is `classify_boundary`'s ring-segment filter, authored
    generously (a box spanning much of the bay's mouth) so a HANDFUL of
    boundary-ring midpoints near the true coast fall inside it regardless of
    exactly how the shoreline was traced -- 10 of Winyah's 605 ring segments
    pass it. Testing polygon-containment alone against every in-domain CELL
    instead of every boundary SEGMENT made 40.5% of this fishery's domain
    (238,106 of 587,325 cells) read as "at the sea": Georgetown Lighthouse,
    a few km up-estuary by every other measure, landed at 0.00 km, tied with
    the jetty itself.

    The edge/depth/polygon triple alone is still not enough: it passes
    scattered shoreline fragments deep inside the bay along with the true
    mouth, and which fragments pass is sensitive to exactly where
    `ocean_max_z_m` falls -- see `_largest_component`, which this function
    applies as a final step to keep only the one contiguous run that is
    actually the coast.

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
        # A self-intersecting (bowtied) hand-authored ring makes .contains()
        # below misclassify silently -- Shapely doesn't raise, it just gives
        # an answer nobody authored. Same failure mode as
        # `mesh.classify_boundary`'s identical check on the same polygon.
        raise ValueError(
            "ocean_boundary_utm_km is not a valid polygon (self-"
            "intersecting or otherwise malformed) -- fix the authored "
            "vertices in the fishery YAML before computing the along-"
            "estuary distance field"
        )
    inside = np.fromiter(
        (poly.contains(Point(x, y)) for x, y in zip(spec.xs, spec.ys, strict=True)),
        dtype=bool,
        count=spec.xs.size,
    )
    deep = np.isfinite(bed_elev_m) & (bed_elev_m < ocean_max_z_m)
    candidates = inside & _domain_edge_mask(spec) & deep
    if not candidates.any():
        # along_estuary_km's own empty-seed check is seed-agnostic and can't
        # say why; this function is where all three conditions -- and their
        # inputs -- are visible enough to diagnose which one emptied it.
        raise ValueError(
            "no cell is simultaneously inside ocean_boundary_utm_km, on the "
            "domain's outer edge, and below ocean_max_z_m -- check that the "
            "polygon actually overlaps real coastline, that ocean_max_z_m "
            "isn't excluding every deep cell, and that the domain mask "
            "reaches the polygon at all"
        )
    return _largest_component(candidates, spec)


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
    if z_lib.shape != spec.shape:
        # The whole point of this function is that z_lib and spec share a
        # source pixel per cell. If grid_spec's own decimation ever changes
        # without this `step` formula changing to match, `z_lib[rows, cols]`
        # below would silently read the wrong pixels -- fancy indexing does
        # not bounds-check against an intended shape, only against z_lib's
        # actual one. Fail loudly here instead.
        raise ValueError(
            f"bed elevation grid {z_lib.shape} does not match the library "
            f"grid {spec.shape} -- flowlib.grid_spec's mask decimation and "
            "this function's z[::step, ::step] have drifted apart"
        )
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    return z_lib[rows, cols].astype("float64")


def build_distance_field(slug: str, fishery: Fishery) -> Path:
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    md = fishery.model_domain
    bed_elev_m = _bed_elevation_m(slug, fishery, spec)
    # The SALT SOURCE, not the tidal boundary -- see ModelDomain for why the
    # two differ and what conflating them cost on Winyah.
    seeds = ocean_seed_mask(
        spec, md.salt_source_polygon_utm_km, bed_elev_m, md.ocean_max_z_m
    )
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


# Distance to the estuary's main stem, THROUGH WATER, above which a station
# is treated as sitting on a different branch. Measured 2026-08-24 against the
# real field: on-axis stations (Winyah Bay main channel plus the bay's own
# NERRS sondes) span 0.048-1.604 km, off-axis ones (North Inlet, Town Creek,
# the AIWW, the North Santee) span 7.798-11.918 -- a 4.8x gap with only Jones
# Creek / Mud Bay (2.170) inside it. This value sits in that gap.
#
# Jones Creek falling OUTSIDE is correct rather than a miss: Mud Bay is the
# physical connection between Winyah Bay and North Inlet, so its membership is
# genuinely ambiguous and the conservative answer is to leave it out of a fit
# that assumes one branch.
ON_AXIS_MAX_KM = 2.0

_STEM_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def descent_path(field_grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Steepest-descent cells from `start` down to a seed, on a 2-D field.

    Follows the same 8-connectivity `along_estuary_km` used to BUILD the
    field, so the path it traces is one the distance actually measures along.
    Stops when no neighbour is strictly lower, which at a seed cell (distance
    0.0) is immediate.
    """
    r, c = start
    out = [(r, c)]
    rows, cols = field_grid.shape
    while True:
        best = None
        for dr, dc in _STEM_NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and not np.isnan(field_grid[rr, cc]):
                if best is None or field_grid[rr, cc] < best[0]:
                    best = (field_grid[rr, cc], rr, cc)
        if best is None or best[0] >= field_grid[r, c]:
            return out
        _, r, c = best
        out.append((r, c))


def main_stem_mask(fishery: Fishery, spec, field: np.ndarray) -> np.ndarray:
    """The estuary's main channel: the union of the descent paths from each
    river inflow point down to the mouth.

    The inflow points are already authored in the fishery YAML (they are where
    `Inlet_operator` injects discharge), so this adds no new hand-authored
    geometry. Walking downhill from each one traces the channel the river
    water actually takes, which is the line the salt front advances along.
    """
    from rasterio.warp import transform as warp_transform

    grid = to_grid(field, spec.flat_index, spec.shape, fill=np.nan)
    lons = [r.inflow_lonlat[0] for r in fishery.rivers]
    lats = [r.inflow_lonlat[1] for r in fishery.rivers]
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", lons, lats)

    stem = np.zeros(spec.shape, dtype=bool)
    for x, y in zip(xs, ys, strict=True):
        d2 = (spec.xs - x) ** 2 + (spec.ys - y) ** 2
        i = int(np.argmin(d2))
        start = np.unravel_index(spec.flat_index[i], spec.shape)
        for rr, cc in descent_path(grid, (int(start[0]), int(start[1]))):
            stem[rr, cc] = True
    return from_grid(stem, spec.flat_index)


def build_stem_distance_field(slug: str, fishery: Fishery) -> Path:
    """Distance from every cell to the main stem, through water.

    The SAME Dijkstra as the along-estuary field, seeded from the stem
    instead of the ocean. Reused rather than reimplemented so the two fields
    cannot drift apart in connectivity or diagonal cost.
    """
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    field = load_distance_field(slug)
    stem = main_stem_mask(fishery, spec, field)
    d = along_estuary_km(spec, stem)
    path = fishery_data_dir(slug) / "stem_km.npy"
    np.save(path, d.astype("float32"))
    return path


def load_stem_distance_field(slug: str) -> np.ndarray:
    path = fishery_data_dir(slug) / "stem_km.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"no distance-to-stem field at {path} -- run "
            f"`tidescout salinity stem {slug}` first"
        )
    return np.load(path)
