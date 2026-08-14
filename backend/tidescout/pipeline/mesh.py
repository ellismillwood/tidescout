"""Bathymetry raster + authored domain polygon -> graded ANUGA mesh.

The shoreline of a real estuary is fractal: Winyah's raw water mask
polygonises to 7,581 exterior vertices, which no triangle generator handles
gracefully. Morphological close/open at the target cell scale removes creeks
too narrow to resolve anyway, and simplify(25 m) then takes the ring to ~486
vertices with area preserved. Meshing 250k triangles from that takes 0.5 s.
"""

import anuga
import numpy as np
import rasterio.features
from matplotlib.path import Path as MplPath
from scipy import ndimage
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union

from tidescout.models import Fishery
from tidescout.pipeline.bathy import read_bathy


def domain_mask(z, transform, fishery: Fishery, polygon=None) -> np.ndarray:
    """Boolean mask of water inside the model domain, single connected body."""
    md = fishery.model_domain
    level = md.wet_level_m if md else 1.5
    valid = ~np.isnan(z)
    wet = valid & (z < level)

    verts = polygon if polygon is not None else (md.polygon_utm_km if md else [])
    if verts:
        h, w = z.shape
        inv = ~transform
        poly_px = np.array([inv * (x * 1000.0, y * 1000.0) for x, y in verts])
        rows, cols = np.mgrid[0:h, 0:w]
        pts = np.column_stack([cols.ravel(), rows.ravel()])
        inside = MplPath(poly_px).contains_points(pts).reshape(h, w)
        wet &= inside

    lbl, n = ndimage.label(wet)
    if n == 0:
        raise ValueError("model domain contains no water")
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()


def clean_mask(mask: np.ndarray, cells: int) -> np.ndarray:
    """Drop sub-mesh-scale detail that would explode the boundary vertex count.

    Islands are deliberately filled rather than carved as mesh holes: ANUGA
    keeps them dry through wetting/drying because they sit above water, and
    that also lets them flood on a spring tide, which a hole never could.
    """
    k = np.ones((cells, cells))
    m = ndimage.binary_closing(mask, k)
    m = ndimage.binary_opening(m, k)
    m = ndimage.binary_fill_holes(m)
    lbl, _ = ndimage.label(m)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()


def domain_polygon(mask: np.ndarray, transform, simplify_m: float) -> Polygon:
    geoms = [
        shape(g)
        for g, v in rasterio.features.shapes(
            mask.astype("uint8"), mask=mask, transform=transform
        )
        if v == 1
    ]
    poly = unary_union(geoms)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly.simplify(simplify_m, preserve_topology=True)


def jetty_regions(fishery: Fishery, boundary: Polygon, epsg: int) -> list:
    """Refinement polygons around jetty structure, clipped inside the boundary.

    The spec singles out the Winyah jetty rips as must-catch, so they get their
    own zone rather than inheriting the channel's resolution.
    """
    from rasterio.warp import transform as warp_transform

    cfg = fishery.anuga
    lines = []
    for jetty in fishery.jetties:
        lons = [c[0] for c in jetty.coords]
        lats = [c[1] for c in jetty.coords]
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", lons, lats)
        lines.append(LineString(list(zip(xs, ys, strict=True))))
    if not lines:
        return []
    buf = unary_union([ln.buffer(cfg.jetty_radius_m) for ln in lines])
    buf = buf.intersection(boundary.buffer(-30.0))
    if buf.is_empty:
        return []
    parts = list(buf.geoms) if buf.geom_type == "MultiPolygon" else [buf]
    return [p.simplify(cfg.jetty_edge_m) for p in parts if not p.is_empty]


def edge_to_area(edge_m: float) -> float:
    """Equilateral triangle area for a target edge length."""
    return edge_m * edge_m * np.sqrt(3.0) / 4.0


def sample_to_centroids(domain, arr: np.ndarray, transform) -> np.ndarray:
    """Nearest-cell lookup of a raster onto mesh centroids."""
    cx, cy = domain.get_centroid_coordinates(absolute=True).T
    cols, rows = ~transform * (cx, cy)
    rows = np.clip(rows.astype(int), 0, arr.shape[0] - 1)
    cols = np.clip(cols.astype(int), 0, arr.shape[1] - 1)
    return arr[rows, cols]


def build_mesh(slug: str, fishery: Fishery):
    z, transform, meta = read_bathy(slug)
    md = fishery.model_domain
    cfg = fishery.anuga
    mask = clean_mask(domain_mask(z, transform, fishery), md.clean_cells if md else 3)
    boundary = domain_polygon(mask, transform, md.simplify_m if md else 25.0)
    ring = [list(c) for c in boundary.exterior.coords[:-1]]

    regions = [
        [[list(c) for c in p.exterior.coords[:-1]], edge_to_area(cfg.jetty_edge_m)]
        for p in jetty_regions(fishery, boundary, fishery.bathymetry.epsg)
    ]
    domain = anuga.create_domain_from_regions(
        ring,
        boundary_tags={"outer": list(range(len(ring)))},
        maximum_triangle_area=edge_to_area(cfg.base_edge_m),
        interior_regions=regions or None,
        verbose=False,
    )
    elev = sample_to_centroids(domain, z, transform)
    # nodata slivers at the boundary become land rather than NaN: a NaN
    # elevation propagates through the solver and kills the whole run.
    elev = np.where(np.isfinite(elev), elev, 1.0)
    domain.set_quantity("elevation", elev, location="centroids")
    return domain


def friction_field(domain, slug: str, fishery: Fishery) -> np.ndarray:
    """Per-centroid Manning n from the zones raster.

    zones.tif is a uint8 enum with 0 = nodata. Anything unclassified falls back
    to the flat value rather than to zero, because a Manning n of 0 is
    frictionless and would produce spectacular nonsense rather than an error.
    """
    import rasterio

    from tidescout.paths import fishery_data_dir

    cfg = fishery.anuga
    path = fishery_data_dir(slug) / "zones.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `tidescout bathy build {slug}` before meshing"
        )
    with rasterio.open(path) as src:
        zones = src.read(1)
        transform = src.transform
    z_at = sample_to_centroids(domain, zones, transform)
    # Enum verified against engine/terrain.py::zones in Task 4 (uint8):
    #   0 = nodata, 1 = land (z >= land_elev_m), 2 = shallow, 3 = mid-depth,
    #   4 = deep (z < deep_min_m).  FIVE values, not four.
    n = np.full(z_at.shape, cfg.manning_flat, dtype="float64")
    n[z_at == 4] = cfg.manning_channel   # deep -> smooth channel bed
    n[z_at == 3] = cfg.manning_flat      # mid-depth
    n[z_at == 2] = cfg.manning_flat      # shallow
    n[z_at == 1] = cfg.manning_marsh     # land / marsh -- rough, vegetated
    n[z_at == 0] = cfg.manning_marsh     # nodata slivers: treat as land, never 0
    return n
