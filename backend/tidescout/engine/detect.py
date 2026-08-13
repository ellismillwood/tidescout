from dataclasses import dataclass, field

import numpy as np
from rasterio import features as rio_features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from skimage.morphology import disk, opening, skeletonize

from tidescout.models import FeatureThresholds, Fishery

# Default only. Static detectors ask "is this cell wet at a representative
# water level"; ANUGA has a time-varying free surface, so the two notions of
# "wet" are about to diverge. Callers pass their own.
WET_LEVEL_M = 0.0


@dataclass
class Feature:
    type: str
    geometry: object
    attrs: dict = field(default_factory=dict)


def orientation_deg(geometry) -> float:
    """Bearing (0-180, compass convention: 0/180 = north-south, 90 = east-west)
    of the geometry's PCA major axis over its exterior/line coords.

    Degenerate guard: a geometry with fewer than 2 distinct coordinate points
    (e.g. a single-point Point, or a zero-length line) has no defined axis.
    np.cov on a single sample divides by (N - ddof) = 0, producing NaN/inf
    that np.linalg.eigh then rejects with a LinAlgError crash. None of the
    current callers (_mask_polygons' rings always have >=4 coords; seed_jetty
    linestrings require >=2 config-validated vertices) hit this, but
    orientation_deg is a public part of this module's interface and a future
    caller could pass a degenerate geometry -- fail soft (0.0) rather than
    crash the pipeline.
    """
    coords = np.asarray(
        geometry.exterior.coords if hasattr(geometry, "exterior") else geometry.coords
    )
    if len(coords) < 2 or not np.any(np.ptp(coords, axis=0) > 0):
        return 0.0
    xy = coords - coords.mean(axis=0)
    cov = np.cov(xy.T)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, int(np.argmax(evals))]
    bearing = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 180.0
    return float(bearing)


def _mask_polygons(
    mask: np.ndarray,
    transform: Affine,
    min_area_m2: float,
    cell_m: float,
    max_area_m2: float | None = None,
):
    """Polygonise a boolean mask, dropping components outside the size band.

    The upper bound is not cosmetic: without it a single connected component
    can span the whole estuary (a 47 km2 'bar' covering 21 x 35 km was what
    the real Winyah raster produced), and every point-in-polygon join against
    it returns distance 0, which destroys 'nearest feature to this cell'.
    """
    polys = []
    cell_area = cell_m * cell_m
    for geom, val in rio_features.shapes(mask.astype("uint8"), transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area < min_area_m2 or g.area < cell_area:
            continue
        if max_area_m2 is not None and g.area > max_area_m2:
            continue
        polys.append(g)
    return polys


def detect_dropoffs(
    z: np.ndarray,
    slope: np.ndarray,
    t: FeatureThresholds,
    transform: Affine,
    wet_level_m: float = WET_LEVEL_M,
) -> list[Feature]:
    wet = ~np.isnan(z) & (z < wet_level_m)
    mask = wet & (slope >= t.dropoff_slope_deg)
    cell = abs(transform.a)
    out = []
    for g in _mask_polygons(mask, transform, t.hole_min_area_m2 / 4.0, cell):
        sel = rio_features.geometry_mask([g], z.shape, transform, invert=True)
        vals = slope[sel]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_slope = float(np.mean(vals))
        p90_slope = float(np.percentile(vals, 90))
        max_slope = float(np.max(vals))
        stat = {"p90": p90_slope, "max": max_slope, "mean": mean_slope}[
            t.wall_slope_estimator
        ]
        ftype = "wall" if stat >= t.wall_slope_deg else "dropoff"
        out.append(
            Feature(
                ftype,
                g,
                {
                    "area_m2": float(g.area),
                    "mean_slope_deg": mean_slope,
                    "p90_slope_deg": p90_slope,
                    "max_slope_deg": max_slope,
                    "min_z": float(np.nanmin(z[sel])),
                    "max_z": float(np.nanmax(z[sel])),
                    "orientation_deg": orientation_deg(g),
                },
            )
        )
    return out


def detect_holes(
    z: np.ndarray,
    t: FeatureThresholds,
    cell_m: float,
    transform: Affine,
    wet_level_m: float = WET_LEVEL_M,
) -> list[Feature]:
    filled = np.nan_to_num(z, nan=1000.0)
    closed = ndimage.grey_closing(filled, size=15)
    pocket = (closed - filled) > t.hole_delta_m
    pocket &= ~np.isnan(z) & (z < wet_level_m)
    out = []
    for g in _mask_polygons(
        pocket, transform, t.hole_min_area_m2, cell_m, max_area_m2=t.hole_max_area_m2
    ):
        sel = rio_features.geometry_mask([g], z.shape, transform, invert=True)
        rim = float(np.nanmax((closed - filled)[sel]))
        out.append(
            Feature(
                "hole",
                g,
                {
                    "area_m2": float(g.area),
                    "depth_below_rim_m": rim,
                    "min_z": float(np.nanmin(z[sel])),
                },
            )
        )
    return out


def detect_flats(
    z: np.ndarray, slope: np.ndarray, t: FeatureThresholds, transform: Affine
) -> list[Feature]:
    lo, hi = t.flat_band_m
    mask = ~np.isnan(z) & (z >= lo) & (z < hi) & (slope <= t.flat_max_slope_deg)
    cell = abs(transform.a)
    return [
        Feature("flat", g, {"area_m2": float(g.area)})
        for g in _mask_polygons(
            mask, transform, 4.0 * t.hole_min_area_m2, cell, max_area_m2=t.flat_max_area_m2
        )
    ]


def detect_creek_mouths(
    z: np.ndarray,
    t: FeatureThresholds,
    cell_m: float,
    transform: Affine,
    wet_level_m: float = WET_LEVEL_M,
) -> list[Feature]:
    wet = ~np.isnan(z) & (z < wet_level_m)
    if not wet.any():
        return []
    open_radius = max(1, round(60.0 / cell_m))
    # mode="ignore" (not the opening()/erosion default "reflect") matches the
    # semantics we rely on: a huge, uninterrupted wet body must not be eroded
    # away near the *array's own* edge just because the footprint runs past
    # it -- verified empirically (test_no_features_on_empty_basin's all-wet
    # 200x200 grid survives opening exactly, no boundary artifact creeks).
    opened = opening(wet, disk(open_radius), mode="ignore")
    labels, n = ndimage.label(opened)
    if n == 0:
        return []
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))
    open_water = labels == (1 + int(np.argmax(sizes)))
    creeks = wet & ~ndimage.binary_dilation(open_water, iterations=2)
    if not creeks.any():
        return []
    skel = skeletonize(creeks)
    dist_to_open = ndimage.distance_transform_edt(~open_water) * cell_m
    width = ndimage.distance_transform_edt(creeks) * cell_m
    # `opening()` rounds convex corners of the wet region wherever one sits
    # near nodata or the array's own edge (a real coastline will have some
    # -- a point, a headland, or just a NaN-bordered mosaic's valid-data
    # boundary). That rounding leaves 1-3 px wet-but-not-opened slivers
    # immediately adjacent to open_water, which `creeks` picks up and
    # `skeletonize` reduces to a 1-2 px stub -- indistinguishable from a
    # real short creek's mouth by proximity-to-open-water alone (confirmed:
    # a genuine 60-100 m feeder creek, well within what Winyah's marsh
    # actually has, also skeletonizes to 2-5 px, so filtering by skeleton
    # *length* drops real creeks along with the artifacts -- verified: both
    # were being dropped before this fix).
    #
    # Origin, not length, is what actually discriminates them: the artifact
    # is *always* within a couple cells of nodata or the boundary (that's
    # the mechanism that produces it -- the disk footprint reaches past the
    # edge/nodata corner and clips a chunk of the wet region right there).
    # A real creek mouth has no reason to sit there. Exclude candidate mouth
    # pixels within `artifact_reach_cells` of either, rather than filtering
    # by how long the skeleton component is. Verified: eliminates all
    # corner/point stubs on both a rectangular NaN border and an irregular
    # diamond-shaped coastline (0 spurious mouths, was 4 on the rectangle),
    # while every real creek from 60 m up finds its mouth (was: only
    # >=110 m survived under the length filter).
    artifact_reach_cells = 2
    near_nodata = ndimage.binary_dilation(np.isnan(z), iterations=artifact_reach_cells)
    near_edge = np.zeros(z.shape, dtype=bool)
    near_edge[:artifact_reach_cells, :] = True
    near_edge[-artifact_reach_cells:, :] = True
    near_edge[:, :artifact_reach_cells] = True
    near_edge[:, -artifact_reach_cells:] = True
    artifact_zone = near_nodata | near_edge
    mouth_px = skel & (dist_to_open <= t.mouth_search_radius_m) & ~artifact_zone
    rows, cols = np.nonzero(mouth_px)
    if rows.size == 0:
        return []
    pts = [Point(transform * (c + 0.5, r + 0.5)) for r, c in zip(rows, cols, strict=True)]
    widths = [2.0 * float(width[r, c]) for r, c in zip(rows, cols, strict=True)]
    # cluster nearby mouth pixels into one feature
    merged = unary_union([p.buffer(3.0 * t.mouth_search_radius_m) for p in pts])
    clusters = merged.geoms if hasattr(merged, "geoms") else [merged]
    out = []
    for cl in clusters:
        members = [(p, w) for p, w in zip(pts, widths, strict=True) if cl.contains(p)]
        if not members:
            continue
        cx = float(np.mean([p.x for p, _ in members]))
        cy = float(np.mean([p.y for p, _ in members]))
        out.append(
            Feature(
                "creek_mouth",
                Point(cx, cy),
                {"creek_width_m": float(np.median([w for _, w in members]))},
            )
        )
    return out


def detect_bars(
    z: np.ndarray, t: FeatureThresholds, cell_m: float, transform: Affine
) -> list[Feature]:
    shallow = ~np.isnan(z) & (z > t.deep_min_m) & (z < t.shallow_max_m)
    deep = ~np.isnan(z) & (z <= t.deep_min_m)
    if not shallow.any() or not deep.any():
        return []
    out = []
    labels, n = ndimage.label(shallow)
    deep_dilated = ndimage.binary_dilation(deep, iterations=3)
    for i in range(1, n + 1):
        region = labels == i
        area = float(region.sum()) * cell_m * cell_m
        if area < t.bar_min_area_m2:
            continue
        # border_value=1 (not scipy's default 0): default erosion treats
        # space beyond the array edge as background, so a region pixel
        # sitting on the array's own boundary always erodes away and gets
        # counted as "boundary" regardless of whether real deep water is
        # anywhere nearby -- verified: a ridge shifted to touch the array
        # edge read pct_deep_boundary=0.9557 (18 phantom edge-boundary px
        # diluting the denominator) vs the true 1.0 with border_value=1,
        # identical on every region that doesn't touch the array edge.
        boundary = region & ~ndimage.binary_erosion(region, border_value=1)
        pct = float((boundary & deep_dilated).sum()) / max(1, int(boundary.sum()))
        if pct < 0.25:
            continue
        # min_area_m2=0.0: already gated on `area` above (raw pixel count),
        # which is exactly the polygon's eventual vector area for an
        # orthogonal raster-to-vector conversion -- re-checking the same
        # threshold here would be redundant. _mask_polygons' own cell_area
        # sanity floor still applies.
        polys = _mask_polygons(region, transform, 0.0, cell_m, max_area_m2=t.bar_max_area_m2)
        for g in polys:
            out.append(
                Feature(
                    "bar",
                    g,
                    {
                        "area_m2": float(g.area),
                        "pct_deep_boundary": pct,
                        "orientation_deg": orientation_deg(g),
                    },
                )
            )
    return out


def seed_jetties(fishery: Fishery, lonlat_to_grid_xy) -> list[Feature]:
    out = []
    for j in fishery.jetties:
        lons = [c[0] for c in j.coords]
        lats = [c[1] for c in j.coords]
        xs, ys = lonlat_to_grid_xy(lons, lats)
        out.append(Feature("jetty", LineString(zip(xs, ys, strict=True)), {"name": j.name}))
    return out
