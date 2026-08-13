from dataclasses import dataclass, field

import numpy as np
from rasterio import features as rio_features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from skimage.morphology import disk, opening, skeletonize

from tidescout.models import FeatureThresholds, Fishery

WET_LEVEL_M = 0.0  # approximate mean-water wetness for static detection


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


def _mask_polygons(mask: np.ndarray, transform: Affine, min_area_m2: float, cell_m: float):
    polys = []
    cell_area = cell_m * cell_m
    for geom, val in rio_features.shapes(mask.astype("uint8"), transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area >= min_area_m2 and g.area >= cell_area:
            polys.append(g)
    return polys


def detect_dropoffs(
    z: np.ndarray, slope: np.ndarray, t: FeatureThresholds, transform: Affine
) -> list[Feature]:
    wet = ~np.isnan(z) & (z < WET_LEVEL_M)
    mask = wet & (slope >= t.dropoff_slope_deg)
    cell = abs(transform.a)
    out = []
    for g in _mask_polygons(mask, transform, t.hole_min_area_m2 / 4.0, cell):
        sel = rio_features.geometry_mask([g], z.shape, transform, invert=True)
        mean_slope = float(np.nanmean(slope[sel])) if sel.any() else 0.0
        ftype = "wall" if mean_slope >= t.wall_slope_deg else "dropoff"
        out.append(
            Feature(
                ftype,
                g,
                {
                    "area_m2": float(g.area),
                    "mean_slope_deg": mean_slope,
                    "min_z": float(np.nanmin(z[sel])),
                    "max_z": float(np.nanmax(z[sel])),
                    "orientation_deg": orientation_deg(g),
                },
            )
        )
    return out


def detect_holes(
    z: np.ndarray, t: FeatureThresholds, cell_m: float, transform: Affine
) -> list[Feature]:
    filled = np.nan_to_num(z, nan=1000.0)
    closed = ndimage.grey_closing(filled, size=15)
    pocket = (closed - filled) > t.hole_delta_m
    pocket &= ~np.isnan(z) & (z < WET_LEVEL_M)
    out = []
    for g in _mask_polygons(pocket, transform, t.hole_min_area_m2, cell_m):
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
        for g in _mask_polygons(mask, transform, 4.0 * t.hole_min_area_m2, cell)
    ]


def detect_creek_mouths(
    z: np.ndarray, t: FeatureThresholds, cell_m: float, transform: Affine
) -> list[Feature]:
    wet = ~np.isnan(z) & (z < WET_LEVEL_M)
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
    # `opening()` rounds convex corners of the wet region (any real coastline
    # will have some -- a point, a headland, or just the valid-data
    # bounding box after a NaN-bordered mosaic/warp). That rounding leaves
    # 1-3 px slivers wet-but-not-opened right next to open_water, which
    # `creeks` then picks up and `skeletonize` reduces to a 1-2 px stub --
    # indistinguishable from a real mouth by the proximity test below.
    # Confirmed empirically on a NaN-bordered flat basin (no real creek
    # anywhere): 4 spurious stubs, exactly one per rectangle corner, each
    # skeletonizing to 1-2 px. A genuine creek is long and thin (that's
    # *why* it survives the opening at all, per this function's docstring),
    # so requiring each skeleton connected-component to have at least
    # `open_radius` pixels -- the same length scale that produced the
    # artifact -- clears the corner stubs (verified: max 2 px) while a real
    # creek's skeleton (verified: 95 px on the synthetic creek) is untouched.
    skel_labels, skel_n = ndimage.label(skel, structure=np.ones((3, 3)))
    if skel_n:
        comp_sizes = ndimage.sum_labels(
            np.ones_like(skel_labels), skel_labels, index=range(1, skel_n + 1)
        )
        keep = {i + 1 for i, sz in enumerate(comp_sizes) if sz >= open_radius}
        skel = np.isin(skel_labels, list(keep))
    dist_to_open = ndimage.distance_transform_edt(~open_water) * cell_m
    width = ndimage.distance_transform_edt(creeks) * cell_m
    mouth_px = skel & (dist_to_open <= t.mouth_search_radius_m)
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
        boundary = region & ~ndimage.binary_erosion(region)
        pct = float((boundary & deep_dilated).sum()) / max(1, int(boundary.sum()))
        if pct < 0.25:
            continue
        polys = _mask_polygons(region, transform, t.bar_min_area_m2, cell_m)
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
