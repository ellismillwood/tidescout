"""Sample derived structure at the static feature inventory.

The join between Plan 2's features and Plan 4's fields. Pure: the caller loads
the library state and the GeoJSON and hands both in.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import shape

from tidescout.engine import structure
from tidescout.engine.flow import wet_mask
from tidescout.models import StructureThresholds

# Fields summarised by their best cell rather than their mean. An ambush point,
# a seam and a convergence front are all defined by their strongest cell -- a
# 150 m disc over a 20 m grid holds ~175 cells, and averaging would dilute a
# real pocket into the channel around it. Speed and depth-derived quantities
# describe the feature as a whole, so those take the mean, and `eddy_share` is
# a mean BY DEFINITION: it is the disc-average of a 0/1 indicator.
_MAX_FIELDS = frozenset({"ambush", "strain", "okubo_w", "convergence"})

# Every field `sample_features` reduces, in the order the table reads.
_SAMPLED_FIELDS = ("speed", "ambush", "strain", "okubo_w", "convergence", "eddy_share")


@dataclass
class FeatureMetrics:
    """One feature's derived structure at one flow state.

    `okubo_w` is MAX-reduced over the disc. Since W > 0 is a seam and W < 0 is
    an eddy, that makes it a SEAM detector and nothing else: a max returns the
    most seam-like cell present, so it structurally cannot report an eddy
    unless the entire disc rotates. Measured over the shipped winyah-bay
    `mean_med` library, all 26 phases: of 13,588 finite per-feature `okubo_w`
    samples exactly 2 were negative, at -8.2e-7 and -4.5e-9 -- both more than
    ten times inside the `quiet_w = 1e-5` dead band, i.e. not eddies at all.
    Zero samples ever crossed -quiet_w.

    `eddy_share` is the eddy channel that leaves `okubo_w` alone: the fraction
    of the feature's disc that `structure.classify_structure` labels an eddy
    (W < -quiet_w), applying at the feature level the same dead band that
    function applies at the cell level. Without that band the obvious
    alternative -- an unthresholded `nanmin(okubo_w) < 0` -- flags 520 of the
    526 evaluated features on floating-point noise, against 104 that genuinely
    contain a rotation-dominated cell.

    THE DENOMINATOR IS THE WET IN-DOMAIN CELLS OF THE DISC, not every cell of
    it. Dry cells are excluded rather than counted as "not an eddy", for the
    same reason `structure_fields` masks them twice: ANUGA writes u = v = 0.0
    on dry ground, and letting that stand in as valid still water is the exact
    artifact the double mask exists to remove. So `eddy_share` answers "of the
    water that is here, how much of it is rotating", and a half-dry disc is
    judged on its water, not on its mud. `n_cells` and `wet_fraction` are
    reported alongside for callers that need the other denominator.

    `flood_phase` is a CIRCULAR mean -- see `_circular_mean_phase`.
    """

    key: str
    type: str
    speed: float
    ambush: float
    strain: float
    okubo_w: float
    convergence: float
    eddy_share: float
    wet_fraction: float
    flood_phase: float
    n_cells: int


def structure_fields(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    spec,
    thresholds: StructureThresholds | None = None,
) -> dict[str, np.ndarray]:
    """Run the whole derived-structure chain for one flow state.

    Scatters the masked library arrays onto the raster, differentiates there,
    then gathers back -- gradients need neighbours, and the stored arrays have
    none. Computed per call rather than stored: the algebra costs milliseconds,
    the storage would cost ~2 GB and a library rebuild, and computing on the
    phase-interpolated velocity is more correct than interpolating these
    nonlinear quantities between phases.

    ANUGA reports u = v = 0.0 on a dry cell, not NaN (regimes.py's
    `_centroid_speed` zeroes momentum where depth <= 0.01 m rather than
    masking it out). Left alone, a dry marsh cell sitting hard against a fast
    channel is indistinguishable from genuine slack water -- a slow cell
    beside a fast one is exactly the shape `ambush_contrast` hunts for, so
    every dry cell next to a channel would score as a perfect ambush pocket,
    and you cannot fish dry marsh. The wet mask is built on the 1-D array
    (via `wet_mask`, the single source of truth for the threshold) and
    scattered to the grid alongside u/v, then applied to ug/vg/speed_g --
    the inputs to every derivative below -- so the NaN it introduces
    propagates through the gradients and through `ambush_contrast`'s own
    NaN-aware max filter the same way out-of-domain NaN already does.

    That propagation is NOT enough on its own for strain/okubo_w/convergence.
    `np.gradient`'s central difference at index i is (a[i+1] - a[i-1]) / 2h --
    it never reads a[i] itself. A dry cell that is only one cell wide along
    the differencing axis therefore has its gradient bridged straight across
    from its two wet neighbours, landing on a finite, plausible-looking value
    instead of NaN (confirmed against a real run: an isolated dry cell on a
    shear line returned strain=0.03, okubo_w=0.0009, convergence=-0.03, all
    finite, while speed and ambush correctly read NaN there). Since those
    three fields are MAX-reduced per feature, one such cell inside a 150 m
    disc can set a feature's entire reported seam or convergence score from
    unfishable dry land. `dry_g` is re-applied explicitly to each of them
    after the tensor is computed, exactly as `ambush_contrast` already does
    internally for its own NaN-aware max filter -- the two masking passes
    (inputs, then outputs) close different gaps and neither one alone is
    sufficient.
    """
    t = thresholds or StructureThresholds()
    ug = structure.to_grid(u, spec.flat_index, spec.shape)
    vg = structure.to_grid(v, spec.flat_index, spec.shape)

    wet = wet_mask(depth)
    wet_g = structure.to_grid(wet.astype("float64"), spec.flat_index, spec.shape, fill=0.0)
    dry_g = wet_g <= 0.5
    ug = np.where(dry_g, np.nan, ug)
    vg = np.where(dry_g, np.nan, vg)
    speed_g = np.hypot(ug, vg)

    tensor = structure.gradient_tensor(ug, vg, spec.cell_m)
    okubo_w = np.where(dry_g, np.nan, structure.okubo_weiss(tensor))
    # A 0/1 indicator per cell; `sample_features` MEAN-reduces it, so the
    # per-feature value is the eddy share of the feature's wet disc -- see
    # FeatureMetrics for why "wet" is the honest denominator.
    #
    # `classify_structure` returns int8, so NaN cannot survive it: a cell whose
    # gradient is undefined comes back 0, "quiet water", which is a claim
    # rather than a gap. That covers more than the dry cells themselves -- a
    # WET cell one in from a dry edge also has an undefined gradient, because
    # np.gradient's central difference reads its NaN neighbour. Masking on
    # `dry_g` alone would count every one of those as "definitely not an eddy"
    # and quietly deflate the share along exactly the shorelines this signal
    # lives on. Re-masking against the already-NaN-carrying `okubo_w` drops
    # them from the denominator instead.
    eddy = np.where(
        np.isfinite(okubo_w),
        (structure.classify_structure(tensor, t.quiet_w) == -1).astype("float64"),
        np.nan,
    )
    fields_2d = {
        "speed": speed_g,
        "ambush": structure.ambush_contrast(speed_g, spec.cell_m, t.ambush_radius_m),
        "strain": np.where(dry_g, np.nan, structure.strain_rate(tensor)),
        "okubo_w": okubo_w,
        "convergence": np.where(dry_g, np.nan, structure.convergence(tensor)),
        "eddy_share": eddy,
    }
    return {k: structure.from_grid(g, spec.flat_index) for k, g in fields_2d.items()}


def sample_features(
    features: list[dict],
    spec,
    fields: dict[str, np.ndarray],
    schedule=None,
    radius_m: float = 150.0,
    already_projected: bool = False,
) -> list[FeatureMetrics]:
    """Per-feature summary of every field, over the cells within `radius_m`.

    `already_projected` is for tests that build coordinates directly in the
    grid CRS; production passes GeoJSON in EPSG:4326 and `_sampling_anchors`
    reprojects.
    """
    xs, ys = _sampling_anchors(features, spec, already_projected)

    out = []
    r2 = radius_m**2
    for f, fx, fy in zip(features, xs, ys, strict=True):
        sel = (spec.xs - fx) ** 2 + (spec.ys - fy) ** 2 <= r2
        n = int(sel.sum())
        vals = {}
        for name in _SAMPLED_FIELDS:
            arr = fields.get(name)
            if arr is None or n == 0:
                vals[name] = float("nan")
                continue
            here = arr[sel]
            reducer = np.nanmax if name in _MAX_FIELDS else np.nanmean
            vals[name] = float(reducer(here)) if np.isfinite(here).any() else float("nan")

        wet_fraction = flood_phase = float("nan")
        if schedule is not None and n:
            wf = schedule.wet_fraction[sel]
            fp = schedule.flood_phase[sel]
            if np.isfinite(wf).any():
                wet_fraction = float(np.nanmean(wf))
            if np.isfinite(fp).any():
                flood_phase = _circular_mean_phase(fp)

        out.append(
            FeatureMetrics(
                key=f["id"],
                type=f["properties"]["type"],
                wet_fraction=wet_fraction,
                flood_phase=flood_phase,
                n_cells=n,
                **vals,
            )
        )
    return out


def _circular_mean_phase(phases: np.ndarray) -> float:
    """Mean of tidal phases on the CIRCLE, returned in [0, 1).

    Phase wraps, so 0.95 and 0.05 are 0.1 of a cycle apart and their centre is
    0.0 -- low water -- not 0.5, which is the other half of the tide.
    `pipeline/schedule.py` spends its module docstring establishing that phase
    is circular and that an ordinary median of one "lands on whichever side of
    0.5 that cluster happens to fall, which is an artifact of the cut point,
    not of the physics"; this used to be `np.nanmedian` and undid that.

    The standard resultant-vector construction: average the unit vectors at
    angle 2*pi*phase and read the angle back with atan2.

    Measured on the shipped winyah-bay `mean_med` library, against the ordinary
    median it replaces: of the 301 features whose disc carries any finite
    `flood_phase`, 261 move at all, 159 by more than 0.01 of a cycle, 19 by
    more than 0.05, and 3 by more than 0.1. The largest single correction is
    `flat-b6a1aec2d79d`, where the median read 0.0 -- "floods exactly at low
    water" -- for a cluster whose circular centre is 0.884, late on the ebb
    half. On a 12.42 h cycle 0.1 of a phase is about 75 minutes.

    Degenerate case, documented rather than guarded: a disc whose flood phases
    are spread right around the cycle has a resultant near zero and no
    meaningful central phase at all, so the angle atan2 recovers from it is
    noise. That is rare and visible -- one of those 301 features has a
    resultant length below 0.2 (it is 0.05, on 55 cells), against a median
    disc resultant of 0.89 -- and adding a cutoff would mean inventing a
    threshold at the merge gate to turn a weakly-determined answer into NaN.
    Exactly antipodal phases resolve to 0.0.
    """
    ang = 2.0 * np.pi * np.asarray(phases, dtype="float64")
    sin_bar = float(np.nanmean(np.sin(ang)))
    cos_bar = float(np.nanmean(np.cos(ang)))
    phase = float(np.arctan2(sin_bar, cos_bar) / (2.0 * np.pi) % 1.0)
    # `% 1.0` does not by itself guarantee the half-open range: atan2 lands a
    # hair BELOW zero for a cluster centred on the wrap (0.95 with 0.05 gives
    # about -1e-17), and 1.0 - 1e-17 rounds to exactly 1.0 in float64. Same
    # point on the circle, outside the documented interval, and a downstream
    # "is this the flood half" test would read it as the end of the ebb.
    return 0.0 if phase >= 1.0 else phase


def _sampling_anchors(
    features: list[dict], spec, already_projected: bool
) -> tuple[list[float], list[float]]:
    """Where each feature is sampled: its geometry's centroid, in the grid CRS.

    THIS MUST BE THE SAME POINT `detect.feature_key` HASHES INTO THE ID.
    Phase 3 keys scoring off `FeatureMetrics.key` and reads the metrics beside
    it, so an id that says "this place" while the metrics describe a disc
    centred somewhere else is a silent mismatch nothing downstream can detect.
    `feature_key` hashes `geometry.centroid`, so the sampling anchor is
    `geometry.centroid` — verified end to end against the shipped inventory:
    feeding these projected geometries back through `feature_key` reproduces
    all 2,162 ids exactly.

    This replaced an unweighted mean of the exterior ring's vertices (which
    also counted the duplicated closing vertex twice). That is not any named
    point of a polygon: measured against the true centroid over the real
    2,162-feature inventory it sat p50 7.6 m / p90 47.7 m / p99 183.5 m / max
    726.2 m away, put 471 of the 2,026 polygon features' anchors OUTSIDE their
    own polygon, and moved 33 of them further than the whole 150 m sampling
    radius.

    `representative_point()` was considered and measured too. It is guaranteed
    to lie inside the geometry (2,026 of 2,026 polygons, against 1,645 for the
    centroid), but it is a scanline construction, not a centre: its offset from
    the id anchor is p50 11.5 m / p99 376.8 m / max 2,228.1 m, WORSE than the
    ring-mean it would have replaced, and 110 features would sample beyond
    their own 150 m radius. Inside-ness buys nothing here anyway — the disc is
    never clipped to the polygon, it is only centred on it — while agreeing
    with the id is the entire point. Of the 381 polygons whose centroid falls
    outside their own ring (concave shapes and bar donuts), 73.5% have it
    within 20 m of the polygon and 97.9% within the sampling radius.

    The geometry is reprojected and THEN the centroid taken, not the other way
    round: a degree of longitude is ~0.83 of a degree of latitude in metres
    here, so an area- or length-weighted centroid computed in lon/lat is not
    the projected centroid. On the shipped inventory that shortcut is within
    0.21 m for every polygon but drifts up to 11.5 m on the long multi-vertex
    jetty lines, and the point of this function is to match `feature_key`
    exactly rather than nearly. `shapely.transform` hands the callable every
    vertex of every geometry in ONE batched call (188,003 vertices for
    winyah-bay), which costs 0.03 s against the ~2 s `sample_features` already
    spends per phase on 587,325-cell boolean masks.
    """
    if not features:
        return [], []
    geoms = [shape(f["geometry"]) for f in features]
    if not already_projected:
        from rasterio.warp import transform as warp_transform

        def to_grid_crs(coords: np.ndarray) -> np.ndarray:
            xs, ys = warp_transform(
                "EPSG:4326", spec.crs, coords[:, 0].tolist(), coords[:, 1].tolist()
            )
            return np.column_stack([xs, ys])

        geoms = shapely.transform(geoms, to_grid_crs)
    centroids = [g.centroid for g in geoms]
    return [c.x for c in centroids], [c.y for c in centroids]
