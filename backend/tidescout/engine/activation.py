"""Sample derived structure at the static feature inventory.

The join between Plan 2's features and Plan 4's fields. Pure: the caller loads
the library state and the GeoJSON and hands both in.
"""

from dataclasses import dataclass

import numpy as np

from tidescout.engine import structure
from tidescout.engine.flow import wet_mask
from tidescout.models import StructureThresholds

# Fields summarised by their best cell rather than their mean. An ambush point,
# a seam and a convergence front are all defined by their strongest cell -- a
# 150 m disc over a 20 m grid holds ~175 cells, and averaging would dilute a
# real pocket into the channel around it. Speed and depth-derived quantities
# describe the feature as a whole, so those take the mean.
_MAX_FIELDS = frozenset({"ambush", "strain", "okubo_w", "convergence"})


@dataclass
class FeatureMetrics:
    key: str
    type: str
    speed: float
    ambush: float
    strain: float
    okubo_w: float
    convergence: float
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
    fields_2d = {
        "speed": speed_g,
        "ambush": structure.ambush_contrast(speed_g, spec.cell_m, t.ambush_radius_m),
        "strain": np.where(dry_g, np.nan, structure.strain_rate(tensor)),
        "okubo_w": np.where(dry_g, np.nan, structure.okubo_weiss(tensor)),
        "convergence": np.where(dry_g, np.nan, structure.convergence(tensor)),
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
    grid CRS; production passes GeoJSON in EPSG:4326 and this reprojects.
    """
    pts = [_centroid_lonlat(f) for f in features]
    if already_projected:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
    else:
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform(
            "EPSG:4326", spec.crs, [p[0] for p in pts], [p[1] for p in pts]
        )

    out = []
    r2 = radius_m**2
    for f, fx, fy in zip(features, xs, ys, strict=True):
        sel = (spec.xs - fx) ** 2 + (spec.ys - fy) ** 2 <= r2
        n = int(sel.sum())
        vals = {}
        for name in ("speed", "ambush", "strain", "okubo_w", "convergence"):
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
                flood_phase = float(np.nanmedian(fp))

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


def _centroid_lonlat(feature: dict) -> tuple[float, float]:
    """Representative point of a GeoJSON geometry, without a shapely round trip."""
    geom = feature["geometry"]
    coords = geom["coordinates"]
    kind = geom["type"]
    if kind == "Point":
        return float(coords[0]), float(coords[1])
    if kind == "LineString":
        pts = coords
    elif kind == "Polygon":
        pts = coords[0]
    else:
        raise TypeError(f"unsupported feature geometry: {kind}")
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)
