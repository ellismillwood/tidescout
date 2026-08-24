"""Boundary forcing for the regime runs.

Two unit traps live here, and both produce plausible-looking wrong answers
rather than errors:
  1. NOAA CO-OPS returns tide heights in FEET; ANUGA works in metres.
  2. CO-OPS predictions are on MLLW; the bathymetry is NAVD88. The offset is
     resolved per-station (Stations.tide_datum_offset_m), never assumed.
"""

import math
from collections.abc import Callable
from datetime import datetime

from tidescout.engine.flow import DISCHARGE_ORDER, bucket_flows
from tidescout.engine.tides import TideEvent, _cosine_height
from tidescout.models import Fishery

FT_TO_M = 0.3048
CFS_TO_M3S = 0.0283168

# Amplitude multipliers about mean water for the three tidal-range regimes.
RANGE_FACTORS = {"neap": 0.72, "mean": 1.0, "spring": 1.28}


def tide_function(
    events: list[TideEvent], datum_offset_m: float, start: datetime
) -> Callable[[float], float]:
    """Stage (m, NAVD88) as a function of seconds since `start`.

    Interpolates between predicted high/low water with the same cosine ramp the
    conditions engine already uses, so the boundary and the displayed tide curve
    cannot drift apart.
    """
    ordered = sorted(events, key=lambda e: e.time)
    if len(ordered) < 2:
        raise ValueError("need at least two tide events to force a boundary")
    times = [(e.time - start).total_seconds() for e in ordered]
    heights = [e.height_ft * FT_TO_M + datum_offset_m for e in ordered]

    def stage(t: float) -> float:
        if t <= times[0]:
            return heights[0]
        if t >= times[-1]:
            return heights[-1]
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                span = times[i + 1] - times[i]
                frac = 0.0 if span == 0 else (t - times[i]) / span
                return _cosine_height(heights[i], heights[i + 1], frac)
        return heights[-1]

    return stage


def range_scaled_tide(
    mean_range_m: float,
    bucket: str,
    period_s: float = 12.42 * 3600.0,
    mean_level_m: float = 0.0,
) -> Callable[[float], float]:
    """Idealised M2 cosine for a tidal-range regime.

    Regime runs are not hindcasts of a particular day -- they are the recurring
    flow patterns the spec's library is indexed by -- so a clean harmonic is
    the right forcing. Real predicted events drive validation runs instead.
    """
    amp = 0.5 * mean_range_m * RANGE_FACTORS[bucket]

    def stage(t: float) -> float:
        return mean_level_m + amp * math.cos(2.0 * math.pi * t / period_s)

    return stage


def river_inflow_m3s(fishery: Fishery, bucket: str) -> dict[str, float]:
    """Steady inflow per river for a discharge regime, in m^3/s.

    Inflow is steady across a run by design: the tidal cycle is 12.42 h and
    river discharge changes over days, so a time-varying inlet would model a
    process the simulation window cannot resolve.

    The composite bucket boundaries are the calibrated percentiles from Plan 3
    Task 1. Each river takes its `inflow_share` of that composite -- NOT its
    gauge `weight`, which serves the different job of building the composite.
    The split (and its guards against partial or mis-summed authorship) lives
    on `Fishery.branch_shares()`, shared with the runtime salinity path.

    The bucket -> cfs map is `engine.flow.bucket_flows` rather than a copy of
    it. This function decides what the library is BUILT at and bucket_flows
    decides what the runtime READS it as; when they were two literals they
    could disagree by a typo and every lookup would then be indexed to a flow
    the water was never run at, with nothing anywhere to catch it.
    """
    flows = bucket_flows(fishery.discharge_buckets)
    if bucket not in flows:
        known = ", ".join(flows)
        extra = ""
        if bucket in DISCHARGE_ORDER:
            # A real bucket the fishery just hasn't measured -- name the fix,
            # because "unknown bucket" would read as a typo and send whoever
            # hits it looking in the wrong place entirely.
            extra = (
                f" -- {bucket!r} is a recognised bucket but this fishery has no "
                f"discharge_buckets.{bucket}_cfs, so there is no flow to force it at"
            )
        raise ValueError(f"unknown discharge bucket {bucket!r}; have {known}{extra}")
    composite_cfs = flows[bucket]

    shares = fishery.branch_shares()
    return {
        r.name: composite_cfs * CFS_TO_M3S * s
        for r, s in zip(fishery.rivers, shares, strict=True)
    }
