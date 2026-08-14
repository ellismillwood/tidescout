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

    The composite bucket boundaries are the calibrated percentiles from Task 1.
    'low' and 'high' sit at those boundaries; 'med' at their midpoint. Each
    river takes its configured share of the composite.
    """
    b = fishery.discharge_buckets
    composite_cfs = {
        "low": b.low_below_cfs,
        "med": 0.5 * (b.low_below_cfs + b.high_above_cfs),
        "high": b.high_above_cfs,
    }[bucket]
    total_weight = sum(r.weight for r in fishery.rivers) or 1.0
    return {
        r.name: composite_cfs * CFS_TO_M3S * (r.weight / total_weight)
        for r in fishery.rivers
    }
