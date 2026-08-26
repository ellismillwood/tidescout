"""The factor pipeline. Pure -- conditions in, sub-scores and reasons out.

Two consumers share it: the fishery-wide hourly score and the per-feature
activation. They differ only in where `flow_speed` and `salinity` come
from -- the bay's representative values, or one feature's own.

Every factor obeys the same contract:
  - its weight comes from the species profile, never from code;
  - its response shape comes from a YAML curve, never from code;
  - a missing input yields missing=True and NaN, never 0.0, because "no data"
    and "conditions are dead" are different claims and spec section 8 requires
    the first to renormalise rather than score;
  - it always returns a reason, because the UI renders factor bars with text.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from tidescout.engine.curves import evaluate
from tidescout.models import SpeciesProfile

FACTORS = (
    "flow", "stage", "light", "solunar", "pressure", "wind",
    "water_temp", "salinity", "season",
)

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


class SalinityProvenance(StrEnum):
    """WHERE a salinity number came from, which is not the same question as
    whether it is in range."""

    MEASURED = "measured"   # a sensor read it -- `day.water.salinity_ppt`
    MODELLED = "modelled"   # `engine.salinity` computed it


@dataclass(frozen=True)
class SalinityReading:
    """A salinity value WITH the provenance a scorer needs to be honest.

    A bare float cannot distinguish a sensor reading from an uncalibrated
    model estimate, and this model is uncalibrated: `SalinityConfig.fitted`
    is False for Winyah Bay. Passing a float here would reproduce, at the
    factor level, exactly the confusion `SalinityField` was built to prevent.
    """

    ppt: float
    provenance: SalinityProvenance
    # Did a calibration EVER constrain these parameters. Config-level, so it
    # is identical at every cell and hour. False for Winyah Bay today.
    fitted: bool = True
    # Was THIS evaluation's discharge outside `calibration_range_cfs`.
    # Per-evaluation; changes with the river. Independent of `fitted`.
    extrapolated: bool = False

    @property
    def constrained(self) -> bool:
        """True only when the number is worth stating without a caveat."""
        return self.provenance is SalinityProvenance.MEASURED or (
            self.fitted and not self.extrapolated
        )


@dataclass
class SubScore:
    factor: str
    value: float
    weight: float
    reason: str
    missing: bool
    # Scored, and counted at full weight, but nothing observed constrains it.
    # DISTINCT from `missing`: missing means excluded and renormalised;
    # provisional means included and disclosed. A UI that renders these the
    # same way is not implementing spec section 10.
    provisional: bool = False


def _missing(factor: str, profile: SpeciesProfile, what: str) -> SubScore:
    return SubScore(factor, float("nan"), profile.weights[factor],
                    f"{factor}: no data ({what})", True)


def _scored(factor: str, profile: SpeciesProfile, x: float, reason: str) -> SubScore:
    return SubScore(factor, evaluate(profile.curves[factor], x),
                    profile.weights[factor], reason, False)


def _hours_from_twilight(t: datetime, sun) -> float | None:
    """Hours to the nearer of sunrise and sunset. Zero at the edge of light."""
    if sun is None or sun.sunrise is None or sun.sunset is None:
        return None
    return min(
        abs((t - sun.sunrise).total_seconds()), abs((t - sun.sunset).total_seconds())
    ) / 3600.0


def _minutes_from_solunar(t: datetime, periods) -> float | None:
    """Minutes to the nearest solunar major or minor."""
    if not periods:
        return None
    return min(abs((t - p.start).total_seconds()) for p in periods) / 60.0


def score_factors(
    hour,
    day,
    profile: SpeciesProfile,
    salinity: SalinityReading | None = None,
    flow_speed: float | None = None,
) -> list[SubScore]:
    """All nine sub-scores for one hour. Always nine, some possibly missing."""
    subs: list[SubScore] = []

    # 1. Tidal flow rate. Prefers the flow library's speed; falls back to the
    # CO-OPS current station, which is a single point and cannot describe the
    # bay, but beats nothing.
    speed = flow_speed
    if speed is None and hour.current_speed_kn is not None:
        speed = hour.current_speed_kn * 0.514444
    if speed is None:
        subs.append(_missing("flow", profile, "no flow state or current station"))
    else:
        kind = "slack" if speed < 0.1 else "moving" if speed < 0.8 else "ripping"
        subs.append(_scored("flow", profile, speed, f"flow {speed:.2f} m/s — {kind}"))

    # 2. Tide stage. tide_frac runs 0 at low water to 0.5 at high, matching the
    # flow library's phase convention -- NOT 0 at high water.
    if hour.tide_frac is None:
        subs.append(_missing("stage", profile, "no tide prediction"))
    else:
        half = "flooding" if hour.tide_frac < 0.5 else "ebbing"
        subs.append(_scored("stage", profile, hour.tide_frac,
                            f"tide {hour.tide_frac:.2f} of cycle — {half}"))

    # 3. Light. Cloud cover widens the low-light window, so heavy cloud is
    # credited as bringing the hour closer to twilight.
    hours_off = _hours_from_twilight(hour.time, getattr(day, "sun", None))
    if hours_off is None:
        subs.append(_missing("light", profile, "no sun times"))
    else:
        cloud = hour.cloud_cover_pct or 0.0
        effective = hours_off * (1.0 - 0.35 * cloud / 100.0)
        note = f", {cloud:.0f}% cloud widens it" if cloud > 50 else ""
        subs.append(_scored("light", profile, effective,
                            f"{hours_off:.1f} h from twilight{note}"))

    # 4. Solunar. Smallest default weight of the nine, per spec section 8.
    mins = _minutes_from_solunar(hour.time, getattr(day, "solunar", None))
    if mins is None:
        subs.append(_missing("solunar", profile, "no solunar periods"))
    else:
        subs.append(_scored("solunar", profile, mins,
                            f"{mins:.0f} min from a solunar period"))

    # 5. Pressure trend.
    if hour.pressure_trend_mb_3h is None:
        subs.append(_missing("pressure", profile, "no pressure trend"))
    else:
        p = hour.pressure_trend_mb_3h
        note = ("falling — pre-frontal feeding window" if p < -0.5
                else "rising sharply — post-frontal shutdown" if p > 2.0
                else "steady")
        subs.append(_scored("pressure", profile, p,
                            f"pressure {p:+.1f} mb/3h — {note}"))

    # 6. Wind.
    if hour.wind_speed_kn is None:
        subs.append(_missing("wind", profile, "no wind forecast"))
    else:
        w = hour.wind_speed_kn
        note = ("calm" if w < 5 else "light" if w < 12
                else "building" if w < 18 else "hard — fishability suffers")
        subs.append(_scored("wind", profile, w, f"wind {w:.0f} kn — {note}"))

    # 7. Water temperature.
    water = getattr(day, "water", None)
    temp = getattr(water, "temp_f", None) if water else None
    if temp is None:
        subs.append(_missing("water_temp", profile, "no water sensor or climatology"))
    else:
        trend = getattr(water, "temp_trend_f_3d", None)
        note = ""
        if trend is not None and abs(trend) >= 1.0:
            note = f", {'warming' if trend > 0 else 'cooling'} {abs(trend):.1f}F/3d"
        subs.append(_scored("water_temp", profile, temp, f"water {temp:.0f}F{note}"))

    # 8. Salinity. Spatial when scoring a feature, bay-representative otherwise.
    # The value is scored the same either way; only the REASON changes, and it
    # must change, because an uncalibrated model estimate and a sensor reading
    # are different claims about the world.
    if salinity is None or not math.isfinite(salinity.ppt):
        subs.append(_missing("salinity", profile, "no salinity estimate"))
    else:
        ppt = salinity.ppt
        note = "near-fresh" if ppt < 5 else "brackish" if ppt < 18 else "salty"
        if salinity.constrained:
            reason = f"salinity {ppt:.1f} ppt — {note}"
        else:
            caveats = []
            if not salinity.fitted:
                caveats.append("UNCALIBRATED model estimate, no observation constrains it")
            if salinity.extrapolated:
                caveats.append("discharge outside the calibrated range")
            # "~" on the number as well as the caveat: the tilde survives
            # truncation in a narrow UI column, the parenthetical may not.
            reason = f"salinity ~{ppt:.1f} ppt — {note} ({'; '.join(caveats)})"
        subs.append(SubScore("salinity", evaluate(profile.salinity, ppt),
                             profile.weights["salinity"], reason, False,
                             provisional=not salinity.constrained))

    # 9. Season. A table lookup, not a curve -- months are discrete.
    month = hour.time.month
    subs.append(SubScore("season", float(profile.months[month]),
                         profile.weights["season"],
                         f"{MONTH_NAMES[month - 1]} (month {month}) seasonal modifier",
                         False))

    return subs
