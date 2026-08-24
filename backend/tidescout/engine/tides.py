import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo


@dataclass
class TideHour:
    time: datetime
    height_ft: float


@dataclass
class TideEvent:
    time: datetime
    kind: str  # "H" | "L"
    height_ft: float


@dataclass
class TideStage:
    phase: str  # "rising" | "falling"
    frac: float
    next_event: TideEvent


@dataclass
class CurrentHour:
    time: datetime
    speed_kn: float  # signed: + flood, - ebb
    dir_deg: float


def stage_at(events: list[TideEvent], t: datetime) -> TideStage | None:
    ordered = sorted(events, key=lambda e: e.time)
    for prev, nxt in pairwise(ordered):
        if prev.time <= t <= nxt.time:
            span = (nxt.time - prev.time).total_seconds()
            frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
            phase = "rising" if nxt.kind == "H" else "falling"
            return TideStage(phase, frac, nxt)
    return None


def _cosine_height(h0: float, h1: float, frac: float) -> float:
    return h0 + (h1 - h0) * (1 - math.cos(math.pi * frac)) / 2


def interpolate_tide_hours(events: list[TideEvent], day: date, tz: str) -> list[TideHour]:
    """Cosine interpolation of hourly heights between consecutive hi/lo events.

    Standard tide-clock approximation for subordinate stations, which reject
    interval=h hourly predictions.
    """
    zone = ZoneInfo(tz)
    start = datetime.combine(day - timedelta(days=1), datetime.min.time(), zone)
    end = datetime.combine(day + timedelta(days=2), datetime.min.time(), zone)
    pairs = list(pairwise(sorted(events, key=lambda e: e.time)))

    hours: list[TideHour] = []
    t = start
    while t < end:
        for prev, nxt in pairs:
            if prev.time <= t <= nxt.time:
                span = (nxt.time - prev.time).total_seconds()
                frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
                hours.append(TideHour(t, _cosine_height(prev.height_ft, nxt.height_ft, frac)))
                break
        t += timedelta(hours=1)
    return hours


# The longest interval between consecutive hi/lo predictions that can still
# be treated as half a tidal cycle. The M2 semidiurnal period is 12.42 h, so
# a real half-cycle is ~6.2 h; diurnal inequality stretches some intervals
# past 8 h. Anything beyond this is a GAP in the predictions, not a long
# half-cycle, and interpolating across it would place phase 0.25 in the
# middle of a cycle that was never predicted.
MAX_HALF_CYCLE_H = 9.0


def phase_at(events: Sequence[TideEvent], t: datetime) -> float | None:
    """The salinity model's tidal phase at `t`, or None if undeterminable.

    Phase 0 is LOW water and 0.5 is high water -- the convention
    `engine/salinity.py:salinity_at` documents and depends on. Reversing it
    inverts the tidal salinity swing across the whole bay, and no test of
    the FIT would catch that: least squares would simply choose different
    parameters to compensate.

    Interpolated LINEARLY IN TIME between the bracketing events, the same
    approximation `interpolate_tide_hours` already rests on. Returns a value
    in [0, 1): a low is always 0.0, never 1.0, so one physical state has one
    number.

    None -- never a default -- when `t` is outside the events, falls in a
    gap, or sits between two events of the same kind (which means the event
    between them was not predicted). A fabricated phase is exactly the error
    this codebase already refuses at parse time when it rejects rows with no
    usable timestamp.
    """
    if t.tzinfo is None:
        raise ValueError("phase_at needs a tz-aware timestamp; a naive one silently shifts phase")
    ordered = sorted(events, key=lambda e: e.time)
    for before, after in zip(ordered, ordered[1:], strict=False):
        if not (before.time <= t <= after.time):
            continue
        if before.kind == after.kind:
            return None  # the event between them was not predicted
        span = (after.time - before.time).total_seconds()
        if span <= 0 or span > MAX_HALF_CYCLE_H * 3600.0:
            return None
        frac = (t - before.time).total_seconds() / span
        # low -> high covers 0.0..0.5; high -> low covers 0.5..1.0
        phase = frac * 0.5 if before.kind == "L" else 0.5 + frac * 0.5
        return phase % 1.0
    return None


def interpolate_current_hours(points: list[CurrentHour], day: date, tz: str) -> list[CurrentHour]:
    """Linear interpolation of signed current speed onto the top-of-hour grid.

    Subordinate current stations predict at irregular times (slack/max ebb/max
    flood); the hourly display grid interpolates between them.
    """
    zone = ZoneInfo(tz)
    start = datetime.combine(day - timedelta(days=1), datetime.min.time(), zone)
    end = datetime.combine(day + timedelta(days=2), datetime.min.time(), zone)
    pairs = list(pairwise(sorted(points, key=lambda p: p.time)))

    hours: list[CurrentHour] = []
    t = start
    while t < end:
        for prev, nxt in pairs:
            if prev.time <= t <= nxt.time:
                span = (nxt.time - prev.time).total_seconds()
                frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
                speed = prev.speed_kn + (nxt.speed_kn - prev.speed_kn) * frac
                # Direction is a step function of flood/ebb regime, not a
                # continuous quantity: keep the leading point's heading until
                # the interpolated sign no longer matches it, then switch.
                same_regime = (speed >= 0) == (prev.speed_kn >= 0)
                dir_deg = prev.dir_deg if same_regime else nxt.dir_deg
                hours.append(CurrentHour(t, speed, dir_deg))
                break
        t += timedelta(hours=1)
    return hours
