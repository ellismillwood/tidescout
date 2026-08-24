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
# be treated as half a tidal cycle. Measured across 5,643 consecutive hi/lo
# intervals at station 8662549 (Springmaid Pier / Winyah Bay area), sampled
# across 1999/2010/2020/2026: min 4.60 h, median 6.28 h, mean 6.21 h,
# max 7.82 h, p99 7.18 h, p99.9 7.36 h, zero intervals exceeding 8 h -- this
# station's tide is semidiurnal with no observed interval near this
# threshold. 9.0 sits between that observed maximum (7.82 h) and the ~12.4 h
# gap a single missing event would produce, so it clears real half-cycles
# with room to spare while still catching a dropped prediction as a GAP, not
# a long half-cycle. Interpolating across a real gap would place phase 0.25
# in the middle of a cycle that was never predicted.
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

    Invariant when `t` lands exactly on an interior event's own timestamp:
    bracketing intervals are inclusive on both ends (`before.time <= t <=
    after.time`), so such a `t` matches two adjacent pairs -- the one ending
    there and the one starting there. This prefers the first VALID pair over
    the first MATCHING one, so an invalid neighbour (a prediction gap or a
    same-kind pair) on one side can't shadow a valid determination from the
    other side. `None` is returned only when every matching pair is
    unusable.
    """
    if t.tzinfo is None:
        raise ValueError("phase_at needs a tz-aware timestamp; a naive one silently shifts phase")
    ordered = sorted(events, key=lambda e: e.time)
    for before, after in zip(ordered, ordered[1:], strict=False):
        if not (before.time <= t <= after.time):
            continue
        if before.kind == after.kind:
            continue  # the event between them was not predicted -- try the other matching pair
        span = (after.time - before.time).total_seconds()
        if span <= 0 or span > MAX_HALF_CYCLE_H * 3600.0:
            continue  # a gap, not a half-cycle -- try the other matching pair
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
