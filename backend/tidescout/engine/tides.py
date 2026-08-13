import math
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
