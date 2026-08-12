import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

import httpx

from tidescout.sources.cache import Cache

DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
PREDICTION_TTL = None  # tide/current predictions are deterministic
OBS_TTL = timedelta(minutes=15)


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


def _get_json(params: dict) -> dict:
    resp = httpx.get(DATAGETTER, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"].get("message", "CO-OPS error"))
    return payload


def _window(day: date) -> tuple[str, str]:
    begin = (day - timedelta(days=1)).strftime("%Y%m%d")
    end = (day + timedelta(days=1)).strftime("%Y%m%d")
    return begin, end


def _parse_t(t: str, tz: ZoneInfo) -> datetime:
    # CO-OPS returns naive "YYYY-MM-DD HH:MM" strings already expressed in
    # the station's local standard/daylight time (we always request
    # time_zone=lst_ldt), so attach that zone directly rather than parsing
    # as UTC and converting.
    return datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=tz)


def tide_hours(station: str, day: date, tz: str, cache: Cache) -> list[TideHour]:
    begin, end = _window(day)
    params = {
        "product": "predictions",
        "application": "tidescout",
        "station": station,
        "begin_date": begin,
        "end_date": end,
        "datum": "MLLW",
        "time_zone": "lst_ldt",
        "units": "english",
        "interval": "h",
        "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"pred:{station}:{begin}:{end}", PREDICTION_TTL, lambda: _get_json(params)
    )
    zone = ZoneInfo(tz)
    return [
        TideHour(_parse_t(p["t"], zone), float(p["v"]))
        for p in cached.payload.get("predictions", [])
    ]


def tide_events(station: str, day: date, tz: str, cache: Cache) -> list[TideEvent]:
    begin, end = _window(day)
    params = {
        "product": "predictions",
        "application": "tidescout",
        "station": station,
        "begin_date": begin,
        "end_date": end,
        "datum": "MLLW",
        "time_zone": "lst_ldt",
        "units": "english",
        "interval": "hilo",
        "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"hilo:{station}:{begin}:{end}", PREDICTION_TTL, lambda: _get_json(params)
    )
    zone = ZoneInfo(tz)
    return [
        TideEvent(_parse_t(p["t"], zone), p["type"], float(p["v"]))
        for p in cached.payload.get("predictions", [])
    ]


def current_hours(station: str, day: date, tz: str, cache: Cache) -> list[CurrentHour]:
    begin, end = _window(day)
    params = {
        "product": "currents_predictions",
        "application": "tidescout",
        "station": station,
        "begin_date": begin,
        "end_date": end,
        "time_zone": "lst_ldt",
        "units": "english",
        "interval": "h",
        "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"cur:{station}:{begin}:{end}", PREDICTION_TTL, lambda: _get_json(params)
    )
    zone = ZoneInfo(tz)
    out = []
    for p in cached.payload.get("current_predictions", {}).get("cp", []):
        speed = float(p["Velocity_Major"])
        dir_deg = float(p["meanFloodDir"] if speed >= 0 else p["meanEbbDir"])
        out.append(CurrentHour(_parse_t(p["Time"], zone), speed, dir_deg))
    return out


def water_temp_latest(station: str, tz: str, cache: Cache) -> tuple[float, datetime] | None:
    params = {
        "product": "water_temperature",
        "application": "tidescout",
        "station": station,
        "date": "latest",
        "time_zone": "lst_ldt",
        "units": "english",
        "format": "json",
    }
    cached = cache.get_or_fetch("coops", f"wtemp:{station}", OBS_TTL, lambda: _get_json(params))
    data = cached.payload.get("data", [])
    if not data:
        return None
    zone = ZoneInfo(tz)
    return float(data[-1]["v"]), _parse_t(data[-1]["t"], zone)


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
