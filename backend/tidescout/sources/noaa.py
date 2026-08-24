from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from tidescout.engine.tides import CurrentHour, TideEvent, TideHour, TideStage
from tidescout.errors import SourceUnavailable
from tidescout.sources.cache import Cache

__all__ = ["CurrentHour", "TideEvent", "TideHour", "TideStage"]

DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
PREDICTION_TTL = None  # tide/current predictions are deterministic
OBS_TTL = timedelta(minutes=15)


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


def _fetch_year_predictions(params: dict, station: str, year: int) -> dict:
    """Fetch one year of hi/lo predictions, rejecting an empty result before
    it can be cached.

    CO-OPS can answer with HTTP 200 and a body like
    {"message": "Network error communicating with endpoint"} when its own
    backend is unhappy -- observed live while capturing this module's test
    fixture. `_get_json` only inspects `payload["error"]`, so that shape
    passes straight through with no "predictions" key. Because
    `PREDICTION_TTL` is `None`, letting it through here would cache zero
    events for that year forever, silently making every grab sample in it
    unphaseable. Every year actually sampled at this station carries
    roughly 1,410 hi/lo events, so an empty result is treated as a fetch
    failure, not a legitimate outcome -- a real empty year is not expected
    here, and caching emptiness forever is strictly worse than a loud
    failure.
    """
    payload = _get_json(params)
    if not payload.get("predictions"):
        raise SourceUnavailable(
            "coops", f"no hi/lo predictions returned for station {station}, {year}"
        )
    return payload


def tide_events_range(
    station: str, start: date, end: date, tz: str, cache: Cache
) -> list[TideEvent]:
    """Hi/lo predictions across a multi-year span, fetched a year at a time.

    The salinity calibration needs a tidal phase for every grab sample it
    holds -- measured 2026-08-24: 1,260 unique dates spanning 1999-2026.
    Looping `tide_events` would be 1,260 requests; yearly chunks are 28.
    Both are one-time (PREDICTION_TTL is None because predictions are
    deterministic), but only one is neighbourly to a federal service.

    Chunks are keyed per year, so extending the range later re-fetches only
    the years actually added.

    Returns every event in each *calendar year* touched by [start, end],
    not just events falling inside that window -- deliberately a superset,
    not a bug. `phase_at` needs the pair of events bracketing each
    observation, and the first and last observations in a caller's window
    are typically bracketed by events that fall just outside it; filtering
    to the window would strip exactly those and leave the edge observations
    unphaseable. A caller that wants a strict window should filter the
    result itself.
    """
    zone = ZoneInfo(tz)
    out: list[TideEvent] = []
    for year in range(start.year, end.year + 1):
        begin = f"{year}0101"
        finish = f"{year}1231"
        params = {
            "product": "predictions",
            "application": "tidescout",
            "station": station,
            "begin_date": begin,
            "end_date": finish,
            "datum": "MLLW",
            "time_zone": "lst_ldt",
            "units": "english",
            "interval": "hilo",
            "format": "json",
        }
        cached = cache.get_or_fetch(
            "coops",
            f"hilo:{station}:{begin}:{finish}",
            PREDICTION_TTL,
            lambda p=params, y=year: _fetch_year_predictions(p, station, y),
        )
        out.extend(
            TideEvent(_parse_t(p["t"], zone), p["type"], float(p["v"]))
            for p in cached.payload.get("predictions", [])
        )
    out.sort(key=lambda e: e.time)
    return out


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
