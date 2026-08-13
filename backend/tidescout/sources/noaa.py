from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from tidescout.engine.tides import CurrentHour, TideEvent, TideHour, TideStage
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
