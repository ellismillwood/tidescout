from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from tidescout.models import Fishery
from tidescout.sources.cache import Cache

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_CUTOFF_DAYS = 7
FORECAST_TTL = timedelta(hours=1)

WEATHER_MODELS = {
    "best": "best_match",
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
    "icon": "icon_seamless",
    "hrrr": "gfs_hrrr",
    "nbm": "ncep_nbm_conus",
}

HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]


@dataclass
class WeatherHour:
    time: datetime
    air_temp_f: float | None
    wind_speed_kn: float | None
    wind_dir_deg: float | None
    wind_gust_kn: float | None
    pressure_mb: float | None
    cloud_cover_pct: float | None
    precip_in: float | None


def _get_json(url: str, params: dict) -> dict:
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _value(hourly: dict, name: str, i: int) -> float | None:
    vals = hourly.get(name)
    return None if vals is None or vals[i] is None else float(vals[i])


def fetch_weather(
    fishery: Fishery, day: date, model_key: str, cache: Cache
) -> tuple[list[WeatherHour], str]:
    model_code = WEATHER_MODELS[model_key]  # KeyError for unknown keys is intended
    lon, lat = fishery.center
    start = day - timedelta(days=1)
    today = datetime.now(UTC).date()
    use_archive = day < today - timedelta(days=ARCHIVE_CUTOFF_DAYS)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start.isoformat(),
        "end_date": day.isoformat(),
        "timezone": fishery.timezone,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "kn",
        "precipitation_unit": "inch",
    }
    if use_archive:
        url, label = ARCHIVE_URL, "era5"
        ttl: timedelta | None = None  # reanalysis of the past never changes
    else:
        url, label = FORECAST_URL, model_key
        params["models"] = model_code
        ttl = FORECAST_TTL
    key = f"{fishery.slug}:{day.isoformat()}:{label}"
    cached = cache.get_or_fetch("open-meteo", key, ttl, lambda: _get_json(url, params))
    hourly = cached.payload["hourly"]
    tz = ZoneInfo(fishery.timezone)
    hours = []
    for i, t in enumerate(hourly["time"]):
        hours.append(
            WeatherHour(
                time=datetime.fromisoformat(t).replace(tzinfo=tz),
                air_temp_f=_value(hourly, "temperature_2m", i),
                wind_speed_kn=_value(hourly, "wind_speed_10m", i),
                wind_dir_deg=_value(hourly, "wind_direction_10m", i),
                wind_gust_kn=_value(hourly, "wind_gusts_10m", i),
                pressure_mb=_value(hourly, "pressure_msl", i),
                cloud_cover_pct=_value(hourly, "cloud_cover", i),
                precip_in=_value(hourly, "precipitation", i),
            )
        )
    return hours, label
