from datetime import date

import pytest
import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources.weather import WEATHER_MODELS, fetch_weather

HOURLY_KEYS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]


def _fixture(n_hours: int) -> dict:
    times = [f"2026-08-{14 + h // 24:02d}T{h % 24:02d}:00" for h in range(n_hours)]
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [80.0] * n_hours,
            "wind_speed_10m": [9.0] * n_hours,
            "wind_direction_10m": [220.0] * n_hours,
            "wind_gusts_10m": [14.0] * n_hours,
            "pressure_msl": [1013.2] * n_hours,
            "cloud_cover": [40] * n_hours,
            "precipitation": [0.0] * n_hours,
        }
    }


@respx.mock
def test_fetch_weather_forecast(tmp_path):
    route = respx.get(url__regex=r"https://api\.open-meteo\.com/v1/forecast.*").mock(
        return_value=Response(200, json=_fixture(48))
    )
    f = load_fishery("winyah-bay")
    hours, label = fetch_weather(f, date(2026, 8, 15), "gfs", Cache(tmp_path / "c.sqlite"))
    assert label == "gfs"
    assert len(hours) == 48
    assert hours[0].wind_speed_kn == 9.0
    assert hours[0].pressure_mb == 1013.2
    sent = route.calls[0].request.url
    assert "models=gfs_seamless" in str(sent)
    for key in HOURLY_KEYS:
        assert key in str(sent)


@respx.mock
def test_old_dates_route_to_archive(tmp_path):
    respx.get(url__regex=r"https://archive-api\.open-meteo\.com/v1/archive.*").mock(
        return_value=Response(200, json=_fixture(48))
    )
    f = load_fishery("winyah-bay")
    hours, label = fetch_weather(f, date(2020, 6, 1), "gfs", Cache(tmp_path / "c.sqlite"))
    assert label == "era5"
    assert len(hours) == 48


def test_unknown_model_rejected(tmp_path):
    f = load_fishery("winyah-bay")
    with pytest.raises(KeyError):
        fetch_weather(f, date(2026, 8, 15), "wrf", Cache(tmp_path / "c.sqlite"))


def test_model_registry_complete():
    assert set(WEATHER_MODELS) == {"best", "gfs", "ecmwf", "icon", "hrrr", "nbm"}
