from datetime import date, timedelta

import pytest
import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources import weather
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
    assert "wind_speed_unit=kn" in str(sent)
    assert "temperature_unit=fahrenheit" in str(sent)
    assert "precipitation_unit=inch" in str(sent)
    assert "timezone=America" in str(sent)


@respx.mock
def test_old_dates_route_to_archive(tmp_path):
    respx.get(url__regex=r"https://archive-api\.open-meteo\.com/v1/archive.*").mock(
        return_value=Response(200, json=_fixture(48))
    )
    f = load_fishery("winyah-bay")
    hours, label = fetch_weather(f, date(2020, 6, 1), "gfs", Cache(tmp_path / "c.sqlite"))
    assert label == "era5"
    assert len(hours) == 48


@respx.mock
def test_archive_cutoff_uses_fishery_local_today(tmp_path, monkeypatch):
    # Pin "today" instead of depending on wall-clock time, so this test is
    # deterministic regardless of what hour/timezone it happens to run in.
    fixed_today = date(2026, 8, 12)
    monkeypatch.setattr(weather, "_today", lambda tz: fixed_today)
    forecast_route = respx.get(url__regex=r"https://api\.open-meteo\.com/v1/forecast.*").mock(
        return_value=Response(200, json=_fixture(48))
    )
    archive_route = respx.get(
        url__regex=r"https://archive-api\.open-meteo\.com/v1/archive.*"
    ).mock(return_value=Response(200, json=_fixture(48)))
    f = load_fishery("winyah-bay")

    # Exactly at the cutoff (today - 7): still forecast.
    hours, label = fetch_weather(
        f, fixed_today - timedelta(days=7), "gfs", Cache(tmp_path / "c1.sqlite")
    )
    assert label == "gfs"
    assert len(hours) == 48
    assert forecast_route.called
    assert not archive_route.called

    # One day older (today - 8): archive.
    hours, label = fetch_weather(
        f, fixed_today - timedelta(days=8), "gfs", Cache(tmp_path / "c2.sqlite")
    )
    assert label == "era5"
    assert len(hours) == 48
    assert archive_route.called


def test_unknown_model_rejected(tmp_path):
    f = load_fishery("winyah-bay")
    with pytest.raises(KeyError):
        fetch_weather(f, date(2026, 8, 15), "wrf", Cache(tmp_path / "c.sqlite"))


def test_model_registry_complete():
    assert set(WEATHER_MODELS) == {"best", "gfs", "ecmwf", "icon", "hrrr", "nbm"}
