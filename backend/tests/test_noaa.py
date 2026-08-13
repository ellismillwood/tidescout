from datetime import date, datetime
from zoneinfo import ZoneInfo

import respx
from httpx import Response

from tidescout.sources.cache import Cache
from tidescout.sources.noaa import (
    current_hours,
    tide_events,
    tide_hours,
    water_temp_latest,
)

ET = ZoneInfo("America/New_York")

PRED_FIXTURE = {
    "predictions": [
        {"t": "2026-08-15 00:00", "v": "2.31"},
        {"t": "2026-08-15 01:00", "v": "3.12"},
    ]
}

HILO_FIXTURE = {
    "predictions": [
        {"t": "2026-08-15 03:12", "v": "5.1", "type": "H"},
        {"t": "2026-08-15 09:30", "v": "0.4", "type": "L"},
    ]
}

CURRENTS_FIXTURE = {
    "current_predictions": {
        "cp": [
            {
                "Time": "2026-08-15 00:00",
                "Velocity_Major": -1.4,
                "meanFloodDir": 315.0,
                "meanEbbDir": 135.0,
            },
            {
                "Time": "2026-08-15 01:00",
                "Velocity_Major": 0.8,
                "meanFloodDir": 315.0,
                "meanEbbDir": 135.0,
            },
        ]
    }
}

TEMP_FIXTURE = {"data": [{"t": "2026-08-15 12:06", "v": "84.2", "f": "0,0,0"}]}


@respx.mock
def test_tide_hours(tmp_path):
    # interval=h&... (not just interval=h...) so this can't also match the
    # interval=hilo request that test_tide_events sends.
    route = respx.get(
        url__regex=r".*datagetter.*product=predictions.*interval=h&.*"
    ).mock(return_value=Response(200, json=PRED_FIXTURE))
    hours = tide_hours(
        "8662245", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite")
    )
    assert hours[0].height_ft == 2.31
    assert hours[0].time == datetime(2026, 8, 15, 0, 0, tzinfo=ET)
    sent_params = route.calls.last.request.url.params
    assert sent_params["units"] == "english"
    assert sent_params["datum"] == "MLLW"
    assert sent_params["time_zone"] == "lst_ldt"


@respx.mock
def test_tide_events(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=predictions.*interval=hilo.*").mock(
        return_value=Response(200, json=HILO_FIXTURE)
    )
    events = tide_events(
        "8662245", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite")
    )
    assert [e.kind for e in events] == ["H", "L"]


@respx.mock
def test_current_hours_signed_direction(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=currents_predictions.*").mock(
        return_value=Response(200, json=CURRENTS_FIXTURE)
    )
    hours = current_hours(
        "WIN1201", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite")
    )
    assert hours[0].speed_kn == -1.4
    assert hours[0].dir_deg == 135.0  # ebbing -> ebb direction
    assert hours[1].dir_deg == 315.0  # flooding -> flood direction


@respx.mock
def test_water_temp_latest(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=water_temperature.*").mock(
        return_value=Response(200, json=TEMP_FIXTURE)
    )
    result = water_temp_latest("8662245", "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert result is not None
    temp, _at = result
    assert temp == 84.2
