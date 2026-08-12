from datetime import date, datetime
from itertools import pairwise
from zoneinfo import ZoneInfo

import respx
from httpx import Response

from tidescout.sources.cache import Cache
from tidescout.sources.noaa import (
    TideEvent,
    _cosine_height,
    current_hours,
    interpolate_tide_hours,
    stage_at,
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
            {"Time": "2026-08-15 00:00", "Velocity_Major": -1.4, "meanFloodDir": 315.0, "meanEbbDir": 135.0},
            {"Time": "2026-08-15 01:00", "Velocity_Major": 0.8, "meanFloodDir": 315.0, "meanEbbDir": 135.0},
        ]
    }
}

TEMP_FIXTURE = {"data": [{"t": "2026-08-15 12:06", "v": "84.2", "f": "0,0,0"}]}


@respx.mock
def test_tide_hours(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=predictions.*interval=h.*").mock(
        return_value=Response(200, json=PRED_FIXTURE)
    )
    hours = tide_hours("8662245", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert hours[0].height_ft == 2.31
    assert hours[0].time == datetime(2026, 8, 15, 0, 0, tzinfo=ET)


@respx.mock
def test_tide_events(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=predictions.*interval=hilo.*").mock(
        return_value=Response(200, json=HILO_FIXTURE)
    )
    events = tide_events("8662245", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert [e.kind for e in events] == ["H", "L"]


@respx.mock
def test_current_hours_signed_direction(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=currents_predictions.*").mock(
        return_value=Response(200, json=CURRENTS_FIXTURE)
    )
    hours = current_hours("WIN1201", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite"))
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


def test_stage_at_interpolates():
    events = [
        TideEvent(datetime(2026, 8, 15, 3, 0, tzinfo=ET), "H", 5.0),
        TideEvent(datetime(2026, 8, 15, 9, 0, tzinfo=ET), "L", 0.5),
    ]
    stage = stage_at(events, datetime(2026, 8, 15, 6, 0, tzinfo=ET))
    assert stage is not None
    assert stage.phase == "falling"
    assert abs(stage.frac - 0.5) < 0.01
    assert stage.next_event.kind == "L"
    assert stage_at(events, datetime(2026, 8, 15, 1, 0, tzinfo=ET)) is None


def test_cosine_height_at_midpoint_frac():
    # Pure formula check, decoupled from wall-clock alignment: H 5.0ft ->
    # L 0.5ft, frac=0.5 is the temporal midpoint between the two events
    # regardless of what clock time that falls on.
    assert abs(_cosine_height(5.0, 0.5, 0.5) - 2.75) < 0.01
    assert _cosine_height(5.0, 0.5, 0.0) == 5.0
    assert abs(_cosine_height(5.0, 0.5, 1.0) - 0.5) < 1e-9


def test_interpolate_tide_hours_subordinate_station():
    # Winyah Bay (8662549) shape: only hi/lo events available, no harmonic
    # hourly predictions. H@03:00 5.0ft, L@09:12 0.5ft -- the 12-minute
    # offset is deliberate so no top-of-hour grid point lands exactly on
    # the true chronological midpoint (covered separately, above, via the
    # pure formula test).
    events = [
        TideEvent(datetime(2026, 8, 15, 3, 0, tzinfo=ET), "H", 5.0),
        TideEvent(datetime(2026, 8, 15, 9, 12, tzinfo=ET), "L", 0.5),
    ]
    hours = interpolate_tide_hours(events, date(2026, 8, 15), "America/New_York")

    by_time = {h.time: h.height_ft for h in hours}
    assert by_time[datetime(2026, 8, 15, 3, 0, tzinfo=ET)] == 5.0

    # Hours before the first event are absent (unbracketed).
    assert all(t >= events[0].time for t in by_time)
    assert datetime(2026, 8, 15, 2, 0, tzinfo=ET) not in by_time

    # Strictly decreasing across the bracketed H->L stretch.
    bracketed = sorted(
        (t, v) for t, v in by_time.items() if events[0].time <= t <= events[1].time
    )
    assert all(a[1] > b[1] for a, b in pairwise(bracketed))

    assert hours == sorted(hours, key=lambda h: h.time)
