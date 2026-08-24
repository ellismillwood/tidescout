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


def test_tide_events_range_chunks_by_year(monkeypatch):
    """1,260 unique dates over 27 years is 28 yearly calls, not 1,260 daily
    ones. The cache makes both one-time, but not both cheap."""
    from datetime import date

    from tidescout.sources import noaa

    calls = []

    def fake_get_or_fetch(source, key, ttl, fetch):
        calls.append(key)
        return type("C", (), {"payload": {"predictions": []}})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    noaa.tide_events_range(
        "8662549", date(1999, 1, 1), date(2001, 12, 31), "America/New_York", cache
    )

    assert len(calls) == 3, f"expected one call per year, got {calls}"


def test_tide_events_range_returns_events_in_time_order(monkeypatch):
    """Yearly chunks are concatenated; phase_at sorts defensively, but the
    seam is where unsorted input would first appear."""
    from datetime import date

    from tidescout.sources import noaa

    payloads = {
        "1999": {"predictions": [{"t": "1999-06-01 05:00", "type": "L", "v": "-0.5"}]},
        "2000": {"predictions": [{"t": "2000-06-01 05:00", "type": "H", "v": "4.0"}]},
    }

    def fake_get_or_fetch(source, key, ttl, fetch):
        year = key.split(":")[2][:4]
        return type("C", (), {"payload": payloads[year]})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    out = noaa.tide_events_range(
        "8662549", date(1999, 1, 1), date(2000, 12, 31), "America/New_York", cache
    )

    assert [e.time for e in out] == sorted(e.time for e in out)
    assert len(out) == 2


def test_tide_events_range_uses_the_permanent_prediction_cache(monkeypatch):
    """Predictions are deterministic; re-fetching 28 years on every run
    would be pure waste."""
    from datetime import date

    from tidescout.sources import noaa

    seen_ttl = []

    def fake_get_or_fetch(source, key, ttl, fetch):
        seen_ttl.append(ttl)
        return type("C", (), {"payload": {"predictions": []}})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    noaa.tide_events_range(
        "8662549", date(2020, 1, 1), date(2020, 12, 31), "America/New_York", cache
    )

    assert seen_ttl == [noaa.PREDICTION_TTL]
    assert noaa.PREDICTION_TTL is None


def test_tide_events_range_parses_a_real_coops_response():
    """A recorded real response, not a hand-built one. CO-OPS returns naive
    local-time strings with `time_zone=lst_ldt`; parsing them as UTC would
    shift every phase by 4-5 hours -- a third of a tidal cycle."""
    import json
    from datetime import date
    from pathlib import Path

    from tidescout.sources import noaa

    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "coops_hilo_1999.json").read_text()
    )
    cache = type("Cache", (), {
        "get_or_fetch": staticmethod(lambda s, k, t, f: type("C", (), {"payload": payload})())
    })()

    out = noaa.tide_events_range("8662549", date(1999, 1, 1), date(1999, 12, 31),
                                 "America/New_York", cache)

    assert out, "the real fixture must yield events"
    assert all(e.time.tzinfo is not None for e in out)
    assert all(e.kind in ("H", "L") for e in out)
    assert out == sorted(out, key=lambda e: e.time)
