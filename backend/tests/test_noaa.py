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
    """Yearly chunks are concatenated in ascending-year order; phase_at
    sorts defensively, but the seam is where unsorted input would first
    appear.

    The "1999" chunk (processed first, by year) is deliberately given a
    *later* timestamp than the "2000" chunk (processed second), so naive
    concatenation would NOT already be time-ordered -- only the explicit
    `out.sort(...)` fixes it. A version of this test using naturally
    ordered dates would still pass with that sort deleted, which is why
    the dates are swapped here."""
    from datetime import date

    from tidescout.sources import noaa

    payloads = {
        "1999": {"predictions": [{"t": "2000-06-01 05:00", "type": "H", "v": "4.0"}]},
        "2000": {"predictions": [{"t": "1999-06-01 05:00", "type": "L", "v": "-0.5"}]},
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
    assert out[0].time.year == 1999  # only true because .sort() reordered them


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


def test_tide_events_range_fetch_closures_do_not_share_the_loop_variable(monkeypatch):
    """Regression for the closure trap the task brief warned about: a bare
    `lambda: _get_json(params)` inside the year loop closes over the
    *variable* `params`, not its value at that point in the loop.

    The real `Cache.get_or_fetch` happens to call `fetch()` synchronously,
    inline, before the loop advances to the next year -- so calling
    `fetch()` immediately (as the other tests' fakes would, if they called
    it at all) can't actually distinguish the fix from the bug: at that
    instant `params` still holds the right value either way. The bug only
    surfaces once a `fetch` callable outlives its loop iteration -- e.g. a
    cache that batches or retries later. So this fake defers every
    `fetch()` call until after `tide_events_range` has fully returned and
    the loop variable has settled on its *last* value. With the bare-lambda
    regression, every deferred call would then report the last year's
    `begin_date`; with the `p=params, y=year` default-argument binding,
    each keeps the year it was built for."""
    from datetime import date

    from tidescout.sources import noaa

    deferred_fetches = []

    def fake_get_or_fetch(source, key, ttl, fetch):
        deferred_fetches.append(fetch)  # NOT called yet -- see docstring
        return type("C", (), {"payload": {"predictions": []}})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    noaa.tide_events_range(
        "8662549", date(1999, 1, 1), date(2002, 12, 31), "America/New_York", cache
    )
    assert len(deferred_fetches) == 4  # the loop variable is now stuck on 2002

    seen_begin_dates = []

    def fake_get_json(params):
        seen_begin_dates.append(params["begin_date"])
        year = params["begin_date"][:4]
        return {"predictions": [{"t": f"{year}-06-01 05:00", "type": "H", "v": "1.0"}]}

    monkeypatch.setattr(noaa, "_get_json", fake_get_json)
    for fetch in deferred_fetches:
        fetch()

    assert seen_begin_dates == ["19990101", "20000101", "20010101", "20020101"]


@respx.mock
def test_tide_events_range_rejects_a_200_status_error_shaped_response(tmp_path):
    """CO-OPS can return HTTP 200 with a body like
    {"message": "Network error communicating with endpoint"} when its own
    backend is unhappy -- not hypothetical: this is exactly what this
    task's own fixture-capture curl got on its first attempt. `_get_json`
    only inspects payload["error"], so that shape passes straight through
    with no "predictions" key. Because PREDICTION_TTL is None, letting it
    through would cache zero events for that year forever, silently making
    every grab sample in it unphaseable.

    Uses a real Cache (not a fake) so this also proves the bad payload
    never reaches the cache in the first place -- a subsequent read for
    the exact key finds nothing, so CO-OPS recovering later is not
    permanently blocked by a stale empty entry."""
    from datetime import date

    import pytest

    from tidescout.errors import SourceUnavailable
    from tidescout.sources import noaa

    respx.get(url__regex=r".*datagetter.*").mock(
        return_value=Response(200, json={"message": "Network error communicating with endpoint"})
    )
    cache = Cache(tmp_path / "c.sqlite")

    with pytest.raises(SourceUnavailable) as exc_info:
        noaa.tide_events_range(
            "8662549", date(1999, 1, 1), date(1999, 12, 31), "America/New_York", cache
        )

    assert "8662549" in str(exc_info.value)
    assert "1999" in str(exc_info.value)
    assert cache._read("coops", "hilo:8662549:19990101:19991231") is None


def test_tide_events_range_parses_a_real_coops_response():
    """A recorded real response, not a hand-built one. CO-OPS returns naive
    local-time strings with `time_zone=lst_ldt`; parsing them as UTC would
    shift every phase by 4-5 hours -- a third of a tidal cycle.

    The non-empty/tz-aware/kind checks below all still pass if `_parse_t`
    used `tzinfo=UTC` instead of the station's local zone -- none of them
    look at the absolute instant a timestamp names. The last two assertions
    do: the fixture's first row is `{"t": "1999-01-01 02:00", "type": "L"}`,
    and comparing the parsed result against that SAME wall-clock reading
    attached to `America/New_York` only matches if `_parse_t` attached the
    right zone -- UTC 02:00 and EST 02:00 (UTC-5) are different instants
    five hours apart, so a UTC-tagged parse fails the equality outright.
    `phase_at` at that instant must then read exactly 0.0 (a low, at its own
    timestamp)."""
    import json
    from datetime import date, datetime
    from pathlib import Path

    from tidescout.engine.tides import phase_at
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
    assert out[0].time == datetime(1999, 1, 1, 2, 0, tzinfo=ET)
    assert phase_at(out, out[0].time) == 0.0
    assert out == sorted(out, key=lambda e: e.time)
