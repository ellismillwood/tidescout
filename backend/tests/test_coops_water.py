"""Recorded-fixture tests. Never hits the live CO-OPS API.

URL matching follows test_noaa.py's established regex pattern rather than a
literal URL: the brief's own literal URL
(https://api.tidesandcurrents.noaa.gov/api/datagetter) omits the "/api/prod/"
segment that noaa.DATAGETTER (and the real, verified-live CO-OPS endpoint)
actually use -- confirmed live 2026-08-22 (that path 403s; .../api/prod/
succeeds). See task-4-report.md.
"""

from datetime import date

import pytest
import respx
from httpx import Response

from tidescout.errors import SourceUnavailable
from tidescout.sources.cache import Cache
from tidescout.sources.coops_water import fetch_ocean_salinity

PAYLOAD = {
    "data": [
        {"t": "2026-08-16 00:00", "s": "33.9"},
        {"t": "2026-08-16 01:00", "s": "34.2"},
        {"t": "2026-08-16 02:00", "s": ""},
    ]
}


@respx.mock
def test_returns_the_mean_of_valid_readings(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(200, json=PAYLOAD)
    )
    got = fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db"))
    assert got == pytest.approx(34.05, abs=0.01)


@respx.mock
def test_blank_readings_are_skipped_not_read_as_zero(tmp_path):
    """CO-OPS returns an empty string for a dark sensor. Parsed as 0.0 it would
    drag the ocean end-member toward fresh -- the model's most sensitive input."""
    respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(200, json={"data": [{"t": "x", "s": ""}]})
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None


@respx.mock
def test_api_error_payload_raises_source_unavailable(tmp_path):
    """A CO-OPS {"error": ...} payload (this is how "No data was found"
    arrives -- the shape Springmaid Pier's salinity product actually returns,
    live-verified) is a real failure, not "the station responded with
    nothing usable" -- it must propagate as SourceUnavailable so it reaches
    the app's one failure-handling path (dayloader.load_day's attempt(),
    which records it by name) instead of vanishing into an indistinguishable
    None alongside a blank reading or an out-of-range value."""
    respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(200, json={"error": {"message": "No data was found"}})
    )
    with pytest.raises(SourceUnavailable):
        fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db"))


@respx.mock
def test_implausible_values_are_rejected(tmp_path):
    """A stuck sensor reading 0 or 300 ppt must not become S_ocean."""
    respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(
            200, json={"data": [{"t": "x", "s": "300.0"}, {"t": "y", "s": "0.0"}]}
        )
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None


@respx.mock
def test_sends_station_and_date_in_request_params(tmp_path):
    """Guards the actual wiring, not just the URL shape: a bug that swapped
    `station` and `begin_date`/`end_date` would pass every other test here
    (they only match on `product=salinity` via regex) but silently query the
    wrong station or the wrong day."""
    route = respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(200, json=PAYLOAD)
    )
    fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db"))
    sent = route.calls.last.request.url.params
    assert sent["station"] == "8661070"
    assert sent["begin_date"] == "20260816"
    assert sent["end_date"] == "20260816"
    assert sent["product"] == "salinity"
    assert sent["datum"] == "MLLW"
    assert sent["units"] == "metric"
    assert sent["time_zone"] == "gmt"
