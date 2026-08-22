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
def test_api_error_payload_returns_none_rather_than_raising(tmp_path):
    """Spec section 10: a dark sensor degrades to the configured default with a
    flag, it does not take down the day's forecast."""
    respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(200, json={"error": {"message": "No data was found"}})
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None


@respx.mock
def test_implausible_values_are_rejected(tmp_path):
    """A stuck sensor reading 0 or 300 ppt must not become S_ocean."""
    respx.get(url__regex=r".*datagetter.*product=salinity.*").mock(
        return_value=Response(
            200, json={"data": [{"t": "x", "s": "300.0"}, {"t": "y", "s": "0.0"}]}
        )
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None
