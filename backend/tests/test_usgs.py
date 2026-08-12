from datetime import UTC, datetime, timedelta

import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources.usgs import discharge_summary, fetch_series, water_summary


def _ts(site: str, param: str, values: list[tuple[str, float]]) -> dict:
    return {
        "sourceInfo": {"siteCode": [{"value": site}]},
        "variable": {"variableCode": [{"value": param}]},
        "values": [{"value": [{"dateTime": t, "value": str(v)} for t, v in values]}],
    }


def _iv_payload(series: list[dict]) -> dict:
    return {"value": {"timeSeries": series}}


def _hours_ago(h: int) -> str:
    return (datetime.now(UTC) - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


@respx.mock
def test_fetch_series_parses(tmp_path):
    payload = _iv_payload([_ts("02131000", "00060", [(_hours_ago(2), 9000.0)])])
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=payload)
    )
    series = fetch_series(["02131000"], ["00060"], 7, Cache(tmp_path / "c.sqlite"))
    assert ("02131000", "00060") in series
    assert series[("02131000", "00060")][0][1] == 9000.0


@respx.mock
def test_discharge_summary_buckets(tmp_path):
    fishery = load_fishery("winyah-bay")
    sites = [r.usgs_site or f"0{i}FAKE" for i, r in enumerate(fishery.rivers)]
    series = [
        _ts(site, "00060", [(_hours_ago(36), 12000.0), (_hours_ago(1), 13000.0)])
        for site in sites
    ]
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload(series))
    )
    summary = discharge_summary(fishery, Cache(tmp_path / "c.sqlite"))
    assert summary.cfs_now == 13000.0 * len(sites)
    assert summary.cfs_lagged == 12000.0 * len(sites)
    assert summary.bucket == "high"  # 36000 > 25000 threshold


@respx.mock
def test_water_summary_falls_back_to_climatology(tmp_path):
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([]))
    )
    fishery = load_fishery("winyah-bay")
    summary = water_summary(fishery, Cache(tmp_path / "c.sqlite"), month=8)
    assert summary.source == "climatology"
    assert summary.temp_f == fishery.climatology.water_temp_f_by_month[8]
    assert summary.salinity_ppt == fishery.climatology.salinity_ppt_by_month[8]
