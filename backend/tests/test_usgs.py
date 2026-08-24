from datetime import UTC, date, datetime, timedelta

import httpx as _httpx
import pytest
import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources import usgs
from tidescout.sources.cache import Cache
from tidescout.sources.usgs import (
    DischargeSummary,
    branch_discharge_cfs,
    classify_limb,
    discharge_summary,
    fetch_series,
    water_summary,
)


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
def test_fetch_series_merges_and_dedupes_repeated_key(tmp_path):
    t_dup = _hours_ago(5)
    t_other = _hours_ago(1)
    entry_a = _ts("02131000", "00060", [(t_dup, 10.0)])
    # Same timestamp reported again in a second values[] block on the same
    # entry (e.g. provisional vs. approved); the later occurrence should win.
    entry_a["values"].append({"value": [{"dateTime": t_dup, "value": "99.0"}]})
    # A second, separate timeSeries entry for the SAME (site, param) key must
    # merge into the same series rather than overwrite entry_a's points.
    entry_b = _ts("02131000", "00060", [(t_other, 50.0)])
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([entry_a, entry_b]))
    )
    series = fetch_series(["02131000"], ["00060"], 7, Cache(tmp_path / "c.sqlite"))
    points = series[("02131000", "00060")]
    assert points == [
        (datetime.fromisoformat(t_dup).astimezone(UTC), 99.0),
        (datetime.fromisoformat(t_other).astimezone(UTC), 50.0),
    ]


@respx.mock
def test_fetch_series_skips_malformed_entries_and_points(tmp_path):
    t_valid = _hours_ago(3)
    malformed_entry = {
        "sourceInfo": {},  # missing siteCode -> extraction fails, entry skipped
        "variable": {"variableCode": [{"value": "00060"}]},
        "values": [{"value": [{"dateTime": _hours_ago(1), "value": "5.0"}]}],
    }
    good_entry = _ts("02131000", "00060", [(t_valid, 100.0)])
    # Malformed points mixed in with the one valid point: missing dateTime,
    # non-ISO dateTime, and missing value should all be skipped silently
    # rather than raising and failing the whole call.
    good_entry["values"][0]["value"].extend(
        [
            {"value": "200.0"},  # no dateTime key
            {"dateTime": "not-a-timestamp", "value": "300.0"},  # non-ISO dateTime
            {"dateTime": _hours_ago(2)},  # no value key
        ]
    )
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([malformed_entry, good_entry]))
    )
    series = fetch_series(["02131000"], ["00060"], 7, Cache(tmp_path / "c.sqlite"))
    assert list(series.keys()) == [("02131000", "00060")]
    assert series[("02131000", "00060")] == [
        (datetime.fromisoformat(t_valid).astimezone(UTC), 100.0)
    ]


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
    assert summary.bucket == "high"  # 36000 > 6292 threshold


@respx.mock
def test_discharge_summary_bucket_falls_back_to_cfs_now(tmp_path):
    fishery = load_fishery("winyah-bay")
    sites = [r.usgs_site for r in fishery.rivers]
    # Only a very-recent reading per site: nothing falls in the 24-48h lag
    # window, so cfs_lagged stays None and the bucket must fall back to cfs_now.
    series = [_ts(site, "00060", [(_hours_ago(1), 500.0)]) for site in sites]
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload(series))
    )
    summary = discharge_summary(fishery, Cache(tmp_path / "c.sqlite"))
    assert summary.cfs_lagged is None
    assert summary.cfs_now == 500.0 * len(sites)
    assert summary.bucket == "low"  # 1500 < 2774 threshold, from cfs_now


@respx.mock
def test_discharge_summary_bucket_med_when_no_data(tmp_path):
    fishery = load_fishery("winyah-bay")
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([]))
    )
    summary = discharge_summary(fishery, Cache(tmp_path / "c.sqlite"))
    assert summary.cfs_now is None
    assert summary.cfs_lagged is None
    assert summary.bucket == "med"


def test_discharge_excludes_stale_sites(monkeypatch, fishery, cache):
    now = datetime.now(UTC)
    fresh = [(now - timedelta(hours=1), 5000.0)]
    stale = [(now - timedelta(days=4), 9000.0)]

    def fake_fetch(sites, params, period_days, cache):
        return {
            ("02131000", usgs.PARAM_DISCHARGE): fresh,
            ("02110500", usgs.PARAM_DISCHARGE): stale,
        }

    monkeypatch.setattr(usgs, "fetch_series", fake_fetch)
    s = usgs.discharge_summary(fishery, cache)
    assert s.cfs_now == 5000.0            # stale site excluded, not summed
    assert s.contributing == ["02131000"]
    assert "02110500" in s.stale


@respx.mock
def test_fetch_daily_parses_and_sorts(cache):
    payload = {"value": {"timeSeries": [{
        "sourceInfo": {"siteCode": [{"value": "02131000"}]},
        "values": [{"value": [
            {"dateTime": "2026-01-02T00:00:00.000", "value": "5200"},
            {"dateTime": "2026-01-01T00:00:00.000", "value": "4800"},
            {"dateTime": "2026-01-03T00:00:00.000", "value": "-999999"},
        ]}],
    }]}}
    respx.get(usgs.DV_URL).mock(return_value=_httpx.Response(200, json=payload))
    out = usgs.fetch_daily(["02131000"], usgs.PARAM_DISCHARGE, "2026-01-01", "2026-01-03", cache)
    assert out["02131000"] == [(date(2026, 1, 1), 4800.0), (date(2026, 1, 2), 5200.0)]


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


@respx.mock
def test_water_summary_uses_live_sensor_temp_and_trend(tmp_path):
    fishery = load_fishery("winyah-bay")
    station = fishery.stations.water[0].station
    # 5 calendar days of temp, one reading per day, each exactly 24h apart so
    # every reading lands on a distinct UTC calendar date regardless of
    # wall-clock time. The oldest day (10.0) sits outside the 3-prior-days
    # trend window and must NOT affect the trend -- it only proves the slice
    # (not the whole history) is what gets used.
    temp_values = [
        (_hours_ago(96), 10.0),
        (_hours_ago(72), 20.0),
        (_hours_ago(48), 22.0),
        (_hours_ago(24), 24.0),
        (_hours_ago(0), 30.0),
    ]
    temp_series = _ts(station, "00010", temp_values)
    sal_series = _ts(station, "00480", [(_hours_ago(0), 12.3)])
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([temp_series, sal_series]))
    )
    summary = water_summary(fishery, Cache(tmp_path / "c.sqlite"), month=8)
    assert summary.source == f"usgs:{station}"
    assert summary.temp_f == 86.0  # 30.0 C -> F, latest reading
    assert summary.temp_trend_f_3d == 14.4  # (30 - mean(20, 22, 24)) * 9/5
    assert summary.salinity_ppt == 12.3


@respx.mock
def test_water_summary_mixed_live_temp_climatology_salinity(tmp_path):
    fishery = load_fishery("winyah-bay")
    station = fishery.stations.water[0].station
    temp_series = _ts(station, "00010", [(_hours_ago(2), 25.0)])
    # No salinity (00480) data anywhere in the payload for any configured site.
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([temp_series]))
    )
    summary = water_summary(fishery, Cache(tmp_path / "c.sqlite"), month=8)
    assert summary.source == f"usgs:{station}"
    assert summary.temp_f == 77.0  # 25.0 C -> F, from the live sensor
    assert summary.temp_trend_f_3d is None  # only 1 day of history
    assert summary.salinity_ppt == fishery.climatology.salinity_ppt_by_month[8]


def _summary(now, lagged):
    return DischargeSummary(now, lagged, "med", [], [], [])


def test_limb_is_rising_when_todays_flow_exceeds_the_lagged_mean():
    assert classify_limb(_summary(12000.0, 4000.0)) == "rising"


def test_limb_is_falling_during_recovery():
    assert classify_limb(_summary(4000.0, 12000.0)) == "falling"


def test_limb_is_steady_inside_the_dead_band():
    """Gauge noise and diurnal variation are not a freshet."""
    assert classify_limb(_summary(4100.0, 4000.0)) == "steady"


def test_limb_is_unknown_when_a_gauge_is_dark():
    assert classify_limb(_summary(None, 4000.0)) == "unknown"
    assert classify_limb(_summary(4000.0, None)) == "unknown"


def test_branch_discharge_splits_by_inflow_share():
    """The Pee Dee carries 78% of the freshwater, so it carries 78% of the
    freshet -- the same correction Phase 1 made to the ANUGA forcing, applied
    to the runtime path that salinity actually reads."""
    f = load_fishery("winyah-bay")
    branches = branch_discharge_cfs(f, _summary(10000.0, 9000.0))
    assert branches["Pee Dee"] == pytest.approx(9000.0 * 0.783, rel=1e-6)
    assert branches["Black"] == pytest.approx(9000.0 * 0.083, rel=1e-6)
    assert sum(branches.values()) == pytest.approx(9000.0, rel=1e-9)


def test_branch_discharge_prefers_lagged_flow_because_the_salt_front_lags():
    """The bay's salinity today reflects the last day or two of river flow,
    not this instant's gauge reading."""
    f = load_fishery("winyah-bay")
    branches = branch_discharge_cfs(f, _summary(20000.0, 5000.0))
    assert sum(branches.values()) == pytest.approx(5000.0, rel=1e-9)


def test_branch_discharge_falls_back_to_now_when_no_lagged_value_exists():
    f = load_fishery("winyah-bay")
    branches = branch_discharge_cfs(f, _summary(7000.0, None))
    assert sum(branches.values()) == pytest.approx(7000.0, rel=1e-9)


def test_branch_discharge_rejects_partial_share_authorship():
    """Half-authoring a split must fail loudly, not half-guess it -- the same
    guard `forcing.river_inflow_m3s` enforces, shared via
    `Fishery.branch_shares()` so the runtime salinity path can't silently
    diverge from the ANUGA forcing path."""
    f = load_fishery("winyah-bay")
    f.rivers[1].inflow_share = None  # Waccamaw left unauthored
    with pytest.raises(ValueError) as exc_info:
        branch_discharge_cfs(f, _summary(10000.0, 9000.0))
    assert "Waccamaw" in str(exc_info.value)


def test_branch_discharge_rejects_shares_that_do_not_sum_to_one():
    """A silent renormalisation would hide an authoring mistake -- and here,
    hide up to a tenth of the river's discharge from a model where intrusion
    length is a strong function of Q."""
    f = load_fishery("winyah-bay")
    f.rivers[0].inflow_share = 0.5  # now sums to ~0.72
    with pytest.raises(ValueError, match="inflow_share"):
        branch_discharge_cfs(f, _summary(10000.0, 9000.0))
