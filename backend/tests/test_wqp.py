"""WQP salinity parsing, against a verbatim excerpt of a real response.

Task 9's CDMO parser was built from documentation and four of its inferences
were wrong, one catastrophically (whole-file header routing would have sent
every station to the MET parser and destroyed every salinity reading). No
parser in this repo is trusted until it has met a real file.
"""

from datetime import UTC, datetime
from pathlib import Path

from tidescout.sources import wqp

FIXTURE = Path(__file__).parent / "fixtures" / "wqp_results_excerpt.csv"


def _report():
    return wqp.parse_results(FIXTURE.read_text().splitlines())


def test_parses_real_samples():
    r = _report()
    assert r.n_admitted > 50, "the real excerpt holds ~90 admitted rows"
    assert all(s.salinity_psu >= 0 for s in r.samples)
    assert {s.station for s in r.samples} <= {
        "21SC60WQ_WQX-WB-06", "21SC60WQ_WQX-WB-05", "21SCSHL_WQX-05-24",
    }


def test_timestamps_are_utc_aware():
    r = _report()
    assert r.samples, "need at least one sample"
    for s in r.samples:
        assert s.ts.tzinfo is not None, "naive timestamps silently shift the tidal phase"
        assert s.ts.utcoffset() == datetime.now(UTC).utcoffset()


def test_edt_is_converted_not_assumed_utc():
    """11:35 EDT is 15:35 UTC. Treating local time as UTC would put the
    sample four hours off, which at a 12.42 h tidal period is most of a
    quarter cycle -- the difference between flood and ebb."""
    rows = [
        "MonitoringLocationIdentifier,CharacteristicName,ResultMeasureValue,"
        "ResultMeasure/MeasureUnitCode,ActivityStartDate,ActivityStartTime/Time,"
        "ActivityStartTime/TimeZoneCode,ActivityTypeCode,ResultStatusIdentifier,"
        "ResultDetectionConditionText,ActivityDepthHeightMeasure/MeasureValue,"
        "ActivityDepthHeightMeasure/MeasureUnitCode",
        "S1,Salinity,20.0,ppt,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,",
    ]
    r = wqp.parse_results(rows)
    assert r.samples[0].ts == datetime(2014, 6, 12, 15, 35, tzinfo=UTC)


def test_per_mille_and_ppt_are_both_admitted_unchanged():
    """`0/00` is per-mille, numerically identical to ppt. Both are real in
    the live data (81 ppt rows, 42 `0/00` rows in one response)."""
    head = (
        "MonitoringLocationIdentifier,CharacteristicName,ResultMeasureValue,"
        "ResultMeasure/MeasureUnitCode,ActivityStartDate,ActivityStartTime/Time,"
        "ActivityStartTime/TimeZoneCode,ActivityTypeCode,ResultStatusIdentifier,"
        "ResultDetectionConditionText,ActivityDepthHeightMeasure/MeasureValue,"
        "ActivityDepthHeightMeasure/MeasureUnitCode"
    )
    a = wqp.parse_results(
        [head, "S1,Salinity,20.0,ppt,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,"]
    )
    b = wqp.parse_results(
        [head, "S1,Salinity,20.0,0/00,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,"]
    )
    assert a.samples[0].salinity_psu == b.samples[0].salinity_psu == 20.0


def test_an_unknown_unit_is_rejected_and_counted_never_coerced():
    r = _report()
    assert r.n_bad_unit >= 1
    assert "uS/cm" in r.unknown_units
    assert all(s.salinity_psu <= 45.0 for s in r.samples), "a uS/cm value would be ~4 digits"


def test_quality_control_activities_are_excluded():
    """Field blanks and lab replicates pass every other filter and would
    enter the fit as real estuary measurements."""
    r = _report()
    assert r.n_qc_activity >= 1


def test_a_row_with_no_time_is_rejected_not_defaulted():
    """Every sample's worth comes from resolving its timestamp to a tidal
    phase. A fabricated noon is a fabricated phase."""
    r = _report()
    assert r.n_no_time >= 1


def test_missing_depth_is_none_not_zero():
    """0.0 m means SURFACE, which is a claim. Depth is missing in 120 of 123
    real rows, and stratification is a measured +3.30 ppt at one distance."""
    r = _report()
    assert any(s.depth_m is None for s in r.samples)
    assert all(s.depth_m is None or s.depth_m > 0 for s in r.samples)


def test_counters_account_for_every_row():
    """A rejection path with no counter is a silent drop."""
    r = _report()
    accounted = (
        r.n_admitted + r.n_no_time + r.n_bad_unit
        + r.n_bad_status + r.n_qc_activity + r.n_no_value
    )
    assert accounted == r.n_rows
