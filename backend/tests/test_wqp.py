"""WQP salinity parsing, against a verbatim excerpt of a real response.

Task 9's CDMO parser was built from documentation and four of its inferences
were wrong, one catastrophically (whole-file header routing would have sent
every station to the MET parser and destroyed every salinity reading). No
parser in this repo is trusted until it has met a real file.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

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


def test_a_bad_status_is_rejected_and_counted():
    """`Preliminary` and `Provisional` are not reviewed and are blocked --
    the same posture cdmo.py takes toward unvetted QAQC flags. A typo in
    ACCEPTED_STATUSES (e.g. "Historical" -> "Historic") must fail a test,
    not just silently narrow admission."""
    r = _report()
    assert r.n_bad_status >= 1
    assert "Preliminary" in r.unknown_statuses


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


def test_depth_m_passes_metres_through_unchanged():
    """The live fixture has zero rows with a recorded depth, so this branch
    is otherwise dead in coverage -- a sign error here would ship silently."""
    assert wqp._depth_m("0.3", "m") == 0.3
    assert wqp._depth_m("1.5", "meters") == 1.5


def test_depth_m_converts_feet_to_metres_numerically():
    """1 ft = 0.3048 m exactly; a wrong constant must fail this, not just
    the vacuous `> 0` check in test_missing_depth_is_none_not_zero."""
    assert wqp._depth_m("1.0", "ft") == pytest.approx(0.3048)
    assert wqp._depth_m("10.0", "feet") == pytest.approx(3.048)


def test_depth_m_rejects_blank_and_unknown_units():
    assert wqp._depth_m("", "m") is None
    assert wqp._depth_m("0.3", "") is None
    assert wqp._depth_m("0.3", "fathoms") is None


def test_a_different_characteristic_is_excluded_and_counted():
    """WQP also serves Specific conductance under a neighbouring
    characteristic; it is a different quantity and not interchangeable
    with salinity (pipeline/salinity_fit.py:832). Mixing them in would be
    silent, and dropping the row with no counter would be too."""
    r = _report()
    assert r.n_other_characteristic >= 1
    assert "SYNTH-OTHER-CHARACTERISTIC-STATION" not in {s.station for s in r.samples}


def test_counters_account_for_every_row():
    """A rejection path with no counter is a silent drop. This must account
    for every row read from the file -- including rows for a different
    CharacteristicName entirely -- not just the salinity subset, since
    Task 2's import_results hands this parser arbitrary multi-characteristic
    exports."""
    r = _report()
    accounted = (
        r.n_admitted + r.n_no_time + r.n_bad_unit
        + r.n_bad_status + r.n_qc_activity + r.n_no_value
        + r.n_other_characteristic
    )
    assert accounted == r.n_rows
