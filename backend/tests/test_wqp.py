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
STATIONS_FIXTURE = Path(__file__).parent / "fixtures" / "wqp_stations_excerpt.csv"


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


def test_ppth_is_admitted_as_the_same_unit_as_ppt():
    """`ppth` (parts per thousand) is `ppt` under a third spelling, not a
    different unit -- see `ACCEPTED_UNITS`'s comment for the measured
    distribution evidence (ppth: 0.00-66.00 median 32.00, n=3,785; ppt:
    -0.01-107.00 median 8.61, n=3,644 -- same physical scale). Admitting it
    recovers 40.7% of real salinity rows that were previously rejected."""
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
        [head, "S1,Salinity,20.0,ppth,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,"]
    )
    assert a.samples[0].salinity_psu == b.samples[0].salinity_psu == 20.0
    assert b.n_bad_unit == 0


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


def test_an_implausible_value_is_rejected_and_counted_not_a_unit_problem():
    """A valid, accepted unit but a value no real estuary water can have
    (e.g. 107 ppt) is a DIFFERENT rejection reason than an unrecognized
    unit, and needs its own counter -- a single 100+ psu point would
    otherwise dominate a least-squares fit (see `_MAX_PLAUSIBLE_PSU`'s
    comment for the measured evidence: 8 rows over 40 psu, 2 negative,
    among 9,306 real rows). The boundary itself (45.0, inclusive) must
    still admit, not reject -- see `_MAX_PLAUSIBLE_PSU`'s comment on why
    the four 40.77-43.50 readings are kept as credible."""
    head = (
        "MonitoringLocationIdentifier,CharacteristicName,ResultMeasureValue,"
        "ResultMeasure/MeasureUnitCode,ActivityStartDate,ActivityStartTime/Time,"
        "ActivityStartTime/TimeZoneCode,ActivityTypeCode,ResultStatusIdentifier,"
        "ResultDetectionConditionText,ActivityDepthHeightMeasure/MeasureValue,"
        "ActivityDepthHeightMeasure/MeasureUnitCode"
    )
    rows = [
        head,
        "S1,Salinity,107.0,ppt,2015-05-12,11:35:00,EDT,Sample-Routine,Final,,,",
        "S1,Salinity,-0.01,ppt,2018-10-02,11:35:00,EDT,Sample-Routine,Final,,,",
        "S1,Salinity,45.01,ppt,2018-10-02,11:35:00,EDT,Sample-Routine,Final,,,",
        "S1,Salinity,45.0,ppt,2018-10-02,11:35:00,EDT,Sample-Routine,Final,,,",
    ]
    r = wqp.parse_results(rows)
    assert r.n_implausible == 3
    assert r.n_admitted == 1
    assert r.samples[0].salinity_psu == 45.0
    assert "107" in r.implausible_values


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
        + r.n_other_characteristic + r.n_implausible
    )
    assert accounted == r.n_rows


def test_store_is_a_separate_file_from_the_nerrs_one(tmp_path, monkeypatch):
    """ndbc.py states as an invariant that its store holds only NIW NERR
    data, and its citation() builds a NERRS citation for everything in it.
    WQP rows in that file would attribute SCDHEC and EPA data to NERRS."""
    from tidescout.sources import ndbc, wqp

    monkeypatch.setattr("tidescout.paths.fishery_data_dir", lambda slug: tmp_path)
    monkeypatch.setattr("tidescout.sources.ndbc.fishery_data_dir", lambda slug: tmp_path)
    monkeypatch.setattr("tidescout.sources.wqp.fishery_data_dir", lambda slug: tmp_path)

    assert wqp.default_store("winyah-bay")._db_path != ndbc.default_store("winyah-bay")._db_path
    assert wqp.default_store("winyah-bay")._db_path.name == "wqp.sqlite"


def test_station_coordinates_are_stored_with_their_readings(tmp_path):
    """WQP stations are discovered from a bbox query, not authored, so their
    positions must travel with their readings rather than living in a second
    artefact that can drift out of step."""
    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    stations_csv = (
        "MonitoringLocationIdentifier,LatitudeMeasure,LongitudeMeasure\n"
        "21SC60WQ_WQX-WB-06,33.2795,-79.2210\n"
    )
    wqp.import_results(FIXTURE.read_text(), store, stations_csv=stations_csv)
    coords = wqp.station_coords_from_store(store)
    assert coords["21SC60WQ_WQX-WB-06"] == pytest.approx((-79.2210, 33.2795))


def test_import_is_idempotent(tmp_path):
    """Re-running an import must add nothing -- the append-and-dedupe
    contract cdmo.py proved at 2.5 M rows, reused unchanged."""
    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    text = FIXTURE.read_text()
    first = wqp.import_results(text, store)
    second = wqp.import_results(text, store)
    assert first.n_new > 0
    assert second.n_new == 0


def test_import_records_provenance_under_its_own_source(tmp_path):
    """Attribution depends on distinguishing WQP rows from NERRS rows, and
    provenance.source is the only field that can carry it."""
    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    wqp.import_results(FIXTURE.read_text(), store)
    sources = {p.source for p in store.provenance()}
    assert sources == {wqp.SOURCE_WQP}


def test_fetch_builds_a_bbox_query_without_a_key(monkeypatch):
    """The endpoint is public and unauthenticated -- unlike CDMO, which
    needs a static IP registered through a human-submitted form."""
    from tidescout.sources import wqp

    seen = {}

    class _Resp:
        text = "CharacteristicName\n"

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, timeout=None):
        seen["url"] = url
        seen["params"] = params
        return _Resp()

    monkeypatch.setattr(wqp.httpx, "get", fake_get)
    wqp.fetch_results((-79.45, 33.15, -79.05, 33.60))

    assert seen["params"]["bBox"] == "-79.45,33.15,-79.05,33.60"
    assert seen["params"]["characteristicName"] == "Salinity"
    assert "key" not in seen["params"] and "apiKey" not in seen["params"]


def test_fetch_stations_builds_a_bbox_query_without_a_key(monkeypatch):
    """Mirrors `test_fetch_builds_a_bbox_query_without_a_key` for the
    sibling Station endpoint -- a wrong URL constant or a dropped
    `raise_for_status()` here would silently return zero station
    coordinates rather than failing loudly."""
    from tidescout.sources import wqp

    seen = {}

    class _Resp:
        text = "MonitoringLocationIdentifier,LatitudeMeasure,LongitudeMeasure\n"

        def raise_for_status(self):
            seen["raise_for_status_called"] = True

    def fake_get(url, params=None, timeout=None):
        seen["url"] = url
        seen["params"] = params
        return _Resp()

    monkeypatch.setattr(wqp.httpx, "get", fake_get)
    text = wqp.fetch_stations((-79.45, 33.15, -79.05, 33.60))

    assert seen["url"] == wqp.WQP_STATION_URL
    assert seen["params"]["bBox"] == "-79.45,33.15,-79.05,33.60"
    assert seen["params"]["characteristicName"] == "Salinity"
    assert "key" not in seen["params"] and "apiKey" not in seen["params"]
    assert seen.get("raise_for_status_called") is True
    assert text == _Resp.text


def test_parse_stations_parses_a_real_station_export():
    """`parse_stations`' column names, proven against a verbatim excerpt of
    a real `Station/search` response -- the same standard Task 1 set for
    `parse_results` (`cdmo.py` had four documentation-derived inferences
    turn out wrong; a parser here isn't trusted on a live-run observation
    alone)."""
    coords = wqp.parse_stations(STATIONS_FIXTURE.read_text().splitlines())
    assert coords["21SC60WQ_WQX-WB-06"] == pytest.approx((-79.18695, 33.2225))
    assert len(coords) == 6
    for lon, lat in coords.values():
        assert -79.45 <= lon <= -79.05, "lon out of the winyah-bay bbox this fixture came from"
        assert 33.15 <= lat <= 33.60, "lat out of the winyah-bay bbox this fixture came from"


def test_station_coords_returns_empty_when_the_store_does_not_exist_yet(tmp_path, monkeypatch):
    """`station_coords`'s explicit contract: no `wqp.sqlite` on disk -> `{}`,
    without creating one as a side effect of merely reading."""
    from tidescout.sources import wqp

    monkeypatch.setattr("tidescout.sources.wqp.fishery_data_dir", lambda slug: tmp_path)
    assert wqp.station_coords("winyah-bay") == {}
    assert not (tmp_path / "wqp.sqlite").exists()


def test_station_coords_reads_back_a_populated_store(tmp_path, monkeypatch):
    """The other half of `station_coords`'s contract: once a store exists
    and has been imported into, it reads the coordinates back."""
    from tidescout.sources import wqp

    monkeypatch.setattr("tidescout.sources.wqp.fishery_data_dir", lambda slug: tmp_path)
    store = wqp.default_store("winyah-bay")
    stations_csv = (
        "MonitoringLocationIdentifier,LatitudeMeasure,LongitudeMeasure\n"
        "21SC60WQ_WQX-WB-06,33.2795,-79.2210\n"
    )
    wqp.import_results(FIXTURE.read_text(), store, stations_csv=stations_csv)

    coords = wqp.station_coords("winyah-bay")
    assert coords["21SC60WQ_WQX-WB-06"] == pytest.approx((-79.2210, 33.2795))
