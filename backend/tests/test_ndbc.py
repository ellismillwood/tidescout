"""Recorded-fixture tests for the NDBC WYSS1 accumulating store.

Never hits the live NDBC endpoint. Fixtures below are REAL captured rows
(fetched live 2026-08-23 from
https://www.ndbc.noaa.gov/data/realtime2/WYSS1.ocean), not hand-written, so
the "MM" missing-value marker and real column widths are exactly what the
station actually sends -- see task-8-report.md for the full capture.

HEADER/LATER/EARLIER simulate the rolling window sliding forward: LATER is
"now", EARLIER is the same real file read ~2.25 h earlier. They overlap on
five timestamps (12:15 through 13:15) with byte-identical values (it is the
same underlying history), which is what makes the union-not-duplicate proof
meaningful -- a second fetch reports old rows again, not new ones.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import ConnectError, Response

from tidescout.errors import SourceUnavailable
from tidescout.sources.ndbc import (
    NDBC_URL,
    NERRS_ACKNOWLEDGEMENT,
    NERRS_DISCLAIMER,
    SOURCE_NDBC_REALTIME2,
    MetObservation,
    NdbcStore,
    Observation,
    fetch_and_store,
    parse_ocean,
)

# Real 4,235-row capture of https://www.ndbc.noaa.gov/data/realtime2/WYSS1.ocean,
# fetched live 2026-08-23 (2026-07-09 00:00 -- 2026-08-23 14:30 UTC). Used only
# to regression-guard parse_ocean against the real file's actual shape.
FULL_CAPTURE = Path(__file__).parent / "fixtures" / "wyss1_2026-08-23.ocean"

HEADER = (
    "#YY  MM DD hh mm   DEPTH  OTMP   COND   SAL   O2% O2PPM  CLCON  TURB    PH    EH\n"
    "#yr  mo dy hr mn       m  degC  mS/cm   psu     %   ppm   ug/l   FTU     -    mv\n"
)

# Real rows, most-recent-first exactly as NDBC serves them (2026-08-23,
# 14:30 down to 12:15 UTC).
LATER = HEADER + (
    "2026 08 23 14 30     0.5 30.40  17.83 10.50  80.5  5.70     MM    10  7.30    MM\n"
    "2026 08 23 14 15     0.5 30.50  18.80 11.10  75.7  5.30     MM    12  7.30    MM\n"
    "2026 08 23 14 00     0.5 30.50  18.32 10.80  71.9  5.10     MM    10  7.20    MM\n"
    "2026 08 23 13 45     0.6 30.20  18.50 10.90  77.6  5.50     MM    10  7.30    MM\n"
    "2026 08 23 13 30     0.6 30.10  17.86 10.50  88.1  6.30     MM     9  7.40    MM\n"
    "2026 08 23 13 15     0.7 30.50  19.92 11.80  70.0  4.90     MM    11  7.30    MM\n"
    "2026 08 23 13 00     0.7 30.40  20.35 12.10  69.9  4.90     MM    10  7.30    MM\n"
    "2026 08 23 12 45     0.8 30.30  20.75 12.30  73.8  5.20     MM     8  7.30    MM\n"
    "2026 08 23 12 30     0.8 30.10  20.85 12.40  73.2  5.20     MM     8  7.30    MM\n"
    "2026 08 23 12 15     0.8 30.20  20.97 12.50  73.2  5.20     MM     8  7.30    MM\n"
)

# Real rows captured ~2.25 h earlier (2026-08-23, 13:15 down to 11:00 UTC) --
# the window as it looked BEFORE the five newest rows in LATER existed.
EARLIER = HEADER + (
    "2026 08 23 13 15     0.7 30.50  19.92 11.80  70.0  4.90     MM    11  7.30    MM\n"
    "2026 08 23 13 00     0.7 30.40  20.35 12.10  69.9  4.90     MM    10  7.30    MM\n"
    "2026 08 23 12 45     0.8 30.30  20.75 12.30  73.8  5.20     MM     8  7.30    MM\n"
    "2026 08 23 12 30     0.8 30.10  20.85 12.40  73.2  5.20     MM     8  7.30    MM\n"
    "2026 08 23 12 15     0.8 30.20  20.97 12.50  73.2  5.20     MM     8  7.30    MM\n"
    "2026 08 23 12 00     0.9 30.00  20.44 12.10  74.5  5.30     MM     8  7.30    MM\n"
    "2026 08 23 11 45     0.9 29.80  20.48 12.20  71.7  5.10     MM     9  7.30    MM\n"
    "2026 08 23 11 30     1.0 30.60  22.50 13.50  66.8  4.60     MM    10  7.30    MM\n"
    "2026 08 23 11 15     1.0 30.80  23.13 13.90  66.4  4.60     MM    10  7.30    MM\n"
    "2026 08 23 11 00     1.0 30.90  22.94 13.70  50.6  3.50     MM    12  7.30    MM\n"
)

EARLIER_TIMES = {
    datetime(2026, 8, 23, h, m, tzinfo=UTC)
    for h, m in [
        (13, 15), (13, 0), (12, 45), (12, 30), (12, 15),
        (12, 0), (11, 45), (11, 30), (11, 15), (11, 0),
    ]
}
LATER_TIMES = {
    datetime(2026, 8, 23, h, m, tzinfo=UTC)
    for h, m in [
        (14, 30), (14, 15), (14, 0), (13, 45), (13, 30),
        (13, 15), (13, 0), (12, 45), (12, 30), (12, 15),
    ]
}
OVERLAP_TIMES = EARLIER_TIMES & LATER_TIMES  # 12:15 .. 13:15, five timestamps
UNION_TIMES = EARLIER_TIMES | LATER_TIMES  # 15 distinct timestamps

# One real row (2026-08-23 13:30 UTC), reused by the single-row parse tests.
ROW = "2026 08 23 13 30     0.6 30.10  17.86 10.50  88.1  6.30     MM     9  7.40    MM\n"


# -- parse_ocean --------------------------------------------------------


def test_parses_every_column():
    rows = parse_ocean(HEADER + ROW)
    assert len(rows) == 1
    r = rows[0]
    assert r.ts == datetime(2026, 8, 23, 13, 30, tzinfo=UTC)
    assert r.depth_m == 0.6
    assert r.water_temp_c == 30.10
    assert r.cond_ms_cm == 17.86
    assert r.salinity_psu == 10.50
    assert r.o2_pct == 88.1
    assert r.o2_ppm == 6.30
    assert r.turbidity_ftu == 9.0
    assert r.ph == 7.40


def test_mm_becomes_none_not_zero():
    """CLCON and EH are "MM" on every real row this station has ever sent
    (no chlorophyll/ORP sensor fitted) -- if MM parsed as 0.0, a chlorophyll
    or ORP zero that never happened would reach a future bite-score input,
    the same failure shape Task 4 caught for CO-OPS blank salinity."""
    rows = parse_ocean(HEADER + ROW)
    r = rows[0]
    assert r.chlorophyll_ug_l is None
    assert r.eh_mv is None
    assert r.chlorophyll_ug_l != 0.0
    assert r.eh_mv != 0.0


def test_timestamps_are_utc_and_timezone_aware():
    rows = parse_ocean(LATER)
    for r in rows:
        assert r.ts.tzinfo is not None
        assert r.ts.utcoffset().total_seconds() == 0


def test_skips_header_and_blank_lines():
    text = HEADER + "\n" + ROW + "\n"
    rows = parse_ocean(text)
    assert len(rows) == 1


def test_conductivity_and_salinity_stay_distinct():
    """COND (mS/cm) must never be read as SAL (psu) -- Task 5's constraints
    name this conflation explicitly. On the real row below they are
    genuinely different numbers (17.86 vs 10.50); a swapped-column bug would
    still "work" numerically, so check both against their real values."""
    rows = parse_ocean(HEADER + ROW)
    r = rows[0]
    assert r.cond_ms_cm == 17.86
    assert r.salinity_psu == 10.50
    assert r.cond_ms_cm != r.salinity_psu


def test_wrong_field_count_raises():
    with pytest.raises(ValueError, match="15 fields"):
        parse_ocean(HEADER + "2026 08 23 13 30     0.6 30.10\n")


def test_parses_the_full_real_capture_without_error():
    """Regression guard against the real file's actual shape, not a
    hand-simplified one -- see task-8-report.md for the live capture."""
    rows = parse_ocean(FULL_CAPTURE.read_text())
    assert len(rows) == 4235
    assert rows[0].ts == datetime(2026, 8, 23, 14, 30, tzinfo=UTC)
    assert rows[-1].ts == datetime(2026, 7, 9, 0, 0, tzinfo=UTC)


# -- NdbcStore ------------------------------------------------------------


def test_append_stores_rows_readable_back(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    rows = parse_ocean(LATER)
    n_new = store.append("WYSS1", rows)
    assert n_new == len(rows) == 10
    assert store.count("WYSS1") == 10
    got = store.read("WYSS1")
    assert {r.ts for r in got} == LATER_TIMES


def test_append_is_idempotent(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    rows = parse_ocean(LATER)
    store.append("WYSS1", rows)
    n_new_second = store.append("WYSS1", rows)  # identical re-fetch
    assert n_new_second == 0
    assert store.count("WYSS1") == 10


def test_append_and_dedupe_across_overlapping_fetches(tmp_path):
    """THE central proof: two overlapping fetches leave the union stored
    exactly once, in order, with nothing dropped and nothing duplicated."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(EARLIER))
    n_new = store.append("WYSS1", parse_ocean(LATER))

    assert n_new == len(LATER_TIMES - EARLIER_TIMES) == 5  # only the truly-new rows counted
    got = store.read("WYSS1")
    got_times = [r.ts for r in got]

    assert len(got_times) == len(UNION_TIMES) == 15
    assert len(set(got_times)) == len(got_times), "duplicate timestamps in the store"
    assert set(got_times) == UNION_TIMES, "the union is not exactly what was fetched"
    assert got_times == sorted(got_times), "store did not read back in time order"

    # The five overlap rows kept their one real value, not a corrupted merge.
    overlap_row = next(r for r in got if r.ts == datetime(2026, 8, 23, 13, 15, tzinfo=UTC))
    assert overlap_row.salinity_psu == 11.80


def test_fetching_the_same_window_a_third_time_adds_nothing(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(EARLIER))
    store.append("WYSS1", parse_ocean(LATER))
    n_new = store.append("WYSS1", parse_ocean(LATER))  # window fetched again, unchanged
    assert n_new == 0
    assert store.count("WYSS1") == 15


def test_partial_write_does_not_corrupt_existing_history(tmp_path):
    """A batch where the third row is unbindable (simulating a corrupted or
    malformed upstream row reaching the store mid-batch) must leave nothing
    behind -- not the two good rows that already ran, and nothing already
    committed by a prior call may be touched either."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(EARLIER))
    before = store.read("WYSS1")
    assert len(before) == 10

    good1 = Observation(
        ts=datetime(2026, 8, 23, 15, 0, tzinfo=UTC), depth_m=0.5, water_temp_c=30.0,
        cond_ms_cm=18.0, salinity_psu=11.0, o2_pct=80.0, o2_ppm=5.5,
        chlorophyll_ug_l=None, turbidity_ftu=10.0, ph=7.3, eh_mv=None,
    )
    good2 = Observation(
        ts=datetime(2026, 8, 23, 15, 15, tzinfo=UTC), depth_m=0.5, water_temp_c=30.1,
        cond_ms_cm=18.1, salinity_psu=11.1, o2_pct=81.0, o2_ppm=5.6,
        chlorophyll_ug_l=None, turbidity_ftu=10.0, ph=7.3, eh_mv=None,
    )
    poisoned = Observation(
        ts=datetime(2026, 8, 23, 15, 30, tzinfo=UTC), depth_m=object(), water_temp_c=30.2,
        cond_ms_cm=18.2, salinity_psu=11.2, o2_pct=82.0, o2_ppm=5.7,
        chlorophyll_ug_l=None, turbidity_ftu=10.0, ph=7.3, eh_mv=None,
    )

    with pytest.raises(sqlite3.ProgrammingError):
        store.append("WYSS1", [good1, good2, poisoned])

    after = store.read("WYSS1")
    assert after == before, "pre-existing history was touched by the failed batch"
    after_times = {r.ts for r in after}
    assert good1.ts not in after_times, "a row from the failed batch leaked in"
    assert good2.ts not in after_times, "a row from the failed batch leaked in"


def test_store_persists_across_reopen(tmp_path):
    db_path = tmp_path / "s.sqlite"
    NdbcStore(db_path).append("WYSS1", parse_ocean(LATER))
    reopened = NdbcStore(db_path)
    assert reopened.count("WYSS1") == 10
    assert {r.ts for r in reopened.read("WYSS1")} == LATER_TIMES


def test_read_windowed_by_start_and_end(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))
    windowed = store.read(
        "WYSS1",
        start=datetime(2026, 8, 23, 13, 0, tzinfo=UTC),
        end=datetime(2026, 8, 23, 14, 0, tzinfo=UTC),
    )
    assert [r.ts.hour for r in windowed] == [13, 13, 13, 13, 14]


def test_salinity_series_pairs_timestamp_with_salinity_only(tmp_path):
    """What Task 5's calibration wants: (timestamp, salinity) at this
    station's one known along-estuary distance -- no other columns."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))
    series = store.salinity_series("WYSS1")
    assert len(series) == 10
    assert series == sorted(series)
    ts, sal = series[0]
    assert ts == datetime(2026, 8, 23, 12, 15, tzinfo=UTC)
    assert sal == 12.50


def test_salinity_series_excludes_missing_readings(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    missing = Observation(
        ts=datetime(2026, 8, 23, 16, 0, tzinfo=UTC), depth_m=0.5, water_temp_c=30.0,
        cond_ms_cm=18.0, salinity_psu=None, o2_pct=80.0, o2_ppm=5.5,
        chlorophyll_ug_l=None, turbidity_ftu=10.0, ph=7.3, eh_mv=None,
    )
    store.append("WYSS1", [missing])
    assert store.salinity_series("WYSS1") == []


def test_latest_returns_the_most_recent_full_row(tmp_path):
    """What Phase 3's bite score wants: temperature, DO, turbidity and
    salinity together for the current instant."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))
    latest = store.latest("WYSS1")
    assert latest is not None
    assert latest.ts == datetime(2026, 8, 23, 14, 30, tzinfo=UTC)
    assert latest.salinity_psu == 10.50
    assert latest.water_temp_c == 30.40


def test_latest_is_none_for_an_empty_station(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    assert store.latest("WYSS1") is None


def test_time_span_reports_min_and_max(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))
    span = store.time_span("WYSS1")
    assert span == (
        datetime(2026, 8, 23, 12, 15, tzinfo=UTC),
        datetime(2026, 8, 23, 14, 30, tzinfo=UTC),
    )


def test_time_span_is_none_for_an_empty_station(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    assert store.time_span("WYSS1") is None


def test_stations_do_not_collide(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))
    store.append("OTHER1", parse_ocean(EARLIER))
    assert store.count("WYSS1") == 10
    assert store.count("OTHER1") == 10


# -- fetch_and_store --------------------------------------------------------


@respx.mock
def test_fetch_and_store_parses_and_appends(tmp_path):
    respx.get(NDBC_URL.format(station="WYSS1")).mock(return_value=Response(200, text=LATER))
    store = NdbcStore(tmp_path / "s.sqlite")
    n_new = fetch_and_store("WYSS1", store)
    assert n_new == 10
    assert store.count("WYSS1") == 10


@respx.mock
def test_fetch_and_store_dedupes_a_second_overlapping_fetch(tmp_path):
    route = respx.get(NDBC_URL.format(station="WYSS1"))
    store = NdbcStore(tmp_path / "s.sqlite")

    route.mock(return_value=Response(200, text=EARLIER))
    fetch_and_store("WYSS1", store)
    route.mock(return_value=Response(200, text=LATER))
    fetch_and_store("WYSS1", store)

    assert store.count("WYSS1") == 15
    assert {r.ts for r in store.read("WYSS1")} == UNION_TIMES


@respx.mock
def test_fetch_failure_raises_source_unavailable(tmp_path):
    respx.get(NDBC_URL.format(station="WYSS1")).mock(side_effect=ConnectError("down"))
    store = NdbcStore(tmp_path / "s.sqlite")
    with pytest.raises(SourceUnavailable):
        fetch_and_store("WYSS1", store)
    assert store.count("WYSS1") == 0


@respx.mock
def test_fetch_failure_does_not_touch_existing_history(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(EARLIER))

    respx.get(NDBC_URL.format(station="WYSS1")).mock(return_value=Response(500))
    with pytest.raises(SourceUnavailable):
        fetch_and_store("WYSS1", store)

    assert store.count("WYSS1") == 10
    assert {r.ts for r in store.read("WYSS1")} == EARLIER_TIMES


@respx.mock
def test_requests_the_right_station():
    route = respx.get(NDBC_URL.format(station="WYSS1")).mock(
        return_value=Response(200, text=LATER)
    )
    from tidescout.sources.ndbc import _fetch_text

    _fetch_text("WYSS1")
    assert route.called
    assert "WYSS1.ocean" in str(route.calls.last.request.url)


# -- met_observations: a sibling table, same contract ------------------------


def _met_row(ts: datetime, air_temp_c: float = 22.0, wind_speed_ms: float = 3.5) -> MetObservation:
    return MetObservation(
        ts=ts, air_temp_c=air_temp_c, rh_pct=65.0, bp_mb=1015.2,
        wind_speed_ms=wind_speed_ms, max_wind_speed_ms=wind_speed_ms + 1.0,
        wind_dir_deg=180.0, wind_dir_sd_deg=12.0, par_mmol_m2=450.0,
        precip_mm=0.0, solar_rad_wm2=210.0,
    )


def test_append_met_stores_rows_readable_back(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    rows = [
        _met_row(datetime(2026, 8, 23, 12, 0, tzinfo=UTC)),
        _met_row(datetime(2026, 8, 23, 12, 15, tzinfo=UTC)),
    ]
    n_new = store.append_met("NIWOLMET", rows)
    assert n_new == 2
    assert store.count_met("NIWOLMET") == 2
    got = store.read_met("NIWOLMET")
    assert [r.ts for r in got] == [rows[0].ts, rows[1].ts]
    assert got[0].air_temp_c == 22.0
    assert got[0].wind_speed_ms == 3.5


def test_append_met_dedupes_overlapping_batches(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    t1 = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 8, 23, 12, 15, tzinfo=UTC)
    t3 = datetime(2026, 8, 23, 12, 30, tzinfo=UTC)
    store.append_met("NIWOLMET", [_met_row(t1), _met_row(t2)])
    n_new = store.append_met("NIWOLMET", [_met_row(t2), _met_row(t3)])
    assert n_new == 1  # only t3 is genuinely new
    assert store.count_met("NIWOLMET") == 3


def test_met_and_wq_tables_do_not_collide_even_under_the_same_station_code(tmp_path):
    """Different parameter families, kept in different tables -- proven by
    writing the SAME station code into both and confirming each table's
    count is independent."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("SAME", [Observation(
        ts=datetime(2026, 8, 23, 12, 0, tzinfo=UTC), depth_m=0.5, water_temp_c=30.0,
        cond_ms_cm=18.0, salinity_psu=11.0, o2_pct=80.0, o2_ppm=5.5,
        chlorophyll_ug_l=None, turbidity_ftu=10.0, ph=7.3, eh_mv=None,
    )])
    store.append_met("SAME", [_met_row(datetime(2026, 8, 23, 12, 0, tzinfo=UTC))])
    assert store.count("SAME") == 1
    assert store.count_met("SAME") == 1


def test_latest_met_returns_the_most_recent_row(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append_met(
        "NIWOLMET",
        [
            _met_row(datetime(2026, 8, 23, 12, 0, tzinfo=UTC), air_temp_c=20.0),
            _met_row(datetime(2026, 8, 23, 12, 15, tzinfo=UTC), air_temp_c=21.0),
        ],
    )
    latest = store.latest_met("NIWOLMET")
    assert latest is not None
    assert latest.ts == datetime(2026, 8, 23, 12, 15, tzinfo=UTC)
    assert latest.air_temp_c == 21.0


def test_latest_met_is_none_for_an_empty_station(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    assert store.latest_met("NIWOLMET") is None


def test_met_time_span_reports_min_and_max(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append_met(
        "NIWOLMET",
        [
            _met_row(datetime(2026, 8, 23, 12, 0, tzinfo=UTC)),
            _met_row(datetime(2026, 8, 23, 14, 0, tzinfo=UTC)),
        ],
    )
    assert store.met_time_span("NIWOLMET") == (
        datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        datetime(2026, 8, 23, 14, 0, tzinfo=UTC),
    )


def test_met_time_span_is_none_for_an_empty_station(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    assert store.met_time_span("NIWOLMET") is None


def test_met_partial_write_does_not_corrupt_existing_history(tmp_path):
    """Same standard Task 8 held `NdbcStore.append` to -- a batch where a
    later row is unbindable rolls back in full, leaving prior MET history
    untouched."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append_met("NIWOLMET", [_met_row(datetime(2026, 8, 23, 12, 0, tzinfo=UTC))])
    before = store.read_met("NIWOLMET")
    assert len(before) == 1

    good = _met_row(datetime(2026, 8, 23, 12, 15, tzinfo=UTC))
    poisoned = MetObservation(
        ts=datetime(2026, 8, 23, 12, 30, tzinfo=UTC), air_temp_c=object(), rh_pct=60.0,
        bp_mb=1013.0, wind_speed_ms=2.0, max_wind_speed_ms=3.0, wind_dir_deg=90.0,
        wind_dir_sd_deg=5.0, par_mmol_m2=300.0, precip_mm=0.0, solar_rad_wm2=150.0,
    )
    with pytest.raises(sqlite3.ProgrammingError):
        store.append_met("NIWOLMET", [good, poisoned])

    after = store.read_met("NIWOLMET")
    assert after == before
    assert good.ts not in {r.ts for r in after}


# -- provenance ---------------------------------------------------------


def test_record_provenance_and_read_back(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    span = (
        datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        datetime(2026, 8, 23, 13, 0, tzinfo=UTC),
    )
    accessed = datetime(2026, 8, 23, 15, 0, tzinfo=UTC)
    store.record_provenance("ndbc:realtime2", ["WYSS1"], span, 5, accessed_at=accessed)

    recs = store.provenance()
    assert len(recs) == 1
    r = recs[0]
    assert r.accessed_at == accessed
    assert r.source == "ndbc:realtime2"
    assert r.stations == ("WYSS1",)
    assert r.span == span
    assert r.n_new == 5


def test_record_provenance_handles_no_span(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.record_provenance("ndbc:realtime2", ["WYSS1"], None, 0)
    recs = store.provenance()
    assert recs[0].span is None


def test_fetch_and_store_records_provenance(tmp_path):
    import respx
    from httpx import Response

    with respx.mock:
        respx.get(NDBC_URL.format(station="WYSS1")).mock(return_value=Response(200, text=LATER))
        store = NdbcStore(tmp_path / "s.sqlite")
        fetch_and_store("WYSS1", store)

    recs = store.provenance()
    assert len(recs) == 1
    assert recs[0].source == SOURCE_NDBC_REALTIME2
    assert recs[0].stations == ("WYSS1",)
    assert recs[0].n_new == 10
    assert recs[0].span == (min(LATER_TIMES), max(LATER_TIMES))


def test_fetch_and_store_records_provenance_even_when_nothing_new(tmp_path):
    """The ACCESS still happened even if every row was already known --
    the citation's accessed date must reflect that."""
    import respx
    from httpx import Response

    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))

    with respx.mock:
        respx.get(NDBC_URL.format(station="WYSS1")).mock(return_value=Response(200, text=LATER))
        fetch_and_store("WYSS1", store)

    recs = store.provenance()
    assert len(recs) == 1
    assert recs[0].n_new == 0


def test_fetch_failure_records_no_provenance(tmp_path):
    import respx
    from httpx import ConnectError

    store = NdbcStore(tmp_path / "s.sqlite")
    with respx.mock:
        respx.get(NDBC_URL.format(station="WYSS1")).mock(side_effect=ConnectError("down"))
        with pytest.raises(SourceUnavailable):
            fetch_and_store("WYSS1", store)
    assert store.provenance() == []


# -- citation() -----------------------------------------------------------


def test_citation_on_an_empty_store_reports_no_recorded_access(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    c = store.citation()
    assert c.accessed_date is None
    assert "NO RECORDED ACCESS" in c.text
    assert c.subset_lines == ("no observations held",)
    assert c.sources == ()


def test_citation_text_matches_the_nerrs_template_with_date_filled_in(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.record_provenance(
        "ndbc:realtime2", ["WYSS1"], None, 0,
        accessed_at=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
    )
    c = store.citation()
    assert c.accessed_date == datetime(2026, 8, 23, 12, 0, tzinfo=UTC).date()
    assert "accessed 23 August 2026" in c.text
    assert c.text.startswith("NOAA National Estuarine Research Reserve System (NERRS).")
    assert c.text.endswith("doi:10.25921/vw8a-8031.")


def test_citation_accessed_date_is_the_most_recent_provenance_record(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.record_provenance(
        "ndbc:realtime2", ["WYSS1"], None, 0,
        accessed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    store.record_provenance(
        "cdmo:water_quality", ["NIWCBWQ"], None, 0,
        accessed_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    c = store.citation()
    assert c.accessed_date == datetime(2026, 8, 23, tzinfo=UTC).date()


def test_citation_subset_lines_describe_each_held_station(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append("WYSS1", parse_ocean(LATER))
    store.append_met(
        "NIWOLMET",
        [_met_row(datetime(2026, 8, 23, 12, 0, tzinfo=UTC))],
    )
    store.record_provenance("ndbc:realtime2", ["WYSS1"], None, 10)
    store.record_provenance("cdmo:meteorological", ["NIWOLMET"], None, 1)

    c = store.citation()
    wq_line = next(ln for ln in c.subset_lines if ln.startswith("WYSS1"))
    met_line = next(ln for ln in c.subset_lines if ln.startswith("NIWOLMET"))
    assert "water quality" in wq_line
    assert "10 observation" in wq_line
    assert "meteorological" in met_line
    assert "1 observation" in met_line


def test_citation_sources_lists_distinct_routes(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    store.record_provenance("ndbc:realtime2", ["WYSS1"], None, 1)
    store.record_provenance("cdmo:water_quality", ["NIWCBWQ"], None, 1)
    store.record_provenance("ndbc:realtime2", ["WYSS1"], None, 0)
    c = store.citation()
    assert c.sources == ("cdmo:water_quality", "ndbc:realtime2")


def test_citation_includes_the_disclaimer_and_niw_acknowledgement(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    c = store.citation()
    assert c.disclaimer == NERRS_DISCLAIMER
    assert c.acknowledgement == NERRS_ACKNOWLEDGEMENT
    assert "North Inlet-Winyah Bay NERR" in c.acknowledgement
    assert "cdmodata@baruch.sc.edu" in c.acknowledgement
    assert "Federal government does not assume liability" in c.disclaimer
    assert "reimburse or indemnify" in c.disclaimer  # the clause the dispatch's quote dropped
