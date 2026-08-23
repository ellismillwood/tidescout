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
