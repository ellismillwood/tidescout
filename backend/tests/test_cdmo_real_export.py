"""Tests against a VERBATIM excerpt of the real CDMO export (Task 11).

Unlike `test_cdmo.py` -- whose fixtures are synthetic, built from
documentation because no real file existed when Task 9 was written -- every
row in `fixtures/cdmo_niw_real_excerpt.csv` is copied byte-for-byte out of
the real 494 MB / 2,521,394-row export `390918.csv` that CDMO's query
interface produced for North Inlet-Winyah Bay, 2016-01-01 to 2026-08-23.
The excerpt keeps the file's CRLF line endings, its space-padded station
codes, its unquoted flag cells and its two genuinely 46-field rows intact,
so these tests fail if the parser stops matching what CDMO actually emits.

`fixtures/cdmo_sampling_stations_excerpt.csv` is likewise a verbatim slice
of the real `sampling_stations.csv` that accompanied the export: all twelve
NIW stations plus four out-of-region ones kept deliberately, so the
west-longitude sign fix and the per-station GMT offset are tested against
stations that would break if either were hardcoded.

Never hits a live endpoint: CDMO has no unauthenticated polling endpoint,
and both fixtures are local files.
"""

import csv
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tidescout.sources import ndbc
from tidescout.sources.cdmo import (
    CDMO_TZ,
    QAQC_CODE_MEANINGS,
    ImportReport,
    canonical_station,
    import_file,
    import_path,
    load_station_metadata,
    parse_cdmo_csv,
    parse_cdmo_met_csv,
    station_kind,
)
from tidescout.sources.ndbc import NdbcStore

FIXTURES = Path(__file__).parent / "fixtures"
REAL = FIXTURES / "cdmo_niw_real_excerpt.csv"
STATIONS = FIXTURES / "cdmo_sampling_stations_excerpt.csv"

# The seven station codes the real export actually contains. The dispatch
# for this task named exactly these seven and was right -- but note the
# station METADATA file lists twelve NIW stations (four `nut` nutrient
# stations and one long-dead `niwcawq`), which is why the importer routes
# on the code it reads rather than on a fixed list.
REAL_WQ_CODES = {"niwcbwq", "niwdcwq", "niwolwq", "niwtawq", "niwwswq", "niwwbwq"}
REAL_MET_CODES = {"niwolmet"}


# -- the format itself: one interleaved file, not two --------------------------


def test_real_header_carries_both_met_and_wq_parameters():
    """The single most consequential difference from Task 9's assumption.
    Task 9 expected SEPARATE WQ and MET exports and routed a whole file by
    its header (`_detect_format`). The real export has ONE header carrying
    both parameter families, so any whole-file routing decision is wrong for
    one of them."""
    header = REAL.read_text(encoding="utf-8-sig").splitlines()[0]
    cols = {c.strip().strip('"').lower() for c in header.split(",")}
    assert {"atemp", "wspd", "bp", "rh"} <= cols  # MET parameters
    assert {"temp", "spcond", "sal", "do_pct"} <= cols  # WQ parameters


def test_real_excerpt_contains_every_station_interleaved():
    """Rows for different stations alternate within one instant, so a parser
    that assumes one station per file (or per contiguous block) is wrong."""
    with REAL.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))[1:]
    codes = [r[0].strip().lower() for r in rows]
    assert set(codes) == REAL_WQ_CODES | REAL_MET_CODES
    first_instant = [c for r, c in zip(rows, codes, strict=True) if r[2] == "01/01/2016 0:00"]
    assert len(first_instant) == len(set(first_instant)) > 1


def test_station_codes_are_space_padded_in_the_real_file():
    raw = REAL.read_text(encoding="utf-8-sig")
    assert '"niwcbwq   "' in raw
    assert '"niwolmet  "' in raw


# -- routing: station code decides the table, not the header ------------------


def test_import_routes_met_rows_to_met_and_wq_rows_to_wq(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    report = import_file(REAL, store, stations=load_station_metadata(STATIONS))

    assert {s.raw_code for s in report.stations} == REAL_WQ_CODES
    assert {s.raw_code for s in report.met_stations} == REAL_MET_CODES
    # The MET station wrote nothing into `observations`...
    assert store.count("NIWOLMET") == 0
    # ...and no water-quality station wrote into `met_observations`.
    for code in REAL_WQ_CODES:
        assert store.count_met(canonical_station(code)) == 0
    assert store.count_met("NIWOLMET") > 0


def test_every_empty_water_quality_row_is_one_cdmo_itself_rejected(tmp_path):
    """Task 9's `_detect_format` would have read the real header as MET and
    sent EVERY row -- all seven stations -- through the MET parser, silently
    manufacturing empty water-quality rows for six stations.

    Empty rows are not banned outright: the real export genuinely contains
    instants where CDMO rejected or never collected every parameter
    (`<-3> [GQR] (CSM)` across the whole row), and storing those under
    `(station, ts)` is a real "we looked, there was nothing usable" record --
    the same convention `parse_ocean` uses for an all-MM NDBC row. What must
    never happen is an empty row MANUFACTURED by mis-routing. So: every
    empty row must be accounted for by a station that reported rejections,
    and the counts must agree exactly.
    """
    store = NdbcStore(tmp_path / "s.sqlite")
    report = import_file(REAL, store, stations=load_station_metadata(STATIONS))
    conn = sqlite3.connect(tmp_path / "s.sqlite")
    empty_by_station = dict(
        conn.execute(
            "SELECT station, COUNT(*) FROM observations WHERE salinity_psu IS NULL "
            "AND water_temp_c IS NULL AND depth_m IS NULL AND ph IS NULL "
            "AND turbidity_ftu IS NULL AND cond_ms_cm IS NULL GROUP BY station"
        )
    )
    reported = {s.canonical: s.n_empty for s in report.stations if s.n_empty}
    assert set(empty_by_station) == set(reported)
    # `n_empty` counts ROWS READ; the store counts rows KEPT. They differ by
    # exactly the export's verbatim duplicate rows -- 07/01/2026 00:00 is
    # repeated for every live station -- which `(station, ts)` absorbs.
    with REAL.open(newline="", encoding="utf-8-sig") as fh:
        raw = [r for r in list(csv.reader(fh))[1:]]
    dupes: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for r in raw:
        key = (canonical_station(r[0]), r[2])
        if key in seen:
            dupes[key[0]] = dupes.get(key[0], 0) + 1
        seen.add(key)
    for canonical, n_reported in reported.items():
        assert empty_by_station[canonical] == n_reported - dupes.get(canonical, 0)
    for canonical in empty_by_station:
        station = next(s for s in report.stations if s.canonical == canonical)
        assert station.n_rejected_by_flag, f"{canonical} has empty rows but no rejections"
    # No station that reports data was left wholly empty, and the MET
    # station contributed nothing here at all.
    assert "NIWOLMET" not in empty_by_station
    assert sum(empty_by_station.values()) < sum(s.n_rows for s in report.stations)


def test_station_kind_is_read_from_the_code_suffix():
    """CDMO's station-code convention is <reserve><site><type>, and across
    all 367 stations in the real metadata file the suffix agrees with the
    file's own `Station Type` column with zero exceptions (wq->1, met->0,
    nut->2)."""
    assert station_kind("niwolwq") == "wq"
    assert station_kind("niwolmet") == "met"
    assert station_kind("niwolnut") == "nut"
    assert station_kind("niwcbwq   ") == "wq"  # padding stripped
    assert station_kind("something-else") == "unknown"


def test_nutrient_station_rows_are_skipped_and_counted_not_stored(tmp_path):
    """No `nut` station appears in this export, but the metadata file lists
    four, so a future export could carry them -- and their parameters
    (PO4F/NH4F/NO23F) have no column in either table here. Skipped and
    REPORTED, never written as empty rows."""
    lines = REAL.read_text(encoding="utf-8-sig").splitlines()
    poisoned = lines[1].replace("niwcbwq   ", "niwcbnut  ", 1)
    path = tmp_path / "with_nut.csv"
    path.write_text("\n".join([lines[0], poisoned, *lines[2:]]) + "\n")

    store = NdbcStore(tmp_path / "s.sqlite")
    report = import_file(path, store, stations=load_station_metadata(STATIONS))

    assert report.skipped_stations == {"niwcbnut": 1}
    assert store.count("NIWCBNUT") == 0
    assert store.count_met("NIWCBNUT") == 0


# -- QA/QC flags and the bracketed qualifier codes -----------------------------


def test_flag_is_parsed_when_qualifier_codes_follow_the_bracket():
    """The real syntax puts codes AFTER the closing angle bracket --
    `<-3> [SSM] (CSM)` -- not inside it. The numeric flag must still gate."""
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    ta = result.stations["niwtawq"]
    # 07/01/2026 00:00 is `<-3> [GQR] (CSM)` on every TA parameter.
    rejected = [
        o for o in ta.observations if o.ts == datetime(2026, 7, 1, 5, 0, tzinfo=UTC)
    ]
    assert rejected, "the 07/01/2026 Thousand Acre rows should have parsed"
    assert all(o.salinity_psu is None for o in rejected)
    assert ta.n_rejected_by_flag["sal"] >= 1


def test_qualifier_codes_are_captured_not_silently_ignored():
    """Task 9's regex matched `<0>` and threw away everything after it, so a
    real file's codes were invisible. They are informational -- they do not
    override the flag -- but they must be visible in the report."""
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    seen: set[str] = set()
    for sp in result.stations.values():
        seen |= set(sp.qualifier_codes)
    assert {"GQR", "CSM", "SDG"} <= seen


def test_qualifier_code_on_an_accepted_flag_does_not_reject_the_value():
    """CDMO's own documentation: the numeric flag determines acceptance;
    letter codes explain WHY it got that flag and do not override it. SWMPr,
    the standard R reader, filters on the flag alone. `<0> (CRE)` -- passed
    QAQC, during a significant rain event -- is real data."""
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    admitted_with_codes = [
        (code, sp.qualifier_codes)
        for code, sp in result.stations.items()
        if sp.qualifier_codes and sp.n_admitted.get("sal", 0)
    ]
    assert admitted_with_codes, "some admitted salinity should carry a qualifier code"


def test_qaqc_code_meanings_cover_every_code_the_real_export_uses():
    """The full vocabulary, from CDMO's own QAQC page. An unknown code in a
    future export is reported, not silently dropped -- but nothing in THIS
    export should be unknown."""
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    seen: set[str] = set()
    for sp in result.stations.values():
        seen |= set(sp.qualifier_codes)
    assert seen
    assert seen <= set(QAQC_CODE_MEANINGS), sorted(seen - set(QAQC_CODE_MEANINGS))


def test_f_record_curly_brace_codes_are_not_read_as_a_parameter_flag():
    """`F_Record` carries record-level comments in CURLY braces (`{CSM}`),
    a third syntax again. It is a metadata column, never a parameter's flag,
    so it must not appear as an unparseable flag."""
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    for sp in result.stations.values():
        assert sp.n_flag_unparseable == {}
    assert "F_Record" not in result.unknown_columns


# -- real-file shape quirks ---------------------------------------------------


def test_rows_with_an_extra_trailing_field_still_parse():
    """63 rows in the real 2.5 M-row export carry 46 fields instead of 45 --
    an extra trailing comma at what look like export-chunk boundaries. Two
    of them are in this excerpt. They are ordinary rows and must not be
    dropped."""
    with REAL.open(newline="", encoding="utf-8-sig") as fh:
        widths = {len(r) for r in list(csv.reader(fh))[1:]}
    assert widths == {45, 46}

    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    ws = result.stations["niwwswq"]
    assert ws.n_bad_timestamp == 0
    long_row = [o for o in ws.observations if o.ts == datetime(2016, 7, 1, 5, 0, tzinfo=UTC)]
    assert long_row and long_row[0].salinity_psu == 7.1


def test_no_unknown_columns_in_the_real_header():
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    assert result.unknown_columns == []


def test_exact_duplicate_rows_import_idempotently(tmp_path):
    """The real export repeats one instant (07/01/2026 00:00) verbatim for
    every live station -- byte-identical rows, an export-stitching artifact.
    `(station, ts)` dedupe absorbs them; nothing is lost or double-counted."""
    store = NdbcStore(tmp_path / "s.sqlite")
    report = import_file(REAL, store, stations=load_station_metadata(STATIONS))
    total_new = sum(s.n_new for s in report.stations) + sum(
        s.n_new for s in report.met_stations
    )
    with REAL.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))[1:]
    distinct = {(r[0].strip().lower(), r[2]) for r in rows}
    assert total_new == len(distinct) < len(rows)


# -- timestamps: fixed EST, verified against the file's own GMT offset --------


def test_niw_gmt_offset_in_the_metadata_file_is_minus_five():
    meta = load_station_metadata(STATIONS)
    niw = {c: m for c, m in meta.items() if c.startswith("niw")}
    assert niw
    assert {m.gmt_offset_hours for m in niw.values()} == {-5}
    assert {m.tz for m in niw.values()} == {CDMO_TZ}


def test_timestamp_uses_the_stations_own_offset_not_a_hardcoded_one():
    """Every NIW station is UTC-5, but the same metadata file gives -8 and
    -9 for other reserves. The offset is READ, so a Kachemak Bay export
    would not be silently shifted four hours."""
    meta = load_station_metadata(STATIONS)
    assert meta["kacsdwq"].gmt_offset_hours == -9
    assert meta["kacsdwq"].tz == timezone(timedelta(hours=-9))
    assert meta["sfbfmwq"].gmt_offset_hours == -8


def test_midnight_est_row_lands_at_0500_utc():
    result = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    cb = result.stations["niwcbwq"]
    first = min(o.ts for o in cb.observations)
    assert first == datetime(2016, 1, 1, 5, 0, tzinfo=UTC)
    assert first.astimezone(CDMO_TZ).hour == 0


# -- station metadata: the west-longitude trap --------------------------------


def test_longitude_is_negated_to_the_western_hemisphere():
    """`sampling_stations.csv` stores 79.1930411 for a station its own
    `Lat Long` text column describes as 79 deg 11' 34.95 W. Taken at face
    value that is western China."""
    meta = load_station_metadata(STATIONS)
    cb = meta["niwcbwq"]
    assert cb.lon == pytest.approx(-79.1930411)
    assert cb.lat == pytest.approx(33.3338636)
    assert all(m.lon < 0 for m in meta.values()), "every NERRS station is west of Greenwich"


def test_longitude_sign_agrees_with_the_files_own_dms_text_column():
    """Cross-check the decimal column against the human-readable DMS string
    in the same row -- two independent renderings of one position."""
    meta = load_station_metadata(STATIONS)
    with STATIONS.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [f.strip() for f in reader.fieldnames or []]
        for row in reader:
            code = row["Station Code"].strip().lower()
            text = row["Lat Long"]
            assert (" W" in text) == (meta[code].lon < 0)
            assert (" N" in text) == (meta[code].lat > 0)


def test_metadata_reads_every_station_in_the_file_not_a_hardcoded_list():
    meta = load_station_metadata(STATIONS)
    assert REAL_WQ_CODES | REAL_MET_CODES <= set(meta)
    # Stations the export does NOT contain are still read -- the loader is
    # driven by the file, so a future export gets them for free.
    assert {"niwcbnut", "niwcawq", "sfbfmwq", "kacsdwq"} <= set(meta)


def test_surface_and_bottom_share_a_position_but_stay_distinguishable():
    """niwwswq and niwwbwq are one mooring, two depths. Same position, two
    store keys -- the pair is what measures stratification."""
    meta = load_station_metadata(STATIONS)
    assert (meta["niwwswq"].lon, meta["niwwswq"].lat) == (
        meta["niwwbwq"].lon,
        meta["niwwbwq"].lat,
    )
    assert canonical_station("niwwswq") != canonical_station("niwwbwq")


def test_niwwswq_still_aliases_onto_wyss1_against_real_rows(tmp_path):
    """Task 9 built the alias from documentation. Prove it against real
    CDMO rows unioning with a real NDBC row rather than duplicating it."""
    store = NdbcStore(tmp_path / "s.sqlite")
    import_file(REAL, store, stations=load_station_metadata(STATIONS))
    assert store.count("WYSS1") > 0
    assert store.count("NIWWSWQ") == 0

    before = store.count("WYSS1")
    # An NDBC row at an instant the CDMO excerpt already holds must not
    # create a second row.
    existing = store.read("WYSS1")[0]
    assert store.append("WYSS1", [existing]) == 0
    assert store.count("WYSS1") == before


# -- streaming: 494 MB must never be materialised ------------------------------


def test_import_never_reads_the_whole_file_into_memory(tmp_path, monkeypatch):
    """The real export is 494 MB / 2,521,394 rows. `Path.read_text` on it
    would need roughly a gigabyte before a single row is parsed, and
    materialising 2.5 M `Observation` objects would need more again."""
    store = NdbcStore(tmp_path / "s.sqlite")

    def boom(*args, **kwargs):
        raise AssertionError("import_file must stream, not read the file whole")

    monkeypatch.setattr(Path, "read_text", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)
    report = import_file(REAL, store, stations=load_station_metadata(STATIONS))
    assert sum(s.n_new for s in report.stations) > 0


def test_streamed_import_flushes_in_batches(tmp_path, monkeypatch):
    """Batching is what bounds memory; prove more than one batch actually
    happens rather than trusting the constant."""
    import tidescout.sources.cdmo as cdmo_mod

    monkeypatch.setattr(cdmo_mod, "BATCH_ROWS", 4)
    store = NdbcStore(tmp_path / "s.sqlite")
    flushes = []
    real_flush = ndbc.BulkWriter.flush

    def counting_flush(self):
        flushes.append(1)
        return real_flush(self)

    monkeypatch.setattr(ndbc.BulkWriter, "flush", counting_flush)
    import_file(REAL, store, stations=load_station_metadata(STATIONS))
    assert len(flushes) > 1


# -- atomicity, at 2.5 M-row scale --------------------------------------------


def test_a_failure_deep_in_the_import_rolls_the_whole_file_back(tmp_path, monkeypatch):
    """Tasks 8 and 9 proved whole-batch atomicity on a handful of rows. The
    same standard has to hold when the write is streamed across many
    batches: a row that fails to bind AFTER earlier batches have already
    been handed to SQLite must leave the store exactly as it was -- neither
    partially loaded with this file, nor missing prior history."""
    import tidescout.sources.cdmo as cdmo_mod

    store = NdbcStore(tmp_path / "s.sqlite")
    seed = ndbc.Observation(
        ts=datetime(2001, 1, 1, tzinfo=UTC), depth_m=1.0, water_temp_c=12.0,
        cond_ms_cm=30.0, salinity_psu=19.0, o2_pct=90.0, o2_ppm=7.0,
        chlorophyll_ug_l=None, turbidity_ftu=3.0, ph=7.5, eh_mv=None,
    )
    assert store.append("WYSS1", [seed]) == 1
    before = store.read("WYSS1")

    monkeypatch.setattr(cdmo_mod, "BATCH_ROWS", 3)
    real_params = ndbc._wq_params
    calls = {"n": 0}

    def poisoned(station, obs):
        calls["n"] += 1
        if calls["n"] == 12:  # well past the first flushed batch
            return (station, obs.ts.astimezone(UTC).isoformat(), object(), *([None] * 9))
        return real_params(station, obs)

    monkeypatch.setattr(ndbc, "_wq_params", poisoned)

    with pytest.raises(sqlite3.ProgrammingError):
        import_file(REAL, store, stations=load_station_metadata(STATIONS))

    assert store.read("WYSS1") == before
    assert store.count("NIWCBWQ") == 0
    assert store.count_met("NIWOLMET") == 0
    assert store.provenance() == []


def test_a_bad_header_touches_nothing(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "broken.csv"
    path.write_text("Temp,F_Temp\n30.0,<0>\n")
    with pytest.raises(ValueError, match="missing required column"):
        import_file(path, store)
    assert store.count("WYSS1") == 0


# -- import_path against the real directory layout ----------------------------


def test_import_path_uses_sampling_stations_as_metadata_not_data(tmp_path):
    """The real export directory holds the data CSV and `sampling_stations
    .csv` side by side. Task 9's `import_path` globbed `*.csv` and would
    have tried to parse the station table as observations, raising on its
    missing `DateTimeStamp` and aborting the whole import."""
    d = tmp_path / "cdmo"
    d.mkdir()
    (d / "390918.csv").write_bytes(REAL.read_bytes())
    (d / "sampling_stations.csv").write_bytes(STATIONS.read_bytes())

    store = NdbcStore(tmp_path / "s.sqlite")
    reports = import_path(d, store)

    assert len(reports) == 1
    assert isinstance(reports[0], ImportReport)
    assert {s.raw_code for s in reports[0].stations} == REAL_WQ_CODES
    assert reports[0].station_metadata_source is not None


def test_import_path_without_a_station_table_still_imports(tmp_path):
    """Geolocation is a reporting nicety; the observations are the point."""
    d = tmp_path / "cdmo"
    d.mkdir()
    (d / "390918.csv").write_bytes(REAL.read_bytes())
    store = NdbcStore(tmp_path / "s.sqlite")
    reports = import_path(d, store)
    assert sum(s.n_new for s in reports[0].stations) > 0
    assert reports[0].station_metadata_source is None


# -- provenance and citation --------------------------------------------------


def test_one_interleaved_file_records_both_wq_and_met_provenance(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    import_file(REAL, store, stations=load_station_metadata(STATIONS))
    sources = {p.source for p in store.provenance()}
    assert sources == {"cdmo:water_quality", "cdmo:meteorological"}


def test_citation_reflects_the_cdmo_subset(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    import_file(REAL, store, stations=load_station_metadata(STATIONS))
    c = store.citation()
    assert c.accessed_date is not None
    assert "NERRS" in c.text
    joined = "\n".join(c.subset_lines)
    assert "WYSS1 (water quality" in joined
    assert "NIWWBWQ (water quality" in joined
    assert "NIWOLMET (meteorological" in joined
    assert set(c.sources) == {"cdmo:water_quality", "cdmo:meteorological"}


# -- salinity stays salinity --------------------------------------------------


def test_spcond_and_sal_stay_in_separate_columns(tmp_path):
    """Task 5's constraint. In the real file the two differ by more than a
    factor of two at every station; conflating them would be catastrophic
    and invisible."""
    store = NdbcStore(tmp_path / "s.sqlite")
    import_file(REAL, store, stations=load_station_metadata(STATIONS))
    rows = [o for o in store.read("WYSS1") if o.salinity_psu and o.cond_ms_cm]
    assert rows
    for o in rows:
        assert o.cond_ms_cm != o.salinity_psu
        assert o.cond_ms_cm > o.salinity_psu


def test_met_parse_view_sees_only_the_met_station():
    met = parse_cdmo_met_csv(REAL.read_text(encoding="utf-8-sig"))
    assert set(met.stations) == REAL_MET_CODES
    ol = met.stations["niwolmet"]
    assert ol.observations[0].air_temp_c == 20.5
    assert ol.observations[0].bp_mb == 1017.0


def test_wq_parse_view_sees_only_the_wq_stations():
    wq = parse_cdmo_csv(REAL.read_text(encoding="utf-8-sig"))
    assert set(wq.stations) == REAL_WQ_CODES
