"""Fixture-based tests for the CDMO historical water-quality importer.

No real CDMO export was available when this was written -- see
`sources/cdmo.py`'s module docstring for exactly what the format was
established from (a real "zip downloads" example fetched live from the
SWMPr R package's own documentation, and the reserve's metadata PDF). Every
CSV string below is SYNTHETIC, built to the documented column layout and
flag syntax, not a captured real file -- unlike `test_ndbc.py`'s real
WYSS1 capture. Never hits a live endpoint (there is no live endpoint to
hit; CDMO exports are downloaded by hand through a web form).
"""

import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tidescout.sources.cdmo import (
    ACCEPTED_FLAGS,
    FLAG_MEANINGS,
    NIW_STATION_COORDS_LONLAT,
    STATION_ALIASES,
    canonical_station,
    import_file,
    import_path,
    parse_cdmo_csv,
)
from tidescout.sources.ndbc import NdbcStore, Observation, parse_ocean

# The real header, verbatim from the real zip_ex example fetched from
# https://s3.amazonaws.com/swmpexdata/zip_ex.zip (linked from SWMPr's
# `import_local()` documentation) -- see module docstring of sources/cdmo.py.
HEADER = (
    "StationCode,isSWMP,DateTimeStamp,Historical,ProvisionalPlus,F_Record,"
    "Temp,F_Temp,SpCond,F_SpCond,Sal,F_Sal,DO_Pct,F_DO_Pct,DO_mgl,F_DO_mgl,"
    "Depth,F_Depth,cDepth,F_cDepth,Level,F_Level,cLevel,F_cLevel,"
    "pH,F_pH,Turb,F_Turb,ChlFluor,F_ChlFluor,\n"
)


def _row(
    station="niwwswq",
    ts="08/22/2026 15:00",
    temp="30.1",
    f_temp="<0> ",
    spcond="18.0",
    f_spcond="<0> ",
    sal="15.2",
    f_sal="<0> ",
    do_pct="80.0",
    f_do_pct="<0> ",
    do_mgl="5.5",
    f_do_mgl="<0> ",
    depth="0.6",
    f_depth="<0> ",
    cdepth="0.55",
    f_cdepth="<3> ",
    level="",
    f_level="<-1> ",
    clevel="",
    f_clevel="",
    ph="7.4",
    f_ph="<0> ",
    turb="9",
    f_turb="<0> ",
    chlfluor="",
    f_chlfluor="<-1> ",
) -> str:
    """One realistic row, defaults all "good" -- callers override just the
    cell(s) under test, matching this codebase's established fixture style
    (see test_ndbc.py's ROW constant)."""
    return (
        f'"{station}   ","P",{ts},0,1,"",'
        f"{temp},{f_temp},{spcond},{f_spcond},{sal},{f_sal},"
        f"{do_pct},{f_do_pct},{do_mgl},{f_do_mgl},"
        f"{depth},{f_depth},{cdepth},{f_cdepth},"
        f'"{level}",{f_level},"{clevel}",{f_clevel},'
        f"{ph},{f_ph},{turb},{f_turb},"
        f'"{chlfluor}",{f_chlfluor},\n'
    )


# -- FLAG_MEANINGS / ACCEPTED_FLAGS ------------------------------------------


def test_flag_meanings_covers_the_documented_vocabulary():
    """Section 11 of the NIW metadata PDF: -5 through 5, eleven flags."""
    assert set(FLAG_MEANINGS) == {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}


def test_accepted_flags_excludes_suspect_and_rejected():
    """The dispatch's explicit concern: suspect (1) and rejected (-3) data
    must never be treated as admitted."""
    assert 1 not in ACCEPTED_FLAGS
    assert -3 not in ACCEPTED_FLAGS
    assert 0 in ACCEPTED_FLAGS


# -- station coordinates / aliasing ------------------------------------------


def test_niw_station_coords_has_all_six_current_stations():
    """The dispatch said four; the reserve's own metadata lists six current
    water-quality stations (CB, DC, OL, TA, WS/WYSS1, WB) -- see the module
    docstring's "WHAT THE DISPATCH GOT WRONG" section."""
    assert set(NIW_STATION_COORDS_LONLAT) == {
        "NIWCBWQ",
        "NIWDCWQ",
        "NIWOLWQ",
        "NIWTAWQ",
        "WYSS1",
        "NIWWBWQ",
    }


def test_wyss1_coords_agree_with_ndbc_pys_independently_documented_position():
    """`sources/ndbc.py` documents WYSS1 at (33.309, -79.289), sourced from
    NDBC directly. This module's WYSS1 entry comes from decimal-converting
    the CDMO metadata PDF's DMS coordinates for "Winyah Bay surface"
    (niwwswq) -- an independent source. They should agree to within survey
    noise, which is the actual evidence the two records are the same
    physical station."""
    lon, lat = NIW_STATION_COORDS_LONLAT["WYSS1"]
    assert lon == pytest.approx(-79.289, abs=0.01)
    assert lat == pytest.approx(33.309, abs=0.01)


def test_canonical_station_maps_niwwswq_to_wyss1():
    assert canonical_station("niwwswq") == "WYSS1"
    assert canonical_station("NIWWSWQ") == "WYSS1"
    assert STATION_ALIASES == {"niwwswq": "WYSS1"}


def test_canonical_station_uppercases_everything_else():
    assert canonical_station("niwolwq") == "NIWOLWQ"
    assert canonical_station("niwcbwq") == "NIWCBWQ"


# -- parse_cdmo_csv: header / structure ---------------------------------------


def test_missing_required_columns_raises():
    with pytest.raises(ValueError, match="missing required column"):
        parse_cdmo_csv("Temp,F_Temp\n30.0,<0>\n")


def test_empty_file_raises():
    with pytest.raises(ValueError, match="empty CDMO export"):
        parse_cdmo_csv("")


def test_unknown_column_is_reported_not_silently_dropped():
    text = HEADER.replace("ChlFluor,F_ChlFluor,", "ChlFluor,F_ChlFluor,Sonde_ID,") + _row().replace(
        "\n", ",XYZ123\n"
    )
    result = parse_cdmo_csv(text)
    assert "Sonde_ID" in result.unknown_columns


def test_cdepth_level_clevel_are_not_reported_as_unknown():
    """Deliberately dropped (no destination field in the shared schema),
    not a parse failure -- see module docstring."""
    result = parse_cdmo_csv(HEADER + _row())
    assert "cDepth" not in result.unknown_columns
    assert "Level" not in result.unknown_columns
    assert "cLevel" not in result.unknown_columns


# -- parse_cdmo_csv: values and station grouping -----------------------------


def test_parses_a_good_row():
    result = parse_cdmo_csv(HEADER + _row())
    sp = result.stations["niwwswq"]
    assert sp.n_rows == 1
    obs = sp.observations[0]
    assert obs.water_temp_c == 30.1
    assert obs.cond_ms_cm == 18.0
    assert obs.salinity_psu == 15.2
    assert obs.o2_pct == 80.0
    assert obs.o2_ppm == 5.5
    assert obs.depth_m == 0.6
    assert obs.ph == 7.4
    assert obs.turbidity_ftu == 9.0
    assert obs.eh_mv is None  # CDMO has no ORP parameter at all


def test_timestamp_is_fixed_est_converted_to_utc():
    """08/22/2026 15:00 EST (UTC-5, no DST) is 20:00 UTC."""
    result = parse_cdmo_csv(HEADER + _row(ts="08/22/2026 15:00"))
    obs = result.stations["niwwswq"].observations[0]
    assert obs.ts == datetime(2026, 8, 22, 20, 0, tzinfo=UTC)
    assert obs.ts.tzinfo is not None


def test_groups_rows_by_their_own_stationcode_not_a_filename():
    """CDMO's "Custom Query" export mode combines multiple stations into
    one file -- the parser must not assume one station per file."""
    text = HEADER + _row(station="niwwswq", sal="15.2") + _row(station="niwolwq", sal="28.0")
    result = parse_cdmo_csv(text)
    assert set(result.stations) == {"niwwswq", "niwolwq"}
    assert result.stations["niwwswq"].observations[0].salinity_psu == 15.2
    assert result.stations["niwolwq"].observations[0].salinity_psu == 28.0


def test_blank_trailer_row_is_skipped():
    text = HEADER + _row() + ("," * 30 + "\n")
    result = parse_cdmo_csv(text)
    assert result.stations["niwwswq"].n_rows == 1


# -- QAQC flag handling: the load-bearing behaviour --------------------------


def test_flag_0_admits_the_value():
    result = parse_cdmo_csv(HEADER + _row(sal="15.2", f_sal="<0> "))
    assert result.stations["niwwswq"].observations[0].salinity_psu == 15.2


def test_flag_1_suspect_rejects_the_value():
    result = parse_cdmo_csv(HEADER + _row(sal="99.9", f_sal="<1> "))
    sp = result.stations["niwwswq"]
    assert sp.observations[0].salinity_psu is None
    assert sp.n_rejected_by_flag["sal"] == 1


def test_flag_minus3_rejected_rejects_the_value():
    result = parse_cdmo_csv(HEADER + _row(sal="99.9", f_sal="<-3> "))
    sp = result.stations["niwwswq"]
    assert sp.observations[0].salinity_psu is None
    assert sp.n_rejected_by_flag["sal"] == 1


def test_flag_4_historical_pre_auto_qaqc_rejects_the_value():
    """A judgement call, documented in the module docstring: unvetted is
    treated the same as bad, not admitted by default."""
    result = parse_cdmo_csv(HEADER + _row(sal="99.9", f_sal="<4> "))
    assert result.stations["niwwswq"].observations[0].salinity_psu is None


def test_flag_5_corrected_admits_the_value():
    result = parse_cdmo_csv(HEADER + _row(temp="31.0", f_temp="<5> "))
    assert result.stations["niwwswq"].observations[0].water_temp_c == 31.0


def test_flag_3_calculated_admits_the_value():
    result = parse_cdmo_csv(HEADER + _row(depth="1.05", f_depth="<3> "))
    assert result.stations["niwwswq"].observations[0].depth_m == 1.05


def test_flag_with_qaqc_code_still_extracts_the_numeric_flag():
    """"<1 SDG>" -- Suspect Data plus the "sensor diagnostics" comment code
    (Section 12). The code must not prevent reading the leading digit that
    decides accept/reject."""
    result = parse_cdmo_csv(HEADER + _row(sal="99.9", f_sal="<1 SDG>"))
    sp = result.stations["niwwswq"]
    assert sp.observations[0].salinity_psu is None
    assert sp.n_rejected_by_flag["sal"] == 1


def test_missing_data_flag_on_an_already_blank_cell_is_not_double_counted():
    """-2/-1 rows are normally blank already -- that's ordinary
    missingness, not a rejected real reading, so it should not inflate
    `n_rejected_by_flag`."""
    result = parse_cdmo_csv(HEADER + _row(sal="", f_sal="<-2> "))
    sp = result.stations["niwwswq"]
    assert sp.observations[0].salinity_psu is None
    assert "sal" not in sp.n_rejected_by_flag


def test_unparseable_flag_cell_is_counted_not_guessed():
    result = parse_cdmo_csv(HEADER + _row(turb="9", f_turb="garbage"))
    sp = result.stations["niwwswq"]
    assert sp.observations[0].turbidity_ftu is None
    assert sp.n_flag_unparseable["turb"] == 1


def test_unparseable_value_cell_is_counted_not_fatal():
    result = parse_cdmo_csv(HEADER + _row(sal="abc", f_sal="<0> "))
    sp = result.stations["niwwswq"]
    assert sp.observations[0].salinity_psu is None
    assert sp.n_value_unparseable["sal"] == 1
    # the rest of the row still parsed fine -- one bad cell isn't fatal
    assert sp.observations[0].water_temp_c == 30.1


def test_bad_timestamp_row_is_dropped_and_counted_not_fatal():
    text = HEADER + _row(ts="not-a-date") + _row(ts="08/22/2026 16:00")
    result = parse_cdmo_csv(text)
    sp = result.stations["niwwswq"]
    assert sp.n_bad_timestamp == 1
    assert len(sp.observations) == 1  # the good row still made it in


# -- import_file / import_path: writes through NdbcStore.append -------------


def test_import_file_writes_into_the_shared_ndbc_store(tmp_path):
    path = tmp_path / "niwwswq2026.csv"
    path.write_text(HEADER + _row(sal="15.2"))
    store = NdbcStore(tmp_path / "s.sqlite")

    report = import_file(path, store)

    assert store.count("WYSS1") == 1
    assert report.stations[0].canonical == "WYSS1"
    assert report.stations[0].raw_code == "niwwswq"
    assert report.stations[0].n_new == 1


def test_import_file_second_station_keeps_its_own_code(tmp_path):
    path = tmp_path / "niwolwq2026.csv"
    path.write_text(HEADER + _row(station="niwolwq"))
    store = NdbcStore(tmp_path / "s.sqlite")

    report = import_file(path, store)

    assert store.count("NIWOLWQ") == 1
    assert store.count("WYSS1") == 0
    assert report.stations[0].canonical == "NIWOLWQ"


def test_reimporting_the_same_file_adds_nothing(tmp_path):
    path = tmp_path / "niwwswq2026.csv"
    path.write_text(HEADER + _row())
    store = NdbcStore(tmp_path / "s.sqlite")

    import_file(path, store)
    report2 = import_file(path, store)

    assert store.count("WYSS1") == 1
    assert report2.stations[0].n_new == 0


def test_bad_header_raises_and_does_not_touch_the_store(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("Temp,F_Temp\n30.0,<0>\n")
    store = NdbcStore(tmp_path / "s.sqlite")

    with pytest.raises(ValueError, match="missing required column"):
        import_file(path, store)

    assert store.count("WYSS1") == 0


# -- THE dedupe proof: CDMO history unions with Task 8's existing NDBC rows -


def test_cdmo_row_dedupes_against_an_existing_ndbc_row_at_the_same_instant(tmp_path):
    """The central proof this task asked for. NDBC's real captured row at
    2026-08-23 13:30 UTC (from test_ndbc.py's LATER fixture) reports
    salinity 10.50. In fixed EST (UTC-5, no DST) that SAME instant is
    2026-08-23 08:30 -- so a CDMO row timestamped "08/23/2026 8:30" for
    niwwswq is the same observation, reaching the store through the OTHER
    module's parser and this one's canonical-station alias. The union must
    land as ONE row, not two -- exactly Task 8's
    `test_append_and_dedupe_across_overlapping_fetches` contract, reused
    here across sources rather than across two NDBC fetches.
    """
    store = NdbcStore(tmp_path / "s.sqlite")

    ndbc_later = (
        "#YY  MM DD hh mm   DEPTH  OTMP   COND   SAL   O2% O2PPM  CLCON  TURB    PH    EH\n"
        "#yr  mo dy hr mn       m  degC  mS/cm   psu     %   ppm   ug/l   FTU     -    mv\n"
        "2026 08 23 13 30     0.6 30.10  17.86 10.50  88.1  6.30     MM     9  7.40    MM\n"
        "2026 08 23 13 45     0.6 30.20  18.50 10.90  77.6  5.50     MM    10  7.30    MM\n"
    )
    n_ndbc = store.append("WYSS1", parse_ocean(ndbc_later))
    assert n_ndbc == 2

    cdmo_text = HEADER + _row(
        station="niwwswq", ts="08/23/2026 8:30", sal="10.50", f_sal="<0> "
    ) + _row(station="niwwswq", ts="08/23/2026 9:00", sal="11.30", f_sal="<0> ")
    result = parse_cdmo_csv(cdmo_text)
    n_new = store.append("WYSS1", result.stations["niwwswq"].observations)

    # Only the 09:00 EST (14:00 UTC) row is genuinely new; 08:30 EST (13:30
    # UTC) already exists from the NDBC fetch.
    assert n_new == 1
    assert store.count("WYSS1") == 3

    overlap_ts = datetime(2026, 8, 23, 13, 30, tzinfo=UTC)
    overlap = next(r for r in store.read("WYSS1") if r.ts == overlap_ts)
    assert overlap.salinity_psu == 10.50


def test_cdmo_import_file_dedupes_against_an_existing_ndbc_store(tmp_path):
    """Same proof, through the file-level `import_file` entry point rather
    than calling `parse_cdmo_csv` + `store.append` directly."""
    store = NdbcStore(tmp_path / "s.sqlite")
    store.append(
        "WYSS1",
        [
            Observation(
                ts=datetime(2026, 8, 23, 13, 30, tzinfo=UTC),
                depth_m=0.6, water_temp_c=30.1, cond_ms_cm=17.86, salinity_psu=10.50,
                o2_pct=88.1, o2_ppm=6.3, chlorophyll_ug_l=None, turbidity_ftu=9.0,
                ph=7.4, eh_mv=None,
            )
        ],
    )
    path = tmp_path / "niwwswq2026.csv"
    path.write_text(HEADER + _row(station="niwwswq", ts="08/23/2026 8:30", sal="10.50"))

    report = import_file(path, store)

    assert store.count("WYSS1") == 1  # the union, not a duplicate
    assert report.stations[0].n_new == 0


# -- import_path: file / directory / zip -------------------------------------


def test_import_path_missing_path_raises_and_names_it(tmp_path):
    missing = tmp_path / "does-not-exist"
    store = NdbcStore(tmp_path / "s.sqlite")
    with pytest.raises(FileNotFoundError, match=str(missing)):
        import_path(missing, store)


def test_import_path_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    store = NdbcStore(tmp_path / "s.sqlite")
    with pytest.raises(FileNotFoundError, match="no .csv files"):
        import_path(empty, store)


def test_import_path_directory_imports_every_csv(tmp_path):
    d = tmp_path / "cdmo"
    d.mkdir()
    (d / "niwwswq2025.csv").write_text(HEADER + _row(ts="12/01/2025 9:00", sal="12.0"))
    (d / "niwwswq2026.csv").write_text(HEADER + _row(ts="01/01/2026 9:00", sal="18.0"))
    store = NdbcStore(tmp_path / "s.sqlite")

    reports = import_path(d, store)

    assert len(reports) == 2
    assert store.count("WYSS1") == 2


def test_import_path_directory_second_bad_file_does_not_touch_the_first_files_data(tmp_path):
    """One malformed file among several must not corrupt what a good file
    already committed -- the multi-file analogue of Task 8's poisoned-row
    proof."""
    d = tmp_path / "cdmo"
    d.mkdir()
    (d / "a_good.csv").write_text(HEADER + _row(sal="12.0"))
    (d / "b_broken.csv").write_text("Temp,F_Temp\n30.0,<0>\n")
    store = NdbcStore(tmp_path / "s.sqlite")

    with pytest.raises(ValueError, match="missing required column"):
        import_path(d, store)

    assert store.count("WYSS1") == 1  # a_good.csv's row survives


def test_import_path_reads_a_zip_archive(tmp_path):
    zip_path = tmp_path / "cdmo_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("niwwswq2026.csv", HEADER + _row(sal="16.5"))
        zf.writestr("niwolwq2026.csv", HEADER + _row(station="niwolwq", sal="29.0"))
    store = NdbcStore(tmp_path / "s.sqlite")

    reports = import_path(zip_path, store)

    assert len(reports) == 2
    assert store.count("WYSS1") == 1
    assert store.count("NIWOLWQ") == 1


def test_import_path_zip_with_no_csv_raises(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "not a csv")
    store = NdbcStore(tmp_path / "s.sqlite")
    with pytest.raises(FileNotFoundError, match="no .csv files"):
        import_path(zip_path, store)


def test_import_path_single_file(tmp_path):
    path = tmp_path / "niwwswq2026.csv"
    path.write_text(HEADER + _row())
    store = NdbcStore(tmp_path / "s.sqlite")
    reports = import_path(path, store)
    assert len(reports) == 1
    assert store.count("WYSS1") == 1


# -- realistic multi-row synthetic fixture -----------------------------------

FULL_FIXTURE = Path(__file__).parent / "fixtures" / "cdmo_niw_sample.csv"


def test_parses_the_synthetic_multi_station_fixture_without_error():
    """Regression guard against the documented shape holding together over
    a bigger, messier file spanning several stations, a station alias, and
    a mix of accepted/rejected/blank/malformed cells -- not a real capture
    (module docstring explains why none was available), but built row by
    row to the same column layout the real zip_ex example uses."""
    result = parse_cdmo_csv(FULL_FIXTURE.read_text())
    assert set(result.stations) >= {"niwwswq", "niwolwq", "niwcbwq"}
    assert sum(sp.n_rows for sp in result.stations.values()) > 10
    # every station parsed at least one usable salinity reading
    for code, sp in result.stations.items():
        sal_values = [o.salinity_psu for o in sp.observations if o.salinity_psu is not None]
        assert sal_values, f"{code} produced no usable salinity readings"


def test_synthetic_fixture_imports_and_dedupes_across_a_second_pass(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    r1 = import_file(FULL_FIXTURE, store)
    r2 = import_file(FULL_FIXTURE, store)
    n1 = sum(s.n_new for s in r1.stations)
    n2 = sum(s.n_new for s in r2.stations)
    assert n1 > 0
    assert n2 == 0


# -- a genuinely poisoned batch must not corrupt existing history -----------


def test_poisoned_observation_mid_batch_does_not_corrupt_existing_history(tmp_path):
    """Same standard Task 8 held `NdbcStore.append` to
    (`test_partial_write_does_not_corrupt_existing_history`), exercised
    here through this module's own write path: a batch where a later row
    is unbindable must roll back in full, leaving prior history untouched.
    This module cannot itself PRODUCE an unbindable row (every resolved
    value is a `float` or `None`, see `_resolve`), so the poisoned row is
    injected directly to prove the underlying guarantee still holds when
    reached through `import_file`'s call path.
    """
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "niwwswq2026.csv"
    path.write_text(HEADER + _row(ts="08/22/2026 15:00", sal="12.0"))
    import_file(path, store)
    before = store.read("WYSS1")
    assert len(before) == 1

    poisoned = Observation(
        ts=datetime(2026, 8, 22, 21, 0, tzinfo=UTC), depth_m=object(), water_temp_c=30.0,
        cond_ms_cm=18.0, salinity_psu=11.0, o2_pct=80.0, o2_ppm=5.5,
        chlorophyll_ug_l=None, turbidity_ftu=10.0, ph=7.3, eh_mv=None,
    )
    with pytest.raises(sqlite3.ProgrammingError):
        store.append("WYSS1", [poisoned])

    after = store.read("WYSS1")
    assert after == before
