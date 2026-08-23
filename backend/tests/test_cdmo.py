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
    MET_ACCEPTED_FLAGS,
    MET_FLAG_MEANINGS,
    NIW_MET_STATION_COORDS_LONLAT,
    NIW_STATION_COORDS_LONLAT,
    SOURCE_CDMO_MET,
    SOURCE_CDMO_WQ,
    STATION_ALIASES,
    canonical_station,
    import_file,
    import_path,
    parse_cdmo_csv,
    parse_cdmo_met_csv,
)
from tidescout.sources.ndbc import MetObservation, NdbcStore, Observation, parse_ocean

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


# =============================================================================
# MET (meteorological) file support -- Task 10
#
# The reserve's own meteorological metadata PDF (fetched live 2026-08-23:
# https://cdmo.baruch.sc.edu/waf/YearlyFiles/North%20Inlet%20Winyah%20Bay/
# meteorological/metadata/niwmet01-12.24m.pdf) plus SWMPr's own
# `R/param_names.R` (fetched live from github.com/fawda123/SWMPr) are what
# the header/flag/coordinate fixtures below are built to -- see
# `sources/cdmo.py`'s "MET FILE SUPPORT" docstring section for the full
# derivation. No real MET export was available (same situation Task 9 was
# in for water quality); every row below is synthetic.
# =============================================================================

# The real MET column order, per SWMPr's `param_names.R` MET list (atemp,
# rh, bp, wspd, maxwspd, wdir, sdwdir, totpar, totprcp, cumprcp, totsorad),
# title-cased to CDMO's export convention and paired with F_ flag columns
# exactly like the WQ header above.
MET_HEADER = (
    "StationCode,isSWMP,DateTimeStamp,Historical,ProvisionalPlus,F_Record,"
    "ATemp,F_ATemp,RH,F_RH,BP,F_BP,WSpd,F_WSpd,MaxWSpd,F_MaxWSpd,"
    "Wdir,F_Wdir,SDWDir,F_SDWDir,TotPAR,F_TotPAR,TotPrcp,F_TotPrcp,"
    "CumPrcp,F_CumPrcp,TotSoRad,F_TotSoRad,\n"
)


def _met_row(
    station="niwolmet",
    ts="08/22/2026 15:00",
    atemp="28.4",
    f_atemp="<0> ",
    rh="71.2",
    f_rh="<0> ",
    bp="1015.3",
    f_bp="<0> ",
    wspd="3.4",
    f_wspd="<0> ",
    maxwspd="5.1",
    f_maxwspd="<0> ",
    wdir="182.0",
    f_wdir="<0> ",
    sdwdir="14.2",
    f_sdwdir="<0> ",
    totpar="612.0",
    f_totpar="<0> ",
    totprcp="0.0",
    f_totprcp="<0> ",
    cumprcp="",
    f_cumprcp="<-1> ",
    totsorad="285.0",
    f_totsorad="<0> ",
) -> str:
    """One realistic MET row, defaults all "good" -- same fixture style as
    the WQ `_row()` above."""
    return (
        f'"{station}   ","P",{ts},0,1,"",'
        f"{atemp},{f_atemp},{rh},{f_rh},{bp},{f_bp},"
        f"{wspd},{f_wspd},{maxwspd},{f_maxwspd},"
        f"{wdir},{f_wdir},{sdwdir},{f_sdwdir},"
        f"{totpar},{f_totpar},{totprcp},{f_totprcp},"
        f'"{cumprcp}",{f_cumprcp},'
        f"{totsorad},{f_totsorad},\n"
    )


# -- MET_FLAG_MEANINGS / MET_ACCEPTED_FLAGS ----------------------------------


def test_met_flag_meanings_covers_the_documented_vocabulary():
    assert set(MET_FLAG_MEANINGS) == {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}


def test_met_accepted_flags_excludes_reserved_flags_2_and_3():
    """A real difference from WQ: the MET metadata PDF's own QAQC section
    documents flags 2 and 3 as "Open - reserved for later flag" for MET
    parameters (WQ uses them for depth-surface/barometric-correction
    meanings that don't apply to weather data at all). Admitting them here
    would invent a meaning CDMO has not assigned."""
    assert 2 not in MET_ACCEPTED_FLAGS
    assert 3 not in MET_ACCEPTED_FLAGS
    assert MET_ACCEPTED_FLAGS == {0, 5}


def test_met_flag_meanings_2_and_3_say_reserved_not_a_wq_meaning():
    assert "reserved" in MET_FLAG_MEANINGS[2].lower()
    assert "reserved" in MET_FLAG_MEANINGS[3].lower()


def test_met_accepted_flags_still_excludes_suspect_and_rejected():
    assert 1 not in MET_ACCEPTED_FLAGS
    assert -3 not in MET_ACCEPTED_FLAGS
    assert 0 in MET_ACCEPTED_FLAGS


# -- station coordinates ------------------------------------------------------


def test_niw_met_station_coords_has_exactly_one_station():
    """NIW runs ONE weather station system-wide (Oyster Landing), unlike
    the six WQ stations -- see the met metadata PDF's own "SWMP station
    timeline" table."""
    assert set(NIW_MET_STATION_COORDS_LONLAT) == {"NIWOLMET"}


def test_met_station_position_is_near_but_not_identical_to_ol_wq_position():
    """Same pier, independently surveyed sensor mounts -- the met metadata
    PDF's own coordinates (33 20'57.85"N, 79 11'20.03"W) differ from the WQ
    metadata PDF's Oyster Landing coordinates (33 20'57.70"N, 79 11'19.97"W)
    by a few feet. Close enough to agree it's the same pier, not so close
    that collapsing them onto one shared position would be honest."""
    met_lon, met_lat = NIW_MET_STATION_COORDS_LONLAT["NIWOLMET"]
    wq_lon, wq_lat = NIW_STATION_COORDS_LONLAT["NIWOLWQ"]
    assert met_lon == pytest.approx(wq_lon, abs=0.001)
    assert met_lat == pytest.approx(wq_lat, abs=0.001)
    assert (met_lon, met_lat) != (wq_lon, wq_lat)


# -- parse_cdmo_met_csv: header / structure -----------------------------------


def test_met_missing_required_columns_raises():
    with pytest.raises(ValueError, match="missing required column"):
        parse_cdmo_met_csv("ATemp,F_ATemp\n28.0,<0>\n")


def test_met_unknown_column_is_reported_not_silently_dropped():
    text = MET_HEADER.replace(
        "TotSoRad,F_TotSoRad,", "TotSoRad,F_TotSoRad,Battery,"
    ) + _met_row().replace("\n", ",12.1\n")
    result = parse_cdmo_met_csv(text)
    assert "Battery" in result.unknown_columns


def test_cumprcp_is_dropped_not_reported_as_unknown():
    """No longer available via CDMO export (met metadata PDF remark 13.d) --
    deliberately dropped, same treatment as WQ's cDepth/Level/cLevel."""
    result = parse_cdmo_met_csv(MET_HEADER + _met_row())
    assert "CumPrcp" not in result.unknown_columns


# -- parse_cdmo_met_csv: values -----------------------------------------------


def test_parses_a_good_met_row():
    result = parse_cdmo_met_csv(MET_HEADER + _met_row())
    sp = result.stations["niwolmet"]
    assert sp.n_rows == 1
    obs = sp.observations[0]
    assert obs.air_temp_c == 28.4
    assert obs.rh_pct == 71.2
    assert obs.bp_mb == 1015.3
    assert obs.wind_speed_ms == 3.4
    assert obs.max_wind_speed_ms == 5.1
    assert obs.wind_dir_deg == 182.0
    assert obs.wind_dir_sd_deg == 14.2
    assert obs.par_mmol_m2 == 612.0
    assert obs.precip_mm == 0.0
    assert obs.solar_rad_wm2 == 285.0


def test_met_timestamp_is_fixed_est_converted_to_utc():
    result = parse_cdmo_met_csv(MET_HEADER + _met_row(ts="08/22/2026 15:00"))
    obs = result.stations["niwolmet"].observations[0]
    assert obs.ts == datetime(2026, 8, 22, 20, 0, tzinfo=UTC)


def test_met_flag_0_and_5_admit_the_value():
    result = parse_cdmo_met_csv(MET_HEADER + _met_row(atemp="30.0", f_atemp="<5> "))
    assert result.stations["niwolmet"].observations[0].air_temp_c == 30.0


def test_met_flag_1_suspect_rejects_the_value():
    result = parse_cdmo_met_csv(MET_HEADER + _met_row(wspd="99.0", f_wspd="<1> "))
    sp = result.stations["niwolmet"]
    assert sp.observations[0].wind_speed_ms is None
    assert sp.n_rejected_by_flag["wspd"] == 1


def test_met_flag_2_is_rejected_unlike_wq():
    """The load-bearing MET-vs-WQ difference under test."""
    result = parse_cdmo_met_csv(MET_HEADER + _met_row(bp="1099.0", f_bp="<2> "))
    sp = result.stations["niwolmet"]
    assert sp.observations[0].bp_mb is None
    assert sp.n_rejected_by_flag["bp"] == 1


def test_met_flag_3_is_rejected_unlike_wq():
    result = parse_cdmo_met_csv(MET_HEADER + _met_row(rh="150.0", f_rh="<3> "))
    sp = result.stations["niwolmet"]
    assert sp.observations[0].rh_pct is None
    assert sp.n_rejected_by_flag["rh"] == 1


def test_met_groups_rows_by_their_own_stationcode():
    text = MET_HEADER + _met_row(station="niwolmet")
    result = parse_cdmo_met_csv(text)
    assert set(result.stations) == {"niwolmet"}


def test_met_bad_timestamp_row_is_dropped_and_counted_not_fatal():
    text = MET_HEADER + _met_row(ts="not-a-date") + _met_row(ts="08/22/2026 16:00")
    result = parse_cdmo_met_csv(text)
    sp = result.stations["niwolmet"]
    assert sp.n_bad_timestamp == 1
    assert len(sp.observations) == 1


def test_met_unparseable_value_cell_is_counted_not_fatal():
    result = parse_cdmo_met_csv(MET_HEADER + _met_row(wspd="abc", f_wspd="<0> "))
    sp = result.stations["niwolmet"]
    assert sp.observations[0].wind_speed_ms is None
    assert sp.n_value_unparseable["wspd"] == 1
    assert sp.observations[0].air_temp_c == 28.4  # rest of the row still parsed


# -- import_file: routes to the MET store path --------------------------------


def test_import_file_detects_a_met_file_and_writes_into_met_observations(tmp_path):
    path = tmp_path / "niwolmet2026.csv"
    path.write_text(MET_HEADER + _met_row())
    store = NdbcStore(tmp_path / "s.sqlite")

    report = import_file(path, store)

    assert store.count_met("NIWOLMET") == 1
    assert store.count("NIWOLMET") == 0  # never written to the WQ table
    assert report.stations[0].canonical == "NIWOLMET"
    assert report.stations[0].raw_code == "niwolmet"
    assert report.stations[0].n_new == 1


def test_reimporting_the_same_met_file_adds_nothing(tmp_path):
    path = tmp_path / "niwolmet2026.csv"
    path.write_text(MET_HEADER + _met_row())
    store = NdbcStore(tmp_path / "s.sqlite")

    import_file(path, store)
    report2 = import_file(path, store)

    assert store.count_met("NIWOLMET") == 1
    assert report2.stations[0].n_new == 0


def test_bad_met_header_raises_and_does_not_touch_the_store(tmp_path):
    path = tmp_path / "broken_met.csv"
    path.write_text("ATemp,F_ATemp\n28.0,<0>\n")
    store = NdbcStore(tmp_path / "s.sqlite")

    with pytest.raises(ValueError, match="missing required column"):
        import_file(path, store)

    assert store.count_met("NIWOLMET") == 0


def test_import_path_directory_with_mixed_wq_and_met_files(tmp_path):
    """An importer that chokes on or ignores MET files wastes the one human
    action this whole data path depends on -- the central proof this task
    asked for."""
    d = tmp_path / "cdmo"
    d.mkdir()
    (d / "niwwswq2026.csv").write_text(HEADER + _row(sal="15.2"))
    (d / "niwolmet2026.csv").write_text(MET_HEADER + _met_row())
    store = NdbcStore(tmp_path / "s.sqlite")

    reports = import_path(d, store)

    assert len(reports) == 2
    assert store.count("WYSS1") == 1
    assert store.count_met("NIWOLMET") == 1


def test_import_path_zip_with_mixed_wq_and_met_files(tmp_path):
    zip_path = tmp_path / "cdmo_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("niwwswq2026.csv", HEADER + _row(sal="16.5"))
        zf.writestr("niwolmet2026.csv", MET_HEADER + _met_row())
    store = NdbcStore(tmp_path / "s.sqlite")

    reports = import_path(zip_path, store)

    assert len(reports) == 2
    assert store.count("WYSS1") == 1
    assert store.count_met("NIWOLMET") == 1


# -- MET dedupe against a pre-existing store, and atomicity -------------------


def test_met_import_file_dedupes_across_a_second_pass(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "niwolmet2026.csv"
    path.write_text(
        MET_HEADER
        + _met_row(ts="08/22/2026 15:00")
        + _met_row(ts="08/22/2026 15:15")
    )

    r1 = import_file(path, store)
    r2 = import_file(path, store)

    assert sum(s.n_new for s in r1.stations) == 2
    assert sum(s.n_new for s in r2.stations) == 0
    assert store.count_met("NIWOLMET") == 2


def test_poisoned_met_observation_mid_batch_does_not_corrupt_existing_history(tmp_path):
    """Same standard Task 8/9 held the WQ path to, exercised for MET
    through `import_file`'s own call path."""
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "niwolmet2026.csv"
    path.write_text(MET_HEADER + _met_row(ts="08/22/2026 15:00"))
    import_file(path, store)
    before = store.read_met("NIWOLMET")
    assert len(before) == 1

    poisoned = MetObservation(
        ts=datetime(2026, 8, 22, 21, 0, tzinfo=UTC), air_temp_c=object(), rh_pct=60.0,
        bp_mb=1013.0, wind_speed_ms=2.0, max_wind_speed_ms=3.0, wind_dir_deg=90.0,
        wind_dir_sd_deg=5.0, par_mmol_m2=300.0, precip_mm=0.0, solar_rad_wm2=150.0,
    )
    with pytest.raises(sqlite3.ProgrammingError):
        store.append_met("NIWOLMET", [poisoned])

    after = store.read_met("NIWOLMET")
    assert after == before


# -- provenance is recorded through the CDMO import path ----------------------


def test_cdmo_wq_import_records_provenance(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "niwwswq2026.csv"
    path.write_text(HEADER + _row(sal="15.2"))

    import_file(path, store)

    recs = store.provenance()
    assert len(recs) == 1
    assert recs[0].source == SOURCE_CDMO_WQ
    assert recs[0].stations == ("WYSS1",)
    assert recs[0].n_new == 1


def test_cdmo_met_import_records_provenance(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "niwolmet2026.csv"
    path.write_text(MET_HEADER + _met_row())

    import_file(path, store)

    recs = store.provenance()
    assert len(recs) == 1
    assert recs[0].source == SOURCE_CDMO_MET
    assert recs[0].stations == ("NIWOLMET",)
    assert recs[0].n_new == 1


def test_cdmo_import_failure_records_no_provenance(tmp_path):
    store = NdbcStore(tmp_path / "s.sqlite")
    path = tmp_path / "broken.csv"
    path.write_text("Temp,F_Temp\n30.0,<0>\n")

    with pytest.raises(ValueError):
        import_file(path, store)

    assert store.provenance() == []
