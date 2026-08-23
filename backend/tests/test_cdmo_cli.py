"""`tidescout salinity import-cdmo` end to end, over a monkeypatched data dir.

Follows `test_flow_structure_cli.py`'s established pattern
(`monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")`) so this never
touches the real `data/winyah-bay/` -- important here specifically because
`sources/cdmo.py`'s tests use SYNTHETIC station histories, and those must
never land in the real accumulating store the live NDBC fetch and any
future CDMO import also write into.

`pipeline.salinity_fit.site_distances_km` is stubbed rather than exercised
for real: it needs a real bathymetry raster and along-estuary distance
field on disk, which is Task 1/5's setup, not this command's. The real
distance field IS exercised manually (see task-9-report.md) against the
real `data/winyah-bay/estuary_km.npy` -- that is what proves the actual
Winyah Bay numbers, not this file. What this file proves is the command's
OWN logic: aggregation across files, the in-domain/out-of-domain split,
the "no known position" path, and that a missing file fails loudly with
the exact path named -- none of which is covered by `test_cdmo.py`'s
direct calls into `sources.cdmo`.
"""

import re

from typer.testing import CliRunner

from tidescout.cli import app

runner = CliRunner()

WIDE = {"COLUMNS": "220"}  # rich wraps table cells at the terminal width otherwise

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result) -> str:
    """`result.stdout` with Rich's colour/highlighting codes stripped.

    Rich auto-highlights anything path-shaped, which splits a single
    printed string into several ANSI-coloured spans -- breaking a naive
    substring check across the split even though the text prints as one
    contiguous, readable path in a real terminal. Strip codes first,
    matching what a human actually reads.
    """
    return _ANSI.sub("", result.stdout)

HEADER = (
    "StationCode,isSWMP,DateTimeStamp,Historical,ProvisionalPlus,F_Record,"
    "Temp,F_Temp,SpCond,F_SpCond,Sal,F_Sal,DO_Pct,F_DO_Pct,DO_mgl,F_DO_mgl,"
    "Depth,F_Depth,cDepth,F_cDepth,Level,F_Level,cLevel,F_cLevel,"
    "pH,F_pH,Turb,F_Turb,ChlFluor,F_ChlFluor,\n"
)


def _row(station, ts, sal, f_sal="<0> "):
    return (
        f'"{station}   ","P",{ts},0,1,"",'
        f"30.1,<0> ,18.0,<0> ,{sal},{f_sal},"
        f"80.0,<0> ,5.5,<0> ,"
        f"0.6,<0> ,0.55,<3> ,"
        f'"",<-1> ,"",,'
        f"7.4,<0> ,9,<0> ,"
        f'"",<-1> ,\n'
    )


MET_HEADER = (
    "StationCode,isSWMP,DateTimeStamp,Historical,ProvisionalPlus,F_Record,"
    "ATemp,F_ATemp,RH,F_RH,BP,F_BP,WSpd,F_WSpd,MaxWSpd,F_MaxWSpd,"
    "Wdir,F_Wdir,SDWDir,F_SDWDir,TotPAR,F_TotPAR,TotPrcp,F_TotPrcp,"
    "CumPrcp,F_CumPrcp,TotSoRad,F_TotSoRad,\n"
)


def _met_row(station, ts):
    return (
        f'"{station}   ","P",{ts},0,1,"",'
        f"28.4,<0> ,71.2,<0> ,1015.3,<0> ,"
        f"3.4,<0> ,5.1,<0> ,"
        f"182.0,<0> ,14.2,<0> ,"
        f"612.0,<0> ,0.0,<0> ,"
        f'"",<-1> ,'
        f"285.0,<0> ,\n"
    )


def _patch_data_dir(monkeypatch, tmp_path):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")


def _stub_distances(monkeypatch, table: dict[str, tuple[float, float]]):
    """`{canonical_station: (distance_km, snap_gap_m)}` -- see module
    docstring for why the real distance field isn't exercised here."""
    from tidescout.pipeline import salinity_fit

    def fake(slug, fishery, sites):
        return {s: table[s] for s in sites if s in table}

    monkeypatch.setattr(salinity_fit, "site_distances_km", fake)


def test_missing_file_fails_loudly_and_names_the_expected_path(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    result = runner.invoke(app, ["salinity", "import-cdmo", "winyah-bay"], env=WIDE)
    assert result.exit_code == 1, result.stdout
    expected = str(tmp_path / "data" / "winyah-bay" / "cdmo")
    assert expected in plain(result)
    assert "import-cdmo winyah-bay --path" in plain(result)


def test_imports_and_reports_in_domain_station(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    _stub_distances(monkeypatch, {"WYSS1": (15.02, 9.5)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "niwwswq2026.csv").write_text(
        HEADER + _row("niwwswq", "08/22/2026 15:00", "15.2")
    )

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    # Check the WYSS1 table ROW specifically, not the whole output blob --
    # a bare substring check against `out` risked a false pass from digits
    # coincidentally inside the random tmp_path used elsewhere in the text.
    wyss1_lines = [ln for ln in out.splitlines() if "WYSS1" in ln]
    assert wyss1_lines, out
    assert "15.02" in wyss1_lines[0]
    assert "10" in wyss1_lines[0]  # snap gap 9.5 -> "10" (round-half-to-even)
    assert "1 in-domain station(s), 1 distinct along-estuary" in out

    # a second identical import must add nothing new -- the store dedupe
    # contract, exercised through the CLI rather than called directly.
    result2 = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )
    assert result2.exit_code == 0, result2.exception or result2.stdout
    out2 = plain(result2)
    lines = [ln for ln in out2.splitlines() if "WYSS1" in ln]
    assert lines, out2
    assert " 0 " in lines[0] or lines[0].strip().endswith("0")


def test_missing_distance_field_does_not_lose_an_already_committed_import(monkeypatch, tmp_path):
    """`site_distances_km` needs the along-estuary distance field
    (`tidescout salinity field <slug>` output) on disk -- a separate build
    step from this command. If it hasn't been run yet, the import (already
    committed to the store by this point) must not be thrown away over a
    reporting nicety; the command should degrade to "position unknown" and
    still exit 0."""
    _patch_data_dir(monkeypatch, tmp_path)

    from tidescout.pipeline import salinity_fit

    def missing_field(slug, fishery, sites):
        raise FileNotFoundError(f"no along-estuary distance field for {slug}")

    monkeypatch.setattr(salinity_fit, "site_distances_km", missing_field)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "niwwswq2026.csv").write_text(
        HEADER + _row("niwwswq", "08/22/2026 15:00", "15.2")
    )

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "no along-estuary distance field" in out
    assert "WYSS1" in out
    assert "no known position" in out

    # the row is durably in the store despite the distance-field failure
    from tidescout.sources.ndbc import default_store

    assert default_store("winyah-bay").count("WYSS1") == 1


def test_out_of_domain_station_is_flagged_not_silently_admitted(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    # A huge snap gap -- the USGS-gauge shape Task 5 documented.
    _stub_distances(monkeypatch, {"WYSS1": (31.57, 9498.0)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "niwwswq2026.csv").write_text(
        HEADER + _row("niwwswq", "08/22/2026 15:00", "15.2")
    )

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    assert "0 in-domain station(s)" in plain(result)


def test_station_with_no_known_position_is_reported_not_dropped(monkeypatch, tmp_path):
    """A CDMO station code this module has no coordinates for (outside the
    six documented NIW stations) must still be imported and named, not
    silently excluded from the report."""
    _patch_data_dir(monkeypatch, tmp_path)
    _stub_distances(monkeypatch, {})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "unknown2026.csv").write_text(
        HEADER + _row("xyzzzwq", "08/22/2026 15:00", "15.2")
    )

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "XYZZZWQ" in out
    assert "no known position" in out


def test_unknown_columns_are_surfaced_in_the_cli_output(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    _stub_distances(monkeypatch, {"WYSS1": (15.02, 9.5)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    header_with_extra = HEADER.replace(
        "ChlFluor,F_ChlFluor,", "ChlFluor,F_ChlFluor,Sonde_ID,"
    )
    row = _row("niwwswq", "08/22/2026 15:00", "15.2").replace("\n", ",ABC123\n")
    (export_dir / "niwwswq2026.csv").write_text(header_with_extra + row)

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    assert "Sonde_ID" in plain(result)


def test_unparseable_cells_and_bad_timestamps_are_surfaced_not_folded_into_new_count(
    monkeypatch, tmp_path
):
    """The parser tracks bad timestamps, unparseable values, and
    unparseable flags separately (see test_cdmo.py) -- this proves the CLI
    actually SHOWS them rather than computing and discarding them."""
    _patch_data_dir(monkeypatch, tmp_path)
    _stub_distances(monkeypatch, {"WYSS1": (15.02, 9.5)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    text = (
        HEADER
        + _row("niwwswq", "08/22/2026 15:00", "15.2")  # one good row
        + _row("niwwswq", "not-a-date", "16.0")  # bad timestamp
        + _row("niwwswq", "08/22/2026 15:15", "abc", f_sal="<0> ")  # unparseable value
        + _row("niwwswq", "08/22/2026 15:30", "16.5", f_sal="garbage")  # unparseable flag
    )
    (export_dir / "niwwswq2026.csv").write_text(text)

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "WYSS1" in out
    assert "could not parse" in out
    assert "unreadable timestamp" in out
    assert "unparseable value" in out and "sal:1" in out
    assert "unparseable flag" in out and "sal:1" in out


# -- MET files: the central proof this task asked for -------------------


def test_met_file_is_imported_and_reported_separately_from_wq(monkeypatch, tmp_path):
    """An importer that chokes on or silently ignores MET files wastes the
    one human action this whole data path depends on -- proved here via
    the CLI end to end, not just `sources.cdmo` directly."""
    _patch_data_dir(monkeypatch, tmp_path)
    _stub_distances(monkeypatch, {"WYSS1": (15.02, 9.5)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "niwwswq2026.csv").write_text(
        HEADER + _row("niwwswq", "08/22/2026 15:00", "15.2")
    )
    (export_dir / "niwolmet2026.csv").write_text(
        MET_HEADER + _met_row("niwolmet", "08/22/2026 15:00")
    )

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "2 file(s) read from" in out
    wyss1_lines = [ln for ln in out.splitlines() if "WYSS1" in ln]
    assert wyss1_lines, out
    assert "NIWOLMET" in out
    assert "meteorolog" in out.lower()

    from tidescout.sources.ndbc import default_store

    store = default_store("winyah-bay")
    assert store.count("WYSS1") == 1
    assert store.count_met("NIWOLMET") == 1


def test_met_only_import_does_not_break_the_wq_table(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "niwolmet2026.csv").write_text(
        MET_HEADER + _met_row("niwolmet", "08/22/2026 15:00")
    )

    result = runner.invoke(
        app, ["salinity", "import-cdmo", "winyah-bay", "--path", str(export_dir)], env=WIDE
    )

    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "NIWOLMET" in out
    assert "0 in-domain station(s)" in out  # the WQ table's own summary, unaffected
