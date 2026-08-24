"""`tidescout salinity citation` end to end, over a monkeypatched data dir.

Follows `test_cdmo_cli.py`'s established pattern so this never touches the
real `data/winyah-bay/ndbc.sqlite`. Proves the CLI surface Task 10 asked
for: "a clean way to ask the store for its citation ... a CLI command is
the obvious one."
"""

import re
from datetime import UTC, datetime

from typer.testing import CliRunner

from tidescout.cli import app

runner = CliRunner()

WIDE = {"COLUMNS": "220"}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result) -> str:
    return _ANSI.sub("", result.stdout)


def _patch_data_dir(monkeypatch, tmp_path):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")


def test_citation_on_an_empty_store_still_prints_a_complete_citation(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    result = runner.invoke(app, ["salinity", "citation", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "NOAA National Estuarine Research Reserve System (NERRS)" in out
    assert "doi:10.25921/vw8a-8031" in out
    assert "NO RECORDED ACCESS" in out
    assert "North Inlet-Winyah Bay NERR" in out
    assert "Federal government does not assume liability" in out


def test_citation_names_the_fishery_and_fails_on_an_unknown_slug(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    result = runner.invoke(app, ["salinity", "citation", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    assert "Winyah Bay" in plain(result)

    bad = runner.invoke(app, ["salinity", "citation", "not-a-real-fishery"], env=WIDE)
    assert bad.exit_code != 0


def test_citation_reflects_real_store_contents(monkeypatch, tmp_path):
    _patch_data_dir(monkeypatch, tmp_path)
    from tidescout.sources.ndbc import default_store

    store = default_store("winyah-bay")
    store.record_provenance(
        "ndbc:realtime2", ["WYSS1"], None, 5,
        accessed_at=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
    )
    from tidescout.sources.ndbc import parse_ocean

    ocean_text = (
        "#YY  MM DD hh mm   DEPTH  OTMP   COND   SAL   O2% O2PPM  CLCON  TURB    PH    EH\n"
        "#yr  mo dy hr mn       m  degC  mS/cm   psu     %   ppm   ug/l   FTU     -    mv\n"
        "2026 08 23 12 00     0.6 30.10  17.86 10.50  88.1  6.30     MM     9  7.40    MM\n"
    )
    store.append("WYSS1", parse_ocean(ocean_text))

    result = runner.invoke(app, ["salinity", "citation", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    assert "accessed 23 August 2026" in out
    assert "WYSS1" in out
    assert "water quality" in out


def test_citation_skips_the_wqp_store_when_it_was_never_imported(monkeypatch, tmp_path):
    """No `wqp.sqlite` on disk yet -- must not error, and must not create
    one as a side effect of a mere citation read (the same posture
    `wqp.station_coords` already takes)."""
    _patch_data_dir(monkeypatch, tmp_path)
    from tidescout import paths

    result = runner.invoke(app, ["salinity", "citation", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    assert not (paths.DATA_DIR / "winyah-bay" / "wqp.sqlite").exists()


def test_citation_includes_both_stores_when_both_exist(monkeypatch, tmp_path):
    """This is the whole point of Task 3's WQP ingestion: SC DES, SCDHEC and
    Coastal Carolina supply the large majority of this fishery's distinct
    along-estuary distances and are owed attribution on the same footing as
    NERRS. Before this fix the command opened only the NDBC store, so a WQP
    contributor's data was cited nowhere `tidescout salinity citation`
    reaches."""
    _patch_data_dir(monkeypatch, tmp_path)
    from tidescout.sources import wqp
    from tidescout.sources.ndbc import Observation, default_store

    ndbc_store = default_store("winyah-bay")
    ndbc_store.record_provenance(
        "ndbc:realtime2", ["WYSS1"], None, 5,
        accessed_at=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
    )

    wqp_store = wqp.default_store("winyah-bay")
    # A real observation row, not just a provenance record -- `subset_lines`
    # is read live from `observations`, not from provenance's own station
    # list (see `NdbcStore.citation`'s docstring), so a station only shows up
    # there once it has an actual row.
    wqp_store.append(
        "21SC60WQ_WQX-WB-06",
        [
            Observation(
                ts=datetime(2018, 7, 18, 12, 0, tzinfo=UTC),
                depth_m=None, water_temp_c=None, cond_ms_cm=None,
                salinity_psu=11.69, o2_pct=None, o2_ppm=None,
                chlorophyll_ug_l=None, turbidity_ftu=None, ph=None, eh_mv=None,
            )
        ],
    )
    wqp_store.record_provenance(
        "wqp:salinity", ["21SC60WQ_WQX-WB-06"], None, 1,
        accessed_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
    )

    result = runner.invoke(app, ["salinity", "citation", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    out = plain(result)
    # NERRS attribution -- still present, from the NDBC store.
    assert "NOAA National Estuarine Research Reserve System (NERRS)" in out
    # WQP attribution -- the fix: previously absent no matter what the WQP
    # store held, because the command never opened it.
    assert "Water Quality Portal" in out
    assert "Environmental Protection Agency" in out
    assert "21SC60WQ_WQX-WB-06" in out
