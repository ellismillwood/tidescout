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
