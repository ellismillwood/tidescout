"""`tidescout score`'s own selection logic: which sub-scores it prints.

Everything else about the command (the real `build_payload` call, the real
engine) is covered by `test_payload.py`; this file stubs `build_payload`
entirely so it can pin the ~15 lines of formatting logic in `cli.py`'s
`score` command in isolation and fast, the same split
`test_flow_structure_cli.py`'s own module docstring describes for its
sibling command.
"""

from typer.testing import CliRunner

from tidescout.cli import app

runner = CliRunner()

# Rich wraps table cells to the terminal width, which would split reason
# text across lines and make substring assertions flaky rather than wrong.
WIDE = {"COLUMNS": "240"}


def _fake_payload():
    """One hour, four subs, deliberately built so the right answer cannot
    come from sorting the wrong direction: `wind`/`pressure`/`season` are
    all HIGH (0.90-0.94) and explain nothing about a 62, while `flow` is
    LOW (0.24, "slack") and is the actual reason the hour is weak -- the
    real numbers measured on winyah-bay 2026-08-16 15:00 (2026-08-26
    review, Important 6).
    """
    hour = {
        "time": "2026-08-16T15:00:00-04:00",
        "score": 62,
        "confidence": 1.0,
        "constrained_share": 1.0,
        "provisional": [],
        "excluded": [],
        "subs": [
            {"factor": "wind", "value": 0.94, "reason": "wind 8 kn — light"},
            {"factor": "pressure", "value": 0.92, "reason": "pressure -0.1 mb/3h — steady"},
            {"factor": "season", "value": 0.90, "reason": "August (month 8) seasonal modifier"},
            {"factor": "flow", "value": 0.24, "reason": "flow 0.04 m/s — slack"},
        ],
    }
    empty = {"hours": [], "features": {}}
    return {
        "slug": "winyah-bay",
        "day": "2026-08-16",
        "model_label": "best",
        "missing": [],
        "flow": {"clamped": False},
        "salinity": {"extrapolated": False, "fitted": True},
        "species": {
            "redfish": {"hours": [hour], "features": {}},
            "speckled_trout": empty,
            "southern_flounder": empty,
        },
    }


def _run(monkeypatch, *args):
    import tidescout.pipeline.payload as payload_module

    monkeypatch.setattr(payload_module, "build_payload", lambda *a, **k: _fake_payload())
    return runner.invoke(
        app, ["score", "winyah-bay", "2026-08-16", "--species", "redfish", *args], env=WIDE
    )


def test_score_cli_shows_the_limiting_factor_not_the_highest(monkeypatch):
    """Under `combine`'s weighted GEOMETRIC mean the LOWEST-value sub is what
    dragged the hour down, not the highest -- an earlier version of this
    command sorted the wrong direction and printed `wind`/`pressure`
    instead, which explain nothing about why this hour is a 62 (2026-08-26
    review, Important 6)."""
    result = _run(monkeypatch)
    assert result.exit_code == 0, result.exception or result.stdout
    assert "flow 0.24" in result.stdout
    assert "flow 0.04 m/s" in result.stdout and "slack" in result.stdout
    assert "wind 0.94" not in result.stdout
    assert "pressure 0.92" not in result.stdout


def test_score_cli_prints_the_reason_text_not_just_the_number(monkeypatch):
    """Spec section 8's "why is 3 PM an 82 always has a visible answer" and
    Step 4's "factor bars AND reasons" both need the words `score_factors`
    actually wrote, not just a bare factor/value pair."""
    result = _run(monkeypatch)
    assert result.exit_code == 0, result.exception or result.stdout
    assert "flow 0.04 m/s — slack" in result.stdout


def test_score_cli_validates_species_before_building_the_payload(monkeypatch):
    """2026-08-26 review, Minor 12: `--species` used to be checked only
    AFTER `build_payload` returned, so a typo cost a full real scoring run
    (~70s) before failing. `load_species()` -- the same source `build_
    payload` uses internally to populate `payload["species"]`'s keys -- is
    now checked first. `build_payload` is stubbed to raise if it is EVER
    called, so this fails loudly if the bad `--species` value ever reaches
    it rather than being caught before.
    """
    import tidescout.pipeline.payload as payload_module

    def _must_not_run(*a, **k):
        raise AssertionError("build_payload must not run for an invalid --species")

    monkeypatch.setattr(payload_module, "build_payload", _must_not_run)
    result = runner.invoke(
        app,
        ["score", "winyah-bay", "2026-08-16", "--species", "nonexistent-species"],
        env=WIDE,
    )
    # A BadParameter is a click UsageError (exit code 2) with the message in
    # the output; an uncaught AssertionError from the stub above (meaning
    # build_payload DID run) would instead exit 1 with a traceback -- these
    # two failure modes are deliberately distinguishable.
    assert result.exit_code == 2, (result.exit_code, result.exception, result.stdout)
    assert "species must be one of" in result.output
