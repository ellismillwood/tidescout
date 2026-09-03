"""The warming job: builds the common dates, skips what is already fresh."""

from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from tidescout import cli
from tidescout.api import store

runner = CliRunner()


def test_warm_skips_dates_that_are_already_fresh(tmp_path, monkeypatch):
    """Warming is ~70 s per date. Rebuilding a payload that is already fresh
    would turn an 8-minute nightly job into a pointless one.

    Both halves: the fresh date is skipped AND the others are still built.
    """
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    built = []

    def fake_build(slug, day, model):
        built.append(day)
        return {"slug": slug, "freshness": {"generated_at": datetime.now(UTC).isoformat()}}

    monkeypatch.setattr(cli, "_warm_build", fake_build)
    today = datetime.now(UTC).date()
    store.write_payload(
        "winyah-bay", today, "best",
        {"slug": "winyah-bay", "freshness": {"generated_at": datetime.now(UTC).isoformat()}},
    )

    result = runner.invoke(cli.app, ["warm", "winyah-bay", "--days", "3"])
    assert result.exit_code == 0, result.output
    assert today not in built
    assert built == [today + timedelta(days=1), today + timedelta(days=2)]


def test_warm_force_rebuilds_even_a_fresh_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    built = []
    monkeypatch.setattr(
        cli,
        "_warm_build",
        lambda s, d, m: (
            built.append(d),
            {"slug": s, "freshness": {"generated_at": datetime.now(UTC).isoformat()}},
        )[1],
    )
    today = datetime.now(UTC).date()
    store.write_payload(
        "winyah-bay", today, "best",
        {"slug": "winyah-bay", "freshness": {"generated_at": datetime.now(UTC).isoformat()}},
    )
    result = runner.invoke(cli.app, ["warm", "winyah-bay", "--days", "1", "--force"])
    assert result.exit_code == 0, result.output
    assert built == [today]


def test_warm_keeps_going_after_one_date_fails(tmp_path, monkeypatch):
    """A single dark source on one date must not abandon the remaining six --
    a nightly job that gives up on the first error is worse than none."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    built = []

    def flaky(slug, day, model):
        if day == datetime.now(UTC).date():
            raise RuntimeError("USGS timed out")
        built.append(day)
        return {"slug": slug, "freshness": {"generated_at": datetime.now(UTC).isoformat()}}

    monkeypatch.setattr(cli, "_warm_build", flaky)
    result = runner.invoke(cli.app, ["warm", "winyah-bay", "--days", "3"])
    assert result.exit_code == 1, "a failure must be visible in the exit code"
    assert len(built) == 2, built
    assert "USGS timed out" in result.output


def test_warm_rejects_an_unknown_fishery(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    result = runner.invoke(cli.app, ["warm", "no-such-bay", "--days", "1"])
    assert result.exit_code != 0
