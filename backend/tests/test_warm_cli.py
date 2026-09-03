"""The warming job: builds the common dates, skips what is already fresh."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from typer.testing import CliRunner

from tidescout import cli, config
from tidescout.api import store

runner = CliRunner()


def _today() -> date:
    """The FISHERY's calendar day, which is what `warm` iterates from.

    Not `datetime.now(UTC).date()`: between 20:00 and 23:59 Eastern the two
    disagree, and a test written against UTC would fail for four hours a day
    for the wrong reason.
    """
    return config.fishery_now("winyah-bay").date()


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
    today = _today()
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
    today = _today()
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
        if day == _today():
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


def test_warm_starts_from_the_fishery_local_date_not_utc(tmp_path, monkeypatch):
    """The 20:00-Eastern trap, pinned.

    Winyah Bay is `America/New_York`. At 20:30 Eastern, UTC is already the NEXT
    day, so a UTC "today" makes `warm --days 7` skip today entirely -- and this
    command's own docstring recommends running it overnight. The next morning
    the user hits a 202 and a 70-second wait for the one date the warming
    subsystem exists to have ready.

    Both halves: the first date built is the Eastern day, AND the UTC day is
    still built after it rather than being dropped off the end.
    """
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    eastern = ZoneInfo("America/New_York")
    frozen = datetime(2026, 9, 4, 0, 30, tzinfo=UTC).astimezone(eastern)
    assert frozen.date() == date(2026, 9, 3), "20:30 ET"
    assert datetime(2026, 9, 4, 0, 30, tzinfo=UTC).date() == date(2026, 9, 4), "the trap"

    monkeypatch.setattr(config, "fishery_now", lambda slug: frozen)
    built = []
    monkeypatch.setattr(
        cli,
        "_warm_build",
        lambda s, d, m: (
            built.append(d),
            {"slug": s, "freshness": {"generated_at": datetime.now(UTC).isoformat()}},
        )[1],
    )

    result = runner.invoke(cli.app, ["warm", "winyah-bay", "--days", "2"])
    assert result.exit_code == 0, result.output
    assert built == [date(2026, 9, 3), date(2026, 9, 4)]


def test_warm_clamps_days_to_the_forecast_horizon(tmp_path, monkeypatch):
    """`--days 30` would spend ~70 s each on 13 dates that `/day` then rejects
    with a 422. Clamped, and the clamp is announced rather than silent."""
    from tidescout.sources.weather import FORECAST_HORIZON_DAYS

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

    result = runner.invoke(cli.app, ["warm", "winyah-bay", "--days", "30"])
    assert result.exit_code == 0, result.output
    assert len(built) == FORECAST_HORIZON_DAYS + 1, built
    assert built[-1] == _today() + timedelta(days=FORECAST_HORIZON_DAYS)
    assert "horizon" in " ".join(result.output.split())


def test_a_days_value_inside_the_horizon_is_not_clamped(tmp_path, monkeypatch):
    """The other half: the clamp must not shorten an ordinary request."""
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
    result = runner.invoke(cli.app, ["warm", "winyah-bay", "--days", "7"])
    assert result.exit_code == 0, result.output
    assert len(built) == 7
    assert "horizon" not in " ".join(result.output.split())
