"""The day endpoint: cache hits, misses, staleness, and honest degradation."""

import gzip
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tidescout.api import store
from tidescout.api.app import create_app
from tidescout.sources.weather import WEATHER_MODELS


class FakeCoordinator:
    def __init__(self):
        self.ensured = []
        self.states = {}

    def ensure(self, slug, day, model):
        self.ensured.append((slug, day, model))
        from tidescout.api.builds import BuildState

        return BuildState("building", datetime.now(UTC))

    def state(self, slug, day, model):
        return self.states.get((slug, day.isoformat(), model))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    coord = FakeCoordinator()
    app = create_app(coordinator=coord)
    return TestClient(app), coord


def _write(day: date, generated_at: datetime, **extra):
    payload = {"slug": "winyah-bay", "day": day.isoformat(),
               "freshness": {"generated_at": generated_at.isoformat()}, **extra}
    store.write_payload("winyah-bay", day, "best", payload)
    return payload


def test_a_fresh_cache_hit_serves_200_and_starts_no_build(client):
    c, coord = client
    day = datetime.now(UTC).date() + timedelta(days=1)
    payload = _write(day, datetime.now(UTC))
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 200
    assert r.json() == payload
    assert coord.ensured == [], "a fresh hit must not trigger a build"


def test_a_miss_returns_202_and_starts_exactly_one_build(client):
    c, coord = client
    day = datetime.now(UTC).date()
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 202
    assert r.json()["status"] == "building"
    assert coord.ensured == [("winyah-bay", day, "best")]


def test_a_stale_hit_serves_immediately_AND_rebuilds(client):
    """Spec §3.1, both halves. Serving without rebuilding leaves the user on
    stale data forever; rebuilding before serving makes them wait 70 s for data
    already on disk. Each is a distinct bug, so each is asserted.
    """
    c, coord = client
    day = datetime.now(UTC).date()
    payload = _write(day, datetime.now(UTC) - timedelta(hours=store.STALE_AFTER_H + 1))
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 200
    assert r.json() == payload
    assert coord.ensured == [("winyah-bay", day, "best")]


def test_a_past_date_is_never_stale_and_never_rebuilt(client):
    c, coord = client
    day = datetime.now(UTC).date() - timedelta(days=30)
    _write(day, datetime.now(UTC) - timedelta(days=29))
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 200
    assert coord.ensured == []


def test_a_degraded_payload_is_served_as_200_with_its_disclosure_intact(client):
    """Spec §6, the rule that matters most. A dark sensor is DATA -- an API
    layer that turned `missing: ['weather']` into a 500, or stripped it, would
    undo the disclosure machinery five PRs went into building.
    """
    c, _ = client
    day = datetime.now(UTC).date() + timedelta(days=1)
    _write(day, datetime.now(UTC), missing=["weather"], confidence=0.79)
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 200
    assert r.json()["missing"] == ["weather"]
    assert r.json()["confidence"] == 0.79


def test_an_unknown_fishery_is_404(client):
    c, _ = client
    r = c.get(f"/api/fisheries/no-such-bay/day/{datetime.now(UTC).date().isoformat()}")
    assert r.status_code == 404


def test_a_known_but_unprocessed_fishery_is_409_naming_what_is_missing(client, monkeypatch):
    """Spec §2: 'one predicate, two callers' -- the same `readiness.readiness`
    backs both the `ready` flag in `/api/fisheries` and this 409, so the list
    and the error can never disagree. The repo's one fixture fishery is
    always fully processed on disk, so this branch is unreachable without a
    monkeypatch. `fake_readiness` still branches on slug (real not-ready vs.
    real unknown) rather than returning one canned result, so this test also
    asserts the 404 sibling below -- catching `_require_ready`'s two `if`
    branches being swapped, which a 409-only check would miss entirely.
    """
    from tidescout.api import readiness as readiness_module
    from tidescout.api.readiness import Readiness

    def fake_readiness(slug):
        if slug == "winyah-bay":
            return Readiness(False, ("flow library", "feature inventory"))
        return Readiness(False, ("unknown fishery",))

    monkeypatch.setattr(readiness_module, "readiness", fake_readiness)
    c, _ = client
    day = datetime.now(UTC).date() + timedelta(days=1)

    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 409
    assert "flow library" in r.json()["detail"]
    assert "feature inventory" in r.json()["detail"]

    r2 = c.get(f"/api/fisheries/no-such-bay/day/{day.isoformat()}")
    assert r2.status_code == 404


def test_a_far_future_date_is_422_and_says_the_range(client):
    c, _ = client
    day = datetime.now(UTC).date() + timedelta(days=40)
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 422
    assert "16" in r.json()["detail"], r.json()

def test_an_unparseable_date_is_422_not_500(client):
    c, _ = client
    r = c.get("/api/fisheries/winyah-bay/day/not-a-date")
    assert r.status_code == 422


def test_status_reports_a_failed_build_with_its_error(client):
    from tidescout.api.builds import BuildState

    c, coord = client
    day = datetime.now(UTC).date()
    coord.states[("winyah-bay", day.isoformat(), "best")] = BuildState(
        "failed", datetime.now(UTC), "RuntimeError: USGS timed out"
    )
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}/status")
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "USGS timed out" in r.json()["error"]


def test_status_reports_ready_for_a_cached_payload_without_any_build(client):
    c, _ = client
    day = datetime.now(UTC).date() + timedelta(days=1)
    _write(day, datetime.now(UTC))
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}/status")
    assert r.json()["status"] == "ready"
    assert r.json()["stale"] is False


@pytest.mark.parametrize(
    "evil",
    ["../../../pwned", "../../../../../escaped", "best/../../../../../x", "..", "gfs;rm"],
)
@pytest.mark.parametrize("suffix", ["", "/status"])
def test_a_traversing_model_is_rejected_before_it_reaches_the_filesystem(
    tmp_path, monkeypatch, evil, suffix
):
    """`?model=` becomes part of a FILENAME. `store.write_payload` mkdirs the
    target's parent, so a `..` chain creates directories and writes files
    OUTSIDE `data/`, and a second request then serves any gzipped JSON on disk
    as a 200. Nothing upstream blocks it -- `weather.fetch_weather` raises
    KeyError on an unknown model, but `dayloader.load_day`'s `attempt()`
    catches bare `Exception` and degrades to `missing: ['weather']`, so the
    build SUCCEEDS and writes.

    Three halves, because a 422 alone would not prove the fix: the status
    code, the fact that NO build was started (the rejection has to precede
    `coord.ensure`), and the fact that nothing appeared anywhere on disk. Both
    `/day` and `/status` take the parameter, so both are covered.
    """
    data_dir = tmp_path / "repo" / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(store, "DATA_DIR", data_dir)
    coord = FakeCoordinator()
    c = TestClient(create_app(coordinator=coord))
    day = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    before = sorted(tmp_path.rglob("*"))

    r = c.get(f"/api/fisheries/winyah-bay/day/{day}{suffix}", params={"model": evil})

    assert r.status_code == 422, r.text
    assert "best" in r.json()["detail"], "the 422 must name the valid models"
    assert coord.ensured == [], "rejection must precede any build"
    assert sorted(tmp_path.rglob("*")) == before, "nothing may be created on disk"


@pytest.mark.parametrize("model", sorted(WEATHER_MODELS))
def test_every_real_weather_model_is_still_accepted(client, model):
    """The other half of the traversal fix: a guard that rejected everything
    would pass a rejection-only test while breaking all six real models."""
    c, coord = client
    day = datetime.now(UTC).date() + timedelta(days=1)
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}", params={"model": model})
    assert r.status_code == 202, r.text
    assert coord.ensured == [("winyah-bay", day, model)]


def test_a_cache_hit_is_served_as_the_stored_gzip_bytes_not_re_serialised(client):
    """Spec §5 sizes the frontend against 1.67 MB gzipped, not 24.59 MB raw.

    Gunzipping the cache entry only to hand it to `JSONResponse` re-serialised
    it and put the uncompressed figure on the wire -- ~270 ms of CPU per
    request and 24.6 MB instead of 1.67 MB, silently invalidating the transfer
    numbers in §5. Three halves: the encoding is declared, the body on the wire
    really is the compressed file (its length equals the file's, not the
    decoded payload's), and the decoded content is byte-identical to what was
    stored -- the contract must be unchanged, only the encoding.
    """
    c, _ = client
    day = datetime.now(UTC).date() + timedelta(days=1)
    # Big enough to compress: the real payload is 24.59 MB raw / 1.67 MB
    # gzipped, and a 100-byte stub would gzip LARGER than it started.
    payload = _write(day, datetime.now(UTC), species={"redfish": [{"flow": 0.5}] * 2000})
    path = store.payload_path("winyah-bay", day, "best")

    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    assert r.headers["content-type"].startswith("application/json")
    assert int(r.headers["content-length"]) == path.stat().st_size
    assert int(r.headers["content-length"]) < len(r.content), "the wire form must be smaller"
    assert r.content == gzip.decompress(path.read_bytes())
    assert r.json() == payload


def test_a_degraded_payload_survives_the_gzip_passthrough(client):
    """Spec §6, re-asserted against the compressed path specifically. Serving
    stored bytes must not become an excuse to reshape the payload: a dark
    sensor is DATA and reaches the client exactly as written."""
    c, _ = client
    day = datetime.now(UTC).date() + timedelta(days=1)
    payload = _write(day, datetime.now(UTC), missing=["weather"], confidence=0.79)
    r = c.get(f"/api/fisheries/winyah-bay/day/{day.isoformat()}")
    assert r.headers["content-encoding"] == "gzip"
    assert r.json() == payload
    assert r.json()["missing"] == ["weather"]
    assert r.json()["confidence"] == 0.79


def test_the_horizon_and_staleness_are_measured_in_the_fishery_zone_not_utc(
    tmp_path, monkeypatch
):
    """20:30 Eastern on 2026-09-03 is already 2026-09-04 in UTC.

    Winyah Bay's own calendar day is the one the user is fishing, and the
    fishery config carries its timezone -- `sources.weather._today` exists to
    say so. Two gates move with the day boundary, and each is asserted against
    the date UTC would have gotten wrong:

    * the forecast horizon -- +16 days from the Eastern day is 2026-09-19, so
      the 19th is inside the range and the 20th is not; in UTC the 20th would
      wrongly be allowed;
    * staleness -- 2026-09-03 is TODAY in the fishery, so an old payload for it
      is stale and gets a background rebuild; in UTC it is yesterday, and
      `is_stale`'s "past dates never go stale" rule would silently pin the user
      to a stale forecast for the day he is actually fishing.
    """
    from zoneinfo import ZoneInfo

    from tidescout.api import app as app_module

    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    eastern = ZoneInfo("America/New_York")
    frozen = datetime(2026, 9, 4, 0, 30, tzinfo=UTC).astimezone(eastern)
    assert (frozen.date(), datetime(2026, 9, 4, 0, 30, tzinfo=UTC).date()) == (
        date(2026, 9, 3),
        date(2026, 9, 4),
    ), "the two calendars disagree here -- that is the point of this test"
    monkeypatch.setattr(app_module, "fishery_now", lambda slug: frozen)

    coord = FakeCoordinator()
    c = TestClient(create_app(coordinator=coord))

    edge = c.get("/api/fisheries/winyah-bay/day/2026-09-19")
    assert edge.status_code == 202, "today+16 in the fishery zone is inside the horizon"
    beyond = c.get("/api/fisheries/winyah-bay/day/2026-09-20")
    assert beyond.status_code == 422, beyond.text

    coord.ensured.clear()
    _write(date(2026, 9, 3), frozen - timedelta(hours=store.STALE_AFTER_H + 1))
    hit = c.get("/api/fisheries/winyah-bay/day/2026-09-03")
    assert hit.status_code == 200
    assert coord.ensured == [("winyah-bay", date(2026, 9, 3), "best")], (
        "the fishery's today is stale and must be rebuilt in the background"
    )
