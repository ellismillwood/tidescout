"""The day endpoint: cache hits, misses, staleness, and honest degradation."""

from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tidescout.api import store
from tidescout.api.app import create_app


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
