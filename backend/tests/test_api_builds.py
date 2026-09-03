"""Background builds: one per key, off the event loop, failures not cached."""

import threading
from datetime import date

from tidescout.api import builds, store


def test_concurrent_callers_start_exactly_one_build(tmp_path, monkeypatch):
    """A double-clicked date picker, or ten status polls, must not start ten
    70-second builds. The second caller JOINS the first.

    Asserts the pair: one build ran, AND every caller got a state back. A
    coordinator that dropped the extra callers on the floor would also see
    `calls == 1`.
    """
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    calls = []
    gate = threading.Event()

    def slow_build(slug, day, model):
        calls.append((slug, day, model))
        gate.wait(timeout=5)
        return {"slug": slug, "freshness": {"generated_at": "2026-09-03T03:00:00+00:00"}}

    c = builds.BuildCoordinator(build_fn=slow_build)
    states = [c.ensure("winyah-bay", date(2026, 9, 3), "best") for _ in range(10)]
    assert all(s.status == "building" for s in states), states
    gate.set()
    c.wait_all(timeout=10)
    assert len(calls) == 1, calls


def test_a_completed_build_is_written_to_the_cache_and_reports_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    payload = {"slug": "winyah-bay", "freshness": {"generated_at": "2026-09-03T03:00:00+00:00"}}
    c = builds.BuildCoordinator(build_fn=lambda s, d, m: payload)

    c.ensure("winyah-bay", date(2026, 9, 3), "best")
    c.wait_all(timeout=10)

    assert c.state("winyah-bay", date(2026, 9, 3), "best").status == "ready"
    assert store.read_payload("winyah-bay", date(2026, 9, 3), "best") == payload


def test_a_failed_build_writes_no_payload_and_reports_the_error(tmp_path, monkeypatch):
    """Spec §6: a transient source outage must not poison the cache with a
    permanently-degraded payload. Both halves: the error is reported AND
    nothing was written."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    def boom(slug, day, model):
        raise RuntimeError("USGS timed out")

    c = builds.BuildCoordinator(build_fn=boom)
    c.ensure("winyah-bay", date(2026, 9, 3), "best")
    c.wait_all(timeout=10)

    st = c.state("winyah-bay", date(2026, 9, 3), "best")
    assert st.status == "failed"
    assert "USGS timed out" in st.error
    assert store.read_payload("winyah-bay", date(2026, 9, 3), "best") is None


def test_a_failed_key_can_be_retried(tmp_path, monkeypatch):
    """A failure must not wedge the key forever -- the next request retries."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    attempts = []

    def flaky(slug, day, model):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")
        return {"slug": slug, "freshness": {"generated_at": "2026-09-03T03:00:00+00:00"}}

    c = builds.BuildCoordinator(build_fn=flaky)
    c.ensure("winyah-bay", date(2026, 9, 3), "best")
    c.wait_all(timeout=10)
    assert c.state("winyah-bay", date(2026, 9, 3), "best").status == "failed"

    c.ensure("winyah-bay", date(2026, 9, 3), "best")
    c.wait_all(timeout=10)
    assert c.state("winyah-bay", date(2026, 9, 3), "best").status == "ready"
    assert len(attempts) == 2


def test_different_keys_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    seen = []
    c = builds.BuildCoordinator(
        build_fn=lambda s, d, m: (
            seen.append((s, d, m)),
            {"slug": s, "freshness": {"generated_at": "2026-09-03T03:00:00+00:00"}},
        )[1]
    )
    c.ensure("winyah-bay", date(2026, 9, 3), "best")
    c.ensure("winyah-bay", date(2026, 9, 4), "best")
    c.ensure("winyah-bay", date(2026, 9, 3), "ecmwf")
    c.wait_all(timeout=10)
    assert len(seen) == 3
