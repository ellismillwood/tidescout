"""Runs the 70-second `build_payload` off the request path, once per key.

Spec §3.2. Three properties this exists to guarantee:
  1. Ten callers for one key start ONE build; the rest join it.
  2. The build never runs on the event loop -- a 70 s CPU-bound numpy call
     there would freeze every other request.
  3. A failed build writes no payload, and does not wedge the key.
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime

from tidescout.api import store

Key = tuple[str, str, str]


@dataclass(frozen=True)
class BuildState:
    status: str  # "building" | "ready" | "failed"
    started_at: datetime
    error: str | None = None


def default_build_fn(slug: str, day: date, model: str) -> dict:
    """The real build. Imported lazily and given its OWN `Cache`.

    `sources.cache.default_cache()` is a module-level singleton wrapping one
    `sqlite3` connection, and sqlite3 raises ProgrammingError when a connection
    crosses threads (verified 2026-09-03). This runs in a worker thread, so it
    must build its own -- passing the singleton in fails on the first request.
    """
    from tidescout.paths import DATA_DIR
    from tidescout.pipeline.payload import build_payload
    from tidescout.sources.cache import Cache

    return build_payload(slug, day, model, Cache(DATA_DIR / "cache.sqlite"))


class BuildCoordinator:
    def __init__(self, build_fn=None, executor: ThreadPoolExecutor | None = None):
        self._build = build_fn or default_build_fn
        # max_workers=2: builds are numpy-heavy and memory-hungry (the flow
        # library is 9.1 GB on disk and states are held in RAM while scoring).
        # Running many at once would thrash rather than finish sooner.
        self._pool = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="build")
        self._lock = threading.Lock()
        self._states: dict[Key, BuildState] = {}
        self._inflight: dict[Key, Future] = {}

    def state(self, slug: str, day: date, model: str) -> BuildState | None:
        with self._lock:
            return self._states.get((slug, day.isoformat(), model))

    def ensure(self, slug: str, day: date, model: str) -> BuildState:
        """Start a build for this key unless one is already running.

        The lock covers the check AND the submit, so two threads arriving
        together cannot both see "not running" and both submit.
        """
        key = (slug, day.isoformat(), model)
        with self._lock:
            existing = self._states.get(key)
            if existing is not None and existing.status == "building":
                return existing
            state = BuildState("building", datetime.now(UTC))
            self._states[key] = state
            fut = self._pool.submit(self._run, slug, day, model, key)
            self._inflight[key] = fut
            return state

    def _run(self, slug: str, day: date, model: str, key: Key) -> None:
        started = datetime.now(UTC)
        try:
            payload = self._build(slug, day, model)
            store.write_payload(slug, day, model, payload)
        except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised to the future
            with self._lock:
                self._states[key] = BuildState("failed", started, f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            self._states[key] = BuildState("ready", started)

    def wait_all(self, timeout: float | None = None) -> None:
        """Test helper: block until every in-flight build has settled."""
        with self._lock:
            futures = list(self._inflight.values())
        for fut in futures:
            fut.result(timeout=timeout)
        with self._lock:
            self._inflight = {k: f for k, f in self._inflight.items() if not f.done()}
