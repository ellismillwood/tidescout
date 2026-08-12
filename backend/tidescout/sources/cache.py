import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tidescout.errors import SourceUnavailable


@dataclass
class Cached:
    payload: Any
    fetched_at: datetime
    fresh: bool


class Cache:
    def __init__(self, db_path: Path, now_fn: Callable[[], datetime] | None = None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  source TEXT NOT NULL, key TEXT NOT NULL, fetched_at TEXT NOT NULL,"
            "  payload TEXT NOT NULL, PRIMARY KEY (source, key))"
        )
        self._now = now_fn or (lambda: datetime.now(UTC))

    def _read(self, source: str, key: str) -> tuple[Any, datetime] | None:
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM cache WHERE source = ? AND key = ?", (source, key)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), datetime.fromisoformat(row[1])

    def get_or_fetch(
        self, source: str, key: str, ttl: timedelta | None, fetch: Callable[[], Any]
    ) -> Cached:
        existing = self._read(source, key)
        now = self._now()
        if existing is not None:
            payload, fetched_at = existing
            if ttl is None or now - fetched_at <= ttl:
                return Cached(payload, fetched_at, fresh=True)
        try:
            payload = fetch()
        except Exception as exc:
            if existing is not None:
                stale_payload, fetched_at = existing
                return Cached(stale_payload, fetched_at, fresh=False)
            raise SourceUnavailable(source, str(exc)) from exc
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (source, key, fetched_at, payload) VALUES (?, ?, ?, ?)",
            (source, key, now.isoformat(), json.dumps(payload)),
        )
        self._conn.commit()
        return Cached(payload, now, fresh=True)


_default: Cache | None = None


def default_cache() -> Cache:
    global _default
    if _default is None:
        db = Path(__file__).resolve().parents[3] / "data" / "cache.sqlite"
        _default = Cache(db)
    return _default
