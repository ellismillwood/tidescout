from datetime import UTC, datetime, timedelta

import pytest

from tidescout.errors import SourceUnavailable
from tidescout.sources.cache import Cache


class Clock:
    def __init__(self):
        self.now = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


def test_fetches_once_within_ttl(tmp_path):
    clock = Clock()
    cache = Cache(tmp_path / "c.sqlite", now_fn=clock)
    calls = []

    def fetch():
        calls.append(1)
        return {"v": 42}

    first = cache.get_or_fetch("noaa", "k1", timedelta(hours=1), fetch)
    clock.now += timedelta(minutes=30)
    second = cache.get_or_fetch("noaa", "k1", timedelta(hours=1), fetch)
    assert first.payload == {"v": 42} and first.fresh
    assert second.payload == {"v": 42} and second.fresh
    assert len(calls) == 1


def test_refetches_after_ttl(tmp_path):
    clock = Clock()
    cache = Cache(tmp_path / "c.sqlite", now_fn=clock)
    values = iter([{"v": 1}, {"v": 2}])
    cache.get_or_fetch("noaa", "k1", timedelta(hours=1), lambda: next(values))
    clock.now += timedelta(hours=2)
    result = cache.get_or_fetch("noaa", "k1", timedelta(hours=1), lambda: next(values))
    assert result.payload == {"v": 2}


def test_stale_fallback_on_fetch_failure(tmp_path):
    clock = Clock()
    cache = Cache(tmp_path / "c.sqlite", now_fn=clock)
    cache.get_or_fetch("noaa", "k1", timedelta(hours=1), lambda: {"v": 1})
    clock.now += timedelta(hours=5)

    def boom():
        raise ConnectionError("down")

    result = cache.get_or_fetch("noaa", "k1", timedelta(hours=1), boom)
    assert result.payload == {"v": 1}
    assert not result.fresh


def test_raises_when_no_fallback(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")

    def boom():
        raise ConnectionError("down")

    with pytest.raises(SourceUnavailable):
        cache.get_or_fetch("noaa", "nope", timedelta(hours=1), boom)


def test_ttl_none_never_refetches(tmp_path):
    clock = Clock()
    cache = Cache(tmp_path / "c.sqlite", now_fn=clock)
    calls = []
    cache.get_or_fetch("astro", "k", None, lambda: calls.append(1) or {"v": 1})
    clock.now += timedelta(days=365)
    cache.get_or_fetch("astro", "k", None, lambda: calls.append(1) or {"v": 2})
    assert len(calls) == 1
