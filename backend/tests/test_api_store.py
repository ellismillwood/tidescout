"""The payload cache: a directory of gzipped JSON, not a database."""

import gzip
from datetime import UTC, date, datetime, timedelta

import pytest

from tidescout.api import store


def _payload(generated_at: datetime) -> dict:
    return {"slug": "winyah-bay", "freshness": {"generated_at": generated_at.isoformat()}}


def test_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    p = _payload(datetime(2026, 9, 3, 3, 0, tzinfo=UTC))
    store.write_payload("winyah-bay", date(2026, 9, 3), "best", p)
    assert store.read_payload("winyah-bay", date(2026, 9, 3), "best") == p


def test_a_miss_is_none_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert store.read_payload("winyah-bay", date(2026, 9, 3), "best") is None


def test_a_failed_write_creates_no_half_payload(tmp_path, monkeypatch):
    """A build killed mid-write must not leave a truncated file to be served
    forever after (spec §3.2). The write goes to a temp file and is renamed, so
    a failure during serialisation leaves the directory as it was.

    This is only half of `write_payload`'s atomicity claim -- "a reader
    either sees the previous payload or the new one, never a partial write."
    This test proves the "nothing new is created" half. It says nothing
    about whether a failed write can destroy an *existing* cached payload;
    see `test_a_failed_write_leaves_a_prior_payload_intact` for that half.
    """
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        store.write_payload("winyah-bay", date(2026, 9, 3), "best", {"bad": Unserialisable()})

    target = store.payload_path("winyah-bay", date(2026, 9, 3), "best")
    assert not target.exists()
    # and no temp litter left behind either
    assert list(target.parent.glob("*")) == []


def test_a_failed_write_leaves_a_prior_payload_intact(tmp_path, monkeypatch):
    """The other half of `write_payload`'s atomicity claim: a failed write
    must not destroy a payload that was already cached at that key. Write a
    good payload, then simulate a crash at the rename step itself -- a full
    disk, a killed process -- on a second, otherwise well-formed write to the
    SAME key, and confirm the original survives untouched.

    Sanity check against a deliberately non-atomic implementation (a single
    `target.write_bytes(gzip.compress(json.dumps(payload).encode()))`, no
    temp file, no rename): it never calls `os.replace`, so this fault
    injection has nothing to intercept -- the second write proceeds straight
    through and silently overwrites the original with the new payload. This
    test would then fail on both counts: no `OSError` propagates, and the
    payload read back afterward is the new one, not the original. That
    failure is the point -- it is exactly the corruption the temp-file-plus-
    rename design exists to prevent, and the earlier failed-serialisation
    test cannot detect it because that failure happens entirely in memory,
    before any implementation -- atomic or not -- ever touches a byte on
    disk.
    """
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    original = _payload(datetime(2026, 9, 3, 3, 0, tzinfo=UTC))
    store.write_payload("winyah-bay", date(2026, 9, 3), "best", original)

    def _boom(*args, **kwargs):
        raise OSError("simulated crash at the rename step")

    monkeypatch.setattr("os.replace", _boom)

    newer = _payload(datetime(2026, 9, 3, 9, 0, tzinfo=UTC))
    with pytest.raises(OSError):
        store.write_payload("winyah-bay", date(2026, 9, 3), "best", newer)

    assert store.read_payload("winyah-bay", date(2026, 9, 3), "best") == original


def test_a_truncated_file_reads_as_a_miss_not_a_crash(tmp_path, monkeypatch):
    """Defence in depth against a file truncated by something other than our
    own writer -- a full disk, a killed process mid-rename on a hostile
    filesystem. A corrupt cache entry must degrade to "rebuild it", never to a
    500 on every future request for that date."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    target = store.payload_path("winyah-bay", date(2026, 9, 3), "best")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(b'{"slug": "winyah'))
    assert store.read_payload("winyah-bay", date(2026, 9, 3), "best") is None


def test_staleness_applies_to_today_and_the_future_but_never_to_the_past():
    """Spec §3.1. A past date is scored from ERA5 reanalysis and USGS daily
    means, neither of which changes after the fact -- rebuilding it would burn
    70 s to produce the same numbers. Today's forecast genuinely moves.

    Both halves asserted: a rule that only ever returned False would pass a
    one-sided test, and so would one that staled everything.
    """
    now = datetime(2026, 9, 3, 18, 0, tzinfo=UTC)
    fresh = _payload(now - timedelta(hours=1))
    old = _payload(now - timedelta(hours=store.STALE_AFTER_H + 1))

    assert store.is_stale(old, date(2026, 9, 3), now) is True
    assert store.is_stale(old, date(2026, 9, 5), now) is True
    assert store.is_stale(old, date(2026, 8, 1), now) is False
    assert store.is_stale(fresh, date(2026, 9, 3), now) is False


def test_a_payload_with_no_generated_at_is_treated_as_stale():
    """Missing provenance is not evidence of freshness."""
    now = datetime(2026, 9, 3, 18, 0, tzinfo=UTC)
    assert store.is_stale({}, date(2026, 9, 3), now) is True


def test_a_naive_generated_at_is_treated_as_stale_not_a_crash():
    """`datetime.fromisoformat` parses an offset-less string into a NAIVE
    datetime instead of raising -- it does not protect us here. `now` is
    always aware (DTZ), so subtracting a naive `at` from it would raise
    TypeError rather than degrade gracefully, breaking `is_stale`'s own
    promise to treat malformed provenance as stale rather than crash the
    caller.
    """
    now = datetime(2026, 9, 3, 18, 0, tzinfo=UTC)
    naive = {"slug": "winyah-bay", "freshness": {"generated_at": "2026-09-03T17:59:00"}}
    assert store.is_stale(naive, date(2026, 9, 3), now) is True
