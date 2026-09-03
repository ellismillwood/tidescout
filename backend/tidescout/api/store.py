"""The payload cache: `data/<slug>/payloads/<date>-<model>.json.gz`.

A directory, not a database (spec §3). Cheap to inspect, cheap to delete,
survives restarts, no schema to migrate. A single-user local tool earns
nothing more.
"""

import gzip
import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from tidescout.paths import DATA_DIR

# Spec §3.1: a judgement call, not a measurement. Short enough that an
# afternoon check reflects the morning forecast update, long enough to avoid
# constant rebuilding. Meant to be moved.
STALE_AFTER_H = 6


def payload_path(slug: str, day: date, model: str) -> Path:
    # DATA_DIR / slug directly, NOT `paths.fishery_data_dir`, which mkdirs its
    # argument -- callers validate the slug first, and this stays a pure path
    # computation so tests can monkeypatch DATA_DIR.
    base = DATA_DIR / slug / "payloads"
    target = base / f"{day.isoformat()}-{model}.json.gz"
    # BACKSTOP, not the gate. `model` and `slug` both arrive from the URL, and
    # `write_payload` mkdirs `target.parent`, so an unchecked `..` chain would
    # create directories and write files OUTSIDE `data/` -- and a later read
    # would then serve any gzipped JSON on disk as a 200. The route validates
    # `model` against `weather.WEATHER_MODELS` and `slug` against
    # `fishery_slugs()` BEFORE calling in here; this refuses to hand back an
    # escaping path even if some future caller forgets to.
    root = DATA_DIR.resolve()
    resolved = target.resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_relative_to(root):
        raise ValueError(f"payload path escapes the cache: slug={slug!r} model={model!r}")
    return target


def write_payload(slug: str, day: date, model: str, payload: dict) -> Path:
    """Serialise to a temp file in the target directory, then rename.

    `os.replace` is atomic within a filesystem, so a reader either sees the
    previous payload or the new one, never a partial write. Serialising BEFORE
    the rename means a payload that cannot be encoded leaves nothing behind --
    the temp file is removed and the target is untouched.
    """
    target = payload_path(slug, day, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def read_payload_gz(slug: str, day: date, model: str) -> tuple[bytes, dict] | None:
    """The stored gzip bytes AND the parsed payload, from ONE read of the file.

    The day route serves the bytes VERBATIM -- spec §5 sizes the frontend
    against 1.67 MB gzipped, and gunzipping only to re-serialise 24.59 MB of
    JSON per request would put the uncompressed figure on the wire instead --
    but it still needs `freshness.generated_at` parsed out to decide staleness.
    Returning both from one read keeps the two in step and avoids a second
    24 MB trip through the filesystem.

    None for absent OR unreadable: a corrupt entry degrades to "rebuild it",
    never to a 500 on every future request for that date.
    """
    path = payload_path(slug, day, model)
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        return raw, json.loads(gzip.decompress(raw))
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
        return None


def read_payload(slug: str, day: date, model: str) -> dict | None:
    """The cached payload, or None if absent OR unreadable."""
    got = read_payload_gz(slug, day, model)
    return None if got is None else got[1]


def is_stale(payload: dict, day: date, now: datetime) -> bool:
    """Spec §3.1. Past dates never go stale -- ERA5 reanalysis and USGS daily
    means do not change after the fact, so rebuilding one burns 70 s to produce
    the same numbers.
    """
    if day < now.date():
        return False
    generated = (payload.get("freshness") or {}).get("generated_at")
    if not generated:
        return True  # missing provenance is not evidence of freshness
    try:
        at = datetime.fromisoformat(generated)
    except ValueError:
        return True
    if at.tzinfo is None:
        # `fromisoformat` happily parses an offset-less string into a naive
        # datetime instead of raising -- it does not protect us here. `now`
        # is always aware (DTZ), so `now - at` on a naive `at` raises
        # TypeError rather than degrading gracefully. Treat naive provenance
        # the same as missing or unparseable: not evidence of freshness.
        return True
    return (now - at) > timedelta(hours=STALE_AFTER_H)
