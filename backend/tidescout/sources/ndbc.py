"""NDBC buoy water quality: an ACCUMULATING store, not a fetch-on-demand cache.

Station WYSS1 ("Winyah Bay Surface"), operated by the National Estuarine
Research Reserve System at North Inlet-Winyah Bay, 33.309N -79.289W. It is
the first salinity observation Phase 2 has found INSIDE the model domain --
Task 5's two USGS 00480 sites both sit outside it and snap to the same
31.57 km cell (see `pipeline/salinity_fit.py`'s module docstring). WYSS1
projects to UTM (659284.5, 3686849.9), 9.5 m from the nearest in-domain
library cell, at along-estuary distance 15.02 km -- independently
re-verified here against the real distance field (see the station's entry
in `fisheries/winyah-bay.yaml`).

WHY THIS IS A STORE, NOT A CACHE (the load-bearing design decision)
---------------------------------------------------------------------
`https://www.ndbc.noaa.gov/data/realtime2/<station>.ocean` is a ROLLING
WINDOW. Measured live 2026-08-23: the file held 4,235 rows spanning
2026-07-09 00:00 through 2026-08-23 14:30 UTC -- 45.6 days, more than this
task's dispatch estimated (~13 days/~1,440 rows), but still finite and still
sliding: the oldest rows drop off the file as new ones are appended at the
top. `sources/noaa.py` and `sources/coops_water.py`'s `Cache.get_or_fetch`
is a TTL cache: the right tool when the upstream API can answer any
historical query on demand and a local copy just needs refreshing. NDBC
cannot answer a query for data outside its current window at all -- that
data is not delayed, it is GONE. A `Cache`-only integration would silently
forget every observation older than the window, in a design explicitly
asked to "store all data from the buoy".

So this module owns a small persistent SQLite store (`NdbcStore`, one file
per fishery at `data/<slug>/ndbc.sqlite` via `default_store`) that every
fetch APPENDS into, deduplicated by (station, timestamp). It does NOT go
through `sources.cache.Cache`: that class's stale-on-failure fallback exists
to paper over an upstream API that can be re-asked later for the same
answer; here the accumulated store already IS the durable copy, so a cache
in front of it would be redundant complexity with no behaviour it does not
already have. A failed fetch instead raises `SourceUnavailable` directly
(`fetch_and_store`) -- Task 4's established convention: real failures
propagate so `dayloader.load_day`'s `attempt()` records them in `missing`,
rather than a silent no-op indistinguishable from "the buoy had nothing new".

WHY SQLITE OVER JSONL/CSV
--------------------------
The repo already models two data shapes -- `data/cache.sqlite` (TTL request
cache, keyed, rows meant to expire) and `.npy`/`.geojson` (static grids and
features, written once per build, never merged). This store is neither: it
is unboundedly growing, keyed, and must survive a crash mid-write without
losing what was already there. SQLite gives all three for free and is
already a proven dependency here (`sources/cache.py`): `PRIMARY KEY
(station, ts)` makes de-duplication a single `INSERT OR REPLACE` rather than
hand-rolled merge logic, the connection used as a context manager commits an
entire batch atomically and rolls back the WHOLE batch on any exception (see
`NdbcStore.append`), and a windowed read uses an indexed query instead of
parsing the entire history into memory. A flat file would need its own
write-to-temp-then-rename discipline for the same crash safety, and would
still need to load-and-merge the full history in Python to deduplicate --
fine at today's ~4,200 rows, but this store is designed to keep growing for
years.

MM -> None, not 0.0. CLCON and EH are "MM" on every real row this station
has sent so far (no chlorophyll/ORP sensor fitted) -- a 0.0 chlorophyll or
ORP reading that never happened must not reach a future bite-score or
calibration input, the same failure shape Task 4 caught for CO-OPS's blank
salinity reading (see `coops_water.py`).

COND (mS/cm, conductivity) is kept as its own column and is never read as
SAL (psu, salinity) anywhere in this module -- Task 5's constraints name
that conflation explicitly as something to avoid.

TWO READ PATHS OUT, NOTHING WIRED IN
--------------------------------------
`NdbcStore.salinity_series` is what Task 5's calibration wants: (timestamp,
salinity) pairs at this station's one known along-estuary distance.
`NdbcStore.read` / `.latest` are what Phase 3's bite score wants: full rows
(temperature, dissolved oxygen, turbidity, salinity together). Neither is
called from `pipeline/salinity_fit.py` or any scoring path by this module --
Phase 3 does not exist yet and scoring belongs there. This module stores and
exposes; it stops at that seam on purpose.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tidescout.errors import SourceUnavailable
from tidescout.paths import fishery_data_dir

__all__ = [
    "NDBC_URL",
    "NdbcStore",
    "Observation",
    "default_store",
    "fetch_and_store",
    "parse_ocean",
]

NDBC_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.ocean"

# The `.ocean` product's own column order, verbatim from its header:
# YY MM DD hh mm DEPTH OTMP COND SAL O2% O2PPM CLCON TURB PH EH
_N_FIELDS = 15


@dataclass(frozen=True)
class Observation:
    """One `.ocean` row. `None` means NDBC sent "MM" (missing) for that field."""

    ts: datetime  # UTC, tz-aware
    depth_m: float | None
    water_temp_c: float | None
    cond_ms_cm: float | None  # conductivity, mS/cm -- NOT salinity, see module docstring
    salinity_psu: float | None
    o2_pct: float | None
    o2_ppm: float | None
    chlorophyll_ug_l: float | None
    turbidity_ftu: float | None
    ph: float | None
    eh_mv: float | None


def _val(token: str) -> float | None:
    return None if token == "MM" else float(token)


def parse_ocean(text: str) -> list[Observation]:
    """Parse the `.ocean` product's whitespace-separated rows.

    Lines starting with "#" (the two-row header) and blank lines are
    skipped. Not defensive beyond that documented shape -- a data row with
    the wrong field count raises `ValueError` rather than being silently
    dropped, the same choice `coops_water.py` makes for its own response
    shape: a format change upstream should be loud, not absorbed as "no
    reading today".
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != _N_FIELDS:
            raise ValueError(f"expected {_N_FIELDS} fields, got {len(fields)}: {line!r}")
        yy, mo, dy, hh, mi, depth, otmp, cond, sal, o2pct, o2ppm, clcon, turb, ph, eh = fields
        ts = datetime(int(yy), int(mo), int(dy), int(hh), int(mi), tzinfo=UTC)
        out.append(
            Observation(
                ts=ts,
                depth_m=_val(depth),
                water_temp_c=_val(otmp),
                cond_ms_cm=_val(cond),
                salinity_psu=_val(sal),
                o2_pct=_val(o2pct),
                o2_ppm=_val(o2ppm),
                chlorophyll_ug_l=_val(clcon),
                turbidity_ftu=_val(turb),
                ph=_val(ph),
                eh_mv=_val(eh),
            )
        )
    return out


_COLUMNS = (
    "ts", "depth_m", "water_temp_c", "cond_ms_cm", "salinity_psu",
    "o2_pct", "o2_ppm", "chlorophyll_ug_l", "turbidity_ftu", "ph", "eh_mv",
)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


class NdbcStore:
    """Accumulating, deduplicated observation history for NDBC stations.

    Every write happens inside one `sqlite3` transaction (the connection
    used as a context manager in `append`): it commits the whole batch on
    success and rolls back the whole batch on any exception, so a fetch that
    dies partway through a batch can never leave the table with only some of
    that batch's rows, and can never touch rows a previous call already
    committed. That is the crash-safety property the task asked for --
    write-then-rename's equivalent, using a mechanism already proven in this
    codebase (`sources/cache.py`) rather than a second one.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS observations ("
            " station TEXT NOT NULL,"
            " ts TEXT NOT NULL,"
            " depth_m REAL, water_temp_c REAL, cond_ms_cm REAL, salinity_psu REAL,"
            " o2_pct REAL, o2_ppm REAL, chlorophyll_ug_l REAL, turbidity_ftu REAL,"
            " ph REAL, eh_mv REAL,"
            " PRIMARY KEY (station, ts))"
        )
        self._conn.commit()

    def append(self, station: str, rows: Sequence[Observation]) -> int:
        """Insert `rows`, de-duplicated by (station, timestamp).

        Returns the count of rows that were actually NEW: re-fetching a
        window that overlaps what is already stored returns 0 for the
        overlap, not `len(rows)` -- see `test_append_and_dedupe_across_
        overlapping_fetches` for the proof this matters.
        """
        before = self.count(station)
        with self._conn:
            for r in rows:
                self._conn.execute(
                    "INSERT OR REPLACE INTO observations "
                    "(station, ts, depth_m, water_temp_c, cond_ms_cm, salinity_psu, "
                    " o2_pct, o2_ppm, chlorophyll_ug_l, turbidity_ftu, ph, eh_mv) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        station, r.ts.astimezone(UTC).isoformat(),
                        r.depth_m, r.water_temp_c, r.cond_ms_cm, r.salinity_psu,
                        r.o2_pct, r.o2_ppm, r.chlorophyll_ug_l, r.turbidity_ftu,
                        r.ph, r.eh_mv,
                    ),
                )
        return self.count(station) - before

    def count(self, station: str) -> int:
        (n,) = self._conn.execute(
            "SELECT COUNT(*) FROM observations WHERE station = ?", (station,)
        ).fetchone()
        return n

    def time_span(self, station: str) -> tuple[datetime, datetime] | None:
        """(earliest, latest) stored timestamp for `station`, or `None` if empty."""
        row = self._conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM observations WHERE station = ?", (station,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return _parse_ts(row[0]), _parse_ts(row[1])

    def read(
        self,
        station: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Observation]:
        """All stored rows for `station`, ordered by timestamp ascending."""
        query = f"SELECT {', '.join(_COLUMNS)} FROM observations WHERE station = ?"
        params: list = [station]
        if start is not None:
            query += " AND ts >= ?"
            params.append(start.astimezone(UTC).isoformat())
        if end is not None:
            query += " AND ts <= ?"
            params.append(end.astimezone(UTC).isoformat())
        query += " ORDER BY ts ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [Observation(_parse_ts(row[0]), *row[1:]) for row in rows]

    def latest(self, station: str) -> Observation | None:
        """Most recent stored row -- what a bite-score consumer wants: the
        current temperature/DO/turbidity/salinity reading, not the history."""
        rows = self.read(station)
        return rows[-1] if rows else None

    def salinity_series(self, station: str) -> list[tuple[datetime, float]]:
        """(timestamp, salinity_psu) pairs with a real reading, ordered
        ascending -- what Task 5's calibration wants: salinity paired with
        time at this station's one along-estuary distance. Rows where SAL
        was "MM" are excluded, not returned as 0.0."""
        rows = self._conn.execute(
            "SELECT ts, salinity_psu FROM observations "
            "WHERE station = ? AND salinity_psu IS NOT NULL ORDER BY ts ASC",
            (station,),
        ).fetchall()
        return [(_parse_ts(ts), sal) for ts, sal in rows]


def default_store(slug: str) -> NdbcStore:
    return NdbcStore(fishery_data_dir(slug) / "ndbc.sqlite")


def _fetch_text(station: str) -> str:
    resp = httpx.get(NDBC_URL.format(station=station), timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_and_store(station: str, store: NdbcStore) -> int:
    """One fetch-parse-append cycle. Returns the count of NEW rows stored.

    Raises `SourceUnavailable` for a network fault or non-2xx response --
    NOT caught here, per Task 4's convention: it is meant to propagate to
    `sources.dayloader.load_day`'s `attempt()`, which records the source by
    name in `missing`, rather than a failed fetch silently leaving the store
    exactly where it was with no trace anything went wrong. A failure here
    changes nothing already committed to `store` -- see `NdbcStore.append`.
    """
    try:
        text = _fetch_text(station)
    except Exception as exc:
        raise SourceUnavailable("ndbc", str(exc)) from exc
    rows = parse_ocean(text)
    return store.append(station, rows)
