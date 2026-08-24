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

A SECOND TABLE FOR METEOROLOGICAL DATA (Task 10)
-------------------------------------------------
`met_observations` is a SIBLING table in the same file, not a widening of
`observations`. Water-quality and meteorological parameters are disjoint
sets (temperature/salinity/DO/turbidity/pH vs. air-temp/wind/pressure/
humidity/PAR/precipitation) with disjoint station namespaces (WYSS1 and
friends vs. the reserve's one weather station) -- see `sources/cdmo.py`'s
"MET FILE SUPPORT" docstring section for the full reasoning (that is where
the parsing and the decision were made; this module just owns the table,
the same way it already owns `observations`). `MetObservation`,
`append_met`, `read_met`, `latest_met`, and `met_time_span` mirror their
WQ-table counterparts exactly -- same atomic-transaction dedupe contract,
same MM/None-not-zero discipline, same "stores and exposes, nothing wired
into scoring yet" seam.

PROVENANCE AND CITATION (Task 10)
-------------------------------------------------
NERRS's citation requirement (nerrsdata.org/data/citation.cfm) names two
facts only the code can honestly supply: the date the data were ACCESSED,
and the SUBSET actually held. Neither can be read off the observation rows
themselves -- a row's `ts` is when the water/air was measured, not when
this program fetched it, and the subset changes every time an import runs.
So every write path that adds real data (`fetch_and_store` here,
`cdmo._apply`/`_apply_met` in the sibling module) also calls
`record_provenance`, appending one row to a third table: accessed-at
(UTC, this call's own wall-clock time), source route (`ndbc:realtime2` vs
`cdmo:water_quality` vs `cdmo:meteorological`), the station codes touched,
and the timestamp span of the rows in THAT call's batch. `citation()` then
derives the citation from what is ACTUALLY on record rather than a
hardcoded string: the accessed date is the most recent `accessed_at` across
every provenance row (how fresh is what we hold), while the subset
description is read LIVE from `observations`/`met_observations` (station,
row count, date span) rather than from the provenance log's own per-import
spans -- the live tables can never go stale relative to what a caller
actually gets back from `read`/`read_met`, whereas summing historical
per-import spans could silently drift from the true union. A store with no
provenance rows at all (a raw `append` call bypassing this bookkeeping, as
most of this module's own tests do) still produces a complete `Citation`;
its `accessed_date` is `None` and its `.text` says so explicitly rather
than fabricating a date.
"""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from tidescout.errors import SourceUnavailable
from tidescout.paths import fishery_data_dir

__all__ = [
    "NDBC_URL",
    "BulkWriter",
    "NERRS_ACKNOWLEDGEMENT",
    "NERRS_CITATION_TEMPLATE",
    "NERRS_DISCLAIMER",
    "SOURCE_NDBC_REALTIME2",
    "Citation",
    "MetObservation",
    "NdbcStore",
    "Observation",
    "ProvenanceRecord",
    "default_store",
    "fetch_and_store",
    "parse_ocean",
]

NDBC_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.ocean"

# Provenance `source` label for a fetch through this module. cdmo.py defines
# its own two sibling labels (`SOURCE_CDMO_WQ`, `SOURCE_CDMO_MET`) -- kept
# next to the code that actually performs each fetch, not centralised here.
SOURCE_NDBC_REALTIME2 = "ndbc:realtime2"

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


@dataclass(frozen=True)
class MetObservation:
    """One meteorological reading -- the reserve's single weather station
    (see `sources/cdmo.py`'s "MET FILE SUPPORT" section: NIW runs exactly
    one, at Oyster Landing, station code `niwolmet`). `None` means the
    parameter was missing, rejected, or not collected -- never 0.0; see
    `sources/cdmo.py` for how each is decided.

    Units, verbatim from the reserve's own meteorological metadata PDF
    (fetched live 2026-08-23): air_temp_c degrees C, rh_pct percent,
    bp_mb millibars, wind speeds m/s, wind directions degrees, par_mmol_m2
    millimoles/m^2 (a 15-minute TOTAL, not an instantaneous rate),
    precip_mm millimeters (also a 15-minute total), solar_rad_wm2 watts/m^2.
    """

    ts: datetime  # UTC, tz-aware
    air_temp_c: float | None
    rh_pct: float | None
    bp_mb: float | None
    wind_speed_ms: float | None
    max_wind_speed_ms: float | None
    wind_dir_deg: float | None
    wind_dir_sd_deg: float | None
    par_mmol_m2: float | None
    precip_mm: float | None
    solar_rad_wm2: float | None


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

_MET_COLUMNS = (
    "ts", "air_temp_c", "rh_pct", "bp_mb", "wind_speed_ms", "max_wind_speed_ms",
    "wind_dir_deg", "wind_dir_sd_deg", "par_mmol_m2", "precip_mm", "solar_rad_wm2",
)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


@dataclass(frozen=True)
class ProvenanceRecord:
    """One import/fetch event, as recorded by `NdbcStore.record_provenance`."""

    accessed_at: datetime  # UTC, tz-aware -- when THIS program touched the source
    source: str  # e.g. "ndbc:realtime2", "cdmo:water_quality", "cdmo:meteorological"
    stations: tuple[str, ...]  # canonical station codes this call wrote
    span: tuple[datetime, datetime] | None  # (earliest, latest) ts in this call's batch
    n_new: int


# Verbatim from https://nerrsdata.org/data/citation.cfm (fetched live
# 2026-08-23) -- the `{date}` placeholder is the only part this codebase
# fills in; everything else is NERRS's own required wording, not paraphrased.
NERRS_CITATION_TEMPLATE = (
    "NOAA National Estuarine Research Reserve System (NERRS). System-wide "
    "Monitoring Program. Data accessed from the NOAA NERRS Centralized Data "
    "Management Office website: http://www.nerrsdata.org; accessed {date}. "
    "doi:10.25921/vw8a-8031."
)

# This store only ever holds North Inlet-Winyah Bay (NIW) NERR data (every
# station in `sources/cdmo.py`'s coordinate tables is NIW), so naming that
# one reserve here is a fact, not a premature generalisation -- see NERRS's
# distribution clause, which requires the SPECIFIC reserve be credited, not
# NERRS-in-the-abstract. Contact per this task's dispatch.
NERRS_ACKNOWLEDGEMENT = (
    "The NERRS retains the right to be fully credited for having collected "
    "and processed the data. Following academic courtesy standards, the "
    "NERR site where the data were collected should be contacted and fully "
    "acknowledged in any subsequent publications in which any part of the "
    "data are used: North Inlet-Winyah Bay NERR, Baruch Marine Field "
    "Laboratory, University of South Carolina, PO Box 1630, Georgetown, SC "
    "29442 (cdmodata@baruch.sc.edu)."
)

# Verbatim from nerrsdata.org/data/citation.cfm AND both reserve metadata
# PDFs' "Distribution" sections (word-for-word identical across all three,
# cross-checked live 2026-08-23). Longer than this task's dispatch quoted --
# the dispatch dropped the trailing "nor will the Federal government
# reimburse or indemnify..." clause; restored here from the primary source.
NERRS_DISCLAIMER = (
    "The user bears all responsibility for its subsequent use/misuse in any "
    "further analyses or comparisons. The Federal government does not "
    "assume liability to the Recipient or third persons, nor will the "
    "Federal government reimburse or indemnify the Recipient for its "
    "liability due to any losses resulting in any way from the use of this "
    "data."
)


@dataclass(frozen=True)
class Citation:
    """Everything `NdbcStore.citation()` derives from the store's own
    provenance and data tables -- see this module's docstring, "PROVENANCE
    AND CITATION". `text` is NERRS's exact requested citation line with the
    access date filled in; `subset_lines` is this codebase's own answer to
    NERRS's OTHER requirement ("the subset of data that was used"), which
    NERRS's template has no placeholder for."""

    text: str
    acknowledgement: str
    disclaimer: str
    accessed_date: date | None
    subset_lines: tuple[str, ...]
    sources: tuple[str, ...]


# -- the one place each table's row shape is written down ---------------------
# `append`/`append_met`/`record_provenance` and `BulkWriter` all bind through
# these, so the small-batch and streaming write paths can never drift apart in
# column order or count.

_INSERT_WQ = (
    "INSERT OR REPLACE INTO observations "
    "(station, ts, depth_m, water_temp_c, cond_ms_cm, salinity_psu, "
    " o2_pct, o2_ppm, chlorophyll_ug_l, turbidity_ftu, ph, eh_mv) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_INSERT_MET = (
    "INSERT OR REPLACE INTO met_observations "
    "(station, ts, air_temp_c, rh_pct, bp_mb, wind_speed_ms, "
    " max_wind_speed_ms, wind_dir_deg, wind_dir_sd_deg, "
    " par_mmol_m2, precip_mm, solar_rad_wm2) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_INSERT_PROVENANCE = (
    "INSERT INTO provenance "
    "(accessed_at, source, stations, span_start, span_end, n_new) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


def _wq_params(station: str, r: Observation) -> tuple:
    return (
        station, r.ts.astimezone(UTC).isoformat(),
        r.depth_m, r.water_temp_c, r.cond_ms_cm, r.salinity_psu,
        r.o2_pct, r.o2_ppm, r.chlorophyll_ug_l, r.turbidity_ftu,
        r.ph, r.eh_mv,
    )


def _met_params(station: str, r: MetObservation) -> tuple:
    return (
        station, r.ts.astimezone(UTC).isoformat(),
        r.air_temp_c, r.rh_pct, r.bp_mb, r.wind_speed_ms,
        r.max_wind_speed_ms, r.wind_dir_deg, r.wind_dir_sd_deg,
        r.par_mmol_m2, r.precip_mm, r.solar_rad_wm2,
    )


def _provenance_params(
    source: str,
    stations: Sequence[str],
    span: tuple[datetime, datetime] | None,
    n_new: int,
    accessed_at: datetime | None = None,
) -> tuple:
    at = (accessed_at or datetime.now(UTC)).astimezone(UTC)
    return (
        at.isoformat(),
        source,
        ",".join(sorted(stations)),
        span[0].astimezone(UTC).isoformat() if span else None,
        span[1].astimezone(UTC).isoformat() if span else None,
        n_new,
    )


class BulkWriter:
    """A streaming write into an open transaction -- see `NdbcStore.bulk_writer`.

    `append` takes a `Sequence`, which means the caller has already built
    every row in memory. That is the right shape for a 4,000-row NDBC fetch
    and the wrong one for a 2.5-million-row CDMO export, where materialising
    the rows costs more than the file itself. This accumulates at most
    `batch_rows` bound tuples per table before handing them to SQLite, so
    peak memory is set by `batch_rows` rather than by the file.

    Both tables and the provenance row are written through ONE transaction
    owned by `bulk_writer`, so "the whole import or none of it" covers a
    multi-station, multi-table, multi-batch import exactly the way
    `append`'s single `with self._conn` covers one batch.
    """

    def __init__(self, conn: sqlite3.Connection, batch_rows: int = 20_000):
        self._conn = conn
        self._batch_rows = batch_rows
        self._wq: list[tuple] = []
        self._met: list[tuple] = []
        # Count at the moment this import first touched each station, taken
        # lazily so a station the file never mentions is never queried.
        self._before_wq: dict[str, int] = {}
        self._before_met: dict[str, int] = {}

    def add(self, station: str, row: Observation) -> None:
        if station not in self._before_wq:
            self._before_wq[station] = self._count("observations", station)
        self._wq.append(_wq_params(station, row))
        if len(self._wq) >= self._batch_rows:
            self.flush()

    def add_met(self, station: str, row: MetObservation) -> None:
        if station not in self._before_met:
            self._before_met[station] = self._count("met_observations", station)
        self._met.append(_met_params(station, row))
        if len(self._met) >= self._batch_rows:
            self.flush()

    def flush(self) -> None:
        """Hand every buffered row to SQLite. Still inside the transaction --
        nothing is durable until `bulk_writer` commits."""
        if self._wq:
            self._conn.executemany(_INSERT_WQ, self._wq)
            self._wq.clear()
        if self._met:
            self._conn.executemany(_INSERT_MET, self._met)
            self._met.clear()

    def n_new(self, station: str) -> int:
        """Rows this import ADDED for `station` -- re-importing a file that
        is already stored reports 0, not its row count. Call after `flush`;
        the uncommitted rows are visible to this connection."""
        before = self._before_wq.get(station, 0)
        return self._count("observations", station) - before

    def n_new_met(self, station: str) -> int:
        before = self._before_met.get(station, 0)
        return self._count("met_observations", station) - before

    def record_provenance(
        self,
        source: str,
        stations: Sequence[str],
        span: tuple[datetime, datetime] | None,
        n_new: int,
        accessed_at: datetime | None = None,
    ) -> None:
        """`NdbcStore.record_provenance`, inside THIS transaction rather than
        one of its own -- so a rollback takes the provenance row with it and
        a failed import leaves no record of data it never committed."""
        self._conn.execute(
            _INSERT_PROVENANCE,
            _provenance_params(source, stations, span, n_new, accessed_at),
        )

    def _count(self, table: str, station: str) -> int:
        (n,) = self._conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE station = ?", (station,)
        ).fetchone()
        return n


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
        # A SIBLING table, not a widened `observations` -- see this module's
        # docstring, "A SECOND TABLE FOR METEOROLOGICAL DATA".
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS met_observations ("
            " station TEXT NOT NULL,"
            " ts TEXT NOT NULL,"
            " air_temp_c REAL, rh_pct REAL, bp_mb REAL,"
            " wind_speed_ms REAL, max_wind_speed_ms REAL,"
            " wind_dir_deg REAL, wind_dir_sd_deg REAL,"
            " par_mmol_m2 REAL, precip_mm REAL, solar_rad_wm2 REAL,"
            " PRIMARY KEY (station, ts))"
        )
        # See this module's docstring, "PROVENANCE AND CITATION" -- one row
        # per real write, never per failed/aborted import.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS provenance ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " accessed_at TEXT NOT NULL,"
            " source TEXT NOT NULL,"
            " stations TEXT NOT NULL,"
            " span_start TEXT, span_end TEXT,"
            " n_new INTEGER NOT NULL)"
        )
        self._conn.commit()

    def append(self, station: str, rows: Sequence[Observation]) -> int:
        """Insert `rows`, de-duplicated by (station, timestamp).

        Returns the count of rows that were actually NEW: re-fetching a
        window that overlaps what is already stored returns 0 for the
        overlap, not `len(rows)` -- see `test_append_and_dedupe_across_
        overlapping_fetches` for the proof this matters.

        For a batch small enough to hold in memory. `bulk_writer` is the
        streaming equivalent for an import too large to materialise (see
        its docstring); both go through `_wq_params` and `_INSERT_WQ`, so
        the row shape is defined in exactly one place.
        """
        before = self.count(station)
        with self._conn:
            self._conn.executemany(_INSERT_WQ, (_wq_params(station, r) for r in rows))
        return self.count(station) - before

    @contextmanager
    def bulk_writer(self, batch_rows: int = 20_000) -> Iterator[BulkWriter]:
        """One transaction spanning an entire streamed import.

        `append` is whole-BATCH atomic; this is whole-IMPORT atomic. The
        difference matters once a single file is too big to be one batch:
        without it a 2.5 M-row import that dies at row 2 million would leave
        the store holding an arbitrary prefix of a file, indistinguishable
        from a complete one. Any exception raised inside the `with` body --
        a malformed row, an unbindable value, a KeyboardInterrupt -- rolls
        the whole thing back, including the provenance row.
        """
        writer = BulkWriter(self._conn, batch_rows=batch_rows)
        try:
            yield writer
            writer.flush()
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

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

    # -- met_observations: same contract as observations, sibling table ----

    def append_met(self, station: str, rows: Sequence[MetObservation]) -> int:
        """`append`'s exact contract (whole-batch-atomic, `(station, ts)`
        dedupe), against `met_observations` instead of `observations`."""
        before = self.count_met(station)
        with self._conn:
            self._conn.executemany(_INSERT_MET, (_met_params(station, r) for r in rows))
        return self.count_met(station) - before

    def count_met(self, station: str) -> int:
        (n,) = self._conn.execute(
            "SELECT COUNT(*) FROM met_observations WHERE station = ?", (station,)
        ).fetchone()
        return n

    def met_time_span(self, station: str) -> tuple[datetime, datetime] | None:
        row = self._conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM met_observations WHERE station = ?", (station,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return _parse_ts(row[0]), _parse_ts(row[1])

    def read_met(
        self,
        station: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MetObservation]:
        """All stored MET rows for `station`, ordered by timestamp ascending."""
        query = f"SELECT {', '.join(_MET_COLUMNS)} FROM met_observations WHERE station = ?"
        params: list = [station]
        if start is not None:
            query += " AND ts >= ?"
            params.append(start.astimezone(UTC).isoformat())
        if end is not None:
            query += " AND ts <= ?"
            params.append(end.astimezone(UTC).isoformat())
        query += " ORDER BY ts ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [MetObservation(_parse_ts(row[0]), *row[1:]) for row in rows]

    def latest_met(self, station: str) -> MetObservation | None:
        """Most recent stored MET row -- what a future bite-score consumer
        wants: current wind/pressure/temperature, not the history."""
        rows = self.read_met(station)
        return rows[-1] if rows else None

    # -- provenance and citation --------------------------------------------

    def record_provenance(
        self,
        source: str,
        stations: Sequence[str],
        span: tuple[datetime, datetime] | None,
        n_new: int,
        accessed_at: datetime | None = None,
    ) -> None:
        """One row per real write -- see this module's docstring,
        "PROVENANCE AND CITATION". Callers (`fetch_and_store` here,
        `cdmo._apply`/`_apply_met`) call this immediately AFTER a
        successful `append`/`append_met`, never before: if the write
        raised, this line is never reached, so a failed import records no
        provenance for data it never actually committed.

        `accessed_at` defaults to now (UTC) -- overridable only for tests
        that need a fixed clock; every real call site uses the default.
        """
        with self._conn:
            self._conn.execute(
                _INSERT_PROVENANCE,
                _provenance_params(source, stations, span, n_new, accessed_at),
            )

    def provenance(self) -> list[ProvenanceRecord]:
        """Every recorded import/fetch, oldest first."""
        rows = self._conn.execute(
            "SELECT accessed_at, source, stations, span_start, span_end, n_new "
            "FROM provenance ORDER BY accessed_at ASC, id ASC"
        ).fetchall()
        out = []
        for accessed_at, source, stations, span_start, span_end, n_new in rows:
            span = (
                (_parse_ts(span_start), _parse_ts(span_end))
                if span_start is not None and span_end is not None
                else None
            )
            out.append(
                ProvenanceRecord(
                    accessed_at=_parse_ts(accessed_at),
                    source=source,
                    stations=tuple(stations.split(",")) if stations else (),
                    span=span,
                    n_new=n_new,
                )
            )
        return out

    def citation(self) -> Citation:
        """Generate the NERRS citation from what THIS store actually holds
        right now -- never a hardcoded string (see this module's docstring).
        Always returns a complete `Citation`, even for an empty store or one
        with data but no provenance (an honest `accessed_date=None` rather
        than a fabricated date -- see `.text`'s wording in that case).
        """
        prov = self.provenance()
        accessed_dt = max((p.accessed_at for p in prov), default=None)
        accessed_date = accessed_dt.astimezone(UTC).date() if accessed_dt is not None else None
        sources = tuple(sorted({p.source for p in prov}))

        if accessed_date is not None:
            date_str = f"{accessed_date.day} {accessed_date:%B} {accessed_date.year}"
        else:
            date_str = "[NO RECORDED ACCESS -- this store predates provenance tracking]"
        text = NERRS_CITATION_TEMPLATE.format(date=date_str)

        wq_stations = [
            r[0] for r in self._conn.execute("SELECT DISTINCT station FROM observations")
        ]
        met_stations = [
            r[0] for r in self._conn.execute("SELECT DISTINCT station FROM met_observations")
        ]
        lines = []
        for st in sorted(wq_stations):
            n = self.count(st)
            span = self.time_span(st)
            span_str = f"{span[0].date()} to {span[1].date()}" if span else "no dated rows"
            lines.append(f"{st} (water quality, {n:,} observation(s), {span_str})")
        for st in sorted(met_stations):
            n = self.count_met(st)
            span = self.met_time_span(st)
            span_str = f"{span[0].date()} to {span[1].date()}" if span else "no dated rows"
            lines.append(f"{st} (meteorological, {n:,} observation(s), {span_str})")
        subset_lines = tuple(lines) if lines else ("no observations held",)

        return Citation(
            text=text,
            acknowledgement=NERRS_ACKNOWLEDGEMENT,
            disclaimer=NERRS_DISCLAIMER,
            accessed_date=accessed_date,
            subset_lines=subset_lines,
            sources=sources,
        )


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

    Also records provenance (see this module's docstring, "PROVENANCE AND
    CITATION") -- but only after `append` returns successfully, so a raised
    `SourceUnavailable` or a poisoned batch records no access to data that
    was never actually committed.
    """
    try:
        text = _fetch_text(station)
    except Exception as exc:
        raise SourceUnavailable("ndbc", str(exc)) from exc
    rows = parse_ocean(text)
    n_new = store.append(station, rows)
    span = (min(r.ts for r in rows), max(r.ts for r in rows)) if rows else None
    store.record_provenance(SOURCE_NDBC_REALTIME2, [station], span, n_new)
    return n_new
