"""Water Quality Portal salinity results -- the anchors the fit never had.

WHY THIS SOURCE EXISTS
----------------------
Phase 2 fitted the intrusion model against the full NERRS record and the
model was FALSIFIED, not merely unconstrained: rmse 4.060 ppt against an
observation resolution of 0.003, a factor of 1,353, with a healthy condition
number. The binding cause is coverage -- only 0.4% of the 2,162-feature
inventory sat where observations bracketed it, and the 2.58-13.05 km reach
holding 35% of features had no salinity observation at all.

WQP (waterqualitydata.us, the EPA/USGS/state aggregator) serves 208 salinity
stations in this bbox, 132 in-domain, 55 of them in that reach -- including
Winyah Bay MAIN CHANNEL stations at 5.56, 10.28 and 12.17 km. Measured live
2026-08-24. It is public and unauthenticated, unlike CDMO.

GRAB SAMPLES, NOT A FEED, AND WHY THAT IS FINE HERE
---------------------------------------------------
These are discrete samples -- WB-06 has 40 over four years, not 40 per day.
That would be useless to a model needing a time series, but this codebase
already holds the tide model and 10.6 years of composite discharge, so each
sample resolves to a known DISTANCE, DISCHARGE and TIDAL PHASE. One grab
sample with all three is a fully-specified observation.

That is also why a row with no usable time is REJECTED rather than defaulted
to noon: the phase is the point, and a fabricated time is a fabricated phase.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

import httpx

from tidescout.paths import fishery_data_dir
from tidescout.sources.ndbc import NdbcStore, Observation

# The WQP characteristic this module reads, and the ONLY one it will read.
# WQP also serves "Specific conductance" and "Conductivity"; `pipeline/salinity_fit.py:832`
# already holds the line that specific conductance is a different quantity and
# is not interchangeable with salinity. Mixing them here would be silent.
CHARACTERISTIC = "Salinity"

# Unit code -> multiplier onto psu. `0/00` is per-mille, numerically identical
# to ppt (81 ppt rows and 42 `0/00` rows in one real response). Anything not
# in here is REJECTED AND COUNTED: WQP serves mg/l and uS/cm under neighbouring
# characteristics and coercing one would inject nonsense at full confidence.
ACCEPTED_UNITS: dict[str, float] = {"ppt": 1.0, "0/00": 1.0, "psu": 1.0, "PSU": 1.0}

# `ResultStatusIdentifier` values admitted. "Final", "Accepted" and "Validated"
# are reviewed; "Preliminary" and "Provisional" are not, and are blocked --
# the same posture cdmo.py takes toward unvetted QAQC flags.
ACCEPTED_STATUSES = frozenset({"Final", "Accepted", "Validated", "Historical"})

# `ActivityTypeCode` prefixes that are NOT estuary measurements: field blanks,
# lab replicates, spikes. They pass every other filter and would enter the fit
# as real readings.
_QC_ACTIVITY_PREFIX = "Quality Control"

# WQP reports local clock time plus a US timezone abbreviation. There is no
# stdlib mapping from those abbreviations to offsets, and getting it wrong
# shifts a sample by up to 4 h -- most of a quarter tidal cycle at 12.42 h.
_TZ_OFFSETS: dict[str, int] = {
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
}


@dataclass(frozen=True)
class Sample:
    """One admitted grab sample."""

    station: str
    ts: datetime  # UTC, tz-aware
    salinity_psu: float
    # None means WQP recorded no depth -- true for 120 of 123 real rows.
    # NEVER 0.0, which would claim "surface".
    depth_m: float | None


@dataclass
class ParseReport:
    """Admitted samples plus a full account of what was not admitted.

    Every rejection path has a counter, and `test_counters_account_for_every_row`
    pins their sum to `n_rows`. `n_rows` counts EVERY row read from the file,
    regardless of `CharacteristicName` -- a caller (`import_results`, Task 2)
    hands this parser arbitrary multi-characteristic WQP exports, so the
    salinity subset is `n_rows - n_other_characteristic`, not `n_rows` itself.
    A rejection with no counter is a silent drop, and silent drops are how a
    fit quietly narrows its own inputs while looking complete.
    """

    samples: list[Sample] = field(default_factory=list)
    n_rows: int = 0
    n_admitted: int = 0
    n_no_time: int = 0
    n_bad_unit: int = 0
    n_bad_status: int = 0
    n_qc_activity: int = 0
    n_no_value: int = 0
    n_other_characteristic: int = 0
    unknown_units: dict[str, int] = field(default_factory=dict)
    unknown_statuses: dict[str, int] = field(default_factory=dict)


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _timestamp(date_s: str, time_s: str, tz_s: str) -> datetime | None:
    """Local clock time + US tz abbreviation -> UTC. None if unusable."""
    date_s, time_s, tz_s = date_s.strip(), time_s.strip(), tz_s.strip().upper()
    if not date_s or not time_s or tz_s not in _TZ_OFFSETS:
        return None
    tz = timezone(timedelta(hours=_TZ_OFFSETS[tz_s]))
    try:
        naive = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    except ValueError:
        return None
    return naive.astimezone(UTC)


def _depth_m(value: str, unit: str) -> float | None:
    value, unit = value.strip(), unit.strip().lower()
    if not value:
        return None
    try:
        d = float(value)
    except ValueError:
        return None
    if unit in ("m", "meters", "meter"):
        return d
    if unit in ("ft", "feet", "foot"):
        return d * 0.3048
    return None


def parse_results(fh: Iterable[str]) -> ParseReport:
    """WQP Result CSV -> admitted salinity samples, plus why the rest went."""
    report = ParseReport()
    for row in csv.DictReader(fh):
        report.n_rows += 1
        if (row.get("CharacteristicName") or "").strip() != CHARACTERISTIC:
            # A different parameter entirely -- e.g. Specific conductance,
            # not interchangeable with salinity (pipeline/salinity_fit.py:832).
            # Still counted: `import_results` (Task 2) hands this parser
            # arbitrary multi-characteristic exports, and an uncounted drop
            # here would be invisible against the raw file.
            report.n_other_characteristic += 1
            continue

        activity = (row.get("ActivityTypeCode") or "").strip()
        if activity.startswith(_QC_ACTIVITY_PREFIX):
            report.n_qc_activity += 1
            continue

        status = (row.get("ResultStatusIdentifier") or "").strip()
        if status and status not in ACCEPTED_STATUSES:
            report.n_bad_status += 1
            _bump(report.unknown_statuses, status)
            continue

        raw = (row.get("ResultMeasureValue") or "").strip()
        if not raw or (row.get("ResultDetectionConditionText") or "").strip():
            report.n_no_value += 1
            continue
        try:
            value = float(raw)
        except ValueError:
            report.n_no_value += 1
            continue

        unit = (row.get("ResultMeasure/MeasureUnitCode") or "").strip()
        if unit not in ACCEPTED_UNITS:
            report.n_bad_unit += 1
            _bump(report.unknown_units, unit or "<blank>")
            continue

        ts = _timestamp(
            row.get("ActivityStartDate", ""),
            row.get("ActivityStartTime/Time", ""),
            row.get("ActivityStartTime/TimeZoneCode", ""),
        )
        if ts is None:
            report.n_no_time += 1
            continue

        report.samples.append(
            Sample(
                station=(row.get("MonitoringLocationIdentifier") or "").strip(),
                ts=ts,
                salinity_psu=value * ACCEPTED_UNITS[unit],
                depth_m=_depth_m(
                    row.get("ActivityDepthHeightMeasure/MeasureValue", ""),
                    row.get("ActivityDepthHeightMeasure/MeasureUnitCode", ""),
                ),
            )
        )
        report.n_admitted += 1
    return report


# -- fetch, store, and station positions -------------------------------------
#
# WQP (waterqualitydata.us) serves this bbox's salinity results from ONE
# endpoint and this bbox's station metadata (lat/lon) from a SIBLING
# endpoint -- same host, same query shape, different `/data/...` path.
# Both are public and unauthenticated: no `key`/`apiKey` param exists to
# send, unlike CDMO's static-IP-registration gate.

WQP_URL = "https://www.waterqualitydata.us/data/Result/search"

# Verified live 2026-08-24: filtering by `characteristicName=Salinity` here
# returns exactly the stations that could ever appear in a `fetch_results`
# response for the same bbox, so the two endpoints stay in lockstep.
WQP_STATION_URL = "https://www.waterqualitydata.us/data/Station/search"

# Provenance source tag. Distinguishing WQP rows from NERRS rows is what makes
# correct attribution possible at all -- see `NdbcStore.citation`.
SOURCE_WQP = "wqp:salinity"

BATCH_ROWS = 20_000


def default_store(slug: str) -> NdbcStore:
    """The WQP store -- a SEPARATE file from the NERRS one.

    Same class, same proven append-and-dedupe contract, different file, so
    `ndbc.py`'s stated "this store only ever holds NIW NERR data" invariant
    survives and its citation cannot overclaim.
    """
    return NdbcStore(fishery_data_dir(slug) / "wqp.sqlite")


def _bbox_params(bbox: tuple[float, float, float, float]) -> dict[str, str]:
    # `:.2f`, not a bare f-string: a plain float drops a trailing zero
    # (33.60 -> "33.6"), and `test_fetch_builds_a_bbox_query_without_a_key`
    # pins the exact query string against the fishery's own 2-decimal bbox.
    lon_min, lat_min, lon_max, lat_max = bbox
    return {
        "bBox": f"{lon_min:.2f},{lat_min:.2f},{lon_max:.2f},{lat_max:.2f}",
        "characteristicName": CHARACTERISTIC,
        "mimeType": "csv",
        "zip": "no",
    }


def fetch_results(
    bbox: tuple[float, float, float, float], timeout: float = 180.0
) -> str:
    """Raw CSV for every salinity result in `bbox`. Public, no key.

    `bbox` is (lon_min, lat_min, lon_max, lat_max) -- the same order
    `Fishery.bbox` uses, so a caller passes it straight through.
    """
    resp = httpx.get(WQP_URL, params=_bbox_params(bbox), timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_stations(
    bbox: tuple[float, float, float, float], timeout: float = 180.0
) -> str:
    """Raw CSV of station metadata (incl. lat/lon) for every station in
    `bbox` that reports Salinity. Same bbox convention as `fetch_results`."""
    resp = httpx.get(WQP_STATION_URL, params=_bbox_params(bbox), timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_stations(fh: Iterable[str]) -> dict[str, tuple[float, float]]:
    """WQP Station CSV -> `{MonitoringLocationIdentifier: (lon, lat)}`.

    A row with a blank id or an unparseable coordinate is skipped -- station
    metadata is a convenience for later positioning, not a fit input in its
    own right, so this has no counter of its own the way `parse_results`
    does for every rejected reading.
    """
    out: dict[str, tuple[float, float]] = {}
    for row in csv.DictReader(fh):
        sid = (row.get("MonitoringLocationIdentifier") or "").strip()
        lat_s = (row.get("LatitudeMeasure") or "").strip()
        lon_s = (row.get("LongitudeMeasure") or "").strip()
        if not sid or not lat_s or not lon_s:
            continue
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue
        out[sid] = (lon, lat)
    return out


@dataclass
class ImportReport:
    parse: ParseReport
    n_new: int
    stations: tuple[str, ...]
    span: tuple[datetime, datetime] | None


# `wqp_stations` is this module's OWN table -- NdbcStore has no idea it
# exists (see this module's docstring, "Station coordinates need a home").
# Created lazily here rather than in `NdbcStore.__init__`, which stays
# generic across every caller that reuses the class.
_CREATE_STATIONS_TABLE = (
    "CREATE TABLE IF NOT EXISTS wqp_stations "
    "(station TEXT PRIMARY KEY, lon REAL NOT NULL, lat REAL NOT NULL)"
)


def import_results(
    text: str, store: NdbcStore, stations_csv: str | None = None
) -> ImportReport:
    """Parse and store, whole-batch-atomic, deduped by (station, ts).

    `stations_csv`, when given (WQP's own Station CSV for the same bbox,
    e.g. from `fetch_stations`), is parsed and written into `wqp_stations`
    inside the SAME transaction as the observations and the provenance row
    -- station positions travel with the readings they belong to, and a
    rollback takes a partial import's coordinates with it too.
    """
    report = parse_results(text.splitlines())
    by_station: dict[str, list[Observation]] = {}
    for s in report.samples:
        by_station.setdefault(s.station, []).append(
            Observation(
                ts=s.ts,
                depth_m=s.depth_m,
                water_temp_c=None,
                cond_ms_cm=None,
                salinity_psu=s.salinity_psu,
                o2_pct=None,
                o2_ppm=None,
                chlorophyll_ug_l=None,
                turbidity_ftu=None,
                ph=None,
                eh_mv=None,
            )
        )
    coords = parse_stations(stations_csv.splitlines()) if stations_csv else {}

    # DDL is idempotent and carries no data of its own, so it is harmless
    # for it to land ahead of (and outside) the transaction below; only the
    # coordinate ROWS need the same all-or-nothing guarantee as the readings.
    store._conn.execute(_CREATE_STATIONS_TABLE)

    n_new = 0
    with store.bulk_writer(batch_rows=BATCH_ROWS) as writer:
        for station, rows in sorted(by_station.items()):
            for row in rows:
                writer.add(station, row)
        writer.flush()
        n_new = sum(writer.n_new(station) for station in by_station)
        span = (
            (min(s.ts for s in report.samples), max(s.ts for s in report.samples))
            if report.samples
            else None
        )
        if coords:
            # Same connection `bulk_writer` commits/rolls back below, so
            # these rows share its atomicity even though they go through
            # `store._conn` rather than `writer` (which only knows about
            # `observations`/`met_observations`).
            store._conn.executemany(
                "INSERT OR REPLACE INTO wqp_stations (station, lon, lat) VALUES (?, ?, ?)",
                [(sid, lon, lat) for sid, (lon, lat) in coords.items()],
            )
        # Inside the transaction, so a rollback takes the provenance row
        # with it -- a failed import must leave no record of data it never
        # wrote.
        writer.record_provenance(SOURCE_WQP, sorted(by_station), span, n_new)
    return ImportReport(report, n_new, tuple(sorted(by_station)), span)


def station_coords_from_store(store: NdbcStore) -> dict[str, tuple[float, float]]:
    """`{station: (lon, lat)}` read back from `wqp_stations`.

    `{}` if the table does not exist yet (a store `import_results` has never
    written into), not an error -- the same "no data yet" posture as an
    empty `citation()`.
    """
    exists = store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'wqp_stations'"
    ).fetchone()
    if exists is None:
        return {}
    rows = store._conn.execute("SELECT station, lon, lat FROM wqp_stations").fetchall()
    return {station: (lon, lat) for station, lon, lat in rows}


def station_coords(slug: str) -> dict[str, tuple[float, float]]:
    """`station_coords_from_store`, opening this fishery's default WQP
    store. `{}` when the store's file does not exist yet -- checked before
    opening it, so a mere read never creates an empty `wqp.sqlite` as a
    side effect the way constructing `NdbcStore` normally would."""
    path = fishery_data_dir(slug) / "wqp.sqlite"
    if not path.exists():
        return {}
    return station_coords_from_store(NdbcStore(path))
