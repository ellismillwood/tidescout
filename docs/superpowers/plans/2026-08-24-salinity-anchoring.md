# Salinity Anchoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anchor the salt-intrusion model in the 2.58–13.05 km reach it has never observed, using
public Water Quality Portal survey data, and screen contributing stations onto the estuary's main
stem by a computed criterion rather than by hand.

**Architecture:** A new `sources/wqp.py` fetches and parses WQP salinity results into a second
`NdbcStore`-backed sqlite file, reusing the append-and-dedupe contract already proven at 2.5 M
rows. `pipeline/estuary.py` gains a main-stem mask and a distance-to-stem field — the same
Dijkstra `along_estuary_km` already runs, seeded from the stem instead of the ocean — which
replaces the hand-set `off_axis` flag with a derived one. `SalinityField` gains a per-cell
coverage ordinal. The model form is deliberately NOT changed; Task 7 refits and reports at a gate.

**Tech Stack:** Python 3.12, httpx, numpy, scipy, pydantic v2, typer, rich, sqlite3, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-24-salinity-anchoring-design.md`

## Global Constraints

- Run everything from the repo root. `make check` = `ruff check` + `pytest`, and must be green
  before every commit. Test count only ever goes UP (558 at the start of this plan).
- Python interpreter is `$(HOME)/.venvs/tidescout/bin/python`. Never `pip install` — the venv is
  managed by `make install`.
- `Salinity` only. NEVER specific conductance — `sources/usgs.py` holds that line already
  ("a different quantity and is not interchangeable"). WQP serves both under one query.
- Units on an explicit allowlist. Real data carries both `ppt` and `0/00`. Anything unrecognised
  is REJECTED AND COUNTED, never coerced.
- Reject-and-report, never silently drop: every rejection path carries a counter that reaches the
  CLI report. This is the `sources/cdmo.py` pattern and it caught four wrong inferences when the
  real CDMO export arrived.
- Do NOT touch `model_domain.ocean_boundary_utm_km`, the ANUGA mesh, or the flow library.
  `mesh.classify_boundary` reads that polygon and changing it invalidates the 12 regimes.
- Do NOT change any value in the `salinity:` block of `fisheries/winyah-bay.yaml`. Task 7 reports;
  Ellis decides.
- Timezone: store everything UTC, tz-aware. `datetime.now(UTC)`, never `utcnow()`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/tidescout/sources/wqp.py` | **Create.** Fetch + parse WQP salinity CSV; unit/QA screening; import into a store. |
| `backend/tidescout/pipeline/estuary.py` | **Modify.** Add `descent_path`, `main_stem_mask`, `build_stem_distance_field`, `load_stem_distance_field`. |
| `backend/tidescout/pipeline/salinity_fit.py` | **Modify.** Read WQP store; derive `off_axis` from the stem distance; report coverage. |
| `backend/tidescout/engine/salinity.py` | **Modify.** Add per-cell `coverage` to `SalinityField`. |
| `backend/tidescout/sources/ndbc.py` | **Modify.** Make `citation()` source-aware instead of always-NERRS. |
| `backend/tidescout/models.py` | **Modify.** `WaterSensor.off_axis` becomes an override, not the source of truth. |
| `backend/tidescout/cli.py` | **Modify.** `salinity import-wqp`, `salinity stem`. |
| `backend/tests/test_wqp.py` | **Create.** Parser + screening tests against a verbatim real excerpt. |
| `backend/tests/fixtures/wqp_results_excerpt.csv` | **Create.** Verbatim rows from a real WQP response. |
| `backend/tests/test_estuary.py` | **Modify.** Stem + distance-to-stem tests. |
| `backend/tests/test_salinity.py` | **Modify.** Derived-off_axis and coverage tests. |

---

## Task 1: Parse WQP salinity results

**Files:**
- Create: `backend/tidescout/sources/wqp.py`
- Create: `backend/tests/test_wqp.py`
- Create: `backend/tests/fixtures/wqp_results_excerpt.csv`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Sample` dataclass: `station: str`, `ts: datetime` (UTC, tz-aware), `salinity_psu: float`, `depth_m: float | None`
  - `ParseReport` dataclass: `samples: list[Sample]`, `n_rows: int`, `n_admitted: int`, `n_no_time: int`, `n_bad_unit: int`, `n_bad_status: int`, `n_qc_activity: int`, `n_no_value: int`, `unknown_units: dict[str, int]`, `unknown_statuses: dict[str, int]`
  - `parse_results(fh: Iterable[str]) -> ParseReport`
  - `ACCEPTED_UNITS: dict[str, float]`, `ACCEPTED_STATUSES: frozenset[str]`, `CHARACTERISTIC: str`

**Background the implementer needs.** WQP (`waterqualitydata.us`) aggregates EPA STORET, USGS
NWIS and state agencies. Its `Result` CSV has 63 columns. The ones that matter, with REAL values
measured from a live response on 2026-08-24:

| column | real values seen |
|---|---|
| `MonitoringLocationIdentifier` | `21SC60WQ_WQX-WB-06` |
| `CharacteristicName` | `Salinity` |
| `ResultMeasureValue` | `28.4` (may be empty) |
| `ResultMeasure/MeasureUnitCode` | `ppt` (81), `0/00` (42) |
| `ActivityStartDate` | `2014-06-12` |
| `ActivityStartTime/Time` | `11:35:00` (may be empty) |
| `ActivityStartTime/TimeZoneCode` | `EDT` (120), empty (3) |
| `ActivityTypeCode` | `Sample-Routine`, `Field Msr/Obs` |
| `ResultStatusIdentifier` | `Final` |
| `ActivityDepthHeightMeasure/MeasureValue` | usually EMPTY; `0.3` when present |
| `ActivityDepthHeightMeasure/MeasureUnitCode` | `m` when present |
| `ResultDetectionConditionText` | empty when the value is real |

Four traps, each of which silently corrupts if missed:

1. **`0/00` is per-mille — numerically identical to `ppt`.** Both admitted, factor 1.0. Any OTHER
   unit is rejected: WQP also serves `mg/l`, `uS/cm` and `PSU` under neighbouring characteristics,
   and coercing one would inject nonsense at full confidence.
2. **QC activities must be excluded.** `ActivityTypeCode` values beginning `Quality Control` are
   field blanks and lab replicates, not estuary measurements. They pass every other filter.
3. **Depth is usually MISSING** (120 of 123 rows). `None`, never 0.0 — 0.0 m means "surface",
   which is a claim, and stratification is one of the two remaining falsification causes.
4. **Rows with no usable time or no timezone are REJECTED, not defaulted.** Each sample's value
   comes from resolving its timestamp to a tidal phase; a fabricated noon would be a fabricated
   phase. Losing 3 of 123 is the right trade.

- [ ] **Step 1: Create the fixture from real data**

Run this to capture a verbatim excerpt (network; the endpoint is public and needs no key):

```bash
cd /Users/ellismillwood/Documents/tidescout
curl -sS --max-time 180 -G "https://www.waterqualitydata.us/data/Result/search" \
  --data-urlencode "siteid=21SC60WQ_WQX-WB-06" \
  --data-urlencode "siteid=21SC60WQ_WQX-WB-05" \
  --data-urlencode "siteid=21SCSHL_WQX-05-24" \
  --data-urlencode "characteristicName=Salinity" \
  --data-urlencode "mimeType=csv" --data-urlencode "zip=no" \
  -o backend/tests/fixtures/wqp_results_excerpt.csv
head -1 backend/tests/fixtures/wqp_results_excerpt.csv | tr ',' '\n' | wc -l   # expect 63
wc -l backend/tests/fixtures/wqp_results_excerpt.csv                          # expect ~95
```

Then hand-append these three synthetic rows to exercise the rejection paths. They MUST have the
same 63-column shape; the easiest correct way is to copy the last real row and edit fields. Add
them with a short comment in the test, not in the CSV (CSV comments are not portable):

- one row with `ResultMeasure/MeasureUnitCode` = `uS/cm`
- one row with `ActivityTypeCode` = `Quality Control Sample-Field Blank`
- one row with `ActivityStartTime/Time` and `ActivityStartTime/TimeZoneCode` both empty

- [ ] **Step 2: Write the failing tests**

```python
"""WQP salinity parsing, against a verbatim excerpt of a real response.

Task 9's CDMO parser was built from documentation and four of its inferences
were wrong, one catastrophically (whole-file header routing would have sent
every station to the MET parser and destroyed every salinity reading). No
parser in this repo is trusted until it has met a real file.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tidescout.sources import wqp

FIXTURE = Path(__file__).parent / "fixtures" / "wqp_results_excerpt.csv"


def _report():
    return wqp.parse_results(FIXTURE.read_text().splitlines())


def test_parses_real_samples():
    r = _report()
    assert r.n_admitted > 50, "the real excerpt holds ~90 admitted rows"
    assert all(s.salinity_psu >= 0 for s in r.samples)
    assert {s.station for s in r.samples} <= {
        "21SC60WQ_WQX-WB-06", "21SC60WQ_WQX-WB-05", "21SCSHL_WQX-05-24",
    }


def test_timestamps_are_utc_aware():
    r = _report()
    assert r.samples, "need at least one sample"
    for s in r.samples:
        assert s.ts.tzinfo is not None, "naive timestamps silently shift the tidal phase"
        assert s.ts.utcoffset() == datetime.now(UTC).utcoffset()


def test_edt_is_converted_not_assumed_utc():
    """11:35 EDT is 15:35 UTC. Treating local time as UTC would put the
    sample four hours off, which at a 12.42 h tidal period is most of a
    quarter cycle -- the difference between flood and ebb."""
    rows = [
        "MonitoringLocationIdentifier,CharacteristicName,ResultMeasureValue,"
        "ResultMeasure/MeasureUnitCode,ActivityStartDate,ActivityStartTime/Time,"
        "ActivityStartTime/TimeZoneCode,ActivityTypeCode,ResultStatusIdentifier,"
        "ResultDetectionConditionText,ActivityDepthHeightMeasure/MeasureValue,"
        "ActivityDepthHeightMeasure/MeasureUnitCode",
        "S1,Salinity,20.0,ppt,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,",
    ]
    r = wqp.parse_results(rows)
    assert r.samples[0].ts == datetime(2014, 6, 12, 15, 35, tzinfo=UTC)


def test_per_mille_and_ppt_are_both_admitted_unchanged():
    """`0/00` is per-mille, numerically identical to ppt. Both are real in
    the live data (81 ppt rows, 42 `0/00` rows in one response)."""
    head = (
        "MonitoringLocationIdentifier,CharacteristicName,ResultMeasureValue,"
        "ResultMeasure/MeasureUnitCode,ActivityStartDate,ActivityStartTime/Time,"
        "ActivityStartTime/TimeZoneCode,ActivityTypeCode,ResultStatusIdentifier,"
        "ResultDetectionConditionText,ActivityDepthHeightMeasure/MeasureValue,"
        "ActivityDepthHeightMeasure/MeasureUnitCode"
    )
    a = wqp.parse_results([head, "S1,Salinity,20.0,ppt,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,"])
    b = wqp.parse_results([head, "S1,Salinity,20.0,0/00,2014-06-12,11:35:00,EDT,Sample-Routine,Final,,,"])
    assert a.samples[0].salinity_psu == b.samples[0].salinity_psu == 20.0


def test_an_unknown_unit_is_rejected_and_counted_never_coerced():
    r = _report()
    assert r.n_bad_unit >= 1
    assert "uS/cm" in r.unknown_units
    assert all(s.salinity_psu <= 45.0 for s in r.samples), "a uS/cm value would be ~4 digits"


def test_quality_control_activities_are_excluded():
    """Field blanks and lab replicates pass every other filter and would
    enter the fit as real estuary measurements."""
    r = _report()
    assert r.n_qc_activity >= 1


def test_a_row_with_no_time_is_rejected_not_defaulted():
    """Every sample's worth comes from resolving its timestamp to a tidal
    phase. A fabricated noon is a fabricated phase."""
    r = _report()
    assert r.n_no_time >= 1


def test_missing_depth_is_none_not_zero():
    """0.0 m means SURFACE, which is a claim. Depth is missing in 120 of 123
    real rows, and stratification is a measured +3.30 ppt at one distance."""
    r = _report()
    assert any(s.depth_m is None for s in r.samples)
    assert all(s.depth_m is None or s.depth_m > 0 for s in r.samples)


def test_counters_account_for_every_row():
    """A rejection path with no counter is a silent drop."""
    r = _report()
    accounted = (
        r.n_admitted + r.n_no_time + r.n_bad_unit
        + r.n_bad_status + r.n_qc_activity + r.n_no_value
    )
    assert accounted == r.n_rows
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_wqp.py -q`
Expected: `ModuleNotFoundError: No module named 'tidescout.sources.wqp'`

- [ ] **Step 4: Implement the parser**

```python
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

# The WQP characteristic this module reads, and the ONLY one it will read.
# WQP also serves "Specific conductance" and "Conductivity"; `sources/usgs.py`
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
    pins their sum to `n_rows`. A rejection with no counter is a silent drop,
    and silent drops are how a fit quietly narrows its own inputs while
    looking complete.
    """

    samples: list[Sample] = field(default_factory=list)
    n_rows: int = 0
    n_admitted: int = 0
    n_no_time: int = 0
    n_bad_unit: int = 0
    n_bad_status: int = 0
    n_qc_activity: int = 0
    n_no_value: int = 0
    unknown_units: dict[str, int] = field(default_factory=dict)
    unknown_statuses: dict[str, int] = field(default_factory=dict)


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _timestamp(date_s: str, time_s: str, tz_s: str) -> datetime | None:
    """Local clock time + US tz abbreviation -> UTC. None if unusable."""
    date_s, time_s, tz_s = date_s.strip(), time_s.strip(), tz_s.strip().upper()
    if not date_s or not time_s or tz_s not in _TZ_OFFSETS:
        return None
    try:
        naive = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    tz = timezone(timedelta(hours=_TZ_OFFSETS[tz_s]))
    return naive.replace(tzinfo=tz).astimezone(UTC)


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
        if (row.get("CharacteristicName") or "").strip() != CHARACTERISTIC:
            continue  # a different parameter entirely; not this module's business
        report.n_rows += 1

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
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_wqp.py -q`
Expected: all PASS.

- [ ] **Step 6: Run the full suite and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/sources/wqp.py backend/tests/test_wqp.py backend/tests/fixtures/wqp_results_excerpt.csv
git commit -m "feat: parse Water Quality Portal salinity results"
```

---

## Task 2: Fetch WQP and store it

**Files:**
- Modify: `backend/tidescout/sources/wqp.py`
- Modify: `backend/tidescout/cli.py`
- Modify: `backend/tests/test_wqp.py`

**Interfaces:**
- Consumes: `Sample`, `ParseReport`, `parse_results` (Task 1); `NdbcStore`, `Observation`,
  `BulkWriter` from `tidescout.sources.ndbc`.
- Produces:
  - `WQP_URL: str`, `WQP_STATION_URL: str`, `SOURCE_WQP: str` (value `"wqp:salinity"`)
  - `default_store(slug: str) -> NdbcStore` — the WQP store, at `data/{slug}/wqp.sqlite`
  - `fetch_results(bbox: tuple[float, float, float, float], timeout: float = 180.0) -> str`
  - `fetch_stations(bbox: tuple[float, float, float, float], timeout: float = 180.0) -> str`
  - `parse_stations(fh: Iterable[str]) -> dict[str, tuple[float, float]]` — `{id: (lon, lat)}`
  - `import_results(text: str, store: NdbcStore, stations_csv: str | None = None) -> ImportReport`
  - `station_coords_from_store(store: NdbcStore) -> dict[str, tuple[float, float]]` —
    `{id: (lon, lat)}` read back from the `wqp_stations` table
  - `station_coords(slug: str) -> dict[str, tuple[float, float]]` — the same, opening
    this fishery's default store; returns `{}` when the store does not exist yet
  - `ImportReport` dataclass: `parse: ParseReport`, `n_new: int`, `stations: tuple[str, ...]`,
    `span: tuple[datetime, datetime] | None`
- Also modifies `NdbcStore` (in `sources/ndbc.py`): adds `self._db_path = db_path` in
  `__init__` and a public `stations() -> list[str]` method. Both are needed by callers here and
  neither exists today — `__init__` currently drops `db_path` after opening the connection, and
  the station list is only reachable through the private `_conn`.

**Station coordinates need a home.** WQP stations are DISCOVERED from a bbox query, not
authored in the fishery YAML — there are ~130 and which are usable is decided by the stem screen.
Their positions therefore have to be stored alongside their readings. `import_results` writes a
`wqp_stations (station TEXT PRIMARY KEY, lon REAL, lat REAL)` table into the same sqlite file, in
the same transaction as the observations, and `station_coords(slug)` reads it back. Positions in
one file with the readings they belong to; no second artefact to keep in step.

**Why a SEPARATE sqlite file rather than the NERRS one.** `sources/ndbc.py`'s module docstring
states as an invariant that its store holds only North Inlet-Winyah Bay NERR data, and
`NdbcStore.citation()` builds a NERRS citation for everything it holds. Writing WQP rows into it
would make that citation overclaim — attributing SCDHEC and EPA data to NERRS. Reusing the CLASS
against a different FILE keeps the proven dedupe contract and the invariant both intact. Task 3
then fixes `citation()` so the WQP store gets correct attribution of its own.

- [ ] **Step 1: Write the failing tests**

```python
def test_store_is_a_separate_file_from_the_nerrs_one(tmp_path, monkeypatch):
    """ndbc.py states as an invariant that its store holds only NIW NERR
    data, and its citation() builds a NERRS citation for everything in it.
    WQP rows in that file would attribute SCDHEC and EPA data to NERRS."""
    from tidescout.sources import ndbc, wqp

    monkeypatch.setattr("tidescout.paths.fishery_data_dir", lambda slug: tmp_path)
    monkeypatch.setattr("tidescout.sources.ndbc.fishery_data_dir", lambda slug: tmp_path)
    monkeypatch.setattr("tidescout.sources.wqp.fishery_data_dir", lambda slug: tmp_path)

    assert wqp.default_store("winyah-bay")._db_path != ndbc.default_store("winyah-bay")._db_path
    assert wqp.default_store("winyah-bay")._db_path.name == "wqp.sqlite"


def test_station_coordinates_are_stored_with_their_readings(tmp_path):
    """WQP stations are discovered from a bbox query, not authored, so their
    positions must travel with their readings rather than living in a second
    artefact that can drift out of step."""
    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    stations_csv = (
        "MonitoringLocationIdentifier,LatitudeMeasure,LongitudeMeasure\n"
        "21SC60WQ_WQX-WB-06,33.2795,-79.2210\n"
    )
    wqp.import_results(FIXTURE.read_text(), store, stations_csv=stations_csv)
    coords = wqp.station_coords_from_store(store)
    assert coords["21SC60WQ_WQX-WB-06"] == pytest.approx((-79.2210, 33.2795))


def test_import_is_idempotent(tmp_path):
    """Re-running an import must add nothing -- the append-and-dedupe
    contract cdmo.py proved at 2.5 M rows, reused unchanged."""
    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    text = FIXTURE.read_text()
    first = wqp.import_results(text, store)
    second = wqp.import_results(text, store)
    assert first.n_new > 0
    assert second.n_new == 0


def test_import_records_provenance_under_its_own_source(tmp_path):
    """Attribution depends on distinguishing WQP rows from NERRS rows, and
    provenance.source is the only field that can carry it."""
    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    wqp.import_results(FIXTURE.read_text(), store)
    sources = {p.source for p in store.provenance()}
    assert sources == {wqp.SOURCE_WQP}


def test_fetch_builds_a_bbox_query_without_a_key(monkeypatch):
    """The endpoint is public and unauthenticated -- unlike CDMO, which
    needs a static IP registered through a human-submitted form."""
    from tidescout.sources import wqp

    seen = {}

    class _Resp:
        text = "CharacteristicName\n"

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, timeout=None):
        seen["url"] = url
        seen["params"] = params
        return _Resp()

    monkeypatch.setattr(wqp.httpx, "get", fake_get)
    wqp.fetch_results((-79.45, 33.15, -79.05, 33.60))

    assert seen["params"]["bBox"] == "-79.45,33.15,-79.05,33.60"
    assert seen["params"]["characteristicName"] == "Salinity"
    assert "key" not in seen["params"] and "apiKey" not in seen["params"]
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_wqp.py -q`
Expected: FAIL with `AttributeError: module 'tidescout.sources.wqp' has no attribute 'default_store'`

- [ ] **Step 3: Implement fetch and import**

Append to `backend/tidescout/sources/wqp.py`:

```python
import httpx

from tidescout.paths import fishery_data_dir
from tidescout.sources.ndbc import NdbcStore, Observation

WQP_URL = "https://www.waterqualitydata.us/data/Result/search"

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


def fetch_results(
    bbox: tuple[float, float, float, float], timeout: float = 180.0
) -> str:
    """Raw CSV for every salinity result in `bbox`. Public, no key.

    `bbox` is (lon_min, lat_min, lon_max, lat_max) -- the same order
    `Fishery.bbox` uses, so a caller passes it straight through.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "bBox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "characteristicName": CHARACTERISTIC,
        "mimeType": "csv",
        "zip": "no",
    }
    resp = httpx.get(WQP_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.text


@dataclass
class ImportReport:
    parse: ParseReport
    n_new: int
    stations: tuple[str, ...]
    span: tuple[datetime, datetime] | None


def import_results(text: str, store: NdbcStore) -> ImportReport:
    """Parse and store, whole-batch-atomic, deduped by (station, ts)."""
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
        # Inside the transaction, so a rollback takes the provenance row with
        # it -- a failed import must leave no record of data it never wrote.
        writer.record_provenance(SOURCE_WQP, sorted(by_station), span, n_new)
    return ImportReport(report, n_new, tuple(sorted(by_station)), span)
```

- [ ] **Step 4: Add the CLI command**

In `backend/tidescout/cli.py`, beside the other `salinity_app` commands:

```python
@salinity_app.command("import-wqp")
def salinity_import_wqp(slug: str) -> None:
    """Fetch Water Quality Portal salinity for this fishery and store it.

    The anchors the fit never had: 55 in-domain stations in the 2.58-13.05 km
    reach that previously held no salinity observation at all.
    """
    from tidescout.config import load_fishery
    from tidescout.sources import wqp

    fishery = load_fishery(slug)
    store = wqp.default_store(slug)
    text = wqp.fetch_results(tuple(fishery.bbox))
    rep = wqp.import_results(text, store)

    table = Table(title=f"{fishery.name} — Water Quality Portal salinity")
    for col in ("metric", "count"):
        table.add_column(col)
    p = rep.parse
    for label, n in [
        ("rows seen", p.n_rows),
        ("admitted", p.n_admitted),
        ("new to the store", rep.n_new),
        ("rejected — no usable time", p.n_no_time),
        ("rejected — unknown unit", p.n_bad_unit),
        ("rejected — unreviewed status", p.n_bad_status),
        ("rejected — QC activity", p.n_qc_activity),
        ("rejected — no value", p.n_no_value),
    ]:
        table.add_row(label, f"{n:,}")
    console.print(table)
    if p.unknown_units:
        console.print(f"[yellow]unknown units seen:[/yellow] {p.unknown_units}")
    if p.unknown_statuses:
        console.print(f"[yellow]unknown statuses seen:[/yellow] {p.unknown_statuses}")
    console.print(f"{len(rep.stations)} station(s); span {rep.span}")
```

- [ ] **Step 5: Run tests, then run it for real**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
$HOME/.venvs/tidescout/bin/tidescout salinity import-wqp winyah-bay
```

Expected: several thousand rows admitted across ~130 stations. Re-run it; "new to the store"
must be 0 the second time.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/sources/wqp.py backend/tidescout/cli.py backend/tests/test_wqp.py
git commit -m "feat: fetch and store Water Quality Portal salinity"
```

---

## Task 3: Make the citation source-aware

**Files:**
- Modify: `backend/tidescout/sources/ndbc.py:719-766` (`NdbcStore.citation`)
- Modify: `backend/tests/test_ndbc.py`

**Interfaces:**
- Consumes: `ProvenanceRecord.source`, `SOURCE_WQP` (Task 2), `Citation`.
- Produces: `Citation.text` becomes a per-source block; new module constant
  `WQP_ATTRIBUTION: str`.

**Why this is a correctness fix, not a nicety.** Task 10 established that the citation must be
GENERATED from what the store actually holds, never hardcoded, precisely so it cannot overclaim
or underclaim. `citation()` currently formats `NERRS_CITATION_TEMPLATE` unconditionally. Point it
at a WQP store and it asserts that SCDHEC and EPA data came from NERRS — the exact failure that
ruling existed to prevent.

- [ ] **Step 1: Write the failing test**

```python
def test_citation_names_the_sources_the_store_actually_holds(tmp_path):
    """Task 10's ruling was that the citation is GENERATED from the store so
    it cannot overclaim. A hardcoded NERRS template does exactly that the
    moment the store holds anything else."""
    from datetime import UTC, datetime

    from tidescout.sources import ndbc, wqp

    store = ndbc.NdbcStore(tmp_path / "wqp.sqlite")
    store.append("WB-06", [ndbc.Observation(
        ts=datetime(2014, 6, 12, 15, 35, tzinfo=UTC), depth_m=None, water_temp_c=None,
        cond_ms_cm=None, salinity_psu=20.0, o2_pct=None, o2_ppm=None,
        chlorophyll_ug_l=None, turbidity_ftu=None, ph=None, eh_mv=None)])
    store.record_provenance(wqp.SOURCE_WQP, ["WB-06"], None, 1)

    cit = store.citation()
    assert "Water Quality Portal" in cit.text
    assert "National Estuarine Research Reserve" not in cit.text, (
        "a WQP-only store must not claim NERRS provenance"
    )


def test_a_nerrs_store_still_gets_the_nerrs_citation(tmp_path):
    from datetime import UTC, datetime

    from tidescout.sources import ndbc

    store = ndbc.NdbcStore(tmp_path / "ndbc.sqlite")
    store.append("WYSS1", [ndbc.Observation(
        ts=datetime(2026, 8, 23, 12, 0, tzinfo=UTC), depth_m=None, water_temp_c=None,
        cond_ms_cm=None, salinity_psu=10.5, o2_pct=None, o2_ppm=None,
        chlorophyll_ug_l=None, turbidity_ftu=None, ph=None, eh_mv=None)])
    store.record_provenance("ndbc:realtime2", ["WYSS1"], None, 1)

    assert "doi:10.25921/vw8a-8031" in store.citation().text
```

- [ ] **Step 2: Run and watch fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_ndbc.py -q -k citation`
Expected: FAIL — `"Water Quality Portal" in cit.text` is False.

- [ ] **Step 3: Implement**

Add near `NERRS_CITATION_TEMPLATE` in `backend/tidescout/sources/ndbc.py`:

```python
# Attribution for Water Quality Portal holdings. WQP is an aggregator: the
# data belongs to the contributing organisation (SC DES, SCDHEC, EPA), and
# WQP itself asks to be named as the access route. `{date}` is the only part
# this codebase fills in, matching NERRS_CITATION_TEMPLATE's contract.
WQP_ATTRIBUTION = (
    "Water quality data accessed from the Water Quality Portal "
    "(https://www.waterqualitydata.us), a cooperative service of the U.S. "
    "Environmental Protection Agency, the U.S. Geological Survey and the "
    "National Water Quality Monitoring Council; accessed {date}. Data are "
    "contributed by the originating monitoring organisations, which retain "
    "credit for having collected them."
)

# Provenance `source` prefix -> the citation block that source requires.
# Keyed on prefix so "cdmo:water_quality" and "cdmo:meteorological" share one
# entry without listing every variant.
_CITATION_BY_SOURCE_PREFIX: tuple[tuple[str, str], ...] = (
    ("wqp:", WQP_ATTRIBUTION),
    ("ndbc:", NERRS_CITATION_TEMPLATE),
    ("cdmo:", NERRS_CITATION_TEMPLATE),
)
```

Then in `citation()`, replace the single line

```python
        text = NERRS_CITATION_TEMPLATE.format(date=date_str)
```

with

```python
        # One block per source family actually present. Task 10's ruling was
        # that this is GENERATED from the store so it can neither overclaim
        # nor underclaim; a fixed NERRS template overclaims the moment the
        # store holds anything else.
        templates: list[str] = []
        for prefix, template in _CITATION_BY_SOURCE_PREFIX:
            if any(s.startswith(prefix) for s in sources) and template not in templates:
                templates.append(template)
        if not templates:
            templates = [NERRS_CITATION_TEMPLATE]
        text = "\n\n".join(t.format(date=date_str) for t in templates)
```

- [ ] **Step 4: Run tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/sources/ndbc.py backend/tests/test_ndbc.py
git commit -m "fix: generate the citation from the sources the store holds"
```

---

## Task 4: Main stem and distance-to-stem

**Files:**
- Modify: `backend/tidescout/pipeline/estuary.py`
- Modify: `backend/tidescout/cli.py`
- Modify: `backend/tests/test_estuary.py`

**Interfaces:**
- Consumes: `along_estuary_km`, `load_distance_field`, `to_grid`, `from_grid`, `grid_spec`.
- Produces:
  - `descent_path(field_grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]`
  - `main_stem_mask(slug: str, fishery: Fishery, spec, field: np.ndarray) -> np.ndarray` (bool, flat layout)
  - `build_stem_distance_field(slug: str, fishery: Fishery) -> Path` (writes `stem_km.npy`)
  - `load_stem_distance_field(slug: str) -> np.ndarray`
  - `ON_AXIS_MAX_KM: float` (value `2.0`)

**The measured basis for this task — do not re-derive it, but DO re-verify it in Step 5.**
Prototyped against the real field on 2026-08-24:

| station | distance to stem |
|---|---|
| WQP `WB-06` (main channel, 5.56 km) | 0.048 km |
| WQP `05-24` Coast Guard Range | 0.200 |
| WQP `05-21` Buoy 17 Range | 0.283 |
| WQP `WB-05` (main channel, 10.28 km) | 0.537 |
| NERRS `NIWTAWQ` (bay) | 0.651 |
| NERRS `WYSS1` / `NIWWBWQ` (bay) | 1.393 |
| WQP `05-25` W. Channel Island | 1.604 |
| *WQP `05-07` Jones Ck / Mud Bay* | *2.170* |
| WQP `06A-03` North Santee River | 7.798 |
| NERRS `NIWCBWQ` (North Inlet) | 7.997 |
| WQP `05-03` North Inlet | 8.768 |
| WQP `05-04` Town Creek | 9.475 |
| WQP `06A-11` AIWW Minum Creek | 9.602 |
| NERRS `NIWOLWQ` (North Inlet) | 9.858 |
| NERRS `NIWDCWQ` (North Inlet) | 11.918 |

A 4.8x gap between 1.604 and 7.798, with only Jones Creek/Mud Bay between. `ON_AXIS_MAX_KM = 2.0`
sits in that gap. Jones Creek landing outside is CORRECT, not a miss: Mud Bay is the physical
connection between the two systems, so its membership is genuinely ambiguous.

An earlier metric — along-path distance until the descent joins the stem — was tried and REJECTED.
It scored the bay's own NERRS stations at 10-14 km, because a station beside a one-cell-wide stem
descends roughly parallel to it and only converges near the mouth. Distance TO the stem is the
right question, and it is the same Dijkstra `along_estuary_km` already runs, seeded differently.

- [ ] **Step 1: Write the failing tests**

```python
# -- Main stem and branch membership ----------------------------------------


def _branch_spec():
    """A main channel with one long side creek joining it near the mouth."""
    mask = np.zeros((20, 20), bool)
    mask[2:15, 3] = True   # main channel, north-south, mouth at the south end
    mask[13, 3:12] = True  # a side creek joining low down
    return _Spec(mask)


def test_descent_path_runs_downhill_to_a_seed():
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    seeds = (rows == 14) & (cols == 3)
    d = estuary.along_estuary_km(spec, seeds)
    grid = to_grid(d, spec.flat_index, spec.shape, fill=np.nan)

    path = estuary.descent_path(grid, (2, 3))

    assert path[0] == (2, 3)
    assert path[-1] == (14, 3), "must terminate at the seed"
    assert len(path) == 13


def test_distance_to_stem_separates_a_side_creek_from_the_channel():
    """The head of a side creek is far from the stem THROUGH WATER even
    when it is close in a straight line."""
    spec = _branch_spec()
    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    stem = (cols == 3) & (rows >= 2) & (rows <= 14)

    to_stem = estuary.along_estuary_km(spec, stem)

    on_channel = (rows == 8) & (cols == 3)
    creek_head = (rows == 13) & (cols == 11)
    assert to_stem[on_channel][0] == pytest.approx(0.0)
    assert to_stem[creek_head][0] == pytest.approx(0.8)  # 8 cells of 100 m


@pytest.mark.parametrize(
    ("station", "expect_off_axis"),
    [
        ("NIWTAWQ", False), ("WYSS1", False), ("NIWWBWQ", False),
        ("NIWCBWQ", True), ("NIWOLWQ", True), ("NIWDCWQ", True),
    ],
)
def test_the_six_nerrs_stations_land_on_the_right_side_of_the_screen(
    station, expect_off_axis
):
    """Regression against the real built field, so a future threshold change
    cannot silently re-admit North Inlet to a Winyah Bay fit.

    Skips rather than fails when the field has not been built -- a fresh
    clone has no data/ directory, and this asserts about real geometry.
    """
    import numpy as np
    from rasterio.warp import transform as warp_transform

    from tidescout.config import load_fishery
    from tidescout.pipeline.flowlib import grid_spec
    from tidescout.sources import cdmo

    fishery = load_fishery("winyah-bay")
    try:
        stem = estuary.load_stem_distance_field("winyah-bay")
    except FileNotFoundError:
        pytest.skip("stem field not built -- run `tidescout salinity stem winyah-bay`")

    spec = grid_spec("winyah-bay", fishery)
    lon, lat = cdmo.NIW_STATION_COORDS_LONLAT[station]
    x, y = (v[0] for v in warp_transform(
        "EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", [lon], [lat]))
    i = int(np.argmin((spec.xs - x) ** 2 + (spec.ys - y) ** 2))

    assert (float(stem[i]) > estuary.ON_AXIS_MAX_KM) is expect_off_axis


def test_on_axis_threshold_sits_in_the_measured_gap():
    """Measured on the real field 2026-08-24: on-axis stations span
    0.048-1.604 km to the stem and off-axis ones 7.798-11.918, with only
    Jones Creek / Mud Bay (2.170) between. The threshold must sit in that
    gap, not on either shoulder."""
    assert 1.604 < estuary.ON_AXIS_MAX_KM < 2.170
```

- [ ] **Step 2: Run and watch fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_estuary.py -q`
Expected: FAIL — `module 'tidescout.pipeline.estuary' has no attribute 'descent_path'`

- [ ] **Step 3: Implement**

Add to `backend/tidescout/pipeline/estuary.py`:

```python
# Distance to the estuary's main stem, THROUGH WATER, above which a station
# is treated as sitting on a different branch. Measured 2026-08-24 against the
# real field: on-axis stations (Winyah Bay main channel plus the bay's own
# NERRS sondes) span 0.048-1.604 km, off-axis ones (North Inlet, Town Creek,
# the AIWW, the North Santee) span 7.798-11.918 -- a 4.8x gap with only Jones
# Creek / Mud Bay (2.170) inside it. This value sits in that gap.
#
# Jones Creek falling OUTSIDE is correct rather than a miss: Mud Bay is the
# physical connection between Winyah Bay and North Inlet, so its membership is
# genuinely ambiguous and the conservative answer is to leave it out of a fit
# that assumes one branch.
ON_AXIS_MAX_KM = 2.0

_STEM_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def descent_path(field_grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Steepest-descent cells from `start` down to a seed, on a 2-D field.

    Follows the same 8-connectivity `along_estuary_km` used to BUILD the
    field, so the path it traces is one the distance actually measures along.
    Stops when no neighbour is strictly lower, which at a seed cell (distance
    0.0) is immediate.
    """
    r, c = start
    out = [(r, c)]
    rows, cols = field_grid.shape
    while True:
        best = None
        for dr, dc in _STEM_NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and not np.isnan(field_grid[rr, cc]):
                if best is None or field_grid[rr, cc] < best[0]:
                    best = (field_grid[rr, cc], rr, cc)
        if best is None or best[0] >= field_grid[r, c]:
            return out
        _, r, c = best
        out.append((r, c))


def main_stem_mask(slug: str, fishery: Fishery, spec, field: np.ndarray) -> np.ndarray:
    """The estuary's main channel: the union of the descent paths from each
    river inflow point down to the mouth.

    The inflow points are already authored in the fishery YAML (they are where
    `Inlet_operator` injects discharge), so this adds no new hand-authored
    geometry. Walking downhill from each one traces the channel the river
    water actually takes, which is the line the salt front advances along.
    """
    from rasterio.warp import transform as warp_transform

    grid = to_grid(field, spec.flat_index, spec.shape, fill=np.nan)
    lons = [r.inflow_lonlat[0] for r in fishery.rivers]
    lats = [r.inflow_lonlat[1] for r in fishery.rivers]
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", lons, lats)

    stem = np.zeros(spec.shape, dtype=bool)
    for x, y in zip(xs, ys, strict=True):
        d2 = (spec.xs - x) ** 2 + (spec.ys - y) ** 2
        i = int(np.argmin(d2))
        start = np.unravel_index(spec.flat_index[i], spec.shape)
        for rr, cc in descent_path(grid, (int(start[0]), int(start[1]))):
            stem[rr, cc] = True
    return from_grid(stem, spec.flat_index)


def build_stem_distance_field(slug: str, fishery: Fishery) -> Path:
    """Distance from every cell to the main stem, through water.

    The SAME Dijkstra as the along-estuary field, seeded from the stem
    instead of the ocean. Reused rather than reimplemented so the two fields
    cannot drift apart in connectivity or diagonal cost.
    """
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    field = load_distance_field(slug)
    stem = main_stem_mask(slug, fishery, spec, field)
    d = along_estuary_km(spec, stem)
    path = fishery_data_dir(slug) / "stem_km.npy"
    np.save(path, d.astype("float32"))
    return path


def load_stem_distance_field(slug: str) -> np.ndarray:
    path = fishery_data_dir(slug) / "stem_km.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"no distance-to-stem field at {path} -- run "
            f"`tidescout salinity stem {slug}` first"
        )
    return np.load(path)
```

Add the import at the top of the file if not already present:
`from tidescout.engine.structure import from_grid, to_grid` (it already imports both).

- [ ] **Step 4: Add the CLI command**

```python
@salinity_app.command("stem")
def salinity_stem(slug: str) -> None:
    """Build the distance-to-main-stem field used to screen off-branch stations."""
    from tidescout.config import load_fishery
    from tidescout.pipeline.estuary import build_stem_distance_field, load_stem_distance_field

    fishery = load_fishery(slug)
    path = build_stem_distance_field(slug, fishery)
    d = load_stem_distance_field(slug)
    console.print(f"distance-to-stem field -> {path}")
    console.print(
        f"min {float(d.min()):.2f}  median {float(np.median(d)):.2f}  "
        f"max {float(d.max()):.2f} km; on-axis threshold "
        f"{__import__('tidescout.pipeline.estuary', fromlist=['x']).ON_AXIS_MAX_KM} km"
    )
```

(Import `numpy as np` at the top of `cli.py` if it is not already imported.)

- [ ] **Step 5: Build it for real and RE-VERIFY the table above**

```bash
cd /Users/ellismillwood/Documents/tidescout
$HOME/.venvs/tidescout/bin/tidescout salinity stem winyah-bay
```

Then check the six NERRS stations against the measured table:

```bash
cd backend && $HOME/.venvs/tidescout/bin/python -c "
import numpy as np
from rasterio.warp import transform as warp_transform
from tidescout.config import load_fishery
from tidescout.pipeline.estuary import load_stem_distance_field, ON_AXIS_MAX_KM
from tidescout.pipeline.flowlib import grid_spec
from tidescout.sources import cdmo
f=load_fishery('winyah-bay'); spec=grid_spec('winyah-bay',f); ts=load_stem_distance_field('winyah-bay')
for s,(lon,lat) in sorted(cdmo.NIW_STATION_COORDS_LONLAT.items()):
    x,y=[v[0] for v in warp_transform('EPSG:4326',f'EPSG:{f.bathymetry.epsg}',[lon],[lat])]
    i=int(np.argmin((spec.xs-x)**2+(spec.ys-y)**2))
    print(f'{s:9} to_stem {ts[i]:7.3f} km  ->  {\"ON\" if ts[i]<=ON_AXIS_MAX_KM else \"OFF\"}-axis')
" 2>&1 | grep -v mpi4py
```

Expected, and this is a HARD GATE — if it does not match, stop and report rather than adjusting
the threshold to fit:

```
NIWCBWQ   to_stem   7.997 km  ->  OFF-axis
NIWDCWQ   to_stem  11.918 km  ->  OFF-axis
NIWOLWQ   to_stem   9.858 km  ->  OFF-axis
NIWTAWQ   to_stem   0.651 km  ->  ON-axis
NIWWBWQ   to_stem   1.393 km  ->  ON-axis
WYSS1     to_stem   1.393 km  ->  ON-axis
```

- [ ] **Step 6: Commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/pipeline/estuary.py backend/tidescout/cli.py backend/tests/test_estuary.py
git commit -m "feat: derive branch membership from distance to the estuary's main stem"
```

---

## Task 5: Derive `off_axis`, and feed WQP into the fit

**Files:**
- Modify: `backend/tidescout/pipeline/salinity_fit.py`
- Modify: `backend/tidescout/models.py:24-45` (`WaterSensor.off_axis` docstring)
- Modify: `backend/tests/test_salinity.py`

**Interfaces:**
- Consumes: `load_stem_distance_field`, `ON_AXIS_MAX_KM` (Task 4); `wqp.default_store` (Task 2);
  `daily_means_and_swings`, `build_site_record`, `site_distances_km` (existing).
- Produces:
  - `station_stem_km(slug: str, fishery: Fishery, sites: Mapping[str, tuple[float, float]]) -> dict[str, float]`
  - `is_off_axis(stem_km: float, declared: bool) -> bool`
  - `CalibrationInput` gains `n_off_axis: int`

**The rule.** The computed value decides, and the YAML `off_axis: true` becomes an OVERRIDE that
can only ever EXCLUDE, never include. A hand flag that could force a station back in would
reintroduce exactly the hand-marking this task exists to remove; a hand flag that can only
exclude is a safety valve for geometry the criterion gets wrong.

- [ ] **Step 1: Write the failing tests**

```python
def test_off_axis_is_computed_from_the_stem_distance():
    """Measured 2026-08-24: North Inlet's stations sit 7.997-11.918 km from
    the main stem and the bay's own sit 0.651-1.393."""
    assert salinity_fit.is_off_axis(stem_km=9.858, declared=False) is True
    assert salinity_fit.is_off_axis(stem_km=0.651, declared=False) is False


def test_the_yaml_flag_can_only_exclude_never_include():
    """A hand flag that could force a station back IN would reintroduce the
    hand-marking this replaces. One that can only exclude is a safety valve."""
    assert salinity_fit.is_off_axis(stem_km=0.05, declared=True) is True
    assert salinity_fit.is_off_axis(stem_km=9.9, declared=False) is True


def test_a_station_with_no_stem_distance_is_excluded_not_admitted():
    """NaN means the cell has no water route to the stem. Admitting it would
    put an unplaceable station into the fit."""
    assert salinity_fit.is_off_axis(stem_km=float("nan"), declared=False) is True
```

- [ ] **Step 2: Run and watch fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_salinity.py -q -k off_axis`
Expected: FAIL — `module 'tidescout.pipeline.salinity_fit' has no attribute 'is_off_axis'`

- [ ] **Step 3: Implement**

Add to `backend/tidescout/pipeline/salinity_fit.py`:

```python
def is_off_axis(stem_km: float, declared: bool) -> bool:
    """Whether a station sits on a branch the 1-D coordinate cannot place.

    The COMPUTED distance decides; `declared` (the fishery YAML's
    `off_axis: true`) is an override that can only ever EXCLUDE. A hand flag
    able to force a station back IN would reintroduce the hand-marking this
    replaces -- and hand-marking 132 WQP stations is precisely how the
    43-point score ambiguity got in unnoticed. One that can only exclude is a
    safety valve for geometry the criterion gets wrong.

    NaN excludes: it means the cell has no water route to the stem at all.
    """
    from tidescout.pipeline.estuary import ON_AXIS_MAX_KM

    if declared:
        return True
    if not np.isfinite(stem_km):
        return True
    return stem_km > ON_AXIS_MAX_KM


def station_stem_km(
    slug: str, fishery: Fishery, sites: Mapping[str, tuple[float, float]]
) -> dict[str, float]:
    """Distance from each {site: (lon, lat)} to the estuary's main stem."""
    from rasterio.warp import transform as warp_transform

    from tidescout.pipeline.estuary import load_stem_distance_field
    from tidescout.pipeline.flowlib import grid_spec

    if not sites:
        return {}
    spec = grid_spec(slug, fishery)
    stem = load_stem_distance_field(slug)
    ids = sorted(sites)
    xs, ys = warp_transform(
        "EPSG:4326",
        f"EPSG:{fishery.bathymetry.epsg}",
        [sites[s][0] for s in ids],
        [sites[s][1] for s in ids],
    )
    out = {}
    for site, x, y in zip(ids, xs, ys, strict=True):
        i = int(np.argmin((spec.xs - x) ** 2 + (spec.ys - y) ** 2))
        out[site] = float(stem[i])
    return out
```

Then in `collect_observations`, replace the line

```python
            off_axis=w.off_axis,
```

with

```python
            off_axis=is_off_axis(store_stem.get(w.station, float("nan")), w.off_axis),
```

and compute `store_stem = station_stem_km(slug, fishery, coords)` alongside `store_dist`, where
`coords` is the same `{station: (lon, lat)}` mapping `_store_distances` already builds. Extract
that mapping into a helper so both callers share it rather than building it twice.

Add `n_off_axis` to `CalibrationInput` and set it from the site records whose note mentions the
axis, so the CLI can report how many stations the screen removed.

- [ ] **Step 4: Add WQP stations as observations**

WQP stations are not declared in the fishery YAML — there are ~130 and they are discovered, not
authored. `collect_observations` therefore reads them from the store's own station list:

```python
def _wqp_sites(slug: str) -> dict[str, list[tuple[datetime, float]]]:
    """Every WQP station held for this fishery, with its salinity series.

    Discovered from the store rather than declared in the fishery YAML:
    there are ~130 of them, they arrive from a bbox query, and which ones
    are usable is decided by the stem screen, not by hand.
    """
    from tidescout.sources import wqp

    try:
        store = wqp.default_store(slug)
    except FileNotFoundError:
        return {}
    return {s: store.salinity_series(s) for s in sorted(store.stations())}
```

Their coordinates come from `wqp.station_coords(slug)`, which Task 2 already stores and reads
back. Feed those into `site_distances_km` and `station_stem_km` exactly as the NERRS stations'
coordinates are fed in today.

WQP samples are single grabs, so they do NOT go through `daily_means_and_swings` (a one-sample
day fails the 40-reading gate, correctly). They enter as individual observations at their own
tidal phase, which is the point of keeping their timestamps exact:

```python
    for site, series in _wqp_sites(slug).items():
        if is_off_axis(wqp_stem.get(site, float("nan")), False):
            continue
        if site not in wqp_dist:
            continue
        for ts, ppt in series:
            day = ts.astimezone(tz).date()
            if day in by_day:
                observations.append((wqp_dist[site], by_day[day], ppt))
```

- [ ] **Step 5: Run tests, then run the calibration**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
$HOME/.venvs/tidescout/bin/tidescout salinity calibrate winyah-bay
```

Expected: the site table now lists WQP stations, the off-axis ones carry the axis note, and
`n_distinct_distances` is well above 2.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/salinity_fit.py backend/tidescout/models.py backend/tests/test_salinity.py
git commit -m "feat: screen stations onto the main stem and fit against WQP anchors"
```

---

## Task 6: Per-cell coverage on `SalinityField`

**Files:**
- Modify: `backend/tidescout/engine/salinity.py:60-95` and `:145-175`
- Modify: `backend/tests/test_salinity.py`

**Interfaces:**
- Consumes: `SalinityField`, `salinity_field`.
- Produces:
  - `Coverage` — a `str` enum with members `MEASURED = "measured"`, `INTERPOLATED = "interpolated"`, `EXTRAPOLATED = "extrapolated"`
  - `SalinityField.coverage: np.ndarray` (dtype `<U12`, same shape as `.ppt`)
  - `classify_coverage(distance_km, observed_km: Sequence[float], near_km: float = 1.0) -> np.ndarray`

**Why per-cell and array-shaped.** Coverage varies ALONG the estuary within a single evaluation —
that is the whole thing the field exists to express — so a scalar would collapse it. `extrapolated`
and `fitted` stay scalar because they are properties of the discharge and the config respectively,
not of position.

- [ ] **Step 1: Write the failing tests**

```python
def test_coverage_is_per_cell_and_aligned_with_ppt():
    from tidescout.engine.salinity import salinity_field

    d = np.array([2.0, 6.0, 12.0, 30.0])
    f = salinity_field(d, 4000.0, 0.25, CFG)
    assert f.coverage.shape == f.ppt.shape


def test_a_cell_at_an_observation_reads_measured():
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([5.56]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.MEASURED


def test_a_cell_between_observations_reads_interpolated():
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([8.0]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.INTERPOLATED


def test_a_cell_outside_the_observed_span_reads_extrapolated():
    """The 53 features seaward of North Jetty have no WQP station below
    2.58 km and must come out extrapolated -- that band gains nothing from
    this work and must not claim otherwise."""
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([1.0, 30.0]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.EXTRAPOLATED
    assert cov[1] == Coverage.EXTRAPOLATED


def test_no_observations_makes_everything_extrapolated():
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([5.0, 15.0]), observed_km=[], near_km=1.0)
    assert set(cov) == {Coverage.EXTRAPOLATED}
```

- [ ] **Step 2: Run and watch fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_salinity.py -q -k coverage`
Expected: FAIL — `cannot import name 'Coverage'`

- [ ] **Step 3: Implement**

```python
from enum import StrEnum


class Coverage(StrEnum):
    """How well OBSERVED this cell's along-estuary position is.

    A coverage statement, not a quality score -- deliberately ordinal rather
    than a 0-1 number, because a number invites being multiplied into
    something and this must not silently scale a bite score.

    Distinct from `SalinityField.extrapolated`, which asks only whether the
    DISCHARGE fell inside `calibration_range_cfs`, and from `fitted`, which
    asks whether the config was ever calibrated at all. A caller can have
    `extrapolated=False` on a cell whose position nothing ever observed.
    """

    MEASURED = "measured"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"


def classify_coverage(distance_km, observed_km, near_km: float = 1.0) -> np.ndarray:
    """Per-cell coverage against the along-estuary distances actually observed.

    MEASURED within `near_km` of an observation, INTERPOLATED inside their
    span, EXTRAPOLATED outside it. With no observations everything is
    EXTRAPOLATED -- the honest answer, and the one Winyah gave before this
    work.
    """
    d = np.atleast_1d(np.asarray(distance_km, dtype="float64"))
    obs = np.asarray(sorted(observed_km), dtype="float64")
    out = np.full(d.shape, str(Coverage.EXTRAPOLATED), dtype="<U12")
    if obs.size == 0:
        return out
    inside = (d >= obs[0]) & (d <= obs[-1])
    out[inside] = str(Coverage.INTERPOLATED)
    nearest = np.min(np.abs(d[:, None] - obs[None, :]), axis=1)
    out[nearest <= near_km] = str(Coverage.MEASURED)
    return out
```

Add `coverage: np.ndarray` to `SalinityField` (default
`field(default_factory=lambda: np.array([], dtype="<U12"))`), and give `salinity_field` a new
keyword argument:

```python
def salinity_field(
    distance_km, cfs: float, phase: float, cfg: SalinityConfig, observed_km: Sequence[float] = ()
) -> SalinityField:
```

populating `coverage=classify_coverage(distance_km, observed_km)`.

**`observed_km` is an ARGUMENT, not a `SalinityConfig` field, and that is deliberate.** Coverage
is a property of the observations currently held, which change every time a store is imported;
`SalinityConfig` is authored in the fishery YAML and is meant to be stable and reviewable. Putting
it in the config would also require Task 7 to edit `fisheries/winyah-bay.yaml`, which Task 7 is
explicitly forbidden to do. Defaulting to `()` means every existing caller keeps working and gets
all-EXTRAPOLATED, which is the honest answer for a caller that has not said what it observed.

- [ ] **Step 4: Run tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/engine/salinity.py backend/tidescout/models.py backend/tests/test_salinity.py
git commit -m "feat: report per-cell salinity coverage alongside the value"
```

---

## Task 7: The gate — refit and report

**Files:**
- Modify: `backend/tidescout/cli.py` (calibrate output)
- Create: `.superpowers/sdd/2026-08-24-salinity-anchoring/gate-report.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a written report for Ellis. NO config values change in this task.

**This task does not decide the model form.** It measures and reports. Ellis chooses.

- [ ] **Step 1: Run the full pipeline end to end**

```bash
cd /Users/ellismillwood/Documents/tidescout
$HOME/.venvs/tidescout/bin/tidescout salinity import-wqp winyah-bay
$HOME/.venvs/tidescout/bin/tidescout salinity stem winyah-bay
$HOME/.venvs/tidescout/bin/tidescout salinity calibrate winyah-bay | tee /tmp/calibrate-after.txt
```

- [ ] **Step 2: Record the comparison**

Write `.superpowers/sdd/2026-08-24-salinity-anchoring/gate-report.md` containing, with real
numbers and no rounding away of a bad result:

| metric | before (PR #4) | after |
|---|---|---|
| observations | 10,864 | ? |
| distinct on-axis distances | 2 | ? |
| distance span | 2.35 km | ? |
| rmse | 4.060 ppt | ? |
| observation resolution | 0.003 ppt | ? |
| **rmse / resolution** | **1,353x** | **?** |
| condition number | 12.2 | ? |
| `fitted` | False | ? |
| stations excluded by the stem screen | n/a (3, by hand) | ? |

Plus per-station bias for every admitted station, and the feature-coverage table from the spec's
§1 recomputed against the new observed span.

- [ ] **Step 3: Report on `ocean_ppt`, which the gate must also examine**

`ocean_ppt` is held at 34.0 and has never been measured — Task 4 of Phase 2 verified no CO-OPS
station within 250 km serves salinity. It is the seaward anchor the whole profile decays from, so
an unmeasured value there propagates everywhere, and it governs precisely the 0–2.58 km band that
has no station.

Two candidates now exist that did not before. Report both, with numbers, and recommend — do not
change anything:

1. The highest-salinity on-axis WQP observations near the mouth (WB-06 reads up to 35.4 ppt at
   5.56 km).
2. North Inlet's three NERRS stations, which are off-axis for the PROFILE but are ocean-flushed
   and read 31.4–32.0 ppt mean with maxima near 39 — arguably the best available shelf-water
   proxy precisely BECAUSE they are not on the river's axis.

State whether `ocean_ppt` should be freed in the fit rather than held, and what would constrain
it if so.

- [ ] **Step 4: Re-run the score-spread probe**

The spec's §1 measured a 26-point trout sub-score spread at North Jetty and 43 at the bay
stations across four healthy fits. Re-run that comparison with the refitted parameters and report
whether the spread narrowed. This is the number that says whether the work helped the SCORE, as
opposed to helping the rmse.

- [ ] **Step 5: Report to Ellis and STOP**

Present the table, state plainly whether `fitted` can now be True, and if it cannot, say which
single cause is now binding. Do NOT edit `fisheries/winyah-bay.yaml`'s `salinity:` block, do not
free `ocean_ppt`, and do not choose a replacement model form. Those are the gate's output, not
its input.

- [ ] **Step 6: Commit the report**

```bash
git add .superpowers/sdd/2026-08-24-salinity-anchoring/gate-report.md backend/tidescout/cli.py
git commit -m "docs: gate report on the anchored salinity refit"
```

---

## Completion Checklist

- [ ] `make check` green; test count > 558
- [ ] `tidescout salinity import-wqp winyah-bay` runs, and a second run reports 0 new
- [ ] `tidescout salinity stem winyah-bay` builds `stem_km.npy`
- [ ] The six NERRS stations land on the correct side of the stem screen, by the computed
      criterion, matching Task 4 Step 5's expected output exactly
- [ ] `citation()` on a WQP store names WQP and does NOT claim NERRS
- [ ] `SalinityField.coverage` is array-shaped and aligned with `.ppt`
- [ ] The 0–2.58 km band still reads EXTRAPOLATED — this work does not close it
- [ ] Gate report written with real numbers, including the `ocean_ppt` recommendation, and
      `fisheries/winyah-bay.yaml` unchanged
