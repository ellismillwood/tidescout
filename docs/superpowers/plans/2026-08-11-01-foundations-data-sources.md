# TideScout Plan 1: Foundations & Data Sources — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `tidescout conditions winyah-bay --date YYYY-MM-DD --model gfs` CLI that assembles and prints a full day of hourly fishing conditions (tide height/stage, station currents, wind, gusts, pressure + trend, cloud, air temp) plus daily context (sun/moon/solunar, water temp + trend, salinity, river discharge bucket) from free public APIs, with SQLite caching and graceful per-source failure.

**Architecture:** `tidescout/sources/` holds one module per external API (Open-Meteo, NOAA CO-OPS, USGS, astronomy) — all network I/O goes through a shared SQLite read-through cache. `tidescout/engine/` stays pure (no I/O): the conditions assembler takes fetched data structures and returns a `DayConditions`. A Typer CLI renders it with Rich. This is Phase 0+1 of the design spec at `docs/superpowers/specs/2026-08-11-tidescout-design.md` (read its §3, §4, §10 before starting).

**Tech Stack:** Python 3.12, pydantic v2, httpx, PyYAML, typer, rich, astral (sun), ephem (moon/solunar), pytest + respx (recorded HTTP fixtures), ruff, uv for env management.

## Global Constraints

- Repo root: `~/Documents/tidescout`. Run all commands from repo root unless a step says otherwise.
- Python 3.12 venv at `~/.venvs/tidescout` (created in Task 1). Never use system Python (it is 3.9).
- All displayed times in `America/New_York`; internal datetimes are timezone-aware.
- Units everywhere: knots, feet, °F, mb (hPa), cfs, ppt.
- Tests NEVER hit live APIs — respx fixtures only. Live calls happen only in steps explicitly labeled **LIVE VERIFICATION**.
- Every network fetch goes through `tidescout/sources/cache.py`.
- Nothing in `tidescout/engine/` may import httpx or sqlite3, or trigger network/disk I/O — importing dataclasses and pure helper functions from `tidescout/sources/` modules is allowed (e.g., `stage_at`).
- Before every commit: `make check` must pass (ruff + pytest).
- Later phases (bathymetry, ANUGA, salinity model, scoring, API, frontend) are OUT OF SCOPE for this plan. So are NERR/CDMO sensors (registration required — config is extensible for them later), SECOFS, CUDEM, and oyster layers.

---

### Task 1: Scaffold backend package, venv, and quality gate

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/tidescout/__init__.py`
- Create: `backend/tidescout/errors.py`
- Create: `backend/tidescout/cli.py`
- Create: `backend/tests/test_smoke.py`
- Create: `Makefile`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: importable `tidescout` package; `tidescout.errors.SourceUnavailable(source: str)` exception; Typer app `tidescout.cli:app` with entry-point script `tidescout`; `make check` gate. All later tasks add modules under `backend/tidescout/` and tests under `backend/tests/`.

- [ ] **Step 1: Create the venv and install uv if missing**

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.venvs/tidescout --python 3.12
```

Expected: venv created; `~/.venvs/tidescout/bin/python --version` prints Python 3.12.x.

- [ ] **Step 2: Write `backend/pyproject.toml`**

```toml
[project]
name = "tidescout"
version = "0.1.0"
description = "SC inshore fishing decision support"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.7",
    "PyYAML>=6",
    "typer>=0.12",
    "rich>=13",
    "astral>=3.2",
    "ephem>=4.1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "respx>=0.21", "ruff>=0.5"]

[project.scripts]
tidescout = "tidescout.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["tidescout"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write the package skeleton**

`backend/tidescout/__init__.py`:

```python
"""TideScout: SC inshore fishing decision support."""

__version__ = "0.1.0"
```

`backend/tidescout/errors.py`:

```python
class SourceUnavailable(Exception):
    """An external data source failed and no cached fallback exists."""

    def __init__(self, source: str, detail: str = ""):
        self.source = source
        self.detail = detail
        super().__init__(f"{source} unavailable{': ' + detail if detail else ''}")
```

`backend/tidescout/cli.py`:

```python
import typer

app = typer.Typer(no_args_is_help=True, help="TideScout: SC inshore fishing decision support.")


@app.callback()
def _root() -> None:
    """TideScout CLI."""


def main() -> None:
    app()
```

`backend/tests/test_smoke.py`:

```python
from typer.testing import CliRunner

import tidescout
from tidescout.cli import app


def test_package_imports():
    assert tidescout.__version__ == "0.1.0"


def test_cli_help_runs():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "TideScout" in result.output
```

- [ ] **Step 4: Write the `Makefile` (repo root)**

Use real tabs for recipe indentation, not spaces.

```makefile
PY := $(HOME)/.venvs/tidescout/bin/python

.PHONY: install check test lint

install:
	uv pip install -p $(PY) -e './backend[dev]'

lint:
	cd backend && $(PY) -m ruff check .

test:
	cd backend && $(PY) -m pytest -q

check: lint test
```

- [ ] **Step 5: Install and run the gate**

```bash
make install
make check
```

Expected: install succeeds; ruff clean; 2 tests pass. If `ephem` fails to build, run `uv pip install -p $(HOME)/.venvs/tidescout/bin/python --only-binary :all: ephem` and retry (wheels exist for macOS arm64).

- [ ] **Step 6: Verify the entry point works**

```bash
~/.venvs/tidescout/bin/tidescout --help
```

Expected: help text prints, exit 0.

- [ ] **Step 7: Commit**

```bash
git add backend Makefile
git commit -m "feat: scaffold tidescout backend package with quality gate"
```

---

### Task 2: Fishery config model and loader

**Files:**
- Create: `backend/tidescout/models.py`
- Create: `backend/tidescout/config.py`
- Create: `fisheries/winyah-bay.yaml`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by every later task):
  - `tidescout.models.Fishery` (pydantic) with fields: `slug: str`, `name: str`, `timezone: str`, `bbox: tuple[float, float, float, float]` (west, south, east, north), `center: tuple[float, float]` (lon, lat), `orientation_deg: float`, `stations: Stations`, `rivers: list[RiverGauge]`, `discharge_buckets: DischargeBuckets`, `climatology: Climatology`.
  - `Stations` fields: `tide: list[str]`, `currents: list[str]`, `water: list[WaterSensor]`; `WaterSensor` fields: `kind: Literal["usgs", "coops"]`, `station: str`, `params: list[str]`.
  - `RiverGauge` fields: `name: str`, `usgs_site: str`, `weight: float = 1.0`.
  - `DischargeBuckets` fields: `low_below_cfs: float`, `high_above_cfs: float`.
  - `Climatology` fields: `water_temp_f_by_month: dict[int, float]`, `salinity_ppt_by_month: dict[int, float]`.
  - `tidescout.config.load_fishery(slug: str, root: Path | None = None) -> Fishery` and `tidescout.config.FISHERIES_DIR: Path` (repo `fisheries/` resolved relative to this file: `Path(__file__).resolve().parents[2] / "fisheries"`).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_config.py`:

```python
import pytest

from tidescout.config import load_fishery


def test_load_winyah_bay():
    f = load_fishery("winyah-bay")
    assert f.name == "Winyah Bay"
    assert f.timezone == "America/New_York"
    west, south, east, north = f.bbox
    assert west < east and south < north
    assert west < f.center[0] < east
    assert south < f.center[1] < north
    assert len(f.rivers) == 3
    assert f.discharge_buckets.low_below_cfs < f.discharge_buckets.high_above_cfs
    assert set(f.climatology.water_temp_f_by_month) == set(range(1, 13))
    assert set(f.climatology.salinity_ppt_by_month) == set(range(1, 13))


def test_unknown_fishery_raises():
    with pytest.raises(FileNotFoundError):
        load_fishery("lake-lanier")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.config).

- [ ] **Step 3: Write models, loader, and the Winyah config**

`backend/tidescout/models.py`:

```python
from typing import Literal

from pydantic import BaseModel


class RiverGauge(BaseModel):
    name: str
    usgs_site: str
    weight: float = 1.0


class WaterSensor(BaseModel):
    kind: Literal["usgs", "coops"]
    station: str
    params: list[str] = []


class Stations(BaseModel):
    tide: list[str] = []
    currents: list[str] = []
    water: list[WaterSensor] = []


class DischargeBuckets(BaseModel):
    low_below_cfs: float
    high_above_cfs: float


class Climatology(BaseModel):
    water_temp_f_by_month: dict[int, float]
    salinity_ppt_by_month: dict[int, float]


class Fishery(BaseModel):
    slug: str
    name: str
    timezone: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    center: tuple[float, float]  # lon, lat
    orientation_deg: float  # direction the bay mouth faces, degrees true
    stations: Stations
    rivers: list[RiverGauge]
    discharge_buckets: DischargeBuckets
    climatology: Climatology
```

`backend/tidescout/config.py`:

```python
from pathlib import Path

import yaml

from tidescout.models import Fishery

FISHERIES_DIR = Path(__file__).resolve().parents[2] / "fisheries"


def load_fishery(slug: str, root: Path | None = None) -> Fishery:
    path = (root or FISHERIES_DIR) / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No fishery config at {path}")
    raw = yaml.safe_load(path.read_text())
    return Fishery.model_validate(raw)
```

`fisheries/winyah-bay.yaml` — station lists start empty and are filled by Task 3's discovery run. Climatology numbers are deliberate initial values (river-dominated estuary), refined in the salinity phase:

```yaml
slug: winyah-bay
name: Winyah Bay
timezone: America/New_York
bbox: [-79.45, 33.15, -79.05, 33.60]
center: [-79.25, 33.35]
orientation_deg: 135 # entrance faces southeast
stations:
  tide: []
  currents: []
  water: []
rivers:
  - name: Pee Dee
    usgs_site: ""
    weight: 1.0
  - name: Waccamaw
    usgs_site: ""
    weight: 1.0
  - name: Black
    usgs_site: ""
    weight: 1.0
discharge_buckets:
  low_below_cfs: 6000
  high_above_cfs: 25000
climatology:
  water_temp_f_by_month:
    {1: 48, 2: 50, 3: 57, 4: 66, 5: 74, 6: 81, 7: 85, 8: 85, 9: 80, 10: 71, 11: 60, 12: 52}
  salinity_ppt_by_month:
    {1: 8, 2: 7, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 15, 9: 14, 10: 12, 11: 10, 12: 9}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_config.py -v`
Expected: 2 PASS. (Empty `usgs_site: ""` strings are valid at this stage; Task 3 fills them.)

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/models.py backend/tidescout/config.py fisheries/winyah-bay.yaml backend/tests/test_config.py
git commit -m "feat: fishery config model with initial winyah-bay definition"
```

---

### Task 3: SQLite read-through cache

**Files:**
- Create: `backend/tidescout/sources/__init__.py` (empty)
- Create: `backend/tidescout/sources/cache.py`
- Test: `backend/tests/test_cache.py`

**Interfaces:**
- Consumes: `tidescout.errors.SourceUnavailable`.
- Produces (every fetcher in Tasks 4–7 uses exactly this):
  - `tidescout.sources.cache.Cached` dataclass: `payload: Any` (parsed JSON), `fetched_at: datetime` (UTC-aware), `fresh: bool`.
  - `tidescout.sources.cache.Cache` with `__init__(self, db_path: Path, now_fn: Callable[[], datetime] | None = None)` and `get_or_fetch(self, source: str, key: str, ttl: timedelta | None, fetch: Callable[[], Any]) -> Cached`. `ttl=None` means immutable (never refetch if present). On fetch failure with a stale row present → return stale `Cached(fresh=False)`. On fetch failure with no row → raise `SourceUnavailable(source)`.
  - `tidescout.sources.cache.default_cache() -> Cache` returning a singleton at repo `data/cache.sqlite` (`Path(__file__).resolve().parents[3] / "data" / "cache.sqlite"`, parent dir created on demand).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_cache.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_cache.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.sources.cache).

- [ ] **Step 3: Implement the cache**

`backend/tidescout/sources/cache.py`:

```python
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

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
```

Also create empty `backend/tidescout/sources/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_cache.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/sources backend/tests/test_cache.py
git commit -m "feat: sqlite read-through cache with stale fallback"
```

---

### Task 4: Station discovery + record real station IDs into config

**Files:**
- Create: `backend/tidescout/sources/discovery.py`
- Modify: `backend/tidescout/cli.py`
- Modify: `fisheries/winyah-bay.yaml` (fill station/river IDs from the LIVE VERIFICATION run)
- Test: `backend/tests/test_discovery.py`

**Interfaces:**
- Consumes: `Fishery` (Task 2), `Cache.get_or_fetch` (Task 3), `load_fishery`.
- Produces:
  - `tidescout.sources.discovery.StationInfo` dataclass: `id: str`, `name: str`, `lat: float`, `lon: float`, `kind: str` (`"tide" | "current" | "usgs"`).
  - `find_tide_stations(fishery: Fishery, cache: Cache) -> list[StationInfo]`
  - `find_current_stations(fishery: Fishery, cache: Cache) -> list[StationInfo]`
  - `find_usgs_sites(fishery: Fishery, cache: Cache, param: str) -> list[StationInfo]` (param = USGS code, e.g. `"00060"` discharge, `"00480"` salinity, `"00010"` water temp; searches an expanded bbox that reaches upstream river gauges)
  - CLI command `tidescout stations SLUG`.

- [ ] **Step 1: Write the failing tests (respx fixtures)**

`backend/tests/test_discovery.py`:

```python
import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources.discovery import (
    find_current_stations,
    find_tide_stations,
    find_usgs_sites,
)

TIDE_FIXTURE = {
    "stations": [
        {"id": "8662245", "name": "South Island Ferry", "lat": 33.2517, "lng": -79.2533},
        {"id": "8661070", "name": "Springmaid Pier", "lat": 33.655, "lng": -78.9183},
    ]
}

CURRENT_FIXTURE = {
    "stations": [
        {"id": "WIN1201", "name": "Winyah Bay Entrance", "lat": 33.21, "lng": -79.17},
        {"id": "CHA0001", "name": "Charleston Harbor", "lat": 32.77, "lng": -79.92},
    ]
}

USGS_RDB = (
    "# comment\n"
    "agency_cd\tsite_no\tstation_nm\tsite_tp_cd\tdec_lat_va\tdec_long_va\n"
    "5s\t15s\t50s\t7s\t16s\t16s\n"
    "USGS\t02131000\tPEE DEE RIVER AT PEEDEE, SC\tST\t34.2043\t-79.5495\n"
    "USGS\t02136000\tBLACK RIVER AT KINGSTREE, SC\tST\t33.6654\t-79.8309\n"
)


@respx.mock
def test_find_tide_stations_filters_to_bbox(tmp_path):
    respx.get(url__regex=r"https://api\.tidesandcurrents\.noaa\.gov/mdapi/.*tidepredictions.*").mock(
        return_value=Response(200, json=TIDE_FIXTURE)
    )
    f = load_fishery("winyah-bay")
    cache = Cache(tmp_path / "c.sqlite")
    stations = find_tide_stations(f, cache)
    assert [s.id for s in stations] == ["8662245"]
    assert stations[0].kind == "tide"


@respx.mock
def test_find_current_stations_filters_to_bbox(tmp_path):
    respx.get(
        url__regex=r"https://api\.tidesandcurrents\.noaa\.gov/mdapi/.*currentpredictions.*"
    ).mock(return_value=Response(200, json=CURRENT_FIXTURE))
    f = load_fishery("winyah-bay")
    cache = Cache(tmp_path / "c.sqlite")
    stations = find_current_stations(f, cache)
    assert [s.id for s in stations] == ["WIN1201"]


@respx.mock
def test_find_usgs_sites_parses_rdb(tmp_path):
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/site/.*").mock(
        return_value=Response(200, text=USGS_RDB)
    )
    f = load_fishery("winyah-bay")
    cache = Cache(tmp_path / "c.sqlite")
    sites = find_usgs_sites(f, cache, "00060")
    assert {s.id for s in sites} == {"02131000", "02136000"}
    assert all(s.kind == "usgs" for s in sites)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_discovery.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.sources.discovery).

- [ ] **Step 3: Implement discovery**

`backend/tidescout/sources/discovery.py`:

```python
from dataclasses import dataclass
from datetime import timedelta

import httpx

from tidescout.models import Fishery
from tidescout.sources.cache import Cache

MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
USGS_SITE = "https://waterservices.usgs.gov/nwis/site/"
TTL = timedelta(days=30)
# Upstream river gauges sit outside the bay bbox; pad generously for USGS search.
USGS_BBOX_PAD_DEG = 1.0


@dataclass
class StationInfo:
    id: str
    name: str
    lat: float
    lon: float
    kind: str


def _get_json(url: str, params: dict) -> dict:
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east


def _noaa_stations(fishery: Fishery, cache: Cache, type_: str, kind: str) -> list[StationInfo]:
    cached = cache.get_or_fetch(
        "noaa-mdapi", type_, TTL, lambda: _get_json(MDAPI, {"type": type_})
    )
    out = []
    for s in cached.payload.get("stations", []):
        lat, lon = float(s["lat"]), float(s["lng"])
        if _in_bbox(lat, lon, fishery.bbox):
            out.append(StationInfo(str(s["id"]), s["name"], lat, lon, kind))
    return out


def find_tide_stations(fishery: Fishery, cache: Cache) -> list[StationInfo]:
    return _noaa_stations(fishery, cache, "tidepredictions", "tide")


def find_current_stations(fishery: Fishery, cache: Cache) -> list[StationInfo]:
    return _noaa_stations(fishery, cache, "currentpredictions", "current")


def find_usgs_sites(fishery: Fishery, cache: Cache, param: str) -> list[StationInfo]:
    west, south, east, north = fishery.bbox
    bbox = (
        f"{west - USGS_BBOX_PAD_DEG:.4f},{south - 0.1:.4f},"
        f"{east + 0.1:.4f},{north + USGS_BBOX_PAD_DEG:.4f}"
    )
    params = {
        "format": "rdb",
        "bBox": bbox,
        "parameterCd": param,
        "siteStatus": "active",
        "hasDataTypeCd": "iv",
    }

    def fetch() -> str:
        resp = httpx.get(USGS_SITE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text

    cached = cache.get_or_fetch("usgs-site", f"{bbox}:{param}", TTL, fetch)
    lines = [ln for ln in cached.payload.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    out = []
    for ln in lines[2:]:  # line after header is the rdb format row (e.g. "5s")
        cols = ln.split("\t")
        try:
            out.append(
                StationInfo(
                    id=cols[idx["site_no"]],
                    name=cols[idx["station_nm"]],
                    lat=float(cols[idx["dec_lat_va"]]),
                    lon=float(cols[idx["dec_long_va"]]),
                    kind="usgs",
                )
            )
        except (KeyError, ValueError, IndexError):
            continue
    return out
```

Replace the full contents of `backend/tidescout/cli.py` with:

```python
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="TideScout: SC inshore fishing decision support.")
console = Console()


@app.callback()
def _root() -> None:
    """TideScout CLI."""


@app.command()
def stations(slug: str) -> None:
    """Discover NOAA and USGS stations for a fishery's area."""
    from tidescout.config import load_fishery
    from tidescout.sources.cache import default_cache
    from tidescout.sources import discovery

    fishery = load_fishery(slug)
    cache = default_cache()
    table = Table(title=f"Stations near {fishery.name}")
    for col in ("kind", "id", "name", "lat", "lon"):
        table.add_column(col)
    rows = (
        discovery.find_tide_stations(fishery, cache)
        + discovery.find_current_stations(fishery, cache)
        + [
            s
            for param in ("00060", "00010", "00480")
            for s in discovery.find_usgs_sites(fishery, cache, param)
        ]
    )
    seen = set()
    for s in rows:
        if (s.kind, s.id) in seen:
            continue
        seen.add((s.kind, s.id))
        table.add_row(s.kind, s.id, s.name, f"{s.lat:.4f}", f"{s.lon:.4f}")
    console.print(table)


def main() -> None:
    app()
```

(The full file after this task contains exactly one `app`, one `_root`, one `stations`, one `main`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_discovery.py -v`
Expected: 3 PASS.

- [ ] **Step 5: LIVE VERIFICATION — run discovery for real**

```bash
~/.venvs/tidescout/bin/tidescout stations winyah-bay
```

Expected: a table with at least one tide station inside the bay (a station named like "South Island Ferry" is the preferred pick), zero or more current stations near the entrance/jetties, and USGS sites on the Pee Dee, Waccamaw, and Black rivers plus any temp/salinity sites near Georgetown. If the table is empty or the request 4xx/5xxs, read the raw response (`httpx.get` the URL from `discovery.py` manually) and adjust parameter names — the mdapi `type` values and USGS `bBox` format are the likely culprits.

- [ ] **Step 6: Record the real IDs into `fisheries/winyah-bay.yaml`**

Fill from the live output:
- `stations.tide`: the in-bay tide prediction station ID(s), best-positioned first (prefer South Island Ferry if present).
- `stations.currents`: current prediction station ID(s) nearest the entrance/jetties (may legitimately be empty if none are in the bbox — leave `[]` and note it in the commit message).
- `stations.water`: any USGS site with params 00010 (temp) and/or 00480 (salinity) in/near the bay, e.g. `{kind: usgs, station: "021108125", params: ["00010", "00480"]}` (use the REAL discovered ID, not this example).
- `rivers[*].usgs_site`: the most-downstream non-tidal discharge gauge per river (candidates to look for: Pee Dee at Peedee `02131000`, Black at Kingstree `02136000`, Waccamaw near Longs/Conway — confirm in output).

Then re-run `make check` (config test still passes) and verify:

```bash
~/.venvs/tidescout/bin/python -c "from tidescout.config import load_fishery; f = load_fishery('winyah-bay'); print(f.stations); print([r.usgs_site for r in f.rivers])"
```

Expected: real IDs print; no empty `usgs_site` remains.

- [ ] **Step 7: Commit**

```bash
git add backend/tidescout/sources/discovery.py backend/tidescout/cli.py fisheries/winyah-bay.yaml backend/tests/test_discovery.py
git commit -m "feat: station discovery CLI; record real Winyah Bay station IDs"
```

---

### Task 5: Open-Meteo weather client with model picker

**Files:**
- Create: `backend/tidescout/sources/weather.py`
- Test: `backend/tests/test_weather.py`

**Interfaces:**
- Consumes: `Fishery`, `Cache`.
- Produces:
  - `tidescout.sources.weather.WeatherHour` dataclass: `time: datetime` (ET-aware), `air_temp_f: float | None`, `wind_speed_kn: float | None`, `wind_dir_deg: float | None`, `wind_gust_kn: float | None`, `pressure_mb: float | None`, `cloud_cover_pct: float | None`, `precip_in: float | None`.
  - `WEATHER_MODELS: dict[str, str]` mapping CLI keys → Open-Meteo codes: `{"best": "best_match", "gfs": "gfs_seamless", "ecmwf": "ecmwf_ifs025", "icon": "icon_seamless", "hrrr": "gfs_hrrr", "nbm": "ncep_nbm_conus"}`.
  - `fetch_weather(fishery: Fishery, date: date, model_key: str, cache: Cache) -> tuple[list[WeatherHour], str]` — returns 48 hours (the day before + the requested day, so pressure trend is computable) and the resolved model label. Dates older than 7 days route to the archive API (ERA5 reanalysis; model choice doesn't apply) and the label returned is `"era5"`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_weather.py`:

```python
from datetime import date

import pytest
import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources.weather import WEATHER_MODELS, fetch_weather

HOURLY_KEYS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]


def _fixture(n_hours: int) -> dict:
    times = [f"2026-08-{14 + h // 24:02d}T{h % 24:02d}:00" for h in range(n_hours)]
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [80.0] * n_hours,
            "wind_speed_10m": [9.0] * n_hours,
            "wind_direction_10m": [220.0] * n_hours,
            "wind_gusts_10m": [14.0] * n_hours,
            "pressure_msl": [1013.2] * n_hours,
            "cloud_cover": [40] * n_hours,
            "precipitation": [0.0] * n_hours,
        }
    }


@respx.mock
def test_fetch_weather_forecast(tmp_path):
    route = respx.get(url__regex=r"https://api\.open-meteo\.com/v1/forecast.*").mock(
        return_value=Response(200, json=_fixture(48))
    )
    f = load_fishery("winyah-bay")
    hours, label = fetch_weather(f, date(2026, 8, 15), "gfs", Cache(tmp_path / "c.sqlite"))
    assert label == "gfs"
    assert len(hours) == 48
    assert hours[0].wind_speed_kn == 9.0
    assert hours[0].pressure_mb == 1013.2
    sent = route.calls[0].request.url
    assert "models=gfs_seamless" in str(sent)
    for key in HOURLY_KEYS:
        assert key in str(sent)


@respx.mock
def test_old_dates_route_to_archive(tmp_path):
    respx.get(url__regex=r"https://archive-api\.open-meteo\.com/v1/archive.*").mock(
        return_value=Response(200, json=_fixture(48))
    )
    f = load_fishery("winyah-bay")
    hours, label = fetch_weather(f, date(2020, 6, 1), "gfs", Cache(tmp_path / "c.sqlite"))
    assert label == "era5"
    assert len(hours) == 48


def test_unknown_model_rejected(tmp_path):
    f = load_fishery("winyah-bay")
    with pytest.raises(KeyError):
        fetch_weather(f, date(2026, 8, 15), "wrf", Cache(tmp_path / "c.sqlite"))


def test_model_registry_complete():
    assert set(WEATHER_MODELS) == {"best", "gfs", "ecmwf", "icon", "hrrr", "nbm"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_weather.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.sources.weather).

- [ ] **Step 3: Implement the weather client**

`backend/tidescout/sources/weather.py`:

```python
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from tidescout.models import Fishery
from tidescout.sources.cache import Cache

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_CUTOFF_DAYS = 7
FORECAST_TTL = timedelta(hours=1)

WEATHER_MODELS = {
    "best": "best_match",
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
    "icon": "icon_seamless",
    "hrrr": "gfs_hrrr",
    "nbm": "ncep_nbm_conus",
}

HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
]


@dataclass
class WeatherHour:
    time: datetime
    air_temp_f: float | None
    wind_speed_kn: float | None
    wind_dir_deg: float | None
    wind_gust_kn: float | None
    pressure_mb: float | None
    cloud_cover_pct: float | None
    precip_in: float | None


def _get_json(url: str, params: dict) -> dict:
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_weather(
    fishery: Fishery, day: date, model_key: str, cache: Cache
) -> tuple[list[WeatherHour], str]:
    model_code = WEATHER_MODELS[model_key]  # KeyError for unknown keys is intended
    lon, lat = fishery.center
    start = day - timedelta(days=1)
    use_archive = day < date.today() - timedelta(days=ARCHIVE_CUTOFF_DAYS)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start.isoformat(),
        "end_date": day.isoformat(),
        "timezone": fishery.timezone,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "kn",
        "precipitation_unit": "inch",
    }
    if use_archive:
        url, label = ARCHIVE_URL, "era5"
        ttl: timedelta | None = None  # reanalysis of the past never changes
    else:
        url, label = FORECAST_URL, model_key
        params["models"] = model_code
        ttl = FORECAST_TTL
    key = f"{fishery.slug}:{day.isoformat()}:{label}"
    cached = cache.get_or_fetch("open-meteo", key, ttl, lambda: _get_json(url, params))
    hourly = cached.payload["hourly"]
    tz = ZoneInfo(fishery.timezone)
    hours = []
    for i, t in enumerate(hourly["time"]):
        def _v(name: str):
            vals = hourly.get(name)
            return None if vals is None or vals[i] is None else float(vals[i])

        hours.append(
            WeatherHour(
                time=datetime.fromisoformat(t).replace(tzinfo=tz),
                air_temp_f=_v("temperature_2m"),
                wind_speed_kn=_v("wind_speed_10m"),
                wind_dir_deg=_v("wind_direction_10m"),
                wind_gust_kn=_v("wind_gusts_10m"),
                pressure_mb=_v("pressure_msl"),
                cloud_cover_pct=_v("cloud_cover"),
                precip_in=_v("precipitation"),
            )
        )
    return hours, label
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_weather.py -v`
Expected: 4 PASS.

- [ ] **Step 5: LIVE VERIFICATION — every model code answers**

```bash
for m in best gfs ecmwf icon hrrr nbm; do
  ~/.venvs/tidescout/bin/python - "$m" <<'EOF'
import sys
from datetime import date
from tidescout.config import load_fishery
from tidescout.sources.cache import default_cache
from tidescout.sources.weather import fetch_weather
hours, label = fetch_weather(load_fishery("winyah-bay"), date.today(), sys.argv[1], default_cache())
ok = sum(1 for h in hours if h.wind_speed_kn is not None)
print(f"{sys.argv[1]:6s} -> {label:6s} {len(hours)} hours, {ok} with wind")
EOF
done
```

Expected: six lines, each with 48 hours and mostly non-null wind. If any model 400s (the `ncep_nbm_conus` code is the least certain), check the error body — Open-Meteo lists valid model names in it — update that entry in `WEATHER_MODELS`, and re-run tests (`test_model_registry_complete` keys stay the same; only codes change).

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/sources/weather.py backend/tests/test_weather.py
git commit -m "feat: open-meteo client with selectable forecast model and era5 hindcast routing"
```

---

### Task 6: NOAA CO-OPS client — tides, currents, water temp

**Files:**
- Create: `backend/tidescout/sources/noaa.py`
- Test: `backend/tests/test_noaa.py`

**Interfaces:**
- Consumes: `Cache`.
- Produces:
  - `TideHour` dataclass: `time: datetime`, `height_ft: float`.
  - `TideEvent` dataclass: `time: datetime`, `kind: str` (`"H" | "L"`), `height_ft: float`.
  - `TideStage` dataclass: `phase: str` (`"rising" | "falling"`), `frac: float` (0.0 just after an extreme → 1.0 at the next), `next_event: TideEvent`.
  - `CurrentHour` dataclass: `time: datetime`, `speed_kn: float` (signed: positive flood, negative ebb), `dir_deg: float`.
  - `tide_hours(station: str, day: date, tz: str, cache: Cache) -> list[TideHour]` (window: day−1 through day+1).
  - `tide_events(station: str, day: date, tz: str, cache: Cache) -> list[TideEvent]` (same window, `interval=hilo`).
  - `current_hours(station: str, day: date, tz: str, cache: Cache) -> list[CurrentHour]`.
  - `water_temp_latest(station: str, tz: str, cache: Cache) -> tuple[float, datetime] | None`.
  - `stage_at(events: list[TideEvent], t: datetime) -> TideStage | None` (pure; None if `t` isn't bracketed by events).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_noaa.py`:

```python
from datetime import date, datetime
from zoneinfo import ZoneInfo

import respx
from httpx import Response

from tidescout.sources.cache import Cache
from tidescout.sources.noaa import (
    TideEvent,
    current_hours,
    stage_at,
    tide_events,
    tide_hours,
    water_temp_latest,
)

ET = ZoneInfo("America/New_York")

PRED_FIXTURE = {
    "predictions": [
        {"t": "2026-08-15 00:00", "v": "2.31"},
        {"t": "2026-08-15 01:00", "v": "3.12"},
    ]
}

HILO_FIXTURE = {
    "predictions": [
        {"t": "2026-08-15 03:12", "v": "5.1", "type": "H"},
        {"t": "2026-08-15 09:30", "v": "0.4", "type": "L"},
    ]
}

CURRENTS_FIXTURE = {
    "current_predictions": {
        "cp": [
            {"Time": "2026-08-15 00:00", "Velocity_Major": -1.4, "meanFloodDir": 315.0, "meanEbbDir": 135.0},
            {"Time": "2026-08-15 01:00", "Velocity_Major": 0.8, "meanFloodDir": 315.0, "meanEbbDir": 135.0},
        ]
    }
}

TEMP_FIXTURE = {"data": [{"t": "2026-08-15 12:06", "v": "84.2", "f": "0,0,0"}]}


@respx.mock
def test_tide_hours(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=predictions.*interval=h.*").mock(
        return_value=Response(200, json=PRED_FIXTURE)
    )
    hours = tide_hours("8662245", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert hours[0].height_ft == 2.31
    assert hours[0].time == datetime(2026, 8, 15, 0, 0, tzinfo=ET)


@respx.mock
def test_tide_events(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=predictions.*interval=hilo.*").mock(
        return_value=Response(200, json=HILO_FIXTURE)
    )
    events = tide_events("8662245", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert [e.kind for e in events] == ["H", "L"]


@respx.mock
def test_current_hours_signed_direction(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=currents_predictions.*").mock(
        return_value=Response(200, json=CURRENTS_FIXTURE)
    )
    hours = current_hours("WIN1201", date(2026, 8, 15), "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert hours[0].speed_kn == -1.4
    assert hours[0].dir_deg == 135.0  # ebbing -> ebb direction
    assert hours[1].dir_deg == 315.0  # flooding -> flood direction


@respx.mock
def test_water_temp_latest(tmp_path):
    respx.get(url__regex=r".*datagetter.*product=water_temperature.*").mock(
        return_value=Response(200, json=TEMP_FIXTURE)
    )
    result = water_temp_latest("8662245", "America/New_York", Cache(tmp_path / "c.sqlite"))
    assert result is not None
    temp, at = result
    assert temp == 84.2


def test_stage_at_interpolates():
    events = [
        TideEvent(datetime(2026, 8, 15, 3, 0, tzinfo=ET), "H", 5.0),
        TideEvent(datetime(2026, 8, 15, 9, 0, tzinfo=ET), "L", 0.5),
    ]
    stage = stage_at(events, datetime(2026, 8, 15, 6, 0, tzinfo=ET))
    assert stage is not None
    assert stage.phase == "falling"
    assert abs(stage.frac - 0.5) < 0.01
    assert stage.next_event.kind == "L"
    assert stage_at(events, datetime(2026, 8, 15, 1, 0, tzinfo=ET)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_noaa.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.sources.noaa).

- [ ] **Step 3: Implement the CO-OPS client**

`backend/tidescout/sources/noaa.py`:

```python
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from tidescout.sources.cache import Cache

DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
PREDICTION_TTL = None  # tide/current predictions are deterministic
OBS_TTL = timedelta(minutes=15)


@dataclass
class TideHour:
    time: datetime
    height_ft: float


@dataclass
class TideEvent:
    time: datetime
    kind: str  # "H" | "L"
    height_ft: float


@dataclass
class TideStage:
    phase: str  # "rising" | "falling"
    frac: float
    next_event: TideEvent


@dataclass
class CurrentHour:
    time: datetime
    speed_kn: float  # signed: + flood, - ebb
    dir_deg: float


def _get_json(params: dict) -> dict:
    resp = httpx.get(DATAGETTER, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"].get("message", "CO-OPS error"))
    return payload


def _window(day: date) -> tuple[str, str]:
    begin = (day - timedelta(days=1)).strftime("%Y%m%d")
    end = (day + timedelta(days=1)).strftime("%Y%m%d")
    return begin, end


def _parse_t(t: str, tz: ZoneInfo) -> datetime:
    return datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=tz)


def tide_hours(station: str, day: date, tz: str, cache: Cache) -> list[TideHour]:
    begin, end = _window(day)
    params = {
        "product": "predictions", "application": "tidescout", "station": station,
        "begin_date": begin, "end_date": end, "datum": "MLLW",
        "time_zone": "lst_ldt", "units": "english", "interval": "h", "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"pred:{station}:{begin}:{end}", PREDICTION_TTL, lambda: _get_json(params)
    )
    zone = ZoneInfo(tz)
    return [
        TideHour(_parse_t(p["t"], zone), float(p["v"]))
        for p in cached.payload.get("predictions", [])
    ]


def tide_events(station: str, day: date, tz: str, cache: Cache) -> list[TideEvent]:
    begin, end = _window(day)
    params = {
        "product": "predictions", "application": "tidescout", "station": station,
        "begin_date": begin, "end_date": end, "datum": "MLLW",
        "time_zone": "lst_ldt", "units": "english", "interval": "hilo", "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"hilo:{station}:{begin}:{end}", PREDICTION_TTL, lambda: _get_json(params)
    )
    zone = ZoneInfo(tz)
    return [
        TideEvent(_parse_t(p["t"], zone), p["type"], float(p["v"]))
        for p in cached.payload.get("predictions", [])
    ]


def current_hours(station: str, day: date, tz: str, cache: Cache) -> list[CurrentHour]:
    begin, end = _window(day)
    params = {
        "product": "currents_predictions", "application": "tidescout", "station": station,
        "begin_date": begin, "end_date": end,
        "time_zone": "lst_ldt", "units": "english", "interval": "h", "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"cur:{station}:{begin}:{end}", PREDICTION_TTL, lambda: _get_json(params)
    )
    zone = ZoneInfo(tz)
    out = []
    for p in cached.payload.get("current_predictions", {}).get("cp", []):
        speed = float(p["Velocity_Major"])
        dir_deg = float(p["meanFloodDir"] if speed >= 0 else p["meanEbbDir"])
        out.append(CurrentHour(_parse_t(p["Time"], zone), speed, dir_deg))
    return out


def water_temp_latest(station: str, tz: str, cache: Cache) -> tuple[float, datetime] | None:
    params = {
        "product": "water_temperature", "application": "tidescout", "station": station,
        "date": "latest", "time_zone": "lst_ldt", "units": "english", "format": "json",
    }
    cached = cache.get_or_fetch(
        "coops", f"wtemp:{station}", OBS_TTL, lambda: _get_json(params)
    )
    data = cached.payload.get("data", [])
    if not data:
        return None
    zone = ZoneInfo(tz)
    return float(data[-1]["v"]), _parse_t(data[-1]["t"], zone)


def stage_at(events: list[TideEvent], t: datetime) -> TideStage | None:
    ordered = sorted(events, key=lambda e: e.time)
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.time <= t <= nxt.time:
            span = (nxt.time - prev.time).total_seconds()
            frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
            phase = "rising" if nxt.kind == "H" else "falling"
            return TideStage(phase, frac, nxt)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_noaa.py -v`
Expected: 5 PASS.

- [ ] **Step 5: LIVE VERIFICATION — real Winyah stations answer**

```bash
~/.venvs/tidescout/bin/python <<'EOF'
from datetime import date
from tidescout.config import load_fishery
from tidescout.sources.cache import default_cache
from tidescout.sources import noaa

f = load_fishery("winyah-bay")
cache = default_cache()
tide_station = f.stations.tide[0]
print("tide hours:", len(noaa.tide_hours(tide_station, date.today(), f.timezone, cache)))
print("hilo:", [(e.kind, str(e.time)) for e in noaa.tide_events(tide_station, date.today(), f.timezone, cache)][:4])
for s in f.stations.currents:
    print("currents", s, len(noaa.current_hours(s, date.today(), f.timezone, cache)))
for w in f.stations.water:
    if w.kind == "coops":
        print("water temp", w.station, noaa.water_temp_latest(w.station, f.timezone, cache))
EOF
```

Expected: ~72 tide hours, 6–8 hilo events, currents hours if a current station was recorded. A `subordinate`-type tide station may reject `interval=h` (error mentions harmonic) — if so, keep hilo (subordinate stations support it) and note that hourly heights need a harmonic station; prefer a harmonic station in `stations.tide[0]` if discovery found one.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/sources/noaa.py backend/tests/test_noaa.py
git commit -m "feat: NOAA CO-OPS client for tide predictions, currents, water temp"
```

---

### Task 7: USGS client — discharge composite, water temp, salinity

**Files:**
- Create: `backend/tidescout/sources/usgs.py`
- Test: `backend/tests/test_usgs.py`

**Interfaces:**
- Consumes: `Fishery`, `Cache`.
- Produces:
  - `DischargeSummary` dataclass: `cfs_now: float | None`, `cfs_lagged: float | None` (mean of readings 24–48 h old), `bucket: str` (`"low" | "med" | "high"`, from `cfs_lagged` falling back to `cfs_now` vs `fishery.discharge_buckets`), `sites: list[str]`.
  - `WaterSummary` dataclass: `temp_f: float | None`, `temp_trend_f_3d: float | None` (latest-day mean minus mean of the 3 prior days), `salinity_ppt: float | None`, `source: str` (`"usgs:<site>"`, `"climatology"`, etc.).
  - `fetch_series(sites: list[str], params: list[str], period_days: int, cache: Cache) -> dict[tuple[str, str], list[tuple[datetime, float]]]` — key is `(site, param)`; times UTC-aware.
  - `discharge_summary(fishery: Fishery, cache: Cache) -> DischargeSummary`.
  - `water_summary(fishery: Fishery, cache: Cache, month: int) -> WaterSummary` — tries configured USGS water sensors (params `00010` temp °C → convert to °F, `00480` salinity); falls back to `fishery.climatology` values for `month` with `source="climatology"`.
  - USGS parameter codes used here: `00060` discharge cfs, `00010` water temp °C, `00480` salinity ppt.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_usgs.py`:

```python
from datetime import UTC, datetime, timedelta

import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources.usgs import discharge_summary, fetch_series, water_summary


def _ts(site: str, param: str, values: list[tuple[str, float]]) -> dict:
    return {
        "sourceInfo": {"siteCode": [{"value": site}]},
        "variable": {"variableCode": [{"value": param}]},
        "values": [{"value": [{"dateTime": t, "value": str(v)} for t, v in values]}],
    }


def _iv_payload(series: list[dict]) -> dict:
    return {"value": {"timeSeries": series}}


def _hours_ago(h: int) -> str:
    return (datetime.now(UTC) - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


@respx.mock
def test_fetch_series_parses(tmp_path):
    payload = _iv_payload([_ts("02131000", "00060", [(_hours_ago(2), 9000.0)])])
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=payload)
    )
    series = fetch_series(["02131000"], ["00060"], 7, Cache(tmp_path / "c.sqlite"))
    assert ("02131000", "00060") in series
    assert series[("02131000", "00060")][0][1] == 9000.0


@respx.mock
def test_discharge_summary_buckets(tmp_path):
    fishery = load_fishery("winyah-bay")
    sites = [r.usgs_site or f"0{i}FAKE" for i, r in enumerate(fishery.rivers)]
    series = [
        _ts(site, "00060", [(_hours_ago(36), 12000.0), (_hours_ago(1), 13000.0)])
        for site in sites
    ]
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload(series))
    )
    summary = discharge_summary(fishery, Cache(tmp_path / "c.sqlite"))
    assert summary.cfs_now == 13000.0 * len(sites)
    assert summary.cfs_lagged == 12000.0 * len(sites)
    assert summary.bucket == "high"  # 36000 > 25000 threshold


@respx.mock
def test_water_summary_falls_back_to_climatology(tmp_path):
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/iv/.*").mock(
        return_value=Response(200, json=_iv_payload([]))
    )
    fishery = load_fishery("winyah-bay")
    summary = water_summary(fishery, Cache(tmp_path / "c.sqlite"), month=8)
    assert summary.source == "climatology"
    assert summary.temp_f == fishery.climatology.water_temp_f_by_month[8]
    assert summary.salinity_ppt == fishery.climatology.salinity_ppt_by_month[8]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_usgs.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.sources.usgs).

- [ ] **Step 3: Implement the USGS client**

`backend/tidescout/sources/usgs.py`:

```python
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean

import httpx

from tidescout.models import Fishery
from tidescout.sources.cache import Cache

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
OBS_TTL = timedelta(minutes=15)

PARAM_DISCHARGE = "00060"
PARAM_TEMP_C = "00010"
PARAM_SALINITY = "00480"


@dataclass
class DischargeSummary:
    cfs_now: float | None
    cfs_lagged: float | None
    bucket: str
    sites: list[str]


@dataclass
class WaterSummary:
    temp_f: float | None
    temp_trend_f_3d: float | None
    salinity_ppt: float | None
    source: str


def fetch_series(
    sites: list[str], params: list[str], period_days: int, cache: Cache
) -> dict[tuple[str, str], list[tuple[datetime, float]]]:
    sites = [s for s in sites if s]
    if not sites:
        return {}
    query = {
        "format": "json",
        "sites": ",".join(sites),
        "parameterCd": ",".join(params),
        "period": f"P{period_days}D",
        "siteStatus": "all",
    }

    def fetch() -> dict:
        resp = httpx.get(IV_URL, params=query, timeout=30)
        resp.raise_for_status()
        return resp.json()

    key = f"{query['sites']}:{query['parameterCd']}:{period_days}"
    cached = cache.get_or_fetch("usgs-iv", key, OBS_TTL, fetch)
    out: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
    for ts in cached.payload.get("value", {}).get("timeSeries", []):
        site = ts["sourceInfo"]["siteCode"][0]["value"]
        param = ts["variable"]["variableCode"][0]["value"]
        points = []
        for block in ts.get("values", []):
            for p in block.get("value", []):
                try:
                    v = float(p["value"])
                except (TypeError, ValueError):
                    continue
                if v <= -999:  # USGS sentinel for missing
                    continue
                points.append((datetime.fromisoformat(p["dateTime"]).astimezone(UTC), v))
        if points:
            out[(site, param)] = sorted(points)
    return out


def discharge_summary(fishery: Fishery, cache: Cache) -> DischargeSummary:
    sites = [r.usgs_site for r in fishery.rivers if r.usgs_site]
    weights = {r.usgs_site: r.weight for r in fishery.rivers if r.usgs_site}
    series = fetch_series(sites, [PARAM_DISCHARGE], 4, cache)
    now = datetime.now(UTC)
    total_now = 0.0
    total_lagged = 0.0
    got_now = got_lagged = False
    for site in sites:
        points = series.get((site, PARAM_DISCHARGE), [])
        if not points:
            continue
        w = weights.get(site, 1.0)
        total_now += points[-1][1] * w
        got_now = True
        lag_window = [v for t, v in points if timedelta(hours=24) <= now - t <= timedelta(hours=48)]
        if lag_window:
            total_lagged += fmean(lag_window) * w
            got_lagged = True
    cfs_now = total_now if got_now else None
    cfs_lagged = total_lagged if got_lagged else None
    basis = cfs_lagged if cfs_lagged is not None else cfs_now
    if basis is None:
        bucket = "med"
    elif basis < fishery.discharge_buckets.low_below_cfs:
        bucket = "low"
    elif basis > fishery.discharge_buckets.high_above_cfs:
        bucket = "high"
    else:
        bucket = "med"
    return DischargeSummary(cfs_now, cfs_lagged, bucket, sites)


def _daily_means(points: list[tuple[datetime, float]]) -> dict:
    days: dict = {}
    for t, v in points:
        days.setdefault(t.date(), []).append(v)
    return {d: fmean(vs) for d, vs in days.items()}


def water_summary(fishery: Fishery, cache: Cache, month: int) -> WaterSummary:
    usgs_sensors = [w for w in fishery.stations.water if w.kind == "usgs"]
    temp_f = trend = salinity = None
    source = "climatology"
    if usgs_sensors:
        sites = [w.station for w in usgs_sensors]
        series = fetch_series(sites, [PARAM_TEMP_C, PARAM_SALINITY], 7, cache)
        for w in usgs_sensors:
            temp_points = series.get((w.station, PARAM_TEMP_C), [])
            if temp_points and temp_f is None:
                temp_f = temp_points[-1][1] * 9 / 5 + 32
                means = _daily_means(temp_points)
                days = sorted(means)
                if len(days) >= 4:
                    latest, prior = days[-1], days[-4:-1]
                    trend = (means[latest] - fmean([means[d] for d in prior])) * 9 / 5
                source = f"usgs:{w.station}"
            sal_points = series.get((w.station, PARAM_SALINITY), [])
            if sal_points and salinity is None:
                salinity = sal_points[-1][1]
    if temp_f is None:
        temp_f = fishery.climatology.water_temp_f_by_month[month]
        source = "climatology"
    if salinity is None:
        salinity = fishery.climatology.salinity_ppt_by_month[month]
    return WaterSummary(temp_f, trend, salinity, source)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_usgs.py -v`
Expected: 3 PASS.

- [ ] **Step 5: LIVE VERIFICATION — real gauges answer**

```bash
~/.venvs/tidescout/bin/python <<'EOF'
from datetime import date
from tidescout.config import load_fishery
from tidescout.sources.cache import default_cache
from tidescout.sources.usgs import discharge_summary, water_summary

f = load_fishery("winyah-bay")
cache = default_cache()
d = discharge_summary(f, cache)
print(f"discharge: now={d.cfs_now} lagged={d.cfs_lagged} bucket={d.bucket} sites={d.sites}")
w = water_summary(f, cache, month=date.today().month)
print(f"water: temp={w.temp_f} trend={w.temp_trend_f_3d} sal={w.salinity_ppt} source={w.source}")
EOF
```

Expected: real cfs numbers (composite Winyah system is typically thousands to tens of thousands cfs) and a sane bucket; water temp either from a USGS sensor (`source=usgs:...`) or climatology. If discharge looks wildly off, re-check the recorded gauge IDs from Task 4.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/sources/usgs.py backend/tests/test_usgs.py
git commit -m "feat: USGS client with discharge composite, water temp trend, salinity"
```

---

### Task 8: Astronomy — sun, moon, solunar periods

**Files:**
- Create: `backend/tidescout/sources/astronomy.py`
- Test: `backend/tests/test_astronomy.py`

**Interfaces:**
- Consumes: `Fishery` (for `center` and `timezone`). No cache needed — pure computation.
- Produces:
  - `SunTimes` dataclass: `dawn: datetime`, `sunrise: datetime`, `sunset: datetime`, `dusk: datetime` (all ET-aware).
  - `MoonInfo` dataclass: `phase_frac: float` (0–1 illuminated), `rise: datetime | None`, `set: datetime | None`, `transits: list[datetime]` (upper + lower culminations within the local day).
  - `SolunarPeriod` dataclass: `kind: str` (`"major" | "minor"`), `start: datetime`, `end: datetime`.
  - `sun_times(fishery: Fishery, day: date) -> SunTimes`
  - `moon_info(fishery: Fishery, day: date) -> MoonInfo`
  - `solunar_periods(fishery: Fishery, day: date) -> list[SolunarPeriod]` — majors = each transit ±1 h; minors = rise ±30 min and set ±30 min; sorted by start.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_astronomy.py`:

```python
from datetime import date, timedelta

from tidescout.config import load_fishery
from tidescout.sources.astronomy import moon_info, solunar_periods, sun_times

DAY = date(2026, 8, 15)


def _fishery():
    return load_fishery("winyah-bay")


def test_sun_times_ordered_and_plausible():
    s = sun_times(_fishery(), DAY)
    assert s.dawn < s.sunrise < s.sunset < s.dusk
    assert s.sunrise.date() == DAY
    # Mid-August sunrise on the SC coast is between 6:15 and 7:15 AM ET.
    assert 6 <= s.sunrise.hour <= 7


def test_moon_info_shape():
    m = moon_info(_fishery(), DAY)
    assert 0.0 <= m.phase_frac <= 1.0
    assert 1 <= len(m.transits) <= 3
    for t in m.transits:
        assert t.date() == DAY


def test_solunar_periods():
    periods = solunar_periods(_fishery(), DAY)
    kinds = {p.kind for p in periods}
    assert kinds <= {"major", "minor"}
    majors = [p for p in periods if p.kind == "major"]
    assert majors, "at least one lunar transit per day"
    for p in periods:
        expected = timedelta(hours=2) if p.kind == "major" else timedelta(hours=1)
        assert p.end - p.start == expected
    assert periods == sorted(periods, key=lambda p: p.start)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_astronomy.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.sources.astronomy).

- [ ] **Step 3: Implement astronomy**

`backend/tidescout/sources/astronomy.py`:

```python
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import ephem
from astral import LocationInfo
from astral.sun import sun as astral_sun

from tidescout.models import Fishery


@dataclass
class SunTimes:
    dawn: datetime
    sunrise: datetime
    sunset: datetime
    dusk: datetime


@dataclass
class MoonInfo:
    phase_frac: float
    rise: datetime | None
    set: datetime | None
    transits: list[datetime]


@dataclass(frozen=True)
class SolunarPeriod:
    kind: str  # "major" | "minor"
    start: datetime
    end: datetime


def _tz(fishery: Fishery) -> ZoneInfo:
    return ZoneInfo(fishery.timezone)


def sun_times(fishery: Fishery, day: date) -> SunTimes:
    lon, lat = fishery.center
    loc = LocationInfo("fishery", "SC", fishery.timezone, lat, lon)
    s = astral_sun(loc.observer, date=day, tzinfo=_tz(fishery))
    return SunTimes(dawn=s["dawn"], sunrise=s["sunrise"], sunset=s["sunset"], dusk=s["dusk"])


def _observer(fishery: Fishery, start_utc: datetime) -> ephem.Observer:
    obs = ephem.Observer()
    lon, lat = fishery.center
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.date = ephem.Date(start_utc.replace(tzinfo=None))
    return obs


def _to_local(edate: ephem.Date, tz: ZoneInfo) -> datetime:
    return edate.datetime().replace(tzinfo=UTC).astimezone(tz)


def moon_info(fishery: Fishery, day: date) -> MoonInfo:
    tz = _tz(fishery)
    day_start = datetime.combine(day, datetime.min.time(), tz)
    day_end = day_start + timedelta(days=1)
    start_utc = day_start.astimezone(UTC)

    moon = ephem.Moon()
    obs = _observer(fishery, start_utc)
    moon.compute(obs)
    phase_frac = float(moon.moon_phase)

    def first_in_day(method_name: str) -> datetime | None:
        o = _observer(fishery, start_utc)
        try:
            t = _to_local(getattr(o, method_name)(ephem.Moon()), tz)
        except (ephem.CircumpolarError, ephem.NeverUpError):
            return None
        return t if day_start <= t < day_end else None

    transits: list[datetime] = []
    for method in ("next_transit", "next_antitransit"):
        o = _observer(fishery, start_utc)
        cursor = start_utc
        while True:
            o.date = ephem.Date(cursor.replace(tzinfo=None))
            t = _to_local(getattr(o, method)(ephem.Moon()), tz)
            if t >= day_end:
                break
            if t >= day_start:
                transits.append(t)
            cursor = t.astimezone(UTC) + timedelta(minutes=1)

    return MoonInfo(
        phase_frac=phase_frac,
        rise=first_in_day("next_rising"),
        set=first_in_day("next_setting"),
        transits=sorted(transits),
    )


def solunar_periods(fishery: Fishery, day: date) -> list[SolunarPeriod]:
    info = moon_info(fishery, day)
    periods = [
        SolunarPeriod("major", t - timedelta(hours=1), t + timedelta(hours=1))
        for t in info.transits
    ]
    for edge in (info.rise, info.set):
        if edge is not None:
            periods.append(
                SolunarPeriod("minor", edge - timedelta(minutes=30), edge + timedelta(minutes=30))
            )
    return sorted(periods, key=lambda p: p.start)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_astronomy.py -v`
Expected: 3 PASS. (ephem returns UTC-naive dates; the `.replace(tzinfo=UTC)` conversion is what the date-membership assertions exercise.)

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/sources/astronomy.py backend/tests/test_astronomy.py
git commit -m "feat: sun, moon, and solunar period computation"
```

---

### Task 9: Conditions assembler (pure) + day loader + `conditions` CLI

**Files:**
- Create: `backend/tidescout/engine/__init__.py` (empty)
- Create: `backend/tidescout/engine/conditions.py`
- Create: `backend/tidescout/sources/dayloader.py`
- Modify: `backend/tidescout/cli.py` (add `conditions` command)
- Test: `backend/tests/test_conditions.py`

**Interfaces:**
- Consumes: everything above — `WeatherHour`, `TideHour`, `TideEvent`, `stage_at`, `CurrentHour`, `DischargeSummary`, `WaterSummary`, `SunTimes`, `MoonInfo`, `SolunarPeriod`, `Fishery`, `Cache`, `SourceUnavailable`, `fetch_weather`, `tide_hours`, `tide_events`, `current_hours`, `discharge_summary`, `water_summary`, `sun_times`, `moon_info`, `solunar_periods`, `load_fishery`, `default_cache`, `WEATHER_MODELS`.
- Produces:
  - `tidescout.engine.conditions.HourlyConditions` dataclass: `time: datetime`, `air_temp_f: float | None`, `wind_speed_kn: float | None`, `wind_dir_deg: float | None`, `wind_gust_kn: float | None`, `pressure_mb: float | None`, `pressure_trend_mb_3h: float | None`, `cloud_cover_pct: float | None`, `precip_in: float | None`, `tide_height_ft: float | None`, `tide_phase: str | None`, `tide_frac: float | None`, `current_speed_kn: float | None`, `current_dir_deg: float | None`, `solunar: list[str]`.
  - `tidescout.engine.conditions.DayConditions` dataclass: `fishery_slug: str`, `day: date`, `model_label: str`, `hours: list[HourlyConditions]` (exactly 24, or 23/25 on DST days), `sun: SunTimes | None`, `moon: MoonInfo | None`, `solunar: list[SolunarPeriod]`, `water: WaterSummary | None`, `discharge: DischargeSummary | None`, `missing: list[str]`.
  - `assemble_day(fishery, day, model_label, weather_48h, tides, events, currents, sun, moon, solunar, water, discharge, missing) -> DayConditions` — pure function, no I/O.
  - `tidescout.sources.dayloader.load_day(fishery: Fishery, day: date, model_key: str, cache: Cache) -> DayConditions` — calls every fetcher, catches `SourceUnavailable` per source, appends the source name to `missing`, passes `None`/empty for that piece.
  - CLI command `tidescout conditions SLUG --date YYYY-MM-DD --model MODELKEY`.

- [ ] **Step 1: Write the failing tests (assembler is pure — no HTTP mocking needed)**

`backend/tests/test_conditions.py`:

```python
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tidescout.config import load_fishery
from tidescout.engine.conditions import assemble_day
from tidescout.sources.noaa import CurrentHour, TideEvent, TideHour
from tidescout.sources.weather import WeatherHour

ET = ZoneInfo("America/New_York")
DAY = date(2026, 8, 15)
START = datetime(2026, 8, 14, 0, 0, tzinfo=ET)


def _weather_48h() -> list[WeatherHour]:
    hours = []
    for i in range(48):
        t = START + timedelta(hours=i)
        hours.append(
            WeatherHour(
                time=t, air_temp_f=80.0, wind_speed_kn=9.0, wind_dir_deg=220.0,
                wind_gust_kn=14.0, pressure_mb=1013.0 + 0.5 * i, cloud_cover_pct=40.0,
                precip_in=0.0,
            )
        )
    return hours


def _tides() -> list[TideHour]:
    return [
        TideHour(START + timedelta(hours=i), 2.0 + (i % 12) / 6) for i in range(72)
    ]


def _events() -> list[TideEvent]:
    out = []
    t = datetime(2026, 8, 14, 3, 0, tzinfo=ET)
    kind = "H"
    while t < datetime(2026, 8, 16, 12, 0, tzinfo=ET):
        out.append(TideEvent(t, kind, 5.0 if kind == "H" else 0.5))
        kind = "L" if kind == "H" else "H"
        t += timedelta(hours=6, minutes=12)
    return out


def test_assemble_day_shape_and_trend():
    f = load_fishery("winyah-bay")
    result = assemble_day(
        fishery=f, day=DAY, model_label="gfs", weather_48h=_weather_48h(),
        tides=_tides(), events=_events(), currents=[], sun=None, moon=None,
        solunar=[], water=None, discharge=None, missing=["currents"],
    )
    assert result.day == DAY
    assert len(result.hours) == 24
    assert all(h.time.date() == DAY for h in result.hours)
    # pressure rises 0.5 mb/hour in the fixture -> 3h trend == 1.5
    assert abs(result.hours[0].pressure_trend_mb_3h - 1.5) < 0.01
    assert result.hours[0].tide_phase in ("rising", "falling")
    assert result.hours[0].current_speed_kn is None
    assert result.missing == ["currents"]


def test_assemble_day_solunar_tags():
    from tidescout.sources.astronomy import SolunarPeriod

    f = load_fishery("winyah-bay")
    noon = datetime(2026, 8, 15, 12, 0, tzinfo=ET)
    periods = [SolunarPeriod("major", noon - timedelta(hours=1), noon + timedelta(hours=1))]
    result = assemble_day(
        fishery=f, day=DAY, model_label="gfs", weather_48h=_weather_48h(),
        tides=_tides(), events=_events(), currents=[], sun=None, moon=None,
        solunar=periods, water=None, discharge=None, missing=[],
    )
    tagged = [h.time.hour for h in result.hours if "major" in h.solunar]
    assert 11 in tagged and 12 in tagged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_conditions.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.engine.conditions).

- [ ] **Step 3: Implement assembler, loader, CLI**

`backend/tidescout/engine/conditions.py` (NOTE: this module must not import httpx or tidescout.sources.* — the type-only imports below come from dataclass modules, which is why `WeatherHour` etc. live in `sources` but contain no I/O; import them under `TYPE_CHECKING` to keep the runtime boundary clean):

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from tidescout.models import Fishery
    from tidescout.sources.astronomy import MoonInfo, SolunarPeriod, SunTimes
    from tidescout.sources.noaa import CurrentHour, TideEvent, TideHour
    from tidescout.sources.usgs import DischargeSummary, WaterSummary
    from tidescout.sources.weather import WeatherHour


@dataclass
class HourlyConditions:
    time: datetime
    air_temp_f: float | None = None
    wind_speed_kn: float | None = None
    wind_dir_deg: float | None = None
    wind_gust_kn: float | None = None
    pressure_mb: float | None = None
    pressure_trend_mb_3h: float | None = None
    cloud_cover_pct: float | None = None
    precip_in: float | None = None
    tide_height_ft: float | None = None
    tide_phase: str | None = None
    tide_frac: float | None = None
    current_speed_kn: float | None = None
    current_dir_deg: float | None = None
    solunar: list[str] = field(default_factory=list)


@dataclass
class DayConditions:
    fishery_slug: str
    day: date
    model_label: str
    hours: list[HourlyConditions]
    sun: SunTimes | None
    moon: MoonInfo | None
    solunar: list[SolunarPeriod]
    water: WaterSummary | None
    discharge: DischargeSummary | None
    missing: list[str]


def assemble_day(
    fishery: Fishery,
    day: date,
    model_label: str,
    weather_48h: list[WeatherHour],
    tides: list[TideHour],
    events: list[TideEvent],
    currents: list[CurrentHour],
    sun: SunTimes | None,
    moon: MoonInfo | None,
    solunar: list[SolunarPeriod],
    water: WaterSummary | None,
    discharge: DischargeSummary | None,
    missing: list[str],
) -> DayConditions:
    from tidescout.sources.noaa import stage_at  # pure function, no I/O

    tz = ZoneInfo(fishery.timezone)
    day_start = datetime.combine(day, datetime.min.time(), tz)
    day_end = day_start + timedelta(days=1)

    weather_by_time = {w.time: w for w in weather_48h}
    tide_by_time = {t.time: t for t in tides}
    current_by_time = {c.time: c for c in currents}

    hours: list[HourlyConditions] = []
    t = day_start
    while t < day_end:
        h = HourlyConditions(time=t)
        w = weather_by_time.get(t)
        if w:
            h.air_temp_f = w.air_temp_f
            h.wind_speed_kn = w.wind_speed_kn
            h.wind_dir_deg = w.wind_dir_deg
            h.wind_gust_kn = w.wind_gust_kn
            h.pressure_mb = w.pressure_mb
            h.cloud_cover_pct = w.cloud_cover_pct
            h.precip_in = w.precip_in
            w3 = weather_by_time.get(t - timedelta(hours=3))
            if w3 and w.pressure_mb is not None and w3.pressure_mb is not None:
                h.pressure_trend_mb_3h = round(w.pressure_mb - w3.pressure_mb, 2)
        tide = tide_by_time.get(t)
        if tide:
            h.tide_height_ft = tide.height_ft
        stage = stage_at(events, t) if events else None
        if stage:
            h.tide_phase = stage.phase
            h.tide_frac = round(stage.frac, 3)
        cur = current_by_time.get(t)
        if cur:
            h.current_speed_kn = cur.speed_kn
            h.current_dir_deg = cur.dir_deg
        hour_end = t + timedelta(hours=1)
        h.solunar = sorted(
            {p.kind for p in solunar if p.start < hour_end and p.end > t}
        )
        hours.append(h)
        t = hour_end

    return DayConditions(
        fishery_slug=fishery.slug, day=day, model_label=model_label, hours=hours,
        sun=sun, moon=moon, solunar=solunar, water=water, discharge=discharge,
        missing=missing,
    )
```

`backend/tidescout/sources/dayloader.py`:

```python
from datetime import date

from tidescout.engine.conditions import DayConditions, assemble_day
from tidescout.errors import SourceUnavailable
from tidescout.models import Fishery
from tidescout.sources import astronomy, noaa, usgs, weather
from tidescout.sources.cache import Cache


def load_day(fishery: Fishery, day: date, model_key: str, cache: Cache) -> DayConditions:
    missing: list[str] = []

    def attempt(name: str, fn, default):
        try:
            return fn()
        except SourceUnavailable:
            missing.append(name)
            return default
        except Exception:
            missing.append(name)
            return default

    weather_48h, label = attempt(
        "weather", lambda: weather.fetch_weather(fishery, day, model_key, cache), ([], model_key)
    )
    tide_station = fishery.stations.tide[0] if fishery.stations.tide else None
    tides = (
        attempt("tides", lambda: noaa.tide_hours(tide_station, day, fishery.timezone, cache), [])
        if tide_station
        else (missing.append("tides") or [])
    )
    events = (
        attempt("tide-events", lambda: noaa.tide_events(tide_station, day, fishery.timezone, cache), [])
        if tide_station
        else []
    )
    current_station = fishery.stations.currents[0] if fishery.stations.currents else None
    currents = (
        attempt(
            "currents", lambda: noaa.current_hours(current_station, day, fishery.timezone, cache), []
        )
        if current_station
        else (missing.append("currents") or [])
    )
    sun = attempt("sun", lambda: astronomy.sun_times(fishery, day), None)
    moon = attempt("moon", lambda: astronomy.moon_info(fishery, day), None)
    solunar = attempt("solunar", lambda: astronomy.solunar_periods(fishery, day), [])
    water = attempt("water", lambda: usgs.water_summary(fishery, cache, day.month), None)
    discharge = attempt("discharge", lambda: usgs.discharge_summary(fishery, cache), None)

    return assemble_day(
        fishery=fishery, day=day, model_label=label, weather_48h=weather_48h,
        tides=tides, events=events, currents=currents, sun=sun, moon=moon,
        solunar=solunar, water=water, discharge=discharge, missing=missing,
    )
```

Append to `backend/tidescout/cli.py`:

```python
@app.command()
def conditions(
    slug: str,
    date_str: str = typer.Option(None, "--date", help="YYYY-MM-DD (default: today)"),
    model: str = typer.Option("best", "--model", help="best|gfs|ecmwf|icon|hrrr|nbm"),
) -> None:
    """Print a day of hourly conditions for a fishery."""
    from datetime import date as date_cls

    from tidescout.config import load_fishery
    from tidescout.sources.cache import default_cache
    from tidescout.sources.dayloader import load_day
    from tidescout.sources.weather import WEATHER_MODELS

    if model not in WEATHER_MODELS:
        raise typer.BadParameter(f"model must be one of {sorted(WEATHER_MODELS)}")
    day = date_cls.fromisoformat(date_str) if date_str else date_cls.today()
    fishery = load_fishery(slug)
    result = load_day(fishery, day, model, default_cache())

    table = Table(title=f"{fishery.name} — {day} — model: {result.model_label}")
    for col in ("hour", "tide ft", "stage", "cur kn", "wind", "gust", "press", "trend", "cloud", "air°F", "solunar"):
        table.add_column(col, justify="right")
    for h in result.hours:
        arrow = {"rising": "↑", "falling": "↓"}.get(h.tide_phase or "", "")
        wind = (
            f"{h.wind_speed_kn:.0f}@{h.wind_dir_deg:.0f}"
            if h.wind_speed_kn is not None and h.wind_dir_deg is not None
            else "—"
        )
        table.add_row(
            h.time.strftime("%H:%M"),
            f"{h.tide_height_ft:.1f}" if h.tide_height_ft is not None else "—",
            f"{arrow}{h.tide_frac:.0%}" if h.tide_frac is not None else "—",
            f"{h.current_speed_kn:+.1f}" if h.current_speed_kn is not None else "—",
            wind,
            f"{h.wind_gust_kn:.0f}" if h.wind_gust_kn is not None else "—",
            f"{h.pressure_mb:.1f}" if h.pressure_mb is not None else "—",
            f"{h.pressure_trend_mb_3h:+.1f}" if h.pressure_trend_mb_3h is not None else "—",
            f"{h.cloud_cover_pct:.0f}%" if h.cloud_cover_pct is not None else "—",
            f"{h.air_temp_f:.0f}" if h.air_temp_f is not None else "—",
            ",".join(h.solunar) or "—",
        )
    console.print(table)
    if result.sun:
        console.print(
            f"sun: dawn {result.sun.dawn:%H:%M} rise {result.sun.sunrise:%H:%M} "
            f"set {result.sun.sunset:%H:%M} dusk {result.sun.dusk:%H:%M}"
        )
    if result.moon:
        console.print(f"moon: {result.moon.phase_frac:.0%} illuminated, transits {[t.strftime('%H:%M') for t in result.moon.transits]}")
    if result.water:
        trend = f" ({result.water.temp_trend_f_3d:+.1f}°F/3d)" if result.water.temp_trend_f_3d is not None else ""
        console.print(f"water: {result.water.temp_f:.0f}°F{trend}, salinity {result.water.salinity_ppt:.0f} ppt [{result.water.source}]")
    if result.discharge:
        console.print(f"discharge: {result.discharge.bucket} ({result.discharge.cfs_now:,.0f} cfs now)" if result.discharge.cfs_now else f"discharge: {result.discharge.bucket}")
    if result.missing:
        console.print(f"[yellow]missing sources: {', '.join(result.missing)}[/yellow]")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_conditions.py -v`
Expected: 2 PASS.

- [ ] **Step 5: LIVE VERIFICATION — the deliverable**

```bash
~/.venvs/tidescout/bin/tidescout conditions winyah-bay --model gfs
~/.venvs/tidescout/bin/tidescout conditions winyah-bay --date 2026-08-01 --model ecmwf
~/.venvs/tidescout/bin/tidescout conditions winyah-bay --date 2025-06-15
```

Expected: three rendered tables. First: today with real tide heights, stages, wind, pressure trends. Second: recent past via forecast API. Third: `model: era5` label (archive routing). Any missing source appears as a yellow warning line, not a crash. Sanity-check tide stage arrows against the heights column (heights climb while stage shows ↑).

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/engine backend/tidescout/sources/dayloader.py backend/tidescout/cli.py backend/tests/test_conditions.py
git commit -m "feat: day conditions assembler and conditions CLI"
```

---

### Task 10: Full gate, docs, wrap-up

**Files:**
- Modify: `README.md` (add run instructions)

**Interfaces:**
- Consumes: everything.
- Produces: Plan 1 complete; Plan 2 (bathymetry + static features) starts from this commit.

- [ ] **Step 1: Run the full gate**

```bash
make check
```

Expected: ruff clean, all ~29 tests pass. Fix anything that fails before proceeding.

- [ ] **Step 2: Add run instructions to `README.md`**

Append this section:

```markdown
## Running (current state: conditions CLI)

    # one-time setup
    uv venv ~/.venvs/tidescout --python 3.12
    make install

    # discover/refresh station IDs for a fishery
    ~/.venvs/tidescout/bin/tidescout stations winyah-bay

    # a day of hourly conditions (tide, currents, wind, pressure, solunar...)
    ~/.venvs/tidescout/bin/tidescout conditions winyah-bay --date 2026-08-15 --model ecmwf

    # quality gate
    make check
```

- [ ] **Step 3: Verify the README instructions verbatim**

Run each command from the README block exactly as written (venv already exists; `uv venv` is idempotent enough to skip — run the other four). Expected: all succeed.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: run instructions for conditions CLI"
```

---

## Self-review checklist (for the plan author — already applied)

- Spec coverage for Phase 0+1: config-driven fisheries (§3), all Phase-1 data sources with caching TTLs (§4), stale-fallback + missing-source flagging (§10), no-live-API testing (§11) — covered. NERR/CDMO explicitly deferred (registration wall) per Global Constraints.
- Placeholders: none — every step has runnable code or an exact command; unknown-at-planning-time values (station IDs, possibly `ncep_nbm_conus`) are resolved by LIVE VERIFICATION steps with explicit fix procedures.
- Type consistency: `Cached`/`Cache.get_or_fetch(source, key, ttl, fetch)` signatures match across Tasks 3–7; `WeatherHour`/`TideHour`/`TideEvent`/`CurrentHour`/`DischargeSummary`/`WaterSummary`/`SunTimes`/`MoonInfo`/`SolunarPeriod` definitions match their uses in Task 9; CLI names (`stations`, `conditions`) consistent.
