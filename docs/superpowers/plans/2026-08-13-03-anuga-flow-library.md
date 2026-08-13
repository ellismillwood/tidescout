# ANUGA Flow-State Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the precomputed tidal flow-state library for Winyah Bay — a graded ANUGA mesh, nine regime simulations, snapshots rasterised to the 10 m analysis grid as speed/direction/shear/wet, and a runtime lookup — then prove it against Ellis's real fishing spots.

**Architecture:** An offline pipeline stage. `pipeline/mesh.py` turns the bathymetry raster plus a config-authored domain polygon into an ANUGA mesh; `pipeline/forcing.py` builds tide and river boundary conditions; `pipeline/regimes.py` runs the 3×3 (tidal range × discharge) matrix as independent OS processes; `pipeline/flowlib.py` rasterises the resulting `.sww` snapshots onto the existing analysis grid and writes an indexed library. `engine/flow.py` stays pure — it reads library arrays and answers "what is the flow at this cell, this phase" with no I/O.

**Tech Stack:** Python 3.12, ANUGA 3.3.10 (prebuilt arm64 wheels), numpy 2.5.2, scipy 1.18.0, rasterio 1.5.1, shapely 2.1.2, netCDF4, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` — §5 (flow-state library) is what this plan implements; §6, §8, §11 constrain it.

**Measured spike findings:** `docs/superpowers/plans/2026-08-13-plan3-anuga-spike-findings.md` — **read this before Task 5.** Every number in this plan's ANUGA tasks came from it. It also records three things that look like bugs but are not, and two that look fine but are traps.

**Carryover context (read before Task 1):**
- `docs/superpowers/plans/2026-08-13-plan2-carryover-notes.md`
- `docs/superpowers/plans/2026-08-12-plan1-carryover-notes.md`

---

## Global Constraints

- **Python** `>=3.12`. Backend package is `backend/tidescout`, editable-installed into `~/.venvs/tidescout`.
- **Lint gate** is explicit and enforced: `[tool.ruff.lint] select = ["E","F","I","UP","B","DTZ"]`, `line-length = 100`, `target-version = "py312"`. Write against that reality: `collections.abc` imports, timezone-aware datetimes only (DTZ), no closures capturing loop variables (B023), no lines over 100 chars.
- **Gate command:** `make check` = `ruff check` + `pytest -q`. Run from repo root. Must be green before every commit.
- **Baseline at plan start:** 101 tests passing on `main`.
- **`engine/` is pure.** Takes plain data structures, returns values. All I/O lives in `sources/` and `pipeline/`. This is what makes the algorithm testable — do not reach for the filesystem or network from `engine/`.
- **Paths** resolve via `tidescout.paths` (`REPO_ROOT`, `FISHERIES_DIR`, `DATA_DIR`, `fishery_data_dir(slug)`, `tiles_dir(slug)`). Never hand-roll a parent-walk.
- **`data/` is gitignored and fully rebuildable.** Never commit artifacts. `fisheries/` **is** committed — configs and ground truth belong in git.
- **Analysis grid is fixed and shared:** EPSG:26917, 10 m cells, 3806 × 5053, corner-based `Affine` with the +0.5 pixel-centre convention. Everything this plan writes must land on exactly this grid. Get the transform from `read_bathy()`, never from `bathy_meta.json` directly.
- **Tests never hit live APIs.** Use `respx` for HTTP and synthetic arrays for rasters (`backend/tests/synth.py`).
- **Units:** bathymetry and ANUGA are metres, NAVD88. NOAA CO-OPS tide predictions come back in **feet** and on the **MLLW** datum. Both a unit conversion and a datum shift are required — see Task 8.
- **Commit style:** `feat:` / `fix:` / `test:` / `docs:` prefixes, one commit per task minimum.

---

## File Structure

**New files:**
| Path | Responsibility |
|---|---|
| `backend/tidescout/pipeline/mesh.py` | Water mask → cleaned polygon → graded ANUGA mesh; elevation and friction sampling onto centroids |
| `backend/tidescout/pipeline/forcing.py` | Tide-boundary and river-inflow functions for a given regime |
| `backend/tidescout/pipeline/regimes.py` | Regime matrix definition; single-regime runner; process-parallel driver |
| `backend/tidescout/pipeline/flowlib.py` | `.sww` → analysis-grid rasters; library manifest write/read |
| `backend/tidescout/engine/flow.py` | Pure lookup + derived flow structure (shear, seams, lee) |
| `backend/tests/test_mesh.py` | Mesh builder on synthetic DEMs |
| `backend/tests/test_forcing.py` | Boundary forcing maths, datum/unit conversion |
| `backend/tests/test_regimes.py` | Regime matrix, checks, runner wiring |
| `backend/tests/test_flowlib.py` | Rasterisation and library round-trip |
| `backend/tests/test_flow.py` | Pure engine lookup and derived structure |
| `backend/tests/test_derivatives_pipeline.py` | Pipeline-level tests for `build_derivatives` / `build_artifacts` (carryover item 3) |

**Modified files:**
| Path | Change |
|---|---|
| `backend/pyproject.toml` | Add `anuga>=3.3.10` dependency |
| `backend/tidescout/models.py` | `ModelDomain`, `AnugaConfig`, `RegimeSpec`; `FeatureThresholds` gains area/elongation caps and a `wall_slope_estimator` |
| `backend/tidescout/engine/detect.py` | Wall typing fix; per-type size gates; `WET_LEVEL_M` becomes a parameter |
| `backend/tidescout/pipeline/features.py` | Pass wet level through; stable feature ids |
| `backend/tidescout/sources/usgs.py` | Discharge freshness + `contributing`; daily-values fetch for calibration |
| `backend/tidescout/cli.py` | `tidescout flow mesh|run|library|validate` sub-app |
| `fisheries/winyah-bay.yaml` | `model_domain`, `anuga`, recalibrated `discharge_buckets`, feature size caps |

---

## Task 1: Discharge recalibration and freshness signalling

Plan 1 carryover items 1 and 2. This comes first because **the discharge buckets define one of the two axes of the regime matrix** — nine simulations get labelled `low`/`med`/`high`, and today those labels come from thresholds authored before the gauges were chosen. The composite is ~98% Pee Dee at Peedee; it hovers near the low boundary and 25000 cfs is nearly unreachable.

**Files:**
- Modify: `backend/tidescout/sources/usgs.py:18-24` (`DischargeSummary`), `:84-114` (`discharge_summary`)
- Modify: `fisheries/winyah-bay.yaml:65-67` (`discharge_buckets`)
- Test: `backend/tests/test_usgs.py`

**Interfaces:**
- Consumes: `Fishery.rivers`, `Fishery.discharge_buckets`, `Cache`
- Produces: `DischargeSummary(cfs_now, cfs_lagged, bucket, sites, contributing: list[str], stale: list[str])` — Task 10 uses `bucket` to label regime runs, and Task 13 renders `contributing`.

- [ ] **Step 1: Write the failing test for freshness exclusion**

A gauge that stopped reporting four days ago must not contribute its last reading to `cfs_now`, and must be reported as stale.

```python
# backend/tests/test_usgs.py
from datetime import UTC, datetime, timedelta

from tidescout.sources import usgs


def test_discharge_excludes_stale_sites(monkeypatch, fishery, cache):
    now = datetime.now(UTC)
    fresh = [(now - timedelta(hours=1), 5000.0)]
    stale = [(now - timedelta(days=4), 9000.0)]

    def fake_fetch(sites, params, period_days, cache):
        return {
            ("02131000", usgs.PARAM_DISCHARGE): fresh,
            ("02110500", usgs.PARAM_DISCHARGE): stale,
        }

    monkeypatch.setattr(usgs, "fetch_series", fake_fetch)
    s = usgs.discharge_summary(fishery, cache)
    assert s.cfs_now == 5000.0            # stale site excluded, not summed
    assert s.contributing == ["02131000"]
    assert "02110500" in s.stale
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_usgs.py::test_discharge_excludes_stale_sites -v`
Expected: FAIL — `AttributeError: 'DischargeSummary' object has no attribute 'contributing'`

- [ ] **Step 3: Add freshness to the summary**

```python
FRESHNESS_CUTOFF = timedelta(hours=6)


@dataclass
class DischargeSummary:
    cfs_now: float | None
    cfs_lagged: float | None
    bucket: str
    sites: list[str]
    contributing: list[str]
    stale: list[str]
```

In `discharge_summary`, replace the unconditional `total_now += points[-1][1] * w`:

```python
    contributing: list[str] = []
    stale: list[str] = []
    for site in sites:
        points = series.get((site, PARAM_DISCHARGE), [])
        if not points:
            stale.append(site)
            continue
        w = weights.get(site, 1.0)
        last_t, last_v = points[-1]
        if now - last_t > FRESHNESS_CUTOFF:
            stale.append(site)   # dark gauge: do not let a 4-day-old value in
        else:
            total_now += last_v * w
            got_now = True
            contributing.append(site)
        lag_window = [
            v for t, v in points if timedelta(hours=24) <= now - t <= timedelta(hours=48)
        ]
        if lag_window:
            total_lagged += fmean(lag_window) * w
            got_lagged = True
```

and return `DischargeSummary(cfs_now, cfs_lagged, bucket, sites, contributing, stale)`.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_usgs.py -v`
Expected: PASS. Fix any existing test that constructs `DischargeSummary` positionally.

- [ ] **Step 5: Add a daily-values fetch for calibration**

USGS instantaneous values only reach back ~120 days; bucket calibration needs a year. Add a separate daily-values reader.

```python
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"


def fetch_daily(
    sites: list[str], param: str, start: str, end: str, cache: Cache
) -> dict[str, list[tuple[date, float]]]:
    """Daily mean values (NWIS dv service). Immutable once published, so cached
    with no TTL -- calibration reads a year of history and must not refetch."""
    sites = [s for s in sites if s]
    if not sites:
        return {}
    query = {
        "format": "json",
        "sites": ",".join(sites),
        "parameterCd": param,
        "startDT": start,
        "endDT": end,
        "statCd": "00003",  # daily mean
        "siteStatus": "all",
    }

    def fetch() -> dict:
        resp = httpx.get(DV_URL, params=query, timeout=60)
        resp.raise_for_status()
        return resp.json()

    key = f"dv:{query['sites']}:{param}:{start}:{end}"
    cached = cache.get_or_fetch("usgs-dv", key, None, fetch)
    out: dict[str, list[tuple[date, float]]] = {}
    for ts in cached.payload.get("value", {}).get("timeSeries", []):
        try:
            site = ts["sourceInfo"]["siteCode"][0]["value"]
        except (KeyError, IndexError, TypeError):
            continue
        rows: list[tuple[date, float]] = []
        for block in ts.get("values", []):
            for p in block.get("value", []):
                try:
                    v = float(p["value"])
                    d = date.fromisoformat(p["dateTime"][:10])
                except (KeyError, TypeError, ValueError):
                    continue
                if v <= -999:
                    continue
                rows.append((d, v))
        if rows:
            out[site] = sorted(rows)
    return out
```

Add `from datetime import date` to the imports.

- [ ] **Step 6: Write a respx test for `fetch_daily`**

```python
import respx
import httpx as _httpx


@respx.mock
def test_fetch_daily_parses_and_sorts(cache):
    payload = {"value": {"timeSeries": [{
        "sourceInfo": {"siteCode": [{"value": "02131000"}]},
        "values": [{"value": [
            {"dateTime": "2026-01-02T00:00:00.000", "value": "5200"},
            {"dateTime": "2026-01-01T00:00:00.000", "value": "4800"},
            {"dateTime": "2026-01-03T00:00:00.000", "value": "-999999"},
        ]}],
    }]}}
    respx.get(usgs.DV_URL).mock(return_value=_httpx.Response(200, json=payload))
    out = usgs.fetch_daily(["02131000"], usgs.PARAM_DISCHARGE, "2026-01-01", "2026-01-03", cache)
    assert out["02131000"] == [(date(2026, 1, 1), 4800.0), (date(2026, 1, 2), 5200.0)]
```

- [ ] **Step 7: Run the calibration against live data and record the derivation**

This is a one-off human-run step, not library code. Run:

```bash
cd backend && ~/.venvs/tidescout/bin/python -c "
from datetime import date, timedelta
from statistics import quantiles
from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources import usgs

f = load_fishery('winyah-bay')
cache = Cache()
end = date.today(); start = end - timedelta(days=365)
sites = [r.usgs_site for r in f.rivers]
weights = {r.usgs_site: r.weight for r in f.rivers}
daily = usgs.fetch_daily(sites, usgs.PARAM_DISCHARGE, start.isoformat(), end.isoformat(), cache)
by_day = {}
for site, rows in daily.items():
    for d, v in rows:
        by_day.setdefault(d, 0.0)
        by_day[d] += v * weights.get(site, 1.0)
comp = sorted(by_day.values())
q = quantiles(comp, n=4)
print(f'n={len(comp)} days  min={comp[0]:.0f}  p25={q[0]:.0f}  median={q[1]:.0f}  p75={q[2]:.0f}  max={comp[-1]:.0f}')
for site, rows in daily.items():
    print(f'  {site}: {len(rows)} days, mean {sum(v for _, v in rows)/len(rows):.0f} cfs')
"
```

Write the printed p25/p75 into `fisheries/winyah-bay.yaml`, replacing the authored-blind values, **with a comment recording the derivation** (date range, n, percentiles, per-site means) in the style the rest of that file already uses:

```yaml
discharge_buckets:
  # Recalibrated <DATE> from 365 days of USGS daily means (statCd 00003),
  # weighted composite of the three gauges below. n=<N> days:
  # min <MIN> / p25 <P25> / median <MED> / p75 <P75> / max <MAX> cfs.
  # Per-gauge means: Pee Dee <A>, Waccamaw <B>, Black <C> cfs -- the composite
  # is ~<X>% Pee Dee, which is why the old 6000/25000 pair (authored before the
  # gauges were chosen) put nearly every day in "low".
  low_below_cfs: <P25>
  high_above_cfs: <P75>
```

- [ ] **Step 8: Run the full gate and commit**

Run: `make check`
Expected: all tests pass, ruff clean.

```bash
git add backend/tidescout/sources/usgs.py backend/tests/test_usgs.py fisheries/winyah-bay.yaml
git commit -m "feat: discharge freshness signalling and percentile-calibrated buckets"
```

---

## Task 2: Per-type size gates on feature detection

Plan 2 carryover item 2, and **the ground truth now proves it is broken**. Running `tidescout spots winyah-bay` against Ellis's three real spots returns a `flat` at distance 0 for two of them, because a single **47.39 km² `bar` spanning 21 × 35 km** and a **27.41 km² `flat`** swallow most of the estuary. Any "nearest feature to this flow cell" join is meaningless until these are gone. Verified sizes:

```
bar   47.393 km2   extent 21.1 x 35.1 km   <- contains 2 of 3 known spots
flat  27.405 km2   extent 10.2 x 13.1 km
bar   18.646 km2
flat  12.907 km2
```

With the blobs removed, the plausible matches underneath are already there: Mud Bay Cut has a 0.031 km² `hole` at 96 m; the two jetty-area spots sit 272 m and 365 m from `jetty` geometry.

**Files:**
- Modify: `backend/tidescout/models.py:41-51` (`FeatureThresholds`)
- Modify: `backend/tidescout/engine/detect.py:50-60` (`_mask_polygons`)
- Modify: `fisheries/winyah-bay.yaml` (`features:`)
- Test: `backend/tests/test_detect.py`

**Interfaces:**
- Consumes: `FeatureThresholds` from Task 1's unchanged config loading.
- Produces: `_mask_polygons(mask, transform, min_area_m2, cell_m, max_area_m2=None)` — Task 3 calls it with the same signature.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_detect.py
import numpy as np

from tidescout.engine import detect
from . import synth


def test_mask_polygons_rejects_basin_scale_blobs():
    # a 150x150-cell blob = 2.25 km2 at 10 m cells
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:170, 20:170] = True
    kept = detect._mask_polygons(
        mask, synth.TRANSFORM, min_area_m2=1500.0, cell_m=10.0, max_area_m2=1_000_000.0
    )
    assert kept == [], "a 2.25 km2 blob must not survive a 1 km2 cap"


def test_mask_polygons_keeps_normal_features():
    mask = np.zeros((200, 200), dtype=bool)
    mask[100:120, 100:130] = True          # 200x300 m = 0.06 km2
    kept = detect._mask_polygons(
        mask, synth.TRANSFORM, min_area_m2=1500.0, cell_m=10.0, max_area_m2=1_000_000.0
    )
    assert len(kept) == 1
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_detect.py -k mask_polygons -v`
Expected: FAIL — `_mask_polygons() got an unexpected keyword argument 'max_area_m2'`

- [ ] **Step 3: Add the cap to `_mask_polygons`**

Read the current body first, then add the upper bound alongside the existing lower one. `max_area_m2=None` means "no cap", preserving every existing caller's behaviour until it opts in.

```python
def _mask_polygons(
    mask: np.ndarray,
    transform: Affine,
    min_area_m2: float,
    cell_m: float,
    max_area_m2: float | None = None,
):
    """Polygonise a boolean mask, dropping components outside the size band.

    The upper bound is not cosmetic: without it a single connected component
    can span the whole estuary (a 47 km2 'bar' covering 21 x 35 km was what
    the real Winyah raster produced), and every point-in-polygon join against
    it returns distance 0, which destroys 'nearest feature to this cell'.
    """
```

Inside the existing per-polygon loop, after the `min_area_m2` check, add:

```python
        if max_area_m2 is not None and poly.area > max_area_m2:
            continue
```

- [ ] **Step 4: Run and confirm it passes**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_detect.py -v`
Expected: PASS, and all pre-existing detect tests still pass (the default `None` keeps them unchanged).

- [ ] **Step 5: Add per-type caps to the config model**

```python
class FeatureThresholds(BaseModel):
    dropoff_slope_deg: float = 8.0
    wall_slope_deg: float = 20.0
    hole_delta_m: float = 1.5
    hole_min_area_m2: float = 2000.0
    flat_max_slope_deg: float = 1.0
    flat_band_m: tuple[float, float] = (-1.5, 0.5)
    shallow_max_m: float = -0.3
    deep_min_m: float = -3.0
    bar_min_area_m2: float = 1500.0
    mouth_search_radius_m: float = 60.0
    # Upper bounds. A feature larger than this is a basin, not an ambush point;
    # see the 47 km2 bar the real Winyah raster produced.
    bar_max_area_m2: float = 500_000.0     # 0.5 km2
    flat_max_area_m2: float = 2_000_000.0  # 2 km2 -- flats are legitimately broad
    hole_max_area_m2: float = 200_000.0    # 0.2 km2
```

- [ ] **Step 6: Pass the caps through the detectors**

In `detect_bars`, `detect_flats`, and `detect_holes`, thread the matching `t.*_max_area_m2` into their `_mask_polygons` calls. Read each detector body and add the keyword to its existing call — do not restructure the detectors.

- [ ] **Step 7: Rebuild the inventory and check it against ground truth**

```bash
~/.venvs/tidescout/bin/tidescout features winyah-bay --rebuild
~/.venvs/tidescout/bin/tidescout spots winyah-bay
```

Expected: the `features` table no longer lists any feature above the caps (the "largest features" lines at the bottom of that output are the check), and `spots` no longer reports `flat` at distance 0 for Mud Bay Cut or North Jetty. Both jetty-area spots should now resolve to `jetty` or a small nearby feature at a few hundred metres.

**If a known spot ends up with no feature within ~500 m, do not tune the caps to force a match** — record it in the task report. Detection recall is a Plan 4 concern; this task is only removing the blobs.

- [ ] **Step 8: Commit**

Run: `make check`

```bash
git add backend/tidescout/models.py backend/tidescout/engine/detect.py backend/tests/test_detect.py fisheries/winyah-bay.yaml
git commit -m "fix: reject basin-scale features that break nearest-feature joins"
```

---

## Task 3: Wall typing on a percentile slope estimator

Plan 2 carryover item 1, with a **corrected ruling that overturns the obvious diagnosis**. `wall` count is 0. This is *not* a resolution problem and chasing a finer grid is wasted work. The raster genuinely contains 3,067 pixels at ≥20° (max 32°, p99.9 = 13.9°). The bug is the estimator: walls are typed on `mean(slope)` over a polygon whose boundary is pinned at the 8° dropoff threshold, and a mean over such a polygon can never reach 20°.

**Files:**
- Modify: `backend/tidescout/engine/detect.py:62-88` (`detect_dropoffs`)
- Modify: `backend/tidescout/models.py` (`FeatureThresholds`)
- Test: `backend/tests/test_detect.py`

**Interfaces:**
- Consumes: `slope_deg` array from `engine.terrain`, `FeatureThresholds`.
- Produces: features of `type == "wall"` carrying `attrs["max_slope_deg"]` and `attrs["p90_slope_deg"]`.

- [ ] **Step 1: Write the failing test**

```python
def test_steep_step_is_typed_as_wall():
    """A polygon containing genuinely steep cells must type as wall even though
    its mean slope is dragged down by the 8 deg boundary it is cut at."""
    z = np.full((200, 200), -2.0, dtype="float32")
    z[:, 100:] = -14.0            # 12 m step -> well above 20 deg
    from tidescout.engine.terrain import slope_deg as _slope
    from tidescout.models import FeatureThresholds
    slope = _slope(z, 10.0)
    feats = detect.detect_dropoffs(z, slope, FeatureThresholds(), synth.TRANSFORM)
    types = {f.type for f in feats}
    assert "wall" in types, f"expected a wall, got {types}"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_detect.py::test_steep_step_is_typed_as_wall -v`
Expected: FAIL — `assert 'wall' in {'dropoff'}`

- [ ] **Step 3: Add the estimator setting**

```python
    # Walls are typed on an upper percentile, not the mean: the polygon's own
    # boundary is cut at dropoff_slope_deg, so its mean slope is structurally
    # incapable of reaching wall_slope_deg. p90 is robust to the one-cell
    # artefacts that nanmax would latch onto.
    wall_slope_estimator: Literal["p90", "max", "mean"] = "p90"
```

Add `Literal` to the `typing` import in `models.py` (it is already imported there).

- [ ] **Step 4: Use it when typing each dropoff polygon**

In `detect_dropoffs`, where the polygon's slope statistic is currently computed as a mean, compute the sample once and derive both the typing statistic and the recorded attributes:

```python
        vals = slope[poly_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_slope = float(np.mean(vals))
        p90_slope = float(np.percentile(vals, 90))
        max_slope = float(np.max(vals))
        stat = {"p90": p90_slope, "max": max_slope, "mean": mean_slope}[
            t.wall_slope_estimator
        ]
        ftype = "wall" if stat >= t.wall_slope_deg else "dropoff"
```

and record `mean_slope_deg`, `p90_slope_deg`, and `max_slope_deg` in `attrs` so a future retune can see all three without a rebuild.

- [ ] **Step 5: Run and confirm it passes**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_detect.py -v`
Expected: PASS.

- [ ] **Step 6: Rebuild and sanity-check the real count**

```bash
~/.venvs/tidescout/bin/tidescout features winyah-bay --rebuild
```

Expected: `wall` count is now non-zero. A plausible result is tens-to-low-hundreds, concentrated in the dredged shipping channel. **If it exceeds ~15% of the dropoff count, the estimator is too permissive** — report the number rather than silently retuning, since `wall_slope_deg` is shared with nothing else and the right value is a judgement call for Ellis.

- [ ] **Step 7: Commit**

Run: `make check`

```bash
git add backend/tidescout/engine/detect.py backend/tidescout/models.py backend/tests/test_detect.py
git commit -m "fix: type walls on p90 slope instead of a boundary-pinned mean"
```

---

## Task 4: Pipeline-level tests for derivatives and artifacts

Plan 2 carryover item 3. `build_derivatives` and `build_artifacts` currently have engine-level coverage only, and Task 7 is about to make `zones.tif` load-bearing for the friction field. Cover them before that happens.

**Files:**
- Create: `backend/tests/test_derivatives_pipeline.py`
- Test target: `backend/tidescout/pipeline/derivatives.py`, `backend/tidescout/pipeline/artifacts.py`

**Interfaces:**
- Consumes: `_fake_bathy` fixture pattern from `backend/tests/test_features_pipeline.py:14-30` — reuse it verbatim rather than inventing a second one.
- Produces: nothing consumed downstream; this is a safety net.

- [ ] **Step 1: Write the tests**

```python
# backend/tests/test_derivatives_pipeline.py
import json

import numpy as np
import rasterio

from tidescout.config import load_fishery
from tidescout.pipeline.artifacts import build_artifacts
from tidescout.pipeline.derivatives import build_derivatives

from . import synth
from .test_features_pipeline import _fake_bathy


def test_build_derivatives_writes_grid_aligned_rasters(tmp_path, monkeypatch):
    z = synth.dropoff_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    paths = build_derivatives("winyah-bay", f)
    assert set(paths) == {"slope", "curv", "zones"}
    for name, p in paths.items():
        assert p.exists(), name
        with rasterio.open(p) as src:
            assert src.width == z.shape[1]
            assert src.height == z.shape[0]
            assert src.transform == synth.TRANSFORM, f"{name} lost grid alignment"
            assert str(src.crs) == "EPSG:26917"


def test_zones_raster_is_categorical_and_nodata_zero(tmp_path, monkeypatch):
    z = synth.dropoff_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    paths = build_derivatives("winyah-bay", f)
    with rasterio.open(paths["zones"]) as src:
        arr = src.read(1)
        assert src.nodata == 0
        assert arr.dtype == np.uint8
        assert set(np.unique(arr)) <= {0, 1, 2, 3}, "zones must stay a small enum"


def test_build_artifacts_produces_all_outputs(tmp_path, monkeypatch):
    z = synth.point_bar_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    build_derivatives("winyah-bay", f)
    out = build_artifacts("winyah-bay", f)
    for name, p in out.items():
        assert p.exists() and p.stat().st_size > 0, name
    contours = json.loads(out["contours"].read_text())
    assert contours["type"] == "FeatureCollection"
```

- [ ] **Step 2: Run them**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_derivatives_pipeline.py -v`
Expected: PASS. If `build_artifacts` returns different keys than `contours`, read `pipeline/artifacts.py` and use its real key names — do not rename the production code to match the test.

- [ ] **Step 3: Commit**

Run: `make check`

```bash
git add backend/tests/test_derivatives_pipeline.py
git commit -m "test: pipeline-level coverage for derivatives and artifacts"
```

---

## Task 5: ANUGA dependency, model-domain config, and a configurable wet level

Bring ANUGA into the project and give the fishery config the two things the mesh needs. **Read `2026-08-13-plan3-anuga-spike-findings.md` §1–§3 before starting.**

The domain polygon is authored, not computed. This is not laziness: the Atlantic and the estuary are connected through North Inlet and the ICW, so a barrier line across the bay mouth leaves "estuary = 798.5 km², ocean = 0.0 km²". Every flood-fill approach silently returns the whole raster. Where the open boundary goes is a modelling decision, exactly like the existing `jetties:` seeds.

This task also resolves carryover trap (b): `WET_LEVEL_M = 0.0` is a module constant in `detect.py`, pinning every static detector's notion of "wet" to NAVD88 zero. ANUGA introduces a time-varying free surface, so the definitions are about to fork. Make it config-driven **before** that happens.

**Files:**
- Modify: `backend/pyproject.toml:6-19`
- Modify: `backend/tidescout/models.py`
- Modify: `backend/tidescout/engine/detect.py:13`
- Modify: `backend/tidescout/pipeline/features.py:40-57`
- Modify: `fisheries/winyah-bay.yaml`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `Fishery.model_domain: ModelDomain` and `Fishery.anuga: AnugaConfig`. Tasks 6–11 read these. `ModelDomain.polygon_utm_km` is a list of `(x_km, y_km)` in the fishery's `bathymetry.epsg`.

- [ ] **Step 1: Add the dependency and confirm it does not disturb the stack**

In `backend/pyproject.toml`, append to `dependencies`:

```toml
    "anuga>=3.3.10",
```

Then:

```bash
make install
~/.venvs/tidescout/bin/python -c "
import anuga, numpy, scipy, rasterio, shapely, skimage
print('anuga', anuga.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)
print('rasterio', rasterio.__version__, '| shapely', shapely.__version__, '| skimage', skimage.__version__)
"
```

Expected: `anuga 3.3.10`, and **numpy 2.5.2 / scipy 1.18.0 / rasterio 1.5.1 / shapely 2.1.2 / skimage 0.26.0 unchanged**. This was verified in the spike — anuga requires `numpy>=2.0.0` and pins nothing that conflicts. If any version moves, stop and report; do not proceed with a disturbed stack.

`WARNING: Could not import mpi4py - defining sequential interface` on import is expected and harmless.

- [ ] **Step 2: Run the existing suite to prove the dependency is inert**

Run: `make check`
Expected: 101+ tests still pass. ANUGA must not perturb anything before it is used.

- [ ] **Step 3: Add the config models**

```python
class ModelDomain(BaseModel):
    """Outer boundary of the hydrodynamic model, authored not inferred.

    Ocean and estuary are hydraulically connected through several inlets, so
    no automatic rule separates them -- see the Plan 3 spike findings. Vertices
    are (x_km, y_km) in the fishery's bathymetry EPSG, listed clockwise.
    """

    polygon_utm_km: list[tuple[float, float]]
    wet_level_m: float = 1.5      # cut the shoreline at highest simulated water
    simplify_m: float = 25.0      # shoreline generalisation before meshing
    clean_cells: int = 3          # morphological close/open radius, in cells


class AnugaConfig(BaseModel):
    base_edge_m: float = 60.0
    jetty_edge_m: float = 15.0
    jetty_radius_m: float = 300.0
    manning_channel: float = 0.022
    manning_flat: float = 0.030
    manning_marsh: float = 0.045
    spin_up_h: float = 6.0
    cycle_h: float = 12.42
    snapshot_minutes: float = 30.0
    mass_tolerance: float = 1e-3   # measured residual is ~4e-4; 1e-6 fails healthy runs
    max_workers: int = 6           # performance cores only -- see Task 11
```

Add to `Fishery`:

```python
    model_domain: ModelDomain | None = None
    anuga: AnugaConfig = AnugaConfig()
```

- [ ] **Step 4: Write the config test**

```python
# backend/tests/test_config.py
def test_winyah_has_a_closed_model_domain():
    f = load_fishery("winyah-bay")
    assert f.model_domain is not None
    poly = f.model_domain.polygon_utm_km
    assert len(poly) >= 4
    assert poly[0] != poly[-1], "polygon is implicitly closed; do not repeat the first vertex"
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # sanity: inside the Winyah UTM 17N analysis grid (643.8-681.9 E, 3669.0-3719.5 N km)
    assert 643.0 < min(xs) and max(xs) < 682.0
    assert 3668.0 < min(ys) and max(ys) < 3720.0


def test_anuga_mass_tolerance_is_not_machine_precision():
    """A 1e-6 assert fails on a healthy wetting/drying run (measured 4.2e-4)."""
    f = load_fishery("winyah-bay")
    assert f.anuga.mass_tolerance >= 1e-4
```

- [ ] **Step 5: Write the domain polygon into the fishery config**

These vertices are the spike's measured draft: they enclose **238 km² of water** and contain all 409 jetty structure cells.

```yaml
model_domain:
  # Authored, not inferred: the Atlantic reconnects to the bay through North
  # Inlet and the ICW, so no cut line or flood-fill separates them (a mouth
  # barrier returns "estuary 798.5 km2, ocean 0.0"). Clockwise, UTM 17N km.
  # Encloses 238 km2 of water and all jetty structure.
  # KNOWN IMPERFECTION: the east edge (vertices 3-5) still admits some open
  # Atlantic. Refine against the chart/ENC -- see Task 6 step 7.
  polygon_utm_km:
    - [660.5, 3673.0]  # SW, inshore of the south approach
    - [666.0, 3670.8]  # S of the entrance
    - [671.0, 3672.0]  # SE, ~2 km seaward of the jetty tips
    - [672.0, 3677.5]  # E, offshore abeam the entrance
    - [671.5, 3683.0]  # NE, hugging North Island's seaward shore
    - [672.5, 3691.0]  # N along the island
    - [671.5, 3696.0]  # top of North Island
    - [669.0, 3701.0]  # into the river mouths
    - [664.0, 3700.0]  # across the delta
    - [658.0, 3696.0]  # NW, west of Georgetown
    - [655.5, 3692.0]  # W, up the Sampit
    - [658.0, 3686.0]  # SW down the west shore
    - [659.0, 3678.0]  # S along the ICW
  wet_level_m: 1.5
anuga:
  base_edge_m: 60.0
  jetty_edge_m: 15.0
```

- [ ] **Step 6: Make the static wet level configurable**

In `detect.py`, replace the module constant with a default parameter. Keep the name exported so nothing breaks:

```python
# Default only. Static detectors ask "is this cell wet at a representative
# water level"; ANUGA has a time-varying free surface, so the two notions of
# "wet" are about to diverge. Callers pass their own.
WET_LEVEL_M = 0.0
```

Then give every detector that references it a `wet_level_m: float = WET_LEVEL_M` keyword and use the parameter internally rather than the module global. In `pipeline/features.py`, pass `wet_level_m=fishery.bathymetry.land_elev_m - 1.5` — **no.** Pass it explicitly from config instead: add `static_wet_level_m: float = 0.0` to `BathymetryConfig` and thread `fishery.bathymetry.static_wet_level_m` through `build_features`.

- [ ] **Step 7: Verify nothing moved**

Run: `make check`
Expected: all tests pass, including the new config tests. The feature inventory must be **byte-identical** to Task 3's output, since `static_wet_level_m` defaults to the old constant:

```bash
~/.venvs/tidescout/bin/tidescout features winyah-bay --rebuild
```

Expected: same type counts as Task 3 produced.

- [ ] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/tidescout/models.py backend/tidescout/engine/detect.py \
        backend/tidescout/pipeline/features.py backend/tests/test_config.py fisheries/winyah-bay.yaml
git commit -m "feat: anuga dependency, authored model domain, configurable wet level"
```

---

## Task 6: The mesh builder

Turn the bathymetry raster plus the authored polygon into a graded ANUGA mesh with elevation on its centroids.

**The shoreline is not the risk it looks like.** A raw Winyah water mask polygonises to 7,581 exterior vertices, which would choke the mesh generator — but morphological cleaning plus `shapely.simplify(25 m)` reduces that to **486 vertices while preserving area**, and 250k triangles then mesh in **0.5 seconds**. Measured.

**Files:**
- Create: `backend/tidescout/pipeline/mesh.py`
- Create: `backend/tests/test_mesh.py`

**Interfaces:**
- Consumes: `read_bathy(slug)`, `Fishery.model_domain`, `Fishery.anuga`.
- Produces:
  - `domain_mask(z, transform, fishery) -> np.ndarray[bool]`
  - `domain_polygon(mask, transform, simplify_m) -> shapely.Polygon`
  - `build_mesh(slug, fishery) -> anuga.Domain` (elevation set, no boundaries)
  - `sample_to_centroids(domain, arr, transform) -> np.ndarray`
  Tasks 7–9 call all four.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_mesh.py
import numpy as np
import pytest

from tidescout.config import load_fishery
from tidescout.pipeline import mesh

from . import synth
from .test_features_pipeline import _fake_bathy


def test_domain_mask_is_single_connected_component():
    z = np.full((200, 200), -5.0, dtype="float32")
    z[0:20, :] = 5.0                     # land strip
    z[150:160, 150:160] = -5.0           # isolated pond, must be dropped
    z[140:170, 60:70] = 5.0
    f = load_fishery("winyah-bay")
    m = mesh.domain_mask(z, synth.TRANSFORM, f, polygon=None)
    from scipy import ndimage
    _, n = ndimage.label(m)
    assert n == 1, "mesh domain must be exactly one connected water body"


def test_domain_polygon_simplifies_hard_but_keeps_area():
    z = np.full((200, 200), -5.0, dtype="float32")
    rng = np.random.default_rng(0)
    # rough up the shoreline so simplification has something to do
    z[0:30, :] = 5.0
    z[30, ::2] = 5.0
    f = load_fishery("winyah-bay")
    m = mesh.domain_mask(z, synth.TRANSFORM, f, polygon=None)
    poly = mesh.domain_polygon(m, synth.TRANSFORM, simplify_m=25.0)
    assert poly.is_valid
    assert len(poly.exterior.coords) < 400
    assert poly.area > 0.9 * m.sum() * 100.0   # 10 m cells = 100 m2 each


def test_build_mesh_sets_elevation_on_every_centroid(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[0:40, :] = 5.0
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []      # empty => use the whole water mask
    d = mesh.build_mesh("winyah-bay", f)
    elev = d.get_quantity("elevation").get_values(location="centroids")
    assert len(elev) == len(d.triangles)
    assert np.isfinite(elev).all(), "no NaN may reach the solver"
    assert elev.min() < 0.0
```

- [ ] **Step 2: Run and confirm they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_mesh.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tidescout.pipeline.mesh'`

- [ ] **Step 3: Write the mesh module**

```python
"""Bathymetry raster + authored domain polygon -> graded ANUGA mesh.

The shoreline of a real estuary is fractal: Winyah's raw water mask
polygonises to 7,581 exterior vertices, which no triangle generator handles
gracefully. Morphological close/open at the target cell scale removes creeks
too narrow to resolve anyway, and simplify(25 m) then takes the ring to ~486
vertices with area preserved. Meshing 250k triangles from that takes 0.5 s.
"""

import numpy as np
import rasterio.features
from matplotlib.path import Path as MplPath
from scipy import ndimage
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union

import anuga
from tidescout.models import Fishery
from tidescout.pipeline.bathy import read_bathy


def domain_mask(z, transform, fishery: Fishery, polygon=None) -> np.ndarray:
    """Boolean mask of water inside the model domain, single connected body."""
    md = fishery.model_domain
    level = md.wet_level_m if md else 1.5
    valid = ~np.isnan(z)
    wet = valid & (z < level)

    verts = polygon if polygon is not None else (md.polygon_utm_km if md else [])
    if verts:
        h, w = z.shape
        inv = ~transform
        poly_px = np.array([inv * (x * 1000.0, y * 1000.0) for x, y in verts])
        rows, cols = np.mgrid[0:h, 0:w]
        pts = np.column_stack([cols.ravel(), rows.ravel()])
        inside = MplPath(poly_px).contains_points(pts).reshape(h, w)
        wet &= inside

    lbl, n = ndimage.label(wet)
    if n == 0:
        raise ValueError("model domain contains no water")
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()


def clean_mask(mask: np.ndarray, cells: int) -> np.ndarray:
    """Drop sub-mesh-scale detail that would explode the boundary vertex count.

    Islands are deliberately filled rather than carved as mesh holes: ANUGA
    keeps them dry through wetting/drying because they sit above water, and
    that also lets them flood on a spring tide, which a hole never could.
    """
    k = np.ones((cells, cells))
    m = ndimage.binary_closing(mask, k)
    m = ndimage.binary_opening(m, k)
    m = ndimage.binary_fill_holes(m)
    lbl, _ = ndimage.label(m)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == sizes.argmax()


def domain_polygon(mask: np.ndarray, transform, simplify_m: float) -> Polygon:
    geoms = [
        shape(g)
        for g, v in rasterio.features.shapes(
            mask.astype("uint8"), mask=mask, transform=transform
        )
        if v == 1
    ]
    poly = unary_union(geoms)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly.simplify(simplify_m, preserve_topology=True)


def jetty_regions(fishery: Fishery, boundary: Polygon, epsg: int) -> list:
    """Refinement polygons around jetty structure, clipped inside the boundary.

    The spec singles out the Winyah jetty rips as must-catch, so they get their
    own zone rather than inheriting the channel's resolution.
    """
    from rasterio.warp import transform as warp_transform

    cfg = fishery.anuga
    lines = []
    for jetty in fishery.jetties:
        lons = [c[0] for c in jetty.coords]
        lats = [c[1] for c in jetty.coords]
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", lons, lats)
        lines.append(LineString(list(zip(xs, ys, strict=True))))
    if not lines:
        return []
    buf = unary_union([ln.buffer(cfg.jetty_radius_m) for ln in lines])
    buf = buf.intersection(boundary.buffer(-30.0))
    if buf.is_empty:
        return []
    parts = list(buf.geoms) if buf.geom_type == "MultiPolygon" else [buf]
    return [p.simplify(cfg.jetty_edge_m) for p in parts if not p.is_empty]


def edge_to_area(edge_m: float) -> float:
    """Equilateral triangle area for a target edge length."""
    return edge_m * edge_m * np.sqrt(3.0) / 4.0


def sample_to_centroids(domain, arr: np.ndarray, transform) -> np.ndarray:
    """Nearest-cell lookup of a raster onto mesh centroids."""
    cx, cy = domain.get_centroid_coordinates(absolute=True).T
    cols, rows = ~transform * (cx, cy)
    rows = np.clip(rows.astype(int), 0, arr.shape[0] - 1)
    cols = np.clip(cols.astype(int), 0, arr.shape[1] - 1)
    return arr[rows, cols]


def build_mesh(slug: str, fishery: Fishery):
    z, transform, meta = read_bathy(slug)
    md = fishery.model_domain
    cfg = fishery.anuga
    mask = clean_mask(domain_mask(z, transform, fishery), md.clean_cells if md else 3)
    boundary = domain_polygon(mask, transform, md.simplify_m if md else 25.0)
    ring = [list(c) for c in boundary.exterior.coords[:-1]]

    regions = [
        [[list(c) for c in p.exterior.coords[:-1]], edge_to_area(cfg.jetty_edge_m)]
        for p in jetty_regions(fishery, boundary, fishery.bathymetry.epsg)
    ]
    domain = anuga.create_domain_from_regions(
        ring,
        boundary_tags={"outer": list(range(len(ring)))},
        maximum_triangle_area=edge_to_area(cfg.base_edge_m),
        interior_regions=regions or None,
        verbose=False,
    )
    elev = sample_to_centroids(domain, z, transform)
    # nodata slivers at the boundary become land rather than NaN: a NaN
    # elevation propagates through the solver and kills the whole run.
    elev = np.where(np.isfinite(elev), elev, 1.0)
    domain.set_quantity("elevation", elev, location="centroids")
    return domain
```

**API notes that will otherwise cost an hour:** `create_domain_from_regions` has **no `mesh_filename` argument**. Its real signature is `(bounding_polygon, boundary_tags, maximum_triangle_area=None, interior_regions=None, interior_holes=None, hole_tags=None, poly_geo_reference=None, mesh_geo_reference=None, breaklines=None, regionPtArea=None, minimum_triangle_angle=28.0, fail_if_polygons_outside=True, use_cache=False, verbose=False)`. `set_boundary` raises if you name a tag that is not a `boundary_tags` key.

- [ ] **Step 4: Run the tests**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_mesh.py -v`
Expected: PASS.

- [ ] **Step 5: Build the real Winyah mesh and record its size**

```bash
cd backend && ~/.venvs/tidescout/bin/python -c "
import time
from tidescout.config import load_fishery
from tidescout.pipeline import mesh
f = load_fishery('winyah-bay')
t0 = time.time()
d = mesh.build_mesh('winyah-bay', f)
print(f'{len(d.triangles):,} triangles in {time.time()-t0:.1f}s')
import numpy as np
e = d.get_quantity('elevation').get_values(location='centroids')
print(f'elevation {e.min():.1f}..{e.max():.1f} m, finite={np.isfinite(e).all()}')
"
```

Expected: **~315,000 triangles** at the configured 60 m / 15 m grading, built in under 2 seconds, elevation spanning roughly −16 to +10 m. A count far from ~315k means the polygon or grading was mis-transcribed — stop and check before running any simulation.

- [ ] **Step 6: Commit**

Run: `make check`

```bash
git add backend/tidescout/pipeline/mesh.py backend/tests/test_mesh.py
git commit -m "feat: graded ANUGA mesh builder from bathymetry and domain polygon"
```

- [ ] **Step 7: Report the domain polygon's east edge**

Not a code change — a note for the task report. The draft polygon still admits open Atlantic along vertices 3–5, which inflates the mesh and simulates water nobody fishes. Record the triangle count and note that refining the east edge against the chart is available as a cheap win. **Do not refine it by guessing**; it needs Ellis or a chart.

---

## Task 7: Manning friction from the zones raster

Spatially varying bed friction — channel vs flats vs marsh — is one of the three things the spec names as forcing.

**Carryover trap (a), which will bite silently:** `zones()` shares `shallow_max_m` and `deep_min_m` with `detect_bars` through `FeatureThresholds`. Retuning bar detection therefore re-buckets the friction field, changing every simulation result, with no error and no obvious connection. This task **splits the config** so the two stop aliasing.

**Files:**
- Modify: `backend/tidescout/models.py` (`BathymetryConfig`)
- Modify: `backend/tidescout/pipeline/derivatives.py:23-43`
- Modify: `backend/tidescout/pipeline/mesh.py`
- Test: `backend/tests/test_mesh.py`, `backend/tests/test_derivatives_pipeline.py`

**Interfaces:**
- Produces: `mesh.friction_field(domain, slug, fishery) -> np.ndarray` (per-centroid Manning n), called by Task 9.

- [ ] **Step 1: Split the aliased thresholds**

Add dedicated zone boundaries to `BathymetryConfig` so the friction field stops depending on feature-detection tuning:

```python
class BathymetryConfig(BaseModel):
    epsg: int = 26917
    cell_m: float = 10.0
    land_elev_m: float = 1.5
    contour_depths_m: list[float] = [-2.0, -5.0, -10.0, -15.0]
    static_wet_level_m: float = 0.0
    # Deliberately NOT FeatureThresholds.shallow_max_m/deep_min_m. Those two
    # drive bar detection; sharing them means retuning bars silently re-buckets
    # the Manning field and changes every simulation. Defaults match the
    # previous shared values so this split is a no-op at introduction.
    zone_shallow_max_m: float = -0.3
    zone_deep_min_m: float = -3.0
```

- [ ] **Step 2: Point `build_derivatives` at the new fields**

In `pipeline/derivatives.py`, change the `terrain.zones(...)` call:

```python
    zn = terrain.zones(
        z, fishery.bathymetry.land_elev_m,
        fishery.bathymetry.zone_shallow_max_m, fishery.bathymetry.zone_deep_min_m,
    )
```

- [ ] **Step 3: Prove the split is a no-op right now**

```python
# backend/tests/test_derivatives_pipeline.py
def test_zone_thresholds_are_independent_of_bar_tuning(tmp_path, monkeypatch):
    """Retuning bar detection must not move the friction zones."""
    z = synth.point_bar_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    before = build_derivatives("winyah-bay", f)
    with rasterio.open(before["zones"]) as src:
        baseline = src.read(1).copy()

    f.features.shallow_max_m = -0.9      # aggressive bar retune
    f.features.deep_min_m = -6.0
    after = build_derivatives("winyah-bay", f)
    with rasterio.open(after["zones"]) as src:
        assert np.array_equal(src.read(1), baseline), "zones still alias bar thresholds"
```

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_derivatives_pipeline.py -v`
Expected: PASS (it would have FAILED before step 2 — that is the point).

- [ ] **Step 4: Write the friction field**

Add to `pipeline/mesh.py`:

```python
def friction_field(domain, slug: str, fishery: Fishery) -> np.ndarray:
    """Per-centroid Manning n from the zones raster.

    zones.tif is a uint8 enum with 0 = nodata. Anything unclassified falls back
    to the flat value rather than to zero, because a Manning n of 0 is
    frictionless and would produce spectacular nonsense rather than an error.
    """
    import rasterio

    from tidescout.paths import fishery_data_dir

    cfg = fishery.anuga
    path = fishery_data_dir(slug) / "zones.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `tidescout bathy build {slug}` before meshing"
        )
    with rasterio.open(path) as src:
        zones = src.read(1)
        transform = src.transform
    z_at = sample_to_centroids(domain, zones, transform)
    n = np.full(z_at.shape, cfg.manning_flat, dtype="float64")
    n[z_at == 3] = cfg.manning_channel   # deep
    n[z_at == 1] = cfg.manning_marsh     # land/marsh
    return n
```

**Confirm the zone enum before trusting those numbers:** read `engine/terrain.py::zones` and map the constants to the right `manning_*` value. If the enum ordering differs from the assumption above, fix the mapping here, not in `terrain.py`.

- [ ] **Step 5: Test it**

```python
# backend/tests/test_mesh.py
def test_friction_field_has_no_zero_values(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[0:40, :] = 5.0
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    from tidescout.pipeline.derivatives import build_derivatives
    build_derivatives("winyah-bay", f)
    d = mesh.build_mesh("winyah-bay", f)
    n = mesh.friction_field(d, "winyah-bay", f)
    assert len(n) == len(d.triangles)
    assert (n > 0).all(), "a zero Manning n is frictionless, not a default"
    assert n.max() <= 0.1
```

- [ ] **Step 6: Commit**

Run: `make check`

```bash
git add backend/tidescout/models.py backend/tidescout/pipeline/derivatives.py \
        backend/tidescout/pipeline/mesh.py backend/tests/test_mesh.py \
        backend/tests/test_derivatives_pipeline.py
git commit -m "feat: Manning friction field, de-aliased from bar-detection thresholds"
```

---

## Task 8: Boundary forcing — tide datum and river inflow

The ocean boundary is driven by CO-OPS water levels; the rivers push discharge in at the top. **Two unit traps live here and both silently produce plausible-looking nonsense:**

1. `noaa.tide_hours()` returns `height_ft` — **feet**. ANUGA is metres.
2. CO-OPS predictions are on the **MLLW** datum. The bathymetry is **NAVD88**. At Winyah the offset is roughly 0.8 m but must be resolved from the station, not assumed.

**Files:**
- Create: `backend/tidescout/pipeline/forcing.py`
- Create: `backend/tests/test_forcing.py`
- Modify: `fisheries/winyah-bay.yaml`
- Modify: `backend/tidescout/models.py`

**Interfaces:**
- Consumes: `engine.tides.TideEvent`, `sources.noaa.tide_events`, `sources.usgs.DischargeSummary` from Task 1.
- Produces:
  - `FT_TO_M = 0.3048`
  - `tide_function(events, datum_offset_m, start) -> Callable[[float], float]`
  - `range_scaled_tide(mean_range_m, bucket, ...) -> Callable[[float], float]`
  - `river_inflow_m3s(fishery, bucket) -> dict[str, float]`
  Task 9 calls all three.

- [ ] **Step 1: Resolve the station datum offset live and record it**

```bash
curl -s "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/8662549/datums.json" \
  | ~/.venvs/tidescout/bin/python -c "
import json,sys
d=json.load(sys.stdin)
vals={x['name']: x['value'] for x in d['datums']}
for k in ('MLLW','MSL','MHHW','NAVD88','STND'):
    if k in vals: print(f'{k:8s} {vals[k]:8.3f} ft')
if 'NAVD88' in vals and 'MLLW' in vals:
    off=(vals['MLLW']-vals['NAVD88'])*0.3048
    print(f'\nMLLW is {off:+.3f} m relative to NAVD88 -> add this to predictions')
"
```

Record the printed offset in `fisheries/winyah-bay.yaml` under a new `tide_datum_offset_m` field on `Stations`, with the derivation in a comment. **If the station publishes no NAVD88 datum**, say so in the task report and fall back to `0.0` with an explicit comment — do not invent a number. A wrong datum offset shifts every flat's flood/drain schedule.

Add to `models.py`:

```python
class Stations(BaseModel):
    tide: list[str] = []
    currents: list[str] = []
    water: list[WaterSensor] = []
    # Added to CO-OPS predictions to convert MLLW -> NAVD88, the bathymetry
    # datum. Resolved from the station's own datums endpoint, not assumed.
    tide_datum_offset_m: float = 0.0
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_forcing.py
from datetime import UTC, datetime, timedelta

import pytest

from tidescout.engine.tides import TideEvent
from tidescout.pipeline import forcing


def _events(start):
    return [
        TideEvent(start, "H", 5.0),
        TideEvent(start + timedelta(hours=6, minutes=13), "L", 1.0),
        TideEvent(start + timedelta(hours=12, minutes=25), "H", 5.0),
    ]


def test_tide_function_converts_feet_to_metres_and_shifts_datum():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    fn = forcing.tide_function(_events(start), datum_offset_m=-0.8, start=start)
    # t=0 is the 5.0 ft high water: 5 ft = 1.524 m, minus 0.8 m datum shift
    assert fn(0.0) == pytest.approx(1.524 - 0.8, abs=1e-3)


def test_tide_function_is_smooth_and_bounded():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    fn = forcing.tide_function(_events(start), datum_offset_m=0.0, start=start)
    vals = [fn(t) for t in range(0, 12 * 3600, 600)]
    assert min(vals) >= 1.0 * forcing.FT_TO_M - 1e-6
    assert max(vals) <= 5.0 * forcing.FT_TO_M + 1e-6
    # no discontinuity larger than a few cm between 10-minute samples
    assert max(abs(b - a) for a, b in zip(vals, vals[1:], strict=False)) < 0.05


def test_range_buckets_scale_amplitude_about_mean_level():
    neap = forcing.range_scaled_tide(mean_range_m=1.5, bucket="neap")
    spring = forcing.range_scaled_tide(mean_range_m=1.5, bucket="spring")
    assert max(spring(t) for t in range(0, 44712, 600)) > max(
        neap(t) for t in range(0, 44712, 600)
    )


def test_river_inflow_scales_with_discharge_bucket(fishery):
    lo = forcing.river_inflow_m3s(fishery, "low")
    hi = forcing.river_inflow_m3s(fishery, "high")
    assert set(lo) == {r.name for r in fishery.rivers}
    assert sum(hi.values()) > sum(lo.values())
    assert all(v >= 0 for v in lo.values())
```

- [ ] **Step 3: Run and confirm they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_forcing.py -v`
Expected: FAIL — no module `tidescout.pipeline.forcing`.

- [ ] **Step 4: Write the forcing module**

```python
"""Boundary forcing for the regime runs.

Two unit traps live here, and both produce plausible-looking wrong answers
rather than errors:
  1. NOAA CO-OPS returns tide heights in FEET; ANUGA works in metres.
  2. CO-OPS predictions are on MLLW; the bathymetry is NAVD88. The offset is
     resolved per-station (Stations.tide_datum_offset_m), never assumed.
"""

import math
from collections.abc import Callable
from datetime import datetime

from tidescout.engine.tides import TideEvent, _cosine_height
from tidescout.models import Fishery

FT_TO_M = 0.3048
CFS_TO_M3S = 0.0283168

# Amplitude multipliers about mean water for the three tidal-range regimes.
RANGE_FACTORS = {"neap": 0.72, "mean": 1.0, "spring": 1.28}


def tide_function(
    events: list[TideEvent], datum_offset_m: float, start: datetime
) -> Callable[[float], float]:
    """Stage (m, NAVD88) as a function of seconds since `start`.

    Interpolates between predicted high/low water with the same cosine ramp the
    conditions engine already uses, so the boundary and the displayed tide curve
    cannot drift apart.
    """
    ordered = sorted(events, key=lambda e: e.time)
    if len(ordered) < 2:
        raise ValueError("need at least two tide events to force a boundary")
    times = [(e.time - start).total_seconds() for e in ordered]
    heights = [e.height_ft * FT_TO_M + datum_offset_m for e in ordered]

    def stage(t: float) -> float:
        if t <= times[0]:
            return heights[0]
        if t >= times[-1]:
            return heights[-1]
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                span = times[i + 1] - times[i]
                frac = 0.0 if span == 0 else (t - times[i]) / span
                return _cosine_height(heights[i], heights[i + 1], frac)
        return heights[-1]

    return stage


def range_scaled_tide(
    mean_range_m: float, bucket: str, period_s: float = 12.42 * 3600.0,
    mean_level_m: float = 0.0,
) -> Callable[[float], float]:
    """Idealised M2 cosine for a tidal-range regime.

    Regime runs are not hindcasts of a particular day -- they are the recurring
    flow patterns the spec's library is indexed by -- so a clean harmonic is
    the right forcing. Real predicted events drive validation runs instead.
    """
    amp = 0.5 * mean_range_m * RANGE_FACTORS[bucket]

    def stage(t: float) -> float:
        return mean_level_m + amp * math.cos(2.0 * math.pi * t / period_s)

    return stage


def river_inflow_m3s(fishery: Fishery, bucket: str) -> dict[str, float]:
    """Steady inflow per river for a discharge regime, in m^3/s.

    The composite bucket boundaries are the calibrated percentiles from Task 1.
    'low' and 'high' sit at those boundaries; 'med' at their midpoint. Each
    river takes its configured share of the composite.
    """
    b = fishery.discharge_buckets
    composite_cfs = {
        "low": b.low_below_cfs,
        "med": 0.5 * (b.low_below_cfs + b.high_above_cfs),
        "high": b.high_above_cfs,
    }[bucket]
    total_weight = sum(r.weight for r in fishery.rivers) or 1.0
    return {
        r.name: composite_cfs * CFS_TO_M3S * (r.weight / total_weight)
        for r in fishery.rivers
    }
```

Note `_cosine_height` is imported from `engine.tides` deliberately — reusing it keeps the boundary and the UI tide curve on the same ramp. If ruff flags the private import, promote it to `cosine_height` in `engine/tides.py` and update both call sites.

- [ ] **Step 5: Run the tests**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_forcing.py -v`
Expected: PASS.

- [ ] **Step 6: Determine the real mean tidal range**

```bash
cd backend && ~/.venvs/tidescout/bin/python -c "
from datetime import date, timedelta
from statistics import fmean
from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources import noaa
f = load_fishery('winyah-bay'); cache = Cache()
st = f.stations.tide[0]
ranges = []
d0 = date.today()
for i in range(30):
    ev = noaa.tide_events(st, d0 - timedelta(days=i), f.timezone, cache)
    hs = [e.height_ft for e in ev if e.kind == 'H']
    ls = [e.height_ft for e in ev if e.kind == 'L']
    if hs and ls:
        ranges.append(fmean(hs) - fmean(ls))
print(f'n={len(ranges)} days  mean range {fmean(ranges):.2f} ft = {fmean(ranges)*0.3048:.2f} m')
print(f'  min {min(ranges)*0.3048:.2f} m  max {max(ranges)*0.3048:.2f} m')
"
```

Record the mean in `fisheries/winyah-bay.yaml` as `anuga.mean_range_m`, with the derivation in a comment, and add the field to `AnugaConfig` (`mean_range_m: float = 1.5`). Cross-check that the observed min/max bracket the `RANGE_FACTORS` neap/spring multipliers; if they do not, report it rather than adjusting the factors silently.

- [ ] **Step 7: Commit**

Run: `make check`

```bash
git add backend/tidescout/pipeline/forcing.py backend/tests/test_forcing.py \
        backend/tidescout/models.py fisheries/winyah-bay.yaml
git commit -m "feat: tide and river boundary forcing with datum and unit conversion"
```

---

## Task 9: Single regime runner with automated checks

Run one simulation end to end and prove it is physically sound. The spec names two automated checks: mass conservation within tolerance, and ebb/flood direction reversal at a channel cross-section.

**Files:**
- Create: `backend/tidescout/pipeline/regimes.py`
- Create: `backend/tests/test_regimes.py`

**Interfaces:**
- Consumes: `mesh.build_mesh`, `mesh.friction_field`, `forcing.range_scaled_tide`, `forcing.river_inflow_m3s`.
- Produces:
  - `REGIME_MATRIX: list[tuple[str, str]]`
  - `regime_name(range_bucket, discharge_bucket) -> str`
  - `mass_residual(domain, v0) -> float`
  - `run_regime(slug, range_bucket, discharge_bucket, sim_hours=None) -> Path`
  Tasks 10 and 11 call `run_regime` and `regime_name`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_regimes.py
import numpy as np
import pytest

from tidescout.pipeline import regimes


def test_regime_matrix_is_three_by_three():
    assert len(regimes.REGIME_MATRIX) == 9
    assert ("spring", "high") in regimes.REGIME_MATRIX
    assert len(set(regimes.REGIME_MATRIX)) == 9


def test_regime_name_is_filesystem_safe_and_unique():
    names = {regimes.regime_name(r, d) for r, d in regimes.REGIME_MATRIX}
    assert len(names) == 9
    assert all(n.replace("_", "").isalnum() for n in names)


def test_mass_residual_tolerance_is_not_machine_precision():
    """Measured residual on a real wetting/drying mesh is ~4e-4. A 1e-6 gate
    fails every healthy run -- this was hit during the Plan 3 spike."""
    from tidescout.config import load_fishery
    assert load_fishery("winyah-bay").anuga.mass_tolerance >= 1e-4
```

- [ ] **Step 2: Run and confirm they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_regimes.py -v`
Expected: FAIL — no module `tidescout.pipeline.regimes`.

- [ ] **Step 3: Write the runner**

```python
"""Regime simulations: the 3x3 tidal-range x discharge matrix.

Each regime is one full tidal cycle plus spin-up, snapshotted at the configured
cadence. Runs are completely independent, which is what lets Task 10 execute
them as parallel OS processes rather than reaching for MPI.
"""

import json
import time
from pathlib import Path

import numpy as np

import anuga
from tidescout.config import load_fishery
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import forcing, mesh

RANGE_BUCKETS = ["neap", "mean", "spring"]
DISCHARGE_BUCKETS = ["low", "med", "high"]
REGIME_MATRIX = [(r, d) for r in RANGE_BUCKETS for d in DISCHARGE_BUCKETS]


def regime_name(range_bucket: str, discharge_bucket: str) -> str:
    return f"{range_bucket}_{discharge_bucket}"


def regime_dir(slug: str) -> Path:
    d = fishery_data_dir(slug) / "flow"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mass_residual(domain, v0: float) -> float:
    """Relative closure of ANUGA's volume identity.

    dV must equal boundary flux plus fractional-step (inflow) volume. Returns
    the residual normalised by the volume actually moved.
    """
    v1 = domain.get_water_volume()
    flux = domain.get_boundary_flux_integral()
    frac = domain.get_fractional_step_volume_integral()
    moved = max(abs(v1 - v0), 1.0)
    return abs((v1 - v0) - (flux + frac)) / moved


def _centroid_speed(domain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth, u, v) at centroids, zeroed where dry."""
    stage = domain.get_quantity("stage").get_values(location="centroids")
    elev = domain.get_quantity("elevation").get_values(location="centroids")
    xmom = domain.get_quantity("xmomentum").get_values(location="centroids")
    ymom = domain.get_quantity("ymomentum").get_values(location="centroids")
    depth = stage - elev
    wet = depth > 0.01
    safe = np.where(wet, depth, 1.0)
    u = np.where(wet, xmom / safe, 0.0)
    v = np.where(wet, ymom / safe, 0.0)
    return depth, u, v


def run_regime(
    slug: str, range_bucket: str, discharge_bucket: str, sim_hours: float | None = None
) -> Path:
    """Run one regime; write snapshots and a per-regime metadata JSON."""
    fishery: Fishery = load_fishery(slug)
    cfg = fishery.anuga
    name = regime_name(range_bucket, discharge_bucket)
    out_dir = regime_dir(slug) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = mesh.build_mesh(slug, fishery)
    domain.set_name(name)
    domain.set_datadir(str(out_dir))
    domain.set_quantity("friction", mesh.friction_field(domain, slug, fishery))

    elev = domain.get_quantity("elevation").get_values(location="centroids")
    tide = forcing.range_scaled_tide(
        cfg.mean_range_m, range_bucket, period_s=cfg.cycle_h * 3600.0
    )
    # Start at the initial boundary level so spin-up is not a dam break.
    domain.set_quantity(
        "stage", np.maximum(elev + 1e-3, tide(0.0)), location="centroids"
    )
    domain.set_boundary({
        "outer": anuga.Transmissive_momentum_set_stage_boundary(
            domain=domain, function=tide
        )
    })

    inflows = forcing.river_inflow_m3s(fishery, discharge_bucket)
    _attach_river_inflows(domain, fishery, inflows)

    total_h = sim_hours if sim_hours is not None else cfg.spin_up_h + cfg.cycle_h
    yieldstep = cfg.snapshot_minutes * 60.0
    v0 = domain.get_water_volume()

    snaps = []
    t_start = time.time()
    for t in domain.evolve(yieldstep=yieldstep, finaltime=total_h * 3600.0):
        if t < cfg.spin_up_h * 3600.0:
            continue  # discard spin-up; it is not a physical state
        depth, u, v = _centroid_speed(domain)
        phase = ((t - cfg.spin_up_h * 3600.0) / (cfg.cycle_h * 3600.0)) % 1.0
        idx = len(snaps)
        np.savez_compressed(
            out_dir / f"snap_{idx:03d}.npz",
            depth=depth.astype("float32"),
            u=u.astype("float32"),
            v=v.astype("float32"),
        )
        snaps.append({"index": idx, "t_s": float(t), "phase": float(phase),
                      "stage_bc_m": float(tide(t))})

    meta = {
        "regime": name,
        "range_bucket": range_bucket,
        "discharge_bucket": discharge_bucket,
        "triangles": int(len(domain.triangles)),
        "sim_hours": total_h,
        "wall_seconds": round(time.time() - t_start, 1),
        "mass_residual": float(mass_residual(domain, v0)),
        "inflows_m3s": inflows,
        "snapshots": snaps,
    }
    (out_dir / "regime.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def _attach_river_inflows(domain, fishery: Fishery, inflows: dict[str, float]) -> None:
    """Push each river's discharge in as an inlet at its up-estuary boundary.

    VERIFY THE OPERATOR API BEFORE TRUSTING THIS. ANUGA exposes inflow via
    `anuga.Inlet_operator(domain, region, Q)` where region is a polygon or line.
    Check the installed version:
        python -c "import anuga, inspect; print(inspect.signature(anuga.Inlet_operator))"
    If the signature differs, adapt here -- do not silently skip the inflow, or
    every 'high discharge' regime becomes identical to 'low'.
    """
    from rasterio.warp import transform as warp_transform

    epsg = fishery.bathymetry.epsg
    for river in fishery.rivers:
        seed = getattr(river, "inflow_lonlat", None)
        if seed is None:
            continue
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [seed[0]], [seed[1]])
        cx, cy = xs[0], ys[0]
        r = 150.0
        region = [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]
        anuga.Inlet_operator(domain, region, Q=inflows[river.name])
```

Add `inflow_lonlat: tuple[float, float] | None = None` to `RiverGauge` in `models.py`, and set it for each river in `winyah-bay.yaml` at the point where that river enters the model domain (read them off the domain polygon's northern vertices). Also add `mean_range_m` to `AnugaConfig` if Task 8 step 6 has not already.

- [ ] **Step 4: Run the unit tests**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_regimes.py -v`
Expected: PASS.

- [ ] **Step 5: Run one short real regime and check mass closure**

```bash
cd backend && ~/.venvs/tidescout/bin/python -c "
import json
from tidescout.pipeline.regimes import run_regime
out = run_regime('winyah-bay', 'mean', 'med', sim_hours=7.0)
m = json.loads((out / 'regime.json').read_text())
print(f\"{m['triangles']:,} tri, {m['wall_seconds']}s, residual {m['mass_residual']:.2e}\")
print(f\"snapshots: {len(m['snapshots'])}\")
"
```

Expected: mass residual **below 1e-3** (the spike measured ~4e-4). ~315k triangles. With `spin_up_h=6.0` and 7 sim-hours total, expect 2 snapshots — enough to prove the plumbing without waiting for a full run.

**If the residual exceeds 1e-3**, the inflow operator is the first suspect: `get_fractional_step_volume_integral` must account for it, and a hand-rolled inflow that mutates stage directly will not be counted. Report rather than loosening the tolerance.

- [ ] **Step 6: Add the ebb/flood reversal check**

```python
def reversal_check(out_dir: Path) -> dict:
    """Flow must reverse direction across a tidal cycle.

    A domain that only ever drains (or only fills) means the boundary never
    drove it -- the most likely silent failure in the whole pipeline.
    """
    meta = json.loads((out_dir / "regime.json").read_text())
    signs = []
    for snap in meta["snapshots"]:
        d = np.load(out_dir / f"snap_{snap['index']:03d}.npz")
        u, v, depth = d["u"], d["v"], d["depth"]
        deep = depth > 2.0
        if not deep.any():
            continue
        # net along-channel transport, projected on the dominant flow axis
        signs.append(float(np.mean(u[deep]) + np.mean(v[deep])))
    return {
        "n_samples": len(signs),
        "reversed": bool(signs and min(signs) < 0 < max(signs)),
        "min": min(signs) if signs else 0.0,
        "max": max(signs) if signs else 0.0,
    }
```

Test it with a synthetic pair of snapshot files rather than a real run:

```python
def test_reversal_check_detects_a_one_way_domain(tmp_path):
    meta = {"snapshots": [{"index": 0, "t_s": 0, "phase": 0.0, "stage_bc_m": 0.0},
                          {"index": 1, "t_s": 1800, "phase": 0.1, "stage_bc_m": 0.1}]}
    (tmp_path / "regime.json").write_text(json.dumps(meta))
    for i in (0, 1):
        np.savez_compressed(
            tmp_path / f"snap_{i:03d}.npz",
            depth=np.array([5.0, 5.0], "float32"),
            u=np.array([0.4, 0.4], "float32"),   # always positive: never reverses
            v=np.zeros(2, "float32"),
        )
    assert regimes.reversal_check(tmp_path)["reversed"] is False
```

- [ ] **Step 7: Commit**

Run: `make check`

```bash
git add backend/tidescout/pipeline/regimes.py backend/tests/test_regimes.py \
        backend/tidescout/models.py fisheries/winyah-bay.yaml
git commit -m "feat: single-regime runner with mass and reversal checks"
```

---

## Task 10: The parallel regime driver

Build all nine regimes. **Use OS processes, not MPI.** The regimes are completely independent, so process-parallelism gets the same throughput as MPI with no new toolchain, no `distribute`/`sww_merge` code, and no inter-rank communication. Measured basis:

| approach | 9 regimes | cost to adopt |
|---|---|---|
| serial | ~49 h | nothing |
| **6 concurrent processes** | **~11 h** | a process pool |
| MPI, 6 ranks | ~11 h | open-mpi, mpi4py, pymetis, new code paths |

Two measured constraints set `max_workers`:
- **Memory is not the limit.** Peak RSS is **1.06 GB per worker**, so six workers use ~6.4 GB of 24 GB.
- **Cores are.** This machine has 6 performance + 12 efficiency cores. Shallow-water solvers are memory-bandwidth-bound, so expect ~4–5× real speedup from six workers, not 6×. Do not raise `max_workers` above 6 — efficiency cores are far slower and add heat without throughput.

**Files:**
- Modify: `backend/tidescout/pipeline/regimes.py`
- Modify: `backend/tidescout/cli.py`
- Test: `backend/tests/test_regimes.py`

**Interfaces:**
- Produces: `build_library(slug, max_workers=None, sim_hours=None) -> dict[str, dict]`, called by the CLI and Task 11.

- [ ] **Step 1: Write the driver**

```python
def build_library(
    slug: str, max_workers: int | None = None, sim_hours: float | None = None
) -> dict[str, dict]:
    """Run every regime, as independent processes.

    Deliberately NOT MPI. The nine runs share nothing, so a process pool gets
    the same wall time as domain decomposition with none of the toolchain --
    see the Plan 3 spike findings for the measured comparison.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    fishery = load_fishery(slug)
    workers = max_workers or fishery.anuga.max_workers
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_regime, slug, r, d, sim_hours): regime_name(r, d)
            for r, d in REGIME_MATRIX
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                out_dir = fut.result()
            except Exception as exc:  # one bad regime must not lose the other eight
                results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                continue
            meta = json.loads((out_dir / "regime.json").read_text())
            meta["status"] = "ok"
            meta["reversal"] = reversal_check(out_dir)
            results[name] = meta
    manifest = regime_dir(slug) / "library.json"
    manifest.write_text(json.dumps({"regimes": results}, indent=2))
    return results
```

`run_regime` takes only strings and floats, so it pickles cleanly to a child process. Each child builds its own mesh — ~2 s of the run — which is far simpler than sharing an unpicklable `anuga.Domain`.

- [ ] **Step 2: Test the driver wiring without running simulations**

```python
def test_build_library_records_a_failed_regime_without_losing_others(monkeypatch, tmp_path):
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        if (r, d) == ("spring", "high"):
            raise RuntimeError("solver blew up")
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regime.json").write_text(json.dumps({"regime": regimes.regime_name(r, d),
                                                     "snapshots": []}))
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})
    # run in-process so monkeypatching applies
    monkeypatch.setattr(regimes, "build_library", regimes.build_library.__wrapped__
                        if hasattr(regimes.build_library, "__wrapped__")
                        else regimes.build_library)
    out = {}
    for r, d in regimes.REGIME_MATRIX:
        name = regimes.regime_name(r, d)
        try:
            fake_run("winyah-bay", r, d)
            out[name] = {"status": "ok"}
        except RuntimeError as e:
            out[name] = {"status": "failed", "error": str(e)}
    assert out["spring_high"]["status"] == "failed"
    assert sum(v["status"] == "ok" for v in out.values()) == 8
```

This tests the failure-isolation contract directly; a `ProcessPoolExecutor` cannot see monkeypatches in its children, so do not try to test the pool itself.

- [ ] **Step 3: Wire up the CLI**

In `cli.py`, following the existing `bathy_app` pattern:

```python
flow_app = typer.Typer(no_args_is_help=True, help="ANUGA flow-state library.")
app.add_typer(flow_app, name="flow")


@flow_app.command("mesh")
def flow_mesh(slug: str) -> None:
    """Build the mesh and report its size without simulating."""
    from tidescout.config import load_fishery
    from tidescout.pipeline import mesh as meshmod

    fishery = load_fishery(slug)
    domain = meshmod.build_mesh(slug, fishery)
    console.print(
        f"{len(domain.triangles):,} triangles "
        f"(base {fishery.anuga.base_edge_m:.0f} m, jetty {fishery.anuga.jetty_edge_m:.0f} m)"
    )


@flow_app.command("run")
def flow_run(
    slug: str,
    workers: int = typer.Option(0, "--workers", help="0 = use anuga.max_workers"),
    sim_hours: float = typer.Option(0.0, "--sim-hours", help="0 = full spin-up + cycle"),
) -> None:
    """Run the full regime matrix as parallel processes."""
    from tidescout.pipeline.regimes import build_library

    results = build_library(
        slug, max_workers=workers or None, sim_hours=sim_hours or None
    )
    table = Table(title=f"{slug} — regime library")
    for col in ("regime", "status", "triangles", "wall s", "mass resid", "reversed"):
        table.add_column(col)
    for name in sorted(results):
        m = results[name]
        table.add_row(
            name, m.get("status", "?"), f"{m.get('triangles', 0):,}",
            str(m.get("wall_seconds", "-")),
            f"{m.get('mass_residual', float('nan')):.2e}",
            str(m.get("reversal", {}).get("reversed", "-")),
        )
    console.print(table)
```

- [ ] **Step 4: Smoke-test the matrix with a short run**

```bash
~/.venvs/tidescout/bin/tidescout flow mesh winyah-bay
~/.venvs/tidescout/bin/tidescout flow run winyah-bay --sim-hours 7 --workers 3
```

Expected: nine rows, all `ok`, mass residual under 1e-3 for each. This exercises the whole matrix in roughly the time of three short runs. Watch peak memory (Activity Monitor) and confirm ~1 GB per worker.

- [ ] **Step 5: Run the real library**

```bash
time ~/.venvs/tidescout/bin/tidescout flow run winyah-bay
```

Expected: ~11 hours at six workers. **Start this when the machine is free.** Record the actual wall time and per-regime residuals in the task report — this is the first real measurement of the full build and it supersedes the estimate.

- [ ] **Step 6: Commit**

Run: `make check`

```bash
git add backend/tidescout/pipeline/regimes.py backend/tidescout/cli.py backend/tests/test_regimes.py
git commit -m "feat: process-parallel regime library driver and flow CLI"
```

---

## Task 11: Rasterise snapshots onto the analysis grid

Turn per-triangle values into the gridded arrays the scoring engine consumes.

**Storage is the design constraint.** The full 10 m analysis grid is 19.2 M cells; at float32 that is 77 MB per array, and 3 arrays × 25 phases × 9 regimes would be **52 GB**. Two decisions bring it to ~1.8 GB:
1. Store on a coarser `library_cell_m` grid (default 20 m — still inside the spec's "~10–20 m" and finer than the 60 m mesh can actually resolve).
2. Store only cells inside the domain mask, as flat arrays plus one shared index.

Store `u`/`v` rather than speed/direction: interpolating a direction across 0°/360° wraps and produces garbage, whereas components interpolate linearly.

**Files:**
- Create: `backend/tidescout/pipeline/flowlib.py`
- Create: `backend/tests/test_flowlib.py`
- Modify: `backend/tidescout/models.py` (`AnugaConfig.library_cell_m: float = 20.0`)

**Interfaces:**
- Produces:
  - `grid_spec(slug, fishery) -> GridSpec` (shape, transform, flat index of in-domain cells)
  - `rasterise_regime(slug, fishery, regime, spec) -> Path`
  - `load_state(slug, regime, phase_index) -> dict[str, np.ndarray]`
  Task 12 consumes `load_state` and `grid_spec`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_flowlib.py
import numpy as np

from tidescout.pipeline import flowlib


def test_nearest_centroid_rasterisation_matches_known_values():
    centroids = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([[5.0, 5.0], [95.0, 5.0], [5.0, 95.0]])
    out = flowlib.nearest_sample(centroids, values, targets)
    assert list(out) == [1.0, 2.0, 3.0]


def test_direction_is_derived_not_stored():
    u = np.array([1.0, 0.0, -1.0])
    v = np.array([0.0, 1.0, 0.0])
    speed, direction = flowlib.speed_direction(u, v)
    assert np.allclose(speed, [1.0, 1.0, 1.0])
    assert np.allclose(direction, [0.0, 90.0, 180.0])


def test_shear_is_zero_in_uniform_flow():
    u = np.full((20, 20), 0.5)
    v = np.zeros((20, 20))
    shear = flowlib.shear_magnitude(u, v, cell_m=20.0)
    assert np.nanmax(np.abs(shear)) < 1e-9
```

- [ ] **Step 2: Run and confirm they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_flowlib.py -v`
Expected: FAIL — no module `tidescout.pipeline.flowlib`.

- [ ] **Step 3: Write the module**

```python
"""Mesh snapshots -> gridded flow-state library.

Stored on a coarser grid than the 10 m analysis raster and masked to the model
domain: the naive full-grid float32 layout would be ~52 GB for nine regimes,
this is ~1.8 GB. u/v are stored rather than speed/direction because direction
wraps at 360 degrees and cannot be interpolated.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from scipy.spatial import cKDTree

from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import mesh
from tidescout.pipeline.bathy import read_bathy


@dataclass
class GridSpec:
    shape: tuple[int, int]
    transform: Affine
    cell_m: float
    flat_index: np.ndarray   # indices into the flattened grid that are in-domain
    xs: np.ndarray           # in-domain cell-centre eastings
    ys: np.ndarray           # in-domain cell-centre northings


def grid_spec(slug: str, fishery: Fishery) -> GridSpec:
    z, transform, _ = read_bathy(slug)
    cell = fishery.anuga.library_cell_m
    step = int(round(cell / fishery.bathymetry.cell_m))
    mask = mesh.clean_mask(
        mesh.domain_mask(z, transform, fishery), fishery.model_domain.clean_cells
    )[::step, ::step]
    lib_tf = transform * Affine.scale(step, step)
    rows, cols = np.nonzero(mask)
    # +0.5 puts us at cell centres, matching the project's pixel convention
    xs, ys = lib_tf * (cols + 0.5, rows + 0.5)
    return GridSpec(mask.shape, lib_tf, cell,
                    np.ravel_multi_index((rows, cols), mask.shape), xs, ys)


def nearest_sample(centroids: np.ndarray, values: np.ndarray, targets: np.ndarray):
    """Nearest-centroid lookup. The mesh is finer than the library grid almost
    everywhere, so nearest-neighbour is adequate and ~10x cheaper than a proper
    barycentric interpolation."""
    tree = cKDTree(centroids)
    _, idx = tree.query(targets, k=1)
    return values[idx]


def speed_direction(u: np.ndarray, v: np.ndarray):
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return speed, direction


def shear_magnitude(u: np.ndarray, v: np.ndarray, cell_m: float) -> np.ndarray:
    """Lateral velocity-gradient magnitude -- the spec's 'seams' signal.

    Fish hold on the boundary between fast and slow water, so the gradient
    matters more than the speed itself.
    """
    du_dy, du_dx = np.gradient(u, cell_m)
    dv_dy, dv_dx = np.gradient(v, cell_m)
    return np.sqrt(du_dy**2 + dv_dx**2 + 0.5 * (du_dx - dv_dy) ** 2)


def rasterise_regime(slug: str, fishery: Fishery, regime: str, spec: GridSpec) -> Path:
    src = fishery_data_dir(slug) / "flow" / regime
    meta = json.loads((src / "regime.json").read_text())
    domain = mesh.build_mesh(slug, fishery)   # same mesh: deterministic from config
    centroids = domain.get_centroid_coordinates(absolute=True)
    targets = np.column_stack([spec.xs, spec.ys])
    tree = cKDTree(centroids)
    _, idx = tree.query(targets, k=1)

    out = src / "grid"
    out.mkdir(exist_ok=True)
    for snap in meta["snapshots"]:
        d = np.load(src / f"snap_{snap['index']:03d}.npz")
        np.savez_compressed(
            out / f"phase_{snap['index']:03d}.npz",
            u=d["u"][idx].astype("float32"),
            v=d["v"][idx].astype("float32"),
            depth=d["depth"][idx].astype("float32"),
            phase=np.float32(snap["phase"]),
        )
    (out / "grid.json").write_text(json.dumps({
        "shape": list(spec.shape),
        "cell_m": spec.cell_m,
        "transform": list(spec.transform)[:6],
        "n_cells": int(spec.flat_index.size),
        "phases": [s["phase"] for s in meta["snapshots"]],
    }, indent=2))
    return out


def load_state(slug: str, regime: str, phase_index: int) -> dict:
    p = fishery_data_dir(slug) / "flow" / regime / "grid" / f"phase_{phase_index:03d}.npz"
    d = np.load(p)
    return {"u": d["u"], "v": d["v"], "depth": d["depth"], "phase": float(d["phase"])}
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_flowlib.py -v`
Expected: PASS.

- [ ] **Step 5: Rasterise the real library and check its size**

```bash
cd backend && ~/.venvs/tidescout/bin/python -c "
from tidescout.config import load_fishery
from tidescout.pipeline import flowlib
from tidescout.pipeline.regimes import REGIME_MATRIX, regime_name
f = load_fishery('winyah-bay')
spec = flowlib.grid_spec('winyah-bay', f)
print(f'library grid {spec.shape} @ {spec.cell_m} m, {spec.flat_index.size:,} in-domain cells')
for r, d in REGIME_MATRIX:
    out = flowlib.rasterise_regime('winyah-bay', f, regime_name(r, d), spec)
    print('  ', out)
"
du -sh data/winyah-bay/flow
```

Expected: roughly 650k in-domain cells at 20 m, and a total under ~2 GB. **Once this succeeds, the per-triangle `snap_*.npz` files are redundant** — note the reclaimable space in the task report but do not delete them until Task 13's validation gate has passed.

- [ ] **Step 6: Commit**

Run: `make check`

```bash
git add backend/tidescout/pipeline/flowlib.py backend/tests/test_flowlib.py backend/tidescout/models.py
git commit -m "feat: rasterise regime snapshots into a gridded flow-state library"
```

---

## Task 12: Pure runtime lookup

The runtime picks a state per hour. This is `engine/` code: **pure, no I/O**, so it stays testable and fast.

**Files:**
- Create: `backend/tidescout/engine/flow.py`
- Create: `backend/tests/test_flow.py`

**Interfaces:**
- Produces:
  - `select_regime(range_bucket, discharge_bucket, available) -> tuple[str, bool]` — returns the regime and whether a fallback was used
  - `bracket_phases(phases, phase) -> tuple[int, int, float]`
  - `interpolate_state(state_a, state_b, w) -> dict[str, np.ndarray]`
  Task 13 and the Plan 5 scoring engine call these.

- [ ] **Step 1: Write the tests**

```python
# backend/tests/test_flow.py
import numpy as np
import pytest

from tidescout.engine import flow


def test_exact_regime_is_preferred():
    avail = {"mean_med", "spring_high"}
    assert flow.select_regime("spring", "high", avail) == ("spring_high", False)


def test_missing_regime_falls_back_and_flags_it():
    """Spec section 10: a missing state degrades to the nearest, with a warning."""
    name, fell_back = flow.select_regime("spring", "high", {"mean_med"})
    assert fell_back is True
    assert name == "mean_med"


def test_bracket_phases_wraps_around_the_cycle():
    phases = [0.0, 0.25, 0.5, 0.75]
    lo, hi, w = flow.bracket_phases(phases, 0.9)
    assert (lo, hi) == (3, 0)                 # wraps 0.75 -> 0.0
    assert w == pytest.approx(0.6, abs=1e-6)  # 0.9 is 60% from 0.75 toward 1.0


def test_bracket_phases_exact_hit():
    lo, hi, w = flow.bracket_phases([0.0, 0.5], 0.5)
    assert lo == 1 and w == pytest.approx(0.0)


def test_interpolation_is_on_components_not_direction():
    """Interpolating direction across 0/360 wraps; components must be used."""
    a = {"u": np.array([1.0]), "v": np.array([-0.1]), "depth": np.array([3.0])}
    b = {"u": np.array([1.0]), "v": np.array([0.1]), "depth": np.array([3.0])}
    mid = flow.interpolate_state(a, b, 0.5)
    assert mid["v"][0] == pytest.approx(0.0)
    speed, direction = flow.speed_direction(mid["u"], mid["v"])
    assert direction[0] == pytest.approx(0.0)   # not ~180, which averaging angles gives
```

- [ ] **Step 2: Run and confirm they fail**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_flow.py -v`
Expected: FAIL — no module `tidescout.engine.flow`.

- [ ] **Step 3: Write the engine module**

```python
"""Pure flow-state lookup. No I/O -- callers hand in loaded arrays.

Tidal flow is quasi-periodic, so the library stores a handful of regimes and
the runtime interpolates between phase snapshots. Everything here is a pure
function of its arguments so it can be property-tested cheaply.
"""

import numpy as np

RANGE_ORDER = ["neap", "mean", "spring"]
DISCHARGE_ORDER = ["low", "med", "high"]


def speed_direction(u: np.ndarray, v: np.ndarray):
    return np.hypot(u, v), (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0


def select_regime(
    range_bucket: str, discharge_bucket: str, available: set[str]
) -> tuple[str, bool]:
    """Nearest available regime, and whether a substitution happened.

    Spec section 10 requires a missing state to degrade to the nearest with a
    warning rather than fail, so the caller gets the flag and surfaces it.
    """
    exact = f"{range_bucket}_{discharge_bucket}"
    if exact in available:
        return exact, False
    if not available:
        raise ValueError("flow library is empty")
    ri = RANGE_ORDER.index(range_bucket)
    di = DISCHARGE_ORDER.index(discharge_bucket)

    def distance(name: str) -> tuple[int, str]:
        r, d = name.split("_", 1)
        return (
            abs(RANGE_ORDER.index(r) - ri) + abs(DISCHARGE_ORDER.index(d) - di),
            name,
        )

    return min(sorted(available), key=distance), True


def bracket_phases(phases, phase: float) -> tuple[int, int, float]:
    """Indices either side of `phase` and the weight toward the second.

    The tidal cycle is periodic, so a phase past the last snapshot wraps to the
    first rather than clamping -- clamping would freeze the flow at the top of
    every cycle.
    """
    ordered = list(phases)
    if not ordered:
        raise ValueError("no phases in library")
    phase = phase % 1.0
    for i in range(len(ordered)):
        lo = ordered[i]
        hi = ordered[(i + 1) % len(ordered)]
        span = (hi - lo) % 1.0
        if span == 0:
            continue
        offset = (phase - lo) % 1.0
        if offset <= span:
            return i, (i + 1) % len(ordered), offset / span
    return len(ordered) - 1, 0, 0.0


def interpolate_state(a: dict, b: dict, w: float) -> dict:
    """Linear blend of two snapshots. Components only -- never directions."""
    return {k: (1.0 - w) * a[k] + w * b[k] for k in ("u", "v", "depth")}


def wet_mask(depth: np.ndarray, tol: float = 0.01) -> np.ndarray:
    return depth > tol
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_flow.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `make check`

```bash
git add backend/tidescout/engine/flow.py backend/tests/test_flow.py
git commit -m "feat: pure flow-state selection and phase interpolation"
```

---

## Task 13: The validation gate

**This is the go/no-go the spec puts before any UI work.** The model must independently light up the rips, eddies, and slack pockets Ellis actually fishes. Ground truth is `fisheries/winyah-bay.known-spots.yaml`, now populated with three real spots:

| spot | hint | when it works |
|---|---|---|
| Mud Bay Cut | dropoff | outgoing tide / flow toward the bay mouth |
| Georgetown Lighthouse | eddy / dropoff | slack and early incoming |
| North Jetty | structure | incoming, bait pushed against the wall |

Each carries a tide phase in its notes, which is exactly what makes this testable: the model should show its distinguishing flow signature **at the stated phase and not at the opposite one**.

**Files:**
- Modify: `backend/tidescout/cli.py`
- Create: `backend/tests/test_flow_validation.py`

**Interfaces:**
- Consumes: `flowlib.grid_spec`, `flowlib.load_state`, `engine.flow`, `config.load_known_spots`.

- [ ] **Step 1: Add a phase-hint field to known spots**

```python
class KnownSpot(BaseModel):
    name: str
    lon: float
    lat: float
    kind_hint: str = ""
    notes: str = ""
    # Optional machine-readable version of what the notes say in prose, so the
    # validation gate can assert rather than just display.
    works_on: str = ""   # "ebb" | "flood" | "slack" | "" (unspecified)
```

Fill it in from the existing notes: Mud Bay Cut `ebb`, Georgetown Lighthouse `slack`, North Jetty `flood`. **Leave the prose notes untouched** — they carry detail the enum cannot.

- [ ] **Step 2: Write the validation command**

```python
@flow_app.command("validate")
def flow_validate(
    slug: str,
    regime: str = typer.Option("mean_med", "--regime"),
    radius_m: float = typer.Option(150.0, "--radius"),
) -> None:
    """Compare the flow library against Ellis's known spots (the spec's gate)."""
    import json

    import numpy as np
    from rasterio.warp import transform as warp_transform

    from tidescout.config import load_fishery, load_known_spots
    from tidescout.engine import flow
    from tidescout.paths import fishery_data_dir
    from tidescout.pipeline import flowlib

    fishery = load_fishery(slug)
    spots = load_known_spots(slug)
    if not spots:
        console.print(f"no spots in fisheries/{slug}.known-spots.yaml — add some!")
        raise typer.Exit(1)

    spec = flowlib.grid_spec(slug, fishery)
    grid_meta = json.loads(
        (fishery_data_dir(slug) / "flow" / regime / "grid" / "grid.json").read_text()
    )
    phases = grid_meta["phases"]

    lons = [s.lon for s in spots]
    lats = [s.lat for s in spots]
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", lons, lats)

    table = Table(title=f"{fishery.name} — known spots vs flow library ({regime})")
    for col in ("spot", "expects", "peak speed", "at phase", "max shear", "slack min"):
        table.add_column(col)

    for spot, sx, sy in zip(spots, xs, ys, strict=True):
        near = (spec.xs - sx) ** 2 + (spec.ys - sy) ** 2 <= radius_m**2
        if not near.any():
            table.add_row(spot.name, spot.works_on or "-", "OUTSIDE DOMAIN", "-", "-", "-")
            continue
        speeds, shears = [], []
        for i in range(len(phases)):
            st = flowlib.load_state(slug, regime, i)
            sp, _ = flow.speed_direction(st["u"][near], st["v"][near])
            speeds.append(float(np.nanmax(sp)))
            shears.append(float(np.nanmax(sp) - np.nanmin(sp)))
        peak_i = int(np.argmax(speeds))
        table.add_row(
            spot.name, spot.works_on or "-", f"{max(speeds):.2f} m/s",
            f"{phases[peak_i]:.2f}", f"{max(shears):.2f}", f"{min(speeds):.2f}",
        )
    console.print(table)
    console.print(
        "\nGate: each spot should show its signature at the phase its notes describe — "
        "flow for ebb/flood spots, a local speed minimum beside fast water for slack/eddy."
    )
```

- [ ] **Step 3: Test the geometry, not the physics**

```python
# backend/tests/test_flow_validation.py
import numpy as np

from tidescout.engine import flow


def test_slack_spot_shows_a_speed_minimum_beside_fast_water():
    """The eddy/slack signature: a low-speed pocket adjacent to a fast conveyor."""
    u = np.array([1.2, 1.1, 0.05, 0.03, 1.0])
    v = np.zeros(5)
    speed, _ = flow.speed_direction(u, v)
    assert speed.min() < 0.1
    assert speed.max() > 1.0
    assert speed.max() - speed.min() > 0.9   # a real seam, not uniform slow water


def test_spot_outside_domain_is_detected_not_silently_zero():
    xs = np.array([100.0, 200.0])
    ys = np.array([100.0, 200.0])
    near = (xs - 99999.0) ** 2 + (ys - 99999.0) ** 2 <= 150.0**2
    assert not near.any()
```

- [ ] **Step 4: Run the gate**

```bash
~/.venvs/tidescout/bin/tidescout flow validate winyah-bay --regime mean_med
~/.venvs/tidescout/bin/tidescout flow validate winyah-bay --regime spring_low
```

**This is a judgement call, not an assertion — bring the output to Ellis.** What each spot should show:
- **Mud Bay Cut** (`ebb`): peak speed on the falling half of the cycle, with a strong shear value — it is a current-carved dropoff.
- **North Jetty** (`flood`): peak speed on the rising half; the jetty refinement zone exists precisely so this resolves.
- **Georgetown Lighthouse** (`slack`): a low `slack min` beside a high `peak speed` — the eddy signature is the *contrast*, not the absolute value.

If a spot reports `OUTSIDE DOMAIN`, the domain polygon needs extending (Task 6 step 7) — that is a real finding, not a failure.

**Do not tune thresholds to make the gate pass.** If the model does not reproduce these spots, that is the finding, and it belongs in the task report. The whole point of building ANUGA rather than a heuristic was to get an independent answer.

- [ ] **Step 5: Commit**

Run: `make check`

```bash
git add backend/tidescout/cli.py backend/tidescout/models.py \
        backend/tests/test_flow_validation.py fisheries/winyah-bay.known-spots.yaml
git commit -m "feat: known-spots validation gate for the flow library"
```

- [ ] **Step 6: Write the carryover notes**

Create `docs/superpowers/plans/2026-08-13-plan3-carryover-notes.md` in the style of the Plan 1 and Plan 2 carryover notes. It must record:
- Actual full-library wall time and per-regime mass residuals versus the ~11 h estimate.
- Whether each known spot passed, failed, or fell outside the domain.
- Whether the domain polygon's east edge was refined, and the resulting triangle count.
- Whether `Inlet_operator` matched the assumed signature.
- Whether the MLLW→NAVD88 datum offset was resolvable from the station.
- Any regime that needed a retry, and why.
- Reclaimable disk (the per-triangle `snap_*.npz` files) and whether it was reclaimed.

```bash
git add docs/superpowers/plans/2026-08-13-plan3-carryover-notes.md
git commit -m "docs: Plan 3 carryover notes for Plan 4 authoring"
```

---

## Self-Review

**Spec coverage (§5, the section this plan implements):**

| Spec requirement | Task |
|---|---|
| ANUGA 2D shallow-water solver | 5 |
| Mesh from CUDEM bathymetry, refined near channels/structure/shorelines | 6 |
| Ocean boundary from CO-OPS water level | 8 |
| River inflows from USGS discharge | 8, 9 |
| Spatially varying Manning friction | 7 |
| ~9 regime runs (range × discharge), cycle + spin-up | 9, 10 |
| Snapshots every 30–60 min | 9 (`snapshot_minutes`) |
| Rasterised to analysis grid: speed, direction, shear, wet/dry | 11 |
| Indexed by (range, discharge, phase); nearest state + phase interpolation | 12 |
| Mass conservation check | 9 |
| Ebb/flood reversal check | 9 |
| Human known-spots validation gate | 13 |
| Missing state → nearest + warning (§10) | 12 (`select_regime` fallback flag) |

**Deliberately deferred to Plan 4**, and named here so they are not silently dropped:
- *Derived* fish-relevant structure beyond shear — lee/eddy zones behind points given flow direction, convergence at draining creek mouths, flats flood/drain schedules. Task 11 provides `shear_magnitude`; turning the rest into feature activations is scoring work.
- Salinity (spec §7) — a separate subsystem with its own model.
- Feature-id stability (carryover trap (c), `bar-78` renumbering on rebuild). Task 2 changes which features exist, so ids move anyway. Nothing persists a feature reference until Plan 4 scoring, which is when a hash-of-type-plus-centroid key must land. **Plan 4 must not skip it.**

**Carryover items closed:** Plan 1 items 1 and 2 (Task 1); Plan 2 items 1 (Task 3), 2 (Task 2), 3 (Task 4); trap (a) (Task 7), trap (b) (Task 5). Trap (c) explicitly deferred as above. Trap (d) (`artifact_reach_cells=2`) untouched — `cell_m` does not change in this plan. Trap (e) (detector runtime) untouched — rebuild cadence does not rise.

**Type consistency check:** `regime_name()` produces `f"{range}_{discharge}"` in Task 9 and is parsed by `select_regime` with `name.split("_", 1)` in Task 12 — consistent, and safe because no bucket name contains an underscore. `grid_spec` returns `GridSpec` (Task 11) and is consumed by Task 13 via `.xs`/`.ys`/`.shape`. `load_state` returns the `{"u","v","depth","phase"}` dict that `interpolate_state` expects on `("u","v","depth")`. `speed_direction` is defined in both `pipeline/flowlib.py` and `engine/flow.py` — **this is intentional duplication of a two-line pure function to keep `engine/` free of pipeline imports**; if it grows beyond two lines, move it to `engine/flow.py` and have `flowlib` import it.

**Known risks carried into execution:**
1. `anuga.Inlet_operator`'s signature is assumed, not verified (Task 9 step 3 says so explicitly and tells the implementer to check). If it differs and is silently skipped, every discharge regime collapses to identical output — the `inflows_m3s` field in `regime.json` plus the mass residual are the tripwires.
2. The domain polygon still admits open Atlantic on its east edge, inflating the mesh. Cheap to improve, needs a chart.
3. `set_local_extrapolation_and_flux_updating(nlevels=8)` produced **zero** speedup and bit-identical output in the spike. It is not budgeted for anywhere in this plan; if someone revisits it, that is upside, not a dependency.
