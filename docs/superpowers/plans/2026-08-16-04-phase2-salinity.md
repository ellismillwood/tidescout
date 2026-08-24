# Plan 4 Phase 2 — Salinity and the Discharge Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every cell in the bay an hourly salinity, driven by real river discharge, and fix the way discharge enters the rest of the system — because the flow library represents it as three steps spanning a quarter of its observed range.

**Architecture:** Salinity is **analytic, not simulated**. One static layer — geodesic along-estuary distance from the sea through the water mask — is computed once; at runtime, salinity at a cell is a closed-form function of that distance, the lagged composite discharge, and the tide phase. No stored field, no ANUGA run, and crucially **no bucketing**: the intrusion model reads discharge in cfs across its full 1,232–22,996 range, where the flow library only knows three values. A parallel change teaches the flow lookup to interpolate along its discharge axis instead of snapping to it.

**Tech Stack:** Python 3.12, numpy 2.5.2, scipy 1.18.0 (`sparse.csgraph.dijkstra`), httpx + respx, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` — §7 (salinity as a fish-distribution layer) is what this plan implements; §4 (data sources), §8 factor 8, and §10 (resilience) constrain it.

**Depends on:** Phase 1 Task 2 (`inflow_share`) and Task 3 (`structure.to_grid`/`from_grid`). Do not start before both have landed.

---

## Global Constraints

Identical to Phase 1 — Python ≥3.12, ruff `["E","F","I","UP","B","DTZ"]` at line-length 100, `make check` green before every commit, `engine/` pure, tests never hit live APIs, library grid `(2527, 1903)` at 20 m with 587,325 in-domain cells, phase 0 is LOW water. Plus:

- **Salinity units are practical salinity units (ppt) throughout.** USGS parameter `00480` reports ppth, which is numerically the same; do not convert. Specific conductance (`00095`) is **not** interchangeable and is out of scope.
- **Discharge units are cfs at the boundary and m³/s inside the model.** `forcing.CFS_TO_M3S` is the only conversion; never inline the constant.
- **The intrusion model must never extrapolate silently.** Every output carries the discharge it was evaluated at and a flag when that value sits outside the calibration range.

---

## What the flow library actually does with discharge, and why this phase exists

Measured from the shipped config on 2026-08-16:

| Bucket | Composite | Origin |
|---|---|---|
| `low` | 2,774 cfs | p25 of 365 days |
| `med` | 4,533 cfs | midpoint of low/high — **not** the median (3,866) |
| `high` | 6,292 cfs | p75 |

The observed range over the same record is **1,232 – 22,996 cfs**. So the library's top bucket is the p75, the observed maximum is **3.7× above it**, and `usgs.discharge_summary` buckets everything over 6,292 as `high` — collapsing the entire top quartile of days onto one simulated point sitting at the *bottom edge* of its own range.

For the flow field this is defensible: Plan 3 measured the discharge axis moving domain-mean depth ~1 cm and barely touching velocity, which is why `RANGE_STEP_COST=3` vs `DISCHARGE_STEP_COST=1` refuses to trade range for discharge. **For salinity it is disqualifying** — intrusion length goes as `Q^(-k)`, so the salt front at 23,000 cfs sits kilometres from where it sits at 6,292.

Hence the split this phase makes: **salinity reads continuous discharge; the flow library keeps its buckets but learns to interpolate between them.**

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `backend/tidescout/pipeline/estuary.py` | Geodesic along-estuary distance field (built once per fishery) |
| `backend/tidescout/engine/salinity.py` | Pure intrusion model: distance + discharge + phase → ppt |
| `backend/tidescout/sources/coops_water.py` | CO-OPS physical-oceanography salinity (the ocean end-member) |
| `backend/tests/test_estuary.py` | Distance field on synthetic channel geometry |
| `backend/tests/test_salinity.py` | Intrusion model behaviour and calibration |
| `backend/tests/test_coops_water.py` | Recorded-fixture tests for the CO-OPS fetcher |

**Modified files:**

| Path | Change |
|---|---|
| `backend/tidescout/models.py` | `SalinityConfig`; `Stations.ocean_salinity` |
| `fisheries/winyah-bay.yaml` | `salinity:` block; Springmaid Pier as ocean end-member |
| `backend/tidescout/sources/usgs.py` | `branch_discharge_cfs`; discharge trend on `DischargeSummary` |
| `backend/tidescout/engine/flow.py` | `blend_regimes` — discharge-axis interpolation |
| `backend/tidescout/cli.py` | `tidescout salinity field|calibrate` |

---

## Task 1: The along-estuary distance field

Salinity is a function of how far the water is from the sea **through water**, not in a straight line — a cell 2 km from the ocean across a barrier island is 30 km from it up the channel. A geodesic distance transform over the domain mask gives every cell that number once, and branching up the Pee Dee, Waccamaw, Black and Sampit falls out for free.

**Files:**
- Create: `backend/tidescout/pipeline/estuary.py`, `backend/tests/test_estuary.py`

**Interfaces:**
- Consumes: `flowlib.grid_spec`, `fishery.model_domain.ocean_boundary_utm_km`
- Produces:
  - `along_estuary_km(spec, seed_mask) -> np.ndarray` — 1-D, library-masked, km from the sea, NaN where unreachable. Seed-agnostic: takes a boolean seed mask, not a polygon, so the Dijkstra core stays pure and Task 3/5 callers can pass their own seeds without going through `ocean_seed_mask`.
  - `ocean_seed_mask(spec, ocean_boundary_utm_km, bed_elev_m, ocean_max_z_m) -> np.ndarray` — the polygon-to-seed adapter: in-domain cells that are on the domain's outer edge, below `ocean_max_z_m`, inside the authored polygon, and in the largest connected component of that set (see `estuary.py` for why all four conditions are needed)
  - `build_distance_field(slug, fishery) -> Path` → `data/<slug>/estuary_km.npy`
  - `load_distance_field(slug) -> np.ndarray`
  Task 3 and Task 5 both consume the 1-D array.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_estuary.py
"""Geodesic distance on hand-built channel geometry.

Straight-line distance is the wrong answer everywhere in an estuary, so these
fixtures are built so the two answers differ measurably.
"""

import numpy as np
import pytest
from affine import Affine

from tidescout.pipeline import estuary


class _Spec:
    """A 20x20 grid of 100 m cells, with an explicit in-domain mask."""

    def __init__(self, mask):
        self.shape = mask.shape
        self.cell_m = 100.0
        self.transform = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 2000.0)
        rows, cols = np.nonzero(mask)
        self.flat_index = np.ravel_multi_index((rows, cols), mask.shape)
        self.xs, self.ys = self.transform * (cols + 0.5, rows + 0.5)


def test_distance_grows_along_a_straight_channel():
    mask = np.zeros((20, 20), bool)
    mask[10, :] = True                       # one east-west channel
    spec = _Spec(mask)
    seeds = spec.xs <= 100.0                 # the westernmost cell is the sea

    d = estuary.along_estuary_km(spec, seed_mask=seeds)

    order = np.argsort(spec.xs)
    assert d[order][0] == pytest.approx(0.0)
    assert np.all(np.diff(d[order]) > 0), "distance must increase away from the sea"
    assert d[order][-1] == pytest.approx(1.9, abs=0.05)  # 19 cells x 100 m


def test_distance_follows_water_around_a_barrier_not_through_it():
    """The whole point: a U-shaped channel puts the far end 100 m away in a
    straight line and ~2 km away through water."""
    mask = np.zeros((20, 20), bool)
    mask[5, 2:18] = True     # north leg
    mask[5:15, 17] = True    # east connector
    mask[14, 2:18] = True    # south leg, ending beside the start
    spec = _Spec(mask)
    seeds = (spec.ys > 1400.0) & (spec.xs < 300.0)   # west end of the north leg

    d = estuary.along_estuary_km(spec, seed_mask=seeds)

    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    far = (rows == 14) & (cols == 2)          # 900 m south in a straight line
    assert d[far][0] > 3.0, "must route the long way around, not across land"


def test_unreachable_water_is_nan_not_zero():
    """An isolated pond has no route to the sea. Zero would read as 'at the
    mouth', which is maximally salty -- the most wrong answer available."""
    mask = np.zeros((20, 20), bool)
    mask[10, 0:5] = True
    mask[2, 15:19] = True       # disconnected
    spec = _Spec(mask)
    seeds = spec.xs <= 100.0

    d = estuary.along_estuary_km(spec, seed_mask=seeds)
    rows, _ = np.unravel_index(spec.flat_index, spec.shape)
    assert np.all(np.isnan(d[rows == 2]))
    assert np.all(np.isfinite(d[rows == 10]))


def test_diagonal_steps_cost_more_than_orthogonal_ones():
    """8-connectivity with equal weights would make a diagonal channel read
    30% shorter than it is."""
    mask = np.zeros((20, 20), bool)
    for i in range(10):
        mask[i, i] = True
    spec = _Spec(mask)
    seeds = (spec.xs < 100.0) & (spec.ys > 1900.0)

    d = estuary.along_estuary_km(spec, seed_mask=seeds)
    assert np.nanmax(d) == pytest.approx(9 * 100.0 * np.sqrt(2) / 1000.0, rel=0.02)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_estuary.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.pipeline.estuary`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/pipeline/estuary.py
"""Along-estuary distance: how far each cell is from the sea THROUGH WATER.

Salinity depends on channel distance, not straight-line distance -- a cell 2 km
from the ocean across a barrier island is 30 km from it up the channel, and the
two answers differ by an order of magnitude over most of Winyah Bay. This walks
the domain mask as a graph, so the branching up the Pee Dee, Waccamaw, Black and
Sampit needs no special handling: each branch simply gets longer.

Built once per fishery. The result is static -- geometry, not state.
"""

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Point, Polygon

from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir

# 8-connectivity. Orthogonal steps cost one cell, diagonals sqrt(2) -- with
# equal weights a diagonal channel would measure ~30% shorter than it is.
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
    (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
]


def along_estuary_km(spec, seed_mask: np.ndarray) -> np.ndarray:
    """Geodesic distance in km from the seeded cells, over in-domain cells only.

    `seed_mask` is a boolean over the same 1-D layout as the library arrays.
    Cells with no water route to a seed come back NaN, never 0.0: zero means
    "at the mouth", which is the saltiest place in the model and so the most
    damaging possible default for an isolated pond.
    """
    n = spec.flat_index.size
    if not seed_mask.any():
        raise ValueError(
            "no seed cells -- the ocean polygon selects nothing inside the "
            "model domain, so there is no sea to measure distance from"
        )

    rows, cols = np.unravel_index(spec.flat_index, spec.shape)
    # Position -> compact node id, for O(1) neighbour lookup.
    lookup = np.full(int(spec.shape[0]) * int(spec.shape[1]), -1, dtype="int64")
    lookup[spec.flat_index] = np.arange(n)

    src, dst, weight = [], [], []
    for dr, dc, cost in _NEIGHBOURS:
        nr, nc = rows + dr, cols + dc
        ok = (nr >= 0) & (nr < spec.shape[0]) & (nc >= 0) & (nc < spec.shape[1])
        nid = np.full(n, -1, dtype="int64")
        nid[ok] = lookup[np.ravel_multi_index((nr[ok], nc[ok]), spec.shape)]
        joined = nid >= 0
        src.append(np.nonzero(joined)[0])
        dst.append(nid[joined])
        weight.append(np.full(int(joined.sum()), cost * spec.cell_m))

    graph = coo_matrix(
        (np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))),
        shape=(n, n),
    ).tocsr()

    d = dijkstra(graph, directed=False, indices=np.nonzero(seed_mask)[0], min_only=True)
    d = np.asarray(d, dtype="float64") / 1000.0
    d[np.isinf(d)] = np.nan
    return d


def ocean_seed_mask(spec, ocean_boundary_utm_km: list) -> np.ndarray:
    """In-domain cells lying inside the authored ocean polygon: the sea itself.

    Reuses `model_domain.ocean_boundary_utm_km` rather than inferring the mouth
    from depth. Plan 3 established twice over that depth cannot classify
    geography -- it put the ocean tide 40 km up the Pee Dee -- and the seaward
    opening is already authored, so there is nothing to infer.
    """
    if not ocean_boundary_utm_km:
        raise ValueError(
            "model_domain.ocean_boundary_utm_km is empty -- the along-estuary "
            "distance field has no sea to measure from"
        )
    poly = Polygon([(x * 1000.0, y * 1000.0) for x, y in ocean_boundary_utm_km])
    if not poly.is_valid:
        raise ValueError("ocean_boundary_utm_km is not a valid polygon")
    return np.fromiter(
        (poly.contains(Point(x, y)) for x, y in zip(spec.xs, spec.ys, strict=True)),
        dtype=bool,
        count=spec.xs.size,
    )


def build_distance_field(slug: str, fishery: Fishery) -> Path:
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    seeds = ocean_seed_mask(spec, fishery.model_domain.ocean_boundary_utm_km)
    d = along_estuary_km(spec, seeds)
    path = fishery_data_dir(slug) / "estuary_km.npy"
    np.save(path, d.astype("float32"))
    return path


def load_distance_field(slug: str) -> np.ndarray:
    path = fishery_data_dir(slug) / "estuary_km.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"no along-estuary distance field at {path} -- run "
            f"`tidescout salinity field {slug}` first"
        )
    return np.load(path)
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_estuary.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Build it for real and check it against known geography**

```bash
python -c "
from tidescout.config import load_fishery
from tidescout.pipeline import estuary
from rasterio.warp import transform as warp_transform
import numpy as np
f = load_fishery('winyah-bay')
p = estuary.build_distance_field('winyah-bay', f)
d = np.load(p)
print(f'{np.isfinite(d).sum():,} reachable of {d.size:,} cells; '
      f'max {np.nanmax(d):.1f} km')
# Sample the three known spots and the three river inlets.
pts = {'North Jetty': (-79.166104, 33.205355),
       'Georgetown Lighthouse': (-79.186965, 33.222133),
       'Mud Bay Cut': (-79.221461, 33.278208)}
for r in f.rivers:
    pts[r.name + ' inlet'] = r.inflow_lonlat
from tidescout.pipeline.flowlib import grid_spec
spec = grid_spec('winyah-bay', f)
for name, (lon, lat) in pts.items():
    x, y = warp_transform('EPSG:4326', 'EPSG:26917', [lon], [lat])
    i = int(np.argmin((spec.xs - x[0])**2 + (spec.ys - y[0])**2))
    print(f'  {name:24s} {d[i]:6.2f} km from the sea')
"
```

Expected ordering, from the geography: **North Jetty ≈ 0 km** (it is the mouth), **Georgetown Lighthouse a few km**, **Mud Bay Cut further**, and the **three river inlets furthest of all (tens of km)**. If a river inlet comes back as NaN, the inlet point sits outside the library mask and Phase 1's inflow work needs revisiting. If North Jetty is not near zero, the seed mask is wrong.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/estuary.py backend/tests/test_estuary.py
git commit -m "feat: geodesic along-estuary distance field"
```

---

## Task 2: Per-branch discharge and the freshet limb

The intrusion model needs discharge as a number, not a bucket, and it needs to know whether that number is rising or falling. The salt front lags discharge by days, so a bay at 12,000 cfs on the rising limb of a freshet is in a different state from the same bay at 12,000 cfs three days into recovery.

`DischargeSummary` already carries `cfs_now` and `cfs_lagged`; their ratio is the limb, free of charge.

**Files:**
- Modify: `backend/tidescout/sources/usgs.py` (`DischargeSummary`, `discharge_summary`)
- Test: `backend/tests/test_usgs.py`

**Interfaces:**
- Consumes: `RiverGauge.inflow_share` (Phase 1 Task 2)
- Produces: `DischargeSummary.trend: float | None` (ratio now/lagged, `None` when either is missing), `DischargeSummary.limb: str` (`"rising"` / `"falling"` / `"steady"` / `"unknown"`), and `branch_discharge_cfs(fishery, summary) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_usgs.py (append)
import pytest

from tidescout.config import load_fishery
from tidescout.sources.usgs import DischargeSummary, branch_discharge_cfs, classify_limb


def _summary(now, lagged):
    return DischargeSummary(now, lagged, "med", [], [], [])


def test_limb_is_rising_when_todays_flow_exceeds_the_lagged_mean():
    assert classify_limb(_summary(12000.0, 4000.0)) == "rising"


def test_limb_is_falling_during_recovery():
    assert classify_limb(_summary(4000.0, 12000.0)) == "falling"


def test_limb_is_steady_inside_the_dead_band():
    """Gauge noise and diurnal variation are not a freshet."""
    assert classify_limb(_summary(4100.0, 4000.0)) == "steady"


def test_limb_is_unknown_when_a_gauge_is_dark():
    assert classify_limb(_summary(None, 4000.0)) == "unknown"
    assert classify_limb(_summary(4000.0, None)) == "unknown"


def test_branch_discharge_splits_by_inflow_share():
    """The Pee Dee carries 78% of the freshwater, so it carries 78% of the
    freshet -- the same correction Phase 1 made to the ANUGA forcing, applied
    to the runtime path that salinity actually reads."""
    f = load_fishery("winyah-bay")
    branches = branch_discharge_cfs(f, _summary(10000.0, 9000.0))
    assert branches["Pee Dee"] == pytest.approx(9000.0 * 0.783, rel=1e-6)
    assert branches["Black"] == pytest.approx(9000.0 * 0.083, rel=1e-6)
    assert sum(branches.values()) == pytest.approx(9000.0, rel=1e-9)


def test_branch_discharge_prefers_lagged_flow_because_the_salt_front_lags():
    """The bay's salinity today reflects the last day or two of river flow,
    not this instant's gauge reading."""
    f = load_fishery("winyah-bay")
    branches = branch_discharge_cfs(f, _summary(20000.0, 5000.0))
    assert sum(branches.values()) == pytest.approx(5000.0, rel=1e-9)


def test_branch_discharge_falls_back_to_now_when_no_lagged_value_exists():
    f = load_fishery("winyah-bay")
    branches = branch_discharge_cfs(f, _summary(7000.0, None))
    assert sum(branches.values()) == pytest.approx(7000.0, rel=1e-9)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_usgs.py -k "limb or branch" -v`
Expected: FAIL — `ImportError: cannot import name 'classify_limb'`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/sources/usgs.py -- add after DischargeSummary

# Fractional change between today's flow and the 24-48 h lagged mean, below
# which the river is called steady. Gauge noise and the diurnal cycle move a
# coastal-plain river a few percent on any quiet day; 15% is comfortably above
# that and comfortably below any real rain event.
LIMB_DEAD_BAND = 0.15


def classify_limb(summary: DischargeSummary) -> str:
    """"rising" / "falling" / "steady" / "unknown".

    The salt front lags discharge by days, so the same flow means different
    things on the way up and on the way down: a rising limb is a freshet
    arriving and fish moving down-bay ahead of it; a falling limb is the slower
    recovery as salt creeps back. The level alone cannot distinguish them.
    """
    if summary.cfs_now is None or summary.cfs_lagged is None or not summary.cfs_lagged:
        return "unknown"
    change = (summary.cfs_now - summary.cfs_lagged) / summary.cfs_lagged
    if change > LIMB_DEAD_BAND:
        return "rising"
    if change < -LIMB_DEAD_BAND:
        return "falling"
    return "steady"


def branch_discharge_cfs(
    fishery: Fishery, summary: DischargeSummary
) -> dict[str, float]:
    """Composite discharge split across rivers by their measured share.

    Uses `cfs_lagged` in preference to `cfs_now`: the bay's salinity today
    reflects the last day or two of river flow, not this instant's reading.

    Splits by `inflow_share`, not gauge `weight` -- see Phase 1 Task 2. This is
    the runtime twin of the ANUGA forcing correction, and it matters more here:
    intrusion length is a strong function of the discharge on each branch, so a
    78/13/8 river system modelled as equal thirds puts the salt front in the
    wrong place on all three.
    """
    basis = summary.cfs_lagged if summary.cfs_lagged is not None else summary.cfs_now
    if basis is None:
        return {}
    shares = [r.inflow_share for r in fishery.rivers]
    if any(s is None for s in shares):
        n = len(fishery.rivers) or 1
        shares = [1.0 / n] * len(fishery.rivers)
    return {
        r.name: basis * s for r, s in zip(fishery.rivers, shares, strict=True)
    }
```

Add `trend` and `limb` to `DischargeSummary` as computed fields, and populate them at the end of `discharge_summary`:

```python
# in the DischargeSummary dataclass
    trend: float | None = None   # cfs_now / cfs_lagged
    limb: str = "unknown"

# at the end of discharge_summary, before the return
    summary = DischargeSummary(cfs_now, cfs_lagged, bucket, sites, contributing, stale)
    summary.trend = (
        cfs_now / cfs_lagged
        if cfs_now is not None and cfs_lagged not in (None, 0.0)
        else None
    )
    summary.limb = classify_limb(summary)
    return summary
```

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_usgs.py -v && make check`

```bash
git add backend/tidescout/sources/usgs.py backend/tests/test_usgs.py
git commit -m "feat: per-branch discharge and freshet limb classification"
```

---

## Task 3: The salt-intrusion model

The core. Salinity at a cell is a closed-form function of three numbers: distance from the sea, discharge, and tide phase.

```
S(x, Q, phase) = S_ocean · exp(−x_eff / L(Q))
L(Q)          = L₀ · (Q / Q₀)^(−k)
x_eff         = max(0, x + E · cos(2π · phase))
```

`L` is the intrusion length scale, shrinking as discharge rises — the classic Savenije-family scaling, with `k ≈ 1/3` as the theoretical starting point. `E` is the tidal excursion: the salt field slides seaward and landward over a cycle. **Phase 0 is LOW water**, so `cos(2π·phase)` is `+1` there — pushing `x_eff` up, making a given cell fresher — and `−1` at high water. Getting that sign backwards inverts the tidal salinity swing everywhere.

**Files:**
- Create: `backend/tidescout/engine/salinity.py`, `backend/tests/test_salinity.py`
- Modify: `backend/tidescout/models.py` (`SalinityConfig`), `fisheries/winyah-bay.yaml`

**Interfaces:**
- Consumes: `estuary.load_distance_field`, `usgs.branch_discharge_cfs`
- Produces:
  - `SalinityConfig` — `ocean_ppt`, `l0_km`, `q0_cfs`, `k`, `excursion_km`, `calibration_range_cfs`
  - `intrusion_length_km(cfs, cfg) -> float`
  - `salinity_at(distance_km, cfs, phase, cfg) -> np.ndarray | float`
  - `SalinityField` dataclass — `ppt: np.ndarray`, `cfs: float`, `extrapolated: bool`
  - `salinity_field(distance_km, cfs, phase, cfg) -> SalinityField`
  Phase 3 Task 5 consumes `SalinityField.ppt` and `.extrapolated`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_salinity.py
"""Intrusion-model behaviour.

These pin SHAPE, not calibrated values -- the constants are fitted in Task 5 and
will move. Every assertion here must survive recalibration.
"""

import numpy as np
import pytest

from tidescout.engine import salinity
from tidescout.models import SalinityConfig

CFG = SalinityConfig(
    ocean_ppt=34.0, l0_km=18.0, q0_cfs=4000.0, k=0.33, excursion_km=7.0,
    calibration_range_cfs=(1232.0, 22996.0),
)


def test_salinity_falls_monotonically_up_the_estuary():
    x = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
    s = salinity.salinity_at(x, cfs=4000.0, phase=0.25, cfg=CFG)
    assert np.all(np.diff(s) < 0)
    assert s[0] == pytest.approx(CFG.ocean_ppt, rel=0.05)


def test_higher_discharge_pushes_the_salt_front_seaward():
    """The whole reason salinity cannot read a three-value bucket."""
    x = np.full(1, 15.0)
    low = salinity.salinity_at(x, cfs=2000.0, phase=0.25, cfg=CFG)[0]
    high = salinity.salinity_at(x, cfs=20000.0, phase=0.25, cfg=CFG)[0]
    assert high < low
    assert low - high > 2.0, "a 10x discharge change must move salinity materially"


def test_intrusion_length_shrinks_as_a_power_law_in_discharge():
    assert salinity.intrusion_length_km(CFG.q0_cfs, CFG) == pytest.approx(CFG.l0_km)
    doubled = salinity.intrusion_length_km(2 * CFG.q0_cfs, CFG)
    assert doubled == pytest.approx(CFG.l0_km * 2 ** (-CFG.k), rel=1e-6)
    assert doubled < CFG.l0_km


def test_high_water_is_saltier_than_low_water_at_the_same_place():
    """Phase 0 is LOW water here. Inverting this inverts the tidal salinity
    swing over the whole bay, which is exactly the trap Plan 3 documented."""
    x = np.full(1, 12.0)
    at_low = salinity.salinity_at(x, cfs=4000.0, phase=0.0, cfg=CFG)[0]
    at_high = salinity.salinity_at(x, cfs=4000.0, phase=0.5, cfg=CFG)[0]
    assert at_high > at_low


def test_tidal_swing_is_bounded_by_the_excursion():
    """The salt field slides; it does not teleport."""
    x = np.full(1, 12.0)
    swing = [
        salinity.salinity_at(x, cfs=4000.0, phase=p, cfg=CFG)[0]
        for p in np.linspace(0, 1, 24, endpoint=False)
    ]
    span_km = CFG.excursion_km * 2
    bound = CFG.ocean_ppt * (1 - np.exp(-span_km / CFG.l0_km))
    assert max(swing) - min(swing) <= bound + 1e-9


def test_salinity_never_exceeds_the_ocean_end_member():
    x = np.array([0.0, 0.5, 1.0])
    for phase in (0.0, 0.25, 0.5, 0.75):
        s = salinity.salinity_at(x, cfs=1232.0, phase=phase, cfg=CFG)
        assert np.all(s <= CFG.ocean_ppt + 1e-9)


def test_salinity_is_never_negative_far_up_river():
    s = salinity.salinity_at(np.array([200.0]), cfs=22996.0, phase=0.5, cfg=CFG)
    assert s[0] >= 0.0


def test_unreachable_cells_stay_nan():
    """NaN distance means no water route to the sea; it must not become 34 ppt."""
    s = salinity.salinity_at(np.array([np.nan, 10.0]), cfs=4000.0, phase=0.25, cfg=CFG)
    assert np.isnan(s[0]) and np.isfinite(s[1])


def test_field_flags_discharge_outside_the_calibration_range():
    """Silent extrapolation is the failure mode this model is most prone to:
    it returns a confident number for a flow nothing was ever fitted against."""
    x = np.array([10.0])
    inside = salinity.salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG)
    outside = salinity.salinity_field(x, cfs=40000.0, phase=0.25, cfg=CFG)
    assert inside.extrapolated is False
    assert outside.extrapolated is True
    assert outside.cfs == 40000.0


def test_zero_discharge_does_not_divide_by_zero():
    s = salinity.salinity_at(np.array([10.0]), cfs=0.0, phase=0.25, cfg=CFG)
    assert np.isfinite(s[0])
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_salinity.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.engine.salinity`.

- [ ] **Step 3: Add the config**

```python
# backend/tidescout/models.py
class SalinityConfig(BaseModel):
    """Empirical salt-intrusion parameters. Fitted in Phase 2 Task 5.

    The defaults here are theoretical starting points, NOT calibrated values --
    k = 1/3 is the Savenije-family scaling exponent and l0_km is a rough guess
    at Winyah's intrusion length at median flow. Task 5 replaces them and
    records the fit residual alongside.
    """

    ocean_ppt: float = 34.0
    # Intrusion length at q0_cfs. Fitted.
    l0_km: float = 18.0
    q0_cfs: float = 4000.0
    # Power-law exponent: L ~ Q^-k. 1/3 is the theoretical value. Fitted.
    k: float = 0.33
    # Tidal excursion -- how far the salt field slides over a cycle.
    # u_tidal * T / pi with u ~ 0.5 m/s and T = 12.42 h gives ~7 km.
    excursion_km: float = 7.0
    # Discharge span the fit was made over. Outside it, results are flagged
    # rather than silently trusted.
    calibration_range_cfs: tuple[float, float] = (1232.0, 22996.0)


# in Fishery
    salinity: SalinityConfig = SalinityConfig()
```

- [ ] **Step 4: Implement the model**

```python
# backend/tidescout/engine/salinity.py
"""Empirical salt-intrusion model. Pure -- no I/O, no library lookup.

Winyah Bay is river-dominated, so salt-wedge position is a first-order control
on where fish are (spec section 7). This computes it analytically rather than
simulating it, which has one decisive advantage: the flow library knows only
three discharge values spanning 2,774-6,292 cfs, while the observed record runs
1,232-22,996. An analytic model reads the real number.

    S(x, Q, phase) = S_ocean * exp(-x_eff / L(Q))
    L(Q)           = L0 * (Q / Q0)^-k
    x_eff          = max(0, x + E * cos(2*pi*phase))
"""

from dataclasses import dataclass

import numpy as np

from tidescout.models import SalinityConfig


@dataclass
class SalinityField:
    ppt: np.ndarray
    cfs: float
    extrapolated: bool


def intrusion_length_km(cfs: float, cfg: SalinityConfig) -> float:
    """Distance scale over which salinity decays, shrinking as discharge rises.

    A floor of 1 cfs keeps a dry-gauge zero from dividing by zero; at that flow
    the estuary is tidally dominated and the length scale saturates anyway.
    """
    q = max(float(cfs), 1.0)
    return cfg.l0_km * (q / cfg.q0_cfs) ** (-cfg.k)


def salinity_at(distance_km, cfs: float, phase: float, cfg: SalinityConfig):
    """Salinity in ppt at one or many along-estuary distances.

    The tidal term slides the whole profile: phase 0 is LOW water (spin-up is
    0.4831 of a cycle), so cos(2*pi*phase) is +1 there, pushing x_eff UP and
    making a given cell fresher, and -1 at high water. Reversing that sign
    inverts the tidal salinity swing across the entire bay.

    NaN distances -- cells with no water route to the sea -- stay NaN. Treating
    them as 0 km would make an isolated pond the saltiest water in the model.
    """
    x = np.asarray(distance_km, dtype="float64")
    shifted = x + cfg.excursion_km * np.cos(2.0 * np.pi * phase)
    x_eff = np.clip(shifted, 0.0, None)
    return cfg.ocean_ppt * np.exp(-x_eff / intrusion_length_km(cfs, cfg))


def salinity_field(
    distance_km, cfs: float, phase: float, cfg: SalinityConfig
) -> SalinityField:
    """`salinity_at` plus provenance: what discharge, and was it in range.

    The extrapolation flag exists because this model's characteristic failure
    is not a crash -- it is returning a confident number for a discharge nothing
    was ever fitted against. Spec section 10 requires degraded data to be
    surfaced, not swallowed.
    """
    lo, hi = cfg.calibration_range_cfs
    return SalinityField(
        ppt=salinity_at(distance_km, cfs, phase, cfg),
        cfs=float(cfs),
        extrapolated=not (lo <= float(cfs) <= hi),
    )
```

- [ ] **Step 5: Run the tests and commit**

Run: `pytest backend/tests/test_salinity.py -v && make check`
Expected: PASS, 10 tests.

```bash
git add backend/tidescout/engine/salinity.py backend/tidescout/models.py \
        backend/tests/test_salinity.py
git commit -m "feat: empirical salt-intrusion model"
```

---

## Task 4: The ocean end-member

The model needs `S_ocean`, and the only usable observation near Winyah is NOAA CO-OPS **Springmaid Pier (8661070)**, ~50 km NE on the open coast. Probed 2026-08-16: it is the sole `physocean` station within 100 km.

**What is NOT available**, established by direct probe on 2026-08-16 — record this, because it constrains Task 5 and should not be rediscovered:

- **NERR / Oyster Landing is blocked.** `cdmo.baruch.sc.edu/webservices2/requests.cfc` returns `<data>Invalid ip …</data>` for every method. CDMO requires IP registration, which is a form a human must submit. Until that happens, the North Inlet–Winyah Bay NERR stations are unavailable.
- **There is no mid-bay salinity gauge.** The only in-bbox USGS sites reporting `00480` are the two already configured, both far up the Waccamaw at 33.51 and 33.44, reading 0–1 ppth.

So calibration is **two-ended with nothing in the middle**: an ocean value 50 km away and a fresh value 30 km up-river. Task 5 is written to be honest about what that can and cannot constrain.

**Files:**
- Create: `backend/tidescout/sources/coops_water.py`, `backend/tests/test_coops_water.py`
- Modify: `backend/tidescout/models.py` (`Stations.ocean_salinity`), `fisheries/winyah-bay.yaml`

**Interfaces:**
- Produces: `fetch_ocean_salinity(station: str, day: date, cache: Cache) -> float | None`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_coops_water.py
"""Recorded-fixture tests. Never hits the live CO-OPS API."""

from datetime import date

import httpx
import pytest
import respx

from tidescout.sources.cache import Cache
from tidescout.sources.coops_water import fetch_ocean_salinity

URL = "https://api.tidesandcurrents.noaa.gov/api/datagetter"

PAYLOAD = {
    "data": [
        {"t": "2026-08-16 00:00", "s": "33.9"},
        {"t": "2026-08-16 01:00", "s": "34.2"},
        {"t": "2026-08-16 02:00", "s": ""},
    ]
}


@respx.mock
def test_returns_the_mean_of_valid_readings(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, json=PAYLOAD))
    got = fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db"))
    assert got == pytest.approx(34.05, abs=0.01)


@respx.mock
def test_blank_readings_are_skipped_not_read_as_zero(tmp_path):
    """CO-OPS returns an empty string for a dark sensor. Parsed as 0.0 it would
    drag the ocean end-member toward fresh -- the model's most sensitive input."""
    respx.get(URL).mock(
        return_value=httpx.Response(200, json={"data": [{"t": "x", "s": ""}]})
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None


@respx.mock
def test_api_error_payload_returns_none_rather_than_raising(tmp_path):
    """Spec section 10: a dark sensor degrades to the configured default with a
    flag, it does not take down the day's forecast."""
    respx.get(URL).mock(
        return_value=httpx.Response(200, json={"error": {"message": "No data was found"}})
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None


@respx.mock
def test_implausible_values_are_rejected(tmp_path):
    """A stuck sensor reading 0 or 300 ppt must not become S_ocean."""
    respx.get(URL).mock(
        return_value=httpx.Response(
            200, json={"data": [{"t": "x", "s": "300.0"}, {"t": "y", "s": "0.0"}]}
        )
    )
    assert fetch_ocean_salinity("8661070", date(2026, 8, 16), Cache(tmp_path / "c.db")) is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_coops_water.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/sources/coops_water.py
"""CO-OPS physical-oceanography salinity: the model's ocean end-member.

Springmaid Pier (8661070) is the only physocean station within 100 km of Winyah
Bay -- checked against the CO-OPS mdapi station list on 2026-08-16. It sits ~50
km NE on the open coast, so it measures shelf water rather than anything inside
the bay, which is exactly the role it plays here: S_ocean, the boundary value
the intrusion profile decays away from.
"""

from datetime import date

import httpx

from tidescout.sources.cache import Cache

DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/datagetter"

# Open shelf water off South Carolina runs ~30-36 ppt. Anything outside this is
# a stuck or miscalibrated sensor, and S_ocean is the single most influential
# constant in the model -- every cell's salinity scales linearly with it.
PLAUSIBLE_PPT = (25.0, 40.0)


def fetch_ocean_salinity(station: str, day: date, cache: Cache) -> float | None:
    """Daily mean salinity in ppt, or None if the sensor gives nothing usable."""
    params = {
        "product": "salinity",
        "station": station,
        "begin_date": day.strftime("%Y%m%d"),
        "end_date": day.strftime("%Y%m%d"),
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "tidescout",
    }

    def fetch() -> dict:
        resp = httpx.get(DATAGETTER, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    payload = cache.get_or_fetch("coops_salinity", params, fetch, ttl_s=900)
    if not payload or "error" in payload:
        return None

    values = []
    for row in payload.get("data", []):
        raw = (row.get("s") or "").strip()
        if not raw:
            continue  # dark sensor: CO-OPS sends "", which float() would not reject
        try:
            values.append(float(raw))
        except ValueError:
            continue
    lo, hi = PLAUSIBLE_PPT
    values = [v for v in values if lo <= v <= hi]
    if not values:
        return None
    return sum(values) / len(values)
```

- [ ] **Step 4: Wire it into config**

```python
# backend/tidescout/models.py, in Stations
    # CO-OPS physocean station supplying S_ocean. Springmaid Pier is the only
    # one within 100 km of Winyah Bay (mdapi probe, 2026-08-16).
    ocean_salinity: str = ""
```

```yaml
# fisheries/winyah-bay.yaml, under stations:
  # Springmaid Pier, Myrtle Beach (33.6550, -78.9183) -- ~50 km NE on the open
  # coast, and the ONLY CO-OPS physocean station within 100 km. It measures
  # shelf water, which is the role it plays: the ocean end-member the intrusion
  # profile decays away from, not an in-bay observation.
  # NOT AVAILABLE, probed 2026-08-16 -- do not re-litigate without new access:
  #   - NERR North Inlet-Winyah Bay (Oyster Landing et al): the CDMO web
  #     service returns "Invalid ip" for every method; it requires IP
  #     registration via a form a human must submit.
  #   - Mid-bay USGS salinity: none exists. The only in-bbox 00480 sites are
  #     the two Waccamaw stations above, both ~30 km up-river reading 0-1 ppth.
  ocean_salinity: "8661070"
```

- [ ] **Step 5: Run the tests and commit**

Run: `pytest backend/tests/test_coops_water.py -v && make check`

```bash
git add backend/tidescout/sources/coops_water.py backend/tests/test_coops_water.py \
        backend/tidescout/models.py fisheries/winyah-bay.yaml
git commit -m "feat: CO-OPS ocean salinity end-member"
```

---

## Task 5: Calibration, and stating plainly what it can constrain

Fit `l0_km` and `k` against the observations that exist. **This task's deliverable is as much an honest uncertainty statement as a pair of numbers.**

With an ocean value 50 km offshore and fresh values 30 km up-river and nothing between, the fit has two anchors for two parameters and no interior leverage. It will reproduce its anchors and can be badly wrong in the middle — which is where the fish are. The mitigation is not a cleverer fit; it is to say so, keep the climatology fallback, and record what would resolve it.

**Files:**
- Create: `backend/tidescout/pipeline/salinity_fit.py`
- Modify: `backend/tidescout/cli.py`
- Test: `backend/tests/test_salinity.py`

**Interfaces:**
- Produces: `fit_intrusion(observations, distances_km, cfg) -> tuple[SalinityConfig, dict]` — fitted config plus a diagnostics dict (`rmse_ppt`, `n_obs`, `n_interior_obs`, `cfs_span`, `warning`)

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_salinity.py (append)
from tidescout.pipeline.salinity_fit import fit_intrusion


def _synthetic_obs(cfg, distances, flows):
    """Observations generated BY the model, so a correct fit must recover it."""
    return [
        (d, q, float(salinity.salinity_at(np.array([d]), q, 0.25, cfg)[0]))
        for d in distances
        for q in flows
    ]


def test_fit_recovers_known_parameters_from_dense_observations():
    truth = SalinityConfig(ocean_ppt=34.0, l0_km=16.0, q0_cfs=4000.0, k=0.40,
                           excursion_km=7.0)
    obs = _synthetic_obs(truth, [2.0, 8.0, 15.0, 25.0, 35.0], [2000.0, 6000.0, 15000.0])
    fitted, diag = fit_intrusion(obs, cfg=truth.model_copy(update={"l0_km": 25.0, "k": 0.2}))
    assert fitted.l0_km == pytest.approx(16.0, rel=0.05)
    assert fitted.k == pytest.approx(0.40, rel=0.10)
    assert diag["rmse_ppt"] < 0.1


def test_fit_warns_when_no_observation_sits_in_the_middle_of_the_gradient():
    """Winyah's real situation: an ocean anchor and a river anchor, nothing
    between. The fit will look excellent and constrain nothing where it matters."""
    truth = SalinityConfig(ocean_ppt=34.0, l0_km=16.0, q0_cfs=4000.0, k=0.40,
                           excursion_km=7.0)
    obs = _synthetic_obs(truth, [0.5, 40.0], [3000.0, 9000.0])   # ends only
    _, diag = fit_intrusion(obs, cfg=truth)
    assert diag["n_interior_obs"] == 0
    assert "interior" in diag["warning"].lower()


def test_fit_refuses_to_run_on_a_single_discharge():
    """k is the response to discharge. One flow cannot constrain it, and a fit
    that returns a number anyway is worse than one that declines."""
    truth = SalinityConfig(ocean_ppt=34.0, l0_km=16.0, q0_cfs=4000.0, k=0.40,
                           excursion_km=7.0)
    obs = _synthetic_obs(truth, [2.0, 10.0, 25.0], [4000.0])
    with pytest.raises(ValueError, match="discharge"):
        fit_intrusion(obs, cfg=truth)


def test_fit_records_the_discharge_span_as_the_calibration_range():
    truth = SalinityConfig(ocean_ppt=34.0, l0_km=16.0, q0_cfs=4000.0, k=0.40,
                           excursion_km=7.0)
    obs = _synthetic_obs(truth, [2.0, 10.0, 25.0], [2500.0, 11000.0])
    fitted, _ = fit_intrusion(obs, cfg=truth)
    assert fitted.calibration_range_cfs == (2500.0, 11000.0)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_salinity.py -k fit -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.pipeline.salinity_fit`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/pipeline/salinity_fit.py
"""Fit the intrusion model to observations.

The fit itself is two free parameters against a handful of points, so the
interesting output is not the numbers but the diagnostics: how many
observations, over what discharge span, and -- decisively for Winyah -- whether
any of them sit in the middle of the gradient rather than at its ends.
"""

import numpy as np
from scipy.optimize import least_squares

from tidescout.engine.salinity import salinity_at
from tidescout.models import SalinityConfig

# An observation is "interior" if the model puts it between 10% and 90% of the
# ocean value. Anchors at the ends fix the asymptotes and say nothing about the
# shape between them, which is the part the scoring layer reads.
INTERIOR_BAND = (0.10, 0.90)


def fit_intrusion(
    observations: list[tuple[float, float, float]], cfg: SalinityConfig
) -> tuple[SalinityConfig, dict]:
    """Least-squares fit of (l0_km, k). `observations` is [(distance_km, cfs, ppt)].

    Phase is taken as 0.25 -- mid-cycle, where the tidal excursion term is zero
    -- because the observations available are daily means, which average the
    tidal swing out. Fitting daily means against an instantaneous phase would
    push the excursion's signal into l0_km.
    """
    if len(observations) < 3:
        raise ValueError(
            f"need at least 3 observations to fit 2 parameters, got {len(observations)}"
        )
    flows = {q for _, q, _ in observations}
    if len(flows) < 2:
        raise ValueError(
            "all observations share one discharge -- k is the model's response TO "
            "discharge and cannot be constrained by a single value. Collect "
            "observations across at least two distinct flows."
        )

    d = np.array([o[0] for o in observations], dtype="float64")
    q = np.array([o[1] for o in observations], dtype="float64")
    y = np.array([o[2] for o in observations], dtype="float64")

    def residual(params):
        l0, k = params
        trial = cfg.model_copy(update={"l0_km": max(l0, 0.1), "k": k})
        return np.array(
            [salinity_at(np.array([di]), qi, 0.25, trial)[0] for di, qi in zip(d, q)]
        ) - y

    sol = least_squares(
        residual, x0=[cfg.l0_km, cfg.k], bounds=([0.5, 0.0], [200.0, 2.0])
    )
    fitted = cfg.model_copy(
        update={
            "l0_km": float(sol.x[0]),
            "k": float(sol.x[1]),
            "calibration_range_cfs": (float(q.min()), float(q.max())),
        }
    )

    lo, hi = INTERIOR_BAND
    frac = y / cfg.ocean_ppt
    n_interior = int(((frac > lo) & (frac < hi)).sum())
    rmse = float(np.sqrt(np.mean(residual(sol.x) ** 2)))

    warning = ""
    if n_interior == 0:
        warning = (
            "NO INTERIOR OBSERVATIONS: every point sits at an end of the "
            "gradient (near-ocean or near-fresh), so the fit reproduces its "
            "anchors and constrains the shape between them barely at all -- "
            "which is where the fish are. Treat l0_km and k as order-of-"
            "magnitude, keep the climatology fallback live, and prefer any "
            "mid-bay observation over more end-member data. Registering the "
            "NERR CDMO IP would supply exactly this."
        )
    return fitted, {
        "rmse_ppt": rmse,
        "n_obs": len(observations),
        "n_interior_obs": n_interior,
        "cfs_span": (float(q.min()), float(q.max())),
        "warning": warning,
    }
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_salinity.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Add the CLI and run a real calibration**

Add `tidescout salinity field <slug>` (builds the distance field) and `tidescout salinity calibrate <slug> --days 90`, which pulls USGS `00480` history at the configured water sensors plus CO-OPS ocean salinity, pairs each with its site's along-estuary distance and that day's composite discharge, then calls `fit_intrusion` and prints the fitted config **and the diagnostics block verbatim**.

Run it, then paste the output — fitted `l0_km`, `k`, `rmse_ppt`, `n_interior_obs` and the warning — into `fisheries/winyah-bay.yaml` above the `salinity:` block as a comment, alongside the date. The warning is expected to fire; that is the finding, not a failure.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/salinity_fit.py backend/tidescout/cli.py \
        backend/tests/test_salinity.py fisheries/winyah-bay.yaml
git commit -m "feat: calibrate the intrusion model, and record what it cannot constrain"
```

---

## Task 6: Discharge-axis interpolation in the flow lookup

The second half of Ellis's question. `select_regime` snaps to one of three discharge buckets; this makes the lookup blend across them, turning a step function into a ramp.

Justified by Plan 3's measurement that depth rises **monotonically and near-linearly** with discharge at every inflow in both range buckets tested. **The range axis is deliberately left alone** — it is the strong axis, `RANGE_STEP_COST=3` exists to protect it, and blending it would change answers everywhere and needs its own validation pass.

Above the top bucket the blend **clamps and flags**; it does not extrapolate a shallow-water model past anything it was run at.

**Files:**
- Modify: `backend/tidescout/engine/flow.py`
- Test: `backend/tests/test_flow.py`

**Interfaces:**
- Produces: `blend_regimes(range_bucket: str, cfs: float, buckets: DischargeBuckets, available: set[str]) -> tuple[list[tuple[str, float]], bool]` — [(regime, weight)] summing to 1, plus a clamped flag.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_flow.py (append)
import pytest

from tidescout.engine.flow import blend_regimes
from tidescout.models import DischargeBuckets

BUCKETS = DischargeBuckets(low_below_cfs=2774.0, high_above_cfs=6292.0)
ALL = {f"{r}_{d}" for r in ("neap", "mean", "spring") for d in ("low", "med", "high")}


def test_exact_bucket_flow_returns_a_single_regime():
    mix, clamped = blend_regimes("mean", 2774.0, BUCKETS, ALL)
    assert mix == [("mean_low", 1.0)]
    assert clamped is False


def test_midway_flow_blends_the_two_bracketing_buckets():
    """4533 cfs is the med point; 3653 is halfway from low to med."""
    mix, _ = blend_regimes("mean", 3653.5, BUCKETS, ALL)
    assert {r for r, _ in mix} == {"mean_low", "mean_med"}
    assert dict(mix)["mean_low"] == pytest.approx(0.5, abs=0.01)
    assert sum(w for _, w in mix) == pytest.approx(1.0)


def test_blend_never_crosses_the_range_axis():
    """Range is the strong axis -- one step rescales the whole tidal forcing.
    A blend that traded it for discharge would be the exact mistake
    RANGE_STEP_COST=3 exists to prevent."""
    mix, _ = blend_regimes("spring", 5000.0, BUCKETS, ALL)
    assert all(r.startswith("spring_") for r, _ in mix)


def test_flow_above_the_top_bucket_clamps_and_flags():
    """22,996 cfs was observed; 6,292 is the highest ever simulated. The model
    must not be extrapolated 3.7x past anything it was run at."""
    mix, clamped = blend_regimes("mean", 22996.0, BUCKETS, ALL)
    assert mix == [("mean_high", 1.0)]
    assert clamped is True


def test_flow_below_the_bottom_bucket_clamps_and_flags():
    mix, clamped = blend_regimes("mean", 500.0, BUCKETS, ALL)
    assert mix == [("mean_low", 1.0)]
    assert clamped is True


def test_blend_falls_back_when_a_bracketing_regime_is_missing():
    """A partial library must degrade, per spec section 10, not raise."""
    partial = {"mean_low", "mean_high"}
    mix, _ = blend_regimes("mean", 4533.0, BUCKETS, partial)
    assert {r for r, _ in mix} <= partial
    assert sum(w for _, w in mix) == pytest.approx(1.0)


def test_weights_are_never_negative():
    for cfs in (1000.0, 2774.0, 3500.0, 4533.0, 5500.0, 6292.0, 30000.0):
        mix, _ = blend_regimes("mean", cfs, BUCKETS, ALL)
        assert all(w >= 0.0 for _, w in mix)
        assert sum(w for _, w in mix) == pytest.approx(1.0)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_flow.py -k blend -v`
Expected: FAIL — `ImportError: cannot import name 'blend_regimes'`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/flow.py (append)
def bucket_flows(buckets) -> dict[str, float]:
    """The cfs each simulated discharge bucket actually represents.

    These are the values `forcing.river_inflow_m3s` injects, not the bucket
    EDGES: 'med' is the midpoint of low and high, which is 4,533 cfs and not
    the record's median of 3,866.
    """
    return {
        "low": buckets.low_below_cfs,
        "med": 0.5 * (buckets.low_below_cfs + buckets.high_above_cfs),
        "high": buckets.high_above_cfs,
    }


def blend_regimes(
    range_bucket: str, cfs: float, buckets, available: set[str]
) -> tuple[list[tuple[str, float]], bool]:
    """Weights over regimes bracketing `cfs` on the discharge axis.

    The library holds three discharge values spanning 2,774-6,292 cfs while the
    observed record runs 1,232-22,996, so snapping to the nearest bucket throws
    away most of the axis. Blending recovers it within the simulated span --
    justified by Plan 3's measurement that depth rises monotonically and
    near-linearly with discharge at every inflow.

    The RANGE axis is deliberately not blended. One range step rescales the
    entire tidal forcing (~15 cm of amplitude on a 1.10 m mean range) against a
    discharge step's ~1 cm of depth; RANGE_STEP_COST=3 exists to stop range
    being traded away, and a blend that crossed it would be the same mistake.

    Outside the simulated span this CLAMPS and returns True. Extrapolating a
    shallow-water solution 3.7x past any flow it was run at would be inventing
    data, and the caller needs to know the difference.
    """
    flows = bucket_flows(buckets)
    order = [b for b in DISCHARGE_ORDER if f"{range_bucket}_{b}" in available]
    if not order:
        # No regime at this range at all: fall back to the existing nearest-
        # regime logic, which is allowed to cross the range axis as a last resort.
        name, _ = select_regime(range_bucket, "med", available)
        return [(name, 1.0)], True

    lo_b, hi_b = order[0], order[-1]
    if cfs <= flows[lo_b]:
        return [(f"{range_bucket}_{lo_b}", 1.0)], cfs < flows[lo_b]
    if cfs >= flows[hi_b]:
        return [(f"{range_bucket}_{hi_b}", 1.0)], cfs > flows[hi_b]

    for a, b in zip(order, order[1:], strict=False):
        fa, fb = flows[a], flows[b]
        if fa <= cfs <= fb:
            w = (cfs - fa) / (fb - fa) if fb > fa else 0.0
            mix = [(f"{range_bucket}_{a}", 1.0 - w), (f"{range_bucket}_{b}", w)]
            return [(n, x) for n, x in mix if x > 0.0] or [(f"{range_bucket}_{a}", 1.0)], False
    return [(f"{range_bucket}_{hi_b}", 1.0)], True
```

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_flow.py -v && make check`

```bash
git add backend/tidescout/engine/flow.py backend/tests/test_flow.py
git commit -m "feat: interpolate the discharge axis instead of snapping to it"
```

---

## Task 7: Extend the discharge axis to freshet flows — GATE RESOLVED, EXTEND

**The gate is closed and the answer is yes. This is the only task in Plan 4 that spends ANUGA compute.**

Two probes ran 2026-08-16, both a `mean_high` regime at **22,996 cfs** (the observed maximum of the 365-day composite record) against production `mean_high` at 6,292 cfs — a 3.65× extrapolation of the shipped axis. One used the shipped equal-thirds inflow split, one the corrected 78/13/8, so the magnitude question and the split question separate cleanly.

**The decision rule was fixed before the results were known:** extend the axis if the velocity field departs from `mean_high` by materially more than the southern-approach variant's 0.77% p99 noise floor; otherwise leave discharge a depth-only axis.

**Measured, over 26 phases and 5.48M wet-centroid samples:**

| Probe | pooled p99 Δ | Mud Bay | Georgetown | North Jetty |
|---|---|---|---|---|
| `flow-variant-freshet` (equal thirds) | **17.20%** | 0.02162 | 0.05376 | 0.02091 |
| `flow-variant-freshet-split` (78/13/8) | **17.62%** | 0.02162 | 0.05378 | 0.02095 |
| *southern-approach noise floor* | *0.77%* | *0.00018* | *0.00059* | *0.00026* |

(spot columns are mean |Δspeed| in m/s within 150 m)

**17.20% is 22× the noise floor, and Georgetown moves by 0.054 m/s against flows of ~1 m/s.** That is not noise, and it overturns the working assumption inherited from Plan 3.

**Why Plan 3 concluded the opposite, and why both conclusions are correct.** Plan 3 measured the discharge axis as weak — domain-mean depth +3.2 mm, "the discharge axis barely moves speed" — and built `RANGE_STEP_COST=3` vs `DISCHARGE_STEP_COST=1` on that finding. But it measured across 2,774 → 6,292 cfs, a **2.3× span entirely inside the p25–p75 band**. Across 6,292 → 22,996 cfs, a further 3.65×, the response is large. **The axis was never weak; it was too short to show its own effect.** The regime-fallback weighting stays correct for the range it governs — this adds range beyond it rather than contradicting it.

The two freshet probes differ from each other by 0.4 percentage points and by 0.00002 m/s at Georgetown, so **at freshet flow the total volume dominates and the inflow split is second-order**. The split fix is still correct (Phase 1 Task 2) and still does not require a rebuild on its own.

**Consistency ruling.** The three new regimes must be run with the **same config as the shipped nine** — old boundary vertex, and whichever inflow split the existing library used — OR all twelve must be rebuilt together. A library whose `freshet` regimes carry corrections the other nine lack would make `blend_regimes` interpolate across a config discontinuity, which is a worse error than either correction fixes. Both corrections measure negligible at the spots (0.00003 and 0.00059 m/s), so **either choice is defensible and the decision is a compute-budget question, not a correctness one.** Present both to Ellis before spending the hours:
- **Three regimes, old config** — ~3.2 h, library internally consistent, retains two measured-negligible errors.
- **All twelve, corrected config** — ~5.5–7 h at 9 workers (the shipped nine took 4.13 h), clean library, both corrections landed.

- [ ] **Step 1: Promote the comparison harness into the repo**

Copy the variant-comparison script to `backend/tools/compare_variant.py` and commit it. Carryover lesson 5: the diagnostic harness has already been lost once and silently drifted once.

```bash
git add backend/tools/compare_variant.py
git commit -m "test: promote the variant-comparison harness into the repo"
```

- [ ] **Step 2: Record the result in the carryover notes**

Add a "Phase 2 results" section to `docs/superpowers/plans/2026-08-13-plan3-carryover-notes.md` with the table above, and correct the standing claim that "the discharge axis barely moves speed" to name the span over which it was measured.

- [ ] **Step 3: Confirm the rebuild scope with Ellis, then add the bucket**

Once he picks a scope: add `freshet` to `DISCHARGE_ORDER` in `engine/flow.py` and to `forcing.river_inflow_m3s`'s composite map at the observed max (22,996 cfs); extend `bucket_flows` and `blend_regimes` to cover it; run the new regimes with `caffeinate -dimsu -t` sized to the job plus margin.

- [ ] **Step 4: Re-validate**

Re-run `tidescout flow validate winyah-bay` on the new regimes and confirm North Jetty and Mud Bay Cut read the same as they do in the shipped nine. Then re-run `blend_regimes`' tests — the clamp boundary has moved from 6,292 to 22,996 cfs, so `test_flow_above_the_top_bucket_clamps_and_flags` needs its input raised above the new top.

---

## Phase 2 Completion Checklist

- [ ] `make check` green; test count ≥ 250
- [ ] Along-estuary distance field built; North Jetty ≈ 0 km, river inlets tens of km, no NaN at an inlet
- [ ] Intrusion model passes all shape tests, including the low-water/high-water sign
- [ ] Ocean end-member fetched from Springmaid Pier with recorded fixtures; NERR and mid-bay unavailability documented in the fishery YAML
- [ ] Calibration run, with `l0_km`, `k`, `rmse_ppt`, `n_interior_obs` and the interior warning pasted into the config
- [ ] `blend_regimes` interpolates within the simulated span and clamps outside it
- [ ] Freshet probe compared, decision recorded in the carryover notes with numbers
- [ ] `compare_variant.py` committed to `backend/tools/`

**Then:** Phase 3 (`2026-08-16-04-phase3-bite-score.md`) consumes `SalinityField`, `FeatureMetrics`, `DischargeSummary.limb` and `blend_regimes`.
