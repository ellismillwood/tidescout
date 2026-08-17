# Plan 4 Phase 1 — Derived Flow Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Plan 3 flow library from raw `u`/`v`/`depth` into the fish-relevant structure the spec asks for — seams, eddy cores, ambush pockets, convergence zones and flats flood/drain schedules — and attach it to the static feature inventory under stable identifiers.

**Architecture:** Every signal in this plan falls out of **one velocity-gradient tensor**. `engine/flow.py` gains `gradient_tensor`, from which strain rate, vorticity, the Okubo–Weiss discriminant and divergence are all cheap algebra; `ambush_contrast` adds a single morphological max-filter. All of it is **computed at runtime, not stored** — the algebra costs milliseconds, precomputing would cost ~2 GB and a 4.6 h library rebuild, and computing on the phase-interpolated velocity is strictly more correct than interpolating precomputed nonlinear quantities. `engine/activation.py` samples those fields at feature geometry. Nothing in this plan runs ANUGA.

**Tech Stack:** Python 3.12, numpy 2.5.2, scipy 1.18.0 (`ndimage.maximum_filter`), shapely 2.1.2, rasterio 1.5.1, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` — §5 ("Derived fish-relevant structure") and §6 (feature inventory) are what this plan implements; §8 and §11 constrain it.

**Carryover context (read before Task 1):** `docs/superpowers/plans/2026-08-13-plan3-carryover-notes.md`. Sections "Conventions Plan 4 must not get wrong" and "Owed / open items" are load-bearing here.

---

## Global Constraints

- **Python** `>=3.12`. Backend package is `backend/tidescout`, editable-installed into `~/.venvs/tidescout`.
- **Lint gate:** `[tool.ruff.lint] select = ["E","F","I","UP","B","DTZ"]`, `line-length = 100`, `target-version = "py312"`. `collections.abc` imports, timezone-aware datetimes only, no closures capturing loop variables.
- **Gate command:** `make check` = `ruff check` + `pytest -q`, run from repo root. Green before every commit.
- **Baseline at plan start:** **193 tests passing** on `plan-03-anuga` (verified 2026-08-16).
- **`engine/` is pure.** Plain data in, values out. No filesystem, no network. This plan *moves code toward* that rule — see Task 3.
- **Paths** resolve via `tidescout.paths`. Never hand-roll a parent-walk.
- **`data/` is gitignored and rebuildable; `fisheries/` is committed.**
- **Tests never hit live APIs.** `respx` for HTTP, synthetic arrays for rasters (`backend/tests/synth.py`).
- **The library grid is fixed:** shape `(2527, 1903)`, 20 m cells, EPSG:26917, transform `(20.0, 0.0, 643813.42, 0.0, -20.0, 3719510.45)`, **587,325 in-domain cells** of 4,808,881 (12.2%). Stored arrays are **1-D and masked**; `flat_index` maps them back. Anything computing a gradient must scatter to 2-D first.
- **Phase 0 is LOW water.** `spin_up_h / cycle_h` = 6.0 / 12.42 = 0.4831 of a cycle. **Flood is the first half** (phase 0→0.5), ebb the second. Never re-derive this — call `engine.flow.tide_states`, which reads it off the recorded `stage_bc_m`.
- **`np.gradient` returns the ROW derivative first**, and rows run south on a north-up raster while `u`/`v` are true east/north. The row derivative is `-d/dy`. Getting this wrong silently converts stretching into divergence.
- **Commit style:** `feat:` / `fix:` / `test:` / `docs:` prefixes, one commit per task minimum.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `backend/tidescout/engine/structure.py` | Pure derived-structure fields: gradient tensor, strain, vorticity, Okubo–Weiss, divergence, ambush contrast |
| `backend/tidescout/engine/activation.py` | Sample structure fields at feature geometry → per-feature metrics |
| `backend/tidescout/pipeline/schedule.py` | Flats flood/drain schedule from the wet mask across a regime's phases |
| `backend/tests/test_structure.py` | Analytic-flow-field tests for every derived signal |
| `backend/tests/test_activation.py` | Feature sampling on synthetic fields |
| `backend/tests/test_schedule.py` | Flood/drain schedule extraction |

**Modified files:**

| Path | Change |
|---|---|
| `fisheries/winyah-bay.yaml` | `ocean_boundary_utm_km` SW vertex; per-river `inflow_share`; `store_sww` |
| `backend/tidescout/models.py` | `RiverGauge.inflow_share`; `AnugaConfig.store_sww`; `StructureThresholds` |
| `backend/tidescout/pipeline/forcing.py` | `river_inflow_m3s` splits by `inflow_share`, not `weight` |
| `backend/tidescout/pipeline/regimes.py` | Honour `store_sww` |
| `backend/tidescout/pipeline/flowlib.py` | `shear_magnitude` moves out to `engine/structure.py`; add `to_grid`/`from_grid` |
| `backend/tidescout/engine/flow.py` | Re-export structure helpers for callers; grid round-trip |
| `backend/tidescout/engine/detect.py` | `feature_key` — stable hash-of-type-plus-centroid |
| `backend/tidescout/pipeline/features.py` | Emit stable ids |
| `backend/tidescout/cli.py` | `tidescout flow structure` |
| `fisheries/winyah-bay.known-spots.yaml` | Georgetown `works_on` |
| `backend/tests/test_flowlib.py` | Shear tests move to `test_structure.py` |

---

## Task 1: Land Plan 3 and the measured corrections

Plan 3's branch is 41 commits ahead of `main` and unmerged. Three carryover items are decided and each is a one-line change, so they land together with the merge rather than as drive-by edits later.

**Decisions being recorded, with their evidence:**

1. **`ocean_boundary_utm_km` SW vertex `(663.0, 3671.5)` → `(662.0, 3672.0)`.** Carryover open item 3. Promotes ring segment 214 — 3,608 m of genuinely open back-barrier water on the southern approach — from `open` (Reflective, a wall) to `ocean`. Verified with production `classify_boundary`: ocean goes 27.59 → 31.20 km, the other four `open` segments (Pee Dee head, Sampit head, both ICW crossings) are untouched, and all 595 `wall` segments are unchanged. A full `mean_med` variant run confirmed it is **stable** (mass residual 1.46e-14, 26/26 snapshots, 2.61 h) and that its effect is **negligible**: p99 speed delta 0.77% of p99 speed bay-wide, and 0.00018–0.00059 m/s at the three known spots against flows of 0.5–1 m/s.
   **Therefore: adopt the config change so any future rebuild inherits it, but do NOT rebuild for it alone.**
2. **`AnugaConfig.store_sww`**, default `True`. Carryover open item 1. ANUGA currently writes 1.5 GB of `.sww` across a build because `run_regime` never calls `set_quantities_to_be_stored(None)`. Keep it for future flow visualisation, but make it a knob.
3. **Georgetown Lighthouse `works_on: flood` → `slack`.** Carryover's open validation gate. Ellis, asked directly (2026-08-16): *"I do not know what tide this spot works best on... I said flood because it hides from the main channel current in that situation but it could also work on an ebb."* A spot defined by **hiding from** the main current is a contrast mechanism, not a peak-current one, and the CLI's `slack` branch already judges on contrast. Supporting evidence: Winyah is river-dominated, so ebb carrying peak current at the mouth is expected rather than anomalous; the model shows the largest contrast of any spot there (a 0.000 m/s pocket beside ~1 m/s water); and the southern-approach variant above independently rules out the boundary as the cause of the ebb lean.

**Files:**
- Modify: `fisheries/winyah-bay.yaml` (`model_domain.ocean_boundary_utm_km`, `anuga`)
- Modify: `fisheries/winyah-bay.known-spots.yaml:26-39`
- Modify: `backend/tidescout/models.py` (`AnugaConfig`)
- Modify: `backend/tidescout/pipeline/regimes.py:106-111`
- Test: `backend/tests/test_regimes.py`

**Interfaces:**
- Produces: `AnugaConfig.store_sww: bool` — Task 2 and Phase 2 both read the same config object.

- [ ] **Step 1: Merge Plan 3 to main**

```bash
cd ~/Documents/tidescout
make check                      # expect 193 passed
git checkout main
git merge --no-ff plan-03-anuga -m "Merge Plan 3: ANUGA flow-state library

Nine regimes x 26 phases, rasterised to a 20 m grid masked to the model
domain. Mass residuals 2.0e-15 - 1.2e-14, all nine reverse."
make check                      # expect 193 passed
git checkout -b plan-04-phase1-structure
```

- [ ] **Step 2: Write the failing test for the `.sww` knob**

```python
# backend/tests/test_regimes.py
def test_store_sww_knob_defaults_on_and_can_be_disabled():
    from tidescout.models import AnugaConfig

    assert AnugaConfig().store_sww is True
    assert AnugaConfig(store_sww=False).store_sww is False
```

- [ ] **Step 3: Run it and watch it fail**

Run: `pytest backend/tests/test_regimes.py::test_store_sww_knob_defaults_on_and_can_be_disabled -v`
Expected: FAIL — `ValidationError` / `AttributeError`, no such field.

- [ ] **Step 4: Add the field**

```python
# backend/tidescout/models.py, in AnugaConfig
    # ANUGA writes a full .sww per regime (~170 MB each, 1.5 GB per build)
    # alongside our snap_*.npz. The pipeline consumes only the npz files, so
    # this is pure surplus -- but it is the only full-resolution record of the
    # run, and the frontend will likely want it for flow visualisation that a
    # 20 m masked grid cannot reconstruct. Kept ON, made switchable.
    store_sww: bool = True
```

- [ ] **Step 5: Honour it in `run_regime`**

```python
# backend/tidescout/pipeline/regimes.py, after domain.set_datadir(str(out_dir))
    if not cfg.store_sww:
        domain.set_quantities_to_be_stored(None)
```

- [ ] **Step 6: Run the tests**

Run: `pytest backend/tests/test_regimes.py -v`
Expected: PASS.

- [ ] **Step 7: Apply the boundary vertex fix**

In `fisheries/winyah-bay.yaml`, change the last entry of `ocean_boundary_utm_km` from `[663.0, 3671.5]` to `[662.0, 3672.0]` and append this comment above the list:

```yaml
  # CORRECTED 2026-08-16 (carryover open item 3). The SW corner was
  # (663.0, 3671.5), which left ring segment 214 -- 3,608 m of genuinely open
  # back-barrier water on the southern approach -- outside the ocean polygon
  # and therefore tagged `open`, which maps to Reflective_boundary: a wall
  # across open water. Nudging the corner to (662.0, 3672.0) promotes exactly
  # that one segment (ocean 27.59 -> 31.20 km) and leaves the four river-head
  # and ICW `open` segments and all 595 `wall` segments untouched -- verified
  # against production classify_boundary before the change.
  #
  # MEASURED, full mean_med variant run 2026-08-16: STABLE (mass residual
  # 1.46e-14, 26/26 snapshots, 2.61 h wall) and NEGLIGIBLE -- pooled p99 speed
  # delta is 0.77% of p99 speed over 5.47M wet-centroid samples, and within
  # 150 m of the known spots the mean delta is 0.00018 (Mud Bay), 0.00059
  # (Georgetown) and 0.00026 m/s (North Jetty) against flows of 0.5-1 m/s.
  # The shipped library was therefore NOT rebuilt for this. It is correct for
  # the next rebuild to inherit, and it independently clears the boundary as
  # an explanation for Georgetown's ebb lean.
```

- [ ] **Step 8: Record the Georgetown decision**

Replace the comment block and `works_on` at `fisheries/winyah-bay.known-spots.yaml:30-35`:

```yaml
    # RESOLVED 2026-08-16. Was `flood`; Ellis, asked directly, said he does not
    # know which half this spot works on -- he chose flood because the spot
    # "hides from the main channel current in that situation, but it could also
    # work on an ebb". Hiding from the current is a CONTRAST mechanism, not a
    # peak-current one, so flood-vs-ebb was the wrong question to put to the
    # gate. `slack` selects the CLI's contrast branch, which is what the notes
    # actually describe.
    # Supporting evidence: the model puts peak current on the ebb here in all
    # nine regimes (~10% margin), which is the EXPECTED result for a
    # river-dominated estuary mouth -- North Jetty is flood-dominant because
    # jetty structure focuses the incoming push, a local effect that does not
    # generalise to the north bank. The model also shows by far the largest
    # contrast of any spot here: a 0.000 m/s pocket beside ~1 m/s water, which
    # IS "the mouth of the bay creates a break in current". And the
    # southern-approach boundary variant (2026-08-16) perturbs this spot by
    # only 0.00059 m/s, ruling out the reflective wall as the cause.
    works_on: slack
```

- [ ] **Step 9: Verify the gate now reads the spot the way the notes describe**

Run: `tidescout flow validate winyah-bay --regime mean_med`
Expected: Georgetown's `agrees` column shows `yes` via the contrast branch. **North Jetty and Mud Bay Cut must be unchanged** — if either moves, the edit touched more than intended.

- [ ] **Step 10: Commit**

```bash
git add fisheries/winyah-bay.yaml fisheries/winyah-bay.known-spots.yaml \
        backend/tidescout/models.py backend/tidescout/pipeline/regimes.py \
        backend/tests/test_regimes.py
git commit -m "fix: adopt measured boundary correction, .sww knob, Georgetown gate

Three carryover decisions, each backed by a measurement rather than an
argument. See the config comments for the evidence."
```

---

## Task 2: Correct the river inflow split

**This is a defect found on 2026-08-16, not a planned enhancement.** `RiverGauge.weight` is overloaded with two incompatible meanings:

- `usgs.discharge_summary` uses it to build the **composite total**: `total_now += last_v * w`. With `weight: 1.0` on all three that is a plain sum, which is correct — the total freshwater input is the sum of the gauges.
- `forcing.river_inflow_m3s` uses the same field to **split that composite across inlets**: `composite_cfs * (r.weight / total_weight)`. With `weight: 1.0` on all three that is **equal thirds**, which is wrong.

The fishery config records the true proportions in its own `discharge_buckets` comment: Pee Dee 3,745 / Waccamaw 642 / Black 397 cfs over 365 days — **78.3 / 13.4 / 8.3 %**. So at `mean_med` (4,533 cfs = 128.4 m³/s) the model injects 42.8 m³/s into each river when it should inject 100.5 / 17.2 / 10.7. **The Black gets 4× its share and the Pee Dee 42% of its own.**

One field cannot serve both uses: for the composite you want 1.0 each (a sum); for the split you want fractions that total 1.0. They need separating.

Plan 3's discharge-axis check corroborates the bug without noticing it — it found the Pee Dee "mid-pack, not weak" among per-river depth responses (+10.9 mm vs Waccamaw +12.5, Black +9.8) and read that as reassuring. Under a correct 78/13/8 split the Pee Dee being mid-pack is the anomaly; under equal thirds it is exactly what you would predict.

**Impact.** Small for Plan 3 — discharge is a weak axis for velocity, moving domain-mean depth ~1 cm. **First-order for Phase 2**, and dangerous in a specific way: both salinity observation stations sit on the Waccamaw, the branch receiving 2.5× too much fresh water, so fitting intrusion parameters against them would absorb the error and then apply it bay-wide. This is the "plausible wrong answer rather than a crash" failure mode from carryover process lesson 6.

**This does not force a rebuild — MEASURED, not inferred.** A full `mean_med` variant run with the corrected 78/13/8 split (2026-08-16, `data/winyah-bay/flow-variant-split/`, 3.16 h, mass residual 2.56e-15) was differenced against production `mean_med` over all 26 phases and 5.48M wet-centroid samples:

- pooled p99 speed delta **1.07%** of p99 speed; max 0.032 m/s, concentrated near the river inlets, which is exactly where moving water between inlets should show;
- within 150 m of the known spots, mean delta **0.00001 m/s** (Mud Bay), **0.00003** (Georgetown), **0.00001** (North Jetty) — three to four orders of magnitude below the flows there.

So the correction is real and worth making, and the shipped library does not need rebuilding for it. Phase 2's salinity model is analytic and takes per-river discharge directly, so it is correct the moment this lands.

**Files:**
- Modify: `backend/tidescout/models.py` (`RiverGauge`)
- Modify: `backend/tidescout/pipeline/forcing.py:74-91`
- Modify: `fisheries/winyah-bay.yaml` (three `inflow_share` entries)
- Test: `backend/tests/test_forcing.py`

**Interfaces:**
- Consumes: `Fishery.rivers`, `Fishery.discharge_buckets`
- Produces: `RiverGauge.inflow_share: float | None` — Phase 2's `salinity.branch_discharge_cfs` reads it. `river_inflow_m3s(fishery, bucket) -> dict[str, float]` keeps its signature.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_forcing.py
import pytest

from tidescout.config import load_fishery
from tidescout.pipeline import forcing


def test_inflow_split_follows_inflow_share_not_gauge_weight():
    """A river's share of the composite is its own long-term flow fraction.

    `weight` means "how this gauge contributes to the composite total" (1.0 =
    include it in the sum). `inflow_share` means "what fraction of that total
    enters here". Conflating them injected equal thirds into three rivers whose
    real split is 78/13/8.
    """
    f = load_fishery("winyah-bay")
    inflows = forcing.river_inflow_m3s(f, "med")
    total = sum(inflows.values())

    assert inflows["Pee Dee"] / total == pytest.approx(0.783, abs=0.01)
    assert inflows["Waccamaw"] / total == pytest.approx(0.134, abs=0.01)
    assert inflows["Black"] / total == pytest.approx(0.083, abs=0.01)
    # The Pee Dee must dominate; equal thirds is the bug this test pins.
    assert inflows["Pee Dee"] > 4 * inflows["Black"]


def test_inflow_total_still_matches_the_composite_bucket():
    """Redistributing shares must not change how much water enters overall."""
    f = load_fishery("winyah-bay")
    for bucket, cfs in (
        ("low", f.discharge_buckets.low_below_cfs),
        ("high", f.discharge_buckets.high_above_cfs),
    ):
        total = sum(forcing.river_inflow_m3s(f, bucket).values())
        assert total == pytest.approx(cfs * forcing.CFS_TO_M3S, rel=1e-9)


def test_inflow_shares_are_rejected_when_they_do_not_sum_to_one():
    """A silent renormalisation would hide an authoring mistake."""
    f = load_fishery("winyah-bay")
    f.rivers[0].inflow_share = 0.5  # now sums to ~0.72
    with pytest.raises(ValueError, match="inflow_share"):
        forcing.river_inflow_m3s(f, "med")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_forcing.py -k inflow -v`
Expected: FAIL — the first asserts 0.333 ≈ 0.783.

- [ ] **Step 3: Add the field**

```python
# backend/tidescout/models.py, in RiverGauge
    # Fraction of the composite discharge that enters the domain at THIS
    # river's inlet. Distinct from `weight`, which says how this gauge
    # contributes to the composite TOTAL (1.0 = include it in the sum). The
    # two were conflated until 2026-08-16, which split the composite into
    # equal thirds across three rivers whose measured split is 78/13/8.
    # None on every river reproduces the old equal-share behaviour, so a
    # fishery that has not authored shares still runs.
    inflow_share: float | None = None
```

- [ ] **Step 4: Split by share**

```python
# backend/tidescout/pipeline/forcing.py, replacing the body of river_inflow_m3s
def river_inflow_m3s(fishery: Fishery, bucket: str) -> dict[str, float]:
    """Steady inflow per river for a discharge regime, in m^3/s.

    Inflow is steady across a run by design: the tidal cycle is 12.42 h and
    river discharge changes over days, so a time-varying inlet would model a
    process the simulation window cannot resolve.

    The composite bucket boundaries are the calibrated percentiles from Plan 3
    Task 1. Each river takes its `inflow_share` of that composite -- NOT its
    gauge `weight`, which serves the different job of building the composite.
    """
    b = fishery.discharge_buckets
    composite_cfs = {
        "low": b.low_below_cfs,
        "med": 0.5 * (b.low_below_cfs + b.high_above_cfs),
        "high": b.high_above_cfs,
    }[bucket]

    shares = [r.inflow_share for r in fishery.rivers]
    if all(s is None for s in shares):
        # Unauthored fishery: fall back to equal shares rather than failing.
        n = len(fishery.rivers) or 1
        shares = [1.0 / n] * len(fishery.rivers)
    elif any(s is None for s in shares):
        missing = [r.name for r in fishery.rivers if r.inflow_share is None]
        raise ValueError(
            f"inflow_share is set on some rivers but missing on {missing} -- "
            "author it on all of them or none, so the split is never half-guessed"
        )
    total_share = sum(shares)
    if abs(total_share - 1.0) > 1e-6:
        raise ValueError(
            f"inflow_share values sum to {total_share:.4f}, not 1.0 -- "
            "renormalising silently would hide an authoring mistake"
        )
    return {
        r.name: composite_cfs * CFS_TO_M3S * s
        for r, s in zip(fishery.rivers, shares, strict=True)
    }
```

- [ ] **Step 5: Author the shares**

Add `inflow_share` beneath each river's `weight` in `fisheries/winyah-bay.yaml`, with this comment above the Pee Dee entry:

```yaml
    # inflow_share added 2026-08-16. Derived from the same 365-day per-gauge
    # daily means that calibrated discharge_buckets: Pee Dee 3745, Waccamaw
    # 642, Black 397 cfs, total 4784 -> 0.783 / 0.134 / 0.083. Before this,
    # `weight: 1.0` on all three made river_inflow_m3s split the composite
    # into equal thirds, injecting 4x the Black's share and 42% of the Pee
    # Dee's. `weight` is deliberately left at 1.0: it means something else
    # (include this gauge in the composite sum) and is correct as it stands.
    inflow_share: 0.783   # Pee Dee
    inflow_share: 0.134   # Waccamaw
    inflow_share: 0.083   # Black
```

(The three lines go on their respective rivers, not together.)

- [ ] **Step 6: Run the tests**

Run: `pytest backend/tests/test_forcing.py -v && make check`
Expected: PASS, 196+ tests.

- [ ] **Step 7: Commit**

```bash
git add backend/tidescout/models.py backend/tidescout/pipeline/forcing.py \
        backend/tests/test_forcing.py fisheries/winyah-bay.yaml
git commit -m "fix: split river inflow by measured share, not gauge weight

RiverGauge.weight served two incompatible jobs; splitting the composite by
it injected equal thirds into rivers whose real split is 78/13/8. First-order
for Phase 2 salinity, millimetre-scale for the existing flow library."
```

---

## Task 3: The velocity-gradient tensor and the grid round-trip

Everything downstream is algebra on four numbers per cell. This task establishes them once, plus the scatter/gather that lets a 1-D masked library array become a 2-D field a gradient can be taken of.

`shear_magnitude` currently lives in `pipeline/flowlib.py`. It is a pure function with no I/O sitting in the I/O layer, in violation of the project's separation rule. It moves here along with its four tests.

**Files:**
- Create: `backend/tidescout/engine/structure.py`
- Create: `backend/tests/test_structure.py`
- Modify: `backend/tidescout/pipeline/flowlib.py` (remove `shear_magnitude` and `_xy_gradients`)
- Modify: `backend/tests/test_flowlib.py` (remove the four shear tests)

**Interfaces:**
- Produces:
  - `to_grid(values: np.ndarray, flat_index: np.ndarray, shape: tuple[int, int], fill: float = np.nan) -> np.ndarray`
  - `from_grid(grid: np.ndarray, flat_index: np.ndarray) -> np.ndarray`
  - `xy_gradients(a: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]`
  - `GradientTensor` dataclass with `du_dx, du_dy, dv_dx, dv_dy`
  - `gradient_tensor(u: np.ndarray, v: np.ndarray, cell_m: float) -> GradientTensor`
  - `strain_rate(t: GradientTensor) -> np.ndarray`
  Tasks 4–6 and 9 all consume `GradientTensor`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_structure.py
"""Analytic flow fields with known answers.

Every test here builds a velocity field whose derived structure can be worked
out on paper, so a failure localises to the formula rather than to the data.
"""

import numpy as np
import pytest

from tidescout.engine import structure


def _xy(n=64, cell=20.0):
    """Centred coordinate grids in metres, north-up (row 0 is the NORTH edge)."""
    c = (np.arange(n) - n / 2) * cell
    x = np.tile(c, (n, 1))
    y = np.tile(-c[:, None], (1, n))  # rows run south, so y decreases downward
    return x, y, cell


def test_grid_round_trip_restores_values_at_their_cells():
    shape = (5, 4)
    flat_index = np.array([0, 6, 19])
    values = np.array([1.5, -2.0, 7.25])
    grid = structure.to_grid(values, flat_index, shape)

    assert grid.shape == shape
    assert grid.ravel()[0] == 1.5
    assert grid.ravel()[6] == -2.0
    assert grid.ravel()[19] == 7.25
    assert np.isnan(grid.ravel()[1])  # out-of-domain stays NaN, never 0.0
    assert np.array_equal(structure.from_grid(grid, flat_index), values)


def test_zero_fill_is_not_the_default_because_zero_is_a_real_speed():
    """0.0 m/s is slack water; NaN is 'not in the domain'. Conflating them
    would make every land cell look like a stagnant ambush pocket."""
    grid = structure.to_grid(np.array([1.0]), np.array([3]), (2, 2))
    assert np.isnan(grid[0, 0])


def test_xy_gradients_return_true_east_and_north_derivatives():
    """np.gradient gives the ROW derivative first, and rows run SOUTH."""
    x, y, cell = _xy()
    d_dx, d_dy = structure.xy_gradients(3.0 * x, cell)
    assert np.allclose(d_dx[1:-1, 1:-1], 3.0)
    assert np.allclose(d_dy[1:-1, 1:-1], 0.0, atol=1e-9)

    d_dx, d_dy = structure.xy_gradients(3.0 * y, cell)
    assert np.allclose(d_dx[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(d_dy[1:-1, 1:-1], 3.0)  # NOT -3.0


def test_gradient_tensor_recovers_a_known_pure_shear():
    x, y, cell = _xy()
    a = 0.004
    t = structure.gradient_tensor(a * y, np.zeros_like(y), cell)
    assert np.allclose(t.du_dy[1:-1, 1:-1], a)
    assert np.allclose(t.du_dx[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(t.dv_dx[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(t.dv_dy[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(structure.strain_rate(t)[1:-1, 1:-1], a)


def test_strain_rate_is_zero_in_solid_body_rotation():
    """A rigidly turning eddy deforms no parcel, so it is not a seam --
    neighbouring water never slides past itself. Task 4 is what finds it."""
    x, y, cell = _xy()
    omega = 0.002
    t = structure.gradient_tensor(-omega * y, omega * x, cell)
    assert np.nanmax(np.abs(structure.strain_rate(t)[1:-1, 1:-1])) < 1e-9


def test_strain_rate_ignores_isotropic_expansion():
    x, y, cell = _xy()
    k = 0.003
    t = structure.gradient_tensor(k * x, k * y, cell)
    assert np.nanmax(np.abs(structure.strain_rate(t)[1:-1, 1:-1])) < 1e-9


def test_strain_rate_is_galilean_invariant():
    """A seam reads the same whether the whole bay is drifting past it."""
    x, y, cell = _xy()
    a = 0.004
    still = structure.strain_rate(structure.gradient_tensor(a * y, np.zeros_like(y), cell))
    drift = structure.strain_rate(
        structure.gradient_tensor(a * y + 0.7, np.full_like(y, -0.3), cell)
    )
    assert np.allclose(still[1:-1, 1:-1], drift[1:-1, 1:-1])
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_structure.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.engine.structure`.

- [ ] **Step 3: Write the module**

```python
# backend/tidescout/engine/structure.py
"""Derived flow structure. Pure -- callers hand in arrays, no I/O.

Every signal in this module comes from one velocity-gradient tensor, so the
four derivatives are computed once and everything else is algebra on them.

The flow library stores 1-D arrays masked to the model domain (587,325 of
4,808,881 cells). A gradient needs neighbours, so anything here that
differentiates works on 2-D grids; `to_grid` and `from_grid` are the bridge.
"""

from dataclasses import dataclass

import numpy as np


def to_grid(
    values: np.ndarray,
    flat_index: np.ndarray,
    shape: tuple[int, int],
    fill: float = np.nan,
) -> np.ndarray:
    """Scatter a masked 1-D library array back onto the full raster.

    `fill` is NaN and not 0.0 on purpose. Zero is a real, common value here --
    it is slack water -- so filling the out-of-domain cells with it would make
    every land cell indistinguishable from a stagnant pocket, which is exactly
    what `ambush_contrast` hunts for. NaN propagates through the gradients and
    is masked off at the end instead.
    """
    grid = np.full(int(shape[0]) * int(shape[1]), fill, dtype="float64")
    grid[flat_index] = values
    return grid.reshape(shape)


def from_grid(grid: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
    """Gather the in-domain cells back out of a full raster."""
    return grid.reshape(-1)[flat_index]


def xy_gradients(a: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """d/dx (true east) and d/dy (true north) of a grid-shaped field.

    `np.gradient` returns the ROW derivative first, and on a north-up raster
    rows run SOUTH while u/v are true east/north. So the row derivative is
    -d/dy. This is not cosmetic: it turns the stretching term du_dx - dv_dy
    into the divergence du_dx + dv_dy, a completely different field.
    """
    d_drow, d_dcol = np.gradient(a, cell_m)
    return d_dcol, -d_drow


@dataclass(frozen=True)
class GradientTensor:
    """The four spatial derivatives of the depth-averaged velocity field."""

    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray


def gradient_tensor(u: np.ndarray, v: np.ndarray, cell_m: float) -> GradientTensor:
    du_dx, du_dy = xy_gradients(u, cell_m)
    dv_dx, dv_dy = xy_gradients(v, cell_m)
    return GradientTensor(du_dx, du_dy, dv_dx, dv_dy)


def strain_rate(t: GradientTensor) -> np.ndarray:
    """Total deformation rate -- the spec's 'seams' signal.

    sqrt((du_dx - dv_dy)^2 + (du_dy + dv_dx)^2), the stretching/shearing pair.

    Deliberately NOT sqrt(du_dy^2 + dv_dx^2 + ...): that returns 1.41*omega for
    solid-body rotation, which contains no seam anywhere -- parcels in a rigidly
    turning eddy never slide past one another -- and would light up the interior
    of every eddy in the bay as holding water. It is also Galilean invariant, so
    a seam reads the same whether the whole body of water is drifting past it,
    and isotropic expansion correctly returns zero.
    """
    stretching = t.du_dx - t.dv_dy
    shearing = t.du_dy + t.dv_dx
    return np.sqrt(stretching**2 + shearing**2)
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_structure.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Remove the duplicates from `flowlib`**

Delete `_xy_gradients` and `shear_magnitude` from `backend/tidescout/pipeline/flowlib.py` (lines 93–128) and the four shear tests from `backend/tests/test_flowlib.py` (they are re-covered above). Leave `flowlib.speed_direction` alone — `engine/flow.py` already has its own copy and both are in use.

- [ ] **Step 6: Run the full gate**

Run: `make check`
Expected: PASS. Test count moves by the four relocated shear tests plus seven new ones.

- [ ] **Step 7: Commit**

```bash
git add backend/tidescout/engine/structure.py backend/tests/test_structure.py \
        backend/tidescout/pipeline/flowlib.py backend/tests/test_flowlib.py
git commit -m "feat: velocity-gradient tensor and grid round-trip

Moves shear_magnitude out of the I/O layer into engine/, where the purity rule
puts it, and generalises it to a tensor the rest of Phase 1 builds on."
```

---

## Task 4: Okubo–Weiss — telling an eddy from a seam

Strain rate alone cannot find an eddy: Task 3 proves it returns zero for solid-body rotation. The missing half is **vorticity**, and the standard oceanographic discriminant between the two is Okubo–Weiss:

```
W = S² − ω²        S = strain rate, ω = dv_dx − du_dy
```

`W > 0` is strain-dominated — a seam, a shear line, fast sliding past slow. `W < 0` is rotation-dominated — an eddy core, the recirculation behind a point or bar that the spec asks for. It costs one extra subtraction on gradients already computed.

**Files:**
- Modify: `backend/tidescout/engine/structure.py`
- Test: `backend/tests/test_structure.py`

**Interfaces:**
- Consumes: `GradientTensor`, `strain_rate` from Task 3
- Produces:
  - `vorticity(t: GradientTensor) -> np.ndarray`
  - `okubo_weiss(t: GradientTensor) -> np.ndarray`
  - `classify_structure(t: GradientTensor, quiet: float = 1e-5) -> np.ndarray` — int8, `+1` seam, `−1` eddy, `0` quiet
  Task 9 samples both `okubo_weiss` and `classify_structure` at feature geometry.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_structure.py (append)
def test_vorticity_recovers_twice_the_rotation_rate():
    x, y, cell = _xy()
    omega = 0.002
    t = structure.gradient_tensor(-omega * y, omega * x, cell)
    assert np.allclose(structure.vorticity(t)[1:-1, 1:-1], 2 * omega)


def test_okubo_weiss_is_negative_inside_a_rotating_eddy():
    """The signal Task 3's strain rate is blind to, by construction."""
    x, y, cell = _xy()
    omega = 0.002
    t = structure.gradient_tensor(-omega * y, omega * x, cell)
    w = structure.okubo_weiss(t)[1:-1, 1:-1]
    assert np.all(w < 0)
    assert np.allclose(w, -((2 * omega) ** 2))


def test_okubo_weiss_is_positive_in_a_pure_shear_seam():
    x, y, cell = _xy()
    a = 0.004
    t = structure.gradient_tensor(a * y, np.zeros_like(y), cell)
    # Pure shear is half strain, half rotation: S = a, omega = -a, so W = 0.
    # Tilting it toward stretching must push W positive.
    w = structure.okubo_weiss(structure.gradient_tensor(a * x, -a * y, cell))[1:-1, 1:-1]
    assert np.all(w > 0)
    assert np.allclose(structure.okubo_weiss(t)[1:-1, 1:-1], 0.0, atol=1e-12)


def test_classify_structure_labels_eddy_seam_and_quiet_water():
    x, y, cell = _xy()
    eddy = structure.gradient_tensor(-0.002 * y, 0.002 * x, cell)
    seam = structure.gradient_tensor(0.004 * x, -0.004 * y, cell)
    still = structure.gradient_tensor(np.full_like(x, 0.3), np.zeros_like(x), cell)

    assert np.all(structure.classify_structure(eddy)[1:-1, 1:-1] == -1)
    assert np.all(structure.classify_structure(seam)[1:-1, 1:-1] == 1)
    assert np.all(structure.classify_structure(still)[1:-1, 1:-1] == 0)


def test_classify_structure_calls_uniform_flow_quiet_not_seam():
    """Water moving fast in a straight line holds no fish. Without the quiet
    band, floating-point noise in a uniform field would sign W arbitrarily."""
    x, y, cell = _xy()
    t = structure.gradient_tensor(np.full_like(x, 1.2), np.full_like(x, -0.4), cell)
    assert np.all(structure.classify_structure(t)[1:-1, 1:-1] == 0)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_structure.py -k "vorticity or okubo or classify" -v`
Expected: FAIL — `AttributeError: module has no attribute 'vorticity'`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/structure.py (append)
def vorticity(t: GradientTensor) -> np.ndarray:
    """Local rotation rate, dv_dx - du_dy. Positive is anticlockwise."""
    return t.dv_dx - t.du_dy


def okubo_weiss(t: GradientTensor) -> np.ndarray:
    """Strain-vs-rotation discriminant, W = S^2 - omega^2.

    Strain rate alone cannot find an eddy -- it is exactly zero for solid-body
    rotation, which is the whole point of the formula Task 3 chose. Vorticity
    alone cannot find a seam, since a shear line and a vortex both spin. The
    difference of their squares separates them, and it is the standard
    oceanographic test:

      W > 0  strain-dominated -- a seam: fast water sliding past slow
      W < 0  rotation-dominated -- an eddy core, the lee behind a point or bar

    Pure parallel shear sits exactly at W = 0, being half of each.
    """
    return strain_rate(t) ** 2 - vorticity(t) ** 2


def classify_structure(t: GradientTensor, quiet: float = 1e-5) -> np.ndarray:
    """+1 seam, -1 eddy, 0 quiet water. int8, grid-shaped.

    `quiet` is a floor on |W| in s^-2, not a tuned threshold: uniform flow has
    W = 0 up to floating-point noise, and without a dead band that noise would
    sign every cell of a featureless channel at random. 1e-5 s^-2 corresponds
    to velocity gradients around 3e-3 s^-1 -- roughly 0.06 m/s across a 20 m
    cell, which is below what the mesh resolves anyway.
    """
    w = okubo_weiss(t)
    out = np.zeros(w.shape, dtype="int8")
    out[w > quiet] = 1
    out[w < -quiet] = -1
    return out
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_structure.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/engine/structure.py backend/tests/test_structure.py
git commit -m "feat: Okubo-Weiss eddy/seam discrimination"
```

---

## Task 5: Ambush contrast — the Georgetown mechanism

Ellis on Georgetown Lighthouse (2026-08-16): the spot works because it *"hides from the main channel current"*. That is the spec's "slow pockets adjacent to fast conveyors", and it is a **neighbourhood** property — a cell is not an ambush point because it is slow, but because it is slow **and** fast water is within a short dart of it. Validation measured exactly this shape at Georgetown: a 0.000 m/s pocket beside water moving ~1 m/s.

```
ambush_contrast(cell) = max(speed within R) − speed(cell)
```

`R` defaults to 150 m — the radius the Task 13 gate already uses for known spots, and roughly the distance a redfish will move to intercept bait.

**Files:**
- Modify: `backend/tidescout/engine/structure.py`
- Modify: `backend/tidescout/models.py` (`StructureThresholds`)
- Test: `backend/tests/test_structure.py`

**Interfaces:**
- Produces: `ambush_contrast(speed: np.ndarray, cell_m: float, radius_m: float = 150.0) -> np.ndarray`, `StructureThresholds`
- Consumes: nothing from Task 4; independent of the tensor.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_structure.py (append)
def test_ambush_contrast_peaks_in_a_slow_pocket_beside_a_fast_conveyor():
    """The Georgetown shape: a stagnant pocket adjacent to the fastest water."""
    speed = np.full((64, 64), 0.05)
    speed[:, 32:] = 1.0          # a fast conveyor filling the east half
    speed[28:36, 24:32] = 0.0    # a dead pocket hard against its west edge

    c = structure.ambush_contrast(speed, cell_m=20.0, radius_m=150.0)

    assert c[31, 28] == pytest.approx(1.0)   # in the pocket, conveyor in reach
    assert c[31, 40] == pytest.approx(0.0)   # inside the conveyor: nothing faster
    assert c[5, 5] < 0.01                    # far slack water: no fast neighbour


def test_ambush_contrast_is_zero_in_uniform_flow_however_fast():
    """Speed alone is not the signal. A uniform 2 m/s river has no ambush."""
    c = structure.ambush_contrast(np.full((32, 32), 2.0), cell_m=20.0, radius_m=150.0)
    assert np.allclose(c, 0.0)


def test_ambush_contrast_reach_is_set_by_radius_not_cell_count():
    """A pocket 200 m from fast water is out of reach at R=100 m, in reach at
    R=300 m. The radius must be interpreted in metres via cell_m."""
    speed = np.full((64, 64), 0.0)
    speed[:, 42:] = 1.0
    at_100 = structure.ambush_contrast(speed, cell_m=20.0, radius_m=100.0)
    at_300 = structure.ambush_contrast(speed, cell_m=20.0, radius_m=300.0)
    assert at_100[32, 32] == pytest.approx(0.0)
    assert at_300[32, 32] == pytest.approx(1.0)


def test_ambush_contrast_ignores_out_of_domain_neighbours():
    """NaN marks land. A pocket beside dry marsh must not inherit its NaN, and
    must not be credited with a fast neighbour that does not exist."""
    speed = np.full((32, 32), 0.1)
    speed[:, 20:] = np.nan
    c = structure.ambush_contrast(speed, cell_m=20.0, radius_m=150.0)
    assert np.isfinite(c[16, 16])
    assert c[16, 16] == pytest.approx(0.0)
    assert np.isnan(c[16, 25])
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_structure.py -k ambush -v`
Expected: FAIL — no attribute `ambush_contrast`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/structure.py -- add to the imports
from scipy import ndimage


def _disk(radius_m: float, cell_m: float) -> np.ndarray:
    """Circular footprint. A square window would reach 1.41x further on the
    diagonals, so a pocket would light up from fast water that is out of reach
    in one direction but not another -- an artefact of the grid, not the flow.
    """
    r = max(int(round(radius_m / cell_m)), 1)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2) <= r**2


def ambush_contrast(
    speed: np.ndarray, cell_m: float, radius_m: float = 150.0
) -> np.ndarray:
    """How much faster the nearby water is than this cell, in m/s.

    The spec's "slow pockets adjacent to fast conveyors", and the mechanism
    Ellis describes at Georgetown Lighthouse: the spot works because it hides
    FROM the main channel current. Being slow is not enough and being fast is
    not enough -- what matters is a low-speed cell with a high-speed neighbour
    within a short dart, so a fish can hold out of the flow and feed in it.

    NaN marks out-of-domain. `maximum_filter` would propagate it across the
    whole footprint, so the max is taken over a NaN-to--inf copy and the mask
    is reapplied afterwards; a cell beside land is credited with no neighbour
    rather than an infinitely fast one.
    """
    invalid = ~np.isfinite(speed)
    filled = np.where(invalid, -np.inf, speed)
    local_max = ndimage.maximum_filter(
        filled, footprint=_disk(radius_m, cell_m), mode="nearest"
    )
    out = local_max - np.where(invalid, np.nan, speed)
    out[invalid] = np.nan
    return np.maximum(out, 0.0, where=np.isfinite(out), out=out)
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_structure.py -v`
Expected: PASS, 16 tests.

- [ ] **Step 5: Add the tunable thresholds**

```python
# backend/tidescout/models.py (new class, above Fishery)
class StructureThresholds(BaseModel):
    """Derived-structure knobs. Tunable per fishery during validation."""

    # Radius a fish will move to intercept bait. Matches the radius the
    # known-spots validation gate already uses, so a spot that reads as an
    # ambush point in the gate reads as one here too.
    ambush_radius_m: float = 150.0
    # Dead band on |Okubo-Weiss| below which water is "quiet" rather than
    # seam or eddy. Not tuned to make anything pass -- it exists because
    # uniform flow sits at W = 0 and floating-point noise would otherwise
    # sign every cell of a featureless channel at random.
    quiet_w: float = 1e-5
    # Minimum convergence (negative divergence) counted as a bait-pinning
    # front, in s^-1. 1e-4 is ~0.002 m/s of closing speed across a 20 m cell.
    convergence_min: float = 1e-4


# in Fishery, alongside `features`
    structure: StructureThresholds = StructureThresholds()
```

- [ ] **Step 6: Run the gate and commit**

Run: `make check`
Expected: PASS.

```bash
git add backend/tidescout/engine/structure.py backend/tidescout/models.py \
        backend/tests/test_structure.py
git commit -m "feat: ambush contrast -- slow pockets beside fast conveyors"
```

---

## Task 6: Convergence at draining creek mouths

The last field the spec names. Convergence is negative divergence: water closing on itself, which is what pins bait at a draining creek mouth on the ebb. It reuses the tensor and is the signal most exposed to the `np.gradient` sign trap, since divergence and stretching differ only by that sign.

**Files:**
- Modify: `backend/tidescout/engine/structure.py`
- Test: `backend/tests/test_structure.py`

**Interfaces:**
- Consumes: `GradientTensor` from Task 3
- Produces: `divergence(t) -> np.ndarray`, `convergence(t) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_structure.py (append)
def test_divergence_is_positive_for_a_source_and_negative_for_a_sink():
    x, y, cell = _xy()
    k = 0.003
    src = structure.gradient_tensor(k * x, k * y, cell)
    snk = structure.gradient_tensor(-k * x, -k * y, cell)
    assert np.allclose(structure.divergence(src)[1:-1, 1:-1], 2 * k)
    assert np.allclose(structure.divergence(snk)[1:-1, 1:-1], -2 * k)


def test_convergence_is_positive_where_water_closes_on_itself():
    """Convergence is what pins bait; the sign is flipped so 'more is better'
    holds for every structure field, which the scoring engine relies on."""
    x, y, cell = _xy()
    k = 0.003
    snk = structure.gradient_tensor(-k * x, -k * y, cell)
    assert np.allclose(structure.convergence(snk)[1:-1, 1:-1], 2 * k)


def test_divergence_is_not_the_stretching_term():
    """The np.gradient row-orientation trap: du_dx - dv_dy and du_dx + dv_dy
    are different fields, and a sign error silently swaps them. This field
    separates them -- stretching is zero, divergence is not."""
    x, y, cell = _xy()
    k = 0.003
    t = structure.gradient_tensor(k * x, k * y, cell)
    assert np.allclose(structure.divergence(t)[1:-1, 1:-1], 2 * k)
    assert np.allclose((t.du_dx - t.dv_dy)[1:-1, 1:-1], 0.0, atol=1e-12)


def test_convergence_is_zero_in_solid_body_rotation():
    x, y, cell = _xy()
    t = structure.gradient_tensor(-0.002 * y, 0.002 * x, cell)
    assert np.nanmax(np.abs(structure.convergence(t)[1:-1, 1:-1])) < 1e-9
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_structure.py -k "divergence or convergence" -v`
Expected: FAIL — no attribute `divergence`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/structure.py (append)
def divergence(t: GradientTensor) -> np.ndarray:
    """du_dx + dv_dy. Positive where water spreads, negative where it closes.

    Differs from the stretching term du_dx - dv_dy by one sign, and the
    np.gradient row-orientation trap turns one into the other silently -- which
    is why `xy_gradients` negates the row derivative once, centrally, and
    nothing else in this module touches np.gradient.
    """
    return t.du_dx + t.dv_dy


def convergence(t: GradientTensor) -> np.ndarray:
    """-divergence: water closing on itself, which pins bait.

    Sign-flipped so that larger is more fish-relevant, the same convention every
    other field here follows. The scoring engine maps each structure field
    through a monotone response curve and would need a special case otherwise.
    """
    return -divergence(t)
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_structure.py -v`
Expected: PASS, 20 tests.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/engine/structure.py backend/tests/test_structure.py
git commit -m "feat: convergence field for draining creek mouths"
```

---

## Task 7: Flats flood/drain schedule

The one derived signal that is **not** instantaneous: a flat's value is *when* it floods and drains, which only exists across a whole cycle. Cheap to precompute (26 phases × one boolean per cell) and small enough to store as a table rather than a raster stack.

> **CORRECTION, 2026-08-17 (found during execution). Step 5's third expectation below is
> WRONG — the algorithm is not.** Step 5 asks that the median drain phase land in
> 0.5–1.0. **Phase is a circular quantity and an ordinary median of one is meaningless.**
> Measured on the shipped library: **33.8% of neap and 35.7% of spring intertidal cells
> drain in phase 0.0–0.2** — slow-draining high marsh that holds water past low water,
> which is physically expected and grows with tidal range. The wrap-safe statistic is
> stable and sensible: wet-window length `(drain − flood) mod 1` has p50 **0.523 in both
> regimes** (a flat is wet for about half a cycle), and the early-draining cells have
> median windows of 0.60–0.64. Only the *median phase* moves across the wrap — 0.765 at
> neap, 0.403 at spring — which is an artifact of the statistic, not the model.
>
> `flood_phase` was never wrong (0.443 median, identically, across all three regimes).
>
> One real refinement did land: `drain_phase` is now the first dry transition at a cyclic
> index *after* the flood, so the two describe one wet window rather than being
> independent first-crossings. On the shipped library this changed **zero** cells — it is
> a correctness guard for multi-window cells, verified by a synthetic regression test, not
> a fix for the numbers above. A first controller ruling misattributed the 0.403 to this
> and was wrong; the pairing was kept because it is right, not because it helped.
>
> **Use `(drain − flood) mod 1` to verify this task, never a raw median of drain phase.**

**Files:**
- Create: `backend/tidescout/pipeline/schedule.py`
- Create: `backend/tests/test_schedule.py`

**Interfaces:**
- Consumes: `flowlib.load_state`, `flow.wet_mask`, `flow.tide_states`
- Produces:
  - `CellSchedule` dataclass: `wet_fraction: np.ndarray`, `flood_phase: np.ndarray`, `drain_phase: np.ndarray` (all 1-D, library-masked, NaN where never wet or always wet)
  - `cell_schedule(slug: str, regime: str) -> CellSchedule`
  - `write_schedule(slug: str, regime: str) -> Path` → `data/<slug>/flow/<regime>/grid/schedule.npz`
  Task 9 reads `wet_fraction` and `flood_phase` per feature.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_schedule.py
"""Flood/drain timing from a synthetic depth series.

Phase 0 is LOW water here -- spin-up is 0.4831 of a cycle -- so a flat that
floods on the rising half floods in phase 0.0-0.5. Getting that backwards
inverts every flat in the bay, so these fixtures pin it explicitly.
"""

import numpy as np
import pytest

from tidescout.pipeline import schedule


def _series(depths_by_phase):
    """depths_by_phase: (n_phases, n_cells) -> the shape load_state returns."""
    return [np.asarray(row, dtype="float32") for row in depths_by_phase]


def test_wet_fraction_counts_the_share_of_the_cycle_a_cell_holds_water():
    depths = _series([[1.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 0.0]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.wet_fraction[0] == pytest.approx(1.0)
    assert s.wet_fraction[1] == pytest.approx(0.25)


def test_flood_phase_is_when_the_cell_first_goes_wet_on_the_rising_half():
    """Cell floods at phase 0.25 (rising) and drains at 0.75 (falling)."""
    depths = _series([[0.0], [0.4], [0.6], [0.0]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.flood_phase[0] == pytest.approx(0.25)
    assert s.drain_phase[0] == pytest.approx(0.75)


def test_always_wet_and_never_wet_cells_have_no_schedule():
    """A channel has no flood time and a marsh hummock has no drain time.
    NaN says 'this question does not apply here' -- 0.0 would be a lie that
    reads as 'floods at low water'."""
    depths = _series([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.wet_fraction[0] == pytest.approx(1.0)
    assert np.isnan(s.flood_phase[0]) and np.isnan(s.drain_phase[0])
    assert s.wet_fraction[1] == pytest.approx(0.0)
    assert np.isnan(s.flood_phase[1]) and np.isnan(s.drain_phase[1])


def test_schedule_wraps_cyclically_across_the_end_of_the_series():
    """A cell wet at the end and start of the record floods before phase 0.
    Treating the series as a line rather than a cycle would report it as
    never flooding."""
    depths = _series([[0.5], [0.0], [0.0], [0.5]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.flood_phase[0] == pytest.approx(0.75)
    assert s.drain_phase[0] == pytest.approx(0.25)
    assert s.wet_fraction[0] == pytest.approx(0.5)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_schedule.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.pipeline.schedule`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/pipeline/schedule.py
"""Flats flood/drain schedule -- the one derived signal that is not
instantaneous.

A flat's fishing value is WHEN it floods and drains, not what the water is
doing at one instant, so this is computed across a whole regime's phase series
and stored as a small per-cell table. Three float32 arrays over 587,325 cells
is ~7 MB per regime, against ~529 MB to store any instantaneous field, which is
why this one is precomputed and the others are not.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tidescout.engine.flow import wet_mask
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.flowlib import load_state


@dataclass
class CellSchedule:
    wet_fraction: np.ndarray  # 0-1, share of the cycle holding water
    flood_phase: np.ndarray   # phase at which it goes wet; NaN if never/always
    drain_phase: np.ndarray   # phase at which it goes dry; NaN if never/always


def schedule_from_depths(depths: list[np.ndarray], phases: list[float]) -> CellSchedule:
    """Per-cell wet fraction and flood/drain phases from a depth series.

    The series is CYCLIC: a cell wet at the last phase and the first floods
    somewhere before phase 0, and treating the record as a line rather than a
    ring would report it as never flooding at all. `np.roll` gives the previous
    phase's state with that wrap built in.

    Phase 0 is LOW water (spin_up_h / cycle_h = 0.4831 of a cycle), so a flat
    that floods on the rising half floods in phase 0.0-0.5.
    """
    wet = np.array([wet_mask(d) for d in depths])  # (n_phases, n_cells)
    ph = np.asarray(phases, dtype="float64")
    n = wet.shape[0]

    wet_fraction = wet.mean(axis=0).astype("float32")
    was_wet = np.roll(wet, 1, axis=0)
    goes_wet = wet & ~was_wet   # dry -> wet transition at this phase
    goes_dry = ~wet & was_wet

    def first_phase(events: np.ndarray) -> np.ndarray:
        # A cell can cross more than once in a cycle; the first crossing is the
        # one that matters for "when can I fish it", and taking argmax of a
        # boolean gives that for free.
        any_event = events.any(axis=0)
        idx = events.argmax(axis=0)
        out = np.where(any_event, ph[idx], np.nan)
        return out.astype("float32")

    flood_phase = first_phase(goes_wet)
    drain_phase = first_phase(goes_dry)
    # Always-wet and never-wet cells have no transition; first_phase already
    # returns NaN for them, but say so explicitly rather than relying on it.
    static = (wet.sum(axis=0) == n) | (wet.sum(axis=0) == 0)
    flood_phase[static] = np.nan
    drain_phase[static] = np.nan
    return CellSchedule(wet_fraction, flood_phase, drain_phase)


def cell_schedule(slug: str, regime: str) -> CellSchedule:
    grid_meta = json.loads(
        (fishery_data_dir(slug) / "flow" / regime / "grid" / "grid.json").read_text()
    )
    phases = grid_meta["phases"]
    depths = [load_state(slug, regime, i)["depth"] for i in range(len(phases))]
    return schedule_from_depths(depths, phases)


def write_schedule(slug: str, regime: str) -> Path:
    s = cell_schedule(slug, regime)
    out = fishery_data_dir(slug) / "flow" / regime / "grid" / "schedule.npz"
    np.savez_compressed(
        out,
        wet_fraction=s.wet_fraction,
        flood_phase=s.flood_phase,
        drain_phase=s.drain_phase,
    )
    return out
```

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_schedule.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Build the schedules for real and sanity-check them**

```bash
python -c "
from tidescout.pipeline import schedule
import numpy as np
for r in ['neap_low','mean_med','spring_high']:
    s = schedule.cell_schedule('winyah-bay', r)
    inter = np.isfinite(s.flood_phase)
    print(f'{r:12s} intertidal {inter.sum():7,d} cells '
          f'({inter.mean()*100:5.2f}%)  '
          f'always-wet {(s.wet_fraction==1).sum():7,d}  '
          f'never-wet {(s.wet_fraction==0).sum():7,d}')
    print(f'{\"\":12s} flood phase p50 {np.nanmedian(s.flood_phase):.3f} '
          f'drain p50 {np.nanmedian(s.drain_phase):.3f}')
"
```

Expected: the intertidal share should **rise from neap to spring** — a bigger tide wets and drains more ground. Median flood phase should land in **0.0–0.5** (the rising half) and median drain phase in **0.5–1.0**. If flood and drain are swapped, the phase convention has been inverted somewhere; stop and find it rather than adjusting the test.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/schedule.py backend/tests/test_schedule.py
git commit -m "feat: flats flood/drain schedule from the wet mask"
```

---

## Task 8: Stable feature identifiers

Carryover open item 4, flagged as **"Plan 4 must not skip it"**. `features.build_features` ids features by a per-type running counter (`f"{f.type}-{counters[f.type]}"`), so `bar-78` becomes a different bar whenever detection reruns and the ordering shifts. Nothing has persisted a feature reference until now; Phase 3 scoring will, and the frontend will store user pins against these ids.

The key must be stable under rebuild, unique, and independent of iteration order — so: a hash of the feature type plus its centroid, quantised so that floating-point jitter in the detector cannot move it.

**Files:**
- Modify: `backend/tidescout/engine/detect.py`
- Modify: `backend/tidescout/pipeline/features.py:59-73`
- Test: `backend/tests/test_detect.py`, `backend/tests/test_features_pipeline.py`

**Interfaces:**
- Produces: `detect.feature_key(feature: Feature, quantise_m: float = 1.0) -> str` — a 12-hex-character id like `dropoff-9f2c1a7b4e05`. Phase 3 uses it as the primary key for per-feature activation.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_detect.py (append)
from shapely.geometry import LineString, Point

from tidescout.engine.detect import Feature, feature_key


def test_feature_key_is_stable_across_rebuilds():
    a = Feature("dropoff", LineString([(100.0, 200.0), (140.0, 260.0)]))
    b = Feature("dropoff", LineString([(100.0, 200.0), (140.0, 260.0)]))
    assert feature_key(a) == feature_key(b)


def test_feature_key_is_independent_of_detection_order():
    """The bug this replaces: ids came from a running counter, so inserting one
    feature renumbered every later feature of that type."""
    feats = [
        Feature("hole", Point(10.0, 20.0)),
        Feature("hole", Point(30.0, 40.0)),
    ]
    keys_forward = [feature_key(f) for f in feats]
    keys_reversed = [feature_key(f) for f in reversed(feats)]
    assert set(keys_forward) == set(keys_reversed)


def test_feature_key_separates_types_at_the_same_place():
    p = Point(10.0, 20.0)
    assert feature_key(Feature("hole", p)) != feature_key(Feature("bar", p))


def test_feature_key_absorbs_sub_metre_detector_jitter():
    """A re-run that moves a centroid by 20 cm must not mint a new feature."""
    a = Feature("bar", Point(1000.0, 2000.0))
    b = Feature("bar", Point(1000.2, 1999.8))
    assert feature_key(a) == feature_key(b)


def test_feature_key_distinguishes_features_a_few_metres_apart():
    a = Feature("bar", Point(1000.0, 2000.0))
    b = Feature("bar", Point(1010.0, 2000.0))
    assert feature_key(a) != feature_key(b)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_detect.py -k feature_key -v`
Expected: FAIL — `ImportError: cannot import name 'feature_key'`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/detect.py -- add near the Feature dataclass
import hashlib


def feature_key(feature: Feature, quantise_m: float = 1.0) -> str:
    """Stable identifier: type plus quantised centroid, hashed.

    Feature ids were a per-type running counter, so `bar-78` became a different
    bar whenever detection reran and the ordering shifted. Nothing persisted a
    feature reference, so nothing broke -- but Phase 3 scoring and the frontend's
    user pins both key off these, and a renumbering would silently reassign
    every one of them.

    The centroid is quantised to `quantise_m` (1 m, a tenth of the 10 m analysis
    cell) before hashing. Detector output moves by centimetres between runs from
    floating-point ordering alone; without quantisation that jitter would mint a
    new id every rebuild, which is the exact bug this replaces. One metre is
    coarse enough to absorb the jitter and far finer than the spacing between
    two genuinely distinct features.

    UTM metres in, so the quantisation is isotropic and in real units -- never
    call this with lon/lat, where 1.0 would be ~100 km.
    """
    c = feature.geometry.centroid
    qx = round(c.x / quantise_m)
    qy = round(c.y / quantise_m)
    digest = hashlib.sha256(f"{feature.type}|{qx}|{qy}".encode()).hexdigest()[:12]
    return f"{feature.type}-{digest}"
```

- [ ] **Step 4: Use it in the pipeline**

```python
# backend/tidescout/pipeline/features.py -- replace the counters block in build_features
    out = []
    for f in feats:
        props = {"type": f.type}
        for k, v in f.attrs.items():
            props[k] = round(v, 2) if isinstance(v, float) else v
        out.append(
            {
                "type": "Feature",
                # Hash of type + quantised UTM centroid, computed BEFORE
                # reprojection: _to4326 would put the centroid in degrees,
                # where a 1 m quantum is ~100 km.
                "id": detect.feature_key(f),
                "properties": props,
                "geometry": mapping(_to4326(f.geometry, epsg)),
            }
        )
```

Delete the now-unused `counters` dict.

- [ ] **Step 5: Write the pipeline-level stability test**

```python
# backend/tests/test_features_pipeline.py (append)
def test_rebuilt_features_keep_their_ids(tmp_path, monkeypatch, synth_fishery):
    """The carryover's trap (c): `bar-78` renumbered on every rebuild."""
    from tidescout.pipeline import features

    first = json.loads(features.build_features(*synth_fishery).read_text())
    second = json.loads(features.build_features(*synth_fishery).read_text())

    ids_first = sorted(f["id"] for f in first["features"])
    ids_second = sorted(f["id"] for f in second["features"])
    assert ids_first == ids_second
    assert len(set(ids_first)) == len(ids_first), "feature ids must be unique"
```

(Reuse whatever synthetic-DEM fixture `test_features_pipeline.py` already defines; name it in place of `synth_fishery`.)

- [ ] **Step 6: Run the tests**

Run: `pytest backend/tests/test_detect.py backend/tests/test_features_pipeline.py -v`
Expected: PASS.

- [ ] **Step 7: Rebuild the real inventory and confirm uniqueness**

```bash
tidescout features winyah-bay
python -c "
import json
d = json.load(open('data/winyah-bay/features.geojson'))
ids = [f['id'] for f in d['features']]
print(f'{len(ids):,} features, {len(set(ids)):,} unique')
assert len(ids) == len(set(ids)), 'collision -- raise the hash width'
"
```

Expected: counts equal. A collision at 12 hex characters (2⁴⁸) over a few thousand features is essentially impossible; if one appears, the centroids are genuinely coincident and the detector is emitting duplicates.

- [ ] **Step 8: Commit**

```bash
git add backend/tidescout/engine/detect.py backend/tidescout/pipeline/features.py \
        backend/tests/test_detect.py backend/tests/test_features_pipeline.py
git commit -m "feat: stable feature ids from type + quantised centroid

Carryover item 4. Replaces a per-type running counter that renumbered every
feature on rebuild, before anything persisted a reference to one."
```

---

## Task 9: Feature activation — sampling structure at the features

Joins the two halves: the static inventory from Plan 2 and the derived fields from Tasks 3–7. For one regime and phase, produce per-feature metrics that Phase 3 turns into activation scores.

**Files:**
- Create: `backend/tidescout/engine/activation.py`
- Create: `backend/tests/test_activation.py`

**Interfaces:**
- Consumes: `structure.*` (Tasks 3–6), `CellSchedule` (Task 7), `detect.feature_key` (Task 8), `flowlib.GridSpec`
- Produces:
  - `FeatureMetrics` dataclass: `key, type, speed, ambush, strain, okubo_w, convergence, wet_fraction, flood_phase, n_cells`
  - `sample_features(features: list[dict], spec, fields: dict[str, np.ndarray], schedule=None, radius_m: float = 150.0) -> list[FeatureMetrics]`
  - `structure_fields(u, v, depth, spec, thresholds) -> dict[str, np.ndarray]` — the one call that runs the whole Task 3–6 chain and returns 1-D masked arrays
  Phase 3 Task 4 consumes `FeatureMetrics` directly.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_activation.py
"""Feature sampling on hand-built fields.

Each fixture puts a known value under a known feature so a wrong answer points
at the sampling, not at the physics.
"""

import numpy as np
import pytest
from affine import Affine

from tidescout.engine import activation


class _Spec:
    """Minimal stand-in for flowlib.GridSpec: an 8x8 grid of 20 m cells."""

    def __init__(self):
        self.shape = (8, 8)
        self.cell_m = 20.0
        self.transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 160.0)
        self.flat_index = np.arange(64)
        cols, rows = np.meshgrid(np.arange(8), np.arange(8))
        self.xs, self.ys = self.transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)


def _feature(key, ftype, lonlat_coords):
    return {
        "id": key,
        "properties": {"type": ftype},
        "geometry": {"type": "Point", "coordinates": lonlat_coords},
    }


def test_sample_features_averages_the_field_within_the_radius():
    spec = _Spec()
    speed = np.zeros(64)
    speed[spec.flat_index] = 0.5
    # One cell much faster, inside the sample radius of the feature below.
    speed[27] = 2.5

    feats = [_feature("hole-abc", "hole", (spec.xs[27], spec.ys[27]))]
    out = activation.sample_features(
        feats, spec, {"speed": speed}, radius_m=25.0, already_projected=True
    )
    assert len(out) == 1
    assert out[0].key == "hole-abc"
    assert out[0].speed == pytest.approx(2.5)


def test_sample_features_reports_the_max_for_ambush_not_the_mean():
    """An ambush point is defined by its best cell. Averaging a 150 m disc over
    a 20 m grid would dilute a real pocket into the channel around it."""
    spec = _Spec()
    ambush = np.zeros(64)
    ambush[27] = 1.0
    feats = [_feature("bar-def", "bar", (spec.xs[27], spec.ys[27]))]
    out = activation.sample_features(
        feats, spec, {"ambush": ambush}, radius_m=60.0, already_projected=True
    )
    assert out[0].ambush == pytest.approx(1.0)


def test_features_with_no_cells_in_the_domain_are_returned_with_nan_not_dropped():
    """Dropping them would make a feature vanish from the map with no
    explanation. NaN plus n_cells=0 says 'outside the model domain'."""
    spec = _Spec()
    feats = [_feature("hole-far", "hole", (999999.0, 999999.0))]
    out = activation.sample_features(
        feats, spec, {"speed": np.zeros(64)}, radius_m=25.0, already_projected=True
    )
    assert len(out) == 1
    assert out[0].n_cells == 0
    assert np.isnan(out[0].speed)


def test_structure_fields_returns_masked_1d_arrays_on_the_library_layout():
    """The round trip must give back exactly the cells it was handed."""
    spec = _Spec()
    n = spec.flat_index.size
    u = np.full(n, 0.4)
    v = np.zeros(n)
    depth = np.full(n, 2.0)
    fields = activation.structure_fields(u, v, depth, spec)
    for name in ("speed", "ambush", "strain", "okubo_w", "convergence"):
        assert fields[name].shape == (n,), name
    assert np.allclose(fields["speed"], 0.4)
    assert np.allclose(fields["ambush"], 0.0)  # uniform flow: no contrast
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_activation.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.engine.activation`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/activation.py
"""Sample derived structure at the static feature inventory.

The join between Plan 2's features and Plan 4's fields. Pure: the caller loads
the library state and the GeoJSON and hands both in.
"""

from dataclasses import dataclass

import numpy as np

from tidescout.engine import structure
from tidescout.models import StructureThresholds

# Fields summarised by their best cell rather than their mean. An ambush point,
# a seam and a convergence front are all defined by their strongest cell -- a
# 150 m disc over a 20 m grid holds ~175 cells, and averaging would dilute a
# real pocket into the channel around it. Speed and depth-derived quantities
# describe the feature as a whole, so those take the mean.
_MAX_FIELDS = frozenset({"ambush", "strain", "okubo_w", "convergence"})


@dataclass
class FeatureMetrics:
    key: str
    type: str
    speed: float
    ambush: float
    strain: float
    okubo_w: float
    convergence: float
    wet_fraction: float
    flood_phase: float
    n_cells: int


def structure_fields(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    spec,
    thresholds: StructureThresholds | None = None,
) -> dict[str, np.ndarray]:
    """Run the whole derived-structure chain for one flow state.

    Scatters the masked library arrays onto the raster, differentiates there,
    then gathers back -- gradients need neighbours, and the stored arrays have
    none. Computed per call rather than stored: the algebra costs milliseconds,
    the storage would cost ~2 GB and a library rebuild, and computing on the
    phase-interpolated velocity is more correct than interpolating these
    nonlinear quantities between phases.
    """
    t = thresholds or StructureThresholds()
    ug = structure.to_grid(u, spec.flat_index, spec.shape)
    vg = structure.to_grid(v, spec.flat_index, spec.shape)
    speed_g = np.hypot(ug, vg)

    tensor = structure.gradient_tensor(ug, vg, spec.cell_m)
    fields_2d = {
        "speed": speed_g,
        "ambush": structure.ambush_contrast(speed_g, spec.cell_m, t.ambush_radius_m),
        "strain": structure.strain_rate(tensor),
        "okubo_w": structure.okubo_weiss(tensor),
        "convergence": structure.convergence(tensor),
    }
    return {k: structure.from_grid(g, spec.flat_index) for k, g in fields_2d.items()}


def sample_features(
    features: list[dict],
    spec,
    fields: dict[str, np.ndarray],
    schedule=None,
    radius_m: float = 150.0,
    already_projected: bool = False,
) -> list[FeatureMetrics]:
    """Per-feature summary of every field, over the cells within `radius_m`.

    `already_projected` is for tests that build coordinates directly in the
    grid CRS; production passes GeoJSON in EPSG:4326 and this reprojects.
    """
    if not already_projected:
        from rasterio.warp import transform as warp_transform

        pts = [_centroid_lonlat(f) for f in features]
        xs, ys = warp_transform(
            "EPSG:4326", spec.crs, [p[0] for p in pts], [p[1] for p in pts]
        )
    else:
        pts = [_centroid_lonlat(f) for f in features]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

    out = []
    r2 = radius_m**2
    for f, fx, fy in zip(features, xs, ys, strict=True):
        sel = (spec.xs - fx) ** 2 + (spec.ys - fy) ** 2 <= r2
        n = int(sel.sum())
        vals = {}
        for name in ("speed", "ambush", "strain", "okubo_w", "convergence"):
            arr = fields.get(name)
            if arr is None or n == 0:
                vals[name] = float("nan")
                continue
            here = arr[sel]
            reducer = np.nanmax if name in _MAX_FIELDS else np.nanmean
            vals[name] = float(reducer(here)) if np.isfinite(here).any() else float("nan")

        wet_fraction = flood_phase = float("nan")
        if schedule is not None and n:
            wf = schedule.wet_fraction[sel]
            fp = schedule.flood_phase[sel]
            if np.isfinite(wf).any():
                wet_fraction = float(np.nanmean(wf))
            if np.isfinite(fp).any():
                flood_phase = float(np.nanmedian(fp))

        out.append(
            FeatureMetrics(
                key=f["id"],
                type=f["properties"]["type"],
                wet_fraction=wet_fraction,
                flood_phase=flood_phase,
                n_cells=n,
                **vals,
            )
        )
    return out


def _centroid_lonlat(feature: dict) -> tuple[float, float]:
    """Representative point of a GeoJSON geometry, without a shapely round trip."""
    geom = feature["geometry"]
    coords = geom["coordinates"]
    kind = geom["type"]
    if kind == "Point":
        return float(coords[0]), float(coords[1])
    if kind == "LineString":
        pts = coords
    elif kind == "Polygon":
        pts = coords[0]
    else:
        raise TypeError(f"unsupported feature geometry: {kind}")
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)
```

Add `crs` to `flowlib.GridSpec` as `f"EPSG:{fishery.bathymetry.epsg}"`, set in `grid_spec`.

- [ ] **Step 4: Run the tests**

Run: `pytest backend/tests/test_activation.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Run the full gate and commit**

Run: `make check`

```bash
git add backend/tidescout/engine/activation.py backend/tests/test_activation.py \
        backend/tidescout/pipeline/flowlib.py
git commit -m "feat: sample derived structure at the feature inventory"
```

---

## Task 10: Oyster habitat as a static feature attribute

Carryover item 6 and spec §6's last feature class. `data/winyah-bay/oyster_reefs.geojson` holds **8,451 SCDNR reef polygons** (median reef 24 m², attributes `calcgeo_ac`, `photo_year`). The carryover is specific about what this is and is not: *"a Plan 4 scoring/habitat layer — not mesh geometry, not an ambush-feature class."*

So oyster reefs do not become features in their own right. They become an **attribute of the features that already exist**: how much reef sits within a short cast of this drop-off, this point, this creek mouth. Oysters hold bait and structure independently of what the tide is doing, so this is static — computed once into `features.geojson`, read by Phase 3 scoring.

**Files:**
- Create: `backend/tidescout/pipeline/oysters.py`, `backend/tests/test_oysters.py`
- Modify: `backend/tidescout/pipeline/features.py`

**Interfaces:**
- Consumes: `data/<slug>/oyster_reefs.geojson`, feature geometries in UTM
- Produces: `reef_area_m2_within(features_utm, reefs_utm, radius_m) -> list[float]`; each GeoJSON feature gains `properties["oyster_area_m2"]` and `properties["oyster_nearest_m"]`. Phase 3 Task 6 reads both.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_oysters.py
"""Reef proximity on hand-placed geometry."""

import pytest
from shapely.geometry import Point, Polygon

from tidescout.pipeline.oysters import reef_area_m2_within, nearest_reef_m


def _square(cx, cy, side):
    h = side / 2.0
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def test_reef_area_counts_only_reefs_inside_the_radius():
    reefs = [_square(0.0, 0.0, 10.0), _square(500.0, 0.0, 10.0)]
    got = reef_area_m2_within([Point(0.0, 0.0)], reefs, radius_m=100.0)
    assert got[0] == pytest.approx(100.0)


def test_reef_area_sums_multiple_nearby_reefs():
    reefs = [_square(10.0, 0.0, 10.0), _square(-10.0, 0.0, 10.0), _square(0.0, 20.0, 10.0)]
    got = reef_area_m2_within([Point(0.0, 0.0)], reefs, radius_m=100.0)
    assert got[0] == pytest.approx(300.0)


def test_a_feature_with_no_reefs_nearby_gets_zero_not_nan():
    """Zero reef is a real, common, meaningful answer -- most of the bay. NaN
    would make Phase 3 exclude the factor and renormalise, turning 'no oysters
    here' into 'we do not know', which are opposite statements."""
    got = reef_area_m2_within([Point(0.0, 0.0)], [_square(9999.0, 9999.0, 10.0)], 100.0)
    assert got[0] == 0.0


def test_nearest_reef_distance_is_zero_when_the_feature_sits_on_one():
    reefs = [_square(0.0, 0.0, 40.0)]
    assert nearest_reef_m([Point(5.0, 5.0)], reefs)[0] == pytest.approx(0.0)


def test_nearest_reef_distance_is_inf_when_there_are_no_reefs():
    import math

    assert math.isinf(nearest_reef_m([Point(0.0, 0.0)], [])[0])


def test_reef_lookup_handles_the_full_layer_size_efficiently():
    """8,451 reefs x thousands of features is a quadratic trap without an index."""
    reefs = [_square(float(i) * 30.0, 0.0, 8.0) for i in range(3000)]
    pts = [Point(float(i) * 30.0, 0.0) for i in range(500)]
    got = reef_area_m2_within(pts, reefs, radius_m=50.0)
    assert len(got) == 500
    assert all(g > 0 for g in got)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_oysters.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.pipeline.oysters`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/pipeline/oysters.py
"""SCDNR oyster reefs as a habitat attribute of the existing features.

Carryover item 6 is explicit that this is a scoring/habitat layer, not an
ambush-feature class and not mesh geometry: 8,451 polygons with a median area
of 24 m^2 would add thousands of markers to the map and resolve nothing, but
"this drop-off has 400 m2 of reef within a cast" is a real signal that holds
independently of the tide.

Static: reefs do not move with the tide, so this is computed once into
features.geojson rather than per hour.
"""

import math

from shapely.strtree import STRtree

# Roughly a cast. Wider and every feature in the bay picks up some reef;
# narrower and the median 24 m^2 reef is missed by features that fish it.
DEFAULT_RADIUS_M = 75.0


def reef_area_m2_within(
    features, reefs, radius_m: float = DEFAULT_RADIUS_M
) -> list[float]:
    """Total reef area within `radius_m` of each feature, in m^2.

    Zero, not NaN, when nothing is nearby: most of the bay has no reef, and that
    is a real answer. NaN would make Phase 3 exclude the factor and renormalise,
    converting "no oysters here" into "we have no oyster data" -- opposite
    claims with opposite consequences for the score.

    STRtree-indexed: the layer is 8,451 reefs against thousands of features, and
    the naive nested loop is 30M+ intersection tests.
    """
    if not reefs:
        return [0.0] * len(features)
    tree = STRtree(reefs)
    out = []
    for geom in features:
        buffered = geom.buffer(radius_m)
        total = 0.0
        for idx in tree.query(buffered):
            reef = reefs[idx]
            if buffered.intersects(reef):
                total += buffered.intersection(reef).area
        out.append(total)
    return out


def nearest_reef_m(features, reefs) -> list[float]:
    """Distance to the closest reef, 0.0 when the feature overlaps one.

    Infinity when there are no reefs at all -- a distance, unlike an area, has
    no meaningful zero-substitute, and inf makes any downstream curve clamp to
    its worst authored value rather than its best.
    """
    if not reefs:
        return [math.inf] * len(features)
    tree = STRtree(reefs)
    return [float(geom.distance(reefs[tree.nearest(geom)])) for geom in features]
```

- [ ] **Step 4: Wire it into the feature build**

In `build_features`, after detection and **before** reprojection to 4326 (both layers must be in the same UTM CRS for metre distances to mean anything), load the reef layer if present, reproject the reefs to UTM once, and attach `oyster_area_m2` and `oyster_nearest_m` to each feature's properties. A missing reef file is not an error — it is the spec's documented contingency ("SCDNR oyster layer unavailable → feature class is optional"); log it and set both to `0.0` and `inf`.

- [ ] **Step 5: Run the tests and rebuild the inventory**

```bash
pytest backend/tests/test_oysters.py -v
tidescout features winyah-bay
python -c "
import json
d = json.load(open('data/winyah-bay/features.geojson'))
withr = [f for f in d['features'] if f['properties'].get('oyster_area_m2', 0) > 0]
print(f'{len(withr):,} of {len(d[\"features\"]):,} features have reef within 75 m')
top = sorted(d['features'], key=lambda f: -f['properties'].get('oyster_area_m2', 0))[:5]
for f in top:
    print(f'  {f[\"id\"]:28s} {f[\"properties\"][\"oyster_area_m2\"]:8.0f} m2')
"
```

Expected: a minority of features carry reef — if nearly all or nearly none do, the CRS of one layer is wrong and the distances are meaningless.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/oysters.py backend/tests/test_oysters.py \
        backend/tidescout/pipeline/features.py
git commit -m "feat: oyster habitat as a static feature attribute

Carryover item 6. 8,451 SCDNR reefs become an attribute of existing features
rather than thousands of their own markers."
```

---

## Task 11: The `flow structure` CLI and the Georgetown re-check

Makes Phase 1 inspectable, and puts the question Ellis's note actually poses to the model: **not** "does Georgetown peak on the flood" but "does the model produce a current shadow there".

**Files:**
- Modify: `backend/tidescout/cli.py`
- Test: manual verification (the CLI is a thin shell over tested functions)

**Interfaces:**
- Consumes: everything above.
- Produces: `tidescout flow structure <slug> --regime <name> [--phase N] [--top N]`

- [ ] **Step 1: Add the command**

```python
# backend/tidescout/cli.py (append to the flow sub-app)
@flow_app.command("structure")
def flow_structure(
    slug: str,
    regime: str = typer.Option("mean_med", "--regime"),
    phase: int = typer.Option(-1, "--phase", help="phase index; -1 = every phase"),
    top: int = typer.Option(15, "--top", help="features to list"),
) -> None:
    """Derived structure at the feature inventory: seams, eddies, ambush points."""
    import numpy as np

    from tidescout.config import load_fishery
    from tidescout.engine import activation, flow
    from tidescout.pipeline import flowlib, schedule
    from tidescout.pipeline.features import load_features

    fishery = load_fishery(slug)
    spec = flowlib.grid_spec(slug, fishery)
    feats = load_features(slug)["features"]
    grid_meta = json.loads(
        (fishery_data_dir(slug) / "flow" / regime / "grid" / "grid.json").read_text()
    )
    states = flow.tide_states(grid_meta["stage_bc_m"])
    sched = schedule.cell_schedule(slug, regime)

    idxs = range(len(grid_meta["phases"])) if phase < 0 else [phase]
    best: dict[str, activation.FeatureMetrics] = {}
    best_state: dict[str, str] = {}
    for i in idxs:
        st = flowlib.load_state(slug, regime, i)
        fields = activation.structure_fields(
            st["u"], st["v"], st["depth"], spec, fishery.structure
        )
        for m in activation.sample_features(
            feats, spec, fields, sched, fishery.structure.ambush_radius_m
        ):
            prev = best.get(m.key)
            if prev is None or (np.nan_to_num(m.ambush) > np.nan_to_num(prev.ambush)):
                best[m.key] = m
                best_state[m.key] = states[i]

    ranked = sorted(
        (m for m in best.values() if m.n_cells),
        key=lambda m: np.nan_to_num(m.ambush),
        reverse=True,
    )[:top]

    table = Table(title=f"{fishery.name} — derived structure ({regime})")
    for col in ("feature", "type", "best state", "ambush m/s", "speed",
                "strain 1/s", "okubo W", "converg.", "wet frac"):
        table.add_column(col)
    for m in ranked:
        table.add_row(
            m.key, m.type, best_state[m.key], f"{m.ambush:.3f}", f"{m.speed:.3f}",
            f"{m.strain:.2e}", f"{m.okubo_w:+.2e}", f"{m.convergence:+.2e}",
            "-" if np.isnan(m.wet_fraction) else f"{m.wet_fraction:.2f}",
        )
    console.print(table)
    console.print(
        "\n[dim]okubo W < 0 is an eddy core (rotation-dominated); W > 0 is a seam "
        "(strain-dominated). `ambush` is how much faster the water within "
        f"{fishery.structure.ambush_radius_m:.0f} m is than the feature itself — "
        "the current-shadow signal.[/dim]"
    )
```

- [ ] **Step 2: Run it**

```bash
tidescout flow structure winyah-bay --regime mean_med --top 20
```

Expected: a ranked table. Sanity checks — **jetty features should rank high on ambush** (a jetty is the canonical current shadow), and **`okubo_w` should be negative for at least some features** (if every feature is a seam, `classify_structure` or the tensor is wrong).

- [ ] **Step 3: Ask the model the Georgetown question the notes actually pose**

```bash
python -c "
import json, numpy as np
from rasterio.warp import transform as warp_transform
from tidescout.config import load_fishery, load_known_spots
from tidescout.engine import activation, flow, structure
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import flowlib

slug, regime = 'winyah-bay', 'mean_med'
f = load_fishery(slug); spec = flowlib.grid_spec(slug, f)
spots = load_known_spots(slug)
meta = json.loads((fishery_data_dir(slug)/'flow'/regime/'grid'/'grid.json').read_text())
states = flow.tide_states(meta['stage_bc_m'])
xs, ys = warp_transform('EPSG:4326', f'EPSG:{f.bathymetry.epsg}',
                        [s.lon for s in spots], [s.lat for s in spots])
sel = {s.name: (spec.xs-x)**2 + (spec.ys-y)**2 <= 150.0**2
       for s, x, y in zip(spots, xs, ys)}
print(f\"{'spot':24s} {'state':6s} {'ambush':>8s} {'eddy%':>7s} {'seam%':>7s}\")
for i in range(len(meta['phases'])):
    st = flowlib.load_state(slug, regime, i)
    fl = activation.structure_fields(st['u'], st['v'], st['depth'], spec, f.structure)
    ug = structure.to_grid(st['u'], spec.flat_index, spec.shape)
    vg = structure.to_grid(st['v'], spec.flat_index, spec.shape)
    cls = structure.from_grid(
        structure.classify_structure(structure.gradient_tensor(ug, vg, spec.cell_m),
                                     f.structure.quiet_w), spec.flat_index)
    for s in spots:
        m = sel[s.name]
        if i % 6: continue
        print(f'{s.name:24s} {states[i]:6s} {np.nanmax(fl[\"ambush\"][m]):8.3f} '
              f'{(cls[m]==-1).mean()*100:6.1f}% {(cls[m]==1).mean()*100:6.1f}%')
" | head -30
```

Expected: **Georgetown Lighthouse should show high ambush contrast and a meaningful eddy share.** That is the current shadow Ellis describes, and it is the question the flood÷ebb ratio was never able to ask. Record the numbers in the carryover notes whatever they say — if the shadow is *not* there, that is a genuine finding about the model and belongs in Phase 2's opening, not a reason to adjust anything here.

- [ ] **Step 4: Commit and write the carryover**

```bash
git add backend/tidescout/cli.py
git commit -m "feat: tidescout flow structure -- derived structure per feature"
```

Then append a "Phase 1 results" section to `docs/superpowers/plans/2026-08-13-plan3-carryover-notes.md` recording: the Georgetown ambush/eddy numbers, the intertidal-share progression from Task 7 step 5, and the feature count and id stability from Task 8 step 7.

---

## Phase 1 Completion Checklist

- [ ] `make check` green; test count ≥ 226
- [ ] Plan 3 merged to `main`; `plan-04-phase1-structure` branched from it
- [ ] Oyster reef attributes attached to features; a minority of features carry reef (not all, not none)
- [ ] Boundary vertex, `store_sww` and Georgetown `works_on` all landed with their evidence in comments
- [ ] River inflow splits 78/13/8, and the total still matches the composite bucket
- [ ] `tidescout flow validate winyah-bay --regime mean_med` — North Jetty and Mud Bay unchanged from Plan 3
- [ ] `tidescout flow structure winyah-bay --regime mean_med` runs and ranks jetty features high on ambush
- [ ] Feature ids stable across two consecutive `tidescout features` runs
- [ ] Georgetown ambush/eddy result recorded in the carryover notes, whichever way it came out

**Then:** Phase 2 (`2026-08-16-04-phase2-salinity.md`) depends on Task 2 (inflow share) and Task 3 (grid round-trip). Nothing else here blocks it.
