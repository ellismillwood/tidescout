# Plan 3 ANUGA Spike — Measured Findings (2026-08-13)

De-risking pass run before authoring Plan 3, against real Winyah artifacts
(`data/winyah-bay/bathy_utm.tif`). Everything below is **measured on this Mac**
(Apple M5 Pro, 18 core / 24 GB, macOS 26.5, serial/single-core) unless labelled
an estimate. Numbers here supersede any earlier guesses.

## 1. Installation: the spec's #1 risk is gone

**ANUGA installs natively. The Docker/OrbStack contingency (spec §5, §13) is moot** —
and it was never free anyway: this Mac has no container runtime installed at all.

- `anuga 3.3.10` publishes prebuilt `macosx_26_0_arm64` wheels for cp310–cp314;
  `meshpy 2026.1` likewise. `uv pip install anuga` completes in seconds, no compiler.
- **No dependency conflict.** ANUGA requires `numpy>=2.0.0` and resolves to
  `numpy 2.5.2` / `scipy 1.18.0` — byte-identical to what `~/.venvs/tidescout`
  already pins. Installing `anuga` alongside the editable `tidescout` package
  changed no existing version. It can go straight into the project venv.
- Added transitively: `meshpy`, `netCDF4`, `pyproj`, `xarray`, `dill`, `matplotlib`,
  `pandas`, `cftime`.
- `WARNING: Could not import mpi4py - defining sequential interface` prints on every
  import. Harmless; serial interface works.

## 2. API facts worth writing into the plan

> **CORRECTION added 2026-08-14 (Plan 3 Task 9).** The spike scripts initialise
> `stage = np.maximum(elev + 1e-3, tide(0))`. That nominal 1 mm keeps land "just
> wet" and is harmless on the spike's smooth synthetic ramp, but it is **unsafe on
> real bathymetry**: on the Winyah mesh it films ~129,000 of 315,564 cells and
> collapses ANUGA's CFL timestep to ~1e-6 s inside the first yieldstep, aborting
> with "Too small timestep … even after 50 steps". Measured A/B on the real mesh:
> with the film, unstable; with `np.maximum(elev, tide(0))` — dry land genuinely
> dry — stable at dt ≈ 0.12 s, both with and without river inflows attached.
> **Dry cells must start at depth exactly 0.** Any future spike copied from here
> inherits the bug otherwise.

- `anuga.create_domain_from_regions(bounding_polygon, boundary_tags, maximum_triangle_area=None, interior_regions=None, interior_holes=None, hole_tags=None, breaklines=None, regionPtArea=None, minimum_triangle_angle=28.0, ...)`.
  There is **no `mesh_filename` argument** (an easy hallucination).
  `breaklines` and `regionPtArea` are available for channel-edge forcing.
- Tide forcing: `anuga.Transmissive_momentum_set_stage_boundary(domain=..., function=lambda t: ...)`
  — the function returns a scalar stage. Preferred over raw `Time_boundary`
  (which wants `[stage, xmom, ymom]`).
- `set_boundary` raises if you name a tag the mesh doesn't have; tags come from
  `boundary_tags` keys only.
- Velocity is not stored — derive it: `u = xmomentum / depth`, guarded on
  `depth > tol`, all via `get_quantity(...).get_values(location="centroids")`.
- `.sww` output carries both vertex and centroid (`*_c`) variables.

### Mass-conservation check — use tolerance 1e-3, not machine precision

The identity is:

```
get_water_volume() - V0  ==  get_boundary_flux_integral() + get_fractional_step_volume_integral()
```

Measured residual on a coarse wetting/drying mesh: **4.2e-4 relative**
(6.8e3 m³ out of 1.6e7 m³ moved). A `< 1e-6` assert **fails on a perfectly
healthy run** — this was hit in the spike. Set the automated check at 1e-3.

## 3. The model domain must be authored, not inferred

- Full water body at `z < +1.5 m`: **798.6 km²**, spanning the entire 38 × 50 km
  raster. Meshing that at 25 m is ~2.7 M triangles — infeasible.
- **Ocean and estuary cannot be separated by connectivity.** A barrier line across
  the bay mouth between the jetties leaves the estuary at 798.5 km² and the ocean
  at 0.0 — because the Atlantic reconnects through North Inlet and the ICW. Any
  "flood-fill the estuary" approach silently returns everything.
- Geodesic (along-channel) distance from the entrance does not size it either:
  the field spreads radially into the open Atlantic, so 10 km already encloses
  228 km², most of it ocean.
- **Therefore: the model-domain boundary belongs in `fisheries/winyah-bay.yaml`
  as an authored polygon**, exactly like the existing `jetties:` seeds. Where to
  put an open boundary is a modelling decision.

A working first draft (UTM 17N km, clockwise; 13 vertices) encloses **238 km² of
water** with all 409 jetty structure cells inside it:

```yaml
model_domain_utm_km:   # EPSG:26917
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
```

Known imperfection: the east edge still admits some open Atlantic near vertices
3–5. An early Plan 3 task should refine it against the chart/ENC. Pricing zones
by depth alone is a **trap** — the ocean is deep too, so `z < -4 m` initially
tagged ~33 km² of Atlantic as "channel" and inflated the mesh by 47%.

## 4. Shoreline meshing is NOT a risk (this surprised me)

The fear was that a fractal marsh shoreline would choke the triangle generator.
It does not, given cheap preprocessing:

- `binary_closing(3) → binary_opening(3) → fill_holes → largest component`,
  then `rasterio.features.shapes` → `shapely.simplify(25 m)`.
- Exterior ring: **7,581 vertices → 486** after simplification. Area preserved
  (261 km²).
- `create_domain_from_regions` then meshes **250k triangles in 0.5 s**.
- Elevation sampling onto 250k centroids by nearest-cell lookup: **<0.1 s**,
  0 nodata slivers.

Note: `binary_fill_holes` removes islands. Prefer **meshing over islands and
letting wetting/drying keep them dry** (they sit at z > 0) rather than carving
mesh holes — simpler and it lets islands flood on spring tides.

## 5. Cost is set by the SMALLEST triangle, not the triangle count

Measured, real Winyah bathymetry, one regime = 18.4 sim-h (12.42 h cycle + 6 h spin-up):

| grading | triangles | 1 run | 9 regimes |
|---|---|---|---|
| uniform 60 m | 256,803 | **1.12 h** | **10.1 h** |
| 60 m + jetty 15 m | 315,564 (+23%) | 5.46 h | 49.2 h |
| 60 m + jetty 12 m | 350,259 (+36%) | 8.49 h | 76.4 h |

**+36% triangles → +658% runtime.** ANUGA uses a global CFL timestep, so one
12 m triangle in a 3.8 km² jetty zone slows the entire 261 km² domain. Mesh
grading must be budgeted by *minimum edge length*, not cell count.

- **A synthetic-domain scaling fit of `wall ∝ N^1.65` (from 1.3k–75k triangles)
  over-predicts real cost by ~15× at 250k. Do not use it.** Uniform real meshes
  are far cheaper than that extrapolation; refined ones are dominated by the CFL
  effect instead. Measure, don't extrapolate.
- `domain.set_local_extrapolation_and_flux_updating(nlevels=8)` — ANUGA's local
  flux-update feature, which should target exactly this — produced **no speedup
  (92.5 s vs 92.6 s) and bit-identical output**. It did not take effect as
  invoked. Worth one more investigation in Plan 3 (ordering vs. other setup
  calls, or `compute_flux_update_frequency`), but do not budget on it.

### Practical budget

The jetty rips are a spec-mandated must-catch feature, so some fine zone is
non-negotiable. `60 m + jetty 15 m ≈ 49 h` for all nine regimes is a
**one-time overnight-times-two offline build** — acceptable for a precompute that
only reruns when bathymetry or config changes. Options if that is too slow:
fewer regimes (6 = 3 range × 2 discharge), shorter spin-up, or MPI.

**MPI is untested.** `anuga[parallel]` needs `brew install open-mpi` (no MPI
toolchain present) plus `mpi4py`/`pymetis`. ANUGA does pymetis domain
decomposition; ~4–6× on the 6 performance cores is the plausible ceiling.
Decide only after one real regime run.

## 6. Not yet established

These timing runs were 300 sim-seconds from rest — they measure **cost, not
physics**. Peak speed was 0.01 m/s, i.e. flow had not developed. Still open:

- Physical realism after full spin-up (ebb/flood reversal, channel speeds).
- Manning `n` field from `zones.tif` — and carryover trap (a): `zones()` shares
  `shallow_max_m`/`deep_min_m` with `detect_bars`, so retuning bar detection
  silently re-buckets the friction field. Freeze or split before tuning.
- Carryover trap (b): `WET_LEVEL_M = 0.0` is a module constant in `detect.py`;
  ANUGA introduces a time-varying free surface. Make it config-driven before the
  definitions fork.
- River inflow boundaries (Pee Dee / Waccamaw / Black) — untested; discharge
  bucket recalibration is still a Plan 1 carryover item.
- Rasterising `.sww` snapshots back onto the 10 m analysis grid.

## Reproduction

Spike scripts are throwaway (scratchpad, not committed). Environment:

```
uv venv --python 3.12 anuga-spike
VIRTUAL_ENV=$PWD/anuga-spike uv pip install anuga
VIRTUAL_ENV=$PWD/anuga-spike uv pip install -e backend
```
