# Plan 3 Carryover Notes (for Plan 4 authoring)

Distilled from Plan 3's ledger, the 2026-08-15 handover, and the four-variant instability round.
Branch `plan-03-anuga`. Tasks 1–13 complete; 193 tests green.

## The headline

Plan 3's whole cost was one hydrodynamic instability that consumed **three library builds and
roughly 17 hours of compute**. It had two apparent causes and one real one:

1. **Cause #1 (real, fixed):** `mesh.classify_boundary` decided ocean-vs-wall on **bed depth
   alone**. Correct at the seaward opening, nonsense wherever the domain polygon severs a deep
   inland channel — a river channel 40 km upstream is indistinguishable from open coast by depth.
   Four segments at the Pee Dee head were tagged `ocean`, imposing the full ocean tide 40 km
   inland through a ~150 m slot in an otherwise solid reflective wall. A jet generator.
   Fixed by a three-class scheme (`ocean` / `wall` / `open`) driven by an **authored polygon**,
   `ocean_boundary_utm_km`, because depth cannot classify geography.
2. **Cause #2 (never existed):** with cause #1 fixed, build #3 failed instead on the east ocean
   apron. That was entirely an artifact of mapping `open` → `anuga.Transmissive_boundary`, which
   also collapsed the timestep 10× (0.016–0.020 s vs ~0.2 s) and pushed per-regime wall time from
   2.7 h to ~14 h. Reverting `open` → `Reflective_boundary` (`cebfb5a`) removed both.
3. **The seaward boundary was never implicated at all.** Proven by running a variant with
   `ocean` → `Time_boundary([stage,0,0])` in parallel; it completed just as cleanly as production's
   `Transmissive_momentum_set_stage_boundary`.

**Round 1 result (2026-08-15):** all four variants ran the full 18.42 sim-h (t = 66,312 s), the
first time any regime had ever completed. Worst values across all 75 samples of all four runs:
mass residual 6.5e-13 – 2.5e-11 (tolerance 1e-3); `dt_max` never below 0.195 s; max 4,624 steps
per 900 s yieldstep against a ~4,400 baseline; zero cells over 3 m/s in any sample; hot-cell
boundary adjacency max 0/1/1 on ocean/open/wall.

## What Plan 4 inherits

- **A working flow library**: 9 regimes × 26 phases, rasterised to a 20 m grid masked to the
  domain — 587,325 in-domain cells covering 234.9 km², 7.05 MB per phase, 737 MB total on disk (compressed).
- **`engine/flow.py`**: `select_regime` (weighted fallback + flag), `bracket_phases` (cyclic),
  `interpolate_state`, `tide_states`, `speed_direction`, `wet_mask`. Pure, no I/O.
- **`pipeline/flowlib.py`**: `grid_spec`, `rasterise_regime`, `load_state`, `shear_magnitude`.
- **The validation gate**: `tidescout flow validate <slug> --regime <name>`.

## Conventions Plan 4 must not get wrong

- **Phase 0 is LOW water, not high.** `phase` is measured from the end of spin-up, and
  `spin_up_h / cycle_h` = 6.0 / 12.42 = **0.4831 of a cycle**. Measured against the shipped config:
  phase 0.000 → −0.547 m, 0.250 → −0.058, 0.500 → +0.547, 0.750 → +0.058. **Flood is the FIRST
  half of the cycle.** Anything that assumes otherwise inverts ebb and flood everywhere. Use
  `engine.flow.tide_states`, which reads it off the recorded `stage_bc_m`, rather than re-deriving.
- **Shear is the strain rate**, `sqrt((du_dx−dv_dy)² + (du_dy+dv_dx)²)` — not the raw velocity-
  gradient magnitude, which returns 1.41ω for solid-body rotation and 1.41k for isotropic
  expansion. Neither deforms a parcel, so neither is a seam; the naive form lights up the interior
  of every eddy as holding water.
- **`np.gradient` returns the ROW derivative first**, and rows run south on a north-up raster while
  u/v are true east/north. Naming the row derivative `du_dy` without a sign flip silently turns
  `du_dx − dv_dy` into the divergence.
- **Regime fallback is not symmetric.** One range step rescales the whole tidal forcing
  (RANGE_FACTORS 0.72/1.0/1.28 on a 1.10 m mean range, ~15 cm of amplitude); one discharge step
  moves domain-mean depth ~1 cm. `RANGE_STEP_COST=3` vs `DISCHARGE_STEP_COST=1` guarantees range
  is never traded for discharge.

## ANUGA facts worth not rediscovering

- Mass identity: `get_water_volume() − V0 == get_boundary_flux_integral() +
  get_fractional_step_volume_integral()`. Config tolerance 1e-3; a correct boundary closes at 1e-13.
- `create_domain_from_regions` has **no `mesh_filename`** argument.
- `set_boundary` raises on any tag the mesh lacks — omit empty tags rather than passing empty lists.
- Omitting `hole_tags` makes ANUGA tag island holes `interior`, which then crashes `set_boundary`.
- Elevation must be set at **vertices**, not centroids — DE0 is a discontinuous-elevation scheme.
- Initial stage must be `max(elev, level)`, never `max(elev + eps, level)`: a 1 mm film over
  ~129,000 land cells collapses the timestep.
- `Inlet_operator(domain, region, Q=...)` — verified. Dies with an unrelated `AttributeError` on an
  empty region, and does **not** raise at all if the region contains only land centroids.
- `domain.number_of_steps`, `recorded_min_timestep`, `recorded_max_timestep` all **reset at each
  yieldstep**, so they describe the interval just completed. `steps` is the honest cost signal;
  instantaneous `dt` at the yield instant is not — it shows transient excursions (8e-3) that the
  mean never sees.
- These knobs are **inert under DE0**: `minimum_allowed_height`, `maximum_allowed_speed`,
  `minimum_triangle_angle` — all produced bit-identical output.

## Performance, measured on this Mac (M5 Pro, 6 performance cores, 247,020 triangles)

| configuration | per 900 s yieldstep | per 18.42 sim-h regime |
|---|---|---|
| 1 process | 144 s | ~2.96 h |
| 4 processes | 176 s (1.22× slowdown) | 3.21–3.35 h |
| 9 processes (build #4) | ~200 s | 236.5–246.9 min (3.94–4.12 h) |

The machine is **mildly memory-bandwidth contended, not core-starved** — 4 processes deliver ~3.3×
the throughput of one. Consequences:

- **MPI is not worth it for this workload.** Nine independent regimes have zero communication, so
  process parallelism cannot be beaten on throughput; MPI's only structural gain is the ragged tail
  (9 regimes on 6 workers is two waves whose second is half-empty). **Oversubscribing to 9 workers
  captures that same gain for free** — ~4.6 h against ~7 h — with no `open-mpi`/mpi4py/pymetis
  toolchain and no perturbation of the numerics. MPI's real value would be single-run *latency*
  for debugging (2.7 h → ~40 min); running four hypotheses at once buys that back instead.
- `anuga.max_workers` defaults to 6. Consider raising it to 9 for this fishery.

## Owed / open items

1. **`run_regime` never calls `set_quantities_to_be_stored(None)`**, so ANUGA writes a full `.sww`
   per regime alongside our `snap_*.npz` — **1.6 GB** across the build, against 548 MB of
   `snap_*.npz` and 737 MB of grid output. Harmless on 746 GB free and
   possibly useful for visualisation, but unrequested. Decide: keep for viz, or disable.
2. **Discharge-axis check** (review Important 3): the Pee Dee `Inlet_operator` sits 2,456 m
   downstream of `open` segment 7, so injected discharge could leave through the boundary above it.
   `cebfb5a` may already fix this as a side effect — a reflective wall cannot pass water out.
   **RESULT: the axis is real.** Depth rises monotonically with discharge at every inflow in
   both range buckets tested. Pee Dee +10.9 mm (mean) / +12.6 mm (spring) low→high, Waccamaw
   +12.5 / +14.3, Black +9.8 / +10.9, against a domain-wide +3.2 / +3.3. The per-river response
   is 3–4× the domain-wide one, which is the correct physical shape; a leaking axis would have
   collapsed the river rows toward the domain row. The Pee Dee — the specific concern, since its
   inlet sits 2,456 m downstream of `open` segment 7 — is mid-pack, not weak. `cebfb5a` closed
   that path as a side effect: a reflective wall cannot pass water out. **CLOSED.**
3. **The `open` tag is not only river heads.** Its largest member is a single **3,608 m** segment at
   (662578, 3672160) — the southern approach — now dammed by `Reflective`. Defensible at a river
   head where discharge enters downstream anyway; a stronger liberty across genuinely open water.
   It did not destabilise anything, but it is a physical assumption Plan 4 should revisit.
4. **Feature-id rebuild stability** (Plan 2 trap (c)): `bar-78` renumbers on rebuild. Nothing
   persists a feature reference until Plan 4 scoring, which is when a hash-of-type-plus-centroid key
   must land. **Plan 4 must not skip it.**
5. **The instrumented harness lives in a gitignored directory** (`.superpowers/sdd/.gitignore` is
   `*`). It is the tool that cracked cause #1, and it has already been lost and rebuilt once — the
   surviving copy had also silently drifted to a two-tag boundary map that would crash on the
   current three-tag mesh. **Promote it into the repo.**
6. **SCDNR oyster layer** (`data/winyah-bay/oyster_reefs.geojson`, 8,451 polygons, median reef
   24 m²) is a Plan 4 *scoring/habitat* layer — not mesh geometry, not an ambush-feature class.
7. **NWI wetlands**: 82.4% of mapped Spartina marsh is classified intertidal by the model, only
   4.7% clearly wrong. **The DEM's marsh boundary is sound — do not re-litigate it.**
8. **Deferred to Plan 4 by the plan's own self-review**: derived structure beyond shear (lee/eddy
   zones, convergence at draining creek mouths, flats flood/drain schedules); salinity (spec §7).

## The validation gate result (Task 13) — OPEN, needs Ellis

Build #4: 9 regimes, 236.5–246.9 min each, **4.13 h wall total** at 9 workers against the plan's
~11 h estimate. Mass residuals 2.0e-15 – 1.2e-14. All nine reverse. All three known spots are
IN DOMAIN, so the east-edge polygon concern (Task 6 step 7) does not bite.

Flood peak ÷ ebb peak within 150 m of each spot, across all nine regimes (>1 = flood-dominant):

| Spot | Expects | Ratio range | Contrast (max−min) | Read |
|---|---|---|---|---|
| North Jetty | flood | **1.17 – 1.23** | 0.56 – 0.78 | Clean pass, every regime |
| Mud Bay Cut | ebb | 0.92 – 1.01 | 0.28 – 0.41 | Correct lean, weak margin |
| Georgetown Lighthouse | flood | **0.88 – 0.92** | 0.76 – 1.12, min 0.000 | Consistently EBB-dominant |

- **North Jetty passes unambiguously** in all nine regimes. The jetty mesh refinement is resolving
  the structure it exists for.
- **Mud Bay Cut leans ebb correctly** in neap and mean, but ties at spring (0.411 flood vs 0.406
  ebb — a 1% margin). The CLI's binary verdict therefore flips between regimes on noise. The
  direction is never wrong, only weakly expressed. **Do not "fix" this by widening the radius or
  changing a threshold** — the plan explicitly forbids tuning the gate to pass.
- **Georgetown Lighthouse is a genuine disagreement.** The model puts peak current on the EBB in
  all nine regimes, ~10% margin, consistent — not marginal. BUT it also shows by far the largest
  contrast of any spot, with a minimum of exactly 0.000 m/s beside water moving ~1 m/s: a fully
  stagnant pocket adjacent to the fastest water there. That IS "the mouth of the bay creates a
  break in current which carries down the north bank". So the model reproduces the STRUCTURE the
  notes describe and disagrees only about which half of the cycle carries peak current.
  Ellis's notes originally said "slack AND early incoming"; he chose `flood` when asked. The
  contrast-based branch the CLI uses for `slack` would pass this spot overwhelmingly. A spot that
  fishes on early incoming does not require peak current on the flood — it requires the break to
  exist, and the break is there. **Whether this is the gate asking the wrong question of an eddy
  spot, or the model genuinely wrong about Georgetown, is Ellis's call and is NOT resolved.**

Physical sanity checks that passed alongside: peak speed is ordered spring > mean > neap at every
spot in every comparison, and the discharge axis barely moves speed (as predicted — it moves depth,
~1 cm locally). Both are independent evidence the range forcing genuinely drives the model.


## Process lessons that actually cost time

1. **A crash dump shows the first cell to FAIL, never the thing that has been GROWING.** Two root
   causes were declared from crash dumps and both were wrong. The instrumented time series found
   the real one in a single run. **Instrument first.**
2. **Run hypotheses in parallel.** Six cores sat idle while diagnostics ran serially, repeatedly,
   after this was pointed out. Round 1 ran four variants at once for the same wall time as one and
   resolved the blocker plus two open questions in a single 3.3 h pass.
3. **Weigh a reviewer's evidence above your own first-principles reasoning.** The review warned
   `Transmissive_boundary` appears in no ANUGA validation script and is weakly ill-posed for
   subcritical inflow. Under-weighted in favour of "it is the only variant that imposes neither
   level nor momentum" — it cost the overnight build, and the reviewer was right on both counts.
4. **Depth alone cannot classify geography.** It failed for the domain polygon and again for the
   ocean boundary. Both had to become authored config. **Assume a third instance exists.**
5. **Verify a "root cause" with a controlled experiment before acting on it.** The no-inflow control
   settled in one run what two five-hour builds could not.
6. **Task briefs are not authoritative.** Four defects were found in the Tasks 11–13 reference
   implementations, three of which would have produced plausible wrong answers rather than crashes:
   the shear formula, the gradient orientation, the regime-fallback weighting, and an ebb/flood
   inversion in the validation gate. Each is now pinned by a test.
7. **The Mac sleeps and kills sessions.** Hold it awake for the whole run: `caffeinate -dimsu -t N`,
   sized to the job — an 8 h window would have expired mid-build #4.
