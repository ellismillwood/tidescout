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

## Phase 1 results (Plan 4, `plan-04-phase1-structure`)

Tasks 1–11 complete, plus the merge-gate fix wave that closed the final whole-branch review: **265
tests green** (249 at the end of Task 11). `tidescout flow structure` is the new inspection CLI
(`backend/tidescout/cli.py`); everything below was produced by it or by the Task 3/11 diagnostics
built on `engine/activation.py` and `engine/structure.py`. Figures are post-fix-wave unless a
sentence says otherwise; the last section of this results block is the before/after for everything
the wave moved.

**Georgetown Lighthouse — the current-shadow question answered.** Plan 3 left this open: the
flood÷ebb ratio said Georgetown peaks on the ebb (contradicting `works_on: flood`), but also showed
the largest speed contrast of any known spot, with a minimum of exactly 0.000 m/s beside ~1 m/s
water. Task 1 of this plan re-read that contrast as an eddy signature and changed `works_on` to
`slack`. Task 11 asked the derived-structure fields directly: does the model produce a current
shadow (high ambush contrast, non-trivial rotation-dominated cell fraction) at Georgetown, the way
it plainly does not at a spot that has none? Over `winyah-bay`'s `mean_med` regime, all 26 phases,
within each spot's 150 m radius, **classifying seam/eddy from `activation.structure_fields`'s
`okubo_w`** (which is wet-masked twice — inputs then outputs — exactly as its docstring specifies,
guarding against the same dry-cell central-difference bridging artifact Task 9 was fixed to remove):

| spot | max ambush (m/s) | max eddy share | phases (of 26) with any eddy cells | max seam share |
|---|---|---|---|---|
| Georgetown Lighthouse | **0.868** | **7.6%** | **19/26** | 9.7% |
| North Jetty | 0.454 | 0.0% | 0/26 | 32.6% |
| Mud Bay Cut | 0.160 | 0.0% | 0/26 | 0.0% |

*(Round-1 review caught that the first pass of this table classified seam/eddy from raw `u`/`v` with
no wet mask, reintroducing exactly the dry-cell bridging artifact `activation.structure_fields`'s
docstring warns about and Task 9 removed. Unmasked, Georgetown's peak read 9.0% eddy / 20-of-26
phases / 11.8% max seam — measurably inflated: at the two eddy-richest phases, masking moves
Georgetown from 9.0%→6.9% and 6.9%→4.9%, and the domain-wide eddy-cell count at phase 0 drops from
1,545 (raw) to 780 (masked), roughly halving. North Jetty and Mud Bay Cut are unaffected — both have
zero dry cells anywhere in their 150 m discs across the whole cycle, so masking changes nothing for
them. The table above is the corrected, masked version and is the number of record; ambush is
untouched either way, since `structure_fields` already wet-masks it before this table ever sees it.)*

**The shadow is there, and it is the most pronounced structure of the three known spots.** Georgetown
shows the highest ambush contrast of any of them (nearly 2× North Jetty's) and is the *only* one of
the three that ever reads as rotation-dominated (Okubo-Weiss < 0) — 19 of 26 phases show some eddy
fraction within its disc, peaking at 7.6%. North Jetty, by contrast, never registers a single eddy
cell across the whole cycle (0/26) despite being the strongest, most consistent seam (up to 32.6%) —
a real current break, but a shear line, not a closed eddy. Mud Bay Cut shows essentially no derived
structure at all at any phase. This is independent confirmation, from a different signal than the
Plan-3 flood÷ebb ratio, that Task 1's `works_on: slack` reasoning for Georgetown was right: it is a
contrast/eddy spot, not a peak-current one, and the model was never wrong about it — the flood÷ebb
question was just the wrong question to ask of it.

One caveat surfaced while producing the table above, and it is what drove the fix wave's `eddy_share`
change. *(Every count in this paragraph and in the two numbered points below was measured at the
PRE-fix sampling anchor — 531 in-domain / 526 with a finite `okubo_w` — and is kept as measured
because it is the evidence the change rests on. The post-fix equivalents are in the SHIPPED block
that follows and in the before/after table at the end of this section.)* `engine/activation.py`'s
`sample_features` reduces `okubo_w` per feature with `np.nanmax` over the 150 m disc, the same
reducer used for `ambush`, `strain`, and `convergence`. Confirmed directly: of the **526** features
that ever record a finite `okubo_w` anywhere in the 26-phase cycle (2,162 total features in the inventory, but only 531
have any in-domain cell at all, and 5 of those never get a finite `okubo_w`), only 2 ever recorded a
negative per-feature `okubo_w` (both within 1e-6 of zero, inside the `quiet_w = 1e-5` dead band —
i.e. not meaningfully different from "quiet"). That two-fact pairing is doing less work than the
first-pass note here claimed, though. Two separate things are true and need to stay separate:

1. **`nanmax` demonstrably suppresses eddy signal in discs that contain one.** The single clearest
   case in the inventory: `creek_mouth-e3a7602b91aa` at phase 20 has 28 of its 179 disc cells
   classified eddy — 15.6% — on a disc that is **fully wet** (179 of 179 cells), so eddy-over-disc
   and eddy-over-wet are the same number and there is no denominator to misread. True in-disc min
   `okubo_w` = −1.48e-4, well past the quiet floor — yet `sample_features`'s `nanmax` reports
   `okubo_w = +1.89e-4` for it, the disc's strongest seam cell, not its eddy core.
   `classify_structure`/the tensor are not wrong; the max-reducer at the feature level is throwing
   the signal away for exactly this feature.
2. **But eddies are sparse enough that most discs never contain one to throw away, regardless of
   reducer.** The domain-wide wet-cell eddy rate (masked) is 0.105% on average across the cycle,
   0.199% at its peak phase, against a typical (median) disc size of 175 cells. Measured directly
   across all 526 evaluated features and all 26 phases (13,676 feature-phase samples), the mean
   eddy-cell count per disc is **0.28** — most individual feature-phases hold zero. Directly counted:
   only **104 of the 526** evaluated features (19.8%) ever have *any* cell crossing the `-quiet_w`
   eddy threshold in their disc across the whole cycle. For the other 422, `nanmax` isn't suppressing
   anything — there is simply nothing there to suppress.

Point 2 also has a sharp edge for whatever Phase 2/3 builds next: a naive `nanmin` field, compared
against 0 rather than against `-quiet_w`, would **not** cleanly recover an eddy signal either. Raw,
unthresholded `nanmin(okubo_w)` goes negative for **520 of the 526** features (98.9%) at some point
in the cycle — because `okubo_w` scatters within a few units of zero as ordinary floating-point/
dead-band noise in slack or otherwise featureless water, the same noise `classify_structure`'s
`quiet_w` dead band exists to absorb at the cell level (see that function's own docstring). Only the
104-feature eddy-containing subset above reflects genuine rotation-dominated content.

> **SHIPPED — this is no longer a Phase 3 item.** The final whole-branch review reversed the
> deferral: `classify_structure` had zero production callers even though the plan's Task 4 interface
> block specifies "Task 9 samples both `okubo_weiss` and `classify_structure` at feature geometry",
> and Phase 3's plan already constructs a test `FeatureMetrics(okubo_w=-1e-5)` that production cannot
> produce. So `engine/activation.py` now returns an **`eddy_share`** field from `structure_fields`
> and carries it on `FeatureMetrics`: a per-cell 0/1 indicator from
> `classify_structure(tensor, quiet_w)`, mean-reduced per feature. That IS the `-quiet_w` dead band
> reapplied at the feature level, so the prescription above is implemented, not outstanding.
> `okubo_w` is unchanged and remains the max-reduced SEAM channel.
>
> **Phase 3 consumes `eddy_share`, not a negative `okubo_w`.** The `_metrics` helper in
> `2026-08-16-04-phase3-bite-score.md` (around line 1074) predates the field and will need it added;
> `okubo_w = -1e-5` is not a value production produces.
>
> Measured after the change, same library and same 26 phases: **106 of the 527** evaluated features
> record `eddy_share > 0` at some phase — the same population the 104-of-526 count above identified,
> shifted by the sampling-anchor fix described below. Peak `eddy_share` anywhere in the inventory is
> 0.25. The reducer is the only thing that changed: of 13,614 finite per-feature `okubo_w` samples,
> still zero ever cross `-quiet_w`.
>
> **Denominator, stated once so it cannot be misread again:** `eddy_share` is eddy cells over the
> disc's **WET** cells, because a dry cell carries ANUGA's u = v = 0.0 and is not slack water — it is
> not water. The known-spot table above uses eddy-over-**DISC** (dry cells scoring False), which is a
> different statistic wherever a disc is partly dry. They agree on a fully wet disc. Georgetown's
> peak is 7.6% over the disc and 8.6% over its wet cells. `creek_mouth-97b7b83992ef` at phase 24, the
> example an earlier fix round retracted for calling 9/42 wet cells a disc fraction, reads
> `eddy_share = 0.214` under the shipped definition — the 21.4% figure was the right number wearing
> the wrong label.

Sanity check on jetty ranking (Step 2, `tidescout flow structure winyah-bay --regime mean_med`)
passed cleanly, and still passes on the post-fix-wave anchors: both `jetty`-type features rank in the
top 10% of the **529** in-domain features by ambush contrast — **24th** (0.536) and **51st** (0.392)
of 529, against a top-10% cut at rank 52. Median ambush across the 2 jetty features is **0.464**, the
highest median of any feature type in the inventory by more than 4×, ahead of `creek_mouth` 0.116,
`hole` 0.111, `wall` 0.100, `bar` 0.100, `dropoff` 0.044, `flat` 0.034. (`wall` and `bar` are
separated by 0.0003 — 0.09982 against 0.09950 — so they print the same at 3 dp. That is a tie for
every practical purpose, not a transcription error.) Jetties read as the canonical current shadow the
model predicts them to be.

The Task 11 version of this paragraph read 0.556 across the jetties and put them 8th and 48th of 531.
Those numbers were measured at a sampling anchor sitting **538.9 m** off North Jetty's own centroid,
so they describe the wrong piece of water; the before/after table below has the full accounting. The
finding — jetties top the inventory on ambush and sit comfortably inside the top 10% — is unchanged.

**Intertidal share progression (Task 7).** Across `winyah-bay`'s three range regimes (fixed at
`_med`/`_low`/`_high` discharge), intertidal cell share rises monotonically with tidal range, as
physically expected:

| regime | intertidal cells | share |
|---|---|---|
| neap_low | 42,645 | 7.26% |
| mean_med | 50,124 | 8.53% |
| spring_high | 57,796 | 9.84% |

Re-run post-fix-wave via `backend/tools/schedule_stats.py winyah-bay`: identical to the cell, as it
must be — nothing in the wave touches `wet_mask` or `schedule_from_depths`. The figures
`pipeline/schedule.py`'s own module docstring quotes reproduce with it (wet-window p50 = 0.523 in all
nine regimes; 33.8% neap / 35.7% spring draining in phase 0.0–0.2; spring_high's 4,037 cells at
`flood_phase == 0.0`, 2,697 of them — 66.8% — overlapping the early-drain bucket).

**Feature inventory count and id stability (Task 8).** The detector produces **2,162 features**
(256 bar + 134 creek_mouth + 847 dropoff + 497 flat + 389 hole + 2 jetty + 37 wall), all with
**2,162 unique ids** — no collisions — and two consecutive rebuilds from the same inputs produced
**identical** id sets (2,162 == 2,162, set-equal), confirming the hash-of-type-plus-centroid key
(Plan 3 carryover item 4, above) closed the `bar-78`-style renumbering trap for good.

**Oyster reef attachment (Task 10, for completeness).** 146 of 2,162 features (6.75%) carry mapped
SC oyster reef within 75 m — a clear minority, as expected (neither "nearly all" nor "none").
Reef *density* (`oyster_density` = `reef_area_m2_within / buffer_area_m2`, where the denominator is
the area of the feature's own 75 m buffer — **not** the feature's own area, which is what an earlier
version of this line said. That would be undefined for 136 of the 2,162 features: the 134 creek
mouths are Points and the 2 jetties are LineStrings, all with zero area. `oysters.py:106-107` states
the real formula correctly; this line did not) ranks the inventory differently from raw
reef *area*: the area-ranked top 5 is entirely large flats/holes (17,000–89,000 m²) with modest
density (0.013–0.066) that merely graze reef at their edges, while the density-ranked top 5 is a
different set of holes/flats carrying far less reef (3,700–14,800 m²) at meaningfully higher density
(0.092–0.139) — area alone is confounded by feature footprint, and density is the scale-free signal.
(The Task 11 version of this line gave the density top 5 as 3,000–7,000 m². Its 4th and 5th places
were tied at 0.09 under the old 2 dp rounding and were ordered by file position; at full precision
`flat-1387858bbedf` takes 5th. See the before/after block below.)

### What the final fix wave moved in the figures above

The merge-gate fix wave changed two things that feed these numbers, so the figures above are stated
as they were measured at Task 11 and are **not** rewritten in place. This block is the before/after.

1. **The sampling anchor.** `sample_features` used to centre each feature's 150 m disc on an
   unweighted mean of its exterior ring's vertices (counting the duplicated closing vertex twice),
   while `detect.feature_key` hashes the geometry's true centroid — so a feature's id and its metrics
   described different places, by a median 7.6 m and up to 726 m. Both are now the centroid.
2. **`oyster_density` precision.** It was rounded to 2 dp with every other float property, which put
   25 of the 146 reef-carrying features at exactly 0.0 and left the inventory with 13 distinct
   values. It is now written at 6 dp; nothing else about the computation changed.

**The known-spot table did NOT move.** Re-measured after both fixes, over the same regime and the
same 26 phases: Georgetown Lighthouse 0.868 max ambush / 7.6% max eddy share / 19 of 26 phases /
9.7% max seam; North Jetty 0.454 / 0.0% / 0 of 26 / 32.6%; Mud Bay Cut 0.160 / 0.0% / 0 of 26 /
0.0%. Identical to the digit. Known spots are sampled from their own lon/lat, not from feature
geometry, so the anchor fix cannot reach them — and the Georgetown conclusion stands untouched.

**What did move, all of it feature-level:**

| figure | Task 11 | after the fix wave |
|---|---|---|
| features with any in-domain cell | 531 | 529 |
| features with any finite `okubo_w` | 526 | 527 |
| features ever containing an eddy cell | 104 (19.8%) | 106 (20.1%) |
| median ambush, `jetty` | 0.556 | 0.464 |
| median ambush, `hole` / `wall` / `bar` | 0.104 / 0.093 / 0.084 | 0.111 / 0.100 / 0.100 |
| median ambush, `flat` | 0.037 | 0.034 |
| jetty ranks by ambush | 8th and 48th of 531 | 24th and 51st of 529 |
| `oyster_density`-ranked top 5, reef area | 3,134–7,340 m² | 3,689–14,812 m² |
| tests green | 249 | 265 |

`creek_mouth` (0.116) and `dropoff` (0.044) are unchanged, and the qualitative claims all survive:
jetties still hold the highest median ambush of any type by a factor of four, both jetty features are
still inside the top 10% (the cut is rank 52 of 529; they land 24th and 51st), and flats and
dropoffs still sit last.

The jetty median is the single largest move and it has a specific cause worth recording: North
Jetty's LineString has five unevenly spaced vertices over 3.2 km, so its vertex mean sat **538.9 m**
from its length-weighted centroid — the disc was being sampled most of a kilometre from the point its
id names. Its ambush reads 0.720 under the old anchor and 0.536 under the new one. The other jetty's
anchor moved 56.4 m and its ambush is unchanged at 0.392.

Two features (`flat-2f450ef65593`, 27 cells, and `bar-6c7aacb7308d`, 3 cells) had their centroid fall
outside the flow domain and now report `n_cells = 0`; both were marginal edge features.

Of the 529 evaluated under both anchors, comparing each feature's metric **at its own best-ambush
phase**, the value differs at all — exact inequality, any bit — for **450** (`speed`), **127**
(`ambush`), **136** (`strain`), **139** (`okubo_w`), **116** (`convergence`) and **50**
(`eddy_share`).

*(An earlier version of this line gave 368 / 101 / 111 / 113 / 93. Those counts came from the same
harness with a `|Δ| > 1e-12` absolute floor, which the words "changed at all" do not describe, so the
exact-inequality figures above are the ones of record. The gap is not noise in the fix — it is
features whose values are themselves at the 1e-14 scale, i.e. water that is numerically motionless.
For `speed`, all 82 of the suppressed features have a best-phase speed below 1e-13 m/s. Both
definitions are stated so a re-measurement can land on either without looking like a contradiction.)*

The 79 features whose `speed` is bit-identical across the two anchors are mostly the ones whose
anchor could not move: 52 of them are `creek_mouth`, and every creek mouth is a Point, whose centroid
is itself.

The `oyster_density` top-5 change is a tie-breaking artifact, not a ranking change: at 2 dp the
fourth and fifth places were tied at 0.09 and were ordered by file position. At full precision
`flat-1387858bbedf` (0.092272) beats `hole-3584c46792e2`. The "0.09–0.14" band and the
"large flats top the area ranking, small holes top the density ranking" conclusion both hold.

**A third fix, `flood_phase`, has no figure above to move.** `sample_features` reduced it with an
ordinary median of a circular quantity; it is now a circular mean. Of the 301 features carrying a
finite `flood_phase`, 261 move at all, 19 by more than 0.05 of a cycle and 3 by more than 0.1. Those
three, largest first: `hole-b7ff6b2a8891` 0.443 → 0.706 (Δ 0.264), `hole-ce5d7d265ad7` 0.523 → 0.694
(Δ 0.171), `flat-b6a1aec2d79d` 0.0 → 0.884 (Δ 0.116).

The biggest is the least informative — `hole-b7ff6b2a8891`'s 55 flood phases are spread right around
the cycle (resultant length 0.05), so neither the old statistic nor the new one means much for it,
and `engine/activation.py`'s `_circular_mean_phase` documents that degenerate case. The clearest is
the smallest: `flat-b6a1aec2d79d`'s median read 0.0 — "floods exactly at low water", the first
instant of the flood half — for a cluster whose circular centre is 0.884, late on the ebb. That is
the cut-point artifact itself rather than a shift in degree. (An earlier version of this paragraph
called `flat-b6a1aec2d79d` the largest correction; it is the smallest of the three.)

