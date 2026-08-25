# Tidal phase — gate report

**Plan:** `docs/superpowers/plans/2026-08-24-tidal-phase.md` (Task 4, the gate)
**Spec:** `docs/superpowers/specs/2026-08-24-tidal-phase-and-ocean-endmember-design.md`
**Branch:** `plan-06-tidal-phase` (PR #6, merged) | **Date:** 2026-08-24
**Predecessor:** `docs/superpowers/plans/2026-08-24-salinity-anchoring-gate-report.md` (PR #5)

*What carrying tidal phase into the salinity fit actually changed.*


**This task measures and reports. It does not decide.** No value in
`fisheries/winyah-bay.yaml`'s `salinity:` block was touched, `ocean_ppt` was not freed, and
`ocean_boundary_utm_km`, the ANUGA mesh, the flow library, and `ON_AXIS_MAX_KM` were not touched.
Everything below is a live re-run against real data, captured 2026-08-24 on branch
`plan-06-tidal-phase`.

Reproduce with:
```
tidescout salinity calibrate winyah-bay
```
Full captured output: `/tmp/calibrate-phase.txt` (1,617 lines) — this is the run captured AFTER the
regression fix in §6; see that section for why a first pass of this report needed correcting before
being finalized.

## Bottom line

The phase wiring in `fit_intrusion` is **correctly wired but a small effect**. WQP's rmse moved
exactly the direction and rough scale the plan predicted (down 0.319 ppt, 5.2% of its own value),
and NERRS's rmse — which the plan required to stay put because those rows were already at the
correct phase — moved by only 0.0018 ppt, three orders of magnitude below NERRS's own 4.06 ppt
rmse, which is what "didn't move" means here. Both halves of the wiring check pass.

But the arithmetic that matters — how much of WQP's *excess* rmse over NERRS this closes — says
the defect was smaller than the framing in the previous gate report implied. Phase closes **15.5%**
of the 2.041 ppt gap between WQP and NERRS, leaving **84.5%** (1.723 of 2.041 ppt) still open. The
previous gate's Jensen's-inequality argument for a systematic phase bias was real and is now
confirmed wired correctly — but it was never a claim about *how much* of the gap phase explained,
and the aggregate number says: not most of it. **`fitted` stays False, and the binding cause is
unchanged from the previous gate: model form (`POOR FIT`), not coverage.** The score-spread probe
at the two spots nearest the mouth — the ones a fishing recommendation depends on most — did not
narrow.

**One regression, found and fixed in the course of writing this report (§6):** `station_bias`, the
per-station diagnostic table `tidescout salinity calibrate` prints, kept scoring every row at
`FIT_PHASE` even after `fit_intrusion` itself started using real per-row phases — so its printed
table for all 56 WQP distances silently stopped matching the fit it sat beside. Fixed, tested with
a deliberate-break check, and re-verified that no `fit_intrusion` number (the ones this report's
headline conclusions rest on) moved as a result. Details and evidence in §6.

## 1. Headline comparison

| metric | BEFORE (PR #5, no phase) | AFTER (measured 2026-08-24) |
|---|---:|---:|
| observations | 12,725 | **12,725** |
| WQP grabs given a real phase | 0 | **1,860** |
| daily means at FIT_PHASE | 12,725 | **10,865** |
| grabs excluded, no phase | n/a | **0** |
| rmse overall | 4.4189 | **4.3542** |
| **rmse WQP** | **6.1020** | **5.7830** |
| rmse NERRS | 4.0614 | **4.0596** |
| `l0_km` | 13.33 | **13.203** |
| `front_width_km` | 14.68 | **14.560** |
| condition number | 7.94 | **8.058** |
| `fitted` | False | **False** |

Exact figures, straight from `fit_intrusion`'s diagnostics on this run:

```
rmse_ppt: 4.354236719156845
rmse_by_source_ppt: {'nerrs': 4.059553283050965, 'wqp': 5.782954764145098}
n_obs: 12725              n_phase_supplied: 12725          n_interior_obs: 8667
cfs_span: (715.2, 183660.0)                                n_dropped: 0
n_swing_obs: 10865         swing_rmse_ppt: 5.091632781651671
n_distinct_distances: 58   distance_span_km: 28.497082233428955
n_distinct_discharges: 3829
fitted_params: ['l0_km', 'k', 'front_width_km', 'excursion_km']
param_sigma: {'l0_km': 0.09698, 'k': 0.01403, 'front_width_km': 0.19368, 'excursion_km': 0.07635}
condition_number: 8.057548319225928
at_bounds: []
```

`l0_km` full precision: 13.202689974552039. `front_width_km`: 14.560081611273919.
`k`: 0.6191930720001856. `excursion_km`: 5.527992438437598 (all fitted, none at a bound).

The CLI's own text confirms the phase split without any extra instrumentation:

```
12725 salinity observations, 10865 tidal-swing observations.
1860 of those 12725 salinity observations carry a tidal phase individually
resolved from their own WQP grab timestamp; the remaining 10865 (NERRS/USGS
daily means) score at the fixed daily-mean FIT_PHASE...
```

No `[yellow]... WQP grab(s) excluded -- no tidal phase could be determined...` line printed,
because that block is conditional on a nonzero count; confirmed directly
(`CalibrationInput.n_no_phase == 0`) rather than inferred from its absence. Every WQP grab that
survived the earlier axis/co-location/discharge-day screens got a real, individually-resolved
phase — none were silently defaulted or silently dropped for lacking one.

**No code change was needed to produce any number in this table.** Every figure above came
straight out of `tidescout salinity calibrate winyah-bay`'s existing, already-tested output
(`fit_intrusion`'s diagnostics dict and `CalibrationInput`'s counters, both wired by Tasks 1-3 of
this plan) and is unaffected by the §6 fix — confirmed by re-running the full calibration after
that fix and diffing the diagnostics block byte-for-byte against the pre-fix run (identical; see
§6). A code change WAS needed elsewhere, for the per-station bias table in §3/§6 below, which is a
separate diagnostic fed by different code (`station_bias`, not `fit_intrusion`) — see §6 for what
changed and why the numbers here were unaffected by it.

## 2. The falsifiable check

The plan's own test of whether the wiring is correct, not merely green: **NERRS rows were already
scored at the correct phase, so their rmse was required NOT to move. WQP was required to move, and
did.**

- **NERRS: 4.061384904005473 → 4.059553283050965 ppt. Moved by 0.0018 ppt** (a 0.045% relative
  change) — down, not up. This is not zero because NERRS rows are not independent of WQP rows in
  the fit: `l0_km`, `k`, `front_width_km`, and `excursion_km` are shared free parameters, fitted
  jointly across both sources. Correcting WQP's phase moves the optimizer's landing point
  slightly, and that small shift changes what the model predicts at NERRS's own two distances too
  — a second-order effect through the shared optimum, not a first-order effect on NERRS's own
  scoring (NERRS rows still evaluate at the same fixed `FIT_PHASE = 0.25` they always did; nothing
  about how they are scored changed). 0.0018 ppt is three orders of magnitude below the 4.06 ppt
  NERRS rmse itself — this is "didn't move," read correctly.
- **WQP: 6.10203118406574 → 5.782954764145098 ppt. Moved by 0.319 ppt** (5.23% of its own value,
  the direction the plan predicted) — down, meaning the model now reproduces WQP's own real grab
  values better once each is scored at the phase it was actually taken at, rather than the
  daily-mean phase it never was.

**Verdict: the wiring is correct.** Both halves of the falsification test behaved exactly as
required — the population whose scoring didn't change (NERRS) didn't move, and the population
whose scoring did change (WQP) did. `swing_rmse_ppt` (5.081 → 5.092 ppt, +0.010) is consistent with
this too: `_swing` is documented as unaffected by the `phases` argument at all (a swing is
high-water-minus-low-water, evaluated at fixed phases 0.0/0.5 regardless), so its own near-zero
movement is the same shared-optimum second-order effect as NERRS's, not a sign anything leaked
into swing scoring.

## 3. The arithmetic that matters most

WQP's rmse exceeded NERRS's by **2.041 ppt** before this work (6.10203118406574 −
4.061384904005473 = 2.040646280060267). That excess is the number a phase defect would be expected
to close, if phase were the dominant reason WQP scored worse than NERRS.

```
excess before  = wqp_rmse − nerrs_rmse = 6.10203118406574 − 4.061384904005473 = 2.040646280060267 ppt
excess after   = wqp_rmse − nerrs_rmse = 5.782954764145098 − 4.059553283050965 = 1.723401481094133 ppt
excess removed = 2.040646280060267 − 1.723401481094133 = 0.317244798966134 ppt
              = 15.5% of the pre-existing excess
excess remaining = 1.723401481094133 ppt = 84.5% of the pre-existing excess
```

**Phase correction closed 15.5% of the WQP-vs-NERRS gap — 0.317 of 2.041 ppt. 84.5% (1.723 ppt)
remains, and is not phase.**

**What that implies, plainly:** the previous gate report's Jensen's-inequality argument (`tanh`'s
curvature means scoring a grab at the wrong phase is a systematic bias, not unstructured noise) was
correct as a mechanism — Section 2 above confirms WQP moved the predicted direction by the
predicted rough scale. But that argument was never quantified against the *aggregate* gap, and now
that it is, phase is a minority contributor. Reading "the model's own tidal swing at WQP distances
is 8.3–12.3 ppt, so a grab could be off by 4–6 ppt from phase alone" as "phase explains most of the
gap" — which is not what the previous report claimed, but is the natural way to over-read that
sentence — would have been wrong. The honest reading is that phase was real, correctly identified,
correctly fixed, and modest.

The remainder is most plausibly two things, both measurable from data already in this run rather
than speculation:

1. **WQP samples 56 distinct along-estuary distances against NERRS's 2**, and disproportionately at
   distances the model fits worst. The per-station bias table (`tidescout salinity calibrate`'s own
   output, now correctly phase-scored per row — see §6) shows the near-mouth WQP stations (4.4–13
   km, the reach nearest the fishing spots) still carrying rmse values of 5.6–12.6 ppt against the
   fitted model — `21SC60WQ_WQX-RO-23321` at 10.327 ppt (n=7), `RO-18423` at 12.594 ppt (n=9),
   `RO-01108`/`RO-01113` (single readings) at 8.911 and 8.109 ppt — while NERRS's two stations sit
   at 3.99–4.19 ppt. NERRS's 2 points constrain nothing about the profile's *shape* (2 points cannot
   separate `l0_km`, `front_width_km`, and `k`); WQP's 56 points probe the actual shape, including
   the near-mouth reach where a single-axis depth-averaged tanh profile is measured to fit worst.
   A population that samples the model's weak spots will always show a higher aggregate rmse than
   one that sits at 2 comparatively well-fit anchors, independent of any phase question at all.
   (Before §6's fix, this table's per-station numbers were somewhat inflated by scoring every WQP
   row at the wrong phase — e.g. `RO-23321` printed 13.878 ppt, `RO-18423` printed 13.695 — but the
   correction changed individual rows up or down by a few ppt each, not the shape of this argument:
   near-mouth WQP stations are still, after the fix, several times NERRS's rmse.)
2. **A single grab carries noise a 96-reading daily mean averages away.** A NERRS daily mean is the
   average of ~96 fifteen-minute readings across a full tidal cycle; naive noise averaging would
   damp its variance by roughly `sqrt(96) ≈ 9.8x` relative to any one of those readings. A WQP grab
   is exactly one reading, with no such averaging, so even after its systematic phase bias is
   removed, its irreducible measurement/sampling noise floor sits far above a daily mean's. This is
   visible in the raw resolution numbers: WQP's own observation resolution is 0.0543 ppt against
   NERRS's 0.00296 ppt — WQP's data is 18x coarser-grained on its face, before any question of fit
   quality. Even after phase correction, WQP's rmse still sits at **106x its own resolution**
   (5.782954764145098 / 0.054307116104868915), a `POOR FIT` by a wide margin on its own terms, not
   only relative to NERRS.

Both of these are structural properties of *where WQP samples and how it was collected*, not of the
tidal-phase defect this plan set out to fix. They are not new problems this task discovered; they
are the previous gate's own "the model reproduces the estuary no better and no worse than it ever
did" conclusion, now with the phase confound subtracted out so it can be stated with more
confidence.

**If the honest reading deflates the previous framing, say so:** yes. The previous gate's own
falsification argument rested on the NERRS-only rmse (4.061 ppt, phase-correct even then), not on
the WQP figure — so the plan's central claim (the model does not reproduce the estuary, on data
that was always trustworthy) is untouched by this result. What this result does deflate is any
expectation that phase-correcting WQP would materially change the *combined* picture or the score
at the spots nearest the mouth (see §4) — it closes a real but minority share of one specific,
correctly-identified defect, on a population that was already known to carry the more severe
`FEW OBSERVATIONS`-per-station and near-mouth-model-misfit problems described above.

## 4. Score-spread probe re-run

**Which question this probe answers: dependence on modelling choices, not parameter uncertainty.**
The fit's own `param_sigma` (§1) reports `front_width_km`'s 1-sigma as 0.194 km against a fitted
value of 14.56 km — a tight, reassuring-looking number describing how much `front_width_km` could
move under the *statistical noise of this exact dataset, at its actual optimum*. This probe asks a
different question: if the model's true front width were actually one of several other physically
plausible values instead — because the model FORM is wrong, which `POOR FIT` already says it is —
how much would the predicted salinity, and the trout score derived from it, change at each spot?
That is a sensitivity-to-modelling-choice question, not a confidence interval, and conflating the
two is exactly how a tight `param_sigma` gets over-read as "the model is well-determined here."

Same protocol as the previous gate's probe: hold `front_width_km` at 5, 8, 12, 16 km and refit
`l0_km`, `k`, `excursion_km` against the current (phase-corrected) dataset — 12,725 level rows using
each row's own resolved phase, 10,865 swing rows. Evaluated at `cfs=q0_cfs=4000` (the config's own
median-flow reference) and `phase=FIT_PHASE=0.25`, matching the previous probe's evaluation point
exactly so the two runs are comparable:

```
front_width_km held  5.0 -> l0_km 16.8258  k 0.2078  excursion_km 1.9839   rmse 5.7101
front_width_km held  8.0 -> l0_km 15.7807  k 0.3157  excursion_km 3.1328   rmse 5.0320
front_width_km held 12.0 -> l0_km 14.2664  k 0.4908  excursion_km 4.6193   rmse 4.7427
front_width_km held 16.0 -> l0_km 12.5790  k 0.6962  excursion_km 6.0186   rmse 4.7167
```

Trout curve (Phase 3 plan, not yet implemented in code — applied here as a standalone piecewise
linear interpolation for this measurement only, identical to the previous gate's):
`x=[0.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0]`, `y=[0.05, 0.10, 0.45, 0.85, 1.00, 0.90, 0.60]`.

| spot | km | ppt range | score range | spread (AFTER) | spread (BEFORE, PR #5) |
|---|---:|---|---|---:|---:|
| North Jetty | 2.58 | 26.43–33.89 | 0.706–0.936 | **0.230 (23.0 pts)** | 0.229 (23 pts) |
| Georgetown Lighthouse | 5.52 | 24.05–33.64 | 0.718–0.959 | **0.241 (24.1 pts)** | 0.240 (24 pts) |
| Mud Bay Cut | 13.05 | 16.50–27.85 | 0.922–0.981 | **0.059 (5.9 pts)** | 0.062 (6 pts) |
| bay stations (16.68 km) | 16.68 | 12.74–17.50 | 0.891–0.962 | **0.071 (7.1 pts)** | 0.069 (7 pts) |
| bay stations (19.03 km) | 19.03 | 9.96–10.59 | 0.846–0.859 | **0.013 (1.3 pts)** | 0.011 (1 pt) |

**Did the spread narrow? No — it is statistically indistinguishable from before, at every spot,
including North Jetty and Georgetown.** North Jetty: 23.0 pts now vs 23 pts before. Georgetown:
24.1 vs 24. The two spots nearest the mouth — the ones §3 already identified as the reach the model
fits worst and a fishing recommendation depends on most — show no improvement at all from this
work. This is consistent with, and further evidence for, §3's conclusion: the score ambiguity at
these spots was never primarily a phase problem, so correcting phase does not touch it. It is a
`front_width_km` / model-form sensitivity, exactly as the previous gate's probe found, still
present after phase correction because phase correction never addressed that axis.

## 5. What is binding now

**`fitted` cannot become True, and the single binding cause is unchanged from the previous gate:
model form (`POOR FIT`), not coverage.** Confirmed directly — the diagnostics warning string for
this run contains exactly one paragraph:

```
POOR FIT: salinity rmse 4.354 ppt against a resolution of 0.005; tidal-swing rmse 5.092
ppt against a resolution of 0.044. ...
```

`DISTANCE COVERAGE TOO THIN` stays cleared, by an even wider margin than before: 58 distinct
distances spanning 28.497 km against a fitted `front_width_km` of 14.560 km (down slightly from
14.68 km, so the coverage-to-width ratio improved marginally). `DISCHARGE SPAN TOO NARROW`
(715–183,660 cfs is a 257x span against a 2x floor), `UNCONSTRAINED PARAMETERS` (condition number
8.06 against a 1,000,000 ceiling, no 1-sigma exceeding its value), and `AT THE OPTIMIZER BOUND`
(`at_bounds: []`) all stay clear, same as before. `n_interior_obs` is 8,667 (nonzero), so
`NO INTERIOR OBSERVATIONS` does not fire either.

`POOR FIT` alone is enough to hold `fitted` at False, and by a large margin: combined rmse is 952x
the combined observation resolution (4.354236719156845 / 0.004575094657130837); on the NERRS-only
subset alone — clean, phase-correct even before this plan — it is 1,369x (4.059553283050965 /
0.002964441909108196), essentially unchanged from the previous gate's measured 1,370x. **Phase
correction did not move this conclusion.** The gap between what the model reproduces and what the
data can resolve is two-to-three orders of magnitude, and no amount of correctly scoring existing
observations at their real tidal phase closes a gap that size — it needs a change to the model
itself (most plausibly the single-layer/no-stratification form, or the branch-offset issue named in
§7 below), which is out of this task's scope and remains a decision for later work.

## 6. A regression found while writing this report, and fixed

**The bug.** `tidescout salinity calibrate`'s per-station bias table (`salinity_fit.station_bias`,
added by the previous plan/gate) kept evaluating every row — WQP grabs included — at the fixed
`FIT_PHASE`, not at each grab's own resolved phase, even after this plan's Tasks 1–3 wired real
per-row phases into `fit_intrusion` itself. The two scoring paths silently diverged: `fit_intrusion`
(and therefore §1's `rmse_by_source_ppt`) was phase-correct; `station_bias` (and therefore the
printed per-station table, and this report's own §3 draft) was not, for all 56 WQP distances. The
table's own docstring and the CLI's caption beside it ("the same scoring `fit_intrusion` itself
uses") were true before this plan and became false the moment Task 3 shipped — a second instance,
after Task 5's up-estuary-lag test, of this plan shipping something that could not fail on the
claim it made. Caught by the coordinator's own source read after the first draft of this report
cited the (then-stale) numbers in §3.

**The fix.** `station_bias` gained an optional `phases` parameter with the identical contract
`fit_intrusion` already uses: a sequence the same length and order as `observations`, filtered
through `_finite_rows` the same way (mirroring `fit_intrusion`'s own `kept_phases`, for the same
row-alignment reason), defaulting to empty — which reproduces the old FIT_PHASE-for-every-row
behaviour exactly, so any other caller is unaffected. The CLI's `salinity_calibrate` command now
passes `data.observation_phases` through, the same array `fit_intrusion` itself is called with a
few lines above. The docstring, the `StationBias` dataclass's inline comment, and the CLI's printed
caption were all corrected to describe what the function now does, not what it used to do.

**Exact-match consistency check.** Aggregating the fixed `station_bias` table's 56 WQP rows
(n-weighted RMS-combine, excluding the two NERRS distances by station name) reproduces
`fit_intrusion`'s own `rmse_by_source_ppt['wqp']` **exactly**:

```
full-precision WQP-weighted rmse from FIXED station_bias: 5.782954764145098   (n=1860)
fit_intrusion rmse_by_source_ppt['wqp']:                  5.782954764145098
difference:                                               0.0

full-precision WQP-weighted rmse from the OLD (FIT_PHASE-only) station_bias: 6.07107988948819
```

Before the fix, the same aggregation gave 6.071 — a different number from what the fit itself
actually scored, which is precisely the bug. After the fix, it is bit-for-bit the same number. 56
of the table's 58 rows changed (the 2 that didn't are NIWTAWQ and NIWWBWQ+WYSS1, NERRS's own
distances, which always scored at `FIT_PHASE` and correctly still do); of those 56, 29 moved down
and 27 moved up — consistent with correcting a phase mismatch that sometimes made a row look better
than it truly fit and sometimes worse, not with a fix that uniformly improves the picture.

**The test, and the deliberate-break evidence the brief for this fix required.** Added
`test_station_bias_scores_each_row_at_its_own_phase` to `backend/tests/test_salinity.py`: it scores
a row (`TRUTH.excursion_km = 7.0`, nonzero) at `phase=0.0` (low water, where
`excursion_km * cos(2*pi*phase)` is nonzero) and asserts the residual matches the low-water
prediction, not the `FIT_PHASE` one — with an explicit guard asserting the two predictions actually
differ, so the test cannot pass by accident if `phases` were silently ignored. A second test,
`test_station_bias_default_phases_reproduces_fit_phase_behaviour`, pins that calling with no
`phases` argument at all still reproduces the old FIT_PHASE-for-every-row behaviour exactly.

To confirm the first test can actually fail on the bug it names (not merely on a signature
mismatch), the residual computation line was reverted in place — `salinity_at(d, q, ph, cfg)` back
to `salinity_at(d, q, FIT_PHASE, cfg)`, a one-line deliberate break simulating the exact historical
defect (the function still accepts `phases`, but silently ignores them when scoring) — and the
station_bias test suite re-run:

```
tests/test_salinity.py::test_station_bias_scores_each_row_at_its_own_phase FAILED
E   assert 24.587633210534676 == 19.28862509455343 ± 1.9e-05

1 failed, 6 passed, 673 deselected in 0.66s
```

**RED**, and isolated exactly where it should be: the one test built to catch this failed; every
other `station_bias` test, including the new default-phase test (which never supplies `phases` and
so is unaffected by the break), stayed green. The break was then reverted line-for-line and the
suite re-run:

```
tests/test_salinity.py::test_station_bias_scores_each_row_at_its_own_phase PASSED
7 passed, 673 deselected in 0.56s
```

**GREEN**, and `git diff backend/tidescout/pipeline/salinity_fit.py` after the revert was diffed
against the committed fix and found byte-for-byte identical — no leftover break markers reached the
commit.

**Confirmed the fix touched nothing `fit_intrusion` itself produces.** Re-ran
`tidescout salinity calibrate winyah-bay` end to end after the fix and diffed its diagnostics block
against the pre-fix run: `rmse_ppt`, `rmse_by_source_ppt`, `n_obs`, `n_phase_supplied`,
`n_interior_obs`, `cfs_span`, `n_dropped`, `n_swing_obs`, `swing_rmse_ppt`, `n_distinct_distances`,
`distance_span_km`, `n_distinct_discharges`, `fitted_params`, `param_sigma`, `condition_number`, and
`at_bounds` are all identical to full float precision. Every number in §1, §2, §4, and §5 of this
report — rmse overall/by-source, the 15.5%/84.5% excess arithmetic in §3, the fitted parameters, the
score-spread probe, and the binding-cause conclusion — is unchanged by this fix; only §3's
per-station citations and this section moved. That is the expected, and confirmed, scope: the fix
corrects a diagnostic (`station_bias`), not the fit (`fit_intrusion`).

Committed as `5bae284` on `plan-06-tidal-phase`: `backend/tidescout/pipeline/salinity_fit.py`
(`station_bias` + docstrings), `backend/tidescout/cli.py` (call site + caption), and
`backend/tests/test_salinity.py` (two new tests). `make check`: **680 passed** (678 → 680, both
additive), ruff clean.

## 7. What this work did NOT do

- **Added no coverage.** Observation count (12,725), distinct distances (58), and distance span
  (28.497 km) are identical before and after — this plan touches only how existing observations are
  *scored*, not which observations exist. Nothing here changes `DISTANCE COVERAGE TOO THIN`'s
  already-cleared status (§5) or the coverage-at-three-spots table from the previous gate report
  (unaffected — it depends only on the set of observed distances, which did not change).
- **Does not touch the 0–2.58 km band.** North Jetty (2.580 km) sits 1.862 km seaward of the lowest
  real observation (4.44 km, a single WQP grab) — the same gap the previous gate measured, still
  open coast with zero stations. North Jetty is still `EXTRAPOLATED`, not `MEASURED`, and this
  task's phase correction does nothing to change that: it corrects how existing observations near
  16–19 km and scattered up-estuary are scored, not whether anything exists at the mouth.
- **Does not address stratification.** The previous gate's measured +3.30 ppt median difference
  between surface and bottom salinity at one distance is a property this model's single depth-
  averaged layer cannot represent under any scoring convention, tidal phase included. Phase-
  correcting a grab's horizontal position in the tidal cycle says nothing about which depth it was
  taken at, and the model has no vertical coordinate to carry that information even if it did.

## Appendix: reproducing this report

`tidescout salinity calibrate winyah-bay`, run end to end against the real `data/winyah-bay/`
store (no synthetic data), full output captured to `/tmp/calibrate-phase.txt` (the post-§6-fix run).
§3's arithmetic and §4's probe were produced by two standalone scripts exercising only existing,
tested library functions (`salinity_fit.collect_observations`, `fit_intrusion`'s internal helpers
`_finite_rows`, `_by_discharge`, `_levels`, `_swing`, `_BOUNDS`, and `engine.salinity.salinity_at`)
— the same functions `fit_intrusion` itself calls, with `front_width_km` pinned rather than free,
mirroring exactly how the previous gate's probe was built. §1/§2/§4/§5's numbers required no
production code change (`fit_intrusion` and `CalibrationInput` were already wired by Tasks 1–3);
§3/§6's per-station table required the `station_bias` fix described in §6, committed as `5bae284`.

`make check`: **680 passed** (678 → 680, both additive — the two `station_bias` phase tests from
§6), ruff clean. Test count only went up, as required.

**Status: one code change, described and evidenced in §6.** `fisheries/winyah-bay.yaml`'s
`salinity:` block, `ocean_ppt`, `ocean_boundary_utm_km`, the ANUGA mesh, the flow library,
`ON_AXIS_MAX_KM`, and `fit_intrusion`'s own maths and `engine/salinity.py` were not touched — the
fix was scoped to a diagnostic function (`station_bias`) and its CLI call site, confirmed by the
byte-identical `fit_intrusion` diagnostics comparison in §6. This report is the gate's entire
output; what happens to `ocean_ppt`, the model form, or stratification next is the owner's
decision, not this task's.

## 8. Fix wave: final whole-branch review, before proposing merge (2026-08-24)

A final whole-branch review of `plan-06-tidal-phase` found nine issues, none touching
`fisheries/winyah-bay.yaml`'s `salinity:` block, `ocean_ppt`, `fitted`, the model form,
`ocean_boundary_utm_km`, the ANUGA mesh, the flow library, `ON_AXIS_MAX_KM`, `engine/salinity.py`'s
arithmetic, or the phase convention (0 = LOW water). All nine are fixed in one pass:

1. **`pipeline/salinity_fit.py`'s `excursion_km` docstring paragraph was stale.** It claimed the
   level residual is IDENTICALLY independent of `excursion_km`, full stop — true only because every
   row used to share `FIT_PHASE` (0.25, where `cos(2*pi*phase)` vanishes). After Tasks 1–3 of this
   plan, 1,860 of 12,725 rows carry individually-resolved phases spanning 0.003–0.999, so that
   independence no longer holds for the whole population — the level residual now has a nonzero
   `excursion_km` gradient on those rows, with or without swing observations. Rewritten to say so,
   and to say plainly that the `swing_obs` gate is now a deliberate, conservative choice, not a
   mathematical necessity. The gate's behaviour is unchanged.
2. **`engine/tides.py`'s `MAX_HALF_CYCLE_H` comment misnamed its station and understated its
   distribution.** It named "Springmaid Pier" for station 8662549, which is actually South Island
   Ferry, Intracoastal Waterway (Springmaid Pier is a different station, 8661070). It also claimed
   zero intervals over 8 h, measured by looking within each year separately — which hides the seams
   a concatenated series actually crosses. Measured properly across all 28 cached chunks
   concatenated (39,519 intervals): min 4.42 h, median 6.28 h, max 12.13 h, 8 over 8 h, 3 over 9 h.
   Comment rewritten with the correct station and distribution, distinguishing the 3 same-kind
   (`L`→`L`) year-seam gaps — already rejected by `phase_at`'s own same-kind guard regardless of this
   threshold — from the 5 DST-artefact intervals fixed by item 5. `MAX_HALF_CYCLE_H` itself unchanged
   at 9.0.
3. **`backend/tests/test_noaa.py::test_tide_events_range_parses_a_real_coops_response` could not
   fail on its own claim.** Its docstring warned that parsing CO-OPS timestamps as UTC would shift
   every phase by 4–5 hours, but its four assertions (non-empty, tz-aware, kind, sorted) all pass
   regardless of which zone `_parse_t` attaches. Added two assertions that do depend on it: the
   fixture's first event's absolute instant (`== datetime(1999, 1, 1, 2, 0,
   tzinfo=ZoneInfo("America/New_York"))`) and `phase_at(out, out[0].time) == 0.0`. Deliberately broke
   `_parse_t` to `tzinfo=UTC` in place and re-ran: **RED** —
   `AssertionError: assert datetime.datetime(1999, 1, 1, 2, 0, tzinfo=datetime.timezone.utc) ==
   datetime.datetime(1999, 1, 1, 2, 0, tzinfo=zoneinfo.ZoneInfo(key='America/New_York'))`. Reverted
   the break and re-ran: **GREEN**, `git diff` on `sources/noaa.py` empty before the real fix was
   applied (confirming no break markers leaked). This is the third time this plan shipped a test
   that could not fail on the thing it named (after Task 5's up-estuary-lag test and §6's
   `station_bias` regression above) — all three are now closed.
4. **`gate-report.md`'s §"Bottom line" claimed a pre-specified numeric tolerance that never
   existed** ("the falsification test's own tolerance for 'didn't move'"). Reworded to the argument
   §2 actually makes two paragraphs later — 0.0018 ppt is three orders of magnitude below NERRS's
   own 4.06 ppt rmse. No number in this file changed; wording only.
5. **`engine/tides.py`'s `phase_at` had a DST bug.** `span` was computed by subtracting two
   `TideEvent.time` values that can share one `ZoneInfo` object — Python's documented behaviour for
   that case is naive wall-clock subtraction, silently dropping any DST offset change between them.
   On the 2026-03-08 spring-forward interval this gave span 8.233 h against a true 7.233 h, and
   `phase_at` at the true midpoint of the falling limb would have returned 0.7196 instead of 0.75.
   Fixed by differencing in UTC (`before.time.astimezone(UTC)`, etc.) for `span`, `frac`'s numerator,
   and `t`. Zero stored WQP rows fall on a DST-transition day today, so this does not change any
   fitted number (confirmed below). This also removes the 5 spurious 8.2–8.4 h intervals from item 2.
6. **`phase_at` re-sorted and linear-scanned on every call.** Measured at production scale: 1,860
   calibration lookups over 39,520 events. Fixed by (a) a cheap O(n) ascending check that skips
   `sorted()` entirely when the input already qualifies — true for every real caller, since
   `tide_events_range` already sorts — and (b) replacing the O(n) linear scan with `bisect_left`/
   `bisect_right` (O(log n)) to find the bracketing pair, preserving the exact original preference
   for the first VALID pair (including the documented tie-handling invariant for `t` landing exactly
   on an interior event). All 17 `test_tides.py` cases, including the unsorted-input and both
   exact-hit-on-interior-event cases, pass unchanged. Timing, measured on a synthetic benchmark at
   production scale (39,520 sorted events, 1,860 queries, matching `phase_at`'s real calling
   pattern): **old ~1.445 ms/call (2.69 s total) → new ~0.448 ms/call (0.83 s total)**, roughly a 3.2x
   speedup on this machine. (This benchmark is a stand-in, not an instrumented run of the actual
   calibrate command; the reported production baseline was 21.45 s / 11.53 ms per call over the same
   event/query counts.)
7. **The WQP site table could claim rows a station did not contribute.** `collect_observations`'s
   site-record loop filtered a WQP station's `rows` (which set `n_days` and the ppt range) by
   `day in by_day` only — not by phase resolvability, which item 5's per-row scoring made a NEW
   per-row rejection (`n_no_phase`). A station could therefore read `used: yes` with `n_days`
   counting rows that never reached the fit. The previous plan closed this exact hole for the
   discharge-day case (comment at the site-table loop); extended the same gate to phase: the tide
   events fetch was moved earlier (before the site-table loop, not just before the
   observation-building loop) and `rows` now requires `phase_at(events, ts) is not None` alongside
   `day in by_day`. On the real Winyah run this changes nothing (`n_no_phase` is 0 today — every row
   that passes the discharge-day gate already resolves a phase), confirmed by the unchanged
   calibrate output below.
8. **Two false cross-references.** `test_salinity.py`'s "one tide station suffices" comment pointed
   at "the module docstring for `pipeline/salinity_fit.py`" for why one station's phase suffices —
   that docstring never discusses tide stations or phase sourcing. Repointed at the spec's §2
   ("Up-estuary phase lag is negligible"), which is where that ruling actually lives.
   `salinity_fit.py`'s `fit_intrusion` docstring quoted the PRE-phase per-source rmse (NERRS 4.061
   ppt/10,880 rows, WQP 6.102 ppt/1,860) as if current, dated the same day as the after-numbers.
   Marked SUPERSEDED and added the current figures alongside: NERRS 4.0596 ppt / 10,865 rows, WQP
   5.783 ppt / 1,860 rows.
9. **`cli.py`'s `salinity calibrate` command didn't catch `SourceUnavailable`.** `collect_observations`
   now fetches tide predictions as part of every call (for WQP phase resolution); a first run with
   no cached predictions and CO-OPS down previously exited with a raw traceback instead of the
   command's usual red one-liner. `except FileNotFoundError` widened to
   `except (FileNotFoundError, SourceUnavailable)`, matching the pattern the command already uses
   for its other source failure.

**Confirmed no fit number moved.** Re-ran `tidescout salinity calibrate winyah-bay` end to end after
all nine fixes: 12,725 observations, no `... WQP grab(s) excluded -- no tidal phase...` line (i.e.
`n_no_phase` still 0), `rmse_ppt` **4.354236719156845**, `rmse_by_source_ppt` nerrs
**4.059553283050965** / wqp **5.782954764145098**, `condition_number` **8.057548319225928** — all
identical to §1's table above, byte-for-byte on the figures this report's conclusions rest on. Items
5 and 7 were flagged as capable of legitimately moving these numbers; on Winyah's real data, neither
did (zero DST-transition-day WQP rows for item 5; `n_no_phase` already 0 for item 7).

`make check`: **680 passed** (unchanged — this wave added assertions to an existing test rather than
new test functions; test count did not go down), ruff clean.
