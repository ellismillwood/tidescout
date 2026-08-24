# Salinity anchoring — gate report

**Plan:** `docs/superpowers/plans/2026-08-24-salinity-anchoring.md` (Task 7, the gate)
**Spec:** `docs/superpowers/specs/2026-08-24-salinity-anchoring-design.md`
**Branch:** `plan-05-salinity-anchoring` (PR #5) | **Date:** 2026-08-24


**This task measures and reports. It does not decide.** No value in `fisheries/winyah-bay.yaml`'s
`salinity:` block was touched, `ocean_ppt` was not freed, and no model form was chosen. Everything
below is a live re-run against real data, captured 2026-08-24 on branch `plan-05-salinity-anchoring`.

Reproduce with:
```
tidescout salinity import-wqp winyah-bay   # 0 new -- already imported by Task 1
tidescout salinity stem winyah-bay         # rebuilds stem_km.npy
tidescout salinity calibrate winyah-bay    # the numbers below
```

## Bottom line

**`fitted` cannot become True, and the binding cause is no longer coverage — it is the model
form.** Every warning this run raised beyond `DISTANCE COVERAGE TOO THIN` has cleared. `POOR FIT`
is now the *only* warning, and it holds even on the clean, phase-correct NERRS subset alone (rmse
4.061 ppt against a resolution of 0.00296 — **1,370x**). More anchoring data will not close a gap
that size. The plan's own honest prediction — `fitted` moving from "false, and structurally
hopeless" to "false, for one identifiable reason" — is exactly what happened.

## 1. Headline comparison

| metric | BEFORE (PR #4) | AFTER (measured 2026-08-24) |
|---|---|---|
| observations | 10,864 | **12,725** |
| distinct on-axis distances | 2 | **58** |
| distance span | 2.35 km | **28.497 km** |
| overall rmse | 4.060 ppt | **4.419 ppt** (contaminated — see §2) |
| rmse, NERRS subset only | 4.060 ppt | **4.061 ppt** (clean, phase-correct — the real comparison) |
| observation resolution | 0.003 ppt | **0.00458 ppt** (combined) / **0.00296 ppt** (NERRS only) |
| rmse / resolution | 1,353x | **966x** (combined, contaminated) / **1,370x** (NERRS-only, clean) |
| condition number | 12.2 | **7.94** |
| stations excluded by the (now computed) axis screen | n/a (3, by hand) | **139** (computed; 203 sites evaluated, 61 admitted, 2 more excluded by domain snap gap, 1 by co-location) |
| `fitted` | False | **False** |
| warnings firing | POOR FIT + DISTANCE COVERAGE TOO THIN | **POOR FIT only** |

Exact figures, straight from `fit_intrusion`'s diagnostics on this run:

```
n_obs: 12725              n_distinct_distances: 58        distance_span_km: 28.497082233428955
rmse_ppt: 4.4188666384259045                               condition_number: 7.935812672076431
rmse_by_source_ppt: {'nerrs': 4.061384904005473, 'wqp': 6.10203118406574}
at_bounds: []              n_swing_obs: 10865              swing_rmse_ppt: 5.081325101946025
fitted_params: ['l0_km', 'k', 'front_width_km', 'excursion_km']
warning: POOR FIT: salinity rmse 4.419 ppt against a resolution of 0.005; tidal-swing rmse 5.081
  ppt against a resolution of 0.044.   [only paragraph — no other warning fires]
```

`observation_resolution_ppt` on the full 12,725-row set: **0.004575094657130837** (unrounded).
On the 10,865-row NERRS-only subset: **0.002964441909108196** — essentially the same 0.003 the
YAML records from before this work, because NERRS daily means are the same clean population they
always were. Per-source counts: `{'nerrs': 10865, 'wqp': 1860}`.

## 2. The rmse rise from 4.060 to 4.419 is partly artefact — read this before the table above

`Observation` is `(distance_km, cfs, ppt)`. It carries no tidal phase. `_levels` evaluates
**every** row — a NERRS daily mean or a WQP instantaneous grab alike — at `FIT_PHASE = 0.25`, the
phase at which the tidal term is exactly zero. That is correct for a daily mean, which already
averages the tide out. It is wrong for a grab sample, which was taken at one real instant that may
sit anywhere in the tidal cycle.

Measured split, this run:

```
NERRS daily means   n=10,865   rmse 4.061 ppt   mean residual (pred - obs) +0.593
WQP grab samples    n= 1,860   rmse 6.102 ppt   mean residual (pred - obs) +1.253
```

The model's own tidal swing at the distances where WQP grabs sit is 8.3–12.3 ppt, so a grab taken
near high or low water is off by 4–6 ppt from phase mismatch alone — before any question of
whether the spatial model is even correct. This is a defect in this plan's own
`Observation`/`fit_intrusion` interface (documented at the point WQP rows are appended in
`collect_observations`), not evidence that the added WQP stations are bad data or that the model
regressed.

**What follows from this:**

- The falsification claim rests on the NERRS daily means, which *are* correctly phase-matched:
  rmse 4.061 ppt against a resolution of 0.00296 ppt — **still roughly 1,370x**, essentially
  unchanged from the 4.060/0.003 ≈ 1,353x measured before this plan's work. The model reproduces
  the estuary no better and no worse than it ever did, on the data that was always trustworthy.
- The coverage win is real and is this plan's actual, measured achievement: **2 → 58 distinct
  distances, 2.35 → 28.50 km span**, and `DISTANCE COVERAGE TOO THIN` has cleared — its own
  test (`n_distinct_d >= 3` and `distance_span >= front_width_km`) is satisfied outright.
- The combined 4.419 ppt headline rmse is contaminated by the phase-mismatch effect above and
  **must not be cited as evidence the model got worse**. It did not get worse; a second, noisier
  population was added to it without a mechanism to score that population correctly.

**Fix-wave correction (2026-08-24, sharper framing, no number or conclusion above changed):** the
paragraphs above state two facts separately that need to be read together. NERRS supplies 10,865
rows at **2** distinct distances (16.68 and 19.03 km); WQP supplies 1,860 rows at the other **56**
of the 58 distinct distances this plan added. That means the fit's entire spatial SHAPE —
`l0_km` and `front_width_km`, the two numbers a scoring layer actually reads — is set almost
entirely by the population that is scored at the wrong tidal phase, while the rmse that carries
the falsification claim (§1's headline, and the "still roughly 1,370x" bullet above) comes almost
entirely from the population that constrains no shape at all (2 distances cannot constrain a
front position AND a front width AND a discharge exponent independently). "2 → 58 distinct
distances" must not be read as "the profile shape is now constrained" until phase is carried —
today it is 56 phase-mismatched points against 2 phase-correct ones.

The phase mismatch is not merely noisier, either — it is a directional BIAS, and the model's own
form says why. `salinity_at` evaluates `tanh((x + E·cos(2π·phase) − L) / W)`; scoring every grab at
`FIT_PHASE` instead of its own real phase means the argument is displaced by up to `E·cos(2π·phase)`
in a direction the grab's own timestamp determines, not zero. Because tanh is concave for positive
arguments and convex for negative ones, averaging over a plausibly-uniform phase distribution does
not average out to the daily-mean value — Jensen's inequality gives a systematic, distance-dependent
offset, largest exactly where the profile's curvature is largest: the 4–13 km reach this plan just
anchored. The measured mean residuals above (NERRS +0.593, WQP +1.253 — both positive, and WQP's
is more than double NERRS's) are consistent with that directional bias, not with unstructured noise
of the kind a noisier-but-unbiased population would leave behind.

Measured effect size, so this reads as a framing correction and not a new defect: `front_width_km`
moved 14.48 → 14.68 km and the clean NERRS-only rmse moved 4.060 → 4.061 ppt (§1) — both
essentially unchanged. The 56 WQP distances did not visibly distort the fitted shape on this run.
But that is a measurement of THIS run, not a guarantee: the correct statement of what this plan
achieved is "56 new distances were added, scored at the wrong phase," not "56 new distances now
constrain the profile" — the latter requires carrying phase through `Observation`/`fit_intrusion`,
which this plan explicitly left as a known limitation (documented at the point WQP rows are
appended in `collect_observations`).

## 3. Per-station bias, every admitted station

`salinity_calibrate` now prints this table itself (see `backend/tidescout/cli.py` /
`tidescout.pipeline.salinity_fit.station_bias`, added this task). Residual is *predicted minus
observed*, evaluated at each row's own discharge and `FIT_PHASE` — the same scoring
`fit_intrusion` itself uses. Two admitted stations that snap to the exact same along-estuary
distance (`Observation` carries no station id — see the note just below this table) are combined into
one row; this happens for the WYSS1/NIWWBWQ surface/bottom pair at 19.03 km and two WQP
legacy/WQX id-split pairs.

| station(s) | km | n | mean residual (ppt) | rmse (ppt) |
|---|---:|---:|---:|---:|
| 21SC60WQ_WQX-RO-046082 | 4.44 | 1 | -2.208 | 2.208 |
| 21SC60WQ_WQX-RO-06317 | 4.60 | 13 | -1.791 | 8.280 |
| 21SC60WQ_WQX-RO-23321 | 4.99 | 7 | +3.011 | 13.893 |
| 21SC60WQ_WQX-RO-19427 | 5.04 | 10 | -3.589 | 5.563 |
| 21SC60WQ_WQX-WB-06 | 5.56 | 34 | -1.373 | 8.014 |
| 21SCSHL-05-24 | 5.76 | 78 | +5.534 | 11.111 |
| 21SCSHL_WQX-05-24 | 5.77 | 14 | -0.691 | 5.852 |
| 21SC60WQ_WQX-RO-056094 | 7.54 | 12 | -6.492 | 9.753 |
| 21SC60WQ_WQX-RO-056110 | 8.73 | 1 | +7.084 | 7.084 |
| 21SC60WQ_WQX-WB-05 | 10.28 | 33 | +0.086 | 7.370 |
| 21SCSHL-05-21 | 10.30 | 76 | +4.891 | 9.411 |
| 21SCSHL_WQX-05-21 | 10.31 | 14 | -0.551 | 5.233 |
| 21SC60WQ_WQX-RO-14347 | 10.65 | 11 | +1.094 | 8.528 |
| 21SC60WQ_WQX-RO-26373 | 10.90 | 1 | +2.153 | 2.153 |
| 21SC60WQ_WQX-RO-19439 | 11.78 | 9 | -1.320 | 6.285 |
| 21SC60WQ_WQX-RO-15375 | 12.07 | 11 | +3.966 | 5.583 |
| 21SCSHL_WQX-05-25 | 12.16 | 14 | -2.839 | 7.017 |
| 21SCSHL-05-25 | 12.17 | 76 | +7.464 | 9.775 |
| 21SC60WQ_WQX-RO-22309 | 12.61 | 12 | -1.839 | 9.490 |
| 21SC60WQ_WQX-RO-01108 | 12.88 | 1 | +12.294 | 12.294 |
| 21SC60WQ_WQX-MD-278 | 13.30 | 224 | +0.467 | 7.179 |
| 21SCSHL-05-20 | 13.50 | 76 | +4.044 | 8.336 |
| 21SCSHL_WQX-05-20 | 13.52 | 14 | -5.467 | 8.334 |
| 21SC60WQ_WQX-WB-09 | 13.92 | 15 | +3.931 | 5.156 |
| 21SC60WQ_WQX-RO-07332 | 14.41 | 16 | -0.089 | 3.850 |
| 21SC60WQ_WQX-RO-10380 | 15.81 | 15 | +4.430 | 6.658 |
| 21SC60WQ_WQX-RO-16391 | 16.05 | 11 | +3.929 | 4.628 |
| 21SC60WQ_WQX-RO-036052 | 16.09 | 1 | +6.071 | 6.071 |
| 21SC60WQ_WQX-RO-01113 | 16.10 | 1 | +13.013 | 13.013 |
| **NIWTAWQ** | 16.68 | 3,574 | +2.227 | 4.234 |
| 21SC60WQ_WQX-RO-13343 | 17.21 | 10 | -1.275 | 6.983 |
| 21SC60WQ_WQX-RO-046064 | 17.36 | 13 | +2.984 | 4.952 |
| 21SC60WQ_WQX-WB-04 | 18.04 | 18 | +2.427 | 3.722 |
| **NIWWBWQ, WYSS1** (surface/bottom pair, same distance) | 19.03 | 7,291 | -0.207 | 3.974 |
| 21SC60WQ_WQX-RO-01161 | 19.22 | 10 | +1.761 | 5.656 |
| 21SC60WQ_WQX-RO-08348 | 20.09 | 14 | -0.491 | 6.722 |
| 21SC60WQ_WQX-RO-17407 | 20.50 | 12 | +1.470 | 2.769 |
| 21SC60WQ_WQX-RO-01121 | 21.35 | 9 | +1.881 | 4.922 |
| 21SC60WQ_WQX-RO-18423 | 22.11 | 9 | -4.820 | 13.673 |
| 21SC60WQ_WQX-RO-036054 | 22.28 | 1 | +2.906 | 2.906 |
| 21SC60WQ_WQX-RO-09364 | 22.92 | 10 | -1.078 | 5.379 |
| 21SC60WQ_WQX-RO-02012 | 23.10 | 11 | +5.435 | 7.529 |
| 21SC60WQ_WQX-RO-24337 | 23.19 | 12 | +1.056 | 3.192 |
| 21SC60WQ_WQX-MD-080 | 23.90 | 34 | -0.479 | 4.242 |
| CCU_EQL-SR 8 | 24.00 | 34 | +0.764 | 2.698 |
| 21SC60WQ_WQX-RO-11315 | 24.01 | 12 | -0.961 | 3.778 |
| 21SC60WQ_WQX-MD-074, 21SC60WQ_WQX-WB-03 | 24.08 | 53 | +0.470 | 2.919 |
| 21SC60WQ_WQX-MD-077 | 24.59 | 233 | +0.615 | 3.360 |
| 21SC60WQ_WQX-RO-046062 | 24.81 | 23 | +0.403 | 4.012 |
| 21SC60WQ_WQX-RO-25353 | 24.85 | 10 | +2.346 | 3.867 |
| 21SC60WQ_WQX-WB-02A, 21SC60WQ_WQX-WB-07 | 24.98 | 22 | +1.240 | 2.069 |
| 21SC60WQ_WQX-RO-21471 | 25.06 | 3 | +1.919 | 1.935 |
| 21SC60WQ_WQX-MD-073 | 25.34 | 45 | -0.341 | 3.303 |
| 21SC60WQ_WQX-WB-02 | 25.84 | 11 | -0.487 | 2.819 |
| 21SC60WQ_WQX-WB-01 | 26.16 | 18 | +1.957 | 2.630 |
| 21SC60WQ_WQX-RO-14359 | 29.19 | 10 | +0.458 | 3.121 |
| 21SC60WQ_WQX-MD-142 | 31.70 | 213 | +0.888 | 3.010 |
| 21SC60WQ_WQX-MD-275 | 32.94 | 209 | +0.179 | 2.741 |

61 admitted stations, 58 distinct distances, 12,725 rows accounted for exactly (the 3 rows above
whose station list holds two names combine both stations' rows under the shared distance — that
is not double counting, it is `Observation`'s real limit: it carries no station id, only a
distance, so two stations at the identical distance cannot be told apart from it).

The two high-volume NERRS rows (NIWTAWQ, NIWWBWQ+WYSS1) carry 85% of all admitted rows between
them and pull the largest bias magnitudes near the true optimum's tightest constraint (+2.227 and
-0.207 ppt respectively) — small next to the near-mouth WQP grabs' bias, which run as high as
±13 ppt on n=1 stations. Those single-reading stations are not a fit failure so much as a sample
size of one; they contribute almost nothing to the least-squares objective and almost everything
to the *appearance* of scatter in this table.

## 4. Coverage at the three known fishing spots

Using `engine.salinity.classify_coverage` against the real 58 observed distances
(`salinity_fit.collect_observations`'s own output, span 4.442–32.939 km), and
`salinity_fit.site_distances_km` for each spot's real along-estuary distance:

| spot | distance | coverage | nearest observation |
|---|---:|---|---|
| Georgetown Lighthouse | 5.517 km | **MEASURED** | 0.040 km away |
| Mud Bay Cut | 13.052 km | **MEASURED** | 0.173 km away |
| North Jetty | 2.580 km | **EXTRAPOLATED** | 1.862 km away |

Before this plan, all three read EXTRAPOLATED (the only NERRS observations sat at 16.68 and
19.03 km, nowhere near any of them). **Two of three are now MEASURED.** North Jetty is not, and
cannot be with data of this kind: it sits 1.862 km seaward of the lowest real observation at
4.442 km (`21SC60WQ_WQX-RO-046082`, a single WQP grab), outside `classify_coverage`'s
`near_km=1.0` window. This is the same 0–2.58 km band the spec named as ungained by this work —
confirmed, not merely asserted: it is still open coast with zero WQP stations and the honest
answer there is the ocean end-member.

Read `coverage=MEASURED` correctly: it means a real observation sits nearby, not that the model
was ever calibrated at that position (`fitted` is False fishery-wide). Georgetown's own admitted
station (WB-06, 34 grabs at 5.56 km) shows a rmse of 8.014 ppt against the fitted model in §3 — a
real, nearby observation with a large residual is exactly what "measured, not trustworthy" means.

## 5. Feature-coverage table recomputed

The spec's §1 table, reproduced against the current 36.193 km distance field for reference
(`data/winyah-bay/estuary_km.npy`, feature centroids sampled the same way production does —
`engine.activation._sampling_anchors`, reproject-then-centroid — against
`data/winyah-bay/features.geojson`'s 2,162 features):

| reach (OLD bands, pre-WQP) | features | share |
|---|---:|---:|
| seaward of North Jetty (0–2.58 km) | 52 | 2.4% |
| jetties → Mud Bay Cut (2.58–13.05 km) | 761 | 35.2% |
| Mud Bay Cut → first station (13.05–16.68 km) | 122 | 5.6% |
| between the two bay stations (16.68–19.03 km) | 9 | 0.4% |
| above the bay stations (>19.03 km) | 1,218 | 56.3% |

(This reproduces the spec's published table almost exactly — 52/761/122 vs the spec's
53/759/123, a handful of features near a band edge, likely field-version noise; the 9/0.4% and
1,218/56.3% bands match exactly. The pattern is identical either way.)

**Recomputed against the new 58-point observed span (4.442–32.939 km):**

| reach (NEW, bracketed by the full observed span) | features | share |
|---|---:|---:|
| below the lowest observation (< 4.442 km) | 95 | 4.4% |
| **within the observed span (bracketed)** | **1,118** | **51.7%** |
| above the highest observation (> 32.939 km) | 949 | 43.9% |

**Only 0.4% of features sat where observations bracketed them. Now 51.7% do** — a ~124x increase
in both count (9 → 1,118) and share. Using the model's actual per-cell semantics
(`classify_coverage`, which extends 1 km past each end observation as MEASURED rather than
requiring strict bracketing):

| `classify_coverage` | features | share |
|---|---:|---:|
| measured | 1,097 | 50.7% |
| interpolated | 87 | 4.0% |
| extrapolated | 978 | 45.2% |

54.8% of the inventory (measured + interpolated) is no longer pure extrapolation, against 0.4%
before. This is the coverage win stated in the terms the original falsification argument used.

## 6. `ocean_ppt` — two candidates, recommendation only, nothing changed

`ocean_ppt` is held at 34.0 in the YAML and has never been measured (no CO-OPS station within
250 km serves salinity — Task 4, re-verified, not repeated here). It is the seaward anchor the
whole profile decays from and it governs the 0–2.58 km band that §4 confirms is still
EXTRAPOLATED — the one reach this plan does not touch.

**Candidate 1 — highest-salinity on-axis WQP observations near the mouth.** The admitted stations
within 8 km of the mouth:

| station | km | n | ppt range | mean |
|---|---:|---:|---|---:|
| 21SC60WQ_WQX-RO-046082 | 4.44 | 1 | 27.1–27.1 | 27.10 |
| 21SC60WQ_WQX-RO-06317 | 4.60 | 13 | 13.6–43.5 | 25.11 |
| 21SC60WQ_WQX-RO-23321 | 4.99 | 7 | 1.4–33.5 | 18.58 |
| 21SC60WQ_WQX-RO-19427 | 5.04 | 10 | 18.2–34.3 | 26.09 |
| **21SC60WQ_WQX-WB-06** | **5.56** | **34** | **6.3–35.4** | **23.21** |
| 21SCSHL-05-24 | 5.76 | 78 | 0.0–34.0 | 16.71 |
| 21SCSHL_WQX-05-24 | 5.77 | 14 | 6.0–35.0 | 23.86 |
| 21SC60WQ_WQX-RO-056094 | 7.54 | 12 | 9.9–33.6 | 25.81 |

WB-06 (35.4 ppt max, n=34, the most-sampled station this close to the mouth) is the credible
candidate. RO-06317's 43.5 ppt maximum is *not* — it is one of the six values admitted at the
plausibility gate's 45 ppt ceiling specifically because summer evaporative concentration in a
shallow estuarine site can exceed open-ocean salinity (Task 2's ruling); using it as an ocean
anchor would import that same evaporative spike as though it were shelf water. None of these
stations' *means* (16.7–27.1 ppt) resemble 34 ppt — they are all still inside the estuary's own
gradient, sampled at whatever discharge and tidal phase happened to occur, not held at the mouth's
saltiest, driest, lowest-flow extreme the way `ocean_ppt` is meant to represent.

**Candidate 2 — North Inlet's three NERRS stations**, off-axis for the profile (excluded from the
fit as a separate branch — see the YAML's own note) but ocean-flushed and read as a genuinely
marine signal:

| station | n | mean | max | min |
|---|---:|---:|---:|---:|
| NIWCBWQ | 345,332 | 31.91 | 37.30 | 1.90 |
| NIWOLWQ | 365,771 | 32.04 | 39.60 | 0.00 |
| NIWDCWQ | 305,780 | 31.37 | 37.10 | 1.20 |

Full multi-year continuous records (300K+ readings each, not grab samples), mean 31.4–32.0 ppt,
maxima near 39. Being off the profile's axis is exactly what makes this a *cleaner* ocean-end
proxy than an on-axis station could be: these three are flushed by their own tidal inlet, not by
the river the intrusion model exists to describe, so their salinity is not confounded by the very
discharge signal `ocean_ppt` is supposed to be held constant against. Their own low tail (0.00,
1.20, 1.90 ppt minima) shows they still receive some Winyah plume on the flood — consistent with
the YAML's note that all six NERRS stations correlate with discharge — so "mean" or a
high-percentile statistic, not the single maximum, is the honest summary of their ocean character.

**Recommendation (not applied): `ocean_ppt` should be freed in the fit, anchored by North Inlet's
continuous record, not held at an unmeasured 34.0.** North Inlet's ~32 ppt mean is a real,
high-volume, continuously-sampled measurement of shelf-adjacent water near this estuary — the
closest thing to ground truth for the seaward end-member that exists today — while 34.0 is a
regional Atlantic-nearshore convention with no station behind it at all. Freeing it would need
either (a) adding North Inlet's readings to the objective as a distinct, explicitly off-axis
anchor term (not folded into the on-axis distance coordinate, which is exactly what the YAML's
own note says a single axis cannot carry), or (b) holding `ocean_ppt` fixed at a value *derived*
from North Inlet (e.g. a high-flow-conditioned mean or percentile) rather than the current
unsourced 34.0 — a smaller change with most of the benefit, since North Inlet's readings still
say nothing about the specific 0–2.58 km reach `ocean_ppt` governs. Either way this is a decision
for the owner, not this task.

## 7. Score-spread probe re-run

Trout curve (Phase 3 plan, not yet implemented in code — applied here as a standalone piecewise
linear interpolation for this measurement only): `x=[0.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0]`,
`y=[0.05, 0.10, 0.45, 0.85, 1.00, 0.90, 0.60]`.

Same protocol as the spec's original probe: hold `front_width_km` at 5, 8, 12, 16 km and refit
`l0_km`, `k`, `excursion_km` against the identical full dataset used above (12,725 level rows,
10,865 swing rows). Evaluated at `cfs=q0_cfs=4000` (the config's own median-flow reference) and
`phase=FIT_PHASE=0.25`:

```
front_width_km held  5.0 -> l0_km 16.841  k 0.2078  excursion_km 1.981   rmse 5.6863
front_width_km held  8.0 -> l0_km 15.834  k 0.3171  excursion_km 3.133   rmse 4.7905
front_width_km held 12.0 -> l0_km 14.395  k 0.4954  excursion_km 4.629   rmse 4.4328
front_width_km held 16.0 -> l0_km 12.775  k 0.7021  excursion_km 6.039   rmse 4.4473
```

**Caveat that changes how to read this:** the original probe's four fits were *statistically
indistinguishable* (rmse spread 0.016 ppt on the old 348-observation data). On this run they are
not: rmse spans 4.4328–5.6863, a 1.25 ppt spread. Holding `front_width_km` at 5 km is now
measurably worse than letting it float to its fitted 14.68 km. That is itself new information —
the enlarged dataset has started to constrain the front's shape, which the old data could not do
at all — but it means this re-run is not a clean apples-to-apples repeat of "four equally
plausible fits"; it is "four fits, three of them closer to plausible than the fourth."

| spot | km | ppt range | score range | spread | BEFORE spread |
|---|---:|---|---|---:|---:|
| North Jetty | 2.58 | 26.57–33.89 | 0.706–0.934 | 0.229 (23 pts) | 0.701–0.960 (26 pts) |
| Georgetown Lighthouse | 5.52 | 24.22–33.64 | 0.718–0.958 | 0.240 (24 pts) | 0.704–0.960 (26 pts) |
| Mud Bay Cut | 13.05 | 16.71–27.88 | 0.921–0.983 | 0.062 (6 pts) | 0.863–0.907 (4 pts) |
| bay stations (16.68 km) | 16.68 | 12.93–17.55 | 0.894–0.963 | 0.069 (7 pts) | — |
| **bay stations (19.03 km)** | 19.03 | 10.00–10.74 | 0.850–0.861 | **0.011 (1 pt)** | **0.083–0.515 (43 pts)** |

**Did the spread narrow? Sharply at the bay stations, marginally everywhere else.** At 19.03 km —
the exact point the spec measured a 43-point ambiguity — the spread has collapsed to essentially
nothing: 1 point. That reach now carries 10,865 real observations across the four refits, so any
plausible parameter set is forced to reproduce close to the same value there. At North Jetty and
Georgetown the spread narrowed only modestly (26→23, 26→24 points) — those points still sit
where `front_width_km` (still poorly determined relative to the mouth's steep gradient there, even
though it is now well-determined overall) has the largest leverage on the predicted value, and
Georgetown's own nearby anchor (WB-06) is itself a noisy, low-volume grab record (§3, rmse
8.014 ppt). Mud Bay Cut's spread widened slightly (4→6 points), a small but real move in the
wrong direction, consistent with it sitting in the transition zone where the four now-more-
different front widths disagree most.

**This is the number that says the work helped the score, not just the rmse — and it says so only
at the bay stations.** The two spots nearest the mouth, which are also the ones a fishing
recommendation most needs sharp, remain nearly as ambiguous as before.

## 8. What this does and does not resolve

- `fitted` stays False. It cannot become True from more anchoring data of this kind: the clean
  NERRS-only rmse/resolution ratio (~1,370x) is the same order of magnitude it was before this
  plan, and `POOR FIT` — a comparison against the model's own residual, not against coverage — is
  the only warning left.
- `DISTANCE COVERAGE TOO THIN` is gone and stays gone; both of its conditions
  (`n_distinct_d >= 3`, `distance_span >= front_width_km`) are now cleared by a wide margin
  (58 distances, 28.5 km span against a 14.68 km fitted front width).
- The 0–2.58 km band is still EXTRAPOLATED. This plan does not close it and was never going to —
  §6 is the only lever available there, and it is a recommendation, not a fix.
- The score-spread evidence (§7) says the coverage win materially helped the *score* at the two
  spots that are now densely observed, and did not meaningfully help it at the two spots nearest
  the mouth, which remain the ones a decision most depends on.

## Appendix: reproducing this report

Everything above was generated by re-running the real pipeline end to end (`import-wqp`, `stem`,
`calibrate`) and by a standalone analysis script exercising only existing, tested library
functions (`salinity_fit.collect_observations`, `fit_intrusion`, `station_bias`,
`engine.salinity.classify_coverage`/`salinity_at`, `engine.activation._sampling_anchors`) — no
synthetic data anywhere in this report. `backend/tidescout/pipeline/salinity_fit.py` gained
`StationBias`/`station_bias` (pure, tested) and `backend/tidescout/cli.py` wires it into
`salinity calibrate`'s own output, so §3's table is now something every future run of that command
prints itself rather than a one-off computed for this report. `make check`: 636 passed (632 → 636,
all additive; 4 new tests for `station_bias`).

## Fix note (post-review)

Review came back spec ✅, report quality Approved, every number independently re-verified
(including the 58-row bias table summing to exactly 12,725, the empty YAML diff, and the
rejection of RO-06317's 43.5 ppt reading as an evaporative-spike artefact). Two items:

1. **(Important, fixed) `station_bias` did not apply the same finite-value filter `fit_intrusion`
   uses before scoring.** `fit_intrusion` runs every observation through `_finite_rows` first;
   `station_bias` was consuming the pre-filter list. In this run it made no difference —
   `n_dropped` is 0, so nothing in this report's numbers is affected — but `station_bias` is now
   permanent output of `tidescout salinity calibrate`, and its own docstring claims the residual
   is scored "the same way the fit itself was scored." A non-finite row reaching it unfiltered
   would put a bare `nan` in a station's `mean_residual_ppt`/`rmse_ppt`, indistinguishable at a
   glance from a station that simply fits badly. Fixed: `station_bias` now runs `observations`
   through `_finite_rows` first, exactly as `fit_intrusion` does, and returns
   `(stations, n_dropped)` — the CLI prints the dropped count when it is nonzero rather than
   letting it disappear. Added `test_station_bias_drops_non_finite_rows_and_counts_them`, pinning
   that two deliberately-injected non-finite rows are excluded from the computation *and* counted,
   not silently dropped. Re-ran the real `tidescout salinity calibrate winyah-bay` before and after
   this fix and diffed the output byte-for-byte: **identical** — confirms the fix is a no-op on
   the real data (`n_dropped=0` here), and only changes behavior when a non-finite row is present.
2. **(Minor, fixed) Dangling `§4b` cross-reference in §3** — this document has no §4b (headings
   run 1–8 plus Bottom line and Appendix). Repointed to "the note just below this table," which is
   where the WYSS1/NIWWBWQ combined-entry explanation actually lives.

No number or conclusion in this report changed. `fisheries/winyah-bay.yaml`, `ocean_ppt`,
`fitted`, the model form, `ocean_boundary_utm_km`, the ANUGA mesh, the flow library, and
`ON_AXIS_MAX_KM` were not touched. `make check`: 637 passed (636 → 637, additive).

## Fix wave (final whole-branch review, 2026-08-24)

Every merge-blocking and cheap-fix item from the final review was addressed in one pass:

- **`salinity citation` now emits both stores.** It previously opened only `ndbc.default_store`,
  so `WQP_ATTRIBUTION`/`WQP_ACKNOWLEDGEMENT` (and the `_blocks_for` citation machinery in
  `ndbc.py`) were unreachable from the CLI, and SC DES/SCDHEC/Coastal Carolina — who supply 56 of
  58 distinct distances — were never credited by this command. `salinity_citation` now prints the
  NERRS store's citation, then the WQP store's (`data/<slug>/wqp.sqlite`) if that file exists,
  skipping it otherwise without creating an empty store as a side effect. Re-ran
  `tidescout salinity citation winyah-bay` against the real store: both blocks print, WQP's naming
  the EPA/USGS/National Water Quality Monitoring Council and the originating agencies. Two new
  tests cover both-stores-present and skip-when-absent.
- **Two stale `pipeline/salinity_fit.py:832` line-number citations in `wqp.py`** (plus one in
  `test_wqp.py`) now cite `collect_observations`'s docstring by name instead of a line number that
  had already drifted once.
- **Two comments this branch's own results had falsified, corrected:** `salinity_fit.py`'s module
  docstring (the off-axis decision is now computed by `is_off_axis`, not hand-marked via
  `WaterSensor.off_axis`; the bay's observations now span 28.50 km across 58 distances, not
  2.35 km) and `models.py`'s `SalinityConfig.fitted` comment (`classify_coverage` now reports
  78.7% of Winyah's cells MEASURED or INTERPOLATED, not extrapolated everywhere) — both updated to
  state the current fact while keeping the historical point, marked as history.
- **`ACCEPTED_STATUSES` no longer admits `"Historical"`**, matching `cdmo.ACCEPTED_FLAGS`'s
  blocking of flag 4 ("Historical Data: Pre-Auto QAQC"). Measured 0 of 9,306 real rows carry it,
  so this is a no-op on real data; a new test pins that it is now rejected and counted.
- **WQP rows with no composite-discharge day are now counted**, not silently dropped: a new
  `n_wqp_no_discharge_day` counter on `CalibrationInput`, surfaced in the CLI when nonzero. The
  WQP site-record loop also now filters rows to those with a matching discharge day before
  computing `n_days`, so a station cannot read `used: yes` while its full row count overstates
  what it actually contributed. Measured on the real fishery: 0 rows dropped, so this run's
  numbers are unaffected.
- **`fetch_results`/`fetch_stations` in `wqp.py` now raise `SourceUnavailable`** on a network
  fault or non-2xx response, matching `ndbc._fetch_text`/`nwi.fetch_page`/`cache.Cache.
  get_or_fetch`. Two new tests cover a bad status and a connection failure.
- **The store off-axis screen no longer reports "off axis" for a station with no known
  coordinates.** `is_off_axis` read the NaN stem distance produced by a missing surveyed position
  as "a real branch the coordinate cannot place" — the wrong reason; the true one is "nobody knows
  where this sonde is." The store path now guards on `w.station in store_coords` the same way the
  WQP path already guards on `wqp_known`. Not reachable on Winyah (every declared store station
  has a surveyed position), but a new test simulates it for the stamp-out fisheries the spec names
  (Charleston, Awendaw, Murrells Inlet).
- **`SalinityField` gains `nearest_observed_km`** (spec §4c, dropped between spec and plan with no
  note) — the raw per-cell distance to the nearest observation, array-shaped and aligned
  elementwise with `.ppt` the same way `coverage` is, NaN-consistent with no observations
  supplied. `classify_coverage` now derives its own nearest-distance computation from the same new
  `nearest_observation_km` function rather than a second copy of the same arithmetic. Six new
  tests cover both the standalone function and the `SalinityField` integration.
- **§2 of this report gains a correction paragraph** (immediately below, unchanged otherwise):
  NERRS's 2 distinct distances vs WQP's 56 means the fit's spatial SHAPE is set almost entirely by
  the phase-mismatched population while the falsifying rmse comes almost entirely from the
  phase-correct one, and `tanh`'s curvature under Jensen's inequality makes the phase-mismatch a
  systematic bias (consistent with the measured +0.593/+1.253 mean residuals), not unstructured
  noise. Framing correction only — front_width_km (14.48 → 14.68) and NERRS-only rmse
  (4.060 → 4.061) barely moved.
- **Cheap minors bundled in:** `build_site_record`'s "USGS gave no coordinates"/"no 00480 history"
  notes are now source-neutral (they fire for WQP and NERRS/store sites too, not just USGS);
  `wqp.py`'s dangling "Station coordinates need a home" docstring cross-reference is repointed to
  `NdbcStore.connection`'s real docstring; the `"wqp.sqlite"` filename literal is now defined once
  (`_WQP_DB_FILENAME`) and used by both `default_store` and `station_coords`; `salinity stem` now
  catches `FileNotFoundError` the way `salinity calibrate` already does; and
  `_stem_km_or_fallback` now distinguishes its own `load_stem_distance_field` error (degrades, as
  before) from a `grid_spec`/`read_bathy` bathymetry error (re-raises with the correct remedy —
  `tidescout bathy build`, not `tidescout salinity stem`, which would fail identically).

**Re-verified after the fix wave:** `tidescout salinity calibrate winyah-bay` against the real
store reproduces the headline numbers in §1 exactly — 12,725 observations, 58 distinct distances,
28.497082233428955 km span, `rmse_by_source_ppt` nerrs 4.061384904005473 / wqp 6.10203118406574,
front_width_km 14.68, condition number 7.94, `fitted` still False, `POOR FIT` still the only
warning. No value in `fisheries/winyah-bay.yaml`'s `salinity:` block, `ocean_ppt`, `fitted`, the
model form, `ocean_boundary_utm_km`, the ANUGA mesh, the flow library, or `ON_AXIS_MAX_KM` was
touched. `make check`: 652 passed (637 → 652, additive; +15 new tests across the items above,
0 removed).
