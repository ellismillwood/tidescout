# Tidal Phase and the Ocean End-Member — Design Spec

**Date:** 2026-08-24
**Status:** Approved (brainstormed, design approved in chat)
**Owner:** Ellis Millwood
**Follows:** `docs/superpowers/specs/2026-08-24-salinity-anchoring-design.md` and its gate report,
`docs/superpowers/plans/2026-08-24-salinity-anchoring-gate-report.md` (PR #5).
**Constrains:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` §7, §10.

## 1. The two problems, in order

### 1a. Tidal phase is not carried into the fit — the binding one

`pipeline/salinity_fit.py` defines `Observation = tuple[float, float, float]` — distance, discharge,
salinity. No phase. `_levels` therefore evaluates **every** observation at `FIT_PHASE = 0.25`, the
phase at which `cos(2*pi*phase)` is exactly zero, i.e. the tidal **average**.

That is correct for a daily mean and wrong for an instantaneous grab sample. Measured on the real
fit (PR #5's gate report, re-verified):

```
NERRS daily means  n=10,865  rmse 4.061 ppt  mean residual +0.593
WQP grab samples   n= 1,860  rmse 6.102 ppt  mean residual +1.253
```

The model's own tidal swing at the distances where grabs sit is **8.3-12.3 ppt**, so a grab taken
near high or low water is off by 4-6 ppt from phase mismatch alone.

**Why this matters more than the rmse gap suggests.** NERRS supplies 10,865 rows at **2** distinct
along-estuary distances; WQP supplies 1,860 rows at **56**. So the fit's entire *spatial shape* —
`l0_km` and `front_width_km`, which is precisely what a scoring layer reads — is set almost
exclusively by the population scored at the wrong phase, while the rmse carrying the falsification
argument comes almost exclusively from the population that constrains no shape at all.

**And the error is biased, not noise.** `salinity_at` is a `tanh` of `x + E*cos(2*pi*phase)`.
Evaluating instantaneous samples at `FIT_PHASE` is not zero-mean even under a uniform phase
distribution — Jensen's inequality gives a **distance-dependent systematic offset**, largest where
curvature is largest, which is the 4-13 km reach PR #5 just anchored. The measured mean residuals
above (+0.593 vs +1.253) are consistent with bias rather than noise.

Until this is fixed, **"2 -> 58 distinct distances" must not be read as "the profile shape is now
constrained."** That sentence is the reason this spec exists.

### 1b. `ocean_ppt` is held at an unmeasured value

`ocean_ppt` is held at 34.0 in `fisheries/winyah-bay.yaml` and has never been measured — Task 4 of
Phase 2 verified no CO-OPS station within 250 km serves salinity. It is the seaward anchor the
whole profile decays from, and it governs the 0-2.58 km band that remains EXTRAPOLATED.

**This is second, deliberately.** The only data near the mouth is WQP grab samples — exactly the
phase-contaminated population, in exactly the reach where the Jensen bias is largest. Freeing
`ocean_ppt` before carrying phase would fit the seaward anchor against the worst data available.

## 2. Facts verified before this spec was written

Each of these was measured, not assumed:

| fact | value |
|---|---|
| WQP grab samples in the store | 8,208 |
| unique local dates needing a phase | **1,260** |
| date range | 1999-01-04 .. 2026-06-24 (27.5 years) |
| CO-OPS yearly fetches required | **28** |
| CO-OPS serves hi/lo predictions for 1999 / 2008 / 2026 | **yes**, all HTTP 200 with events |
| sampling hour-of-day | median 11.4 h, p10-p90 9.8-13.4 h, **98.4% in 08:00-16:00** |

**The narrow sampling window does not bias tidal phase.** Samples cluster in a ~3.6 h band of the
*solar* day, but the M2 period is 12.42 h and precesses ~50 min/day against the solar day, so a
fixed clock-time window sweeps all tidal phases over a ~29.5-day cycle. Across 27.5 years the phase
distribution is effectively uniform. (This does **not** rescue the current code: the Jensen bias in
§1a is a property of evaluating a nonlinear function at the mean, and uniformity does not remove it.)

### Up-estuary phase lag is negligible — one tide station suffices

Measured empirically from the NERRS stations' own `depth_m` records against CO-OPS predicted highs
for station 8662549, over 2026-07-01..21 (37-39 matched highs each):

| station | along-estuary | median lag | phase units |
|---|---|---|---|
| NIWTAWQ | 16.68 km | −2.0 min | −0.003 |
| WYSS1 | 19.03 km | +4.0 min | +0.005 |
| NIWWBWQ | 19.03 km | +8.0 min | +0.011 |

All within ±8 minutes, i.e. |lag| <= 0.011 phase units, against the ~0.25 phase-unit error the
current code makes. **No per-location lag model is needed.** This was the largest complexity risk
in the work and it is measured away.

*Limit of this measurement, stated rather than glossed:* it covers 16.68-19.03 km. Grabs span
4.44-32.94 km. Above 19 km the lag is unmeasured — but the model's own swing there is smaller
(4.85 ppt at 25 km vs 12.29 at 10 km), so a given phase error costs less exactly where the lag is
least known.

## 3. Decisions taken

| Decision | Choice |
|---|---|
| Order | **Phase first, then `ocean_ppt`** |
| Phase source | CO-OPS hi/lo predictions, station `8662549`, yearly chunks |
| Per-location lag | **Not modelled** — measured at <= 0.011 phase units |
| Interface | A parallel `phases` sequence, mirroring the existing `sources` argument |
| Daily means | Keep `FIT_PHASE` — they genuinely ARE tidal averages |
| Undeterminable phase | **Exclude and count**, never default |

## 4. Architecture

### 4a. Phase derivation

A function mapping a tz-aware timestamp to a model phase in [0, 1), where **0 = low water and
0.5 = high water** — the convention `engine/salinity.py` already uses and which
`.superpowers`-recorded project notes pin as "phase 0 is LOW water (flood = first half)".

Derived by locating the bracketing hi/lo events and interpolating linearly in time between them:
low -> high maps to 0 -> 0.5, high -> low maps to 0.5 -> 1.0. Linear-in-time between hi/lo is the
same approximation `engine/tides.py`'s existing `_cosine_height`/`stage_at` machinery already
rests on; this spec does not introduce a new tidal model, it reuses the one in place.

Predictions are fetched in yearly chunks and cached. `sources/noaa.py` already sets
`PREDICTION_TTL = None` because predictions are deterministic, so this is a one-time cost per
fishery that never re-fetches.

### 4b. Carrying phase into the fit

`fit_intrusion` gains `phases: Sequence[float] = ()`, a parallel sequence to `observations`,
**mirroring the `sources` argument added in PR #5** — same length validation against the
pre-filter observation list, same `ValueError` on mismatch, same backward compatibility.

- Empty `phases` means every observation evaluates at `FIT_PHASE`, so every existing caller is
  unchanged and no test needs rewriting to accommodate the new parameter.
- `_levels` evaluates each observation at its own phase rather than one global constant. The
  existing grouping by discharge must not silently collapse observations that now differ in phase.
- `_swing` is **untouched**. It already evaluates at 0.0 and 0.5 deliberately, and swings are a
  different quantity from levels.

### 4c. Which observations get a real phase

- **WQP grab samples:** their computed phase.
- **NERRS daily means:** `FIT_PHASE`, and this is correct rather than a fallback — a daily mean IS
  a tidal average, and 0.25 is the phase at which the model's tidal term vanishes.
- **USGS daily means:** likewise `FIT_PHASE`.
- **A grab whose phase cannot be determined:** excluded and counted, surfaced in the CLI. This
  repo already rejects rows with no usable timestamp at parse time on the reasoning that "a
  fabricated time is a fabricated phase"; defaulting a phase here would be the same error one
  layer down.

### 4d. Then, and only then, `ocean_ppt`

With phase carried, revisit the gate report's recommendation. Two routes it identified:

- **(a) Free `ocean_ppt` in the fit**, with North Inlet's continuous record entering as an
  explicitly off-axis anchor term — not folded into the on-axis distance coordinate, which the
  fishery YAML's own note says a single axis cannot carry.
- **(b) Hold it fixed at a value *derived* from North Inlet** (a flow-conditioned mean or
  percentile) rather than the current unsourced 34.0 — smaller, with most of the benefit.

The evidence for either, from the gate report: North Inlet's three stations hold 305,780-365,771
readings each, means 31.37-32.04 ppt, maxima near 39. They are ocean-flushed by their own inlet
rather than by the river the intrusion model describes, which is what makes them a cleaner
end-member proxy than any on-axis station. The best on-axis candidate, WB-06 at 5.56 km, has
n=34 and a 35.4 ppt maximum but a 23.21 ppt mean — still inside the estuary's own gradient.

**Which route is chosen is a decision for the owner at a gate, after phase lands and the refit is
measured.** This spec does not pick one.

## 5. What success looks like

The per-source rmse split added in PR #5 makes this directly measurable. Report, before and after:

- `rmse_by_source_ppt` — the WQP figure (6.102) is what should move; NERRS (4.061) should not.
- `l0_km` and `front_width_km` — the shape parameters, currently 13.33 and 14.68.
- Per-station bias, which PR #5 already computes for all 58 admitted stations.
- The count of grabs excluded for undeterminable phase.

**Honest expectation, recorded so the gate is not read as optimism:** carrying phase removes a
known artefact. It does not add coverage, does not touch the 0-2.58 km band, and does not address
stratification — a measured +3.30 ppt median between surface and bottom at one distance, which a
depth-averaged single-layer model cannot represent. `fitted` becoming True is not the expected
outcome and should not be treated as the success criterion. The expected outcome is that the
falsification argument becomes **clean**: resting on all 12,725 observations rather than on the
10,865 that happen to be phase-correct.

## 6. Testing

- Phase derivation against hand-computed cases: exactly at a low (0.0), exactly at a high (0.5),
  midway on the rise (0.25) and on the fall (0.75), and either side of a day boundary.
- A timestamp with no bracketing events available -> excluded and counted, never defaulted.
- `phases` length mismatch -> `ValueError`, matching `sources`' existing contract.
- Empty `phases` reproduces today's behaviour **exactly** — this is the backward-compatibility
  guarantee that keeps every existing caller and test valid, and it deserves an explicit test.
- A real-data regression pinning the measured lag figures in §2, so a future change to the tide
  station or the phase convention cannot silently invalidate the "one station suffices" ruling.
- Round-trip against real CO-OPS data for at least one historical year, following this repo's
  standard that a parser is not trusted until it has met a real response.

## 7. Out of scope

- Stratification / a two-layer model. Named repeatedly as a falsification cause; not designed here.
- Any change to `ocean_boundary_utm_km`, the ANUGA mesh, or the flow library.
- Deriving spatial structure from the flow library's velocity fields.
- Closing the 0-2.58 km observational gap. Nothing observes it and this work does not change that.
