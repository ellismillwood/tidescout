# Salinity Model Form: Flow-Dependent Front Width and Discharge Memory — Design Spec

**Date:** 2026-08-25
**Status:** Approved (brainstormed from measurement; design approved in chat)
**Owner:** Ellis Millwood
**Follows:** PRs #4-#8 and their gate reports, notably
`docs/superpowers/plans/2026-08-24-tidal-phase-gate-report.md`.
**Constrains:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` §7, §10.

## 1. Why this exists

Five merged PRs eliminated, by measurement, every cheap explanation for the salt-intrusion
model's misfit: the distance field routed the bay out the wrong mouth (fixed), coverage was
2 distinct distances (now 58), tidal phase was uncorrected (now carried), and `ocean_ppt` was an
unsourced convention (now measured). `fitted` is still False and `POOR FIT` is still the sole
warning. What is left is the model form itself.

This spec addresses the **two largest structures actually present in the residual**, which are
NOT the two the earlier gate reports assumed.

### The diagnosis, measured 2026-08-25

**The dominant systematic error is the response to discharge, and it is not confounded with
position.** Controlling for distance by using the two NERRS stations (which sit at fixed x and
carry 10,865 rows), the mean residual trends monotonically with flow:

| | lowest-flow quintile | highest-flow quintile |
|---|---|---|
| x = 16.68 km (n=3,574) | −1.33 ppt | −3.72 |
| x = 19.03 km (n=7,291) | +1.33 | −2.03 |

Negative means the model predicts too salty. At high flow it is far too salty even though its
intrusion length has collapsed.

**The mechanism is a constant front width.** `L(Q)` ranges **37.14 km → 1.13 km** across the
observed 257x discharge span, while `front_width_km` is a constant. The front cannot be sharp at
high flow and broad at low flow at once, so the tanh transitions too slowly wherever `L` is small.

**After fixing width, 90.7% of the remaining variance is WITHIN-distance scatter.** Decomposed
over the 58 distance groups: between-group (structural bias a profile shape could still capture)
1.520 ppt², **9.3%**; within-group 14.748 ppt², **90.7%**. So a perfect along-estuary profile
would take rmse only from 4.03 to ~3.84. The profile shape is no longer where the error lives.

> **Two populations are quoted in this spec and they are not interchangeable.** The decomposition
> above was measured on ALL 12,725 observations using a `w0 + α·L` width. The A/B table in §2 was
> measured on the **11,688** observations that map back to a calendar day (memory needs a date, so
> the comparison had to hold the population fixed across all rows of that table). Do not compare
> 4.03 against 4.0875 and conclude anything — they are different row sets. Within each block the
> comparisons are like-for-like.

**Two things drive that scatter, both measured:**

1. **Stratification.** WYSS1 and NIWWBWQ share one piling and differ only in depth. Over 3,537
   shared days the bottom reads **+3.622 ppt** above the surface (sd 2.085), and the split alone
   explains **24.0%** of the variance at that distance. A depth-averaged model must sit between
   them, carrying **±1.81 ppt of irreducible bias** there. **No reformulation of the along-estuary
   profile can touch this** — the model has no depth axis. It is out of scope here, deliberately,
   and is the clean next target once the two changes below land.

2. **Estuary memory.** The residual correlates with discharge averaged over PRIOR days, and the
   correlation strengthens with lag before weakening — WYSS1 −0.06/−0.13/**−0.22**/−0.15 and
   NIWWBWQ −0.23/−0.39/**−0.46**/−0.37 at 1/7/14/60 days. The bottom sensor shows it roughly twice
   as strongly, which is physically right: the bottom layer carries the long-memory salt wedge.
   The model reads **same-day discharge only**.

## 2. The two changes, and their measured effect

Fitted on the identical 11,688 observations that map to a calendar day:

| form | rmse | discharge-trend spread |
|---|---|---|
| baseline (constant W, same-day Q) | 4.0875 | 2.96 ppt |
| **A.** `W(Q) = front_width_km · (Q/q0)^-k` | **3.8609** | **0.55** |
| **A + B**, memory τ = 3 d | 3.5133 | 0.39 |
| **A + B, memory τ = 7 d** | **3.4163** | **0.27** |
| A + B, τ = 14 d | 3.6422 | 0.55 |
| A + B, τ = 21 d | 3.8619 | 0.83 |
| A + B, τ = 30 d | 4.0691 | 0.92 |

**Together: rmse 4.0875 → 3.4163, a 16.4% reduction, for ONE net new parameter (τ).**

### A note on τ, because reasoning alone gets it wrong

The raw residual correlation peaks at **14 days**, and that is what an analyst reading only the
correlation would choose. Fitting τ properly gives a clear optimum at **7 days**, with a sharp
penalty either side — at 30 days the fit is no better than no memory at all. Correlation-with-
residual and best-fit-timescale are different questions and only the second is being asked. τ is
to be FITTED, not authored, and its profile is to be reported.

### Why `front_width_km` is reparametrised rather than replaced

`W = α·L` fits equally well (rmse 4.0430 vs 4.0333 for `w0 + α·L`) and removes a parameter, but
leaves a bare dimensionless α in the config. Since `L = l0·(Q/q0)^-k`, the form
`W(Q) = front_width_km · (Q/q0)^-k` is the **same family**, adds no parameter, and keeps
`front_width_km` meaning something a reader can hold: *the front's width at the reference
discharge `q0_cfs`*. Its fitted value under the new form is ~23.7 km at q0.

**Consequence for the fishery YAML:** `front_width_km: 5.0` is currently authored as a constant
width with a documented derivation (a 3-8 km sweep). Under the new form its meaning changes, so
that comment becomes false and must be rewritten. The VALUE is a theoretical starting point and
`fitted` is False, so nothing fitted is written — but the comment must describe the new meaning.

## 3. Decisions taken

| Decision | Choice |
|---|---|
| Scope | Width and memory only. **Two layers explicitly deferred** — re-diagnose first. |
| Width form | `W(Q) = front_width_km · (Q/q0)^-k` — reparametrise, do not replace |
| Memory form | Exponentially-weighted mean of prior daily discharge, timescale τ |
| τ | **Fitted**, not authored. Report its profile. Bounded to a physically defensible range (roughly 1-60 days); a τ resting on a bound is not a fitted value and must be reported as such, per this repo's existing `at_bounds` convention. |
| Stratification | Out of scope. Measured at +3.622 ppt; needs a depth axis, not a shape change. |

## 4. Architecture

### 4a. Flow-dependent front width

`engine/salinity.py:salinity_at` currently computes
`ocean_ppt * 0.5 * (1 - tanh((x_eff - L) / cfg.front_width_km))`.
The denominator becomes `front_width_km * (Q/q0_cfs)^-k`, i.e. the same scaling `L` already
carries. `intrusion_length_km` already computes that factor; the width must reuse it rather than
recompute the exponent independently, so the two cannot drift.

This changes the model's predictions everywhere, so it is NOT backward compatible and must not
pretend to be. Every existing shape test must be re-examined against the new form rather than
mechanically updated to pass.

**Swings are affected too, and this is easy to miss.** `_swing` evaluates `salinity_at` at phases
0.0 and 0.5 and takes the difference, so a discharge-dependent width changes the modelled tidal
swing as well as the level — most at high flow, where the front is now sharp and a given tidal
excursion sweeps a steeper gradient. `excursion_km` is fitted against those swings, so its fitted
value will move. That is expected, not a defect; it must be reported rather than discovered.

### 4b. Discharge memory

The model consumes ONE discharge number. Memory changes what that number MEANS: from "today's
composite discharge" to "an exponentially-weighted mean of composite discharge over the preceding
days, timescale τ".

**The critical constraint: it must be computed identically at fit time and at prediction time.**
If the calibration fits against smoothed discharge and the runtime path passes raw same-day
discharge, the fitted parameters do not apply and nothing will error. τ therefore belongs in
`SalinityConfig`, and the smoothing belongs in one shared function both paths call.

`salinity_at`'s signature does NOT change — it still takes one `cfs`. The caller supplies the
smoothed value.

**Missing history is a rejection, not a default.** An observation whose preceding τ-window is not
covered by the discharge record must be excluded and counted, never smoothed over a short window
and treated as equivalent. The probe behind this spec excluded **1,037 of 12,725** observations
that could not be mapped back to a calendar day; the implementation must handle that population
explicitly and report it, not silently drop it.

## 5. What success looks like — pre-registered

State these before running, and report them against the outcome:

- **rmse 4.0875 → ~3.42** on the mapped population.
- **Discharge-trend spread 2.96 → ~0.27 ppt** — the trend is the thing being removed; rmse is a
  side effect. If rmse falls but the trend does not flatten, the change did not do what it claims.
- **τ fits near 7 days**, with a worse fit at 3 and at 14. If it lands at a bound or the profile is
  flat, memory is not identifiable on this data and that must be reported as such.
- **`fitted` stays False.** 3.42 ppt is still ~1,140x the observation resolution. Neither change
  un-falsifies the model and no report may imply otherwise.
- **The stratification signal should be LARGER and cleaner afterwards**, not smaller — removing
  two competing structures makes the remaining depth split easier to see. That is the point of
  doing these first.

## 6. Testing

- Shape tests for the new width: the front must be demonstrably sharper at high discharge and
  broader at low, pinned numerically rather than by inspection.
- The width factor and `intrusion_length_km` must be proven to use the same exponent, so a change
  to one cannot silently diverge from the other.
- Memory: a known discharge series with a known τ must produce a hand-computable smoothed value.
- **Fit-time and prediction-time smoothing must be proven identical** — a test that would fail if
  one path smoothed and the other did not. This is the failure mode that would silently invalidate
  every fitted parameter.
- An observation with insufficient discharge history is excluded AND counted.
- Every existing `engine/salinity.py` shape test re-examined, with any change justified against
  the new form rather than adjusted until green.

## 7. Out of scope

- **A two-layer / stratified model.** The largest single identified bias (+3.622 ppt) and the
  obvious next plan, deliberately deferred so its benefit can be measured against a cleaner
  residual.
- Freeing `ocean_ppt`. Settled in PR #8: an honestly-weighted anchor still yields 22.25 ppt
  because a falsified model absorbs misfit into whatever is free.
- Any change to `ocean_boundary_utm_km`, the ANUGA mesh, or the flow library.
- The 0-2.58 km observational gap. Nothing observes it and this does not change that.
