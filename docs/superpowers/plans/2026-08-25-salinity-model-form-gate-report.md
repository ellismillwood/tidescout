# Salinity model form — gate report

**Plan:** `docs/superpowers/plans/2026-08-25-salinity-model-form.md` (Task 4, the gate)
**Spec:** `docs/superpowers/specs/2026-08-25-salinity-model-form-design.md`
**Branch:** `plan-07-model-form`, at `c02a04d` | **Date:** 2026-08-25
**Predecessors:** `docs/superpowers/plans/2026-08-24-tidal-phase-gate-report.md` (PR #6),
`docs/superpowers/plans/2026-08-24-salinity-anchoring-gate-report.md` (PR #5)

*What scaling the front's width with discharge, and profiling an estuary memory, actually did.*


**This task measures and reports. It changes no config and decides nothing.** No value in
`fisheries/winyah-bay.yaml`'s `salinity:` block was touched, `discharge_memory_days` was not set,
`fitted` was not set True, no two-layer model was begun, and `ocean_boundary_utm_km`, the ANUGA
mesh, the flow library and `ON_AXIS_MAX_KM` were not touched. Everything below is a live re-run
against the real `data/winyah-bay/` store, captured 2026-08-25 at `c02a04d`.

Reproduce with:
```
tidescout salinity calibrate winyah-bay
```
Full captured output: `/tmp/calibrate-form.txt` (1,643 lines). Everything the CLI does not print
itself came from one standalone script over existing, tested library functions — see the Appendix.

**The targets in §3 are PRE-REGISTRATIONS, not goals.** They were computed before this work by a
method that is not recoverable from the spec, on a different population (11,688 observations; the
shipped scan reports 12,204). Nothing here was tuned, re-binned, re-scoped or re-fitted to move a
number toward a target. Two of them are missed and §9 says so in plain terms.

## Bottom line

**The primary criterion passes, and it passes on change A alone.** Measured by one function
applied identically to three model forms on one fixed 12,204-row population, the discharge-trend
spread at the two fixed-distance NERRS stations falls **3.0087 → 0.3070 ppt, an 89.8% reduction**,
and that is entirely the work of the flow-dependent front width — the change that actually ships.
Adding memory at τ=7 does not flatten it further (0.4229 ppt binned on the discharge that form
reads; 0.3030 ppt when all three forms are binned on the same raw-discharge axis). The residual's
dominant systematic response to discharge, the structure this plan existed to remove, is gone.

**The rmse target is missed and the miss is real.** The pre-registration was
`4.0875 → ~3.42`. The best measured point on the whole τ grid is **3.5602 ppt**, 0.14 ppt above
the target, and it is a form that **does not ship**. The form that ships reads **4.0669 ppt** on
that same population (**4.0806 ppt** on the full 12,725-row headline fit). This is not explained
away below; it is reported as the disagreement it is.

**Read the two rows apart or the whole report inverts.** `discharge_memory_days` defaults to `0.0`
and no fishery YAML sets it, so **what ships today is width-scaling with NO memory**. The
headline `4.0875 → 3.4163` prediction was about width-scaling **plus** memory at τ=7 — a form
this branch measures and deliberately does not adopt. Anyone reading "16.4% rmse reduction" as
live behaviour is reading the wrong row.

**Stratification is now the largest identified structure, and the spec's prediction about it was
half right in a way worth knowing.** The mean surface/bottom split cannot grow — WYSS1 and
NIWWBWQ share an exact along-estuary distance and a calendar day, so the model's prediction
cancels out of their residual difference by construction, and the split reads **+3.6222 ppt (sd
2.0855, 3,537 paired days)** under every model form tested, reproducing the pre-registered figure
to four decimals. What did move is the share of that station-pair's residual variance it explains:
**20.3% → 24.0% (shipped) → 33.3% (A+B τ=7)**. The signal got *cleaner*, exactly as predicted, but
it could not have got *larger*, and the spec's wording implies it could.

**`fitted` stays False**, by a margin of 1,159x the observation resolution at the best τ.

## 1. Which row ships, and which row the prediction was about

The spec's §2 table covers three distinct model forms. They are not interchangeable and only one
of them is live.

| # | form | `discharge_memory_days` | ships today? | spec's pre-registered rmse / spread |
|---|---|---|---|---|
| 1 | **baseline** — constant width, same-day Q | 0.0 | no (this is the *before* state) | 4.0875 / 2.96 |
| 2 | **A only** — `W(Q) = front_width_km·(Q/q0)^-k`, same-day Q | 0.0 | **YES — this is what ships** | 3.8609 / 0.55 |
| 3 | **A + B** — width scaling **plus** memory at τ=7 | 7.0 | **no — measured, not adopted** | 3.4163 / 0.27 |

Verified directly rather than assumed: `fishery.salinity.model_dump()` on this run reads
`"discharge_memory_days": 0.0`, `grep` finds no `discharge_memory_days` key in any file under
`fisheries/`, and the CLI prints `discharge memory window: 0 day(s)` alongside
`discharge_memory_days stays whatever the fishery YAML sets (0 day(s) here)`. Row 2 is production.
Row 3 is a diagnostic scan.

**The headline prediction (`4.0875 → 3.42`, "a 16.4% reduction, for ONE net new parameter (τ)")
is about row 3.** It is not a claim about anything running today. Nothing in this branch adopts
τ, `profile_memory` explicitly discards the config it builds internally, and the CLI caption says
so on every run.

## 2. Populations — read this before comparing any number

Five different denominators appear in this report and in the documents it measures against. They
are not the same row sets and comparing across them without saying so would make the gate
meaningless.

| figure | population | n | comparable to its pre-registered counterpart? |
|---|---|---:|---|
| headline `calibrate` fit | all collected observations + swings, memory 0 | 12,725 + 10,865 | **No** — see the `ocean_ppt` note below |
| τ scan / the three-form comparison | days the largest grid τ (30) retains | **12,204** | **No** — spec's A/B table used 11,688 |
| spec §2's A/B table | observations mapping to a calendar day, 2026-08-25 probe | 11,688 | — (the pre-registration itself) |
| spec §1's variance decomposition | all observations, `w0 + α·L` width | 12,725 | — |
| stratification pairing | days both 19.03 km stations report **and** discharge exists | 3,537 (3,402 on the retained set) | **Yes** — reproduces exactly, see §7 |

**Two reconciliations that change what may be compared:**

1. **`ocean_ppt` moved between PR #6's gate and this branch.** `a0cfe57` ("derive the ocean
   end-member from North Inlet instead of a convention", PR #8, an ancestor of `HEAD`) changed
   `ocean_ppt: 34.0 → 35.5`. So the previous gate's headline rmse of **4.3542 cannot be used as
   the "before" for change A** — it is a different model, not merely a different fit. The
   corroborating symptom: `n_interior_obs` reads 8,573 today against 8,667 then, on an otherwise
   byte-identical population (12,725 obs, 10,865 swings, 58 distinct distances, distance span
   28.497082233428955 km, cfs span 715.2–183,660). `n_interior_obs` counts rows falling between
   10% and 90% of `ocean_ppt`, so it moves when `ocean_ppt` moves and nothing else does.

   **Recomputing the old number on the new population** (the direction the brief asks for): the
   *constant-width* form refitted on today's identical 12,725-row headline population, at
   `ocean_ppt = 35.5`, with swings supplied and `excursion_km` freed, gives **rmse 4.3866 ppt**
   (nerrs 4.0891, wqp 5.8287, swing rmse 5.1233, condition 7.9074, `at_bounds: []`;
   `l0_km` 12.565, `k` 0.6294, `front_width_km` 14.7239, `excursion_km` 5.4492). Against the
   shipped form's **4.0806**, change A is worth **−0.3060 ppt (−7.0%)** at the headline level,
   like-for-like.

2. **The τ scan's 12,204 rows are not the spec's 11,688.** The composite discharge series holds
   10,091 days (1999-01-04 .. 2026-08-24). `smooth_discharge` at τ=30 drops **399** of them for
   insufficient preceding history (112 at τ=7), leaving 9,692; the observations landing on those
   retained days are the 12,204 every candidate τ is scored on, spanning 4,017 distinct calendar
   days (1999-05-10 .. 2026-07-14), of which 10,437 are NERRS daily means and 1,767 are WQP grabs.
   The spec's 11,688 came from a probe that "excluded 1,037 of 12,725 observations that could not
   be mapped back to a calendar day" — a different exclusion rule from
   `smooth_discharge`'s window, applied to a differently-anchored series. On this run
   `n_wqp_no_discharge_day` is **0** and `n_no_phase` is **0**, so no row here is lost the way
   that probe's 1,037 were.

**Everything in §4–§6 is measured on one fixed 12,204-row set**, so the three forms are
like-for-like against *each other* even though none of them is like-for-like against the
pre-registration. That internal comparison is what the conclusions rest on.

## 3. The gate table

| prediction | target | actual | comparable? |
|---|---|---|---|
| rmse | 4.0875 → ~3.42 | **baseline 4.4068 → A+B τ=7 3.5602**; the form that **ships** reads **4.0669** (12,204 rows) / **4.0806** (headline, 12,725) | **No.** Different population, different `ocean_ppt`, unrecoverable original method. **Target MISSED by 0.14 ppt**, and by 0.65 ppt against what actually ships. |
| **discharge-trend spread** | **2.96 → ~0.27 ppt** | **baseline 3.0087 → A only (ships) 0.3070 → A+B τ=7 0.4229** (0.3030 on a shared raw-Q axis) | **No.** The spec never defines the recipe; §4 defines one and applies it identically to all three. Direction and magnitude agree; the A-vs-B split does not. **PASSES on its own terms.** |
| fitted τ | ~7 days, worse at 3 and 14 | **argmin τ = 7**, interior to the grid, not at a bound. Worse at 3 (+0.1363 ppt) and at 14 (+0.2076 ppt), both resolved. **τ=5 vs τ=7 is NOT resolved.** | **Yes** — same grid, same code, same rows. |
| `fitted` | stays False | **False.** `POOR FIT` is the sole warning; residual is 1,159x the NERRS observation resolution at the best τ. | — |

**The primary criterion is the trend spread, not rmse, and it passed.** The spec's own test —
"if rmse falls but the trend does not flatten, the change did not do what it claims" — is
satisfied for the shipped form: the trend flattened by 89.8% while rmse fell 7.7% on the same
rows. The converse worth stating plainly is that **memory is the opposite case**: it buys a real
12.5% further rmse cut (4.0669 → 3.5602) while flattening the trend by nothing at all.

## 4. The trend spread — defined here, because the spec never defined it

**This is the gate's primary criterion and the spec gives no recipe for it.** `2.96` is not
reproducible from the spec's own quintile table, whose two per-distance ranges
(−1.33→−3.72 and +1.33→−2.03) average to **2.875**, not 2.96. So it is not quoted here as though
it were; a measurement is defined instead, and run over every form through the same code path.

### The definition

```
For each of the two fixed-distance NERRS stations (x = 16.68 km, x = 19.03 km):
    take that station's rows out of the fitted model's residuals
    sort them by THE DISCHARGE THAT MODEL ACTUALLY READS
        (raw same-day Q for a memoryless form; the tau-smoothed value for a memory form)
    split into 5 equal-count bins by rank (stable sort, so ties are deterministic)
    that station's trend := mean(residual in top bin) - mean(residual in bottom bin)

spread := mean of the two stations' ABSOLUTE trends
```

Three properties of it, stated so a reader can check rather than trust:

* **Sign convention: residual = OBSERVED − PREDICTED**, the spec's convention ("negative means the
  model predicts too salty"). This is the *opposite* of the repo's `station_bias` table, which
  prints predicted − observed. The spread is a mean of absolute differences and is therefore
  invariant to the choice; the per-quintile signs below are not, and are oriented to match the
  spec's table.
* **Distance is controlled by construction, so the trend cannot be confounded with position.**
  Verified rather than assumed: every row at each of those two exact distances in the scan
  population is a NERRS daily mean (3,439 at 16.68 km, 6,998 at 19.03 km; the assertion fails the
  script if any WQP row shares those distances). WYSS1 and NIWWBWQ share the 19.03 km distance
  bit-for-bit, so that station's bin holds both depths.
* **Each form is fitted fresh** — `l0_km`, `k`, `front_width_km` refitted at each — before its
  residuals are taken. Nothing is scored against another form's parameters.

### The 2×5 tables, all three forms, same 12,204 rows

**Form 1 — baseline (constant W, same-day Q). rmse 4.4068. Spread 3.0087.**
Fitted: `l0_km` 12.2285, `k` 0.6411, `front_width_km` 14.2512.

| quintile | x = 16.68 km, cfs range | mean resid | x = 19.03 km, cfs range | mean resid |
|---|---|---:|---|---:|
| Q1 | 1,232 – 4,343 (n=688) | −0.9001 | 1,232 – 4,313 (n=1,400) | +1.8385 |
| Q2 | 4,344 – 6,667 (n=688) | −0.3522 | 4,314 – 6,466 (n=1,400) | +2.3055 |
| Q3 | 6,667 – 10,330 (n=688) | −1.2156 | 6,468 – 9,646 (n=1,400) | +1.4576 |
| Q4 | 10,330 – 17,030 (n=688) | −2.5905 | 9,670 – 16,041 (n=1,399) | −0.0376 |
| Q5 | 17,050 – 183,660 (n=687) | −3.3387 | 16,041 – 183,660 (n=1,399) | −1.7403 |
| **trend** | | **−2.4387** | | **−3.5788** |

Monotone-ish and large, in the direction the spec diagnosed: at high flow the model is far too
salty at both distances.

**Form 2 — A only (W(Q), same-day Q). THIS IS WHAT SHIPS. rmse 4.0669. Spread 0.3070.**
Fitted: `l0_km` 9.9666, `k` 0.5728, `front_width_km` 25.0343.

| quintile | x = 16.68 km | mean resid | x = 19.03 km | mean resid |
|---|---|---:|---|---:|
| Q1 | 1,232 – 4,343 | −0.4388 | 1,232 – 4,313 | +1.1174 |
| Q2 | 4,344 – 6,667 | −1.3169 | 4,314 – 6,466 | +0.7468 |
| Q3 | 6,667 – 10,330 | −1.4540 | 6,468 – 9,646 | +0.8319 |
| Q4 | 10,330 – 17,030 | −1.1950 | 9,670 – 16,041 | +0.9714 |
| Q5 | 17,050 – 183,660 | −0.1375 | 16,041 – 183,660 | +0.8048 |
| **trend** | | **+0.3013** | | **−0.3127** |

The monotone ramp is gone at both stations. What is left is a *bowl* at 16.68 km (both extremes
less biased than the middle) and near-flat scatter at 19.03 km, and the two residual trends now
point in **opposite** directions — the signature of leftover noise, not of a remaining response
to discharge.

**Form 3 — A + B, memory τ=7. NOT ADOPTED. rmse 3.5602. Spread 0.4229.**
Fitted: `l0_km` 11.414, `k` 0.676, `front_width_km` 27.4281. Bins here are on the τ=7-smoothed
discharge, which is what this form reads.

| quintile | x = 16.68 km, smoothed cfs | mean resid | x = 19.03 km, smoothed cfs | mean resid |
|---|---|---:|---|---:|
| Q1 | 1,351 – 4,874 | −0.8659 | 1,351 – 4,889 | +0.7689 |
| Q2 | 4,883 – 7,121 | −0.8165 | 4,890 – 6,985 | +1.0873 |
| Q3 | 7,125 – 10,737 | −1.3720 | 6,989 – 10,120 | +0.9326 |
| Q4 | 10,737 – 17,842 | −1.1496 | 10,120 – 16,517 | +1.0472 |
| Q5 | 17,851 – 86,165 | −0.3772 | 16,517 – 86,165 | +0.4118 |
| **trend** | | **+0.4887** | | **−0.3571** |

### Secondary check: the same three forms, all binned on raw same-day discharge

Form 3 reads a different discharge from forms 1 and 2, so its bins are drawn on a different axis.
That is the definition the brief specifies and it is the primary number above — but it invites the
objection that the memory row was measured against its own convenient x-axis. Binning all three
identically, on raw same-day discharge, removes the objection:

| form | spread (own-Q bins) | spread (shared raw-Q bins) | trend at 16.68 | trend at 19.03 |
|---|---:|---:|---:|---:|
| baseline | 3.0087 | **3.0087** | −2.4387 | −3.5788 |
| A only — ships | 0.3070 | **0.3070** | +0.3013 | −0.3127 |
| A + B τ=7 | 0.4229 | **0.3030** | +0.3391 | −0.2670 |

The conclusion holds either way. **Change A removes ~90% of the trend. Change B removes between
none of the remainder and, at best, a further 0.13% of the original — well inside the scatter of
what is left.**

### Against the pre-registration

| | pre-registered (11,688 rows, unrecoverable method) | measured here (12,204 rows, recipe above) |
|---|---:|---:|
| baseline | 2.96 | **3.0087** |
| A only | 0.55 | **0.3070** |
| A + B τ=7 | 0.27 | **0.4229** / 0.3030 |

The **baseline agrees well** — 3.0087 against 2.96, a 1.6% difference on a different population,
and notably closer than the 2.875 the spec's own quintile table implies. That is a useful
independent check that the recipe defined here is at least the same *kind* of measurement as the
one the pre-registration used.

**The split between A and B does not agree, and that is the substantive finding of §4.** The
pre-registration expected roughly half the remaining trend to be removed by memory (0.55 → 0.27).
The measurement says change A alone reaches ~0.31 and memory has nothing left to take.

## 5. rmse

On the identical 12,204-row population, one code path, `excursion_km` held throughout (no swings
are supplied to the scan, exactly as `profile_memory` does it):

| form | rmse (ppt) | vs baseline | vs the form before it |
|---|---:|---:|---:|
| baseline (constant W, same-day Q) | **4.4068** | — | — |
| **A only (W(Q)) — SHIPS** | **4.0669** | **−7.71%** | −7.71% |
| A + B, τ=7 — not adopted | **3.5602** | −19.21% | −12.46% |

And on the full headline population (12,725 obs + 10,865 swings, `excursion_km` freed):

| metric | constant W (recomputed today) | **A only — SHIPPED** | PR #6 gate (different `ocean_ppt`) |
|---|---:|---:|---:|
| rmse overall | 4.3866 | **4.0806** | 4.3542 |
| rmse NERRS | 4.0891 | **3.7870** | 4.0596 |
| rmse WQP | 5.8287 | **5.4905** | 5.7830 |
| swing rmse | 5.1233 | **5.0855** | 5.0916 |
| `l0_km` | 12.565 | **11.2078** | 13.203 |
| `k` | 0.6294 | **0.5169** | 0.6192 |
| `front_width_km` | 14.7239 (constant) | **21.2898 (at q0)** | 14.560 (constant) |
| `excursion_km` | 5.4492 | **6.5890** | 5.528 |
| condition number | 7.9074 | **7.7208** | 8.058 |
| `at_bounds` | [] | **[]** | [] |
| `fitted` | False | **False** | False |

**`excursion_km` moved, as spec §4a required be reported rather than discovered.** 5.4492 → 6.5890
(+20.9%) between the two width forms on the identical population. The mechanism is the one the
spec named: `_swing` evaluates `salinity_at` at phases 0.0 and 0.5, so a discharge-dependent width
changes the modelled swing, and `excursion_km` is fitted against those swings. This is expected,
not a defect.

**`front_width_km`'s fitted value is not what the spec estimated.** Spec §2 said "its fitted value
under the new form is ~23.7 km at q0". Measured: **21.29 km** on the headline fit and **25.03 km**
on the τ=0 scan fit (excursion held, 12,204 rows). Both bracket 23.7 without landing on it. Since
nothing is written back to the YAML, this changes no behaviour — but the spec's figure should not
be quoted as measured.

**The ~3.42 target is missed.** The best point on the entire τ grid is 3.5602. Stated without
softening: on this population, in this codebase, at this `ocean_ppt`, the two changes together do
not reach the pre-registered rmse, and the change that ships does not come close to it. The
*relative* improvement is in fact larger than pre-registered (−19.21% measured against −16.42%
predicted) — the absolute floor is simply higher, on every form, than the probe found.

## 6. The τ profile as a curve, and whether its minimum is resolvable

Straight from the CLI's own scan table, re-run at `c02a04d` and independently reproduced twice
more by the standalone script (bit-identical rmse to four decimals on every candidate):

| τ (days) | 0 | 3 | 5 | **7** | 10 | 14 | 21 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rmse (ppt) | 4.0669 | 3.6964 | 3.5716 | **3.5602** | 3.6344 | 3.7678 | 3.9720 | 4.1776 |
| rows scored | 12,204 | 12,204 | 12,204 | 12,204 | 12,204 | 12,204 | 12,204 | 12,204 |
| `l0_km` | 9.9666 | 10.864 | 11.2177 | 11.414 | 11.5663 | 11.6724 | 11.7546 | 11.7949 |
| `k` | 0.5728 | 0.6402 | 0.6654 | 0.676 | 0.6804 | 0.6802 | 0.6789 | 0.6791 |
| `front_width_km` | 25.0343 | 26.0409 | 26.8366 | 27.4281 | 28.0831 | 28.7339 | 29.7078 | 30.8552 |

Shape: a clear U with a single interior minimum. Total range across the grid is 0.6174 ppt.

**τ does not land on a bound.** The grid's bounds are 0 and 30; the argmin is 7 — index 3 on
`(0, 3, 5, 7, 10, 14, 21, 30)`, three grid points in from the low end and four from the high end.
This is not the `at_bounds` case the spec's decision table required be flagged.

**The profile is not flat, but it is flat NEAR the minimum, and that distinction matters.** The
margin between best (τ=7, 3.5602) and second-best (τ=5, 3.5716) is **0.0114 ppt — 0.32% of the
rmse it is measured on.** To decide whether that is meaningful rather than eyeball it, the
difference was tested by a **day-clustered paired bootstrap**: rows on the same calendar day share
a discharge and a weather state and are not independent, so days (4,017 of them) are the
resampling unit, 2,000 replicates, comparing `rmse(τ) − rmse(7)` on the identical rows.

| comparison | Δ rmse (ppt) | 95% CI | verdict |
|---|---:|---|---|
| τ=5 vs τ=7 | +0.0115 | [−0.0010, +0.0237] | **NOT resolved** — CI includes zero |
| τ=10 vs τ=7 | +0.0742 | [+0.0587, +0.0905] | resolved |
| τ=3 vs τ=7 | +0.1363 | [+0.1089, +0.1646] | resolved |
| τ=14 vs τ=7 | +0.2076 | [+0.1776, +0.2378] | resolved |
| τ=0 vs τ=7 | +0.5067 | [+0.4513, +0.5607] | resolved |

**Read plainly: memory is identifiable on this data; the value 7 is not.** That some memory beats
no memory is overwhelming (0.5067 ppt, CI nowhere near zero). That 3 and 14 are worse than 7 is
the spec's exact prediction and it holds, both resolved. But **τ=5 and τ=7 are not distinguishable
at all**, so the honest statement of the result is *"the fitted timescale is in the neighbourhood
of 5–7 days"*, not *"τ = 7 days"*. Any later task that adopts a number off this curve should adopt
it as a band, and should not treat 7 as measured to the day.

**One consequence of memory that the discharge-span diagnostics would not surface.** Smoothing
compresses the discharge axis the model sees: raw same-day discharge on the scan rows spans
715.2 – 183,660 cfs (**257x**), while the τ=7-smoothed value spans 737.5 – 86,165 cfs (**117x**).
`calibration_range_cfs` and the `extrapolated` flag are defined against raw discharge. If memory
were ever adopted, that pairing would need re-deriving, or `extrapolated` would be answering a
question about a quantity the model no longer reads. Noted here because nothing in the current
code would catch it.

## 7. Stratification, re-measured — and why "larger" was never possible

Method, stated so before and after are comparable: WYSS1 (**surface**) and NIWWBWQ (**bottom**)
are paired on shared calendar days; the model's residual is taken at each; the statistic is
`mean(bottom residual − surface residual)` with its sd, plus the share of that station-pair's
pooled residual variance the depth split explains (between-group variance over total pooled
variance, equal group sizes, so between = `(Δ/2)²`). Residual is observed − predicted, matching
the pre-registration's sign.

| form | paired days | mean(bottom − surface) | sd | pooled resid var | between (depth) | **share** |
|---|---:|---:|---:|---:|---:|---:|
| **pre-registered "before this work"** | **3,537** | **+3.622** | **2.085** | — | — | **24.0%** |
| baseline (constant W), all shared days | 3,537 | **+3.6222** | 2.0855 | 16.1454 | 3.2802 | **20.3%** |
| baseline, days retained at τ=30 | 3,402 | +3.6250 | 2.0919 | 16.1012 | 3.2851 | 20.4% |
| **A only — SHIPS**, all shared days | 3,537 | **+3.6222** | 2.0855 | 13.6427 | 3.2802 | **24.0%** |
| A only — ships, days retained at τ=30 | 3,402 | +3.6250 | 2.0919 | 13.4924 | 3.2851 | 24.3% |
| A + B τ=7, days retained at τ=30 | 3,402 | +3.6250 | 2.0919 | 9.8516 | 3.2851 | **33.3%** |

Per-station means, for a reader who wants the two sides rather than the difference: under the
shipped form the surface station reads **−1.0040 ppt** and the bottom station **+2.6183 ppt**
against one depth-averaged prediction. The model sits between them and carries **±1.81 ppt of
irreducible bias** at that distance, matching spec §1 exactly.

**Three things this says.**

1. **The mean split reproduces the pre-registration to four decimals** (+3.6222 vs +3.622, sd
   2.0855 vs 2.085, 3,537 days vs 3,537). Unlike the trend spread, the stratification recipe *is*
   recoverable, and it was recovered.

2. **The mean split cannot move with model form, by construction, and did not.** Both stations sit
   at bit-identical along-estuary distance `19.029830932617188` km, both are daily means scored at
   the same `FIT_PHASE`, and on a shared day both read the same discharge — so the model's
   prediction is *the same number* for both and cancels exactly out of their difference. What
   remains is `observed_bottom − observed_surface`, a property of the sensors alone. Measured
   confirmation: +3.6222 under baseline and under A on the same 3,537 days; +3.6250 under all
   three forms on the 3,402-day retained set. **The spec's prediction that the signal would be
   LARGER afterwards was not achievable as the statistic is defined** — not wrong about the
   physics, wrong about what this particular number can express.

3. **"Cleaner" did happen, and it is the part worth carrying forward.** The share of the
   station-pair's residual variance attributable to depth rises **20.3% → 24.0%** with the shipped
   change and **→ 33.3%** with memory added, because the *other* error at that distance shrank
   (pooled variance 16.15 → 13.64 → 9.85) while the depth term stayed at a fixed 3.2802–3.2851.
   One third of the residual variance at the estuary's best-observed distance is now a depth split
   a single-layer model has no axis for.

**Note on the pre-registered 24.0%:** it matches the **A-only** row here, not the baseline row
(20.3%). That is consistent with spec §1's own footnote — the decomposition behind it was measured
"using a `w0 + α·L` width", i.e. a width that already scaled with discharge. So the figure
labelled "before this work" was in fact computed under a width-scaling model. Flagged rather than
reconciled: it does not change any conclusion, and adjusting a measurement to make it agree is
precisely what this gate must not do.

**Paired-day count moved, as the brief anticipated.** 3,537 shared days on the full record; 3,402
once restricted to days the τ=30 window retains — the memory window costs 135 paired days (3.8%).
The mean and sd are unchanged to two decimals across that restriction, so the population shift did
not drive any of the numbers above.

## 8. What is binding now

**`fitted` cannot become True.** `POOR FIT` is the sole warning this run raises — verified from
the diagnostics string, which is one paragraph:

```
POOR FIT: salinity rmse 4.081 ppt against a resolution of 0.005; tidal-swing rmse 5.086 ppt
against a resolution of 0.044. ...
```

The residual-to-resolution ratios, with the denominator named each time because the spec's
"~1,140x" is quoted against a different one than the CLI's warning uses:

| residual | ÷ combined resolution (0.0047308 on the scan set) | ÷ NERRS-only resolution (0.0030728 on the scan set) |
|---|---:|---:|
| shipped, τ=0: 4.0669 | 860x | **1,324x** |
| best measured, τ=7: 3.5602 | 753x | **1,159x** |
| spec's target, 3.4163 | 722x | **1,112x** |
| headline fit, 4.0806 (÷ 0.0045751 / 0.0029644 on the full set) | 892x | **1,377x** |

The spec's and the brief's "**~1,140x**" is the NERRS-only column, and the measured value at the
best τ is **1,159x** — slightly worse than the pre-registration assumed, and the conclusion is
identical under either denominator. **Neither change un-falsifies the model. No sentence in this
report should be read as saying the model is calibrated; it is not, and `fitted: false` in
`fisheries/winyah-bay.yaml` is unchanged.**

Every other warning stays clear, by the same or wider margins than the previous gate:
`DISTANCE COVERAGE TOO THIN` (58 distinct distances over 28.497 km against a fitted
`front_width_km` of 21.29 km at q0), `DISCHARGE SPAN TOO NARROW` (257x against a 2x floor),
`UNCONSTRAINED PARAMETERS` (condition 7.7208 against a 1e6 ceiling; every 1-sigma far below its
value — `l0_km` 0.1361, `k` 0.0085, `front_width_km` 0.3547, `excursion_km` 0.0931),
`AT THE OPTIMIZER BOUND` (`at_bounds: []`), `NO INTERIOR OBSERVATIONS` (8,573).

**The largest remaining identified structure is stratification**, and this work made it easier to
see rather than smaller: a fixed **+3.62 ppt** surface-to-bottom split at the estuary's
best-observed distance, now **24.0%** of the residual variance there under the shipped form and
**33.3%** under the memory form. It needs a **depth axis**, not another shape change to a
depth-averaged profile — no reformulation of `salinity_at` can represent two values at one cell.
That work is deliberately out of scope here and **is not begun by this branch**.

The second-largest is the **WQP-vs-NERRS gap**, which change A narrowed but did not close: WQP
5.4905 against NERRS 3.7870, an excess of **1.7035 ppt** (was 1.7396 under constant width on the
same population, and 1.7234 at PR #6). Change A took 2.1% of it. The previous gate's reading —
that WQP samples 56 distinct distances including the near-mouth reach the model fits worst, while
NERRS sits at 2 comparatively well-fit anchors — is unaffected by anything measured here.

## 9. Where a measurement disagreed with its pre-registration

Collected in one place, none of them adjusted to agree.

1. **rmse misses its target.** Pre-registered `~3.42`; best measured on the whole grid **3.5602**
   (+0.14). The form that ships reads **4.0669** / **4.0806**. §5.
2. **The baseline rmse is higher than pre-registered on every comparable framing.** 4.4068 on the
   scan population, 4.3866 on the headline population, against a pre-registered 4.0875. Every form
   in this branch sits above its pre-registered counterpart by 0.14–0.32 ppt while the *relative*
   improvement exceeds prediction (−19.2% measured vs −16.4% predicted). §2, §5.
3. **The trend spread's A/B split is wrong in the pre-registration.** Predicted 2.96 → 0.55 (A)
   → 0.27 (A+B). Measured **3.0087 → 0.3070 (A) → 0.4229** (or 0.3030 on a shared axis). Change A
   does essentially all of it; **memory contributes no measurable trend flattening**, though it
   does cut rmse by 12.5%. §4.
4. **`2.96` is not reproducible from the spec's own table**, which averages to 2.875. The recipe
   was unrecoverable and is defined for the first time in §4. §4.
5. **τ is not resolved to 7 days.** τ=5 vs τ=7 differ by 0.0114 ppt with a 95% CI of
   [−0.0010, +0.0237] — the ordering is not established. The spec's "clear optimum at 7 days with
   a sharp penalty either side" overstates the sharpness near the minimum; the penalty at 3 and 14
   is real and resolved. §6.
6. **The stratification signal did not get larger, and could not have.** The statistic is
   model-invariant by construction at a shared distance. It got *cleaner* (20.3% → 24.0% → 33.3%
   of variance). §7.
7. **The pre-registered "before this work" stratification share of 24.0% was measured under a
   width-scaling model**, not the constant-width baseline (which reads 20.3% here). §7.
8. **`front_width_km`'s fitted value is not ~23.7 km at q0** as spec §2 estimated: 21.29 km on the
   headline fit, 25.03 km on the τ=0 scan fit. §5.
9. **`fisheries/winyah-bay.yaml` now carries a pre-registered figure as though it were measured.**
   The `front_width_km` comment rewritten by Task 1 states *"Scaling the width the same way cuts
   that trend spread from 2.96 to 0.55 ppt."* Both numbers are the spec's pre-registration; this
   gate measures **3.0087 → 0.3070** by the recipe in §4. The claim is directionally right and
> **Superseded — see the Addendum at the end of this report.**

   understates the improvement. **Not corrected here** — this task is forbidden to edit the
   `salinity:` block, and the correction is a human decision. Flagged for whoever takes the next
   plan.

## 10. What this work did NOT do

* **It did not calibrate the model.** `fitted` is False, `POOR FIT` fires, and the residual is
  1,159x the NERRS observation resolution at the best point on the τ grid.
* **It did not adopt memory.** `discharge_memory_days` is `0.0` on every fishery and no YAML sets
  it. The τ scan is a diagnostic; `profile_memory` discards the config it builds.
* **It did not begin a two-layer model.** Stratification is measured (§7) and deferred, exactly as
  spec §7 requires.
* **It did not change any authored value.** The only diff to `fisheries/winyah-bay.yaml` on this
> **Superseded — see the Addendum at the end of this report.**

  branch is comment text (20 insertions, 5 deletions, all inside comment lines); every value in
  the `salinity:` block — `ocean_ppt` 35.5, `l0_km` 18.0, `q0_cfs` 4000.0, `k` 0.33,
  `excursion_km` 7.0, `front_width_km` 5.0, `calibration_range_cfs` [1232.0, 22996.0],
  `fitted: false` — is byte-identical to the branch point at `16609ce`. Confirmed by
  `git diff 16609ce..HEAD -- fisheries/winyah-bay.yaml`.
* **It did not touch** `ocean_boundary_utm_km`, the ANUGA mesh, the flow library, or
  `ON_AXIS_MAX_KM`.
* **It did not free `ocean_ppt`.** Settled in PR #8 and untouched here.
* **It did not close the 0–2.58 km observational gap.** Nothing observes it and nothing in this
  branch changes that.

## 11. One test-coverage claim this report must make honestly

Per the plan's own completion checklist: **the FIT path is proven to route through
`smooth_discharge` at the configured τ** by a spy test that fails if the call is removed or the τ
hardcoded (`test_the_fit_path_routes_its_discharge_through_smooth_discharge`, plus
`test_smoothing_at_the_configured_tau_is_not_a_no_op`).

**The PREDICTION half is NOT test-covered.** There is no production caller of
`engine.salinity.salinity_field` anywhere in `backend/tidescout/` — verified by grep:
`grep -rn salinity_field backend/tidescout/` returns four matches (the function's own definition
in `engine/salinity.py`, two module comments naming it, and `cli.py:358`'s `salinity_field` CLI
command — a same-named but unrelated command that builds its distance field via
`pipeline.estuary.build_distance_field` and never calls this function) — none of the four is a
production call site, so the substantive claim holds. The constraint spec §4b calls decisive
("if the calibration fits against smoothed discharge and the runtime path passes raw same-day
discharge, the fitted parameters do not apply and nothing will error") currently rests on a
**module note** in `engine/salinity.py`, not on a test. That note is explicit about it, and this report does not
claim parity between the two paths. Today the exposure is nil because `discharge_memory_days` is
0.0 everywhere; **the first caller that reads a memory-configured `SalinityConfig` and passes a
same-day discharge reading will be silently wrong**, and no assertion in this codebase will catch
it. Whoever adopts τ must close that gap in the same change.

## Appendix: reproducing this report

`tidescout salinity calibrate winyah-bay`, run end to end at `c02a04d` against the real
`data/winyah-bay/` store (no synthetic data), full output captured to `/tmp/calibrate-form.txt`
(1,643 lines). The τ scan table, the headline diagnostics, the fitted-config table and the
per-station bias table are printed by that command directly and were not recomputed.

Everything else — the three-form comparison, the quintile tables, the bootstrap, and the
stratification pairing — came from **one standalone script**, `/tmp/gate_measure.py`, which
exercises only existing, tested library functions: `salinity_fit.collect_observations`,
`fit_intrusion`, `smooth_discharge`, `_memory_rows_by_tau`, `_memory_row_phases`,
`_largest_tau_retained_days`, `_finite_rows`, `_by_discharge`, `_levels`,
`observation_resolution_ppt`, `daily_means_and_swings`, and `engine.salinity.salinity_at` — the
same functions `fit_intrusion` itself calls. The baseline form was produced by a context manager
that temporarily rebinds `engine.salinity.front_width_at` to return `cfg.front_width_km`
unscaled, in that process only; **no repository file was edited to produce any number here.**
The script asserts, and would fail on, three preconditions: that every row at each fixed station
distance is a NERRS row, that the τ=0 and τ=7 row sets share identical distances and salinity
values, and that its own recomputed rmse matches `fit_intrusion`'s to 1e-12.

The τ profile was independently reproduced twice by that script, bit-identical to the CLI's table
at four decimals on all eight candidates, on top of the two reproductions the plan already had.
The bootstrap uses seed 20260825; its point estimates are exact and only the CI endpoints vary
between runs (τ=0 vs τ=7 gave [+0.4512, +0.5611] and [+0.4513, +0.5607] on two draws).

**`make check`: 702 passed, ruff clean, tree clean.** Test count 681 at plan start → **702** now,
additive throughout, as required.

> **Superseded — see the Addendum at the end of this report.**

**Status: no code change was needed to produce this report.** `fisheries/winyah-bay.yaml`'s
`salinity:` block values, `ocean_ppt`, `ocean_boundary_utm_km`, the ANUGA mesh, the flow library,
`ON_AXIS_MAX_KM`, `engine/salinity.py` and `pipeline/salinity_fit.py` were all left untouched by
this task. This report is the gate's entire output. Whether width scaling stays, whether τ is ever
adopted and at what band, whether the YAML comment in §9.9 is corrected, and whether a depth axis
is the next plan are the owner's decisions, not this task's.

## Addendum (added 2026-08-26, after `8079990`)

This report was written and pinned at `c02a04d`, and everything above is left as written — its
measurements, tables and conclusions are unchanged. Two things happened on this same branch
immediately afterward, before a separate whole-branch review closed out further prose findings
(including the ones below); a reader merging this branch should take the following as current,
not §9.9, §10's diff-stat line, or the Appendix's "Status" line above:

* **Finding 9 (§9.9) is closed.** Commit `8079990`, the very next commit on this branch, replaced
  the pre-registered "2.96 to 0.55" figure in `fisheries/winyah-bay.yaml`'s `front_width_km`
  comment with this gate's own measurement (3.0087 → 0.3070) and an explicit note distinguishing
  it from the superseded pre-registration it replaced. §9's point 9, and §10's "**Not corrected
  here**" / "Flagged for whoever takes the next plan" language, describe a state that no longer
  holds — read them as history of what this task found, not as an open to-do.
* **§10's diff-stat line is stale.** §10 states the YAML diff against `16609ce` is "20 insertions,
  5 deletions." That was accurate at `c02a04d`. `8079990` grew it to 35 insertions, 5 deletions
  (comment-only, per that commit's own message). A subsequent editing pass — closing this same
  branch's whole-branch review, including a companion finding that several OTHER pre-registered
  figures in this same YAML comment block and in `engine/salinity.py` had likewise shipped as
  measurements — added further comment-only clarifications, bringing it to **40 insertions, 5
  deletions** as of this addendum. Every value in the `salinity:` block remains byte-identical to
  `16609ce` throughout, each time reverified by `yaml.safe_load` equality rather than by eyeballing
  the diff.
* **The Appendix's "Status: no code change was needed to produce this report" line** is true of
  what THIS task produced at `c02a04d` and is not rewritten above. It should not be read as a
  claim that the files it lists are still byte-identical to that commit: `fisheries/winyah-bay.yaml`
  has since received the comment-only correction above (and further comment-only corrections to
  the same block), and `engine/salinity.py` / `models.py` have likewise had stale or
  pre-registration-as-measurement prose corrected — comments and docstrings only, no behaviour
  change, `fitted` still False, `discharge_memory_days` still 0.0.

No measurement, table, or conclusion elsewhere in this report is altered by this addendum.
