# The Score's Dynamic Range — Why the Day Curve Is Flat

**Date:** 2026-09-04
**Status:** Findings — measures a property of the shipped scoring design. No change proposed;
any change here is a calibration decision blocked on ground truth.
**Owner:** Ellis Millwood
**Explains:** the hour strip's near-uniform bars, first noticed on the Phase 4 frontend
(~12 px of spread on an 82 px plot).
**Bears on:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` §8 (scoring and
combination), `docs/superpowers/plans/2026-08-16-hindcast-log.md` (weight tuning).

## 1. Why this note exists

The bite score is a 0–100 number, and across every payload on disk it uses **58–85**. The hour
strip therefore reads as a row of near-equal bars, and the obvious reading — "something is
broken" — is wrong. Nothing is broken. The compression is a straightforward arithmetic
consequence of the combination rule spec §8 asks for, and it is worth writing down once so the
question is not re-opened as a bug.

**What is genuinely open** is whether the resulting range is the right one. That is a
calibration question, and calibration is blocked on the hindcast `actual` column.

## 2. The mechanism

`engine/score.py:391` `combine` is a weighted geometric mean over the present factors:

```python
log_sum = sum(s.weight * math.log(max(s.value, SCORE_FLOOR)) for s in present)
raw = min(max(math.exp(log_sum / live_weight), 0.0), 1.0)
```

Equivalently:

```
ln(score) = Σ (wᵢ / W) · ln(vᵢ)          W = Σ wᵢ
```

Each factor's swing enters **damped by its weight share** `wᵢ/W`. With nine factors and
W ≈ 6.3–7.1, no single factor exceeds a share of 0.16 — including `flow`, the highest-weighted
at 1.00. A factor that triples in value moves the score by 3^0.16 ≈ 1.19×.

This is not a flaw in the implementation. It is what a weighted geometric mean over nine terms
does, and §8 chose it deliberately.

## 3. Three compounding causes, measured

All figures below: `redfish`, `2026-09-04`, the shipped payload.

### 3.1 Dilution across nine factors

Each factor's marginal effect on the 0–100 score, holding the other eight at their daily mean:

| factor | weight | moves the score |
|---|---|---|
| flow | 1.00 | **11.0 pts** |
| stage | 0.90 | 7.6 |
| light | 0.70 | 7.1 |
| pressure | 0.60 | 4.7 |
| solunar | 0.30 | 2.2 |
| wind | 0.70 | 1.9 |
| salinity | 0.50 | 1.8 |
| water_temp | 0.80 | **0.0** |
| season | 0.80 | **0.0** |

Rebuilding the mean from subsets shows the dilution directly:

| combination | score range | spread |
|---|---|---|
| `flow` alone | 19.5 – 54.5 | **35.0** |
| `flow` + `stage` | 36.2 – 72.2 | 36.0 |
| the 7 varying factors | 56.2 – 75.7 | 19.5 |
| **all 9 (shipped)** | **62.7 – 78.3** | **15.6** |

`flow` on its own would swing the score 35 points across the day. Inside the nine-factor mean it
moves it 11, and the combined result is 15.6.

### 3.2 A quarter to a third of the weight cannot vary within a day

`water_temp` and `season` are **constant across all 24 hours** — they are day-level quantities
sitting inside an hour-level average. They contribute exactly zero variance while occupying
weight in the denominator, shrinking every other factor's share:

| species | constant weight | share of W | spread with them | without | gain |
|---|---|---|---|---|---|
| redfish | 1.60 | 25.4% | 15.6 | 19.5 | **×1.25** |
| speckled_trout | 1.80 | 25.4% | 20.3 | 27.2 | **×1.34** |
| southern_flounder | 2.00 | 31.2% | 18.1 | 27.2 | **×1.50** |

This is the largest tractable lever in the system, and the only one that is arguably a modelling
mismatch rather than a tuning choice: the two factors are doing real work on the day's *level*,
but they are applied in the one place where they can only flatten its *shape*.

### 3.3 Cancellation removes about 57%

The nine marginal contributions in §3.1 sum to **36.3 points**. The realised spread is **15.6**.

In log space: observed `sd(ln score)` = **0.0625**, against **0.1607** if every factor moved
together — a **2.57× reduction**. The independence bound is 0.0713, so the factors are close to
independent with mild net cancellation. This is structural, not accidental: `light` peaks at dawn
and dusk, `stage` peaks on the tide, and the two are unrelated clocks.

| factor | share `wᵢ/W` | `sd(ln vᵢ)` | contribution | of total swing |
|---|---|---|---|---|
| flow | 0.159 | 0.303 | 0.0480 | 29.9% |
| stage | 0.143 | 0.242 | 0.0346 | 21.5% |
| light | 0.111 | 0.281 | 0.0313 | 19.5% |
| pressure | 0.095 | 0.186 | 0.0177 | 11.0% |
| salinity | 0.079 | 0.127 | 0.0101 | 6.3% |
| wind | 0.111 | 0.088 | 0.0098 | 6.1% |
| solunar | 0.048 | 0.194 | 0.0092 | 5.7% |
| water_temp | 0.127 | 0.000 | 0.0000 | 0.0% |
| season | 0.127 | 0.000 | 0.0000 | 0.0% |

### 3.4 The factors never approach zero

A quieter fourth cause. On these days the sub-scores live in the **upper half of [0, 1]**: `wind`
never drops below 0.785, `salinity` below 0.723, `stage` below 0.465. The geometric mean's
defining behaviour — "a near-zero critical factor tanks the hour rather than averaging away",
the entire stated reason §8 chose geometric over arithmetic — therefore **never engages on an
ordinary day**. It is correct insurance against conditions that did not occur in this sample.

## 4. What range is actually used

Four days × three species × 24 hours = 288 scored hours:

| species | n | range | spread | mean | sd |
|---|---|---|---|---|---|
| redfish | 96 | 62 – 78 | 16 | 70.8 | 4.03 |
| speckled_trout | 96 | 58 – 79 | 21 | 67.7 | 4.90 |
| southern_flounder | 96 | 65 – 85 | 20 | 75.3 | 4.89 |

**All 288 hours span 58–85 — 27 of 100 points**, p1–p99 of 60–84. The hindcast log's three
dates, chosen for regime diversity, span 51–85. The bottom half of the scale is never used.

### 4.1 A caveat on between-day differentiation

Measured on the four consecutive payloads on disk, day-to-day movement looks almost absent —
redfish daily means of 71.5 / 71.1 / 70.3 / 70.2, a between-day sd of 0.53 against a within-day
sd of 3.99.

**That sample understates it.** All four are early-September days in similar conditions. Across
the hindcast log's genuinely different regimes — 2,318 vs 8,854 cfs discharge, neap vs spring,
83 vs 87 °F — the redfish daily peak moves 74 → 77 → 82. The score does respond to real regime
change. Any future claim about between-day resolution needs a sample chosen for regime spread,
not four adjacent dates.

## 5. This is not a defect

Every mechanism above is the specified design computing correctly:

- The geometric mean is what §8 asks for, for the reason §8 gives.
- Missing-factor renormalisation, `confidence` and `constrained_share` all behave as documented.
- No factor curve is misbehaving; they are simply being averaged.

The open question is whether 58–85 is the right range for the product's purpose, and that cannot
be settled from the engine. `docs/superpowers/plans/2026-08-16-hindcast-log.md` states the
constraint in its own header: **no row is usable for tuning until the `actual` column is filled
in by Ellis**, and filling it with plausible-looking guesses would tune every future weight
against fiction while looking rigorous.

## 6. Levers, if the range is judged wrong

Recorded in order of leverage. None is recommended here; each needs ground truth first.

1. **Move `water_temp` and `season` out of the hourly mean** (§3.2). They are day-level
   quantities. Applied as a day multiplier they would keep their influence on the day's level
   without flattening its shape — worth ×1.25–1.50 on within-day spread. This is the only lever
   that corrects a scope mismatch rather than retuning a judgement.
2. **Steepen the factor curves** so they use more of [0, 1] (§3.4). `wind` spanning 0.785–0.999
   across a whole day is barely participating.
3. **Concentrate weight** on fewer, more dynamic factors (§3.1) — accepting that this trades away
   the breadth the ten-factor explainability panel is built on.
4. **Rescale the output.** Cosmetic: it changes the numbers a reader sees without adding any
   information, and would break comparability with every score recorded to date.

## 7. Reproducing the measurements

Run against the shipped payloads; no network, no rebuild.

```python
import gzip, json, glob, numpy as np

d = json.load(gzip.open("data/winyah-bay/payloads/2026-09-04-best.json.gz"))
hrs = d["species"]["redfish"]["hours"]
names = np.array([s["factor"] for s in hrs[0]["subs"]])
V = np.array([[s["value"] for s in h["subs"]] for h in hrs])
w = np.array([s["weight"] for s in hrs[0]["subs"]])

def gm(V, w):                                     # combine(), vectorised
    return 100 * np.exp((np.log(np.clip(V, 1e-9, None)) * w).sum(axis=1) / w.sum())

allf = gm(V, w)
varying = np.where(V.std(axis=0) > 1e-9)[0]       # drops water_temp, season
print(allf.max() - allf.min())                    # 15.6  (shipped)
print(np.ptp(gm(V[:, varying], w[varying])))      # 19.5  (x1.25)

# marginal effect of one factor, others held at their daily mean
base = np.log(np.clip(V, 1e-9, None)).mean(axis=0)
for i in range(len(names)):
    L = np.tile(base, (24, 1)); L[:, i] = np.log(np.clip(V[:, i], 1e-9, None))
    s = 100 * np.exp((L * w).sum(axis=1) / w.sum())
    print(f"{names[i]:<12} {np.ptp(s):4.1f} pts") # flow 11.0 ... season 0.0
```

**One trap.** `V.std(axis=0) > 1e-9` is the honest test for "does this factor vary within the
day" — do not infer it from the factor's name or its `sub_scope` entry. `sub_scope.hour` lists
`water_temp` and `season` as hour-scope factors, and they are, in the sense that they are
resolved per hour and carried on every `HourScore`. They simply resolve to the *same value* all
24 times. Reading `sub_scope` as "these vary hourly" is what makes the flatness surprising in the
first place.
