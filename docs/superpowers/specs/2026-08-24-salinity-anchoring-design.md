# Salinity Anchoring — Design Spec

**Date:** 2026-08-24
**Status:** Approved (brainstormed section-by-section)
**Owner:** Ellis Millwood
**Constrains:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` §7 (salinity effect on
scoring), §10 (resilience). Follows Plan 4 Phase 2 (PR #4).

## 1. The problem this exists to solve

Phase 2 fitted the salt-intrusion model against the full NERRS record — 10,864 observations and
10,864 tidal swings over 2016-2026 — and the model was **falsified, not merely unconstrained**:
rmse 4.060 ppt against an observation resolution of 0.003 ppt, a factor of **1,353**, with a
healthy condition number (12.2), every 1-sigma below its value and nothing at a bound.

The binding cause is coverage. Measured against the 2,162-feature inventory:

| reach | features | share | status |
|---|---|---|---|
| seaward of North Jetty (0–2.58 km) | 53 | 2.5% | below observed range |
| jetties → Mud Bay Cut (2.58–13.05 km) | 759 | 35.1% | below observed range |
| Mud Bay Cut → first station (13.05–16.68 km) | 123 | 5.7% | below observed range |
| **between the two bay stations (16.68–19.03 km)** | **9** | **0.4%** | **interpolated** |
| above the bay stations (>19.03 km) | 1,218 | 56.3% | above observed range |

**Only 0.4% of features sit where observations bracket them.** Everything else is extrapolation.

The ambiguity survives into the score. Using the Phase 3 plan's trout curve (salinity weight 0.9,
the sharpest of the three species), the median salinity sub-score across four statistically
healthy fits spans:

- North Jetty (2.58 km): **0.701 – 0.960** — 26 points of a 0–100 factor
- Georgetown Lighthouse (5.52 km): **0.704 – 0.960** — 26 points
- Mud Bay Cut (13.05 km): 0.863 – 0.907 — 4 points
- At the bay stations (19.03 km): **0.083 – 0.515** — 43 points

Two notes that shape the design. First, near the mouth every fit lands in 0.70–1.00, so the
*direction* never changes there: salinity is not the limiting factor for trout at the jetties
whichever fit you pick. The severe ambiguity is upstream, where the curve is steepest and 56% of
features sit. Second, rank correlation between fits is exactly 1.0000 at every spot — but that is
**structural**, not evidence of agreement: the model is monotone in discharge by construction, so
any monotone-in-Q model gives 1.0. It must not be read as a constraint.

### What is explicitly NOT the cause

More NERRS history will not fix this, and neither will the CDMO IP registration — that data is
already in hand and is what produced the falsification. The CDMO import moved the observations
**further** from the fishing spots, not closer.

## 2. Decisions taken

| Decision | Choice |
|---|---|
| Role of salinity where unconstrained | Best estimate everywhere **plus explicit confidence**, not dropping the factor and not coarse zones |
| Source of new anchors | **Published survey data**, not new field readings and not physics-only |
| Branch membership | **Computed**, not hand-marked |
| Model form | **Ingest and refit first, decide at a measured gate** |
| ANUGA-derived spatial shape | Out of scope — noted as the successor once the model has data worth being faithful to |

## 3. The data, verified before this spec was written

The Water Quality Portal (`waterqualitydata.us`, EPA/USGS/state aggregator) serves **208 salinity
stations** in the Winyah Bay bbox, **132 of them in-domain** (snap gap ≤ 500 m), **55 of those in
the 2.58–13.05 km reach that had nothing**. Main-channel stations, live-verified 2026-08-24:

| station | along-estuary km | n | period | salinity |
|---|---|---|---|---|
| `21SC60WQ_WQX-WB-06` | 5.56 | 40 | 2014-06-12 – 2018-09-05 | 6.3–35.4 ppt |
| `21SCSHL_WQX-05-24` (Coast Guard DK Range) | 5.77 | 14 | 2009-12-29 – 2012-06-19 | 6.0–35.0 |
| `21SC60WQ_WQX-WB-05` | 10.28 | 38 | 2014-06-12 – 2018-09-05 | 3.7–29.3 |
| `21SCSHL_WQX-05-21` (Buoy 17 Range E) | 10.31 | 14 | 2009-12-29 – 2012-06-19 | 2.0–28.0 |
| `21SCSHL_WQX-05-25` (W. Channel Island) | 12.17 | 14 | 2009-12-29 – 2012-06-19 | 1.0–30.0 |

Contributing organisations: SC Dept of Environmental Services (144 stations), SCDHEC Shellfish
(39), SCDHEC (9), EPA EMAP (9), Coastal Carolina University (4), EPA NARS (3).

These are **discrete grab samples, not a feed**. That is not a weakness here: each carries a
timestamp, and this codebase already holds the tide model and 10.6 years of composite discharge,
so every sample resolves to a known **distance, discharge and tidal phase** — a fully-specified
observation. The endpoint is public, unauthenticated, and needs no registration, unlike CDMO.

## 4. Architecture

Three units, each independently testable.

### 4a. `sources/wqp.py` — ingestion

A sibling to `usgs.py`/`cdmo.py`. Fetches WQP `Result` CSV for the fishery bbox and writes into
the **existing** `NdbcStore`; `cdmo.py` already reuses that store unmodified, so the
append-and-dedupe contract proven at 2.5 M rows is shared rather than reimplemented. A WQP
salinity sample maps onto `ts` + `salinity_psu` + `depth_m`.

Three rules, each with an existing precedent in this repo:

- **`Salinity` only — never specific conductance.** `usgs.py` already holds that line ("a
  different quantity and is not interchangeable"). WQP serves both under one query and mixing
  them would be silent.
- **Units on an explicit allowlist.** The real data carries both `ppt` and `0/00`. Anything
  unrecognised is rejected and *reported*, never coerced.
- **QA flags on an allowlist; blocked rows counted and surfaced.** The `cdmo.py` pattern, which
  caught four wrong inferences when the real export arrived.

Depth is recorded rather than discarded, because stratification is one of the two remaining
falsification causes and a bottom sample is not a surface sample. Per-import provenance goes into
the existing provenance table so Task 10's citation machinery covers SCDES/SCDHEC/EPA attribution
on the same footing as NERRS.

### 4b. The computed on-axis screen

The novel unit, and the reason approach A alone was rejected. Of the 55 in-reach stations, several
are *North Santee River*, *AIWW at Minum Creek*, and Town/Jones Creek — the North Inlet failure at
20× the scale. Hand-marking 132 stations would reproduce the defect that caused the 43-point
ambiguity, only less visibly.

- **Main stem:** the union of the steepest-descent paths on the distance field from the three
  river inflow points already in `fisheries/winyah-bay.yaml` down to the mouth.
- **Tributary offset:** for a station, the along-path distance its own descent travels before
  joining the main stem. A main-channel station scores ≈ 0; a station up a side creek scores that
  creek's length.
- **Threshold:** measured, not authored. Swept against the distribution and validated against
  known cases — the three bay NERRS stations must come out on-axis and North Inlet's three must
  come out off-axis, by the computed criterion alone.

This replaces the hand-set `WaterSensor.off_axis` flag with a derived value, retaining the YAML
field as an explicit override for cases the geometry gets wrong. It generalises to the
stamp-out fisheries (Charleston, Awendaw/Cape Romain, Murrells Inlet), all of which branch.

### 4c. Confidence on `SalinityField`

`SalinityField` gains an ordinal — **measured / interpolated / extrapolated** — derived from
along-axis distance to the nearest on-axis observation, with the raw distance exposed alongside
it. Ordinal rather than a 0–1 score because it is a *coverage* statement and should read as one;
a continuous score invites being multiplied into something.

It sits beside the existing `extrapolated` and `fitted` flags, which answer different questions
and are already documented as such. Phase 3's `score_factors(..., salinity_ppt=None)` already
accepts an absent value, so this rides into the sub-score's reason string.

## 5. The gate

Model form is deliberately not decided here. The first task ingests WQP and re-runs the existing
fit on the enlarged dataset, then reports **rmse against observation resolution, per-station
bias, distinct on-axis distances and their span**. That report goes to Ellis with numbers, and
the replacement form — if one is needed — is chosen then.

This matches how the freshet gate and the Georgetown gate already work on this project: measured
first, decided by Ellis, never spent unilaterally.

**Honest expectation, recorded so the gate is not read as optimism:** better anchoring does not
touch the other two falsification causes. Stratification is a measured +3.30 ppt median between
surface and bottom at one distance, which a depth-averaged single layer cannot represent, and
dropping to surface-only alone moved rmse 4.07 → 3.37. The branch offset is handled by §4b for
*fitting*, but the model still has one axis. The realistic outcome is `fitted` moving from
"false, and structurally hopeless" to "false, for one identifiable reason" — with `fitted=True`
genuinely unknown until the data is in.

## 6. Testing

- **Branch geometry:** hand-built fixtures in the style of the two-mouth tests added in PR #4 —
  a main stem with a side creek, asserting the offset separates them.
- **Real-data parsing:** tests against a verbatim excerpt of a real WQP CSV, the `cdmo.py`
  precedent. Task 9 built a parser from documentation and four inferences were wrong, one
  catastrophically; no parser here is trusted until it has met a real file.
- **Screen regression:** the six NERRS stations pinned to the correct side of the computed
  criterion, so a future threshold change cannot silently re-admit North Inlet.
- **Unit rejection:** an unrecognised unit must raise or be reported, never coerced.
- **Store contract:** re-running an import yields `new = 0`, matching the CDMO proof.

## 7. Out of scope

- Deriving the spatial shape from the ANUGA library's velocity fields (approach C). Real, and the
  natural successor, but a substantial new subsystem that does not fix anchoring.
- A two-layer / stratified salinity model. Named as a falsification cause; not designed here.
- Any change to `ocean_boundary_utm_km` or the flow library.
- Field readings by the user.
