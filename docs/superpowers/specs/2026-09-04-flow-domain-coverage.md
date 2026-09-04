# Flow-Model Coverage — Why 75% of Features Are Unscored

**Date:** 2026-09-04
**Status:** Findings — documents an existing, deliberate constraint. No change proposed.
**Owner:** Ellis Millwood
**Explains:** `docs/superpowers/specs/2026-09-03-frontend-design.md` §4.4 ("the 2162-vs-529
problem"), and the map key's "1,633 of 2,162 features sit outside the flow model" line.
**Bears on:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` §7 (scoring),
`docs/superpowers/plans/2026-08-13-03-anuga-flow-library.md` (the domain polygon).

## 1. Why this note exists

The shipped UI states, honestly and prominently, that it scores 529 of the 2,162 features it
detected. That number has been carried in the code and the plans since Plan 3, but it has never
been written down in one place *with its cause*, and the cause is not what the phrasing
("outside the flow model") suggests to a reader encountering it fresh. This note records the
mechanism, the measurement, and what changing it would actually cost, so the question does not
have to be re-derived.

**Nothing here is a defect.** The scoring code behaves correctly and reports its own limits. The
75% is a scope decision, made deliberately in Plan 3 and inherited unchanged.

## 2. The mechanism

`engine/activation.py:167` `sample_features` summarises every field over the library cells within
a **150 m disc** of each feature's anchor:

```python
sel = (spec.xs - fx) ** 2 + (spec.ys - fy) ** 2 <= r2   # radius_m = 150.0
n = int(sel.sum())
```

When `n == 0`, every metric on that feature is NaN by construction. `engine/score.py:757`
short-circuits before `score_factors` ever runs, rather than computing a number on NaN:

```python
if metrics.n_cells == 0:
    return FeatureActivation(
        key=..., activation=0, subs=[], confidence=0.0, constrained_share=0.0,
        reason=f"{metrics.key} is outside the model domain — no library cells "
               "fall within the feature's sampling disc",
    )
```

The payload therefore carries no entry for that feature, `join.ts` writes no `a_*` properties,
and the map's `["coalesce", ["get", key], -1]` branch paints it muted. "Unscored" means exactly
**"no modelled water within 150 m"** — not "dead water", and not "scored zero".

## 3. The root cause: two extents that disagree

Two config-driven extents govern different stages, and they were authored independently.

| | source | extent |
|---|---|---|
| Feature detection | `bbox` in `fisheries/winyah-bay.yaml` | `[-79.45, 33.15, -79.05, 33.60]` ≈ 37 × 50 km |
| Flow model | `model_domain.polygon_utm_km` | 13 authored vertices, **235 km²** |

The bathymetry raster holds **751.6 km² of connected tidal water** (`z < wet_level_m` = 0.9 m,
1,857.5 km² of non-nodata cells). The authored polygon meshes **31%** of it. The unscored
features are the ones detected in the other 69%.

The cross-tab is exact — the polygon *is* the explanation, not a contributing factor:

| | scored | unscored |
|---|---|---|
| inside polygon | 523 | 6 |
| outside polygon | 6 | **1,627** |

The twelve off-diagonal features are boundary effects at the 150 m disc, and are the only slack
in the account.

### 3.1 The sampling radius is not implicated

The natural first hypothesis — that the 150 m disc is too tight — is wrong by three orders of
magnitude. Distance from each unscored feature to the *nearest* library cell:

| percentile | p0 | p10 | p25 | **p50** | p75 | p90 | p100 |
|---|---|---|---|---|---|---|---|
| metres | 152 | 1,645 | 3,460 | **8,591** | 13,255 | 16,993 | 25,941 |

Widening the disc recovers almost nothing, because the features are not near the domain edge —
they are kilometres beyond it:

| radius | 150 m | 300 m | 500 m | 1 km | 2 km | 5 km |
|---|---|---|---|---|---|---|
| scored | 529 | 545 | 557 | 612 | 741 | 1,068 |
| share | 24.5% | 25.2% | 25.8% | 28.3% | 34.3% | 49.4% |

A 5 km sampling disc — physically meaningless for a feature-level flow metric — still reaches
only half the inventory.

### 3.2 Where the unscored features actually are

Of the 1,627 never-modelled features, **964 sit north of the polygon's top edge**: up the
Waccamaw, Pee Dee and Black, in **131.8 km² of tidal water that is hydraulically connected to
the bay** (same `ndimage.label` component as the modelled body). The remainder are offshore,
outer marsh, and side creeks.

| | library grid | detected features |
|---|---|---|
| lat | 33.1655 – 33.4340 | 33.1516 – **33.5990** |
| lon | −79.3306 – −79.1336 | −79.4545 – −79.0484 |

By feature type, the loss is heaviest exactly where the river reaches are:

| type | n | unscored | share | median distance |
|---|---|---|---|---|
| dropoff | 847 | 752 | 88.8% | 7,903 m |
| bar | 256 | 218 | 85.2% | 7,709 m |
| hole | 389 | 257 | 66.1% | 3,928 m |
| wall | 37 | 23 | 62.2% | 3,222 m |
| creek_mouth | 134 | 82 | 61.2% | 2,118 m |
| flat | 497 | 301 | 60.6% | 2,178 m |
| **jetty** | **2** | **0** | **0.0%** | **9 m** |

The scored quarter is the *intended* quarter: both jetties are covered, which
`2026-08-13-03-anuga-flow-library.md` singles out as must-catch structure, and the polygon's east
edge was pushed to 673.5 km specifically to get them.

## 4. This was known, and is recorded in three places

- **Plan 3 spike** (2026-08-13) authored the polygon as enclosing "238 km² of water and all jetty
  structure", with the open-boundary placement called out as "a modelling decision, exactly like
  the existing `jetties:` seeds" — not something to be computed.
- **Plan 3 carryover notes** measured the consequence directly: "2,162 total features in the
  inventory, but only 531 have any in-domain cell at all." (531 then; 529 now, after the
  sampling-anchor fix.)
- **Frontend design spec §4.4** is titled "The 2162-vs-529 problem falls out for free" and builds
  the muted styling branch around it, explicitly to avoid "a parallel code path and a lookup that
  silently returns nothing."

The salinity-anchoring spec independently measured the same upriver mass from a different angle:
1,218 features (56.3%) sit above the bay salinity stations.

## 5. What closing the gap would cost

Extending the domain north is **not** primarily a compute-budget question. Three costs, in
increasing order of difficulty:

**5.1 Mesh area** — 235 km² → ~367 km² (+56%) to take in the connected upriver water.

**5.2 The CFL trap, which dominates.** Plan 3 §5 measured this on real Winyah bathymetry:

| grading | triangles | 1 run | 9 regimes |
|---|---|---|---|
| uniform 60 m | 256,803 | 1.12 h | 10.1 h |
| 60 m + jetty 15 m | 315,564 (+23%) | 5.46 h | 49.2 h |
| 60 m + jetty 12 m | 350,259 (+36%) | 8.49 h | 76.4 h |

**+36% triangles → +658% runtime.** ANUGA uses a global CFL timestep, so the smallest triangle
anywhere sets the cost everywhere. Narrow river channels need finer triangles than the open bay
to resolve at all — so the upriver extension lands on the wrong side of exactly this
nonlinearity. Area is the misleading number here; **minimum edge length is the budget**.

**5.3 The river boundary conditions would have to move.** All three inflows currently sit just
inside the polygon's northern edge:

| river | `inflow_lonlat` |
|---|---|
| Pee Dee | −79.22587, 33.40619 |
| Waccamaw | −79.20462, 33.42022 |
| Black | −79.26360, 33.38999 |

Extending the domain upstream means relocating all three into narrower, shallower channels. The
config carries an explicit warning against precisely this failure mode — inflow points "must sit
well inside the domain, in the channel, in water deeper than −2 m — NEVER on or near the
`model_domain.polygon_utm_km` boundary" — recorded as the root cause of the 2026-08-14 build
failure in which **6 of 9 regimes died with "Too small timestep."** The per-river discharge split
(78/13/8) would also need re-deriving at the new attachment points.

Any extension therefore requires a full 9-regime library rebuild plus revalidation, not a config
tweak.

## 6. Cheaper options, if coverage is the goal

Recorded for completeness; none is evaluated or recommended here.

1. **Accept and clarify.** The status quo. The UI already discloses the ratio honestly. The
   product claim narrows to "the bay proper, scored well" rather than "the fishery, scored".
2. **Clip detection to the domain.** Stop detecting features that can never be scored. Makes the
   ratio disappear from the UI without adding any coverage — cosmetic, and arguably *less*
   honest, since the upriver structure is real and detected.
3. **A coarse upriver regime.** Mesh the rivers at deliberately coarse resolution as a separate,
   lower-fidelity domain. Sidesteps §5.2 only if the channels can be resolved coarsely, which is
   unmeasured.
4. **Score upriver features without flow.** The scoring engine already renormalises around
   missing factors and reports `confidence` / `constrained_share`. An upriver feature could in
   principle be scored on its other nine factors with `flow` excluded and confidence reported
   accordingly. **This is the cheapest option by a wide margin** — no mesh, no rebuild — but it
   changes what a score means across the map, and `flow` carries weight 1.00, the highest of the
   ten. Not obviously right; worth its own design pass if pursued.

## 7. Reproducing the measurements

Every number above comes from this, run against the shipped `winyah-bay` data:

```python
import json, numpy as np
from matplotlib.path import Path as MPath
from scipy.spatial import cKDTree
from tidescout.config import load_fishery
from tidescout.pipeline.bathy import read_bathy
from tidescout.pipeline.flowlib import grid_spec
from tidescout.engine.activation import _sampling_anchors
from tidescout.paths import fishery_data_dir

slug = "winyah-bay"
fishery = load_fishery(slug)
spec = grid_spec(slug, fishery)
feats = json.loads((fishery_data_dir(slug) / "features.geojson").read_text())["features"]

xs, ys = (np.asarray(a) for a in _sampling_anchors(feats, spec, False))
dist, _ = cKDTree(np.column_stack([spec.xs, spec.ys])).query(np.column_stack([xs, ys]), k=1)
poly_m = np.array(fishery.model_domain.polygon_utm_km, float) * 1000.0
inside = MPath(poly_m).contains_points(np.column_stack([xs, ys]))

print((dist <= 150).sum(), "scored /", len(feats))          # 529 / 2162
print(np.percentile(dist[dist > 150], 50), "m median gap")  # 8590.6
print(inside.sum(), (~inside).sum())                        # 529, 1633
```

**One trap, worth stating because it reverses the conclusion.** `mesh.domain_mask(z, transform,
fishery)` applies `model_domain.polygon_utm_km` *internally* when no `polygon=` argument is
passed. Measuring "total water in the raster" with it returns the already-clipped 235 km² and
makes the upriver water look nonexistent. For the raw figure, threshold directly:

```python
z, transform, _ = read_bathy(slug)
raw_wet = ~np.isnan(z) & (z < fishery.model_domain.wet_level_m)   # 751.6 km², not 235
```
