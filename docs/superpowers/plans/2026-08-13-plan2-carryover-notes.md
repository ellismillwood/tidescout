# Plan 2 Carryover Notes (for Plan 3+ authoring)

Distilled from Plan 2's ledger and final whole-branch review (2026-08-13) before workspace cleanup. Branch `plan-02-bathymetry` (ea3211a..f30c0b7): 11/11 tasks, final review "with fixes" applied + re-reviewed clean, 101/101 tests.

## What Plan 3 (ANUGA flow library) inherits

- **Analysis grid is sound and safe to mesh from**: EPSG:26917 @ 10 m, 3806×5053, corner-based Affine with consistent +0.5 pixel-center convention; nodata (3.4%) is **entirely projection wedges at the corners — no interior gaps** (verified via grid-convergence arithmetic). `read_bathy` now takes its transform from the GeoTIFF itself and raises on stale meta.
- **Real raster facts**: z −16.3..+19.3 m NAVD88; jetties physically present as ridges (South Jetty found in bathymetry at the config seed endpoint); 767 MB tiles cached in `data/winyah-bay/tiles/` (fileServer URLs in committed manifest `fisheries/winyah-bay.tiles.yaml` — re-downloadable, THREDDS stalls happen: retry + curl -C - fallback exists).
- **Feature inventory**: 2,217 features in `data/winyah-bay/features.geojson` (4326, interior rings preserved): dropoff 884, flat 509, hole 416, bar 272, creek_mouth 134, jetty 2, wall 0.

## Plan 3 kickoff tasks (ordered, from the final review)

1. **Wall re-diagnosis — corrected ruling**: wall=0 is an ESTIMATOR artifact, not resolution. 3,067 pixels ≥20° exist (max 32°, p99.9 13.9°); `mean(slope)` over a polygon whose boundary is pinned at the 8° dropoff threshold can never reach 20°. Fix: type walls on `nanmax`/p90 slope within the polygon, or a second connected-component pass at `wall_slope_deg`. Do NOT chase a finer grid first.
2. **Per-type maximum area / elongation gates**: largest bar is 47.4 km², largest flat 27.4 km² — basin-scale blobs that will wreck "nearest feature to this flow cell" joins (`p.distance`=0 for any contained point). Add upper bounds/segmentation, then have **Ellis populate `fisheries/winyah-bay.known-spots.yaml`** and run `tidescout spots winyah-bay` — the validation aid has never seen real ground truth, and it is the gate ANUGA work is measured against.
3. **Pipeline-level tests** for `build_derivatives`/`build_artifacts` (currently engine-only coverage) before Plan 3 consumes `zones.tif`.
4. Plan 1 parked items also land here: discharge freshness signalling + bucket recalibration (see 2026-08-12-plan1-carryover-notes.md).

## Traps named by the final review

- **(a) zones.tif aliases detector thresholds** (`shallow_max_m`/`deep_min_m` shared between `zones()` and `detect_bars`): retuning bar detection silently re-buckets zones. If zones feed Manning n / wet-dry init, freeze or split the config before tuning.
- **(b) `WET_LEVEL_M = 0.0` is a module constant in detect.py**: every static detector's "wet" is pinned to NAVD88 zero; ANUGA introduces a time-varying free surface — make it config-driven before the definitions fork.
- **(c) Feature ids are not rebuild-stable** (`bar-78` renumbers on `--rebuild`): decide a stable key (hash of type+centroid) before anything persists a feature reference (scores, user notes).
- **(d) `artifact_reach_cells=2` in the creek-mouth filter is hardcoded** — recheck if `cell_m` ever changes. Real creek mouths within 2 cells of survey nodata are deliberately suppressed (perimeter only, given no interior nodata).
- **(e) `detect_bars`/dropoffs/holes do O(N×components) full-raster work** (56 s of 72 s runtime) — fine at one-shot cadence, optimize only if rebuild cadence rises.

## Environment/process facts

- ANUGA install still unattempted (Plan 3's first risk; container fallback per spec §5). THREDDS/NCEI is the bathy source of record — no CUDEM S3 bucket exists.
- Deferred minors (final review triaged all OK-TO-DEFER): contour ring seam-splitting (~40 m loss cases, stitch recipe in ledger), hillshade nodata-vs-black collision, quicklook not georeferenced, `noaa.__all__` under-declares, `JettySeed` lacks min-2-vertex validator, 3 early-bound path value-imports (patch-target gotchas documented in test_spots.py), README section header stale, empty-feats `min()` edge, Rich-table test parse brittleness, `_write` cross-module private import (rename to `write_raster` when touched).
- Subagent report prose can drift on specifics (bar-78 vs bar-123 attribution) — verify numbers against artifacts, not summaries. Three more API connection drops this plan; work always survived in the tree/commits — check `git log` before re-dispatching.

## SCDNR intertidal oyster reef layer — OBTAINED 2026-08-14 (Plan 4 input)

Spec §4 lists oyster beds as a feature class "if downloadable"; §13 treats it as optional. It was
deferred by name in Plan 1 and recorded `oyster ✗` in Plan 2 without anyone checking obtainability.
It has now been checked, downloaded, and characterised.

**It IS publicly downloadable — no token.** An initial probe wrongly concluded otherwise: the
service that web searches surface, `MRD/Sc_Intertidal_Oyster_Reefs20190402/MapServer`, is secured
(HTTP 499 Token Required) and no longer listed in the public `MRD` folder. That is a stale
endpoint. The LIVE one is found by walking the public ArcGIS Online web app to its web map to its
operational layer:

```
scdnr.maps.arcgis.com item 5bc898b455be43bea4a908491d2b3414   (access: public)
  -> webmap db780044ce3348e785f653a72cc6c6b7
    -> https://arcweb.dnr.sc.gov/server/rest/services/Hosted/SCDNROyster2015Live/FeatureServer/0
```

Public, unauthenticated, `maxRecordCount` 2000 with pagination, served in Web Mercator (3857)
despite metadata claiming UTM 17N — request `outSR=4326` to match the pipeline's GeoJSON
convention. Fields: `objectid, id, calcgeo_ac, photoedit, photo_year, shape_leng`.
Working downloader: `.superpowers/sdd/<plan>/fetch_oysters.py`; Plan 4 should promote it to
`sources/scdnr.py` with the usual SQLite cache. Acknowledge SCDNR as source per their use
constraints.

**Downloaded for Winyah** to `data/winyah-bay/oyster_reefs.geojson` (gitignored, rebuildable):
8,451 polygons in the bbox, 6,527 (77%) inside the model domain, `photo_year` uniformly 2008.

**What it is good for — and what it is NOT.** These are fringing reef patches, not bars:

| metric | value |
|---|---|
| total area, whole bbox | 0.607 km² |
| median reef | 24 m² (~5 x 5 m) |
| p90 | 142 m² |
| max | 6,458 m² |
| >= 500 m² (resolvable at the 20 m mesh / 20 m library grid) | 181 |
| >= 2000 m² | 18 |

- **NOT mesh geometry.** A 24 m² reef is far below the 20 m triangle edge and the 20 m
  `library_cell_m` raster. Oysters cannot influence the hydrodynamics at our resolution, and
  meshing them would be meaningless CFL cost. Do not add them to the ANUGA domain.
- **NOT an ambush-feature class as-is.** 8,451 polygons of median 24 m² would swamp the 2,162-strong
  feature inventory with noise, and would re-create exactly the nearest-feature problem Task 2 fixed
  from the other direction (too many tiny features rather than one huge one). If used as features at
  all, gate on area (>= ~500 m², i.e. 181 of them) or cluster adjacent patches into reef complexes.
- **YES as a scoring/habitat layer (Plan 4).** Proximity-to-oyster-habitat is a legitimate
  bite-score factor, especially for redfish. A rasterised reef-density field on the analysis grid is
  the natural form — it degrades gracefully at any resolution and sidesteps the polygon-count problem.

**Calibration note:** none of Ellis's three known spots sit near a reef — nearest reef is 1.4 km
(Mud Bay Cut), 3.9 km (Georgetown Lighthouse), 6.5 km (North Jetty). So oyster habitat is not what
drives his current spots, and any oyster factor should carry a small default weight until validated
against spots he chooses specifically for oysters. Imagery is 2008, ~18 years stale; intertidal
reefs migrate, so treat this as a prior, not ground truth.
