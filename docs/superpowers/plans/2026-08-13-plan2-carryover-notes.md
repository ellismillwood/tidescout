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

## SCDNR intertidal oyster reef layer — availability probed 2026-08-14 (Plan 4 input)

Spec §4 lists oyster beds as a feature class "if downloadable, adds a feature class; else
manual pins later", and §13 treats it as optional. It was deferred by name in Plan 1
("...SECOFS, CUDEM, and oyster layers") and recorded as `oyster ✗` in Plan 2's spec-coverage
line. **Neither deferral ever checked whether the layer was actually obtainable.** It has now
been checked, so Plan 4 does not have to re-derive this:

- **The dataset exists and is a good fit.** `SCDNRoyster2015Live` — 168,373 intertidal reef
  polygons, NAD83 **UTM 17N** (the same CRS as our analysis grid, so no reprojection risk),
  extent -80.94..-78.53 lon / 32.02..33.91 lat, which covers Winyah Bay. Digitised from
  0.25 m 4-band orthophotos (2003-2006), updated from helicopter photography 2011-2015.
  Metadata: https://www.dnr.sc.gov/GIS/metadata/SCDNR_Oyster2015Live.html
- **It is NOT publicly downloadable as of this check.** The service that public search results
  point at — `MRD/Sc_Intertidal_Oyster_Reefs20190402/MapServer` on arcweb.dnr.sc.gov — returns
  `{"error":{"code":499,"message":"Token Required"}}`. SCDNR's service directory *is* browsable
  without auth, and the `MRD` folder now lists only `sfpermit` and `SSG19_20test`: the oyster
  service is no longer published there. So it was unpublished/secured, not merely mis-addressed.
- **The metadata states Access Constraints: "None"** but publishes no download URL or ordering
  process. That combination (no policy restriction, no public endpoint) means the realistic route
  is a direct data request to SCDNR's Marine Resources Division, not scraping.
- **`MRD/sfpermit` (layer `sfpermit22`) is NOT a substitute** — it is shellfish *permit/harvest
  ground* boundaries, i.e. management polygons, not reef geometry. Useless for ambush-feature
  detection.
- **Auth mechanics if a login is ever granted:** the server reports
  `isTokenBasedSecurity: true` with `tokenServicesUrl =
  https://arcweb.dnr.sc.gov/portal/sharing/rest/generateToken`, and `owningSystemUrl =
  https://arcweb.dnr.sc.gov/portal`. So it is federated to ArcGIS Enterprise Portal: a token is
  minted by POSTing existing portal credentials (username/password/client/referer/expiration) to
  that endpoint — the token is the *consequence* of an SCDNR-issued account, never a way to
  obtain one. `/server/tokens/generateToken` returns HTTP 405 to GET (POST-only).
- **Scope ruling:** this belongs to Plan 4 (scoring/feature classes), not Plan 3 (flow library).
  The spec already permits shipping without it. If the request to SCDNR succeeds, the layer drops
  in as a new `oyster` feature type alongside dropoff/flat/hole/bar/creek_mouth/wall/jetty; if it
  does not, spec §13's fallback is manual pins in the fishery config.
