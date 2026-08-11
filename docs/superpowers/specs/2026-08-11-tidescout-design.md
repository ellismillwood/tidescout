# TideScout — Design Spec

**Date:** 2026-08-11
**Status:** Approved (brainstormed section-by-section, all sections user-approved)
**Owner:** Ellis Millwood

## 1. Purpose

A personal decision-support app for South Carolina inshore fishing that out-predicts consumer apps (Fishbrain, Salt Strong) by modeling what they ignore: how tidal flow interacts with local bathymetry to create eddies, seams, and ambush points, and how salinity and water temperature move fish around an estuary.

The user picks a **fishery, date, and weather forecast model**; the app shows a **bathymetric map with likely ambush points highlighted** for the selected hour, plus an **hour-by-hour bite score (0–100) per species** the user can scrub through, with every contributing condition (tide flow/direction, wind, pressure, temps, salinity, light, solunar) visible and explained.

### Non-goals

- No accounts, hosting, or multi-user features — runs locally for one user.
- No machine learning — scoring is a transparent, tunable heuristic.
- No catch logging or trip journaling in v1.
- No shore/kayak accessibility filtering — user fishes from a boat/skiff.
- No mobile app — desktop browser UI served from localhost.

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Audience | Ellis only, local machine |
| Species | Redfish (red drum), speckled trout, southern flounder — species-specific scoring lenses |
| Fisheries (eventual) | Charleston, Awendaw/Cape Romain, Winyah Bay, Murrells Inlet |
| First fishery (v1 validation) | **Winyah Bay** |
| Flow modeling | **ANUGA 2D shallow-water simulation from day one** (no interim heuristic solver) |
| Scoring | Transparent weighted heuristic; weighted geometric mean of factor sub-scores |
| Backend | Python 3.12, FastAPI; venv at `~/.venvs/tidescout`; editable install |
| Frontend | React + Vite + TypeScript + MapLibre GL |
| Delivery | Local web app: one launcher command starts the API and opens the browser |
| Repo | `~/Documents/tidescout` (git) |

## 3. Architecture

```
tidescout/
  backend/
    pyproject.toml
    tidescout/
      engine/        # scoring + feature activation (pure functions, no I/O)
      sources/       # external data fetchers + cache (Open-Meteo, NOAA, USGS, astral)
      pipeline/      # offline per-fishery prep: bathymetry, features, ANUGA states
      api/           # FastAPI app: day-payload endpoint, serves built frontend
    tests/
  frontend/          # React + Vite + TS + MapLibre GL
  fisheries/         # per-fishery config + user ground truth (committed)
    winyah-bay.yaml
    winyah-bay.known-spots.yaml
    species_weights.yaml
  data/              # processed artifacts (gitignored, fully rebuildable)
  docs/superpowers/  # specs and plans
```

**Separation rule:** `engine/` is pure — takes plain data structures in, returns scores and explanations out; all I/O lives in `sources/` and `pipeline/`. This is what makes the algorithm testable and tunable.

**Data flow:**
1. **Offline pipeline** (run once per fishery, rerun only when inputs change): bathymetry download → map artifacts (hillshade, contours, depth tint) → static feature inventory (GeoJSON) → ANUGA flow-state library.
2. **Runtime:** request `(fishery, date, weather_model, species)` → fetch/cached externals (forecast, tide + current predictions, discharge, water temp, salinity obs) → map each hour to nearest flow state → score 24 hours × all features × 3 species → **one JSON day-payload** → React renders; scrubbing never refetches.

## 4. Data sources (all free, no API keys)

| Input | Source | Notes |
|---|---|---|
| Bathymetry | NOAA NCEI CUDEM, 1/9 arc-second (~3 m) GeoTIFF tiles | One-time download per fishery bbox; fallback to 1/3″ Coastal Relief Model where CUDEM has gaps up-river |
| Tide levels & harmonics | NOAA CO-OPS API | Winyah candidates: South Island Ferry, Georgetown-area stations — **resolve exact station IDs from the CO-OPS directory during implementation** |
| Tidal current predictions | NOAA CO-OPS current predictions stations near Winyah entrance | Used for boundary forcing + calibration, not spatial answers |
| Weather forecast | Open-Meteo forecast API, `models=` parameter | **This is the weather-model picker**: GFS, ECMWF, ICON, HRRR, NBM, best_match; hourly wind speed/dir/gust, MSL pressure, cloud cover, precip, air temp |
| Historical weather | Open-Meteo archive API | Enables hindcasting past dates |
| River discharge | USGS NWIS instantaneous values: Pee Dee, Waccamaw, Black | Composite, lagged 1–2 days, drives salinity + flow regime selection |
| Water temp & salinity obs | NOAA CO-OPS sensors, USGS coastal gauges, North Inlet–Winyah Bay NERR stations (e.g., Oyster Landing) | Nearest healthy sensor wins; 30-day rolling history kept for trend; seasonal climatology fallback |
| Sun/moon/solunar | Computed locally with `astral` (+ moon transit math) | No network at all |
| Modeled currents/salinity (evaluate) | NOAA SECOFS | Southeast Coastal Operational Forecast System — evaluate resolution/coverage inside the bay during implementation before relying on it; likely useful only as boundary/calibration data |
| Oyster beds | SCDNR shellfish GIS layer | If downloadable, adds a feature class; else manual pins later |

**Caching:** every fetcher writes through a local SQLite cache keyed by (source, params, date). TTLs: forecasts ~1 h, observations ~15 min, predictions/astronomy immutable. Stale data is served with a freshness flag rather than failing.

## 5. Flow-state library (the novel core)

Tidal flow patterns are quasi-periodic: the flow field at "mid-ebb, spring range, high discharge" recurs. So we **simulate offline, look up at runtime**.

- **Model:** ANUGA (open-source Python 2D depth-averaged shallow-water solver, unstructured triangular mesh, native wetting/drying).
- **Mesh:** built from CUDEM bathymetry; refined near channels, structure, and shorelines; coarser in open water.
- **Forcing:** ocean-boundary water level from CO-OPS harmonic predictions; river inflows from USGS discharge; spatially varying Manning friction (channel vs marsh vs flats).
- **Regime runs:** ~9 simulations = tidal range {neap, mean, spring} × discharge {low, med, high}, each a full tidal cycle plus spin-up, snapshotted every 30–60 simulated minutes.
- **Library:** snapshots rasterized to the analysis grid (~10–20 m cells) as float arrays: **speed, direction, shear (lateral velocity gradient), wet/dry**. Indexed by (range bucket, discharge bucket, tide phase). Runtime picks the nearest state, interpolating by phase.
- **Derived fish-relevant structure:** slow pockets adjacent to fast conveyors, seams (high shear), lee/eddy zones behind points and bars given flow direction, convergence at draining creek mouths, flats flood/drain schedule from wet/dry.
- **Automated checks:** mass conservation within tolerance; ebb/flood direction reversal at known channel cross-sections.
- **Human validation gate (go/no-go, precedes UI polish):** the model must independently light up the rips, eddies, and slack pockets Ellis records in `winyah-bay.known-spots.yaml`.
- **Contingency:** if ANUGA installation fights macOS, run it in a Linux container (Docker/OrbStack); outputs identical.

## 6. Static ambush-feature inventory

Extracted offline from bathymetry + shoreline; every feature carries geometry, type, depth band, orientation, and adjacency to deep water.

Feature classes: **drop-offs/channel edges** (slope + depth-change thresholds), **holes** (local deep pockets), **creek mouths** (junctions in the skeletonized channel network), **points and bars** (shoreline convexity + submerged ridges beside deep water), **flats** (low-slope shallows with flood/drain schedule from ANUGA wet/dry), **dredged channel walls** (Winyah shipping channel), **jetty structure** (chart-seeded manual geometry — the Winyah jetty rips must be catchable), **oyster beds** (SCDNR layer if available).

Detection is validated on **synthetic DEMs first** (an idealized point bar and creek mouth the detectors must find), then against Winyah reality via known-spots.

## 7. Salinity as a fish-distribution layer

Winyah Bay is heavily river-dominated; salt-wedge position is a first-order control on where fish are.

- **Model:** empirical salt-intrusion model — salinity as a function of along-estuary distance, lagged composite USGS discharge, and tide phase, using standard intrusion-length scaling; calibrated against NERR/USGS sensor observations (and SECOFS salinity if it proves usable). Output: hourly 2D salinity field.
- **Effect on scoring:** per-species **salinity suitability curves** (trout ≈ 10–30 ppt, avoid near-fresh; redfish broadly tolerant; flounder tolerant, bait follows salt) multiply feature activation **spatially** — the same eddy scores near zero up-bay after a freshet and lights up 5 miles down-bay.
- **UI:** salinity overlay toggle + a distribution-shift indicator ("high discharge: fish pushed down-bay").

## 8. Bite-score engine

Two outputs from one factor pipeline:
- **Fishery-wide hourly bite score, 0–100, per species** — the scrubbable strip.
- **Per-feature activation, per hour, per species** — the map markers.

**Factors** (each → 0–1 sub-score through a species-specific response curve):
1. **Tidal flow rate** (from the hour's flow state; station fallback) — bell curve peaking mid-ebb/mid-flood, cratering at slack; stage-direction interactions (flounder bias to ebb).
2. **Tide stage** — rising/falling/high/low + which flats are actually flooded.
3. **Light** — dawn/dusk peaks; cloud cover widens windows.
4. **Solunar** — moon transit majors/minors; small default weight, zeroable.
5. **Pressure trend** — falling/pre-front boost; sharp post-front rise penalized.
6. **Wind** — speed curve + direction vs fishery orientation (bait push vs fishability).
7. **Water temp + trend** — optimal ranges and seasonal behavior states (falling-through-low-60s triggers the flounder run; cold snap flips trout to winter-hole logic).
8. **Salinity regime** — freshet penalty + spatial suitability (Section 7).
9. **Season/spawn calendar** — monthly modifiers (fall bull-red jetty run, trout spawn aggregation).

**Combination:** weighted **geometric mean** so a near-zero critical factor (dead slack, cold shock) tanks the hour instead of averaging away. Missing factors are **excluded with weights renormalized and flagged** — never silently defaulted.

**Tunability:** all curves and weights in `fisheries/species_weights.yaml`; edit → re-score instantly (dev hot-reload).

**Explainability:** the API returns every sub-score with a one-line reason; the UI renders factor bars — "why is 3 PM an 82" always has a visible answer.

**Hindcasting:** past dates fully supported (Open-Meteo archive + deterministic tide/current predictions) so known-good days can be checked against their predicted scores when tuning weights.

## 9. UI

```
┌────────────────────────────────────────────────────────────────┐
│ TideScout  [Winyah Bay ▾] [Wed Aug 12 ▾] [ECMWF ▾] [Redfish ▾] │
├───────────────────────────────────────────────┬────────────────┤
│              MAP  (MapLibre GL)               │  3:00 PM   82  │
│   bathy hillshade + depth tint + contours     │  tide, flow,   │
│   ambush markers (size/color = activation)    │  wind, pressure│
│   flow arrows for selected hour               │  temps, salin, │
│   toggles: [flow] [salinity] [contours]       │  sun/moon      │
│   click marker → "what + why active" popover  │  factor bars   │
├───────────────────────────────────────────────┴────────────────┤
│  ◀ │ 24 bite-score bars, tide curve underlay, drag playhead │ ▶ │
└────────────────────────────────────────────────────────────────┘
```

- Top bar: fishery (unprocessed ones grayed out), date picker (past = hindcast), weather model picker, species selector (switch re-colors instantly; all species pre-scored in the payload).
- Basemap: satellite/light toggle; choose a token-free tile source during implementation (verify terms).
- Right rail: selected hour's full conditions + factor bars.
- Bottom strip: draggable playhead, arrow-key scrubbing, prev/next day.
- Implementation note: build the frontend with the `frontend-design` skill (visual direction) and the `dataviz` skill (hour strip, factor bars, tide curve) — user explicitly wants these applied.

## 10. Resilience

- All external sources cached with stale-fallback; UI shows freshness badges and a confidence indicator when a factor is missing or a sensor is dark.
- Tides/currents computed from locally cached harmonic predictions — effectively never dark.
- Requested weather model unavailable → best_match fallback with banner.
- Missing flow state for a regime → nearest available state + warning.

## 11. Testing

- **Engine:** pytest golden tests (hand-built scenarios with expected sub-scores) + property tests (scores bounded 0–100, monotone response to single-factor changes, weight renormalization sums correctly).
- **Feature detection:** synthetic-DEM fixtures (idealized point bar, creek mouth) must be found before real bathymetry is trusted.
- **Flow states:** mass-conservation and ebb/flood-reversal checks; the human known-spots gate.
- **Fetchers:** recorded HTTP fixtures (respx) — tests never hit live APIs.
- **Frontend:** strict TypeScript, component smoke tests, Playwright smoke once stable.
- **Gate:** `make check` = ruff + pytest. No CI (personal tool).

## 12. Rollout

1. Winyah Bay end-to-end (phases: data foundations → bathymetry + features → ANUGA library + validation gate → salinity → engine → API → frontend → hindcast tuning).
2. Stamp out Charleston, Awendaw/Cape Romain, Murrells Inlet: each is a new `fisheries/*.yaml` (bbox, stations, orientation) + pipeline run + known-spots validation. No new code expected.

## 13. Risks & contingencies

| Risk | Contingency |
|---|---|
| ANUGA macOS install/runtime pain | Linux container (Docker/OrbStack), identical outputs |
| CUDEM gaps up-river | 1/3″ Coastal Relief Model fallback |
| SCDNR oyster layer unavailable | Manual oyster pins later; feature class is optional |
| SECOFS too coarse in-bay | It's evaluate-only; empirical models don't depend on it |
| Sparse current stations for calibration | Lean on the human validation gate |
| Basemap terms-of-service | Deliberately select a token-free source during implementation |
| System Python is 3.9 | Create the 3.12 venv from a modern interpreter (uv or Homebrew) |
