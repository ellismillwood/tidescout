# TideScout frontend design (spec §9)

**Status:** approved 2026-09-03. Implements design spec §9 (UI) against the API
from PR #12 (`docs/superpowers/specs/2026-09-03-api-design.md`). This is the last
greenfield piece of the original rollout.

## 1. What this is, and four things §9 assumes that do not exist

§9 sketches a map with ambush markers, a scrubbing 24-hour strip, a conditions
rail, and four layer toggles. Most of it is served by the day payload today.
**Four pieces of it have no data path**, all found by reading the pipeline rather
than the sketch:

| §9 asks for | Status | Resolution |
|---|---|---|
| Marker activation, per hour, per species | Ships today | — |
| Depth contours | Ships today (812 features, each with `depth_m`) | — |
| Hillshade underlay | Built in PR #12 | — |
| **Flow arrows for the selected hour** | `payload["flow"]` is regime metadata — no vector field | New endpoint (§3) |
| **Salinity overlay** | `payload["salinity"]` is a bay-wide scalar per hour — no spatial field | New endpoint (§3) |
| **Depth tint** | No artifact; only `bathy_utm.tif` (EPSG:26917, float32) | New offline artifact (§4) |
| **Right-rail conditions + tide curve** | `_hour_to_dict` emits no raw conditions — they survive only as prose inside `reason` strings | New payload block (§5) |

The last is the largest: you cannot draw a tide curve from the sentence
`"pressure +0.7 mb/3h — steady"`.

### 1.1 Owner decisions

- **Salinity is drawn, and heavily marked as modelled.** The model is falsified
  (`fitted: false`, residual ~1,159x observation resolution). It is rendered
  hatched, with a permanent UNCALIBRATED badge and no crisp isolines that would
  imply precision. This follows Phase 3's ratified principle: **flagged, not
  discounted**. Excluding it would be the discounting that decision rejected.
- **Basemap is OSM raster now, satellite later.** The layer stack takes its
  basemap from one config entry so a second source is a config change, not a
  rewrite. Attribution is required and rendered.
- **Full §9 in one plan.** Not staged.

## 2. Structure and toolchain

`frontend/` beside `backend/`. Vite + React + TypeScript with **strict mode on**
(spec §11). Vite dev on `:5173` proxying `/api` to `:8000`; `npm run build` emits
`frontend/dist`, which `create_app` already mounts with the SPA fallback added in
PR #12. No new backend serving work.

```
frontend/src/
  api/     client.ts, types.ts          fetch + the payload contract as TS types
  state/   DayContext.tsx               payload, hour, species, fishery
  map/     MapView.tsx, layers.ts, join.ts
  strip/   HourStrip.tsx                24 bars, tide underlay, playhead
  rail/    ConditionsRail.tsx, FactorBars.tsx
  ui/      TopBar.tsx, Freshness.tsx, Disclosure.tsx
```

Split by responsibility, not by technical layer — the map's join logic lives
beside the map that consumes it.

**`types.ts` is hand-written and pinned by a contract test.** An earlier draft
of this spec said to generate it from `/api/openapi.json`. That is wrong twice
over, and the reason matters:

- `get_day` has no return-type annotation and returns a bare `JSONResponse`, so
  OpenAPI describes its response as untyped. There is nothing to generate from.
- Adding a Pydantic response model to fix that would be **actively harmful**.
  FastAPI filters responses through the declared model, and PR #12's Task 4
  review confirmed that the *absence* of one is exactly why `missing` and
  `confidence` reach the client verbatim. A response model is one schema
  omission away from silently stripping the disclosure fields — the failure
  §7's rule exists to prevent.

So the types are written by hand, and drift is caught by a **contract test**
instead: a committed real payload fixture, plus a test asserting the key set the
TS types expect is exactly the key set the payload carries — at the top level,
per hour, per feature-hour, and inside `sub_scope`. A new field in the payload
fails that test rather than being silently ignored by the UI.

**No state-management library.** One payload, one selected hour, one species, one
fishery. React context plus `useMemo` covers it. Zustand or Redux would be
machinery without a job.

## 3. Two new API endpoints

```
GET /api/fisheries/{slug}/flow-vectors/{date}?model=best&hour=15
    → a DOWNSAMPLED u/v grid for that hour's blended state (~2-3K arrows,
      not 587,325 cells), plus its grid spec.

GET /api/fisheries/{slug}/salinity-field/{date}?model=best&hour=15
    → ppt over the along-estuary distance field at that hour's phase and
      discharge. Carries `fitted` and `extrapolated` so the client cannot
      render it without the disclosure.
```

Both reuse PR #12's existing `_check_model` allowlist and readiness gate — the
same code that closed the path-traversal vulnerability, **not a parallel path**.
A third endpoint taking an unvalidated `model` would reintroduce it.

**Both are cheap.** `_blended_state` measured at **24 ms** on the real library;
the salinity field is a pure function over a 587,325-cell array that is already
loaded. Neither touches `structure_fields` or `sample_features`, which is where
all 70 seconds of `build_payload` actually goes.

**A deliberate exception to "scrubbing never refetches" (design spec §3).**
These are per-hour requests, so scrubbing *with an overlay enabled* does refetch.
The alternative — prefetching 24 hours x 2 overlays into the day payload — would
undo the 49% size reduction PR #11 achieved. Fetching a small thing on demand
beats inflating the large thing. Both overlays are **off by default**, so the
default scrub path still refetches nothing.

## 4. The map

### 4.1 Layer stack, bottom to top

```
1  basemap         OSM raster (satellite swappable via one config entry)
2  depth tint      NEW artifact, colour-ramped by depth
3  hillshade       PR #12's PNG, blended for relief
4  contours        line layer, labelled from depth_m
5  oyster reefs    fill; 1.9 MB, all 8451 reefs
6  flow arrows     OPTIONAL — per-hour fetch (two layers: casing + shaft)
7  markers         circles; radius and colour = activation
```

**AMENDED, 2026-09-04 — salinity is not a map layer.** This list originally
had the salinity field as layer 6, hatched, under the flow arrows. It shipped
as a hatched along-estuary PROFILE in the chart's margin instead (`MapView`'s
`SalinityInset`), and the layer stack above is what the code actually draws.

Why, and what it would take. `/salinity-field` returns a **1-D profile**: one
ppt per along-estuary kilometre bin, because that is what the model is — a
function of distance, discharge and phase, with no second spatial dimension in
it. Painting a 2-D field from that needs `estuary_km.npy`, the per-cell
distance field the backend joins those bins back onto. The client has no such
field: `api/layers.py`'s `LAYERS` allowlist does not carry it, so there is no
URL that serves it.

So this is **not impossible — it is one served artifact away.** The array
exists (`data/winyah-bay/estuary_km.npy`, 587,325 cells) and the backend
already loads it (`pipeline.estuary.load_distance_field`, called by the
endpoint itself). Serving it would mean a web-ready warp (the raw `.npy` is
neither web-mercator nor an image) plus a `LAYERS` entry, i.e. a fifth entry
in `webartifacts` beside the depth tint.

What was NOT acceptable was the alternative: inventing an estuary axis on the
client — interpolating distance from the shoreline, or from the marker
positions — and tinting the bay with it. The salinity model is `fitted:
false`, an uncalibrated fit whose own disclosure says so on every reading. A
falsified model painted across geography the client made up is exactly the
overclaiming §1.1 forbids, and it would be the most confident-looking thing on
the chart. A profile in the margin says what the model says and no more.

### 4.2 The depth tint artifact

Colour-ramp `bathy_utm.tif` (3806 x 5053, float32, nodata −9999, EPSG:26917),
reproject to EPSG:3857, emit a PNG sharing the hillshade's existing bounds
sidecar. Added to the same `tidescout bathy artifacts` step and the same
`LAYERS` allowlist. Marginal cost is low: `webartifacts.hillshade_png` already
does this warp on an identically-shaped grid.

### 4.3 The scrub loop

At load, `join.ts` writes the payload's activations onto the `features.geojson`
FeatureCollection as flat numeric properties — `a_redfish_0` through
`a_southern_flounder_23`. That is 529 x 72 ≈ 38K numbers, written **once**,
before the source reaches MapLibre.

Scrubbing then calls only:

```ts
map.setPaintProperty("markers", "circle-radius", radiusExpr(species, hour))
map.setPaintProperty("markers", "circle-color",  colorExpr(species, hour))
```

Each builder returns `["interpolate", ["linear"], ["get", `a_${species}_${hour}`], …]`.

Note the species ids themselves contain underscores (`speckled_trout`,
`southern_flounder`), so a key like `a_speckled_trout_11` cannot be split back
apart unambiguously. **These keys are only ever constructed, never parsed** — the
species and hour are always known from state. Anything that needed to parse one
should be reading state instead.

**No JavaScript touches a feature per tick.** The renderer re-evaluates a small
expression internally, so a playhead drag stays smooth and species switching is
the same operation with a different key — which is what §9's "switch re-colors
instantly" requires.

Rejected alternatives, recorded so they are not re-proposed:

- **`setFeatureState` per feature per tick** — 529 calls at frame rate. The
  idiomatic answer, and the one most likely to feel sticky under a fast drag.
- **Rebuilding the GeoJSON source's `data` per hour** — re-parses and re-uploads
  geometry that never changes.
- **A dynamic `["get", <expression>]` key** — would avoid rebuilding the
  expression, but depends on `get` accepting a computed string, which is easy to
  get subtly wrong across MapLibre versions, for a nil win.

### 4.4 The 2162-vs-529 problem falls out for free

`features.geojson` holds 2162 detected features; the payload scores only the 529
inside the flow-model domain. Unscored features simply have no `a_*` properties,
so `["coalesce", ["get", key], -1]` yields −1 and a `["case", ["<", v, 0], …]`
branch paints them muted and non-interactive.

**No parallel code path and no lookup that silently returns nothing** — the trap
named in the API spec becomes a styling branch.

### 4.5 Marker popover

Click assembles the ten-factor breakdown by the positional merge PR #11
specified: `species[name].hours[i].subs` for the seven hour-scope factors,
`features[key].hours[i].subs` for the three feature-scope ones. The split is read
from the payload's **`sub_scope`**, never hardcoded — that field exists precisely
so a factor changing scope does not break the frontend silently.

## 5. The payload's new `conditions` block

24 entries of the raw `HourlyConditions` values (tide height and phase, wind
speed/direction/gust, pressure and 3 h trend, air and water temperature, cloud
cover, precipitation) plus the day's sun and moon times.

**At the payload's TOP LEVEL, not per species.** These are fishery-wide hour
facts. Duplicating them across three species would repeat exactly the modelling
error PR #11 corrected — and `sub_scope` exists to document that hour-scope and
feature-scope facts live in different places. Cost is ~24 x 12 numbers, noise
against 24.59 MB.

## 6. Strip, rail, and disclosure

**Bottom strip.** 24 bars, height = the selected species' score; tide curve
underlaid from `conditions[i].tide_height_ft`; draggable playhead, arrow-key
scrubbing, prev/next day. This is the only view of a whole day at once, so it
carries disclosure too: an hour with `confidence < 1.0` is visually distinct from
a fully-observed hour.

**Right rail.** The hour's conditions as structured values, plus one factor bar
per factor — length from the sub-score, label from its `reason` string verbatim.
The reason strings are the honest part; they are where "UNCALIBRATED model
estimate, no observation constrains it" actually reaches a person.

**Disclosure is a component, not a badge someone remembers.** Four signals with
four distinct meanings:

| Signal | Source | Means |
|---|---|---|
| Freshness | `freshness.generated_at`, `/status` `stale` | how old this scoring run is |
| Confidence | `hour.confidence` | how many factors resolved |
| Constrained share | `hour.constrained_share` | how much rests on measurement vs model |
| Provisional | `hour.provisional[]` | *which* factors are unconstrained, by name |

`confidence` and `constrained_share` are **separate numbers in the payload on
purpose**. A component that merged them into one "quality" score would erase a
distinction five PRs went into establishing. The salinity overlay's UNCALIBRATED
badge is this same component in its most emphatic mode.

## 7. Error handling

**A degraded payload is not an error.** `missing: ['weather']` with
`confidence: 0.79` renders the full UI with disclosure showing — never an error
screen, never a silent omission. This is the same rule the API layer was
forbidden from breaking, carried to the client.

| State | Source | UI |
|---|---|---|
| Building | `202` from `/day` | Progress state, poll `/status`. Show **elapsed**, not a fake progress bar — the 70 s is honest |
| Build failed | `/status` → `failed` | Show the error text, offer retry; the next request genuinely retries |
| Fishery unprocessed | `ready: false` / `409` | Greyed in the picker, with `reason` naming what is missing |
| Date out of range | `422` | Picker constrained to the usable range up front; the 422 is the backstop |
| Layer missing | `404` from `/layers` | Map renders without it. A missing depth tint must not blank the map |
| Overlay fetch fails | `/flow-vectors`, `/salinity-field` | Toggle reverts with an inline error; base map unaffected |

## 8. Testing

Strict TypeScript, Vitest for units, Playwright smoke once stable (spec §11).

The units worth real tests are the pure ones:

- **`join.ts`** — a scored feature gets all 72 `a_*` properties **and** an
  unscored feature gets none, so §4.4's muted branch is genuinely reachable. A
  test checking only the scored side would pass against a join that wrote
  defaults to all 2162.
- **The payload contract test** (§2) — the TS types' key set matches a real
  committed payload fixture's key set, at every level.
- **Paint expression builders** — the expression references the expected key
  *for the given species and hour*, not merely that an object came back.
- **The popover merge** — produces all ten factors **and** reads `sub_scope` from
  the payload. A test with a hardcoded factor list would pass against a frontend
  that hardcodes the split, which is the failure `sub_scope` exists to prevent.
- **The polling machine** — `202 → building → ready` observed as a transition,
  not just a final state.

**Playwright smoke:** load a warmed day; drag the playhead and confirm marker
styling changes; click a marker and confirm ten factors; switch species and
confirm re-colour. The last is the cheapest real check that §4.3 works.

### 8.1 The test standard, stated because it has bitten every review

**A test asserting something is *present* without asserting it is *true* does not
count.** That includes assertions inside loops that never execute — the defect
that made four API tests pass against completely unmodified code during PR #12's
execution. On a frontend the trap is sharper: `expect(markers).toBeDefined()`
passes when zero markers rendered. **Every collection assertion states a count or
a value.**

## 9. Out of scope

- **Visual design decisions** — palette, type scale, chart forms. Design spec §9
  calls for the `frontend-design` and `dataviz` skills, and those belong at
  implementation time. The plan names where each applies (dataviz: hour strip,
  factor bars, tide curve; frontend-design: overall direction). Choosing colours
  in a spec document would be guessing without pixels.
- **Mobile layout.** Desktop-only is the stated target; see the API spec §1.1.
- **Offline/PWA behaviour.**
- **Multi-fishery comparison.** The picker switches; it does not compare.
