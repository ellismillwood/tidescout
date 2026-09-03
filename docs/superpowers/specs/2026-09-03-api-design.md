# TideScout API design (spec §3)

**Status:** approved 2026-09-03. Implements the `api/` half of design spec §3
("FastAPI app: day-payload endpoint, serves built frontend"). The frontend
(§9) is a separate spec and is not designed here.

## 1. The problem this API exists to solve

`pipeline.payload.build_payload` takes **70 seconds** and does not get faster on
repeat calls — the cost is compute, not fetching. Profiled on the real
winyah-bay library, 2026-09-02:

| Stage | Time |
|---|---|
| `activation.sample_features`, 24 hourly calls | 39.1 s |
| `activation.structure_fields` → `structure.ambush_contrast` → scipy `maximum_filter` | 26.4 s |
| Scoring proper — 38,088 `score_feature` calls | 1.9 s |
| **Total** | **~70 s** |

Scoring is free. Essentially all of it is deriving flow structure from the
library, recomputed per hour. A synchronous `GET` that takes 70 s times out in
proxies and gives the UI nothing to render, so the central design question is
not "how do we serve a slow endpoint" but **"what is precomputed, and when."**

### 1.1 Decisions that fixed the shape

Two owner decisions (2026-09-02/03), recorded because they close off options a
later reader would otherwise reopen:

- **Target is DESKTOP ONLY for now.** This is why no lazy-loading, pagination
  or columnar payload encoding is specified. See §7.
- **Usage is "today plus the next few days, mostly."** This is what makes a
  nightly warming job sufficient and an on-demand build a rare path.
- **Only the `best` weather model is warmed.** The other five stay available
  on demand at one 70 s build each, then cached. The §9 picker survives
  without a 6× warming bill.

### 1.2 The option deliberately NOT taken

`structure_fields` and `sample_features` depend on `(regime, phase)` — **not on
the date**. The library holds 9 regimes × 26 stored phases = **234 distinct
states**, recomputed from scratch for every date ever opened. Precomputing
`FeatureMetrics` for all 234 would make any payload ~2–5 s, fast enough to serve
synchronously, deleting this document's entire async subsystem.

It is not specified here for two reasons. It optimises the path the owner says
he rarely takes, and it is **numerically unvalidated**: today the code blends
flow *states* and then derives structure, whereas that approach would
interpolate precomputed *metrics*. `structure_fields` is nonlinear — gradients,
Okubo-Weiss, a maximum filter — so blending metrics is not equal to metrics of
the blend, and the difference has never been measured.

**Revisit trigger:** if arbitrary-date hindcasting becomes a routine habit, or
the target moves to mobile. The first task then is to measure that
interpolation error against the current path, not to assume it is small.

## 2. Endpoints

```
GET  /api/fisheries
     → 200 [{slug, name, center, timezone, ready, reason?}]

GET  /api/fisheries/{slug}/day/{date}?model=best
     → 200 the day payload
     → 202 {status:"building", started_at, key}
     → 404 unknown fishery
     → 409 fishery not processed, naming what is missing
     → 422 date outside the usable range, stating the range

GET  /api/fisheries/{slug}/day/{date}/status?model=best
     → 200 {status:"ready"|"building"|"failed", generated_at?, stale?, error?}

GET  /api/fisheries/{slug}/layers/{name}
     → 200 a static per-fishery artifact (§4)
     → 404 unknown fishery or unknown layer name
```

`ready` in `/api/fisheries` drives §9's "unprocessed fisheries grayed out". A
fishery is ready when its flow library, along-estuary distance field and
`features.geojson` all exist. **The same predicate backs the `409`**, so the
list and the error can never disagree — one function, two callers.

`/status` is separate from the payload endpoint so that polling costs a few
hundred bytes rather than re-transferring megabytes on every tick.

## 3. The payload cache

**A directory, not a database:** `data/<slug>/payloads/<date>-<model>.json.gz`.
Cheap to inspect, cheap to delete, survives restarts, needs no schema or
migration. A single-user local tool earns nothing more than this.

### 3.1 Staleness

A cached payload is **stale** when `freshness.generated_at` is older than
`STALE_AFTER_H = 6` **and** its date is today or in the future. Past dates never
go stale: they are scored from ERA5 reanalysis and USGS daily means, neither of
which changes after the fact.

A stale hit is **served immediately** — with `generated_at` in the response so
§10's freshness badge can show its age — and triggers a background rebuild. The
user never waits on a rebuild for data already on disk.

**`STALE_AFTER_H = 6` is a judgement call, not a measurement.** Short enough
that an afternoon check reflects the morning forecast update, long enough to
avoid constant rebuilding. It is one constant, in one place, meant to be moved.

### 3.2 Concurrency and atomicity

- An **in-process dict of in-flight builds**, keyed by `(slug, date, model)`,
  so ten status polls or a double-clicked date picker cannot start ten
  70-second builds. Second and subsequent callers join the existing build.
- Builds run in a **thread pool, never on the event loop** — a 70 s CPU-bound
  numpy call on the loop would freeze every other request.
- Cache writes go to a **temp file and are atomically renamed** into place. A
  build killed mid-write must not leave a truncated payload to be served
  forever after.

### 3.3 Warming

`tidescout warm <slug> --days 7 --model best` builds today…+6, skipping any
payload already present and fresh. Roughly **8 minutes** for seven dates. It is
an ordinary CLI command — testable, runnable by hand, and schedulable from cron
or launchd. The API does not schedule anything itself.

## 4. Static artifacts

One endpoint, `GET /api/fisheries/{slug}/layers/{name}`. Both `slug` and `name`
are user input that reaches the filesystem, so **both are validated against
allowlists** and never concatenated into a path: `slug` against the known
fisheries, and `name` against exactly

```
{"features", "contours", "oysters", "hillshade", "hillshade-bounds"}
```

Anything else is a `404`, including a name that merely resolves to a real file.
This gets a test, not a code comment (§6).

These files are rewritten IN PLACE by `tidescout bathy artifacts` — same URL,
new bytes — so they are served with a strong `ETag` and `Cache-Control:
no-cache`, meaning *revalidate every time*, not *do not store*. A repeat load
costs one ~200-byte `304`, not another 8 MB.

**Not `Cache-Control: immutable`**, which an earlier draft of this section
specified: `immutable` is only sound for content-addressed URLs. On a fixed
path it tells the browser to serve its cached copy for a year without ever
sending `If-None-Match`, so a regenerated layer stays invisible until a hard
reload and the server's conditional-request handling becomes dead code.

| Layer | Source | Served size |
|---|---|---|
| `features` | `features.geojson` | 8.0 MB (~2 MB gzipped) |
| `contours` | `contours.geojson` | 6.8 MB |
| `oysters` | `oyster_reefs.web.geojson` | **1.9 MB (0.3 MB gzipped)** |
| `hillshade` | `hillshade.png` + `hillshade.bounds.json` | ~3 MB |

### 4.1 Two artifacts do not exist yet

Both are **offline pipeline steps added to `tidescout artifacts`**, not API
responsibilities. The API only serves their output.

**Hillshade.** `hillshade.tif` is a single-band GeoTIFF in EPSG:26917 (UTM 17N),
3806 × 5053. A browser can render neither the format nor the projection.
Requires reprojection to EPSG:3857 and export as `hillshade.png` with a
`hillshade.bounds.json` sidecar. Spec §9 assumes this underlay already exists;
it does not.

**Oysters.** `oyster_reefs.geojson` is 37.6 MB (11.5 MB gzipped) for 8451
reefs, of which 34.7 MB is geometry carrying **14 decimal places** per
coordinate. Trimming to 6 dp (~11 cm at this latitude), dropping the unused
`objectid`/`photo_year`/`calcgeo_ac` properties, and applying a
topology-preserving Douglas-Peucker simplification at ~2 m gives **1.9 MB raw,
0.3 MB gzipped, with all 8451 reefs retained**. Two metres of outline detail on
a shellfish bed is invisible at any fishing zoom.

> **`oyster_reefs.web.geojson` is a DISPLAY artifact only.** Feature scoring
> reads real oyster geometry — `features.geojson` carries `oyster_area_m2`,
> `oyster_density` and `oyster_nearest_m` derived from the full-precision
> source. Nothing in `pipeline/` may read the web version, or ambush scoring
> silently degrades to match a rendering optimisation.

### 4.2 The 2162-vs-529 trap

`features.geojson` contains **2162** detected features. The payload scores only
the **529** inside the flow-model domain. A frontend that renders all 2162 and
looks up activations will silently find nothing for 1633 of them.

**`species[name].features` keys are the authority on what is scored.** Unscored
features render as a visually distinct, non-interactive class — not as markers
that look scored and read as dead water.

## 4.3 Serving the built frontend

Spec §3 makes the API responsible for serving the built frontend, so it mounts
`frontend/dist` at `/` with an SPA fallback: any non-`/api` path that does not
match a file returns `index.html`, so client-side routes survive a page reload.
`/api/*` is matched first and never falls through to the SPA.

**Development runs two servers** — Vite on `:5173`, the API on `:8000` — so CORS
is enabled for `localhost:5173` **in development only**. In production the
frontend is same-origin and no CORS headers are emitted. The app binds
`127.0.0.1`, not `0.0.0.0`: this is a single-user local tool with no auth (§7),
and binding it to every interface would put an unauthenticated filesystem-backed
API on the local network.

## 5. Payload contract notes for the frontend

Settled 2026-09-02 (PR #11) and not to be re-litigated:

- **24.59 MB raw, 1.67 MB gzipped, 30 ms `JSON.parse`, 40.1 MB JS heap.**
  Scrubbing to any hour is **0.21 ms**, so §3's "scrubbing never refetches"
  holds with room to spare.
- A feature-hour carries **only** the factors that vary per feature
  (`flow`, `salinity`, `structure`) and **no `time`** — it is positionally
  aligned with `species[name].hours[i]`.
- The payload publishes **`sub_scope`**, naming which factors are hour-level
  and which are feature-level. The frontend **reads `sub_scope`** rather than
  hardcoding the split, so a factor changing scope does not break it silently.
  A marker popover's full ten-factor breakdown is the positional merge of the
  two.

## 6. Error handling

**A degraded payload is a `200`, not an error.** A dark sensor is already
modelled in the data — `missing: ['weather']`, a lowered `confidence`, a
`constrained_share` below 1.0. An API layer that turned those into a `500`, or
hid them behind a generic "unavailable", would undo the disclosure machinery
five PRs went into building. Degradation is data, and gets served as data.

| Case | Response |
|---|---|
| Unknown fishery | `404` |
| Fishery not processed | `409`, naming what is missing |
| Date out of range | `422`, stating the range |
| Build in progress | `202` + `{status:"building"}` |
| Build failed | `/status` reports `failed` + message; next request **retries** |
| Unknown or traversing layer name | `404` |

**Usable date range, measured 2026-09-03 rather than assumed:** Open-Meteo's
forecast endpoint currently allows through **+16 days**; anything older than
`ARCHIVE_CUTOFF_DAYS = 7` routes to the ERA5 archive, which reaches back years.
The bound that actually bites is therefore the future one.

**A failed build never writes a payload file.** The failure lives in the job
record, so a transient source outage does not poison the cache with a
permanently-degraded payload.

## 7. Explicitly out of scope

Named so a later reader does not mistake omission for oversight:

- **Lazy-loading, pagination, columnar encoding.** Measured unnecessary on
  desktop (§5). Revisit trigger: a second day held resident beside a MapLibre
  GL context on a phone — not a byte count.
- **The `(regime, phase)` structure cache.** See §1.2, including its revisit
  trigger and the measurement that must precede it.
- **Auth, multi-user, rate limiting.** Single-user tool bound to localhost.
- **Vector tiles.** The simplification in §4.1 makes them unnecessary at this
  scale.
- **The frontend itself.** Spec §9, its own design document, to be built with
  the `frontend-design` and `dataviz` skills as the design spec requires. This
  API *serves* the built frontend (§4.3); it does not design or build it.

## 8. Testing

`TestClient`, no network. Most tests monkeypatch `build_payload` and run in
milliseconds; **one** integration test exercises the real 70 s path, matching
how `synthetic_day_with_flow` already earns its cost in `test_payload.py`.

Required cases:

1. Cache hit serves `200` without invoking a build.
2. Cache miss returns `202` and starts exactly one build.
3. `building → ready` transition observable through `/status`.
4. A stale-but-present payload serves **immediately** and **also** triggers a
   rebuild — both halves asserted; serving without rebuilding, and rebuilding
   before serving, are each a distinct bug.
5. A failed build leaves **no** cache file and reports `failed`.
6. Concurrent requests for one key start **exactly one** build.
7. Path traversal on `/layers/{name}` is rejected.
8. A payload with `missing: ['weather']` is served as `200` with its
   `missing`/`confidence` fields intact — the §6 rule, pinned.
9. `ready: false` in `/api/fisheries` and the `409` agree, because they call
   the same predicate.

Tests assert **pairs, not points**, per this project's recurring defect: a test
that asserts a thing is present without asserting it is true. Cases 4 and 9
exist specifically because each has two halves that a single-sided assertion
would miss.
