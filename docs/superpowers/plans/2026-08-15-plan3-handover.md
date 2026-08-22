# Plan 3 Handover — ANUGA Flow-State Library

**Written 2026-08-15. Branch `plan-03-anuga`, HEAD `cebfb5a`, 146 tests green, working tree clean.**

Read this before touching anything. It exists because Plan 3 burned three full library
builds (~17 hours of compute) on a hydrodynamic instability, and the expensive knowledge is
in *what was ruled out*, not in the code.

Companion documents:
- `docs/superpowers/specs/2026-08-11-tidescout-design.md` — the spec (authority)
- `docs/superpowers/plans/2026-08-13-03-anuga-flow-library.md` — the plan
- `docs/superpowers/plans/2026-08-13-plan3-anuga-spike-findings.md` — measured ANUGA facts
- `.superpowers/sdd/2026-08-13-03-anuga-flow-library/progress.md` — the full chronological ledger
  (every ruling, every dead end, all raw numbers)

---

## 1. Where the work actually stands

| Task | State |
|---|---|
| 1 Discharge recalibration + freshness | complete, reviewed clean |
| 2 Feature size gates | complete, reviewed clean |
| 3 Wall typing (p90 slope) | complete, reviewed clean |
| 4 Pipeline tests (derivatives/artifacts) | complete, reviewed clean |
| 5 ANUGA dep + model domain config | complete, reviewed clean |
| 6 Mesh builder | complete, reviewed clean |
| 7 Manning friction from zones | complete, reviewed clean |
| 8 Boundary forcing (tide datum, river inflow) | complete, reviewed clean |
| 9 Single-regime runner + checks | complete, reviewed clean |
| 10 Parallel driver + `flow` CLI | code complete + reviewed; **library never successfully built** |
| 11 Rasterise snapshots to grid | NOT STARTED |
| 12 Pure runtime lookup engine | NOT STARTED |
| 13 Known-spots validation gate | NOT STARTED |

**The blocker: no regime has ever completed.** Three builds, twelve-plus regimes, all died with
ANUGA's `Too small timestep ... even after 50 steps`.

---

## 2. The instability — what is known, and what is still open

### Ruled OUT, with evidence (do not re-test these)

| Hypothesis | How it died |
|---|---|
| Bed relief / steep channel banks | Failing cell was 99.8th-percentile relief (5.25 m), but saved snapshots showed several of the 10 fastest cells at LOW relief (0.78–1.14 m) and ALL deep (2.8–6.5 m). Correlation, not cause. |
| River inflow placement | Repositioned Pee Dee 2,476 m and Waccamaw 2,469 m off the domain boundary. Build #2 crashed at the same sim-times as #1. |
| River discharge at all | Instrumented run with inflows **entirely disabled** reproduced the identical hotspot at the identical location — 38 of 42 samples. |
| Mass drift | Residual 1e-13 to 1e-15 for the entire run, every sample. |
| Flow-energy threshold | neap/mean/spring and low/med/high all fail at the same place. |
| `minimum_allowed_height` (1e-3, 1e-2) | Bit-identical failures. Inert under DE0 — only read by Python-level velocity helpers, not the flux kernel. |
| `maximum_allowed_speed` (3.0) | Bit-identical failure. Inert. |
| `minimum_triangle_angle` (20/25/28/30/32) | Bit-identical output at every value. Inert. |
| DE1 flow algorithm | Unstable at t≈1800–2100 s in a probe. |
| DEM smoothing (sigma 1, 2) | Better baseline dt, still failed. **Rejected on principle too** — it blurs the dropoffs and channel edges this app exists to find. |
| Graded/annular jetty refinement | Actively worse: min radius collapses 1.84 → 0.15 → 0.10 → 0.02 m as rings are added. |
| Removing jetty refinement entirely | Buys only ~8% (min radius 1.84 → 1.98 m). The timestep is NOT jetty-dominated. |

### CONFIRMED cause #1 — mis-tagged ocean boundary at river heads (FIXED)

`mesh.classify_boundary` decided ocean-vs-wall on **bed depth alone**. Correct at the seaward
opening; nonsense wherever the domain polygon severs a deep inland channel, because a river
channel 40 km upstream is indistinguishable from open coast by depth.

Four segments at the Pee Dee head were tagged `ocean`, so the full ocean tide was imposed 40 km
inland through a ~150 m slot in an otherwise solid reflective wall. Instrumented evidence: a
perfectly stationary hotspot at UTM **(662770, 3698800)**, 9 m from those segments,
`hot20_persist` 15–20 of 20 at every sample, growing 0.47 → 6.25 m/s over one tidal cycle,
tidally modulated (worst at mid-ebb), and phase-locked to just after the SECOND high water.

**Fixed** by the three-class scheme (`a3159c4` + review fixes). **This worked** — that hotspot does
not appear anywhere in build #3.

### OPEN cause #2 — seaward-boundary instability (build #3)

With the Pee Dee hotspot gone, build #3 failed instead at:
```
Triangle #232026 (672405, 3690284)  13,282,282 m/s
Triangle #221981 (672619, 3679888)  10,551,011 m/s
```
Both on the **east ocean apron** along North Island — the legitimate seaward face.

Build #3 also mapped `open` → `anuga.Transmissive_boundary`, which regressed badly:
max dt **0.016–0.020 s** vs ~0.2 s, 248,664 steps for 33,431 s of sim, per-regime wall time
2.7 h → ~14 h, and crashes ~12,000 s earlier in sim time.

`cebfb5a` reverts `open` → `Reflective_boundary`. **This is UNTESTED.** The run that would have
tested it was killed at t=0.

### The immediately next experiment

Run instrumented regimes to t≈50,400 s and read the timestep:
- **dt recovers to ~0.2 s** → `Transmissive_boundary` was the whole regression; then see whether
  the east-boundary failure also disappears.
- **dt stays ~0.016 s** → something else degrades it and the transmissive choice was a red herring.

**Run several variants in parallel — the machine has 6 performance cores and this was repeatedly
done serially, which is the single biggest process failure of this plan.** Useful variants:
`mean_med` (baseline), `neap_low`, `spring_high`, and a control with the east apron trimmed out of
the domain polygon to test cause #2 directly.

Instrumented harness: `.superpowers/sdd/2026-08-13-03-anuga-flow-library/_instrumented_regime.py`
(also copied there as `instrumented*`). Usage:
`_instrumented_regime.py <range> <discharge> <sim_hours> <tag> [--no-inflow]`.
It logs per yieldstep to `instr-<tag>.csv`: mass identity terms, dt, wet fraction, speed
distribution, counts over 1 and 3 m/s, location/depth/radius of the fastest cell, and
`hot20_persist` (how many of the top-20 fastest cells carry over — this is what distinguishes a
stationary growing hotspot from noise, and it is what cracked cause #1).
**It calls `regimes._boundary_map` directly so it cannot drift from production.**

---

## 3. Hard-won operational facts

- **The Mac sleeps and kills subagent sessions.** Four died to "computer went to sleep
  mid-response"; build #1 accumulated 8 CPU-minutes across 65 wall-minutes before anyone noticed.
  **Hold the system awake for the whole session**: `caffeinate -dimsu -t 21600 &`. Wrapping
  individual commands is not enough.
- **`tidescout flow run` prints nothing until all nine regimes finish.** Build #1's six failures sat
  undetected for over an hour. Always arm a monitor on `grep -c "Too small timestep"` in the log,
  on `snap_NNN.npz` progress, and on completion.
- **Never wait on `pgrep -f <pattern>` where the pattern appears in the waiting command** — it
  matches itself and never exits. Wait on a PID (`while kill -0 <pid>`) or on file contents.
- **Commit before writing reports.** Every implementer that died mid-task kept its work only
  because a commit already existed.
- Timings on this Mac (M5 Pro, 6 performance cores), 247,020-triangle mesh:
  ~530 s wall per sim-hour ⇒ ~2.7 h per 18.4 sim-h regime, ~5–6 h for nine at 6 workers.
  Build #3 with transmissive boundaries: ~14 h per regime.

## 4. Measured ANUGA facts worth not rediscovering

- Mass-conservation identity: `get_water_volume() - V0 == get_boundary_flux_integral() +
  get_fractional_step_volume_integral()`. Tolerance **1e-3**, not machine precision — though with a
  correct boundary it actually closes at 1e-13.
- `create_domain_from_regions` has **no `mesh_filename`** argument.
- `set_boundary` raises on any tag the mesh lacks — omit empty tags rather than passing empty lists.
- Omitting `hole_tags` makes ANUGA tag island holes `interior`, which then crashes `set_boundary`.
- Elevation must be set at **vertices**, not centroids — DE0 is a discontinuous-elevation scheme.
- Initial stage must be `max(elev, level)`, never `max(elev + eps, level)`: a 1 mm film over
  ~129,000 land cells collapses the timestep.
- `Inlet_operator(domain, region, Q=...)` — verified signature. It dies with an unrelated
  `AttributeError` on an empty region, and does NOT raise at all if the region contains only land
  centroids, hence the wet-centroid guard in `_attach_river_inflows`.

---

## 5. Owed work, beyond the instability

1. **Discharge-axis check (review Important 3, documented not mitigated).** The Pee Dee
   `Inlet_operator` sits 2,456 m downstream of `open` segment 7, so injected discharge can leave
   through the upstream open boundary. Will not crash; `mass_residual` correctly will not catch it.
   **After any successful build, compare snapshots across low/med/high and confirm they differ.**
2. **Tasks 11–13 unstarted** — rasterise to grid, pure lookup engine, known-spots validation gate.
   Task 13 is the spec's go/no-go and needs Ellis's judgement, not a metric.
3. **Deferred minors** are listed in the ledger (search `minor (deferred)`), including feature-id
   rebuild stability (carryover trap (c)) which Plan 4 must not skip.
4. **SCDNR oyster layer** is downloaded (`data/winyah-bay/oyster_reefs.geojson`, 8,451 polygons) and
   characterised: median reef 24 m², so it is a Plan 4 *scoring/habitat* layer, not mesh geometry
   and not an ambush-feature class. See the Plan 2 carryover notes.
5. **NWI wetlands** fetched and validated against the DEM: 82.4% of mapped Spartina marsh is
   classified intertidal by the model, only 4.7% clearly wrong. **The DEM's marsh boundary is sound
   — do not re-litigate it.**

---

## 6. Process lessons (the expensive ones)

1. **A crash dump shows the first cell to FAIL, never the thing that has been GROWING.** Two root
   causes were declared from crash dumps and both were wrong. The instrumented time series found
   the real one in a single run. **Instrument first.**
2. **Run hypotheses in parallel.** Six performance cores sat idle while single diagnostic runs
   executed serially, repeatedly, even after this was pointed out. Four parallel variants cost the
   same wall time as one and would have collapsed the search dramatically.
3. **Weigh a reviewer's evidence above your own first-principles reasoning.** The review warned
   `Transmissive_boundary` appears in no ANUGA validation script and is weakly ill-posed for
   subcritical inflow. That warning was under-weighted in favour of "it is the only variant that
   imposes neither level nor momentum," and it cost the overnight build.
4. **Depth alone cannot classify geography.** It failed for the domain polygon (ocean and estuary
   are hydraulically connected) and again for the ocean boundary (a deep river reads as open coast).
   Both had to become authored config. Assume the third instance exists.
5. **Verify a "root cause" with a controlled experiment before acting on it.** The no-inflow control
   settled in one run what two five-hour builds could not.
