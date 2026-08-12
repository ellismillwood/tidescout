# Plan 1 Carryover Notes (for Plan 2+ authoring)

Distilled from Plan 1's execution ledger and final whole-branch review (2026-08-12) before workspace cleanup. Plan 1 landed on branch `plan-01-foundations` (9034c59..e43040a): all 10 tasks complete, final review "with fixes" and the fix wave applied + re-reviewed clean.

## Parked findings (real, deliberately deferred — with rulings)

1. **Discharge composite freshness/completeness signalling** (`usgs.py` `discharge_summary`): a gauge that stops reporting still contributes its last in-window reading to `cfs_now` (up to 4 days old), and a gauge returning nothing silently shrinks the composite with no indication. Ruling: nothing consumes discharge until Plan 3 scoring — fix as a Plan 3 kickoff task (per-site freshness cutoff ~6h + `contributing` list rendered as "2/3 gauges").
2. **`discharge_buckets` thresholds uncalibrated** (`fisheries/winyah-bay.yaml`: low<6000 / high>25000): authored before gauges were chosen. The real composite is ~98% Pee Dee at Peedee (Waccamaw-at-Longs and Black-at-Kingstree are tiny upstream gauges) — the composite hovers near the low boundary and 25000 is nearly unreachable. Ruling: Plan 3 kickoff — pull ~1 year of daily values, set thresholds at composite ~25th/75th percentiles, record derivation in a YAML comment.

## Notes that MUST inform Plan 2's brief

- **Always-24 DST contract**: `DayConditions.hours` is always exactly 24 wall-clock-labeled rows (spring-forward phantom hour appears as a label; fall-back repeated hour appears once). The Plan 1 document's "23/25" wording is stale; the code/tests are authoritative.
- **Pure-math placement**: `stage_at`, `_cosine_height`, `interpolate_tide_hours`, `interpolate_current_hours` are pure functions living in `sources/noaa.py`; `engine/` reaches back for `stage_at`. Final review recommends moving pure tide math to `engine/tides.py` before Plan 2 adds engine modules — make the boundary structural, not conventional.
- **Path resolution**: `fisheries/` and `data/` resolve via `__file__` parent-walks (`config.py` parents[2], `cache.py` parents[3]) — editable-install-dependent, deliberate for a local-only tool. Plan 2's bathymetry artifacts should follow the same pattern knowingly (or centralize a `paths.py`).
- **Lint gate is now explicit**: `[tool.ruff.lint] select = ["E","F","I","UP","B","DTZ"]`, line-length 100. Write brief code against that reality (collections.abc imports, tz-aware datetimes, no closures capturing loop vars, ≤100-char lines).
- **Slug sanitization**: `load_fishery(slug)` does no path sanitization — fine for CLI; MUST be fixed in the plan that first exposes slug over HTTP (Plan 5 API).
- **Winyah data realities** (live-verified): no harmonic tide station in the bbox (heights/currents are interpolated from subordinate-station events — already implemented); recorded stations: tide 8662549 (South Island Ferry), currents ACT6531 (entrance); rivers 02131000 / 02110500 / 02110500→(Waccamaw at Longs) / 02136000; water sensors 021108125 + 02110815 (temp+salinity), 02136371 (temp). Discovery output includes out-of-watershed basins (Cooper/Ashley, South Santee) — human-curated recording is the filter.
- **Salinity sensor placement**: configured USGS sensors are upriver — read ~0 ppt vs 15 ppt August mid-bay climatology. Plan 4's salinity model must use bay-appropriate observation points (NERR/SECOFS evaluation) and the CLI water line should label salinity provenance separately if touched earlier.

## Smaller deferred minors (final review: all OK-TO-DEFER)

Interpolator skeleton duplication (`_bracket_hours` helper candidate); `-0.0` possible in `tide_height_ft` and water-temp trend displays (`_snap_zero` reusable — mind that negative tide heights below MLLW are meaningful); `WaterSensor.params` recorded in YAML but unread by code (use it or rename); `kind: "coops"` water sensors accepted by the model but unconsumed (`noaa.water_temp_latest` has no production caller); `moon_info` computed twice per `load_day`; `_daily_means` keys by UTC date not fishery-local; `usgs-iv` fetches ~390KB/15min for mostly-unused breadth; `stations` CLI has no error degradation; no structural guard against tests hitting live APIs (autouse `respx.mock(assert_all_mocked=True)` fixture candidate); `ttl=None` rows permanent (README documents `rm data/cache.sqlite`); cache-behavior edges (TTL boundary test, connection close) — see Plan 1 ledger history in git if archaeology is ever needed.

## Process facts worth keeping

- Subagent reports can contain confident fabrications (one implementer invented table digits where terminal truncation hid columns — self-caught and corrected). Reviews must verify against diffs/files, never trust report prose.
- Three implementer sessions died to API connection drops at report-writing time; work was always already committed — check `git log` before re-dispatching anything.
- CO-OPS mdapi station metadata casing is unreliable for harmonic/subordinate classification — test the actual endpoint instead.
