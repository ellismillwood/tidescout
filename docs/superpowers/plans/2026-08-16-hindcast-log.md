# Hindcast log: predicted vs actual

**No row in this log is usable for tuning any curve until the `actual`
column is filled in by Ellis.** Every number under "predicted" below came
from `tidescout score`, i.e. from the engine as shipped in this branch --
it is a forecast, not ground truth. The ONLY ground truth this project has
is what actually happened on days Ellis fished, and no agent has that.
Filling `actual` with plausible-looking guesses would mean every future
weight is tuned against fiction while the tuning looks rigorous -- do not
do this, ever, to this file.

**This log was regenerated 2026-08-26** after a code review found the first
version unusable in two ways, both fixed before these rows were produced:
the three dates were in the FUTURE (Ellis cannot report `actual` for a day
that has not happened), and the header's discharge claim was wrong. Both
are explained below rather than just fixed silently, because the same
mistake is easy to make again the next time this log is regenerated.

## Read `constrained_share` before trusting a row

`constrained_share < 1.0` on a row means the salinity factor THAT hour
rests on `engine.salinity`'s uncalibrated model (`fitted: false` in
`fisheries/winyah-bay.yaml` -- Winyah's live, permanent state) rather than
an observation. All nine rows below read `constrained_share = 1.00`,
because the live USGS gauge (021108125) is currently reporting a real
salinity value (0.0 ppt -- a genuinely near-fresh tidal reach) and
`build_payload` prefers a real sensor reading over the model whenever one
exists. **A stale/dark salinity gauge is NOT what would flip this** --
`usgs.water_summary` never returns `salinity_ppt=None`; a gauge that stops
reporting SALINITY specifically falls back to a monthly climatology GUESS
while `source` stays labelled by whichever sensor (if any) still reports
TEMPERATURE, and `payload._bay_salinity_reading` treats that fallback as
MEASURED too unless `source` itself reads `"climatology"`. The actual
trigger is narrower: `water` failing entirely (the whole USGS water fetch
errors out) or EVERY configured sensor's TEMPERATURE series going dark at
once (so `source` itself falls to `"climatology"`). `test_an_uncalibrated_
salinity_reaches_the_payload_as_provisional` (`backend/tests/test_
payload.py`) exercises that path directly with a synthetic day, since none
of these nine rows do. The per-FEATURE numbers behind "top-ranked features"
are unaffected either way: `score_feature` always uses the spatial model at
the feature's own along-estuary distance, never the bay-wide sensor value,
so every map marker's activation rests on the unfitted model regardless of
what the gauge says.

## Discharge now varies correctly by date; salinity and water temperature still do not

**Fixed 2026-08-26 (this regeneration):** `sources.usgs.discharge_summary`
used to read only the LIVE gauge (`datetime.now(UTC)`, no `day` parameter)
-- `dayloader.load_day` called it the same way regardless of which date was
requested, so the first version of this log had all three rows sharing one
discharge reading despite naming different dates. `discharge_summary` now
takes an optional `day`; a date strictly before today reads THAT day's own
USGS daily mean (`fetch_daily`, the NWIS `dv` service already used by
`pipeline.salinity_fit` for calibration) instead of the live instantaneous
feed, and `dayloader.load_day` passes the requested date through. The three
discharge readings below (2,317.94 / 4,037.80 / 8,853.80 cfs) are each that
date's own real historical composite, genuinely spanning low/med/high.

**Still NOT fixed, found while regenerating this log:** `usgs.water_summary`
(water temperature AND the bay-wide salinity fallback used above) has the
SAME wiring gap `discharge_summary` had -- it takes no `day` and always
reads the live 7-day window. Every "water 88F" reason string and every
`representative_ppt: 0.0` below is TODAY's live reading, identically,
regardless of which of the three dates the row names -- confirmed by all
three rows reading the exact same water temperature. This was out of scope
for the review that prompted this regeneration (which named `discharge_
summary` specifically), so it was not fixed here, but it is the same class
of bug and should be closed the same way (`fetch_daily` against `PARAM_TEMP_C`
and `PARAM_SALINITY`, keyed by `day`) before this log is regenerated again.

## How these were produced

```
tidescout score winyah-bay 2026-07-21   # low discharge, neap tide
tidescout score winyah-bay 2026-08-05   # med discharge, neap tide
tidescout score winyah-bay 2026-07-28   # high discharge, spring tide
```

All three are real past dates (today is 2026-08-26). Each command scores
all three species from one `build_payload` call; the table below pulls the
day's score range, its single highest hour, and the TWO OR THREE LOWEST-
VALUE sub-scores that hour -- under `combine`'s weighted geometric mean, a
low value (not a high one) is what actually constrains the score, and the
CLI itself was corrected to show the same thing this same review round
(`tidescout score`, `cli.py`'s `score` command -- an earlier version showed
the highest-value subs, which are precisely the ones that did NOT move a
geometric mean).

## Predictions

| date | tide regime | discharge (cfs) | bucket | species | score range (day) | peak hour | peak score | limiting factors at peak (value — reason) | confidence | constrained_share | actual | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-21 | neap | 2,317.94 | low (clamped to `neap_low`) | redfish | 57-70 | 03:00 | 70 | salinity 0.45 — "salinity 0.0 ppt — near-fresh"; flow 0.46 — "flow 0.10 m/s — moving" | 1.00 | 1.00 | | |
| 2026-07-21 | neap | 2,317.94 | low | speckled_trout | 39-47 | 03:00 | 47 | salinity 0.05 — "salinity 0.0 ppt — near-fresh"; water_temp 0.45 — "water 88F" | 1.00 | 1.00 | | |
| 2026-07-21 | neap | 2,317.94 | low | southern_flounder | 52-65 | 07:00 | 65 | salinity 0.20 — "salinity 0.0 ppt — near-fresh"; water_temp 0.41 — "water 88F" | 1.00 | 1.00 | | |
| 2026-08-05 | neap | 4,037.80 | med (`neap_low`/`neap_med` blend) | redfish | 56-69 | 13:00 | 69 | light 0.43 — "4.9 h from twilight, 70% cloud widened it from 6.5 h"; flow 0.45 — "flow 0.10 m/s — slack" | 1.00 | 1.00 | | |
| 2026-08-05 | neap | 4,037.80 | med | speckled_trout | 38-49 | 19:00 | 49 | salinity 0.05 — "salinity 0.0 ppt — near-fresh"; water_temp 0.45 — "water 88F" | 1.00 | 1.00 | | |
| 2026-08-05 | neap | 4,037.80 | med | southern_flounder | 52-67 | 19:00 | 67 | salinity 0.20 — "salinity 0.0 ppt — near-fresh"; water_temp 0.41 — "water 88F" | 1.00 | 1.00 | | |
| 2026-07-28 | spring | 8,853.80 | high (`spring_high`/`spring_freshet` blend) | redfish | 61-77 | 20:00 | 77 | salinity 0.45 — "salinity 0.0 ppt — near-fresh"; flow 0.57 — "flow 0.14 m/s — moving" | 1.00 | 1.00 | | |
| 2026-07-28 | spring | 8,853.80 | high | speckled_trout | 42-52 | 20:00 | 52 | salinity 0.05 — "salinity 0.0 ppt — near-fresh"; water_temp 0.45 — "water 88F" | 1.00 | 1.00 | | |
| 2026-07-28 | spring | 8,853.80 | high | southern_flounder | 56-66 | 13:00 | 66 | salinity 0.20 — "salinity 0.0 ppt — near-fresh"; water_temp 0.41 — "water 88F" | 1.00 | 1.00 | | |

`flow.clamped` was `True` on 2026-07-21 (2,317.94 cfs sits genuinely below
`neap_low`'s own simulated flow, 2,774 cfs -- a real single-regime pin, not
just the boundary-inclusive-as-suspect case `payload.py` also flags) and
`False` on the other two (genuine two-regime blends: `neap_low`/`neap_med`
and `spring_high`/`spring_freshet`). `salinity.extrapolated` was `False` on
all three -- none of these real discharges reach 22,996 cfs, the top of the
1,232-22,996 cfs calibrated span. Neither `test_payload_flags_a_clamped_
discharge_blend` nor `test_payload_flags_an_extrapolated_salinity`
(`backend/tests/test_payload.py`) depends on these dates; both use a
synthetic freshet day instead.

`salinity` reads identically "near-fresh, 0.0 ppt" on every row for the
reason given above (the live gauge, not a historical one) -- read it as
"today's live salinity applied uniformly," not as each date's own
condition, until `water_summary` gets the same `day` fix `discharge_
summary` got here.

## Ellis's homework

Pick three days you remember well -- ideally one excellent, one poor, one
middling -- and fill in `actual` (what you caught, how the bite felt,
whether it matched the hour the model called out as best) and `notes`
(tide stage, wind, anything the model didn't have -- water clarity, bait
present). They do not need to be these exact three dates; these three exist
to prove the pipeline runs end to end against real, genuinely different
discharge, and to show the tidal-range regime blend responding correctly to
a real neap/spring spread. Once you have three real days with `actual`
filled in, that is the point to start looking at whether any curve in
`fisheries/species_weights.yaml` should move -- and even then, three days
is a first look, not a fit.
