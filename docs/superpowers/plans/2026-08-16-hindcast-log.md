# Hindcast log: predicted vs actual

**No row in this log is usable for tuning any curve until the `actual`
column is filled in by Ellis.** Every number under "predicted" below came
from `tidescout score`, i.e. from the engine as shipped in this branch --
it is a forecast, not ground truth. The ONLY ground truth this project has
is what actually happened on days Ellis fished, and no agent has that.
Filling `actual` with plausible-looking guesses would mean every future
weight is tuned against fiction while the tuning looks rigorous -- do not
do this, ever, to this file.

**Read `constrained_share` before trusting a row.** `constrained_share < 1.0`
on a row means the salinity factor THAT hour rests on `engine.salinity`'s
uncalibrated model (`fitted: false` in `fisheries/winyah-bay.yaml` --
Winyah's live, permanent state; see that file's "CALIBRATION ATTEMPTED AND
DECLINED" block) rather than an observation. All nine rows below happen to
read `constrained_share = 1.00` for the FISHERY-WIDE score, because the
live USGS gauge (021108125) is currently reporting a real salinity value
(0.0 ppt -- a genuinely near-fresh tidal reach) and `build_payload` prefers
a real sensor reading over the model whenever one exists (see
`pipeline/payload.py::_bay_salinity_reading`). This is a **coincidence of
when this log was generated**, not a property of the model: on any day that
gauge is stale or dark, the fishery-wide row would read
`constrained_share < 1.0` too, exactly as `test_an_uncalibrated_salinity_
reaches_the_payload_as_provisional` (`backend/tests/test_payload.py`)
verifies happens on Winyah's actual, ordinary path. The per-FEATURE numbers
behind the "top-ranked features" the CLI prints are unaffected by that
coincidence either way: `score_feature` always uses the spatial model at
the feature's own along-estuary distance, never the bay-wide sensor value
(see `engine/score.py`'s module docstring), so every map marker's activation
in this log rests on the unfitted model regardless of what the gauge says
today.

**Discharge does not vary by date in this build.** `sources.usgs.
discharge_summary` reads the LIVE USGS gauge (`datetime.now(UTC)`,
no `day` parameter) -- `dayloader.load_day` calls it the same way
regardless of which date is requested. All three dates below were
generated in the same session and so share one discharge reading
(3,354.6 cfs, `med` bucket) even though they were chosen to span
different real TIDAL ranges (spring / mean / neap, from the moon's
illuminated fraction -- see `pipeline/payload.py::_range_bucket_for_day`).
**"Three dates spanning the discharge range," as asked for in the task
brief, is not actually obtainable from this pipeline today** -- there is no
historical discharge lookup, only a live one. That is a real, current
limitation worth a follow-up task (a USGS daily-values query keyed on
`day`, alongside the existing live `iv` one), not something this log should
paper over. What DOES vary correctly below is the tidal-range regime blend
(`spring_low`/`spring_med` vs `mean_low`/`mean_med` vs `neap_low`/`neap_med`)
and the real CO-OPS tide predictions for each date.

**Collect all three before tuning anything.** Even once `actual` is filled
in, a single day's agreement or disagreement fits noise -- these curves
(`fisheries/species_weights.yaml`) are the most over-fittable surface in the
project. Wait for all three rows' `actual` before touching any weight or
curve breakpoint, and even then, treat one round of three days as a first
look, not a calibration.

## How these were produced

```
tidescout score winyah-bay 2026-08-28   # spring tide
tidescout score winyah-bay 2026-09-01   # mean tide
tidescout score winyah-bay 2026-09-04   # neap tide
```

Each command scores all three species from one `build_payload` call (the
whole point of the payload is that switching species never re-scores); the
table below pulls the day's score range, its single highest hour, and the
two-to-three factors that hour's own `subs` names as driving it -- read
straight from the JSON, not eyeballed off the printed table.

## Predictions

| date | tide regime | discharge (cfs) | species | score range (day) | peak hour | peak score | top factors at peak | confidence | constrained_share | actual | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-28 | spring (`spring_low`/`spring_med` blend) | 3,354.6 (`med`) | redfish | 56-75 | 20:00 | 75 | wind 1.00, light 0.98, stage 0.94 | 1.00 | 1.00 | | |
| 2026-08-28 | spring | 3,354.6 (`med`) | speckled_trout | 39-53 | 20:00 | 53 | wind 1.00, light 0.97, stage 0.97 | 1.00 | 1.00 | | |
| 2026-08-28 | spring | 3,354.6 (`med`) | southern_flounder | 60-69 | 02:00 | 69 | wind 0.99, stage 0.93, season 0.90 | 1.00 | 1.00 | | |
| 2026-09-01 | mean (`mean_low`/`mean_med` blend) | 3,354.6 (`med`) | redfish | 61-72 | 11:00 | 72 | season 1.00, stage 0.99, solunar 0.96 | 1.00 | 1.00 | | |
| 2026-09-01 | mean | 3,354.6 (`med`) | speckled_trout | 44-50 | 06:00 | 50 | wind 0.95, season 0.95, light 0.83 | 1.00 | 1.00 | | |
| 2026-09-01 | mean | 3,354.6 (`med`) | southern_flounder | 57-70 | 04:00 | 70 | season 1.00, stage 0.96, wind 0.95 | 1.00 | 1.00 | | |
| 2026-09-04 | neap (`neap_low`/`neap_med` blend) | 3,354.6 (`med`) | redfish | 60-72 | 14:00 | 72 | season 1.00, stage 1.00, pressure 0.99 | 1.00 | 1.00 | | |
| 2026-09-04 | neap | 3,354.6 (`med`) | speckled_trout | 41-52 | 07:00 | 52 | light 0.98, season 0.95, wind 0.91 | 1.00 | 1.00 | | |
| 2026-09-04 | neap | 3,354.6 (`med`) | southern_flounder | 56-70 | 07:00 | 70 | season 1.00, light 0.99, stage 0.94 | 1.00 | 1.00 | | |

`flow.clamped` was `False` on all three days (a genuine two-regime discharge
blend each time, not a pinned edge); `salinity.extrapolated` was `False` on
all three (3,354.6 cfs sits well inside the 1,232-22,996 cfs calibrated
span). Neither disclosure flag is exercised by these particular three days
-- `backend/tests/test_payload.py::test_payload_flags_a_clamped_discharge_
blend` and `::test_payload_flags_an_extrapolated_salinity` cover that path
with a synthetic freshet day instead, since (per the discharge caveat above)
no date reachable today can produce a real one.

## Ellis's homework

Pick three days you remember well -- ideally one excellent, one poor, one
middling -- and fill in `actual` (what you caught, how the bite felt,
whether it matched the hour the model called out as best) and `notes` (tide
stage, wind, anything the model didn't have -- e.g. water clarity, bait
present). They do not need to be these exact three dates; these three exist
to prove the pipeline runs end to end and to show the tidal-range regime
blend responding correctly to a real spring/mean/neap spread. Once you have
three real days with `actual` filled in, that is the point to start looking
at whether any curve in `fisheries/species_weights.yaml` should move -- and
even then, three days is a first look, not a fit.
