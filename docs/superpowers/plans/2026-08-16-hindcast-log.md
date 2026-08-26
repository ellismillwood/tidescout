# Hindcast log: predicted vs actual

**No row in this log is usable for tuning any curve until the `actual`
column is filled in by Ellis.** Every number under "predicted" below came
from `tidescout score`, i.e. from the engine as shipped in this branch --
it is a forecast, not ground truth. The ONLY ground truth this project has
is what actually happened on days Ellis fished, and no agent has that.
Filling `actual` with plausible-looking guesses would mean every future
weight is tuned against fiction while the tuning looks rigorous -- do not
do this, ever, to this file.

**This is the third generation of this log (2026-08-26).** The first had
two problems: its three dates were in the future (Ellis cannot report
`actual` for a day that has not happened), and its discharge was
day-blind (every row read today's live gauge regardless of the date
named). The second fixed both, on real past dates with genuinely different
discharge -- but its water temperature and salinity were STILL day-blind,
which this generation fixes. Every input this log's three dates depend on
-- weather, tides, currents, sun, moon, solunar, discharge, water
temperature, and the bay-wide salinity fallback -- now reads that
SPECIFIC date, not "now".

## Read `constrained_share` before trusting a row

`constrained_share < 1.0` on a row means the salinity factor THAT hour
rests on `engine.salinity`'s uncalibrated model (`fitted: false` in
`fisheries/winyah-bay.yaml` -- Winyah's live, permanent state) rather than
an observation. All nine rows below read `constrained_share = 1.00`,
because each date's USGS gauge (021108125) reported a real salinity value
that day (all three happen to read 0.0 ppt -- a genuinely near-fresh tidal
reach on every one of these three dates) and `build_payload` prefers a
real sensor reading over the model whenever one exists. **A stale/dark
salinity gauge is NOT what would flip this** -- `usgs.water_summary` never
returns `salinity_ppt=None`; a gauge that stops reporting SALINITY
specifically falls back to a monthly climatology GUESS while `source`
stays labelled by whichever sensor (if any) still reports TEMPERATURE, and
`payload._bay_salinity_reading` treats that fallback as MEASURED too
unless `source` itself reads `"climatology"`. The actual trigger is
narrower: `water` failing entirely (the whole USGS water fetch errors out)
or EVERY configured sensor's TEMPERATURE series going dark at once (so
`source` itself falls to `"climatology"`). `test_an_uncalibrated_salinity_
reaches_the_payload_as_provisional` (`backend/tests/test_payload.py`)
exercises that path directly with a synthetic day, since none of these
nine rows do. The per-FEATURE numbers behind "top-ranked features" are
unaffected either way: `score_feature` always uses the spatial model at
the feature's own along-estuary distance, never the bay-wide sensor value,
so every map marker's activation rests on the unfitted model regardless of
what the gauge says.

## Every input below is now date-faithful

**Fixed 2026-08-26 (second regeneration):** `sources.usgs.discharge_summary`
used to read only the LIVE gauge (`datetime.now(UTC)`, no `day` parameter)
regardless of which date `dayloader.load_day` was assembling. It now takes
an optional `day`; a date strictly before today reads THAT day's own USGS
daily mean (`fetch_daily`, the NWIS `dv` service already used by
`pipeline.salinity_fit` for calibration) instead of the live instantaneous
feed.

**Fixed 2026-08-26 (this, third regeneration):** `sources.usgs.water_summary`
had the identical bug -- water temperature (weight 1.0 for speckled trout
and southern flounder, joint-highest of the nine factors, 0.8 for redfish)
and the bay-wide salinity fallback both always read the live 7-day window.
It now takes the same optional `day` and, split the same way into
`_live_water_summary`/`_historical_water_summary`, reads that date's own
daily means for both `PARAM_TEMP_C` and `PARAM_SALINITY` when `day` is in
the past. Confirmed working on real data: 2026-07-21 read 87.44°F,
2026-08-05 read 82.76°F, 2026-07-28 read 86.54°F -- three different real
historical readings from the SAME station (021108125), not one live value
repeated three times (the earlier all-live version, and the un-rounded
`temp_f` the two-decimal precision above is pulled from -- the printed
`reason` string itself rounds to the nearest whole degree, which is why
07-21 and 07-28 both display as "87F" in the table below despite being
87.44 and 86.54 underneath).

`source` stays exactly as honest, and exactly as limited, as it always
was: it names whichever sensor supplied TEMPERATURE (`usgs:021108125` on
all three rows below), never necessarily the one that supplied SALINITY.
That gap is real (a fishery whose temperature- and salinity-reporting
sensors disagree could still slip a climatology salinity value through
labelled by a temperature station's name) but is unreachable on Winyah's
current sensor config -- station 021108125 reports both parameters, tried
first -- and closing it fully is a `WaterSensor`/`WaterSummary` shape
change, not a `day`-threading one. Every input this log depends on is now
correctly date-specific; this one residual gap is about attribution
labelling on a path that, for Winyah today, always resolves the same
station for both parameters anyway.

## How these were produced

```
tidescout score winyah-bay 2026-07-21   # low discharge, neap tide
tidescout score winyah-bay 2026-08-05   # med discharge, neap tide
tidescout score winyah-bay 2026-07-28   # high discharge, spring tide
```

All three are real past dates (today is 2026-08-26). Each command scores
all three species from one `build_payload` call; the table below pulls the
day's score range, its single highest hour, the TWO OR THREE LOWEST-VALUE
sub-scores that hour (under `combine`'s weighted geometric mean, a low
value -- not a high one -- is what actually constrains the score), and
that hour's own `water_temp` reason string.

## Predictions

| date | tide regime | discharge (cfs) | bucket | water temp | species | score range (day) | peak hour | peak score | limiting factors at peak (value — reason) | confidence | constrained_share | actual | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-21 | neap | 2,317.94 | low (clamped to `neap_low`) | 87°F (87.44°F) | redfish | 57-70 | 03:00 | 70 | salinity 0.45 — "salinity 0.0 ppt — near-fresh"; flow 0.46 — "flow 0.10 m/s — moving" | 1.00 | 1.00 | | |
| 2026-07-21 | neap | 2,317.94 | low | 87°F (87.44°F) | speckled_trout | 39-47 | 03:00 | 47 | salinity 0.05 — "salinity 0.0 ppt — near-fresh"; water_temp 0.46 — "water 87F" | 1.00 | 1.00 | | |
| 2026-07-21 | neap | 2,317.94 | low | 87°F (87.44°F) | southern_flounder | 52-65 | 07:00 | 65 | salinity 0.20 — "salinity 0.0 ppt — near-fresh"; water_temp 0.42 — "water 87F" | 1.00 | 1.00 | | |
| 2026-08-05 | neap | 4,037.80 | med (`neap_low`/`neap_med` blend) | 83°F (82.76°F) | redfish | 57-70 | 13:00 | 70 | light 0.43 — "4.9 h from twilight, 70% cloud widened it from 6.5 h"; flow 0.45 — "flow 0.10 m/s — slack" | 1.00 | 1.00 | | |
| 2026-08-05 | neap | 4,037.80 | med | 83°F (82.76°F) | speckled_trout | 41-52 | 19:00 | 52 | salinity 0.05 — "salinity 0.0 ppt — near-fresh"; flow 0.51 — "flow 0.10 m/s — moving"; water_temp 0.69 — "water 83F" | 1.00 | 1.00 | | |
| 2026-08-05 | neap | 4,037.80 | med | 83°F (82.76°F) | southern_flounder | 56-72 | 19:00 | 72 | salinity 0.20 — "salinity 0.0 ppt — near-fresh"; water_temp 0.66 — "water 83F" | 1.00 | 1.00 | | |
| 2026-07-28 | spring | 8,853.80 | high (`spring_high`/`spring_freshet` blend) | 87°F (86.54°F) | redfish | 61-77 | 20:00 | 77 | salinity 0.45 — "salinity 0.0 ppt — near-fresh"; flow 0.57 — "flow 0.14 m/s — moving"; water_temp 0.74 — "water 87F" | 1.00 | 1.00 | | |
| 2026-07-28 | spring | 8,853.80 | high | 87°F (86.54°F) | speckled_trout | 42-53 | 20:00 | 53 | salinity 0.05 — "salinity 0.0 ppt — near-fresh"; water_temp 0.51 — "water 87F" | 1.00 | 1.00 | | |
| 2026-07-28 | spring | 8,853.80 | high | 87°F (86.54°F) | southern_flounder | 57-67 | 13:00 | 67 | salinity 0.20 — "salinity 0.0 ppt — near-fresh"; light 0.58 — "4.6 h from twilight, 84% cloud widened it from 6.5 h" | 1.00 | 1.00 | | |

`flow.clamped` was `True` on 2026-07-21 (2,317.94 cfs sits genuinely below
`neap_low`'s own simulated flow, 2,774 cfs -- a real single-regime pin, not
just the boundary-inclusive-as-suspect case `payload.py` also flags) and
`False` on the other two (genuine two-regime blends: `neap_low`/`neap_med`
and `spring_high`/`spring_freshet`). `salinity.extrapolated` was `False` on
all three -- none of these real discharges reach 22,996 cfs, the top of the
1,232-22,996 cfs calibrated span (also no longer "3.7x the highest flow
ever simulated" -- every discharge bucket, including `freshet`, is now
rasterised at exactly that flow; see `conftest.py`'s `synthetic_day_freshet`
fixture docstring). Neither `test_payload_flags_a_clamped_discharge_blend`
nor `test_payload_flags_an_extrapolated_salinity` (`backend/tests/test_
payload.py`) depends on these dates; both use a synthetic freshet day
instead.

`salinity` reads "near-fresh, 0.0 ppt" on every row -- unlike water
temperature, this is NOT a residual live-only artifact: `water_summary`'s
historical path fetches `PARAM_SALINITY` daily means exactly the way it
fetches `PARAM_TEMP_C`, and station 021108125 genuinely reported ~0 ppt on
all three of these specific dates (mid-to-late summer, a tidal-freshwater
reach). It is a real, if repetitive, three-for-three coincidence in the
data, not a wiring gap.

## Ellis's homework

Pick three days you remember well -- ideally one excellent, one poor, one
middling -- and fill in `actual` (what you caught, how the bite felt,
whether it matched the hour the model called out as best) and `notes`
(tide stage, wind, anything the model didn't have -- water clarity, bait
present). They do not need to be these exact three dates; these three exist
to prove the pipeline runs end to end against real, genuinely different
discharge and water temperature, and to show the tidal-range regime blend
responding correctly to a real neap/spring spread. Once you have three real
days with `actual` filled in, that is the point to start looking at whether
any curve in `fisheries/species_weights.yaml` should move -- and even then,
three days is a first look, not a fit.
