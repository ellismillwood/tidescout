# Hindcast log: predicted vs actual

**No row in this log is usable for tuning any curve until the `actual`
column is filled in by Ellis.** Every number under "predicted" below came
from `tidescout score`, i.e. from the engine as shipped in this branch --
it is a forecast, not ground truth. The ONLY ground truth this project has
is what actually happened on days Ellis fished, and no agent has that.
Filling `actual` with plausible-looking guesses would mean every future
weight is tuned against fiction while the tuning looks rigorous -- do not
do this, ever, to this file.

**This is the FIFTH generation of this log (2026-09-02, whole-branch
review of PR #10).** That review found nine defects, all nine verified
against the code before any fix was written. Two of them move the numbers
in this table, so every row below was regenerated against the corrected
engine. What moved, and why:

- **Bay flow speed was averaging DRY cells (Finding 3).** ANUGA writes a
  dry cell as u = v = 0, indistinguishable from genuine slack water, and
  the bay-wide hourly mean included all of them -- 16.8-18.9% of the
  domain depending on phase. Every row in the fourth generation read
  `flow 0.10 m/s` or `0.14 m/s`; the same hours now read 0.12-0.13 and
  0.17. That is the single largest change to this table.
- **Solunar distance was measured to a period's leading EDGE, not its
  centre (Finding 2).** A major is the moon's transit +/- 1 h, so the
  factor peaked a full hour before the transit and called the transit
  itself "60 min from a solunar period". 2026-07-21 / redfish moves from
  "180 min" to "169 min".

- **Exactly ONE peak hour moved:** southern_flounder / 2026-07-28, from
  13:00 to 01:00. The other eight rows kept the hour they had. That single
  shift is the combined effect of the two fixes above on a day whose top
  hours were already within a few points of each other -- not a separately
  diagnosed change, and not a strong claim about that date.
- **A climatology salinity could ship as MEASURED (Finding 5).** Not
  triggered on these three dates -- all nine rows resolve salinity through
  the spatial model, as the section below describes -- but the gate that
  let it happen is closed, so `provenance` on a future regeneration means
  what it says.
- Findings 1, 4, 6, 8 (signed ebb current, DST phase arithmetic, the flat
  gate's phase convention, and the library's duplicate closing snapshot)
  do not move these particular rows: none of these dates crosses a DST
  boundary, the flow library resolved for every hour so the CO-OPS
  fallback never ran, and the last two affect per-FEATURE activation
  rather than the fishery-wide hourly score tabulated here.

The `actual` column is still empty, and still Ellis's to fill.

**The fourth generation (2026-08-26, whole-branch review) said:**
The first had two problems: its three dates were in the future
(Ellis cannot report `actual` for a day that has not happened), and its
discharge was day-blind (every row read today's live gauge regardless of
the date named). The second fixed both, on real past dates with genuinely
different discharge. The third closed the same day-blindness in water
temperature and the bay-wide salinity fallback. **This regeneration fixes
two further defects the review found, both of which change every number
below:**

- **Important 1 (out-of-domain gauge mislabelled MEASURED).** Station
  021108125 -- the gauge behind every "salinity 0.0 ppt" reading in the
  previous generation -- sits 9,498 m outside the model domain (see
  `fisheries/winyah-bay.yaml`'s water-station comments), snapped to the
  along-estuary distance field's own extreme fresh end. `payload.
  _bay_salinity_reading` labelled it `MEASURED` anyway, which made
  `constrained_share = 1.00` on every row below regardless of species --
  the exact case the "Read `constrained_share`" section used to (wrongly)
  say could not happen from a live-reporting gauge. `WaterSensor.in_domain`
  (now `false` for this station) makes the payload fall through to the
  spatial MODELLED estimate instead, carrying the existing `fitted`/
  `extrapolated` disclosure machinery. Salinity numbers below are
  therefore genuinely different from the third generation, not just
  relabelled -- see "Salinity is now the model, not the gauge" below.
- **Important 3 (wrong limiting factors on one row).** The previous
  generation's southern_flounder / 2026-07-28 row printed "salinity 0.20;
  light 0.58" as the two lowest sub-scores at the 13:00 peak. The true
  three lowest, confirmed by direct measurement against that row's own
  scored hour, are salinity 0.20, **water_temp 0.47**, light 0.58 -- index
  1 of the ascending-sorted list (water_temp, weight 1.0 for flounder) was
  skipped in favour of index 2 (light, weight 0.5), an off-by-one in
  whatever selected that row's factors. The other eight rows were
  independently confirmed correct at generation time. Rather than patch
  one row inside a table whose selection rule already varied between "2"
  and "3" factors per row with no stated criterion for which, this
  regeneration adopts one fixed, unambiguous rule for every row: **always
  the three lowest-value sub-scores, ascending, ties broken by list
  order** -- a rule an off-by-one has no room to hide in.

Every input this log's three dates depend on -- weather, tides, currents,
sun, moon, solunar, discharge, water temperature, and salinity (now
correctly resolved to the spatial model, not a mislabelled gauge) -- reads
that SPECIFIC date, not "now".

## Read `constrained_share` before trusting a row

`constrained_share < 1.0` on a row means the salinity factor THAT hour
rests on `engine.salinity`'s uncalibrated model (`fitted: false` in
`fisheries/winyah-bay.yaml` -- Winyah's live, permanent state) rather than
an observation. **All nine rows below now read `constrained_share < 1.0`**
-- unlike the previous generation, where all nine read exactly `1.00`
because station 021108125's out-of-domain reading was wrongly accepted as
`MEASURED` (see Important 1 above). With that fixed, no station Winyah Bay
currently declares can make `constrained_share` read 1.00 on a row
generated the way these nine were: every declared USGS salinity sensor is
either climatology-graded or now correctly excluded as out-of-domain, so
the bay-wide salinity reading falls through to the uncalibrated spatial
model, and `fitted: false` keeps it provisional. **This is not a
structural guarantee, and this file already documents the gap that breaks
it 60 lines below ("Every input below is now date-faithful"):**
`_measured_salinity_in_domain` gates on `WaterSummary.source`, which names
whichever sensor supplied TEMPERATURE, not necessarily the one that
supplied SALINITY. If both of Winyah's salinity-capable USGS stations ever
went dark on TEMPERATURE specifically while the in-domain, temperature-only
02136371 kept reporting, `source` would read `"usgs:02136371"` -- correctly
`in_domain` -- while `salinity_ppt` itself had silently fallen to
climatology, and the reading would be labelled `MEASURED` on a guess.
Closing that fully needs `WaterSummary` to carry the salinity station's
own identity, not just the temperature one's (see `_measured_salinity_
in_domain`'s docstring). `constrained_share` differs slightly BY SPECIES,
not by date, because it is a function of each species' own relative factor
weights (`(total_weight - salinity_weight) / total_weight`) and salinity
is the only factor provisional on any of these rows: 0.92 for redfish
(salinity weight 0.5 of 6.3), 0.87 for speckled trout (0.9 of **7.1** --
trout weights salinity heaviest of the three, so losing it to
"provisional" costs `constrained_share` the most, NOT `confidence`, which
reads 1.00 on all nine rows regardless -- `confidence` and
`constrained_share` answer different questions and this pair of columns is
the entire reason both are carried), 0.91 for southern flounder (0.6 of
6.4). `test_an_uncalibrated_salinity_reaches_the_payload_as_provisional`
(`backend/tests/test_payload.py`) exercises the negative path with a
synthetic day; `test_an_out_of_domain_gauge_is_never_labelled_measured`
(same file) proves it against a REAL declared Winyah Bay station rather
than only a synthetic one; `test_an_in_domain_gauge_is_labelled_measured`
and `test_a_non_usgs_salinity_source_still_defaults_to_measured` prove the
POSITIVE half -- a genuinely in-domain or unrecognised-source reading is
still `MEASURED` -- which an earlier version of this fix had no test for.
The per-FEATURE numbers behind "top-ranked features" are unaffected
either way and always were: `score_feature` always uses the spatial model
at the feature's own along-estuary distance, never the bay-wide sensor
value, so every map marker's activation rested on the unfitted model
regardless of what the gauge said, in every generation of this log.

**`constrained_share` (and `confidence`) are a SNAPSHOT of this run, not a
property of the date or species.** Both are computed from whichever
factors actually resolved that hour, and `combine` renormalises over
whatever survives -- so a source going dark between one regeneration and
the next moves the number even for the identical date and species.
Measured directly: one run of this exact table read southern_flounder /
2026-07-28's `constrained_share` at 0.9063 with `missing: []` (every
source live); a second run, with `missing: ['weather']` (pressure and wind
both excluded that day), read 0.8868 for the SAME species and date --
`(5.3 - 0.6) / 5.3` instead of `(6.4 - 0.6) / 6.4`, because losing two more
factors to `missing` shrinks the denominator `constrained_share` divides
by. Both numbers are correct for the run that produced them; neither is
"the" constrained_share for southern_flounder on 2026-07-28. This is spec
section 10 working as designed -- a source going dark is supposed to move
the disclosure, not hide behind a number that looks fixed -- and it is the
best evidence on this branch that the renormalisation machinery is real
end-to-end rather than only exercised by a fixture. All nine rows in the
table below were regenerated together in one `build_payload` pass per
date, each with `missing: []`; a future regeneration during a source
outage will legitimately read different `constrained_share` values on the
same dates without either version being wrong.

## Salinity is now the model, not the gauge

The previous generation's "salinity 0.0 ppt -- near-fresh" on every row
was a real number from a real, live-reporting station -- it was simply the
WRONG station's number for the reach being scored (see Important 1 above).
With the gauge correctly excluded, the bay-wide salinity reading below
comes from `engine.salinity`'s spatial model evaluated at the domain's
median along-estuary distance and that hour's tidal phase, and it now
MOVES with discharge and tide the way the model actually predicts, rather
than sitting fixed at the gauge's repeated 0.0 ppt. It is still
UNCALIBRATED (`fitted: false`), so every salinity number below carries
that caveat in its `reason` and counts toward `provisional`/
`constrained_share` exactly as any other unfitted model estimate does --
the number changed, the honesty did not.

## Every input below is now date-faithful

**Fixed 2026-08-26 (second regeneration):** `sources.usgs.discharge_summary`
used to read only the LIVE gauge (`datetime.now(UTC)`, no `day` parameter)
regardless of which date `dayloader.load_day` was assembling. It now takes
an optional `day`; a date strictly before today reads THAT day's own USGS
daily mean (`fetch_daily`, the NWIS `dv` service already used by
`pipeline.salinity_fit` for calibration) instead of the live instantaneous
feed.

**Fixed 2026-08-26 (third regeneration):** `sources.usgs.water_summary`
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
87.44 and 86.54 underneath). Water temperature is unaffected by this
generation's two fixes (Important 1 is salinity-only; Important 3 is
selection-only) and these three readings are unchanged from the previous
generation.

`source` stays exactly as honest, and exactly as limited, as it always
was: it names whichever sensor supplied TEMPERATURE (`usgs:021108125` on
all three rows below), never necessarily the one that supplied SALINITY.
That gap is real (a fishery whose temperature- and salinity-reporting
sensors disagree could still slip a climatology salinity value through
labelled by a temperature station's name) but is unreachable on Winyah's
current sensor config -- station 021108125 reports both parameters, tried
first -- and closing it fully is a `WaterSensor`/`WaterSummary` shape
change, not a `day`-threading one. It is now ALSO the station Important 1
excludes from `MEASURED` salinity (`in_domain: false`) -- that exclusion is
about salinity only; this station's TEMPERATURE reading is unaffected and
still names `source` correctly.

## How these were produced

```
tidescout score winyah-bay 2026-07-21   # low discharge, neap tide
tidescout score winyah-bay 2026-08-05   # med discharge, neap tide
tidescout score winyah-bay 2026-07-28   # high discharge, spring tide
```

All three are real past dates (regenerated 2026-09-02; they were already
past when this log was first written on 2026-08-26). Each command scores
all three species from one `build_payload` call; the table below pulls the
day's score range, its single highest hour, the THREE LOWEST-VALUE
sub-scores that hour (under `combine`'s weighted geometric mean, a low
value -- not a high one -- is what actually constrains the score; always
exactly three, not "two or three" -- see Important 3 above for why the
previous generation's variable count is what let one row go wrong), and
that hour's own `water_temp` reason string.

## Predictions

| date | tide regime | discharge (cfs) | bucket | water temp | species | score range (day) | peak hour | peak score | limiting factors at peak (value — reason) | confidence | constrained_share | actual | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-21 | neap | 2,317.94 | low (clamped to `neap_low`) | 87°F (87.44°F) | redfish | 60-74 | 03:00 | 74 | flow 0.53 — "flow 0.13 m/s — moving"; solunar 0.56 — "169 min from a solunar period"; light 0.63 — "2.2 h from twilight, 96% cloud widened it from 3.4 h" | 1.00 | 0.92 |  |  |
| 2026-07-21 | neap | 2,317.94 | low | 87°F (87.44°F) | speckled_trout | 55-69 | 07:00 | 69 | water_temp 0.46 — "water 87F"; flow 0.55 — "flow 0.12 m/s — moving"; pressure 0.58 — "pressure +0.7 mb/3h — steady" | 1.00 | 0.87 |  |  |
| 2026-07-21 | neap | 2,317.94 | low | 87°F (87.44°F) | southern_flounder | 61-77 | 07:00 | 77 | water_temp 0.42 — "water 87F"; pressure 0.72 — "pressure +0.7 mb/3h — steady"; flow 0.72 — "flow 0.12 m/s — moving" | 1.00 | 0.91 |  |  |
| 2026-08-05 | neap | 4,037.80 | med (`neap_low`/`neap_med` blend) | 83°F (82.76°F) | redfish | 62-77 | 19:00 | 77 | flow 0.53 — "flow 0.13 m/s — moving"; stage 0.59 — "tide 0.83 of cycle — ebbing"; light 0.85 — "1.0 h from twilight, 47% cloud widened it from 1.2 h" | 1.00 | 0.92 |  |  |
| 2026-08-05 | neap | 4,037.80 | med | 83°F (82.76°F) | speckled_trout | 60-77 | 19:00 | 77 | flow 0.58 — "flow 0.13 m/s — moving"; water_temp 0.69 — "water 83F"; stage 0.72 — "tide 0.83 of cycle — ebbing" | 1.00 | 0.87 |  |  |
| 2026-08-05 | neap | 4,037.80 | med | 83°F (82.76°F) | southern_flounder | 66-85 | 19:00 | 85 | water_temp 0.66 — "water 83F"; flow 0.76 — "flow 0.13 m/s — moving"; pressure 0.88 — "pressure -0.9 mb/3h — falling — pre-frontal feeding window" | 1.00 | 0.91 |  |  |
| 2026-07-28 | spring | 8,853.80 | high (`spring_high`/`spring_freshet` blend) | 87°F (86.54°F) | redfish | 64-82 | 20:00 | 82 | flow 0.63 — "flow 0.17 m/s — moving"; salinity 0.74 — "salinity ~34.9 ppt — salty (UNCALIBRATED model estimate, no observation constrains it)"; water_temp 0.74 — "water 87F" | 1.00 | 0.92 |  |  |
| 2026-07-28 | spring | 8,853.80 | high | 87°F (86.54°F) | speckled_trout | 51-76 | 20:00 | 76 | water_temp 0.51 — "water 87F"; salinity 0.66 — "salinity ~34.9 ppt — salty (UNCALIBRATED model estimate, no observation constrains it)"; flow 0.70 — "flow 0.17 m/s — moving" | 1.00 | 0.87 |  |  |
| 2026-07-28 | spring | 8,853.80 | high | 87°F (86.54°F) | southern_flounder | 65-78 | 01:00 | 78 | water_temp 0.47 — "water 87F"; light 0.58 — "4.8 h from twilight, 33% cloud widened it from 5.5 h"; pressure 0.83 — "pressure -0.3 mb/3h — steady" | 1.00 | 0.91 |  |  |

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

Salinity no longer reads "near-fresh, 0.0 ppt" on every row (see "Salinity
is now the model, not the gauge" above) -- it only surfaces among a row's
three lowest sub-scores on 2026-07-28, the highest-discharge date, where
the model's tidal/discharge state pushes it to 0.74/0.66 ("salty") for
redfish and speckled trout. On the other two, lower-discharge dates the
modelled value scores well enough that some OTHER factor is always among
the three lowest instead -- salinity is still computed, still `provisional`,
and still counted at full weight every hour (`constrained_share` above
proves that), it simply is not always among the worst three at the single
peak hour this table reports.

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
