"""Payload shape. The frontend contract lives here."""

import json


def test_payload_has_24_hours_for_every_species(synthetic_day):
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert set(p["species"]) == {"redfish", "speckled_trout", "southern_flounder"}
    for name, rows in p["species"].items():
        assert len(rows["hours"]) == 24, name


def test_every_hour_carries_its_sub_scores_and_reasons(synthetic_day):
    """Spec section 8: 'why is 3 PM an 82' always has a visible answer."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    hour = p["species"]["redfish"]["hours"][15]
    assert 0 <= hour["score"] <= 100
    assert len(hour["subs"]) == 9
    assert all(s["reason"] for s in hour["subs"])


def test_payload_is_json_serialisable(synthetic_day):
    """NaN is not valid JSON and numpy floats are not serialisable -- both are
    easy to leak from the scoring path."""
    from tidescout.pipeline.payload import build_payload

    text = json.dumps(build_payload(**synthetic_day), allow_nan=False)
    assert "NaN" not in text


def test_payload_records_missing_inputs_at_the_top_level(synthetic_day):
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert isinstance(p["missing"], list)
    assert "freshness" in p


def test_every_hour_carries_its_provenance_pair(synthetic_day):
    """The payload is the frontend contract, so the disclosure has to reach it.
    `confidence` and `constrained_share` answer different questions and BOTH
    must be present -- an hour on full data with an uncalibrated salinity reads
    1.0 and something lower, and a UI given only the first cannot tell."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    for name, rows in p["species"].items():
        for h in rows["hours"]:
            assert "confidence" in h and "constrained_share" in h, name
            assert isinstance(h["provisional"], list)
            assert 0.0 <= h["constrained_share"] <= 1.0


def test_an_uncalibrated_salinity_reaches_the_payload_as_provisional(synthetic_day):
    """Winyah Bay ships with `fitted: false`, so this is the live path, not an
    edge case. If the payload ever reports an empty `provisional` list for
    every hour while the fishery is unfitted, the disclosure has been lost
    somewhere between the factor and the JSON."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    flagged = [h for rows in p["species"].values() for h in rows["hours"]
               if "salinity" in h["provisional"]]
    assert flagged, "an unfitted salinity model must surface on some hour"
    assert all(h["constrained_share"] < 1.0 for h in flagged)


def test_an_out_of_domain_gauge_is_never_labelled_measured(synthetic_day_out_of_domain_gauge):
    """2026-08-26 review, Important 1: station 021108125 is a real,
    live-reporting USGS sensor -- but it sits 9,498 m outside the model
    domain and snaps, along with a second station 1,362 m outside it, to the
    along-estuary distance field's own extreme fresh end, so neither
    station's number describes the reach the scoring layer actually reads.
    Before the fix, `water.source != "climatology"` was the ONLY gate
    `_bay_salinity_reading` applied, so this station's 0.0 ppt reached
    every hour as `SalinityProvenance.MEASURED` -- `constrained`
    unconditionally True, no caveat, on the one factor this project has
    spent five PRs learning to disclose. This is exactly the hourly half of
    the payload a person actually sees (`tidescout score`'s table), unlike
    the per-feature half, which never took this sensor-vs-model path at all.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_out_of_domain_gauge)

    assert p["salinity"]["series"], "no bay-wide salinity reading resolved at all"
    for row in p["salinity"]["series"]:
        assert row["provenance"] != "measured", row

    flagged = [
        h for rows in p["species"].values() for h in rows["hours"]
        if "salinity" in h["provisional"]
    ]
    assert flagged, "an out-of-domain gauge must not silently pass as constrained salinity"
    assert all(h["constrained_share"] < 1.0 for h in flagged)


def test_an_in_domain_gauge_is_labelled_measured(synthetic_day_in_domain_gauge):
    """2026-08-26 re-review: Important 1's fix had no test on its POSITIVE
    half. Every existing provenance test in this codebase carries both
    halves (`test_a_measured_salinity_carries_no_caveat` sits beside
    `test_an_uncalibrated_salinity_reaches_the_payload_as_provisional`;
    `test_constrained_share_is_one_when_nothing_is_provisional` sits beside
    the one that proves it moves) -- this one was the exception, and three
    over-correction mutations (flip the non-USGS default, ignore
    `w.in_domain` and always exclude a declared station, or make
    `_measured_salinity_in_domain` return `False` unconditionally) all
    passed the full suite without it. Station 02136371 (Sampit River) is a
    real Winyah Bay sensor, declared with no `in_domain: false` override,
    so a genuine reading from it must still reach the payload confidently:
    `MEASURED`, no `~` and no caveat in the reason, `constrained_share`
    exactly 1.00 (nothing else on this fixture's full-data day is
    provisional or missing).
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_in_domain_gauge)

    assert p["salinity"]["series"], "no bay-wide salinity reading resolved at all"
    for row in p["salinity"]["series"]:
        assert row["provenance"] == "measured", row

    for rows in p["species"].values():
        for h in rows["hours"]:
            assert "salinity" not in h["provisional"], h
            assert h["constrained_share"] == 1.0, h
            sal = next(s for s in h["subs"] if s["factor"] == "salinity")
            assert sal["provisional"] is False, sal
            assert "~" not in sal["reason"] and "UNCALIBRATED" not in sal["reason"], sal


def test_a_non_usgs_salinity_source_still_defaults_to_measured(
    synthetic_day_non_usgs_salinity_source,
):
    """2026-08-26 re-review: `_measured_salinity_in_domain` has TWO
    default-True branches -- one for a `source` that names no declared USGS
    sensor, one for a `source` that is not USGS-shaped at all
    (`not source.startswith("usgs:")`). `test_an_in_domain_gauge_is_
    labelled_measured`'s `"usgs:02136371"` source cannot exercise this
    second branch -- it starts with `"usgs:"` and IS declared, so a mutation
    that flips ONLY the non-USGS default would leave that test green. This
    fixture's `"coops:8661070"` source is what closes that gap.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_non_usgs_salinity_source)

    assert p["salinity"]["series"], "no bay-wide salinity reading resolved at all"
    for row in p["salinity"]["series"]:
        assert row["provenance"] == "measured", row


def test_payload_flags_an_extrapolated_salinity(synthetic_day_freshet):
    """Spec section 10: degraded inputs surface, they do not hide."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_freshet)
    assert p["salinity"]["extrapolated"] is True


def test_salinity_representative_ppt_is_labelled_with_its_own_hour(synthetic_day):
    """`_bay_salinity_reading` runs every hour and the tidal shift term
    genuinely moves it across the day on the modelled path -- an earlier
    version of this module kept only the LAST hour's reading, silently
    overwriting the other 23 and publishing it as `representative_ppt` with
    no hour attached (2026-08-26 review, Important 7: measured 13.7 ->
    35.3 ppt across a real day, reported as 23.5 -- hour 23's value -- with
    nothing saying so). The full `series` must carry all 24, and
    `representative_ppt` must be labelled with the SPECIFIC hour it came
    from, not whichever hour a loop happened to finish on."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    series = p["salinity"]["series"]
    assert len(series) == 24
    # The series must show real movement across the day (the tidal term is
    # not a constant) -- a degenerate fixture where every hour reads the
    # same ppt would pass the shape assertions above while proving nothing
    # about the last-write bug this test protects against.
    assert len({round(row["ppt"], 3) for row in series}) > 1

    mid = series[len(series) // 2]
    assert p["salinity"]["representative_ppt"] == mid["ppt"]
    assert p["salinity"]["representative_hour"] == mid["time"]
    # And it must NOT be the (bugged) last-write answer, except by the kind
    # of coincidence this synthetic day is built to avoid.
    assert p["salinity"]["representative_ppt"] != series[-1]["ppt"]


def test_payload_flags_a_clamped_discharge_blend(synthetic_day_freshet):
    """22,996 cfs is the exact top of the simulated discharge axis (every
    range bucket's `freshet` regime is rasterised at exactly this flow --
    see `synthetic_day_freshet`'s docstring for why this is no longer "3.7x
    the highest flow ever simulated"). The payload must still say a single
    simulated regime was pinned rather than presenting it as a genuine
    two-regime blend -- `payload.py`'s `clamped` is deliberately truer than
    `flow.blend_regimes`'s own boundary-inclusive flag here."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_freshet)
    assert p["flow"]["clamped"] is True


def test_payload_sanitises_a_real_nan_from_a_dead_source(synthetic_day_no_weather):
    """`weather` going dark makes `pressure` and `wind` genuinely missing
    (`SubScore.value = NaN`, straight out of `score_factors`'s own
    `_missing` helper) -- the one fixture in this file that hands
    `_json_safe` a real NaN, rather than a NaN appended after it already
    ran. 2026-08-26 review, Important 1: without this, deleting both
    branches of `_json_safe` left every other test in this file green,
    because none of their fixtures ever produced a missing factor."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_no_weather)
    text = json.dumps(p, allow_nan=False)
    assert "NaN" not in text

    hour = p["species"]["redfish"]["hours"][12]
    by_factor = {s["factor"]: s for s in hour["subs"]}
    assert by_factor["pressure"]["missing"] is True
    assert by_factor["pressure"]["value"] is None
    assert by_factor["wind"]["missing"] is True
    assert by_factor["wind"]["value"] is None
    assert "pressure" in hour["excluded"]
    assert "wind" in hour["excluded"]


def test_the_feature_path_runs_against_the_real_flow_library(synthetic_day_with_flow):
    """Every other fixture in this file passes `cache=None`, which skips the
    flow-library phase entirely (see `payload._flow_events`) -- so
    `_blended_state`, `flowlib.load_state`, `activation.structure_fields`
    and `activation.sample_features` never ran under any of them
    (2026-08-26 review, Important 2). This is the one test that activates
    that path, against winyah-bay's REAL rasterised flow library, and
    checks the per-feature output it produces: a real in-domain feature,
    scored for all 24 hours, each hour carrying its own sub-score detail
    (Important 5) as well as the activation/provenance summary.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_with_flow)
    redfish_features = p["species"]["redfish"]["features"]
    assert redfish_features, "no in-domain feature was scored -- the flow-library path did not run"

    some_key = next(iter(redfish_features))
    feature = redfish_features[some_key]
    assert feature["type"]
    hours = feature["hours"]
    assert len(hours) == 24
    for h in hours:
        assert 0 <= h["activation"] <= 100
        assert 0.0 <= h["confidence"] <= 1.0
        assert 0.0 <= h["constrained_share"] <= 1.0
        assert isinstance(h["provisional"], list)
        assert isinstance(h["excluded"], list)
        assert h["subs"], "no sub-score detail on this feature-hour"
        assert all(s["reason"] for s in h["subs"])


def test_bay_flow_speed_ignores_dry_cells():
    """ANUGA writes a dry cell as u = v = 0, which the flow curve cannot tell
    from genuine slack water. `activation.structure_fields` already masks
    this; the bay-wide hourly speed did not, so every dry marsh cell voted
    "slack" in the average that feeds the flow factor.

    Measured on the real winyah-bay/mean_med library: 16.8-18.9% of cells are
    dry across phases, and phase 0 reads 0.1127 m/s unmasked against 0.1355
    wet-only -- a 17% understatement, worth about 0.09 of the redfish flow
    sub-score and enough to cross the 0.1 m/s "slack" label boundary.

    The synthetic state below is that situation in miniature: half the cells
    genuinely moving, half bone dry. The wet answer is 0.3; the unmasked
    average is 0.15, which the redfish curve would call "moving" instead of
    the truth.
    """
    import numpy as np

    from tidescout.pipeline.payload import _bay_flow_speed

    state = {
        "u": np.array([0.3, 0.3, 0.0, 0.0]),
        "v": np.array([0.0, 0.0, 0.0, 0.0]),
        "depth": np.array([2.0, 2.0, 0.0, 0.0]),
    }
    assert _bay_flow_speed(state) == 0.3
    # An all-dry state has no representative speed at all -- None, not 0.0,
    # so the factor goes MISSING and renormalises rather than scoring a
    # confident "dead slack" the model never observed.
    assert _bay_flow_speed({k: v * 0 for k, v in state.items()}) is None


def test_a_measured_in_domain_sensor_survives_a_missing_distance_field():
    """`_bay_salinity_reading`'s MEASURED branch returns before it ever
    touches the distance field -- only the MODELLED branch needs one, for
    `np.nanmedian`. `build_payload` used to gate the whole call on the field
    being present, so a missing `estuary_km.npy` dropped the salinity factor
    for all 24 hours x every species even with a live in-domain gauge
    reporting.
    """
    from types import SimpleNamespace

    from tidescout.config import load_fishery
    from tidescout.engine.score import SalinityProvenance
    from tidescout.pipeline.payload import _bay_salinity_reading

    fishery = load_fishery("winyah-bay")
    # A source naming no declared USGS sensor is in-domain by default (see
    # `_measured_salinity_in_domain`), so this exercises the MEASURED branch
    # unconditionally rather than depending on which of Winyah's stations
    # happens to be flagged in_domain today -- both of its salinity-capable
    # USGS gauges are currently `in_domain: false`.
    day = SimpleNamespace(
        water=SimpleNamespace(salinity_ppt=18.0, source="nerrs:northinlet",
                              salinity_source="nerrs:northinlet"),
        discharge=SimpleNamespace(cfs_now=5000.0),
    )
    reading = _bay_salinity_reading(day, fishery, None, 0.25)
    assert reading is not None, "a measured in-domain reading needs no distance field"
    assert reading.provenance is SalinityProvenance.MEASURED
    assert reading.ppt == 18.0
    # Without a distance field the MODELLED branch must decline rather than
    # crash on np.nanmedian(None).
    modelled_only = SimpleNamespace(
        water=None, discharge=SimpleNamespace(cfs_now=5000.0)
    )
    assert _bay_salinity_reading(modelled_only, fishery, None, 0.25) is None


def test_a_climatology_salinity_is_never_published_as_measured(
    synthetic_day_climatology_salinity_in_domain_temp,
):
    """2026-09-02 review, Finding 5. `WaterSummary.source` names whichever
    sensor supplied TEMPERATURE; salinity is resolved by a separate
    per-parameter fallback and can be monthly climatology while `source`
    still names a real, in-domain gauge.

    Gated on `source`, the payload published that climatology guess as
    `MEASURED` with `provisional=False` and `constrained_share` 1.00 -- no
    `~`, no caveat, indistinguishable from a live sensor. That is reachable
    on the shipped Winyah config (see the fixture), where the only in-domain
    USGS gauge reports temperature and no salinity at all.

    The gate now reads `salinity_source`, so this falls through to the
    MODELLED path and picks up the existing `fitted`/`extrapolated`
    disclosure machinery instead.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_climatology_salinity_in_domain_temp)

    for row in p["salinity"]["series"]:
        assert row["provenance"] != "measured", row

    # Winyah's salinity model is unfitted, so the honest label downstream is
    # provisional -- the point is that SOMETHING discloses, not silence.
    saw_hour = False
    for rows in p["species"].values():
        for h in rows["hours"]:
            saw_hour = True
            assert h["constrained_share"] < 1.0, h
            sal = next(s for s in h["subs"] if s["factor"] == "salinity")
            assert sal["provisional"] is True, sal
    assert saw_hour, "fixture produced no hours to check"
