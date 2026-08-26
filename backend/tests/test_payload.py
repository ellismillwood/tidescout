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
