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


def test_payload_flags_a_clamped_discharge_blend(synthetic_day_freshet):
    """22,996 cfs is 3.7x the highest flow ever simulated. The payload must say
    the flow state was clamped rather than presenting it as a lookup."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_freshet)
    assert p["flow"]["clamped"] is True
