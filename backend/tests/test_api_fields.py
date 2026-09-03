"""Payload fields the frontend needs that scoring alone does not produce."""

CONDITION_KEYS = {
    "time", "air_temp_f", "wind_speed_kn", "wind_dir_deg", "wind_gust_kn",
    "pressure_mb", "pressure_trend_mb_3h", "cloud_cover_pct", "precip_in",
    "tide_height_ft", "tide_phase", "tide_frac",
}


def test_conditions_are_top_level_not_duplicated_per_species(synthetic_day):
    """Spec §5. These are FISHERY-WIDE hour facts -- they depend on the hour,
    never on which species is being scored. Putting them under `species[name]`
    would repeat exactly the modelling error PR #11 corrected, where seven
    hour-scope factors were shipped on every feature-hour and cost 21.1 MB to
    assert something false.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert "conditions" in p, "the tide curve and right rail have no other source"
    assert len(p["conditions"]) == 24
    for blob in p["species"].values():
        assert "conditions" not in blob


def test_every_hour_carries_the_keys_the_rail_and_tide_curve_need(synthetic_day):
    """`_hour_to_dict` emits only score/subs/confidence -- the raw values
    survive nowhere else. You cannot draw a tide curve from the string
    "pressure +0.7 mb/3h -- steady".

    Asserts the KEYS are present AND that tide_height_ft is a real number
    somewhere in the day: a block of 24 dicts full of nulls would satisfy a
    keys-only check while leaving the curve unplottable.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    for row in p["conditions"]:
        assert CONDITION_KEYS <= set(row), sorted(CONDITION_KEYS - set(row))
    heights = [r["tide_height_ft"] for r in p["conditions"]]
    assert any(h is not None for h in heights), "no plottable tide data"


def test_conditions_are_positionally_aligned_with_the_species_hours(synthetic_day):
    """The strip draws bar i and tide point i from the same index. If these
    axes ever diverge, every bar is labelled with the wrong hour's weather and
    nothing raises."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    for blob in p["species"].values():
        assert len(blob["hours"]) == len(p["conditions"])
        for i, hour in enumerate(blob["hours"]):
            assert hour["time"] == p["conditions"][i]["time"]
