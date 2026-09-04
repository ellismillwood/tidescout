"""Payload fields the frontend needs that scoring alone does not produce."""

import dataclasses
from datetime import date

# The WHOLE top-level key set of a day payload, not a floor. `frontend/
# fixtures/day-payload.json` is a frozen recording and `frontend/tests/
# contract.test.ts` asserts exact key sets against it -- so a field added to
# `build_payload` after that recording is invisible to the frontend suite and,
# with a subset check here, invisible to this one too. It would pass ruff, all
# 935 backend tests and all 108 frontend tests while the UI silently ignored
# it. Equality is what makes adding a field a deliberate edit to a Python
# constant, at which point RE-RECORDING the fixture is the next step (the
# frontend contract test fails against the old recording until it is).
PAYLOAD_KEYS = {
    "slug", "day", "model_label", "missing", "freshness", "sub_scope",
    "flow", "species", "conditions", "salinity", "water", "astro",
}

CONDITION_KEYS = {
    "time", "air_temp_f", "wind_speed_kn", "wind_dir_deg", "wind_gust_kn",
    "pressure_mb", "pressure_trend_mb_3h", "cloud_cover_pct", "precip_in",
    "tide_height_ft", "tide_phase", "tide_frac",
}

# Which `CONDITION_KEYS` fields the `synthetic_day` fixture (conftest.py's
# `_day_conditions`) actually populates with a real value on EVERY hour --
# verified empirically (`_day_conditions(4_200.0, "med")`, all 24 hours),
# not assumed. `wind_gust_kn` and `precip_in` are the only two fields the
# fixture never sets (no `HourlyConditions(...)` kwarg for either, so both
# default to `None`); asserting those non-null would be an assertion that
# only holds by accident, so they are deliberately excluded here.
REAL_EVERY_HOUR = CONDITION_KEYS - {"wind_gust_kn", "precip_in"}


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

    Asserts the KEYS are present AND that every field the fixture actually
    populates is a real number on EVERY hour, not merely one -- an `any()`
    check over `tide_height_ft` let 23 null rows plus one real value pass
    even though `_day_conditions` sets it on all 24 (2026-09-03 review,
    Important 2). `wind_gust_kn`/`precip_in` are excluded from this check
    (see `REAL_EVERY_HOUR`'s docstring): the fixture genuinely never sets
    them, so asserting them non-null would only pass by luck.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    for row in p["conditions"]:
        # EQUALITY, not `<=`. A subset check answers "does the UI have what it
        # needs" and says nothing about drift the other way -- a new key in
        # `_conditions_to_dict` passes it while the frozen frontend fixture
        # (and so the frontend's own exact-key contract test) never sees the
        # field at all.
        assert set(row) == CONDITION_KEYS, {
            "extra": sorted(set(row) - CONDITION_KEYS),
            "missing": sorted(CONDITION_KEYS - set(row)),
        }
    for key in REAL_EVERY_HOUR:
        values = [r[key] for r in p["conditions"]]
        assert all(v is not None for v in values), f"{key} has a null hour: {values}"


def test_the_payload_carries_exactly_the_top_level_keys_the_client_declares(
    synthetic_day,
):
    """The backend half of `frontend/tests/contract.test.ts`'s top-level
    assertion, which can only see the frozen fixture.

    Spec §2 claims "a new field in the payload fails that test rather than
    being silently ignored by the UI". That is only true if a check on THIS
    side sees the live `build_payload` output, because the frontend's copy is
    a recording: without this, adding a key to the payload is a silent,
    green-suite no-op for the UI. With it, the field cannot ship without
    someone editing `PAYLOAD_KEYS`, `frontend/src/api/types.ts`, that test's
    `TOP_LEVEL`, and re-recording `frontend/fixtures/day-payload.json`.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert set(p) == PAYLOAD_KEYS, {
        "extra": sorted(set(p) - PAYLOAD_KEYS),
        "missing": sorted(PAYLOAD_KEYS - set(p)),
    }


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


def test_water_block_carries_temperature_not_salinity(synthetic_day):
    """Spec §5 asks the rail for water temperature alongside air temperature.
    `payload["water"]` is a DAY fact (one `WaterSummary` per day, no per-hour
    reading), so it is a sibling of `conditions`, not a field inside it.

    Asserted as an exact dict, not just key presence: this also proves
    `salinity_ppt`/`source`/`salinity_source` did NOT leak in alongside
    `temp_f`/`temp_trend_f_3d` -- `payload["salinity"]` already owns that
    reading WITH its provenance, and a second, provenance-free copy here
    would give a reader two places to look for one fact.
    """
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert p["water"] == {"temp_f": 84.0, "temp_trend_f_3d": 0.4}


def test_water_block_is_null_when_the_day_has_no_water_summary(monkeypatch):
    """A fishery whose water sensor AND climatology fallback both produced
    nothing (`DayConditions.water is None` -- distinct from a `WaterSummary`
    whose fields happen to be `None`) must degrade `payload["water"]` to
    `null`, not raise. Builds its own dark day rather than reusing
    `synthetic_day`: that fixture's `water` is never `None`."""
    from tests.conftest import _day_conditions
    from tidescout.pipeline.payload import build_payload
    from tidescout.sources import dayloader

    dark = dataclasses.replace(_day_conditions(4_200.0, "med"), water=None)
    monkeypatch.setattr(dayloader, "load_day", lambda *a, **k: dark)
    p = build_payload(
        slug="winyah-bay", day=date(2026, 8, 16), model_label="gfs_seamless", cache=None
    )
    assert p["water"] is None


def test_astro_block_carries_the_days_sun_and_moon_times(synthetic_day):
    """Spec §5: "plus the day's sun and moon times" -- another DAY fact,
    computed once (`sources.astronomy`), never once per hour, so it is a
    sibling of `conditions` rather than folded into it.

    Compared against the fixture's OWN `SunTimes`/`MoonInfo` (built the same
    way `synthetic_day` builds them internally) rather than a hand-typed ISO
    string, so a timezone slip in this test and one in `_astro_to_dict`
    cannot silently cancel out. The distinctness check guards against an
    implementation that wrote one field's value into all four sun keys --
    the fixture's dawn/sunrise/sunset/dusk genuinely differ, so that bug
    would still be caught even though every key would individually be
    non-null.
    """
    from tests.conftest import _day_conditions
    from tidescout.pipeline.payload import build_payload

    expected = _day_conditions(4_200.0, "med")
    p = build_payload(**synthetic_day)
    astro = p["astro"]
    assert astro == {
        "dawn": expected.sun.dawn.isoformat(),
        "sunrise": expected.sun.sunrise.isoformat(),
        "sunset": expected.sun.sunset.isoformat(),
        "dusk": expected.sun.dusk.isoformat(),
        "moon_phase_frac": expected.moon.phase_frac,
        "moonrise": expected.moon.rise.isoformat(),
        "moonset": expected.moon.set.isoformat(),
    }
    assert len({astro["dawn"], astro["sunrise"], astro["sunset"], astro["dusk"]}) == 4


def test_astro_block_is_null_per_source_when_sun_or_moon_is_missing(monkeypatch):
    """`sun` and `moon` are two independent upstream sources -- `dayloader.
    load_day`'s own wrapper degrades each on its own dead source, never both
    together. Nulls both here at once (the cheapest way to prove neither
    field access raises); a fishery that lost only one source would see the
    other half of `astro` stay real, per `_astro_to_dict`'s per-source
    guards.
    """
    from tests.conftest import _day_conditions
    from tidescout.pipeline.payload import build_payload
    from tidescout.sources import dayloader

    dark = dataclasses.replace(_day_conditions(4_200.0, "med"), sun=None, moon=None)
    monkeypatch.setattr(dayloader, "load_day", lambda *a, **k: dark)
    p = build_payload(
        slug="winyah-bay", day=date(2026, 8, 16), model_label="gfs_seamless", cache=None
    )
    assert p["astro"] == {
        "dawn": None, "sunrise": None, "sunset": None, "dusk": None,
        "moon_phase_frac": None, "moonrise": None, "moonset": None,
    }
