"""Factor behaviour. Each test isolates one factor and asserts the DIRECTION
of its response, not a calibrated value -- the curves are tuned by hindcasting
and every number in them will move."""

import dataclasses
import math

import pytest
import yaml

from tidescout.config import load_species
from tidescout.engine.activation import FeatureMetrics
from tidescout.engine.conditions import HourlyConditions
from tidescout.engine.score import (
    FACTORS,
    HourScore,
    SubScore,
    combine,
    score_factors,
    score_feature,
)
from tidescout.models import StructureThresholds
from tidescout.paths import FISHERIES_DIR


def _hour(**kw):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    base = {"time": datetime(2026, 10, 15, 15, 0, tzinfo=ZoneInfo("America/New_York"))}
    return HourlyConditions(**{**base, **kw})


def _by_factor(subs):
    return {s.factor: s for s in subs}


def _sal(ppt: float, *, fitted: bool = False, extrapolated: bool = False):
    """A salinity reading for tests.

    `fitted=False` is the DEFAULT because it is Winyah Bay's actual state --
    a helper defaulting to True would quietly exercise a configuration this
    project does not have.
    """
    from tidescout.engine.score import SalinityProvenance, SalinityReading

    return SalinityReading(ppt, SalinityProvenance.MODELLED,
                           fitted=fitted, extrapolated=extrapolated)


def test_the_factor_list_and_the_authored_yaml_cannot_drift_apart():
    """Task 3's test file declares its own FACTORS set and this module declares
    a FACTORS tuple. They agree today -- verified 2026-08-26 -- and nothing
    else makes them keep agreeing.

    Drift is silent and expensive in both directions: a factor named here but
    absent from the YAML raises KeyError deep inside `_scored` at scoring time,
    and a factor weighted in the YAML but missing here is simply never
    evaluated, so its weight quietly vanishes from the geometric mean and every
    score shifts with no error anywhere.

    This is the one place that can see both, because Task 3 ships before this
    module exists and its tests cannot import FACTORS from here.
    """
    for name, profile in load_species().items():
        assert set(FACTORS) == set(profile.weights), name


def test_slack_water_craters_the_flow_factor():
    p = load_species()["redfish"]
    slack = _by_factor(score_factors(_hour(), None, p, flow_speed=0.01))["flow"]
    running = _by_factor(score_factors(_hour(), None, p, flow_speed=0.5))["flow"]
    assert slack.value < 0.3
    assert running.value > slack.value
    # Absolute pins, not just the two relative checks above (2026-08-26
    # review, Minor 5): `flow_speed` is m/s already on this path -- only the
    # CO-OPS knot fallback converts units -- so a bug that quietly
    # re-applied the knot->m/s factor here would still satisfy "slack <
    # running" but must fail these. redfish's flow curve is exact at both
    # points: x=0.01 sits on the 0.0->0.10 segment (y 0.10->0.45), x=0.5 is
    # an authored breakpoint (y=1.00).
    assert abs(slack.value - 0.135) < 0.01
    assert abs(running.value - 1.0) < 0.01


def test_flow_reason_labels_the_regime_the_code_actually_computed():
    """2026-08-26 re-review: inverting slack<->ripping stayed GREEN under
    the label-swap gap Important 2 narrowed but did not close -- this is
    one of the four factors named. Picks a speed inside each of the three
    regimes (thresholds at 0.1 and 0.8 m/s) and ties the word to the side
    of the boundary it came from."""
    p = load_species()["redfish"]
    slack = _by_factor(score_factors(_hour(), None, p, flow_speed=0.01))["flow"]
    moving = _by_factor(score_factors(_hour(), None, p, flow_speed=0.5))["flow"]
    ripping = _by_factor(score_factors(_hour(), None, p, flow_speed=1.2))["flow"]
    assert "slack" in slack.reason
    assert "moving" not in slack.reason and "ripping" not in slack.reason
    assert "moving" in moving.reason
    assert "slack" not in moving.reason and "ripping" not in moving.reason
    assert "ripping" in ripping.reason
    assert "slack" not in ripping.reason and "moving" not in ripping.reason


def test_the_same_printed_flow_speed_never_carries_two_labels():
    """2026-08-26 review, Minor 7: the label used to compare the RAW speed
    against the 0.1/0.8 thresholds while the reason printed the speed
    rounded to 2 dp. 0.0999 m/s is genuinely below 0.1 but displays as
    "0.10 m/s" -- the same string a speed just ABOVE 0.1 also displays,
    which the old code labelled "moving". Without the fix this reads
    "flow 0.10 m/s — slack", identically formatted to a real "moving"
    reading -- a reader (or a hindcast log scraping the reason string) can
    no longer tell the two apart from the number and label alone.
    """
    p = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(), None, p, flow_speed=0.0999))["flow"]
    assert "0.10" in sub.reason, sub.reason
    assert "moving" in sub.reason, sub.reason
    assert "slack" not in sub.reason, sub.reason


def test_the_same_printed_pressure_trend_never_carries_two_labels():
    """2026-08-26 re-review: Minor 7's fix touched flow, pressure, wind and
    salinity, but only flow had a regression test -- reverting pressure's
    rounding alone left all 791 tests green. -0.501 mb/3h is genuinely below
    the -0.5 "falling" threshold but displays as "-0.5" at `:+.1f` -- the
    same string a genuinely steady -0.499 also displays, which raw
    comparison would label "pre-frontal feeding window" while a reader sees
    the identical "-0.5 mb/3h" a steady reading also shows.
    """
    p = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(pressure_trend_mb_3h=-0.501), None, p))["pressure"]
    assert "-0.5" in sub.reason, sub.reason
    assert "steady" in sub.reason, sub.reason
    assert "pre-frontal feeding window" not in sub.reason, sub.reason


def test_the_same_printed_wind_speed_never_carries_two_labels():
    """2026-08-26 re-review: see the pressure test above -- wind is the
    second of the three unpinned factors. 4.6 kn is genuinely below the
    "calm" threshold of 5 but rounds to "5" at `:.0f`, the same string a
    genuine 5.0 kn ("light", 5 is not < 5) also displays.
    """
    p = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(wind_speed_kn=4.6), None, p))["wind"]
    assert "5 kn" in sub.reason, sub.reason
    assert "light" in sub.reason, sub.reason
    assert "calm" not in sub.reason, sub.reason


def test_the_same_printed_salinity_never_carries_two_labels():
    """2026-08-26 re-review: see the pressure test above -- salinity is the
    third of the three unpinned factors. 4.96 ppt is genuinely below the
    "near-fresh" threshold of 5 but rounds to "5.0" at `:.1f`, the same
    string a genuine 5.0 ppt ("brackish", 5.0 is not < 5) also displays.
    """
    p = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(), None, p, salinity=_sal(4.96, fitted=True)))["salinity"]
    assert "5.0 ppt" in sub.reason, sub.reason
    assert "brackish" in sub.reason, sub.reason
    assert "near-fresh" not in sub.reason, sub.reason


def test_stage_factor_converts_the_half_cycle_frac_to_a_full_cycle_and_labels_it_right():
    """2026-08-26 review, Important 1: `stage_at`'s `frac` resets to 0 at
    every hi/lo turn -- a HALF-cycle fraction -- but the YAML curves are
    authored against the FULL 0 (low water) .. 1 (next low water) cycle.
    `tide_phase` says which half, so the factor must recombine them:
    flooding halves the half-cycle frac, ebbing adds it to the far side.

    This is also the flood/ebb label test Important 2 asked for: it ties the
    reason's "flooding"/"ebbing" word to the `tide_phase` that produced it,
    so swapping the labels -- the exact defect that shipped -- fails here,
    and so does skipping the conversion (the same raw 0.8 would appear in
    both reasons instead of 0.40 and 0.90)."""
    p = load_species()["redfish"]
    flood = _by_factor(score_factors(
        _hour(tide_frac=0.8, tide_phase="rising"), None, p))["stage"]
    ebb = _by_factor(score_factors(
        _hour(tide_frac=0.8, tide_phase="falling"), None, p))["stage"]
    assert "flooding" in flood.reason
    assert "0.40" in flood.reason, flood.reason  # 0.8 / 2
    assert "ebbing" in ebb.reason
    assert "0.90" in ebb.reason, ebb.reason  # 0.5 + 0.8 / 2
    # Same half-cycle frac, opposite phase -- must NOT produce the same
    # full-cycle value or the curve cannot express a direction bias at all.
    assert flood.value != ebb.value


def test_stage_missing_phase_marks_missing_even_with_a_frac_present():
    """`tide_phase` can be None independently of `tide_frac` (see
    `stage_at`) -- a frac with no phase to interpret it is not enough
    information to guess a half, so it must not silently default to one."""
    p = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(tide_frac=0.3, tide_phase=None), None, p))["stage"]
    assert sub.missing is True


def test_light_reason_quotes_the_cloud_widened_value_actually_scored():
    """2026-08-26 review, Minor 3/4: the curve is evaluated at `effective`
    (cloud-widened), not the raw hours-from-twilight, and the disclosure
    must fire whenever cloud cover is non-zero, not only above some cutoff.
    """
    from datetime import datetime
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from tidescout.sources.astronomy import SunTimes

    tz = ZoneInfo("America/New_York")
    sunrise = datetime(2026, 10, 15, 7, 0, tzinfo=tz)
    sunset = datetime(2026, 10, 15, 19, 0, tzinfo=tz)
    sun = SunTimes(dawn=sunrise, sunrise=sunrise, sunset=sunset, dusk=sunset)
    day = SimpleNamespace(sun=sun, solunar=[], water=None)
    p = load_species()["redfish"]

    # 09:00 is 2.0 h after sunrise (the nearer of the two twilights); 100%
    # cloud widens it to 2.0 * (1 - 0.35) = 1.30 h -- the value the curve is
    # actually evaluated at.
    cloudy = _by_factor(score_factors(
        _hour(time=datetime(2026, 10, 15, 9, 0, tzinfo=tz), cloud_cover_pct=100.0),
        day, p))["light"]
    # The number the curve was actually evaluated at leads the reason;
    # the raw pre-widening value may still appear afterward as context
    # (it does here, in "widened it from 2.0 h"), so this checks WHERE
    # 1.3 appears rather than banning 2.0 outright.
    assert cloudy.reason.startswith("1.3"), cloudy.reason
    assert "cloud" in cloudy.reason.lower()

    # Clear sky: effective == raw, no cloud note needed.
    clear = _by_factor(score_factors(
        _hour(time=datetime(2026, 10, 15, 9, 0, tzinfo=tz), cloud_cover_pct=0.0),
        day, p))["light"]
    assert "2.0" in clear.reason
    assert "cloud" not in clear.reason.lower()

    # Mid-range: cloud=50 is exactly where the 2026-08-26 re-review measured
    # a silent +0.070 score movement under the OLD `cloud > 50` cutoff
    # (50 is not strictly greater than 50, so the old code disclosed
    # nothing here). Pinning only the two extremes let that cutoff pass;
    # this pins the boundary itself. effective = 2.0 * (1 - 0.35*0.5) = 1.65,
    # which `:.1f` renders as "1.6".
    mid = _by_factor(score_factors(
        _hour(time=datetime(2026, 10, 15, 9, 0, tzinfo=tz), cloud_cover_pct=50.0),
        day, p))["light"]
    assert mid.reason.startswith("1.6"), mid.reason
    assert "cloud" in mid.reason.lower()


def test_editing_light_cloud_widen_in_yaml_alone_moves_the_hour_score(tmp_path):
    """2026-08-26 review, Important 2: the cloud-widening coefficient in the
    light factor (`effective = hours_off * (1.0 - light_cloud_widen *
    cloud / 100.0)`) used to be a hard-coded Python `0.35`, unreachable from
    YAML and species-independent, even though `light`'s WEIGHT (0.5-0.9) and
    its CURVE are both authored per species right beside it -- the same
    class of violation `structure_ambush` shipped with until Task 6 pulled
    its ramp into YAML. Same style of proof
    `test_editing_structure_weight_in_yaml_alone_moves_the_activation` uses:
    copy the real `species_weights.yaml`, edit ONE number, reload through
    the real `load_species` loader, and watch the light sub-score -- and the
    hour score built from it -- move with zero Python touched.
    """
    from datetime import datetime
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from tidescout.sources.astronomy import SunTimes

    raw = yaml.safe_load((FISHERIES_DIR / "species_weights.yaml").read_text())
    assert raw["redfish"]["light_cloud_widen"] == pytest.approx(0.35), "baseline assumption"
    raw["redfish"]["light_cloud_widen"] = 0.60
    edited = tmp_path / "species_weights.yaml"
    edited.write_text(yaml.safe_dump(raw))

    baseline = load_species()["redfish"]
    mutated = load_species(edited)["redfish"]

    tz = ZoneInfo("America/New_York")
    sunrise = datetime(2026, 10, 15, 7, 0, tzinfo=tz)
    sunset = datetime(2026, 10, 15, 19, 0, tzinfo=tz)
    sun = SunTimes(dawn=sunrise, sunrise=sunrise, sunset=sunset, dusk=sunset)
    day = SimpleNamespace(sun=sun, solunar=[], water=None)
    # 09:00, 100% cloud -- same fixture shape as the reason test above, so
    # `effective` genuinely differs between baseline (1.3 h) and mutated
    # (0.8 h) rather than both clamping to the same curve breakpoint.
    hour = _hour(time=datetime(2026, 10, 15, 9, 0, tzinfo=tz), cloud_cover_pct=100.0)

    baseline_light = _by_factor(score_factors(hour, day, baseline))["light"]
    mutated_light = _by_factor(score_factors(hour, day, mutated))["light"]
    assert mutated_light.value != baseline_light.value
    # More widening at the same cloud cover pushes `effective` FURTHER below
    # `hours_off`, i.e. closer to twilight, which this curve scores higher.
    assert mutated_light.value > baseline_light.value, (baseline_light, mutated_light)

    baseline_score = combine(score_factors(hour, day, baseline)).score
    mutated_score = combine(score_factors(hour, day, mutated)).score
    assert mutated_score != baseline_score, "the hour score must move too, not just the sub-score"


def test_every_sub_score_carries_a_reason():
    """The UI renders factor bars with text; a sub-score with no explanation
    is an incomplete implementation."""
    p = load_species()["redfish"]
    for s in score_factors(_hour(wind_speed_kn=8.0), None, p, flow_speed=0.4):
        assert s.reason, f"{s.factor} has no reason"
        assert not s.reason.endswith(".."), f"{s.factor} reason looks malformed"


def test_missing_input_marks_the_factor_missing_rather_than_scoring_zero():
    """A dark anemometer must not read as dead calm."""
    p = load_species()["redfish"]
    wind = _by_factor(score_factors(_hour(wind_speed_kn=None), None, p))["wind"]
    assert wind.missing is True
    assert "no data" in wind.reason.lower() or "missing" in wind.reason.lower()


def test_strong_wind_suppresses_the_wind_factor():
    p = load_species()["redfish"]
    calm = _by_factor(score_factors(_hour(wind_speed_kn=6.0), None, p))["wind"]
    gale = _by_factor(score_factors(_hour(wind_speed_kn=30.0), None, p))["wind"]
    assert gale.value < calm.value
    assert gale.value < 0.25


def test_falling_pressure_scores_above_sharply_rising_pressure():
    p = load_species()["redfish"]
    falling = _by_factor(score_factors(_hour(pressure_trend_mb_3h=-2.5), None, p))["pressure"]
    rising = _by_factor(score_factors(_hour(pressure_trend_mb_3h=+3.0), None, p))["pressure"]
    assert falling.value > rising.value


def test_wind_reason_labels_the_category_the_code_actually_computed():
    """2026-08-26 re-review: inverting calm<->"hard -- fishability suffers"
    stayed GREEN under the label-swap gap Important 2 narrowed but did not
    close -- one of the four factors named. Picks a speed inside each of the
    four categories (thresholds at 5, 12, 18 kn)."""
    p = load_species()["redfish"]
    calm = _by_factor(score_factors(_hour(wind_speed_kn=3.0), None, p))["wind"]
    light = _by_factor(score_factors(_hour(wind_speed_kn=8.0), None, p))["wind"]
    building = _by_factor(score_factors(_hour(wind_speed_kn=15.0), None, p))["wind"]
    hard = _by_factor(score_factors(_hour(wind_speed_kn=25.0), None, p))["wind"]
    assert "calm" in calm.reason
    assert "light" in light.reason and "calm" not in light.reason
    assert "building" in building.reason
    assert "hard" in hard.reason and "fishability suffers" in hard.reason
    # The specific opposite-ends pair named in the review's mutation table.
    assert "calm" not in hard.reason
    assert "hard" not in calm.reason and "fishability suffers" not in calm.reason


def test_pressure_reason_labels_the_trend_the_code_actually_computed():
    """2026-08-26 re-review: swapping "pre-frontal feeding window" and
    "post-frontal shutdown" stayed GREEN under the label-swap gap Important 2
    narrowed but did not close -- one of the four factors named."""
    p = load_species()["redfish"]
    falling = _by_factor(score_factors(_hour(pressure_trend_mb_3h=-2.5), None, p))["pressure"]
    steady = _by_factor(score_factors(_hour(pressure_trend_mb_3h=0.0), None, p))["pressure"]
    rising = _by_factor(score_factors(_hour(pressure_trend_mb_3h=+3.0), None, p))["pressure"]
    assert "pre-frontal feeding window" in falling.reason
    assert "post-frontal shutdown" not in falling.reason
    assert "steady" in steady.reason
    assert "post-frontal shutdown" in rising.reason
    assert "pre-frontal feeding window" not in rising.reason


def test_near_fresh_water_penalises_trout_far_more_than_redfish():
    """Spec section 7: the same eddy scores near zero up-bay after a freshet."""
    trout = load_species()["speckled_trout"]
    red = load_species()["redfish"]
    t = _by_factor(score_factors(_hour(), None, trout, salinity=_sal(2.0)))["salinity"]
    r = _by_factor(score_factors(_hour(), None, red, salinity=_sal(2.0)))["salinity"]
    assert t.value < r.value
    assert t.value < 0.3


def test_salinity_reason_labels_the_band_the_code_actually_computed():
    """2026-08-26 re-review: inverting near-fresh<->salty stayed GREEN under
    the label-swap gap Important 2 narrowed but did not close -- one of the
    four factors named. Picks a ppt inside each of the three bands
    (thresholds at 5 and 18 ppt). `fitted=True` keeps the reading
    unconstrained-free so the band word is not buried in a caveat."""
    p = load_species()["redfish"]
    fresh = _by_factor(score_factors(_hour(), None, p, salinity=_sal(2.0, fitted=True)))["salinity"]
    brackish = _by_factor(score_factors(
        _hour(), None, p, salinity=_sal(10.0, fitted=True)))["salinity"]
    salty = _by_factor(score_factors(
        _hour(), None, p, salinity=_sal(25.0, fitted=True)))["salinity"]
    assert "near-fresh" in fresh.reason
    assert "brackish" not in fresh.reason and "salty" not in fresh.reason
    assert "brackish" in brackish.reason
    assert "near-fresh" not in brackish.reason and "salty" not in brackish.reason
    assert "salty" in salty.reason
    assert "near-fresh" not in salty.reason and "brackish" not in salty.reason


def test_an_uncalibrated_salinity_is_scored_but_marked_provisional():
    """The owner's 2026-08-26 call: include it, flag it. `fitted=False` is
    Winyah Bay's real state, so this is the path that actually runs -- the
    score must still be a number, and the caveat must be in the text a person
    reads, not only in a payload field the UI may not render."""
    red = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(), None, red, salinity=_sal(22.0)))["salinity"]
    assert sub.missing is False, "unconstrained is not the same as absent"
    assert math.isfinite(sub.value) and 0.0 <= sub.value <= 1.0
    assert sub.weight == red.weights["salinity"], "flagged, not discounted"
    assert sub.provisional is True
    assert "UNCALIBRATED" in sub.reason


def test_a_measured_salinity_carries_no_caveat():
    """The discriminating half. Without this, a bug marking EVERYTHING
    provisional would pass the test above.

    `fitted=False` here is deliberate, not an oversight: `SalinityReading.
    fitted` is a required field (2026-08-26 review, Minor 5), and passing
    `False` -- the "worst case" for that field -- proves `constrained`
    genuinely short-circuits on `provenance is MEASURED` rather than merely
    happening to agree with a `True` default that was never exercised.
    """
    from tidescout.engine.score import SalinityProvenance, SalinityReading

    red = load_species()["redfish"]
    reading = SalinityReading(22.0, SalinityProvenance.MEASURED, fitted=False)
    sub = _by_factor(score_factors(_hour(), None, red, salinity=reading))["salinity"]
    assert sub.provisional is False
    assert "UNCALIBRATED" not in sub.reason and "~" not in sub.reason


def test_extrapolation_is_disclosed_separately_from_calibration():
    """`fitted` and `extrapolated` answer different questions and a reading can
    fail either independently -- see `SalinityField`'s docstring."""
    red = load_species()["redfish"]
    sub = _by_factor(score_factors(
        _hour(), None, red, salinity=_sal(22.0, fitted=True, extrapolated=True),
    ))["salinity"]
    assert sub.provisional is True
    assert "outside the calibrated range" in sub.reason
    assert "UNCALIBRATED" not in sub.reason, "this one IS fitted"


def test_season_factor_uses_the_month_of_the_hour():
    p = load_species()["redfish"]
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    oct_ = _by_factor(score_factors(
        _hour(time=datetime(2026, 10, 15, 12, tzinfo=tz)), None, p))["season"]
    assert 0.0 <= oct_.value <= 1.0
    assert str(10) in oct_.reason or "octo" in oct_.reason.lower()

    # A second month, checked against the profile's own table rather than a
    # second hardcoded literal (2026-08-26 review, Minor 5): the previous
    # version passed even with the month hardcoded to 10, since it never
    # tried any other month.
    jan = _by_factor(score_factors(
        _hour(time=datetime(2026, 1, 15, 12, tzinfo=tz)), None, p))["season"]
    assert "january" in jan.reason.lower()
    assert jan.value == p.months[1]
    assert oct_.value == p.months[10]
    assert jan.value != oct_.value


def test_every_species_stage_curve_is_cyclic_closed():
    """2026-08-26 review: `stage`'s x runs a FULL tidal cycle (Important 1),
    so x=0.0 and x=1.0 are the SAME physical instant -- low water -- and y
    must match at both ends or the curve has a discontinuity at the wrap.
    flounder's y(1.0) was 0.60 against y(0.0)'s 0.50 while redfish and
    speckled_trout were already authored closed; fixed to 0.50 in
    species_weights.yaml. This generalises the fix so a species added later
    with the same slip is caught here instead of by inspection."""
    for name, profile in load_species().items():
        stage = profile.curves["stage"]
        assert stage.x[0] == 0.0 and stage.x[-1] == 1.0, name
        assert stage.y[0] == stage.y[-1], (
            f"{name}: y(0.0)={stage.y[0]} != y(1.0)={stage.y[-1]} at the low-water wrap"
        )


def test_all_nine_factors_are_always_present():
    """Even when every input is missing -- the UI shows nine bars, some greyed."""
    p = load_species()["redfish"]
    subs = score_factors(_hour(), None, p)
    assert len(subs) == 9
    assert len({s.factor for s in subs}) == 9


def _subs(**vals):
    return [SubScore(f, v, 1.0, f"{f} reason", False) for f, v in vals.items()]


def test_score_is_bounded_zero_to_one_hundred():
    for v in (0.0, 0.5, 1.0):
        s = combine(_subs(a=v, b=v, c=v))
        assert 0 <= s.score <= 100


def test_one_dead_factor_tanks_the_hour():
    """The whole reason for a geometric mean. Arithmetic would give 0.67 here."""
    tanked = combine(_subs(a=0.0, b=1.0, c=1.0))
    assert tanked.score < 20


def test_all_good_factors_score_near_one_hundred():
    assert combine(_subs(a=1.0, b=1.0, c=1.0)).score >= 99


def test_missing_factors_are_excluded_and_weights_renormalised():
    """A dark sensor must not drag the score down -- spec section 8 requires
    exclusion with renormalisation, never a silent default."""
    present = combine(_subs(a=0.8, b=0.8))
    with_missing = combine(
        _subs(a=0.8, b=0.8) + [SubScore("c", float("nan"), 1.0, "c: no data", True)]
    )
    assert with_missing.score == present.score
    assert with_missing.excluded == ["c"]


def test_confidence_falls_as_factors_go_missing():
    full = combine(_subs(a=0.8, b=0.8, c=0.8))
    partial = combine(
        _subs(a=0.8, b=0.8) + [SubScore("c", float("nan"), 1.0, "c: no data", True)]
    )
    assert partial.confidence < full.confidence
    assert full.confidence == pytest.approx(1.0)


def test_weights_actually_weight():
    heavy = [SubScore("a", 0.2, 9.0, "", False), SubScore("b", 1.0, 1.0, "", False)]
    light = [SubScore("a", 0.2, 1.0, "", False), SubScore("b", 1.0, 9.0, "", False)]
    assert combine(heavy).score < combine(light).score


def test_a_provisional_factor_keeps_confidence_but_lowers_constrained_share():
    """The two numbers answer different questions. If a regression made
    `constrained_share` an alias of `confidence`, this is what catches it."""
    full = combine([
        SubScore("flow", 0.8, 1.0, "", False),
        SubScore("salinity", 0.6, 1.0, "", False, provisional=True),
    ])
    assert full.confidence == pytest.approx(1.0), "nothing was excluded"
    assert full.constrained_share == pytest.approx(0.5)
    assert full.provisional == ["salinity"]


def test_constrained_share_is_one_when_nothing_is_provisional():
    """The discriminating half -- without it, hardcoding constrained_share to
    0.5 would pass the test above."""
    s = combine([
        SubScore("flow", 0.8, 1.0, "", False),
        SubScore("salinity", 0.6, 1.0, "", False),
    ])
    assert s.constrained_share == pytest.approx(1.0)
    assert s.provisional == []


def test_everything_missing_returns_zero_confidence_not_a_crash():
    s = combine([SubScore("a", float("nan"), 1.0, "a: no data", True)])
    assert s.confidence == 0.0
    assert s.score == 0


def test_score_is_monotone_in_a_single_factor():
    """Property test: improving one input can never lower the score."""
    previous = -1
    for v in [i / 20 for i in range(21)]:
        s = combine(_subs(a=v, b=0.7, c=0.7)).score
        assert s >= previous
        previous = s


def test_zero_is_floored_rather_than_producing_negative_infinity():
    """log(0) is -inf, which would propagate NaN through the whole payload.
    The floor must still tank the score -- it is a guard, not a rescue."""
    s = combine(_subs(a=0.0, b=1.0, c=1.0))
    assert math.isfinite(s.score)
    assert s.score < 20


def _metrics(**kw):
    base = dict(
        key="dropoff-abc123", type="dropoff", speed=0.5, ambush=0.4, strain=2e-3,
        okubo_w=-1e-5, convergence=1e-4, wet_fraction=1.0, flood_phase=float("nan"),
        n_cells=42,
        # Required on FeatureMetrics and NOT optional -- omitting it is a
        # TypeError, not a default. Added to the dataclass after this plan was
        # written. 0.0 is the neutral value (no wet disc cell classifies as an
        # eddy).
        eddy_share=0.0,
    )
    return FeatureMetrics(**{**base, **kw})


_T = StructureThresholds()  # the class defaults; `score_feature` requires this argument.


def _full_hour(**kw):
    """An hour with EVERY factor's INPUT live -- unlike `_hour()` above,
    which deliberately carries only a timestamp so the sparse-input tests in
    this file can isolate one factor at a time.

    `score_feature`'s tests need the OPPOSITE (2026-08-26 review, Important
    1): the original version of the headline salinity test ran on a plain
    `_hour()`, under which 6 of 9 factors come back missing and `confidence`
    lands at 0.38-0.40. Excluding 6 factors doesn't just shrink the sample --
    it RENORMALISES the remaining weights over a much smaller total, so
    salinity's share of the geometric mean roughly triples versus what it is
    on a real, fully-observed hour. That inflated share is what let the
    original test pass; on a fully-populated hour the same assertion fails
    (measured below). Every field here is live, so `confidence` is 1.0 and
    every factor counts at its AUTHORED weight, not an inflated one.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    base = dict(
        time=datetime(2026, 10, 15, 15, 0, tzinfo=ZoneInfo("America/New_York")),
        air_temp_f=78.0, wind_speed_kn=8.0, wind_dir_deg=180.0, wind_gust_kn=12.0,
        pressure_mb=1015.0, pressure_trend_mb_3h=-0.8, cloud_cover_pct=30.0,
        precip_in=0.0, tide_height_ft=2.8, tide_phase="rising", tide_frac=0.4,
        current_speed_kn=None, current_dir_deg=90.0,
    )
    return HourlyConditions(**{**base, **kw})


def _full_day():
    """A `day` with `.sun`, `.solunar` and `.water` all populated, paired
    with `_full_hour`.

    A `SimpleNamespace`, not a real `DayConditions`: `score_factors` only
    ever reads these three attributes off `day`, via `getattr`, so a real
    `DayConditions` (with the `discharge`/`missing`/`hours` machinery Task 7
    needs) would be scaffolding this task has no use for -- the same idiom
    `test_light_reason_quotes_the_cloud_widened_value_actually_scored`
    already uses above, just with `solunar` and `water` filled in too instead
    of left empty/None.
    """
    from datetime import datetime
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from tidescout.sources.astronomy import SolunarPeriod, SunTimes
    from tidescout.sources.usgs import WaterSummary

    tz = ZoneInfo("America/New_York")
    d = datetime(2026, 10, 15, tzinfo=tz)
    return SimpleNamespace(
        sun=SunTimes(dawn=d.replace(hour=6, minute=30), sunrise=d.replace(hour=7),
                    sunset=d.replace(hour=18, minute=45),
                    dusk=d.replace(hour=19, minute=15)),
        solunar=[SolunarPeriod(kind="major", start=d.replace(hour=14),
                               end=d.replace(hour=16))],
        water=WaterSummary(temp_f=71.0, temp_trend_f_3d=-0.5,
                          salinity_ppt=None, source="synthetic"),
    )


def test_the_same_feature_scores_lower_in_fresh_water_for_trout():
    """Spec section 7's actual, owner-ratified requirement (2026-08-26
    review): salinity moves a feature's activation DIRECTIONALLY, by a
    margin that holds species by species -- NOT the literal multiplier spec
    section 7's prose also describes, which the owner rejected because it
    hands one still-uncalibrated factor veto power over the whole map
    (Winyah's `fitted=False`). Spec section 8 lists salinity as factor 8 of
    nine, weighted like the others, and that is the design this asserts.

    Runs on `_full_hour()`/`_full_day()`, not the sparse `_hour()`: see
    `_full_hour`'s docstring for why a sparse fixture inflates salinity's
    share and makes this test pass for the wrong reason. Measured directly
    against the shipped code 2026-08-26, fresh (1 ppt) vs salty (22 ppt),
    same eddy (`_metrics()`'s defaults), same hour, same day:

        trout:     salty 84, fresh 61  (ratio 0.73)
        flounder:  salty 82, fresh 73  (ratio 0.89)
        redfish:   salty 86, fresh 82  (ratio 0.95) -- "broadly tolerant"

    Trout moves the most (spec section 7: "~10-30 ppt, avoid near-fresh"),
    flounder moves less but still down, and redfish is nearly flat, which
    spec section 7 calls out by name rather than treating as a bug -- so
    this pins that near-flatness too, rather than only the two species that
    move.
    """
    trout = load_species()["speckled_trout"]
    flounder = load_species()["southern_flounder"]
    red = load_species()["redfish"]

    def scores(profile):
        salty = score_feature(_metrics(), _full_hour(), _full_day(), profile,
                              salinity=_sal(22.0), thresholds=_T)
        fresh = score_feature(_metrics(), _full_hour(), _full_day(), profile,
                              salinity=_sal(1.0), thresholds=_T)
        return salty.activation, fresh.activation

    trout_salty, trout_fresh = scores(trout)
    flounder_salty, flounder_fresh = scores(flounder)
    red_salty, red_fresh = scores(red)

    # Directional and by a real margin for the two salinity-sensitive species.
    assert trout_fresh < trout_salty - 15, (trout_fresh, trout_salty)
    assert flounder_fresh < flounder_salty - 5, (flounder_fresh, flounder_salty)
    # Trout is the sharper of the two (spec section 7 singles it out).
    trout_drop = trout_salty - trout_fresh
    flounder_drop = flounder_salty - flounder_fresh
    assert trout_drop > flounder_drop, (trout_drop, flounder_drop)
    # Redfish is "broadly tolerant": present, not absent, but small.
    assert 0 <= red_salty - red_fresh < 10, (red_salty, red_fresh)


def test_a_strong_ambush_pocket_outscores_featureless_water():
    red = load_species()["redfish"]
    strong = score_feature(_metrics(ambush=0.9), _hour(), None, red,
                           salinity=_sal(18.0), thresholds=_T)
    weak = score_feature(_metrics(ambush=0.0), _hour(), None, red,
                         salinity=_sal(18.0), thresholds=_T)
    assert strong.activation > weak.activation


def test_a_strong_eddy_share_reading_registers_as_structure():
    """2026-08-26 review, Finding 3: `eddy_share` is Phase 1's DEDICATED eddy
    channel -- "the eddy channel that leaves `okubo_w` alone", per
    `FeatureMetrics`'s own docstring -- because `okubo_w` is MAX-reduced per
    feature and structurally cannot report an eddy (of 13,614 real
    per-feature samples measured over the whole winyah-bay library, the most
    negative is -8.8e-7, ten times inside the quiet band -- floating-point
    residue, not a rotation). Spec section 7's headline object IS an eddy,
    so this pins that a strong `eddy_share` reading actually moves the
    structure sub-score, with every OTHER structural signal held quiet --
    the case the PREVIOUS version of `_structure_subscore` (deriving "eddy"
    from negative `okubo_w` instead) could not recognise at all.
    """
    red = load_species()["redfish"]
    quiet = _metrics(ambush=0.0, okubo_w=0.0, convergence=0.0, eddy_share=0.0)
    eddying = _metrics(ambush=0.0, okubo_w=0.0, convergence=0.0, eddy_share=0.2)
    no_eddy = score_feature(quiet, _hour(), None, red, salinity=_sal(18.0), thresholds=_T)
    has_eddy = score_feature(eddying, _hour(), None, red, salinity=_sal(18.0), thresholds=_T)
    no_eddy_structure = _by_factor(no_eddy.subs)["structure"]
    has_eddy_structure = _by_factor(has_eddy.subs)["structure"]
    assert "eddy" in has_eddy_structure.reason
    assert has_eddy_structure.value > no_eddy_structure.value
    assert has_eddy.activation > no_eddy.activation


def test_a_feature_outside_the_domain_scores_zero_with_an_explanation():
    """n_cells == 0 means the feature has no library cells; it must not be
    silently scored on NaN metrics."""
    red = load_species()["redfish"]
    out = score_feature(_metrics(n_cells=0), _hour(), None, red,
                        salinity=_sal(18.0), thresholds=_T)
    assert out.activation == 0
    assert "outside" in out.reason.lower() or "no cells" in out.reason.lower()


def test_activation_carries_the_feature_key_unchanged():
    """The frontend keys markers off this; Phase 1 Task 8 made it stable."""
    red = load_species()["redfish"]
    got = score_feature(_metrics(key="bar-9f2c1a7b4e05"), _hour(), None, red,
                        salinity=_sal(18.0), thresholds=_T)
    assert got.key == "bar-9f2c1a7b4e05"


def test_a_dry_flat_scores_zero_however_good_the_conditions():
    """You cannot fish a flat that has no water on it, at any hour --
    `wet_fraction <= 0.0` gates to zero regardless of the tide."""
    red = load_species()["redfish"]
    dry = score_feature(_metrics(type="flat", wet_fraction=0.0, speed=0.0),
                        _hour(), None, red, salinity=_sal(18.0), thresholds=_T)
    assert dry.activation < 10


def test_a_partly_wet_flat_scores_full_at_high_tide_and_zero_at_low():
    """2026-08-26 review, Finding 4: the case that ACTUALLY happens to most
    real flats (median shipped `wet_fraction` 0.735) was untested -- only the
    always-dry edge case (`wet_fraction == 0.0`, which no real flat hits;
    the shipped minimum is 0.143) was pinned. A flat with a partial
    `wet_fraction` must score near its full, un-gated activation at an hour
    when it IS flooded, and near zero at an hour when it is NOT -- not a
    fixed haircut applied at every hour alike.

    `flood_phase=0.2` means this flat's wet window opens at tide fraction
    0.2 (just past low water, still on the flood) and, with
    `wet_fraction=0.6`, closes at (0.2 + 0.6) % 1.0 = 0.8. tide_frac/
    tide_phase are chosen so `_recombine_tide_frac` lands well inside that
    window for "flooded" (full=0.5, high water) and well outside it for
    "dry" (full=0.05, just past low water).

    "Full" is checked against a CONTROL with the identical metrics but
    `type="dropoff"` instead of `"flat"` -- so it is never gated at all --
    scored at the SAME flooded hour, rather than against some other
    plausible-looking number: the flat multiplier at a flooded hour is
    exactly 1.0, so a flat and a non-flat with otherwise identical inputs
    must land on the identical activation. A comparison against the cycle
    average (`wet_fraction` alone, no tide reading) was tried first and
    rejected -- 0.6 vs the true per-hour 1.0 differ by exactly enough to
    make that comparison assert nothing useful.
    """
    red = load_species()["redfish"]
    metrics = _metrics(type="flat", wet_fraction=0.6, flood_phase=0.2)
    control = dataclasses.replace(metrics, type="dropoff")

    flooded_hour = _hour(tide_frac=1.0, tide_phase="rising")
    dry_hour = _hour(tide_frac=0.1, tide_phase="rising")
    # tide_frac=1.0 rising -> full = 1.0/2 = 0.5 (high water) -- inside
    # [0.2, 0.8). tide_frac=0.1 rising -> full = 0.05 -- outside it.

    flooded = score_feature(metrics, flooded_hour, None, red,
                            salinity=_sal(18.0), thresholds=_T)
    dry = score_feature(metrics, dry_hour, None, red, salinity=_sal(18.0), thresholds=_T)
    unrestricted = score_feature(control, flooded_hour, None, red,
                                 salinity=_sal(18.0), thresholds=_T)

    assert dry.activation < 5
    assert flooded.activation > dry.activation + 30
    assert flooded.activation == unrestricted.activation


def test_a_features_flow_comes_from_its_own_metrics_not_a_bay_wide_default():
    """Spec section 7's actual mechanism, restated for flow rather than
    salinity: `score_feature` has no `flow_speed` parameter in its signature
    at all, so the only way the flow factor can vary between two features is
    if it is read from `FeatureMetrics.speed`. `_hour()` carries no
    `current_speed_kn`, so a bay-wide fallback would leave flow MISSING for
    both calls and this test would catch that too -- not just a wrong value,
    but the wrong SOURCE entirely.
    """
    red = load_species()["redfish"]
    slack = score_feature(_metrics(speed=0.01), _hour(), None, red,
                          salinity=_sal(18.0), thresholds=_T)
    ripping = score_feature(_metrics(speed=1.2), _hour(), None, red,
                            salinity=_sal(18.0), thresholds=_T)
    slack_flow = _by_factor(slack.subs)["flow"]
    ripping_flow = _by_factor(ripping.subs)["flow"]
    assert slack_flow.missing is False and ripping_flow.missing is False
    assert "slack" in slack_flow.reason
    assert "ripping" in ripping_flow.reason
    assert slack_flow.value < ripping_flow.value
    assert slack.activation < ripping.activation


def test_an_uncalibrated_salinity_still_flags_a_feature_as_provisional():
    """The owner's 2026-08-26 include-and-flag ruling has to hold at the
    feature boundary, not just the hourly one: the map cannot quietly show an
    unconstrained per-feature number as a confident one. Winyah's salinity
    model is `fitted=False` -- the default `_sal()` produces -- so this is
    the path that actually runs in production. Checks BOTH the structured
    `FeatureActivation.provisional` field (2026-08-26 review, Finding 5 --
    it used to survive only as prose) and the sub-score it names.

    Runs on `_full_hour()`/`_full_day()`, not the sparse `_hour()`: with
    only a timestamp set, 6 of 9 factors come back missing, `confidence`
    lands around 0.38, and "nothing was EXCLUDED, only flagged" would not
    actually be true of the fixture -- the same degenerate-fixture trap
    `_full_hour`'s docstring describes for the headline salinity test.
    """
    red = load_species()["redfish"]
    got = score_feature(_metrics(), _full_hour(), _full_day(), red,
                        salinity=_sal(18.0), thresholds=_T)
    salinity_sub = _by_factor(got.subs)["salinity"]
    assert salinity_sub.missing is False
    assert salinity_sub.provisional is True
    assert "UNCALIBRATED" in salinity_sub.reason
    assert got.provisional == ["salinity"]
    assert got.confidence == pytest.approx(1.0), "nothing was EXCLUDED, only flagged"
    assert got.constrained_share < 1.0
    assert "provisional" in got.reason.lower()


def test_a_feature_with_no_finite_structure_sample_is_missing_not_zero():
    """`sample_features` can leave ambush/okubo_w/eddy_share/convergence all
    NaN even with `n_cells > 0` -- an entirely dry disc, for instance. That
    must read as missing data, never as a silently-computed zero."""
    red = load_species()["redfish"]
    got = score_feature(
        _metrics(ambush=float("nan"), okubo_w=float("nan"), eddy_share=float("nan"),
                 convergence=float("nan")),
        _hour(), None, red, salinity=_sal(18.0), thresholds=_T,
    )
    structure_sub = _by_factor(got.subs)["structure"]
    assert structure_sub.missing is True
    assert math.isnan(structure_sub.value)
    assert "structure" in got.excluded


def test_editing_structure_weight_in_yaml_alone_moves_the_activation(tmp_path):
    """2026-08-26 re-review, unpinned fix 1 of 4: `structure_weight` has to
    be genuinely READ from the species profile at runtime, not merely
    declared as a field while a stale Python constant keeps doing the real
    work -- the reviewer proved this exact regression stays GREEN if all
    that exists is the field declaration. Copies the real
    `species_weights.yaml`, edits ONE number, reloads through the real
    `load_species` loader, and checks the ambush weak/strong activation gap
    actually widens -- the same style of proof Task 3's own review used for
    the nine hourly factors ("copied the YAML, edited one y-value, reloaded
    and watched the score move with zero Python touched"), extended to the
    tenth. Measured manually during this fix: 0.2 gives a 20-point gap,
    5.0 gives an 85-point gap; the threshold below is set well inside that.
    """
    raw = yaml.safe_load((FISHERIES_DIR / "species_weights.yaml").read_text())
    assert raw["redfish"]["structure_weight"] == pytest.approx(0.2), "baseline assumption"
    raw["redfish"]["structure_weight"] = 5.0
    edited = tmp_path / "species_weights.yaml"
    edited.write_text(yaml.safe_dump(raw))

    baseline = load_species()["redfish"]
    mutated = load_species(edited)["redfish"]

    weak, strong = _metrics(ambush=0.0), _metrics(ambush=0.9)

    def gap(profile):
        lo = score_feature(weak, _hour(), None, profile, salinity=_sal(18.0), thresholds=_T)
        hi = score_feature(strong, _hour(), None, profile, salinity=_sal(18.0), thresholds=_T)
        return hi.activation - lo.activation

    baseline_gap = gap(baseline)
    mutated_gap = gap(mutated)
    assert mutated_gap > baseline_gap + 20, (baseline_gap, mutated_gap)


def test_editing_a_structure_curve_in_yaml_alone_moves_the_subscore(tmp_path):
    """2026-08-26 re-review, unpinned fix 2 of 4 -- the constraint this task
    was cited for in the first place was never only the WEIGHT.
    `_structure_subscore`'s response shape (Important 2's second sentence:
    "a clamped linear ramp ... IS a response curve") has to come from YAML
    too, or nothing stops a future edit from quietly moving it back into
    Python the way it started. Flattens redfish's `structure_ambush` curve
    to a constant 0.05 and checks the structure sub-score actually drops --
    if the ramp were still hardcoded in Python, editing this YAML key would
    change nothing at all, which is exactly the regression the reviewer's
    mutation produced.
    """
    raw = yaml.safe_load((FISHERIES_DIR / "species_weights.yaml").read_text())
    original = raw["redfish"]["curves"]["structure_ambush"]
    assert original["y"][-1] == pytest.approx(1.0), "baseline assumption"
    raw["redfish"]["curves"]["structure_ambush"] = {"x": [0.0, 1.0], "y": [0.05, 0.05]}
    edited = tmp_path / "species_weights.yaml"
    edited.write_text(yaml.safe_dump(raw))

    baseline = load_species()["redfish"]
    mutated = load_species(edited)["redfish"]

    # ambush is the only nonzero structural signal here (okubo_w, convergence
    # and eddy_share all sit at their curves' x=0 -> y=0), so the structure
    # sub-score is read straight off the ambush curve with nothing else able
    # to win the MAX and mask a flat curve.
    metrics = _metrics(ambush=0.9, okubo_w=0.0, convergence=0.0, eddy_share=0.0)

    baseline_structure = _by_factor(
        score_feature(metrics, _hour(), None, baseline, salinity=_sal(18.0),
                      thresholds=_T).subs
    )["structure"]
    mutated_structure = _by_factor(
        score_feature(metrics, _hour(), None, mutated, salinity=_sal(18.0),
                      thresholds=_T).subs
    )["structure"]
    assert baseline_structure.value == pytest.approx(1.0)
    assert mutated_structure.value == pytest.approx(0.05)


def test_an_unknown_wet_fraction_gates_a_flat_to_dry_not_full_credit():
    """2026-08-26 re-review, unpinned fix 3 of 4. Finding 9: "a NaN
    wet_fraction skips the gate and scores FULL -- the optimistic default
    the plan forbids." A flat with NO schedule data reaching it at all must
    be treated as NOT confirmed wet, never as fully wet -- the reviewer's
    mutation (skip the gate on NaN, same as the pre-fix code) stayed GREEN
    because nothing exercised this specific input.
    """
    red = load_species()["redfish"]
    metrics = _metrics(type="flat", wet_fraction=float("nan"), flood_phase=float("nan"))
    control = dataclasses.replace(metrics, type="dropoff")

    got = score_feature(metrics, _hour(), None, red, salinity=_sal(18.0), thresholds=_T)
    unrestricted = score_feature(control, _hour(), None, red, salinity=_sal(18.0),
                                 thresholds=_T)

    assert got.activation == 0
    # Sanity check that the gate, not otherwise-poor conditions, is what
    # zeroed this: the identical inputs on a non-flat type score well.
    assert unrestricted.activation > 50
    assert "unknown" in got.reason.lower()


def test_single_rounding_not_double_on_a_worked_example(monkeypatch):
    """2026-08-26 re-review, unpinned fix 4 of 4. Finding 9: the OLD code
    rounded `combine()`'s raw [0, 1] value into `combined.score` (an int),
    THEN multiplied by the flat's wet multiplier and rounded a second time
    -- two roundings where one suffices, and the two orders measurably
    disagree. Worked example the review supplied, reproduced here exactly:
    raw=0.5082, wet multiplier=0.579 (the "no tide reading this hour, use
    the cycle average" path, so the multiplier equals wet_fraction exactly).

        single (correct, what `score_feature` must produce):
            round(100 * 0.5082 * 0.579)        = round(29.4248) = 29
        double (the old bug, reintroduced by the reviewer's mutation):
            round(round(100 * 0.5082) * 0.579) = round(51 * 0.579)
                                                = round(29.529)  = 30

    `combine` is monkeypatched to return this exact `raw` deterministically:
    reproducing 0.5082 from real factor inputs would mean reverse-engineering
    nine factors' worth of curve values to four decimal places, which tests
    nothing this arithmetic-precision check needs -- what is under test is
    the ONE line of arithmetic downstream of `combine`, not `combine` itself
    (already covered elsewhere).
    """
    single = int(round(100 * 0.5082 * 0.579))
    double = int(round(int(round(100 * 0.5082)) * 0.579))
    assert (single, double) == (29, 30), "the worked example itself must produce these two numbers"

    import tidescout.engine.score as score_mod

    fixed = HourScore(
        score=51, subs=[], excluded=[], confidence=1.0, constrained_share=1.0,
        provisional=[], raw=0.5082,
    )
    monkeypatch.setattr(score_mod, "combine", lambda subs: fixed)

    red = load_species()["redfish"]
    # No tide_frac/tide_phase -> _flat_wet_multiplier's "no tide reading"
    # fallback, which returns wet_fraction itself as the multiplier.
    metrics = _metrics(type="flat", wet_fraction=0.579, flood_phase=0.1)
    got = score_feature(metrics, _hour(), None, red, salinity=_sal(18.0), thresholds=_T)
    assert got.activation == single == 29
