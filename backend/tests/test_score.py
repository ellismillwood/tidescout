"""Factor behaviour. Each test isolates one factor and asserts the DIRECTION
of its response, not a calibrated value -- the curves are tuned by hindcasting
and every number in them will move."""

import math

from tidescout.config import load_species
from tidescout.engine.conditions import HourlyConditions
from tidescout.engine.score import FACTORS, score_factors


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
    provisional would pass the test above."""
    from tidescout.engine.score import SalinityProvenance, SalinityReading

    red = load_species()["redfish"]
    reading = SalinityReading(22.0, SalinityProvenance.MEASURED)
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
