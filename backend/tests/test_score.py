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


def test_near_fresh_water_penalises_trout_far_more_than_redfish():
    """Spec section 7: the same eddy scores near zero up-bay after a freshet."""
    trout = load_species()["speckled_trout"]
    red = load_species()["redfish"]
    t = _by_factor(score_factors(_hour(), None, trout, salinity=_sal(2.0)))["salinity"]
    r = _by_factor(score_factors(_hour(), None, red, salinity=_sal(2.0)))["salinity"]
    assert t.value < r.value
    assert t.value < 0.3


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


def test_all_nine_factors_are_always_present():
    """Even when every input is missing -- the UI shows nine bars, some greyed."""
    p = load_species()["redfish"]
    subs = score_factors(_hour(), None, p)
    assert len(subs) == 9
    assert len({s.factor for s in subs}) == 9
