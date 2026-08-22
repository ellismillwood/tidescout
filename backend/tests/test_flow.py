import numpy as np
import pytest

from tidescout.engine import flow
from tidescout.engine.flow import blend_regimes
from tidescout.models import DischargeBuckets


def test_exact_regime_is_preferred():
    avail = {"mean_med", "spring_high"}
    assert flow.select_regime("spring", "high", avail) == ("spring_high", False)


def test_missing_regime_falls_back_and_flags_it():
    """Spec section 10: a missing state degrades to the nearest, with a warning."""
    name, fell_back = flow.select_regime("spring", "high", {"mean_med"})
    assert fell_back is True
    assert name == "mean_med"


def test_fallback_keeps_the_range_bucket_before_the_discharge_bucket():
    """Substituting discharge is far cheaper than substituting tidal range.

    One range step rescales the whole tidal forcing -- RANGE_FACTORS are
    0.72/1.0/1.28 on a 1.10 m mean range, so ~15 cm of amplitude. One
    discharge step moves domain-mean depth by ~1 cm (the entire low->high
    span is +99.6 m3/s, ~4.45e6 m3 over a cycle against a 6.7e8 m3 domain).
    A plain Manhattan distance calls these equal and then breaks the tie
    alphabetically, which picks `mean_low` over `spring_med` here -- degrading
    the axis that matters roughly fifteen times more.
    """
    name, fell_back = flow.select_regime("spring", "low", {"mean_low", "spring_med"})
    assert fell_back is True
    assert name == "spring_med"


def test_fallback_prefers_one_range_step_over_two_discharge_steps_only_when_forced():
    """With the range bucket unavailable at any discharge, it must still pick
    the nearest range, not give up and take an arbitrary name."""
    name, fell_back = flow.select_regime("spring", "low", {"neap_low", "mean_low"})
    assert (name, fell_back) == ("mean_low", True)  # mean is one range step, neap is two


def test_fallback_is_deterministic():
    """Two genuinely equidistant candidates must resolve the same way every
    call, or a forecast changes between runs for no reason."""
    avail = {"neap_med", "spring_med"}
    first = flow.select_regime("mean", "med", avail)
    assert all(flow.select_regime("mean", "med", avail) == first for _ in range(5))


def test_empty_library_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="empty"):
        flow.select_regime("mean", "med", set())


def test_unknown_bucket_names_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        flow.select_regime("torrential", "med", {"mean_med"})


def test_malformed_regime_names_are_ignored_not_crashed_on():
    """A stray directory in the library must not take the whole lookup down."""
    name, _ = flow.select_regime("mean", "med", {"not-a-regime", "neap_med"})
    assert name == "neap_med"


def test_bracket_phases_wraps_around_the_cycle():
    phases = [0.0, 0.25, 0.5, 0.75]
    lo, hi, w = flow.bracket_phases(phases, 0.9)
    assert (lo, hi) == (3, 0)                 # wraps 0.75 -> 0.0
    assert w == pytest.approx(0.6, abs=1e-6)  # 0.9 is 60% from 0.75 toward 1.0


def test_bracket_phases_exact_hit_lands_on_a_snapshot():
    """Landing exactly on a snapshot must weight it fully. The first bracket
    that contains the phase wins, so 0.5 comes back as the far end of [0.0,0.5]
    (weight 1.0) rather than the near end of the next span -- equivalent, and
    this is the branch the implementation actually takes."""
    lo, hi, w = flow.bracket_phases([0.0, 0.5], 0.5)
    assert (lo, hi) == (0, 1)
    assert w == pytest.approx(1.0)


def test_bracket_phases_handles_a_single_snapshot():
    """A one-phase library is degenerate but must not divide by a zero span."""
    assert flow.bracket_phases([0.3], 0.9) == (0, 0, 0.0)


def test_bracket_phases_rejects_unsorted_phases():
    """`bracket_phases` walks the list in order and treats it as a cycle, so
    unsorted input silently returns the wrong bracket rather than failing."""
    with pytest.raises(ValueError, match="ascending"):
        flow.bracket_phases([0.0, 0.75, 0.25], 0.3)


def test_bracket_phases_accepts_phase_outside_the_unit_interval():
    """Hour 30 of a forecast is phase 2.4 of the cycle; it must wrap."""
    assert flow.bracket_phases([0.0, 0.5], 2.25) == flow.bracket_phases([0.0, 0.5], 0.25)


def test_interpolation_is_on_components_not_direction():
    """Interpolating direction across 0/360 wraps; components must be used."""
    a = {"u": np.array([1.0]), "v": np.array([-0.1]), "depth": np.array([3.0])}
    b = {"u": np.array([1.0]), "v": np.array([0.1]), "depth": np.array([3.0])}
    mid = flow.interpolate_state(a, b, 0.5)
    assert mid["v"][0] == pytest.approx(0.0)
    speed, direction = flow.speed_direction(mid["u"], mid["v"])
    assert direction[0] == pytest.approx(0.0)   # not ~180, which averaging angles gives


def test_interpolation_at_the_endpoints_returns_the_endpoints():
    a = {"u": np.array([1.0]), "v": np.array([2.0]), "depth": np.array([3.0])}
    b = {"u": np.array([9.0]), "v": np.array([8.0]), "depth": np.array([7.0])}
    assert flow.interpolate_state(a, b, 0.0)["u"][0] == pytest.approx(1.0)
    assert flow.interpolate_state(a, b, 1.0)["u"][0] == pytest.approx(9.0)


def test_interpolation_does_not_mutate_its_inputs():
    a = {"u": np.array([1.0]), "v": np.array([2.0]), "depth": np.array([3.0])}
    b = {"u": np.array([9.0]), "v": np.array([8.0]), "depth": np.array([7.0])}
    flow.interpolate_state(a, b, 0.5)
    assert a["u"][0] == pytest.approx(1.0) and b["u"][0] == pytest.approx(9.0)


def test_wet_mask_excludes_the_numerically_dry():
    depth = np.array([0.0, 0.005, 0.5])
    assert list(flow.wet_mask(depth)) == [False, False, True]


def test_known_spots_carry_a_machine_readable_phase_hint():
    """Task 13's gate asserts against `works_on`, so the shipped ground truth
    must actually have it filled -- an unfilled hint makes the go/no-go gate
    silently untestable for that spot."""
    from tidescout.config import load_known_spots

    spots = {s.name: s for s in load_known_spots("winyah-bay")}
    assert spots["Mud Bay Cut"].works_on == "ebb"
    # RESOLVED 2026-08-16: was flood. Notes describe a contrast mechanism
    # (hiding from current), not a peak-current one, so slack branch applies.
    assert spots["Georgetown Lighthouse"].works_on == "slack"
    assert spots["North Jetty"].works_on == "flood"


def test_known_spot_rejects_an_unrecognised_phase_hint():
    """A typo must fail loudly rather than degrade to 'unspecified', which the
    gate treats as 'no expectation to check'."""
    import pydantic

    from tidescout.models import KnownSpot

    with pytest.raises(pydantic.ValidationError):
        KnownSpot(name="x", lon=0.0, lat=0.0, works_on="outgoing")


def test_known_spot_notes_are_left_intact_by_the_hint():
    """The prose stays authoritative -- it carries detail the enum cannot."""
    from tidescout.config import load_known_spots

    spots = {s.name: s for s in load_known_spots("winyah-bay")}
    assert "early incoming" in spots["Georgetown Lighthouse"].notes


def _real_stage_series(n=24):
    """The stage series a real regime.json records, using the shipped config.

    phase is measured from the END of spin-up, and spin_up_h / cycle_h is
    0.4831 of a cycle -- so this deliberately does NOT start at high water.
    """
    import math

    return [
        0.55 * math.cos(2.0 * math.pi * (0.4831 + i / n)) for i in range(n)
    ]


def test_tide_states_finds_flood_on_the_rising_half():
    """Phase 0 is LOW water in this project's convention, so the phases just
    after it are flooding. Getting this backwards inverts the whole gate."""
    states = flow.tide_states(_real_stage_series())
    assert states[3] == "flood"   # a quarter of the way up
    assert states[15] == "ebb"    # past high water, falling


def test_tide_states_marks_the_turning_points_slack():
    states = flow.tide_states(_real_stage_series())
    assert states[0] == "slack"    # low water
    assert states[12] == "slack"   # high water


def test_tide_states_covers_both_halves_of_the_cycle():
    states = flow.tide_states(_real_stage_series())
    assert {"flood", "ebb", "slack"} == set(states)
    # A full cycle spends comparable time flooding and ebbing.
    assert abs(states.count("flood") - states.count("ebb")) <= 2


def test_tide_states_is_derived_from_stage_not_from_phase_order():
    """Reversing the stage series must swap flood and ebb, proving the label
    comes from the water and not from the index."""
    forward = flow.tide_states(_real_stage_series())
    backward = flow.tide_states(list(reversed(_real_stage_series())))
    assert forward.count("flood") == backward.count("ebb")


def test_tide_states_rejects_an_empty_series():
    with pytest.raises(ValueError, match="no boundary stage"):
        flow.tide_states([])


def test_tide_states_handles_a_flat_series():
    """A regime whose boundary never moved is entirely slack, not a crash."""
    assert flow.tide_states([1.0, 1.0, 1.0, 1.0]) == ["slack"] * 4


BUCKETS = DischargeBuckets(low_below_cfs=2774.0, high_above_cfs=6292.0)
ALL = {f"{r}_{d}" for r in ("neap", "mean", "spring") for d in ("low", "med", "high")}


def test_exact_bucket_flow_returns_a_single_regime():
    mix, clamped = blend_regimes("mean", 2774.0, BUCKETS, ALL)
    assert mix == [("mean_low", 1.0)]
    assert clamped is False


def test_midway_flow_blends_the_two_bracketing_buckets():
    """4533 cfs is the med point; 3653 is halfway from low to med."""
    mix, _ = blend_regimes("mean", 3653.5, BUCKETS, ALL)
    assert {r for r, _ in mix} == {"mean_low", "mean_med"}
    assert dict(mix)["mean_low"] == pytest.approx(0.5, abs=0.01)
    assert sum(w for _, w in mix) == pytest.approx(1.0)


def test_blend_never_crosses_the_range_axis():
    """Range is the strong axis -- one step rescales the whole tidal forcing.
    A blend that traded it for discharge would be the exact mistake
    RANGE_STEP_COST=3 exists to prevent."""
    mix, _ = blend_regimes("spring", 5000.0, BUCKETS, ALL)
    assert all(r.startswith("spring_") for r, _ in mix)


def test_flow_above_the_top_bucket_clamps_and_flags():
    """22,996 cfs was observed; 6,292 is the highest ever simulated. The model
    must not be extrapolated 3.7x past anything it was run at."""
    mix, clamped = blend_regimes("mean", 22996.0, BUCKETS, ALL)
    assert mix == [("mean_high", 1.0)]
    assert clamped is True


def test_flow_below_the_bottom_bucket_clamps_and_flags():
    mix, clamped = blend_regimes("mean", 500.0, BUCKETS, ALL)
    assert mix == [("mean_low", 1.0)]
    assert clamped is True


def test_blend_falls_back_when_a_bracketing_regime_is_missing():
    """A partial library must degrade, per spec section 10, not raise."""
    partial = {"mean_low", "mean_high"}
    mix, _ = blend_regimes("mean", 4533.0, BUCKETS, partial)
    assert {r for r, _ in mix} <= partial
    assert sum(w for _, w in mix) == pytest.approx(1.0)


def test_weights_are_never_negative():
    for cfs in (1000.0, 2774.0, 3500.0, 4533.0, 5500.0, 6292.0, 30000.0):
        mix, _ = blend_regimes("mean", cfs, BUCKETS, ALL)
        assert all(w >= 0.0 for _, w in mix)
        assert sum(w for _, w in mix) == pytest.approx(1.0)


def test_blend_falls_back_across_the_range_axis_when_the_range_bucket_has_no_regimes():
    """Basic coverage for the `not order` branch: when `range_bucket` has no
    regime at ANY discharge, blend_regimes cannot blend at all and must fall
    through to select_regime's cross-range fallback rather than crash or
    silently drop the axis.

    mean is one range step from spring; neap is two. RANGE_STEP_COST=3 beats
    even the worst possible discharge mismatch (DISCHARGE_STEP_COST=1 * 2),
    so a `mean_*` regime always wins over any `neap_*` one here regardless of
    which discharge bucket 4533 cfs (the med point) is nearest to -- and
    since it's nearest med itself, mean_med is the unique minimum-cost pick."""
    no_spring = ALL - {"spring_low", "spring_med", "spring_high"}
    mix, clamped = blend_regimes("spring", 4533.0, BUCKETS, no_spring)
    assert mix == [("mean_med", 1.0)]
    assert clamped is True


def test_blend_fallback_breaks_a_range_tie_by_actual_discharge_not_a_fixed_med():
    """Regression for a defect found in review: the total-fallback branch
    once scored every candidate's discharge distance against a hardcoded
    "med" instead of the bucket nearest `cfs`, so two range-adjacent
    candidates that tied on range distance fell through to an alphabetic
    tiebreak instead of the real discharge distance.

    neap_high and spring_low are both exactly one range step from "mean"
    (RANGE_STEP_COST=3 either way) -- a genuine tie that must be broken by
    discharge. cfs=100 sits far below every bucket, so "low" is the correct
    discharge match: scored against the true nearest bucket ("low"), the
    costs are neap_high=3*1+1*2=5 vs spring_low=3*1+1*0=3, and spring_low
    wins outright. Scored against a fixed "med" they would instead tie at
    cost 4 apiece (3*1+1*1 for both), and the name-alphabetical tiebreak
    would hand this to neap_high -- the high-discharge regime, wrong for
    cfs=100 -- which is exactly the bug this pins against regressing."""
    avail = {"neap_high", "spring_low"}
    mix, clamped = blend_regimes("mean", 100.0, BUCKETS, avail)
    assert mix == [("spring_low", 1.0)]
    assert clamped is True
