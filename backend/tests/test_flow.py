import numpy as np
import pytest

from tidescout.engine import flow


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
