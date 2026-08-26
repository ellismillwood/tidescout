"""Intrusion-model behaviour.

These pin SHAPE, not calibrated values -- the constants are fitted in Task 5 and
will move. Every assertion here must survive recalibration.

The model is a bounded logistic (sigmoid) of the tidally-shifted distance, not
a clipped exponential -- see `engine.salinity`'s module docstring for the
real-data review that found the clipped form made 47.40% of Winyah's domain
read bit-identical salinity across a 19x discharge swing at high water. That
review is why several tests below deliberately probe high water (phase=0.5)
and x < excursion_km -- exactly where the old plateau lived and the new form
must not.
"""

import re

import numpy as np
import pytest
from pydantic import ValidationError

from tidescout.engine import salinity
from tidescout.models import SalinityConfig

CFG = SalinityConfig(
    ocean_ppt=34.0, l0_km=18.0, q0_cfs=4000.0, k=0.33, excursion_km=7.0,
    front_width_km=5.0, calibration_range_cfs=(1232.0, 22996.0),
)


def test_salinity_falls_monotonically_up_the_estuary():
    x = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
    s = salinity.salinity_at(x, cfs=4000.0, phase=0.25, cfg=CFG)
    assert np.all(np.diff(s) < 0)
    assert s[0] == pytest.approx(CFG.ocean_ppt, rel=0.05)


@pytest.mark.parametrize("phase", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
def test_salinity_falls_monotonically_at_every_phase(phase):
    """The clipped exponential this model replaced was only monotone at
    phases where the clip never fired -- at phase 0.5 it went flat (34.0,
    34.0) over the first two points before decaying. Every phase must be
    strictly monotone now, with no clip left to interrupt it."""
    x = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
    s = salinity.salinity_at(x, cfs=4000.0, phase=phase, cfg=CFG)
    assert np.all(np.diff(s) < 0), f"not strictly monotone at phase={phase}: {s}"


def test_higher_discharge_pushes_the_salt_front_seaward():
    """The whole reason salinity cannot read a three-value bucket."""
    x = np.full(1, 15.0)
    low = salinity.salinity_at(x, cfs=2000.0, phase=0.25, cfg=CFG)[0]
    high = salinity.salinity_at(x, cfs=20000.0, phase=0.25, cfg=CFG)[0]
    assert high < low
    assert low - high > 2.0, "a 10x discharge change must move salinity materially"


def test_discharge_sensitivity_survives_at_high_water_near_the_mouth():
    """Reproduces the exact defect a real-data review found: at phase 0.5
    (high water) and x < excursion_km, the old clipped-exponential form
    pinned every such cell to exactly ocean_ppt regardless of discharge --
    North Jetty (2.58 km) was discharge-blind 37.9% of every tidal cycle.
    That bit-identical plateau is the property that must never regress, and
    under the new form it does not: checked here across a grid of near-mouth
    positions and every quarter-phase, spanning the full calibration
    discharge range, zero points are bit-identical (0.00%, against 47.40%
    under the old clipped form).

    Wherever the tidal shift leaves x_eff > 0 (landward of the mouth, not
    ocean-saturated), raising discharge must still measurably freshen the
    water -- the real content of "discharge sensitivity must not regress".

    Seaward of the mouth (x_eff < 0) the model is already within a fraction
    of a ppt of ocean_ppt regardless of flow, by physical construction
    (marine water is marine water there). Measured 2026-08-25 across a
    4,525-point (x, phase) grid over the calibration range: 284 points
    (6.28%) there read a HIGHER discharge as a hair SALTIER instead of
    fresher, by at most 0.0091 ppt (median 0.0074 ppt), always within
    0.03 ppt of ocean_ppt. That is a known, BOUNDED consequence of `W`
    sharing `L`'s exponent (see `_discharge_scale`'s docstring) -- not the
    bit-identical plateau defect 1 was about, since it is never exactly tied
    and never material.

    The bound below is enforced INSIDE the loop, for every x_eff < 0 point
    this grid visits -- not just at one named coordinate. A single hardcoded
    exception (e.g. only at North Jetty) would let the reversal grow
    unboundedly at any other seaward point without failing anything; the
    grid's own worst point (x=5.0, phase=0.5, x_eff=-2.0) is the one that
    actually exercises the bound, not North Jetty."""
    lo, hi = CFG.calibration_range_cfs
    xs = np.array([0.5, 2.58, 5.0, 8.0, 12.0, 20.0, 31.57])
    phases = np.array([0.0, 0.25, 0.5, 0.75])
    # Measured maxima across this exact grid: reversal 0.008646 ppt (at
    # x=5.0, phase=0.5), distance from ocean_ppt 0.014751 ppt (same point).
    # These bounds carry roomy headroom above both so the test is not
    # fragile to float noise, while still catching an order-of-magnitude
    # regression (e.g. a reversal that grew to 0.5 ppt).
    MAX_SATURATED_REVERSAL_PPT = 0.02
    MAX_SATURATED_DISTANCE_FROM_OCEAN_PPT = 0.03

    ties = 0
    saturated_points_checked = 0
    for x in xs:
        for phase in phases:
            x_eff = x + CFG.excursion_km * np.cos(2.0 * np.pi * phase)
            low = float(salinity.salinity_at(np.array([x]), cfs=lo, phase=phase, cfg=CFG)[0])
            high = float(salinity.salinity_at(np.array([x]), cfs=hi, phase=phase, cfg=CFG)[0])
            if low == high:
                ties += 1
            if x_eff > 0:
                assert high < low, (
                    f"x={x}, phase={phase}, x_eff={x_eff:.2f} (landward of the mouth): "
                    "higher discharge must still freshen the cell"
                )
            else:
                # Seaward of the mouth: a reversal is sanctioned, but ONLY
                # within the measured bound -- enforced for every such point
                # the grid visits, so a regression cannot hide at a
                # coordinate this loop happens not to name.
                saturated_points_checked += 1
                assert low != high, (
                    f"x={x}, phase={phase}, x_eff={x_eff:.2f}: must never be "
                    "bit-identical, even in the saturated exception"
                )
                assert abs(high - low) < MAX_SATURATED_REVERSAL_PPT, (
                    f"x={x}, phase={phase}, x_eff={x_eff:.2f}: reversal "
                    f"{high - low:.6f} ppt exceeds the known-bounded exception -- "
                    "this is no longer the saturated corner the ruling accepted"
                )
                assert (
                    CFG.ocean_ppt - high < MAX_SATURATED_DISTANCE_FROM_OCEAN_PPT
                    and CFG.ocean_ppt - low < MAX_SATURATED_DISTANCE_FROM_OCEAN_PPT
                ), (
                    f"x={x}, phase={phase}, x_eff={x_eff:.2f}: reading strayed "
                    "outside the saturated band near ocean_ppt"
                )
    assert ties == 0, (
        "no position/phase may read bit-identical salinity across the full discharge range"
    )
    assert saturated_points_checked > 0, (
        "the grid must include at least one x_eff < 0 point to exercise the bound above"
    )

    # North Jetty at high water: the headline example of the saturated
    # exception (defect 1's original reproduction case), kept as a named,
    # documentary pin -- x=2.58, phase=0.5 is already one of the points the
    # loop above visits and bounds; this does not add new enforcement, only
    # a reader-facing anchor for the historical defect this test traces to.
    x_eff = 2.58 + CFG.excursion_km * np.cos(2.0 * np.pi * 0.5)
    assert x_eff < 0, "this exception is sanctioned only seaward of the mouth"


def test_intrusion_length_shrinks_as_a_power_law_in_discharge():
    assert salinity.intrusion_length_km(CFG.q0_cfs, CFG) == pytest.approx(CFG.l0_km)
    doubled = salinity.intrusion_length_km(2 * CFG.q0_cfs, CFG)
    assert doubled == pytest.approx(CFG.l0_km * 2 ** (-CFG.k), rel=1e-6)
    assert doubled < CFG.l0_km


# -- Task 1: the front's width scales with discharge -------------------------
# `L(Q)` ranges 37.14 km -> 1.13 km across the observed 257x discharge span
# while `front_width_km` was a constant, so the front could not be sharp at
# high flow and broad at low flow at once. Measured at FIXED distance (so it
# cannot be confounded with position), the mean residual trended monotonically
# with flow: -1.33 -> -3.72 ppt at x=16.68 km, +1.33 -> -2.03 at x=19.03 km.
# Making the width carry the same scaling as the length cuts that trend
# spread from 2.96 to 0.55 ppt.


def test_front_width_scales_down_as_discharge_rises():
    """A constant width cannot be sharp at high flow and broad at low flow.
    Measured consequence of the old form: the residual trended -1.33 -> -3.72
    ppt across flow quintiles at a FIXED distance."""
    from tidescout.engine.salinity import front_width_at

    low = front_width_at(1_000.0, CFG)
    high = front_width_at(100_000.0, CFG)
    assert high < low
    assert high < 0.25 * low, "a 100x discharge range must sharpen the front substantially"


def test_front_width_equals_the_authored_value_at_the_reference_discharge():
    """`front_width_km` now means 'the front's width at q0_cfs'. At exactly
    q0 the scaling is 1.0, so the authored number must come back unchanged --
    that is what keeps the config field readable."""
    from tidescout.engine.salinity import front_width_at

    assert front_width_at(CFG.q0_cfs, CFG) == pytest.approx(CFG.front_width_km)


def test_width_and_length_share_one_discharge_scaling():
    """If these ever read different exponents the front's shape stops meaning
    anything, and no output would look wrong. Pinned as a ratio so the test
    survives any future change to the scaling itself."""
    from tidescout.engine.salinity import front_width_at, intrusion_length_km

    for cfs in (500.0, 4_000.0, 30_000.0, 200_000.0):
        ratio = front_width_at(cfs, CFG) / intrusion_length_km(cfs, CFG)
        assert ratio == pytest.approx(CFG.front_width_km / CFG.l0_km)


def test_intrusion_length_is_unchanged_by_the_refactor():
    """`intrusion_length_km` is reimplemented on the shared helper. Its VALUES
    must not move -- l0_km's fitted meaning depends on them."""
    from tidescout.engine.salinity import intrusion_length_km

    for cfs, expected in ((1_000.0, CFG.l0_km * (1_000.0 / CFG.q0_cfs) ** -CFG.k),
                          (CFG.q0_cfs, CFG.l0_km),
                          (50_000.0, CFG.l0_km * (50_000.0 / CFG.q0_cfs) ** -CFG.k)):
        assert intrusion_length_km(cfs, CFG) == pytest.approx(expected)


def test_a_sharper_front_at_high_flow_drops_salinity_faster_with_distance():
    """The point of the change, isolated from `L(Q)` shrinking with
    discharge -- which already predated this task and is, by itself, enough
    to make a low-vs-high-discharge comparison LOOK sharper over a fixed x
    window. Confirmed: hand-building a constant-width control at the same
    two discharges below (L still scales with Q, W frozen at
    `front_width_km`) also satisfies `hi[-1] < lo[-1]` with both drops
    positive -- the ORIGINAL, unisolated version of this test could not fail
    on the thing it named.

    This version holds discharge -- and therefore `L(Q)` and `x_eff` --
    FIXED, and compares the real, discharge-scaled width against a
    constant-width control at that SAME discharge. With `L` and `x`
    identical between the two, any extra drop-off is attributable to the
    width term alone, checked at both a low and a high discharge so the
    isolation is not an artefact of picking one flow.

    NOTE the extra drop comes from opposite mechanisms at the two flows,
    and only the high one is literally "sharper" -- this test's name
    describes that case, not both. At 60,000 cfs the scaled width (2.046
    km) is genuinely narrower than the 5.0 km control and the front really
    is sharper. At 2,000 cfs -- BELOW `q0_cfs` -- the scaled width is
    WIDER (6.285 km), and the x window (2-14 km, against L = 22.626 km)
    sits entirely on the sigmoid's landward tail: there the narrower
    control saturates toward `ocean_ppt` and goes nearly flat, while the
    wider real curve is still transitioning, so the wider front is the one
    that drops more. Both directions are the width term doing the work,
    which is what is asserted; neither is `L` moving, since `L` is pinned
    per-discharge above."""
    x = np.array([2.0, 6.0, 10.0, 14.0])
    phase = 0.25

    for cfs in (2_000.0, 60_000.0):
        length = salinity.intrusion_length_km(cfs, CFG)
        x_eff = x + CFG.excursion_km * np.cos(2.0 * np.pi * phase)
        scaled_width = salinity.front_width_at(cfs, CFG)

        real = CFG.ocean_ppt * 0.5 * (1.0 - np.tanh((x_eff - length) / scaled_width))
        control = CFG.ocean_ppt * 0.5 * (1.0 - np.tanh((x_eff - length) / CFG.front_width_km))
        # `real` must be exactly what production `salinity_at` computes, not
        # a reimplementation that could silently drift from it.
        assert np.allclose(real, salinity.salinity_at(x, cfs, phase, CFG))

        real_drop = float(real[0] - real[-1])
        control_drop = float(control[0] - control[-1])
        assert real_drop > control_drop, (
            f"cfs={cfs}: at the SAME discharge (same L, same x_eff), the "
            "discharge-scaled width must fall off faster than a constant-width "
            "control -- the width term itself, not L moving, must be doing the work"
        )


def test_high_water_is_saltier_than_low_water_at_the_same_place():
    """Phase 0 is LOW water here. Inverting this inverts the tidal salinity
    swing over the whole bay, which is exactly the trap Plan 3 documented."""
    x = np.full(1, 12.0)
    at_low = salinity.salinity_at(x, cfs=4000.0, phase=0.0, cfg=CFG)[0]
    at_high = salinity.salinity_at(x, cfs=4000.0, phase=0.5, cfg=CFG)[0]
    assert at_high > at_low


def test_tidal_swing_peaks_at_high_water_and_troughs_at_low_water():
    """The salt field slides; it does not teleport. `x_eff` is a monotonic
    function of phase over each half-cycle (cos is monotone on [0, pi] and
    [pi, 2pi]) and salinity is monotonic in x_eff, so the extremes of a full
    phase sweep must land exactly at phase 0.5 (high water) and phase 0.0
    (low water) -- nowhere else, and the swing must equal exactly the
    difference between those two evaluations. This holds for any monotone
    form, not just this one, so it survives a future change of shape too."""
    x = np.full(1, 12.0)
    phases = np.linspace(0, 1, 240, endpoint=False)
    swing = np.array(
        [salinity.salinity_at(x, cfs=4000.0, phase=p, cfg=CFG)[0] for p in phases]
    )
    at_high = salinity.salinity_at(x, cfs=4000.0, phase=0.5, cfg=CFG)[0]
    at_low = salinity.salinity_at(x, cfs=4000.0, phase=0.0, cfg=CFG)[0]
    assert swing.max() == pytest.approx(at_high, abs=1e-6)
    assert swing.min() == pytest.approx(at_low, abs=1e-6)
    assert swing.max() - swing.min() == pytest.approx(at_high - at_low, abs=1e-6)


def test_salinity_never_exceeds_the_ocean_end_member():
    """True by construction now -- tanh's range is [-1, 1], so S sits in
    [0, ocean_ppt] with no clip anywhere in the implementation. Checked at
    x=0 specifically (not just x>=0) since the old clipped form saturated to
    exactly ocean_ppt there, hiding a discharge-blind bug behind a
    trivially-true bound; this asserts the tighter property directly."""
    x = np.array([0.0, 0.5, 1.0])
    for phase in (0.0, 0.25, 0.5, 0.75):
        s = salinity.salinity_at(x, cfs=1232.0, phase=phase, cfg=CFG)
        assert np.all(s < CFG.ocean_ppt), f"phase={phase}: {s} touched or exceeded ocean_ppt"


def test_salinity_is_never_negative_far_up_river():
    s = salinity.salinity_at(np.array([200.0]), cfs=22996.0, phase=0.5, cfg=CFG)
    assert s[0] >= 0.0


def test_unreachable_cells_stay_nan():
    """NaN distance means no water route to the sea; it must not become 34 ppt."""
    s = salinity.salinity_at(np.array([np.nan, 10.0]), cfs=4000.0, phase=0.25, cfg=CFG)
    assert np.isnan(s[0]) and np.isfinite(s[1])


def test_field_flags_discharge_outside_the_calibration_range():
    """Silent extrapolation is the failure mode this model is most prone to:
    it returns a confident number for a flow nothing was ever fitted against."""
    x = np.array([10.0])
    inside = salinity.salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG)
    outside = salinity.salinity_field(x, cfs=40000.0, phase=0.25, cfg=CFG)
    assert inside.extrapolated is False
    assert outside.extrapolated is True
    assert outside.cfs == 40000.0


def test_field_flags_the_calibration_boundary_inclusively():
    """Both ends of calibration_range_cfs are trusted, not flagged -- a
    discharge reading of exactly the boundary is not "outside" it."""
    x = np.array([10.0])
    lo, hi = CFG.calibration_range_cfs
    assert salinity.salinity_field(x, cfs=lo, phase=0.25, cfg=CFG).extrapolated is False
    assert salinity.salinity_field(x, cfs=hi, phase=0.25, cfg=CFG).extrapolated is False
    assert salinity.salinity_field(x, cfs=lo - 1e-6, phase=0.25, cfg=CFG).extrapolated is True
    assert salinity.salinity_field(x, cfs=hi + 1e-6, phase=0.25, cfg=CFG).extrapolated is True


def test_zero_discharge_does_not_divide_by_zero():
    s = salinity.salinity_at(np.array([10.0]), cfs=0.0, phase=0.25, cfg=CFG)
    assert np.isfinite(s[0])


def test_negative_discharge_does_not_divide_by_zero():
    s = salinity.salinity_at(np.array([10.0]), cfs=-500.0, phase=0.25, cfg=CFG)
    assert np.isfinite(s[0])
    field = salinity.salinity_field(np.array([10.0]), cfs=-500.0, phase=0.25, cfg=CFG)
    assert field.extrapolated is True
    assert field.cfs == 1.0, "cfs must report the floored value it was evaluated at"


def test_nan_discharge_propagates_as_nan():
    """A missing discharge reading must read as 'unknown', not silently become
    the 1 cfs floor -- `max(cfs, 1.0)`'s argument order is what preserves
    this, and a tidy-looking reorder would swap it silently."""
    s = salinity.salinity_at(np.array([10.0]), cfs=float("nan"), phase=0.25, cfg=CFG)
    assert np.isnan(s[0])
    length = salinity.intrusion_length_km(float("nan"), CFG)
    assert np.isnan(length)


# -- SalinityConfig bounds validation --------------------------------------
# Task 5's fit is an unconstrained optimizer; these guard against it landing
# somewhere the model's own invariants (S in [0, ocean_ppt]) silently break.


def test_nonpositive_ocean_ppt_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(ocean_ppt=0.0)


def test_negative_l0_km_is_rejected():
    """Verified before this guard existed: l0_km=-18 produced 59.26 ppt,
    above ocean_ppt, with no error anywhere."""
    with pytest.raises(ValidationError):
        SalinityConfig(l0_km=-18.0)


def test_zero_l0_km_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(l0_km=0.0)


def test_nonpositive_q0_cfs_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(q0_cfs=0.0)


def test_negative_k_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(k=-0.1)


def test_zero_k_is_allowed():
    """k=0 means a discharge-independent front -- degenerate, but not the
    divide/root blowup that negative k or a zero length scale causes."""
    SalinityConfig(k=0.0)


def test_nonpositive_excursion_km_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(excursion_km=0.0)


def test_nonpositive_front_width_km_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(front_width_km=0.0)


def test_reversed_calibration_range_is_rejected():
    """A reversed range would make `not (lo <= cfs <= hi)` true for every
    discharge -- extrapolated would never be False again."""
    with pytest.raises(ValidationError):
        SalinityConfig(calibration_range_cfs=(22996.0, 1232.0))


def test_degenerate_calibration_range_is_rejected():
    with pytest.raises(ValidationError):
        SalinityConfig(calibration_range_cfs=(4000.0, 4000.0))


def test_unknown_salinity_key_is_rejected():
    """A typo'd YAML key under `salinity:` must fail loudly, not be silently
    ignored -- especially since the YAML block duplicates these defaults
    value-for-value, so Task 5 has to keep both in lockstep by hand."""
    with pytest.raises(ValidationError):
        SalinityConfig(oceon_ppt=34.0)


# -- Fitting the model to observations (Task 5) -----------------------------
# The interesting output of a fit is not the numbers -- least squares always
# returns numbers -- but whether the data could constrain them. These tests
# pin both: that a correct fit recovers parameters from data the model itself
# generated, and that the diagnostics say so loudly when it cannot.

from tidescout.pipeline.salinity_fit import fit_intrusion  # noqa: E402


def _synthetic_obs(cfg, distances, flows):
    """Observations generated BY the model, so a correct fit must recover it.

    Phase 0.25 matches the fit's own convention: the tidal term is exactly
    zero there, which is what makes these daily-mean-shaped observations.
    """
    return [
        (d, q, float(salinity.salinity_at(np.array([d]), q, 0.25, cfg)[0]))
        for d in distances
        for q in flows
    ]


def _synthetic_swings(cfg, distances, flows):
    """High-water minus low-water salinity, also generated by the model."""
    return [
        (
            d,
            q,
            float(
                salinity.salinity_at(np.array([d]), q, 0.5, cfg)[0]
                - salinity.salinity_at(np.array([d]), q, 0.0, cfg)[0]
            ),
        )
        for d in distances
        for q in flows
    ]


TRUTH = SalinityConfig(
    ocean_ppt=34.0, l0_km=16.0, q0_cfs=4000.0, k=0.40, excursion_km=7.0,
    front_width_km=5.0,
)


def test_fit_recovers_known_parameters_from_dense_observations():
    obs = _synthetic_obs(TRUTH, [2.0, 8.0, 15.0, 25.0, 35.0], [2000.0, 6000.0, 15000.0])
    # Every free parameter starts away from truth -- front_width_km included.
    # Starting W at 5.0 would have recovered 5.0 nearly by construction and
    # demonstrated nothing about whether the optimizer can find it.
    fitted, diag = fit_intrusion(
        obs, cfg=TRUTH.model_copy(update={"l0_km": 25.0, "k": 0.2, "front_width_km": 12.0})
    )
    assert fitted.l0_km == pytest.approx(16.0, rel=0.05)
    assert fitted.k == pytest.approx(0.40, rel=0.10)
    assert fitted.front_width_km == pytest.approx(5.0, rel=0.05)
    assert diag["rmse_ppt"] < 0.1


def test_fit_warns_when_no_observation_sits_in_the_middle_of_the_gradient():
    """Winyah's real situation: an ocean anchor and a river anchor, nothing
    between. The fit will look excellent and constrain nothing where it
    matters."""
    obs = _synthetic_obs(TRUTH, [0.5, 40.0], [3000.0, 9000.0])  # ends only
    _, diag = fit_intrusion(obs, cfg=TRUTH)
    assert diag["n_interior_obs"] == 0
    assert "interior" in diag["warning"].lower()


def test_fit_refuses_to_run_on_a_single_discharge():
    """k is the response to discharge. One flow cannot constrain it, and a fit
    that returns a number anyway is worse than one that declines."""
    obs = _synthetic_obs(TRUTH, [2.0, 10.0, 25.0], [4000.0])
    with pytest.raises(ValueError, match="discharge"):
        fit_intrusion(obs, cfg=TRUTH)


def test_fit_records_the_discharge_span_as_the_calibration_range():
    obs = _synthetic_obs(TRUTH, [2.0, 10.0, 25.0], [2500.0, 11000.0])
    fitted, diag = fit_intrusion(obs, cfg=TRUTH)
    assert fitted.calibration_range_cfs == (2500.0, 11000.0)
    assert diag["cfs_span"] == (2500.0, 11000.0)


def test_fit_recovers_the_tidal_excursion_from_a_swing_target():
    """`excursion_km` is a free parameter (Task 3's finding: held at 7.0 it
    implies a 22-29 ppt tidal swing at North Jetty and Mud Bay Cut, which is
    unphysical). It is recoverable ONLY from a swing target -- see the
    companion test below for why."""
    truth = TRUTH.model_copy(update={"excursion_km": 3.0})
    distances = [2.0, 8.0, 15.0, 25.0, 35.0]
    flows = [2000.0, 6000.0, 15000.0]
    fitted, diag = fit_intrusion(
        _synthetic_obs(truth, distances, flows),
        cfg=truth.model_copy(update={"l0_km": 25.0, "k": 0.2, "excursion_km": 7.0}),
        swings=_synthetic_swings(truth, distances, flows),
    )
    assert fitted.excursion_km == pytest.approx(3.0, rel=0.05)
    assert fitted.l0_km == pytest.approx(16.0, rel=0.05)
    assert diag["n_swing_obs"] == len(distances) * len(flows)
    assert diag["rmse_ppt"] < 0.1


def test_fit_holds_the_excursion_and_says_so_when_there_is_no_swing_target():
    """The trap this exists to catch: at the daily-mean phase the tidal term
    is EXACTLY zero, so a spatial-only objective has identically zero
    gradient in `excursion_km`. An optimizer handed it as a free parameter
    returns the starting value unchanged and it reads as 'fitted'."""
    obs = _synthetic_obs(TRUTH, [2.0, 8.0, 15.0, 25.0, 35.0], [2000.0, 6000.0])
    start = TRUTH.model_copy(update={"excursion_km": 6.25})
    fitted, diag = fit_intrusion(obs, cfg=start)
    assert fitted.excursion_km == 6.25
    assert "excursion" in diag["warning"].lower()
    assert diag["n_swing_obs"] == 0
    assert "excursion_km" not in diag["param_sigma"]


def test_fit_warns_when_every_observation_sits_at_one_distance():
    """Winyah's ACTUAL situation, measured: both USGS 00480 sites lie outside
    the model domain and snap to the same cell, so every real observation
    carries one along-estuary distance. A profile fitted at one point cannot
    constrain a profile."""
    obs = _synthetic_obs(TRUTH, [31.57], [1500.0, 4000.0, 12000.0, 20000.0])
    _, diag = fit_intrusion(obs, cfg=TRUTH)
    assert diag["n_distinct_distances"] == 1
    assert diag["distance_span_km"] == 0.0
    assert "distance" in diag["warning"].lower()


def test_fit_warns_when_the_discharge_span_is_too_narrow():
    """Two flows 6% apart satisfy the 'more than one discharge' rule and still
    cannot constrain a power law in discharge."""
    obs = _synthetic_obs(TRUTH, [2.0, 10.0, 25.0], [4000.0, 4250.0])
    _, diag = fit_intrusion(obs, cfg=TRUTH)
    assert "discharge span" in diag["warning"].lower()


def _noisy_obs(cfg, distances, flows, sigma_ppt=0.3, seed=7):
    """Synthetic observations with a little measurement noise.

    The noise is load-bearing, not decoration. With EXACT data the fit's
    residual variance is zero, so every parameter's 1-sigma comes back 0.000
    however degenerate the design -- measured, on the single-distance case
    below. That is precisely why `condition_number` (a property of WHICH
    distances and discharges were observed) is reported beside `param_sigma`
    (which converts it into ppt using the residual actually seen).
    """
    rng = np.random.default_rng(seed)
    return [
        (
            d,
            q,
            float(salinity.salinity_at(np.array([d]), q, 0.25, cfg)[0])
            + float(rng.normal(0.0, sigma_ppt)),
        )
        for d in distances
        for q in flows
    ]


def test_fit_reports_tight_uncertainty_when_the_data_constrains_the_model():
    obs = _noisy_obs(
        TRUTH, [2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 35.0], [1500.0, 4000.0, 12000.0]
    )
    fitted, diag = fit_intrusion(obs, cfg=TRUTH.model_copy(update={"l0_km": 20.0}))
    assert diag["condition_number"] < 100.0
    for name, sigma in diag["param_sigma"].items():
        value = getattr(fitted, name)
        assert sigma is not None, f"{name} has no uncertainty estimate"
        assert sigma < 0.1 * abs(value), f"{name}: sigma {sigma} vs value {value}"


def test_fit_reports_useless_uncertainty_when_the_data_does_not():
    """The same machinery and the same noise, at one distance instead of
    seven. Measured: condition number 1.09e7, 1-sigma on l0_km of 709 km
    against a fitted 31.3 km -- an uncertainty 22x the length of the estuary
    it describes -- and `front_width_km` driven onto its lower bound. This is
    Winyah's real design: both USGS 00480 sites snap to the same cell."""
    obs = _noisy_obs(TRUTH, [31.57], [1500.0, 4000.0, 12000.0, 20000.0])
    fitted, diag = fit_intrusion(obs, cfg=TRUTH)
    assert diag["condition_number"] > 1.0e6
    loose = [
        n
        for n, s in diag["param_sigma"].items()
        if s is None or s > abs(getattr(fitted, n))
    ]
    assert loose, f"expected an unconstrained parameter, got {diag['param_sigma']}"
    assert "unconstrained" in diag["warning"].lower()


def test_fit_never_returns_a_config_the_model_rejects():
    """The optimizer is the one caller that can write a SalinityConfig
    nobody reviewed. Fed an inverted profile (fresh at the mouth, ocean at the
    head) it must still return a config that revalidates -- `model_copy`
    would NOT, since it skips validation."""
    obs = [
        (0.5, 2000.0, 0.0), (0.5, 12000.0, 0.2),
        (40.0, 2000.0, 34.0), (40.0, 12000.0, 33.5),
    ]
    fitted, _ = fit_intrusion(obs, cfg=TRUTH)
    SalinityConfig(**fitted.model_dump())  # raises if the fit left it invalid
    assert fitted.l0_km > 0 and fitted.front_width_km > 0 and fitted.k >= 0


def test_fit_drops_non_finite_observations_and_counts_them():
    """A dark sensor arrives as NaN. Feeding it to least_squares makes every
    residual NaN and the fit returns the starting guess, looking successful."""
    obs = _synthetic_obs(TRUTH, [2.0, 10.0, 25.0], [2500.0, 11000.0])
    dirty = [*obs, (12.0, 5000.0, float("nan")), (float("nan"), 5000.0, 12.0)]
    fitted, diag = fit_intrusion(dirty, cfg=TRUTH.model_copy(update={"l0_km": 22.0}))
    assert diag["n_obs"] == len(obs)
    assert diag["n_dropped"] == 2
    assert fitted.l0_km == pytest.approx(16.0, rel=0.05)


def test_fit_refuses_an_empty_observation_set():
    with pytest.raises(ValueError, match="at least"):
        fit_intrusion([], cfg=TRUTH)


# -- rmse split by observation source ----------------------------------------
# Measured 2026-08-24 on the real Winyah run: NERRS daily means (n=10,880)
# rmse 4.061 ppt, WQP grab samples (n=1,860) rmse 6.102 ppt -- two
# populations that residualise very differently must not hide behind one
# headline number.


def test_fit_reports_rmse_split_by_source():
    distances = [2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    flows = [1500.0, 4000.0, 12000.0]
    clean = _synthetic_obs(TRUTH, distances, flows)
    biased = [(d, q, y + 3.0) for d, q, y in clean]
    obs = clean + biased
    sources = ["clean"] * len(clean) + ["biased"] * len(biased)

    _, diag = fit_intrusion(
        obs, cfg=TRUTH.model_copy(update={"l0_km": 20.0}), sources=sources
    )

    by_source = diag["rmse_by_source_ppt"]
    assert set(by_source) == {"clean", "biased"}
    assert by_source["biased"] > by_source["clean"]


def test_fit_omits_rmse_by_source_when_no_sources_are_given():
    obs = _synthetic_obs(TRUTH, [2.0, 8.0, 15.0], [2000.0, 6000.0])
    _, diag = fit_intrusion(obs, cfg=TRUTH)
    assert diag["rmse_by_source_ppt"] == {}


def test_fit_rejects_a_sources_list_the_wrong_length():
    obs = _synthetic_obs(TRUTH, [2.0, 8.0, 15.0], [2000.0, 6000.0])
    with pytest.raises(ValueError, match="same length"):
        fit_intrusion(obs, cfg=TRUTH, sources=["only_one"])


def test_fit_rmse_by_source_matches_a_hand_computed_group_rmse():
    """Not just "different from each other" -- the split rmse must equal the
    rmse of exactly the rows tagged with that source, computed the same way
    the headline `rmse_ppt` is."""
    distances = [2.0, 10.0, 25.0]
    flows = [2000.0, 8000.0]
    a = _synthetic_obs(TRUTH, distances, flows)
    b = [(d, q, y - 1.5) for d, q, y in a]
    fitted, diag = fit_intrusion(
        a + b, cfg=TRUTH.model_copy(update={"l0_km": 20.0}),
        sources=["a"] * len(a) + ["b"] * len(b),
    )
    resid_b = [
        y - float(salinity.salinity_at(np.array([d]), q, 0.25, fitted)[0]) for d, q, y in b
    ]
    expected_b = float(np.sqrt(np.mean(np.square(resid_b))))
    assert diag["rmse_by_source_ppt"]["b"] == pytest.approx(expected_b, rel=1e-9)


# -- per-row tidal phase ------------------------------------------------------
# Task 3: 56 of the fit's 58 distinct along-estuary distances come from
# instantaneous WQP grabs, every one of which was previously scored at
# FIT_PHASE as though it were a tidal average. `phases` carries each row's
# own phase instead.


def test_phases_default_to_fit_phase_and_reproduce_todays_behaviour():
    """The backward-compatibility guarantee. Every existing caller passes no
    phases; all of them must keep getting exactly what they get today."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    a, da = fit_intrusion(obs, cfg=CFG)
    b, db = fit_intrusion(obs, cfg=CFG, phases=[salinity_fit.FIT_PHASE] * 3)
    assert da["rmse_ppt"] == pytest.approx(db["rmse_ppt"], rel=0, abs=0)
    assert a.l0_km == pytest.approx(b.l0_km, rel=0, abs=0)


def test_a_phase_actually_changes_the_residual():
    """If phase were ignored, these two fits would be identical -- which is
    exactly the bug this task exists to fix."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    _, flat = fit_intrusion(obs, cfg=CFG, phases=[0.25, 0.25, 0.25])
    _, tidal = fit_intrusion(obs, cfg=CFG, phases=[0.0, 0.5, 0.0])
    assert flat["rmse_ppt"] != pytest.approx(tidal["rmse_ppt"])


def test_phases_length_mismatch_raises():
    """Mirrors the contract `sources` already holds."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    with pytest.raises(ValueError, match="phases"):
        fit_intrusion(obs, cfg=CFG, phases=[0.25, 0.5])


def test_phases_is_validated_against_observations_not_swings():
    """The two sequences are different lengths in the real fit (12,725 vs
    10,865). Validating against the wrong one would misalign every phase."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    swings = [(5.0, 4000.0, 8.0)]
    fitted, _ = fit_intrusion(obs, cfg=CFG, swings=swings, phases=[0.1, 0.2, 0.3])
    assert fitted is not None  # 3 phases for 3 observations, 1 swing -- valid


# -- Assembling real observations -------------------------------------------

from datetime import UTC, date, datetime  # noqa: E402

from tidescout.config import load_fishery  # noqa: E402
from tidescout.pipeline import salinity_fit  # noqa: E402


def _bracketing_tide_events(start: datetime, end: datetime) -> list:
    """Alternating hi/lo events spanning [start, end], every ~6h12m -- enough
    to bracket any timestamp in that range so `phase_at` resolves rather than
    returning `None`. For tests exercising the WQP phase-lookup PATH itself,
    not `phase_at`'s own edge cases (covered in `test_tides.py`)."""
    from datetime import timedelta

    from tidescout.engine.tides import TideEvent

    events = []
    t = start
    kind = "L"
    while t <= end:
        events.append(TideEvent(t, kind, 3.0 if kind == "H" else 0.5))
        kind = "H" if kind == "L" else "L"
        t += timedelta(hours=6, minutes=12)
    return events


def test_composite_discharge_drops_days_a_gauge_is_dark():
    """A dark Pee Dee gauge would otherwise arrive as a 78% drop in river
    flow, which the model reads as the salt front driving up the bay -- a
    fabricated freshet recovery, fitted as though it were real."""
    fishery = load_fishery("winyah-bay")
    sites = [r.usgs_site for r in fishery.rivers]
    d1, d2 = date(2026, 5, 1), date(2026, 5, 2)
    daily = {sites[0]: [(d1, 3000.0), (d2, 3200.0)],
             sites[1]: [(d1, 500.0), (d2, 520.0)],
             sites[2]: [(d1, 300.0)]}  # dark on d2
    out = salinity_fit.composite_discharge_by_day(fishery, daily)
    assert out == {d1: 3800.0}


def test_composite_discharge_is_empty_when_a_gauge_never_reports():
    fishery = load_fishery("winyah-bay")
    sites = [r.usgs_site for r in fishery.rivers]
    daily = {sites[0]: [(date(2026, 5, 1), 3000.0)]}
    assert salinity_fit.composite_discharge_by_day(fishery, daily) == {}


def test_pair_daily_means_keeps_only_days_with_a_discharge():
    d1, d2 = date(2026, 5, 1), date(2026, 5, 2)
    obs = salinity_fit.pair_daily_means(
        {"A": [(d1, 4.0), (d2, 5.0)]}, {d1: 3800.0}, {"A": 31.57}
    )
    assert obs == [(31.57, 3800.0, 4.0)]


def test_daily_swings_drops_partial_days():
    """A day with a handful of readings understates the range, which would
    drag the fitted excursion down without changing the residual much."""
    day = date(2026, 5, 1)
    full = [(datetime(2026, 5, 1, 0, 0, tzinfo=UTC), 2.0 + (i % 5)) for i in range(48)]
    thin = [(datetime(2026, 5, 2, 0, 0, tzinfo=UTC), 9.0)]
    swings = salinity_fit.daily_swings(
        {("A", "00480"): full + thin}, "00480", {day: 3800.0, date(2026, 5, 2): 3800.0},
        {"A": 31.57},
    )
    assert swings == [(31.57, 3800.0, 4.0)]


def test_daily_swings_ignores_other_parameters():
    """`fetch_series` keys by (site, param); a temperature series arriving in
    the same dict must not be read as a salinity swing."""
    day = date(2026, 5, 1)
    temp = [(datetime(2026, 5, 1, 0, 0, tzinfo=UTC), 28.0 + (i % 3)) for i in range(48)]
    assert salinity_fit.daily_swings(
        {("A", "00010"): temp}, "00480", {day: 3800.0}, {"A": 31.57}
    ) == []


def test_fit_flags_a_parameter_that_lands_on_its_optimizer_bound():
    """Found on the real Winyah run, where `k` came back as exactly 2.0 -- its
    upper bound -- with a 1-sigma of 0.82 and a condition number of 46, so
    every numerical health check read as healthy. The bound stopped it, not
    the evidence, so it is not a fitted value and must not be presented as
    one. Reproduced here with the same shape of data: one distance, salinity
    collapsing to zero over the discharge range."""
    obs = [
        (31.57, 1300.0, 8.0), (31.57, 1600.0, 6.0), (31.57, 2000.0, 3.0),
        (31.57, 2600.0, 1.0), (31.57, 3400.0, 0.0), (31.57, 6000.0, 0.0),
        (31.57, 11000.0, 0.0),
    ]
    fitted, diag = fit_intrusion(obs, cfg=CFG)
    assert diag["at_bounds"], "a parameter ran to its bound and was not flagged"
    for name in diag["at_bounds"]:
        lo, hi = salinity_fit._BOUNDS[name]
        value = getattr(fitted, name)
        assert min(abs(value - lo), abs(value - hi)) < 1e-3, f"{name}={value}"
    assert "optimizer bound" in diag["warning"].lower()


def test_fit_reports_no_bound_hits_on_a_well_posed_problem():
    obs = _synthetic_obs(TRUTH, [2.0, 8.0, 15.0, 25.0, 35.0], [2000.0, 6000.0, 15000.0])
    _, diag = fit_intrusion(obs, cfg=TRUTH.model_copy(update={"l0_km": 25.0}))
    assert diag["at_bounds"] == []
    assert "optimizer bound" not in diag["warning"].lower()


# -- The `fitted` marker ----------------------------------------------------
# `extrapolated` answers "was this DISCHARGE in range". Nothing answered "did
# any observation ever constrain these parameters", which is currently False
# for every cell in Winyah Bay -- so a caller checking only `extrapolated`
# saw green on numbers carrying no observational signal at all.


def test_config_defaults_to_unfitted():
    """The default must be the pessimistic one. A theoretical config that
    reads as calibrated is the exact failure this flag exists to prevent."""
    assert SalinityConfig().fitted is False


def test_the_shipped_winyah_config_is_marked_unfitted():
    """Task 5 ran the calibration and declined to write its output. If this
    ever flips to True, a fit was accepted -- check it raised no warning."""
    assert load_fishery("winyah-bay").salinity.fitted is False


def test_field_carries_the_fitted_flag_beside_extrapolated():
    x = np.array([10.0])
    unfitted = salinity.salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG)
    assert unfitted.fitted is False
    assert unfitted.extrapolated is False, "in-range discharge, but still unfitted"
    ok = salinity.salinity_field(
        x, cfs=4000.0, phase=0.25, cfg=CFG.model_copy(update={"fitted": True})
    )
    assert ok.fitted is True


def test_the_fitted_flag_changes_no_computed_value():
    """It is metadata riding alongside the numbers, nothing more."""
    x = np.array([2.58, 9.47, 20.0, 31.57])
    a = salinity.salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG)
    b = salinity.salinity_field(
        x, cfs=4000.0, phase=0.25, cfg=CFG.model_copy(update={"fitted": True})
    )
    assert np.array_equal(a.ppt, b.ppt)
    assert a.cfs == b.cfs and a.extrapolated == b.extrapolated


def test_a_fit_that_raises_any_warning_is_not_marked_fitted():
    """Winyah's real shape: one distance. The residual is fine and every
    numerical health check passes; the flag must still say False."""
    obs = _synthetic_obs(TRUTH, [31.57], [1500.0, 4000.0, 12000.0, 20000.0])
    fit, diag = fit_intrusion(obs, cfg=TRUTH)
    assert diag["warning"] != ""
    assert fit.fitted is False


def test_a_clean_fit_is_marked_fitted():
    """The positive case, so the flag is not just permanently False: enough
    observations, spanning the gradient, over a wide discharge range, with a
    swing target to constrain the excursion."""
    distances = [2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    flows = [1500.0, 4000.0, 12000.0]
    fit, diag = fit_intrusion(
        _synthetic_obs(TRUTH, distances, flows),
        cfg=TRUTH.model_copy(update={"l0_km": 20.0}),
        swings=_synthetic_swings(TRUTH, distances, flows),
    )
    assert diag["warning"] == "", f"unexpected warning: {diag['warning']}"
    assert fit.fitted is True


def test_the_interior_count_is_named_as_value_based_in_the_warning():
    """36 'interior observations' on a spatially degenerate set is the
    misleading combination, and diagnostics get read alone."""
    obs = _synthetic_obs(TRUTH, [12.0], [1500.0, 4000.0, 12000.0, 20000.0])
    _, diag = fit_intrusion(obs, cfg=TRUTH)
    assert diag["n_interior_obs"] > 0
    assert "n_interior_obs" in diag["warning"]
    assert "blind to this" in diag["warning"]


# -- Goodness of fit is part of the `fitted` gate ---------------------------
# `fitted = not warning` originally had NO residual component: a config could
# miss every observation by a seventh of the ocean-to-fresh range and still be
# certified. Rejecting "clean residual" as a SUFFICIENT bar was right; dropping
# it as a NECESSARY one was not.


def test_resolution_recovers_the_quantum_of_integer_observations():
    """Winyah's 348 daily means take the integers 0..10. The derived
    resolution must come back as exactly the 1 ppt quantum -- this is the
    threshold, so it is measured, not authored."""
    assert salinity_fit.observation_resolution_ppt(range(11)) == 1.0
    assert salinity_fit.observation_resolution_ppt([0.0, 0.0, 5.0, 10.0]) == 5.0


def test_resolution_is_undefined_below_two_distinct_values():
    assert np.isnan(salinity_fit.observation_resolution_ppt([3.0, 3.0, 3.0]))
    assert np.isnan(salinity_fit.observation_resolution_ppt([]))


def test_a_fit_that_reproduces_nothing_is_not_marked_fitted():
    """A per-site bias the model cannot represent. The design is otherwise
    ideal -- 7 distances, 3 flows, well conditioned, nothing on a bound -- so
    every non-residual check passes and only the residual catches it."""
    distances = [2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    flows = [1500.0, 4000.0, 12000.0]
    bias = {2.0: -6.0, 6.0: 6.0, 20.0: 5.0, 35.0: 4.0}
    obs = [(d, q, y + bias.get(d, 0.0)) for d, q, y in _synthetic_obs(TRUTH, distances, flows)]
    fitted, diag = fit_intrusion(
        obs,
        cfg=TRUTH.model_copy(update={"l0_km": 20.0}),
        swings=_synthetic_swings(TRUTH, distances, flows),
    )
    assert diag["rmse_ppt"] > 1.0
    assert "poor fit" in diag["warning"].lower()
    assert fitted.fitted is False, "a config reproducing nothing was certified"


def test_a_legitimately_noisy_fit_is_still_marked_fitted():
    """The other side of the threshold: real data has noise, and a fit inside
    what the observations can resolve must not be failed for it."""
    distances = [2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    flows = [1500.0, 4000.0, 12000.0]
    obs = _noisy_obs(TRUTH, distances, flows, sigma_ppt=0.3)
    fitted, diag = fit_intrusion(
        obs,
        cfg=TRUTH.model_copy(update={"l0_km": 20.0}),
        swings=_synthetic_swings(TRUTH, distances, flows),
    )
    assert 0.0 < diag["rmse_ppt"] < 1.0
    assert "poor fit" not in diag["warning"].lower()
    assert fitted.fitted is True


# -- Site admission: the reason must be the real one ------------------------

NAN = float("nan")


@pytest.mark.parametrize(
    "rows,located,dist,gap,used,expect",
    [
        ([(date(2026, 5, 1), 4.0)], True, 10.0, 12.0, True, ""),
        ([(date(2026, 5, 1), 4.0)], True, 10.0, 900.0, False, "outside the domain"),
        # An unlocated site is never QUERIED, so it arrives with no rows. The
        # old order tested `not rows` first and reported "no salinity
        # history" -- a data-availability claim about a site whose problem
        # is that nobody knows where it is.
        ([], False, NAN, float("inf"), False, "no coordinates"),
        ([], True, 10.0, 12.0, False, "no salinity history"),
        ([(date(2026, 5, 1), 4.0)], True, NAN, 12.0, False, "no water route"),
    ],
)
def test_site_record_names_the_real_reason(rows, located, dist, gap, used, expect):
    r = salinity_fit.build_site_record(
        "X", rows, located=located, distance_km=dist, snap_gap_m=gap, max_snap_m=500.0
    )
    assert r.used is used
    assert expect in r.note
    assert r.n_days == len(rows)


def test_site_record_reports_the_observed_ppt_range():
    rows = [(date(2026, 5, 1), 0.0), (date(2026, 5, 2), 8.0), (date(2026, 5, 3), 3.0)]
    r = salinity_fit.build_site_record(
        "X", rows, located=True, distance_km=31.57, snap_gap_m=5.0, max_snap_m=500.0
    )
    assert r.ppt_range == (0.0, 8.0)


# -- Task 7: per-station bias against a fitted config ------------------------
# The gate report needs a bias/rmse figure per admitted station, not just the
# one headline rmse. `Observation` carries a distance, not a station id, so
# `station_bias` groups admitted `SiteRecord`s by their exact distance_km --
# the same value that built `observations` in the first place.


def test_station_bias_matches_a_hand_computed_residual():
    site = salinity_fit.build_site_record(
        "A", [(date(2026, 5, 1), 10.0)], located=True, distance_km=5.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [(5.0, 4000.0, 9.0), (5.0, 4000.0, 11.0)]
    out, dropped = salinity_fit.station_bias([site], obs, TRUTH)

    assert dropped == 0
    assert len(out) == 1
    b = out[0]
    assert b.sites == ("A",)
    assert b.distance_km == 5.0
    assert b.n == 2
    predicted = float(
        salinity.salinity_at(np.array([5.0]), 4000.0, salinity_fit.FIT_PHASE, TRUTH)[0]
    )
    expected = [predicted - 9.0, predicted - 11.0]
    assert b.mean_residual_ppt == pytest.approx(float(np.mean(expected)))
    assert b.rmse_ppt == pytest.approx(float(np.sqrt(np.mean(np.square(expected)))))


def test_station_bias_drops_non_finite_rows_and_counts_them():
    """`station_bias` must apply the same `_finite_rows` filter
    `fit_intrusion` itself runs before scoring. A NaN reaching it unfiltered
    would put a bare `nan` in `mean_residual_ppt`/`rmse_ppt` -- visually
    indistinguishable, in a printed table, from a station that simply fits
    badly. The two bad rows here (a NaN salinity, a NaN discharge) must be
    excluded from the computation AND counted, not silently dropped."""
    site = salinity_fit.build_site_record(
        "A", [(date(2026, 5, 1), 10.0)], located=True, distance_km=5.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [
        (5.0, 4000.0, 9.0),
        (5.0, 4000.0, float("nan")),
        (5.0, float("nan"), 11.0),
    ]
    out, dropped = salinity_fit.station_bias([site], obs, TRUTH)

    assert dropped == 2
    assert len(out) == 1
    assert out[0].n == 1
    assert np.isfinite(out[0].mean_residual_ppt)
    assert np.isfinite(out[0].rmse_ppt)


def test_station_bias_combines_stations_sharing_one_distance():
    """WYSS1 and NIWWBWQ, the real surface/bottom pair on Winyah, snap to the
    identical 19.03 km. `Observation` cannot tell their rows apart, so they
    must be reported as one combined entry rather than silently double-
    counted or arbitrarily assigned to whichever name sorts first."""
    a = salinity_fit.build_site_record(
        "NIWWBWQ", [(date(2026, 5, 1), 10.0)], located=True, distance_km=19.03,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    b = salinity_fit.build_site_record(
        "WYSS1", [(date(2026, 5, 1), 10.0)], located=True, distance_km=19.03,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [(19.03, 4000.0, 8.0), (19.03, 9000.0, 5.0)]
    out, dropped = salinity_fit.station_bias([a, b], obs, TRUTH)

    assert dropped == 0
    assert len(out) == 1
    assert out[0].sites == ("NIWWBWQ", "WYSS1")
    assert out[0].n == 2


def test_station_bias_omits_unused_stations_and_stations_with_no_matching_rows():
    used = salinity_fit.build_site_record(
        "A", [(date(2026, 5, 1), 10.0)], located=True, distance_km=5.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    # Outside the domain -> used=False; must not appear even though it is
    # "in" `sites`.
    excluded = salinity_fit.build_site_record(
        "B", [], located=True, distance_km=31.0, snap_gap_m=9000.0, max_snap_m=500.0
    )
    # Admitted (used=True) but nothing in `observations` sits at its
    # distance -- e.g. a station whose only history is swings, not levels.
    no_rows = salinity_fit.build_site_record(
        "C", [(date(2026, 5, 1), 4.0)], located=True, distance_km=25.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [(5.0, 4000.0, 9.0)]
    out, dropped = salinity_fit.station_bias([used, excluded, no_rows], obs, TRUTH)

    assert dropped == 0
    assert [b.sites for b in out] == [("A",)]


def test_station_bias_is_sorted_by_distance():
    far = salinity_fit.build_site_record(
        "FAR", [(date(2026, 5, 1), 4.0)], located=True, distance_km=20.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    near = salinity_fit.build_site_record(
        "NEAR", [(date(2026, 5, 1), 4.0)], located=True, distance_km=5.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [(20.0, 4000.0, 4.0), (5.0, 4000.0, 9.0)]
    out, dropped = salinity_fit.station_bias([far, near], obs, TRUTH)

    assert dropped == 0
    assert [b.distance_km for b in out] == [5.0, 20.0]


def test_station_bias_scores_each_row_at_its_own_phase():
    """A WQP-style grab must be scored at ITS OWN resolved tidal phase, the
    same `phases` contract `fit_intrusion` takes -- not silently defaulted to
    the daily-mean FIT_PHASE. `TRUTH.excursion_km` is 7.0 (nonzero), so
    `salinity_at`'s tidal term (`excursion_km * cos(2*pi*phase)`) is exactly
    zero at FIT_PHASE=0.25 but nonzero at phase=0.0 (low water) -- if
    `station_bias` ignored `phases` and always scored at FIT_PHASE (the bug
    this test pins against), this row's residual would come out equal to the
    FIT_PHASE prediction minus the observation, not the low-water one, and
    the assertion below would fail.
    """
    site = salinity_fit.build_site_record(
        "WQP1", [(date(2026, 5, 1), 10.0)], located=True, distance_km=5.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [(5.0, 4000.0, 9.0)]
    low_water_phase = 0.0
    out, dropped = salinity_fit.station_bias([site], obs, TRUTH, phases=[low_water_phase])

    assert dropped == 0
    assert len(out) == 1
    predicted_low_water = float(
        salinity.salinity_at(np.array([5.0]), 4000.0, low_water_phase, TRUTH)[0]
    )
    predicted_fit_phase = float(
        salinity.salinity_at(np.array([5.0]), 4000.0, salinity_fit.FIT_PHASE, TRUTH)[0]
    )
    # Guard against a degenerate test: if these two predictions happened to
    # be equal, the assertion below would pass whether or not `phases` were
    # honoured, and the test would not be able to fail on the bug it names.
    assert predicted_low_water != pytest.approx(predicted_fit_phase)
    assert out[0].mean_residual_ppt == pytest.approx(predicted_low_water - 9.0)


def test_station_bias_default_phases_reproduces_fit_phase_behaviour():
    """No `phases` argument at all (the pre-Task-3 call shape) must still
    score every row at FIT_PHASE, exactly as before -- existing callers that
    never carried phase are unaffected by this fix."""
    site = salinity_fit.build_site_record(
        "A", [(date(2026, 5, 1), 10.0)], located=True, distance_km=5.0,
        snap_gap_m=1.0, max_snap_m=500.0,
    )
    obs = [(5.0, 4000.0, 9.0)]
    out, dropped = salinity_fit.station_bias([site], obs, TRUTH)

    assert dropped == 0
    predicted = float(
        salinity.salinity_at(np.array([5.0]), 4000.0, salinity_fit.FIT_PHASE, TRUTH)[0]
    )
    assert out[0].mean_residual_ppt == pytest.approx(predicted - 9.0)


# -- The CLI refusal path ---------------------------------------------------


def test_calibrate_refuses_and_names_every_rejected_site(monkeypatch):
    """The refusal is the DEFAULT outcome on this fishery, so its message is
    the most-read output this command has."""
    from typer.testing import CliRunner

    from tidescout.cli import app

    rejected = [
        salinity_fit.build_site_record(
            "021108125", [], located=True, distance_km=31.57,
            snap_gap_m=9497.907, max_snap_m=500.0,
        ),
        salinity_fit.build_site_record(
            "02110815", [(date(2026, 5, 1), 4.0)], located=True, distance_km=31.57,
            snap_gap_m=1362.043, max_snap_m=500.0,
        ),
    ]
    monkeypatch.setattr(
        salinity_fit,
        "collect_observations",
        lambda *a, **k: salinity_fit.CalibrationInput([], [], rejected, 90, None, 90),
    )
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    assert result.exit_code == 1
    # Rich wraps to the terminal width and would split these phrases across
    # lines; collapse whitespace so the assertions test content, not layout.
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "CANNOT CALIBRATE" in out
    assert "2 of 2" in out
    assert "021108125" in out and "02110815" in out
    # ceil, not round: admission is `gap <= max_snap_m`, so 9497.907 must
    # suggest 9498 and never 9497.
    assert "--max-snap-m 9498" in out
    assert "fitted" in out


# -- Reading the NERRS store into the fit -----------------------------------
# The store holds 15-minute readings; the fit wants daily means (and daily
# swings, which are what free `excursion_km`). Aggregating them is pure and
# is tested as such -- the network and the sqlite file stay in
# `collect_observations`.


def _series(day, hours, value_fn, tz_offset=0):
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=tz_offset))
    return [
        (datetime(day.year, day.month, day.day, 0, 0, tzinfo=tz) + timedelta(hours=h),
         value_fn(h))
        for h in hours
    ]


def test_daily_means_and_swings_aggregates_a_full_day():
    from zoneinfo import ZoneInfo

    day = date(2026, 5, 1)
    # 96 readings, a clean 10 ppt swing about a mean of 20.
    series = _series(day, [i * 0.25 for i in range(96)],
                     lambda h: 20.0 + 5.0 * np.sin(2 * np.pi * h / 12.42),
                     tz_offset=-4)

    means, swings = salinity_fit.daily_means_and_swings(
        series, ZoneInfo("America/New_York"), min_readings=40
    )

    assert means[day] == pytest.approx(20.0, abs=0.2)
    assert swings[day] == pytest.approx(10.0, abs=0.2)


def test_a_thin_day_yields_neither_a_mean_nor_a_swing():
    """A partial day understates the RANGE, and -- because these readings
    are tidal -- it also biases the MEAN, by up to the full swing if the few
    readings happen to land on one phase. Winyah's measured daily swing is
    11.9 ppt median, so that bias is larger than the whole signal the fit is
    trying to resolve. Both outputs are gated on the same count."""
    from zoneinfo import ZoneInfo

    day = date(2026, 5, 1)
    series = _series(day, [0.0, 0.25, 0.5], lambda h: 30.0, tz_offset=-4)

    means, swings = salinity_fit.daily_means_and_swings(
        series, ZoneInfo("America/New_York"), min_readings=40
    )

    assert day not in means
    assert day not in swings


def test_days_are_local_not_utc():
    """The store keeps UTC. Grouping on the UTC date would put the four
    hours after 20:00 local into the NEXT day, splitting every day's tidal
    cycle across two means and pairing them with the wrong day's discharge.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # 2026-05-02 01:00 UTC is 2026-05-01 21:00 in New York (EDT, UTC-4).
    series = [(datetime(2026, 5, 2, 1, 0, tzinfo=UTC), 12.0)]

    means, _ = salinity_fit.daily_means_and_swings(
        series, ZoneInfo("America/New_York"), min_readings=1
    )

    assert list(means) == [date(2026, 5, 1)]


def test_store_stations_are_read_and_off_axis_ones_are_skipped(monkeypatch):
    """`collect_observations` must read the declared store stations, and
    must not read the ones marked off_axis -- that flag is the only thing
    standing between the fit and a second estuary's worth of 32 ppt data."""
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")
    asked: list[str] = []

    class _Store:
        def salinity_series(self, station):
            asked.append(station)
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 10.0 + (i % 8)) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {date(2026, 5, 1): 4000.0, date(2026, 5, 2): 4200.0}, [], {}),
    )
    # NERRS/USGS daily means always score at FIT_PHASE regardless of tide
    # events (see the comment in `collect_observations`); this test has no
    # WQP fixture that needs a resolvable phase, so an empty events list is
    # enough to avoid the real `tide_events_range` hitting `cache=None`.
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    assert set(asked) == {"WYSS1", "NIWWBWQ", "NIWTAWQ"}
    assert "NIWCBWQ" not in asked and "NIWOLWQ" not in asked and "NIWDCWQ" not in asked
    assert data.observations, "store observations must reach the fit"


def test_off_axis_stations_are_still_reported_as_sites(monkeypatch):
    """Excluded is not invisible. A station dropped from the fit must still
    appear in the site table with the REASON, the same contract
    `build_site_record` already holds for out-of-domain USGS sites --
    otherwise the fit silently narrows and the report looks complete."""
    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")

    class _Store:
        def salinity_series(self, station):
            return []

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    monkeypatch.setattr(salinity_fit, "_usgs_inputs", lambda *a, **k: ({}, {}, [], {}))

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)
    notes = {r.site: r.note for r in data.sites}

    for station in ("NIWCBWQ", "NIWOLWQ", "NIWDCWQ"):
        assert station in notes, f"{station} must still be reported"
        assert "off" in notes[station].lower() or "branch" in notes[station].lower()


def test_store_station_with_no_known_position_reports_no_coordinates_not_off_axis(monkeypatch):
    """A store sensor absent from `cdmo.NIW_STATION_COORDS_LONLAT` has no
    surveyed position at all. Before this guard, `is_off_axis` read the
    resulting NaN stem distance as "off the axis" -- the wrong REASON: NaN
    there means "nobody knows where this sonde is," not "a real branch the
    coordinate cannot place." `located` must win, the same guard the WQP
    path already applies over `wqp_known` (see `wqp_off_axis`). Not
    reachable on Winyah today (every declared store station has a surveyed
    position) but simulated here for the stamp-out fisheries the spec names
    (Charleston, Awendaw, Murrells Inlet), whose sondes will not all be in
    that table."""
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")

    class _Store:
        def salinity_series(self, station):
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 10.0 + (i % 8)) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    # No known position for ANY store station -- simulates a fishery whose
    # sondes are not in `cdmo.NIW_STATION_COORDS_LONLAT`.
    monkeypatch.setattr(salinity_fit, "_store_coords", lambda sites: {})
    monkeypatch.setattr(salinity_fit, "_store_distances", lambda slug, fishery, sites: {})
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {date(2026, 5, 1): 4000.0, date(2026, 5, 2): 4200.0}, [], {}),
    )
    monkeypatch.setattr(salinity_fit, "_wqp_sites", lambda slug: {})
    # No WQP fixture here needs a resolvable phase; an empty events list
    # just avoids the real `tide_events_range` hitting `cache=None`.
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)
    notes = {r.site: r.note for r in data.sites}

    # Declared off_axis=False in the YAML, and now unlocatable -- must
    # report "no coordinates", never "off axis".
    for station in ("WYSS1", "NIWWBWQ", "NIWTAWQ"):
        assert "coordinates" in notes[station].lower()
        assert "axis" not in notes[station].lower()


def test_store_station_declared_off_axis_true_and_unlocated_reports_no_coordinates(
    monkeypatch,
):
    """The other half of the truth table the previous test does not cover.
    `store_stem_ok and w.station in store_coords` correctly guards the
    `off_axis: false` case (previous test), but the `else` branch used to
    fall back to `w.off_axis` -- the DECLARED flag -- which can be `True`.
    `NIWCBWQ`/`NIWOLWQ`/`NIWDCWQ` are declared `off_axis: true` in
    `winyah-bay.yaml` (North Inlet, a separate branch); simulated here with
    no surveyed position at all. Before the fix, `is_off_axis` was never
    even consulted -- the bare `w.off_axis` fallback handed `True` straight
    to `build_site_record`, which tests `off_axis` BEFORE `located`, so the
    station reported "off the salt-intrusion axis" instead of the true
    reason: nobody knows where it is. Being unplaceable is a fact about the
    station; the YAML flag is a claim about which branch it sits on, and
    that claim cannot be asserted for a station whose position is unknown.
    Not reachable on Winyah today (every declared store station has a
    surveyed position) but simulated here for the stamp-out fisheries the
    spec names (Charleston, Awendaw, Murrells Inlet)."""
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")

    class _Store:
        def salinity_series(self, station):
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 32.0 + (i % 8)) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(salinity_fit, "_store_coords", lambda sites: {})
    monkeypatch.setattr(salinity_fit, "_store_distances", lambda slug, fishery, sites: {})
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {date(2026, 5, 1): 4000.0, date(2026, 5, 2): 4200.0}, [], {}),
    )
    monkeypatch.setattr(salinity_fit, "_wqp_sites", lambda slug: {})
    # No WQP fixture here needs a resolvable phase; an empty events list
    # just avoids the real `tide_events_range` hitting `cache=None`.
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)
    notes = {r.site: r.note for r in data.sites}

    # Declared off_axis=True in the YAML, and now unlocatable -- must
    # report "no coordinates", never "off axis".
    for station in ("NIWCBWQ", "NIWOLWQ", "NIWDCWQ"):
        assert "coordinates" in notes[station].lower()
        assert "axis" not in notes[station].lower()


def test_calibrate_reports_both_sources_not_just_the_usgs_window(monkeypatch):
    """The header claimed "00480 sensors over the last N days". Once the
    NERRS store contributes its full held history that is wrong twice over:
    these are not all 00480 sensors, and the store's decade is not `--days`
    long. A report that misstates its own provenance is worse than a terse
    one -- this output is what gets pasted into the fishery YAML."""
    from typer.testing import CliRunner

    from tidescout.cli import app

    records = [
        salinity_fit.build_site_record(
            "WYSS1", [(date(2016, 1, 1), 6.0), (date(2026, 8, 22), 7.0)],
            located=True, distance_km=19.03, snap_gap_m=5.0, max_snap_m=500.0,
        ),
        salinity_fit.build_site_record(
            "NIWCBWQ", [], located=True, distance_km=12.88, snap_gap_m=4.0,
            max_snap_m=500.0, off_axis=True,
        ),
    ]
    monkeypatch.setattr(
        salinity_fit,
        "collect_observations",
        lambda *a, **k: salinity_fit.CalibrationInput(
            [(19.03, 4000.0, 6.0), (16.68, 9000.0, 3.0), (19.03, 9000.0, 2.0)],
            [],
            records,
            90,
            (date(2016, 1, 1), date(2026, 8, 22)),
            90,
            observation_sources=["nerrs", "nerrs", "nerrs"],
        ),
    )
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))

    assert "00480 sensors over the last 90 days" not in out
    assert "NERRS store" in out
    assert "2016-01-01 .. 2026-08-22" in out


# -- Task 5: the computed stem screen, and WQP anchors ----------------------
# The COMPUTED distance to the main stem decides `off_axis`; the YAML's
# declared flag becomes an override that can only ever EXCLUDE, never
# include (see `is_off_axis`'s own docstring for why the other direction
# would reintroduce the hand-marking this replaces).


def test_off_axis_is_computed_from_the_stem_distance():
    """Measured 2026-08-24: North Inlet's stations sit 7.997-11.918 km from
    the main stem and the bay's own sit 0.651-1.393."""
    assert salinity_fit.is_off_axis(stem_km=9.858, declared=False) is True
    assert salinity_fit.is_off_axis(stem_km=0.651, declared=False) is False


def test_the_yaml_flag_can_only_exclude_never_include():
    """A hand flag that could force a station back IN would reintroduce the
    hand-marking this replaces. One that can only exclude is a safety valve."""
    assert salinity_fit.is_off_axis(stem_km=0.05, declared=True) is True
    assert salinity_fit.is_off_axis(stem_km=9.9, declared=False) is True


def test_a_station_with_no_stem_distance_is_excluded_not_admitted():
    """NaN means the cell has no water route to the stem at all. Admitting
    it would put an unplaceable station into the fit."""
    assert salinity_fit.is_off_axis(stem_km=float("nan"), declared=False) is True


def test_stem_command_reports_missing_distance_field_instead_of_a_traceback(monkeypatch):
    """`build_stem_distance_field` reads the along-estuary field via
    `load_distance_field`, which raises `FileNotFoundError` (with its own
    "run `tidescout salinity field` first" guidance) until `salinity field`
    has run. `salinity calibrate` already catches this and prints it in red
    rather than a raw traceback; `salinity stem` did not."""
    from typer.testing import CliRunner

    from tidescout.cli import app
    from tidescout.pipeline import estuary

    def _raise(slug, fishery):
        raise FileNotFoundError(
            f"no along-estuary distance field at /fake/estuary_km.npy -- run "
            f"`tidescout salinity field {slug}` first"
        )

    monkeypatch.setattr(estuary, "build_stem_distance_field", _raise)
    result = CliRunner().invoke(app, ["salinity", "stem", "winyah-bay"])

    assert result.exit_code == 1
    # Rich wraps to the terminal width and would split this phrase across
    # lines; collapse whitespace so the assertion tests content, not layout
    # (same technique `test_calibrate_refuses_and_names_every_rejected_site`
    # already uses).
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "tidescout salinity field winyah-bay" in out


def test_station_stem_km_agrees_with_the_built_field():
    """Regression against the real built field, mirroring
    `test_estuary.test_the_six_nerrs_stations_land_on_the_right_side_of_the_screen`
    -- `station_stem_km` must recover exactly the value the field itself
    holds at each station's nearest cell, not a reimplementation that could
    drift from it. Skips rather than fails when the field has not been
    built -- a fresh clone has no data/ directory."""
    from tidescout.pipeline import estuary
    from tidescout.pipeline.flowlib import grid_spec
    from tidescout.sources import cdmo

    fishery = load_fishery("winyah-bay")
    try:
        stem_field = estuary.load_stem_distance_field("winyah-bay")
    except FileNotFoundError:
        pytest.skip("stem field not built -- run `tidescout salinity stem winyah-bay`")

    spec = grid_spec("winyah-bay", fishery)
    stations = ["NIWTAWQ", "WYSS1", "NIWWBWQ", "NIWCBWQ", "NIWOLWQ", "NIWDCWQ"]
    sites = {s: cdmo.NIW_STATION_COORDS_LONLAT[s] for s in stations}

    out = salinity_fit.station_stem_km("winyah-bay", fishery, sites)

    from rasterio.warp import transform as warp_transform

    for station in stations:
        lon, lat = sites[station]
        x, y = (v[0] for v in warp_transform(
            "EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", [lon], [lat]))
        i = int(np.argmin((spec.xs - x) ** 2 + (spec.ys - y) ** 2))
        assert out[station] == pytest.approx(float(stem_field[i]))
    # And the on/off split matches the six-station gate Task 4 pinned.
    for station in ("NIWTAWQ", "WYSS1", "NIWWBWQ"):
        assert out[station] <= estuary.ON_AXIS_MAX_KM
    for station in ("NIWCBWQ", "NIWOLWQ", "NIWDCWQ"):
        assert out[station] > estuary.ON_AXIS_MAX_KM


def test_stem_field_missing_falls_back_to_the_declared_flag_not_a_crash(monkeypatch):
    """`load_stem_distance_field` raises `FileNotFoundError` on a machine
    that has not run `salinity stem` yet. That must not turn `salinity
    calibrate` into a hard crash: off_axis for NERRS/NDBC stations falls
    back to the YAML's declared flag (the pre-Task-5 behaviour), and the
    fallback is reported on the returned `CalibrationInput` rather than
    silently swallowed."""
    from datetime import datetime, timedelta

    fishery = load_fishery("winyah-bay")

    class _Store:
        def salinity_series(self, station):
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 10.0 + (i % 8)) for i in range(96)]

    def _raise(*a, **k):
        raise FileNotFoundError("no distance-to-stem field")

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {date(2026, 5, 1): 4000.0, date(2026, 5, 2): 4200.0}, [], {}),
    )
    monkeypatch.setattr(salinity_fit, "station_stem_km", _raise)
    monkeypatch.setattr(salinity_fit, "_wqp_sites", lambda slug: {})
    # No WQP fixture here needs a resolvable phase; an empty events list
    # just avoids the real `tide_events_range` hitting `cache=None`.
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    assert data.stem_field_missing is True
    # WYSS1/NIWWBWQ/NIWTAWQ are declared off_axis=false, NIWCBWQ/NIWOLWQ/
    # NIWDCWQ declared off_axis=true -- with the field unavailable, that
    # declared flag alone must decide, exactly as it did before this task.
    notes = {r.site: r.note for r in data.sites}
    for station in ("WYSS1", "NIWWBWQ", "NIWTAWQ"):
        assert notes[station] == ""
    for station in ("NIWCBWQ", "NIWOLWQ", "NIWDCWQ"):
        assert "axis" in notes[station].lower()
    assert data.observations, "on-axis stations must still reach the fit"


def test_stem_km_or_fallback_distinguishes_a_missing_bathymetry_raster(monkeypatch):
    """`station_stem_km` reads the bathymetry raster too (via `grid_spec` ->
    `read_bathy`), on a machine that has not run `tidescout bathy build`
    either -- a DIFFERENT missing file than the stem field, surfacing as the
    same bare `FileNotFoundError`. Silently degrading for that one too would
    tell the caller to run `tidescout salinity stem`, which calls this exact
    same `grid_spec` and would fail identically -- the wrong remedy. Only
    `load_stem_distance_field`'s own, distinctively-worded error should be
    read as "the stem field itself is missing"."""
    fishery = load_fishery("winyah-bay")

    def _raise_bathy(*a, **k):
        raise FileNotFoundError(
            "[Errno 2] No such file or directory: '.../bathy_meta.json'"
        )

    monkeypatch.setattr(salinity_fit, "station_stem_km", _raise_bathy)

    with pytest.raises(FileNotFoundError) as excinfo:
        salinity_fit._stem_km_or_fallback("winyah-bay", fishery, {"WYSS1": (-79.29, 33.35)})

    assert "bathymetry raster" in str(excinfo.value)
    assert "tidescout bathy build winyah-bay" in str(excinfo.value)


def test_stem_km_or_fallback_still_degrades_for_the_stem_field_itself(monkeypatch):
    """The distinguishing fix above must not break the ORIGINAL fallback --
    `load_stem_distance_field`'s own error still degrades to `({}, False)`,
    not a re-raise."""
    fishery = load_fishery("winyah-bay")

    def _raise_stem(*a, **k):
        raise FileNotFoundError(
            "no distance-to-stem field at /fake/stem_km.npy -- run "
            "`tidescout salinity stem winyah-bay` first"
        )

    monkeypatch.setattr(salinity_fit, "station_stem_km", _raise_stem)

    result = salinity_fit._stem_km_or_fallback(
        "winyah-bay", fishery, {"WYSS1": (-79.29, 33.35)}
    )

    assert result == ({}, False)


def test_calibrate_reports_off_axis_count_and_stem_fallback(monkeypatch):
    """Excluded is not invisible, and neither is a degraded run: the CLI
    must print how many stations the screen removed, and must say so out
    loud when the stem field itself was unavailable rather than leaving a
    machine that has not run `salinity stem` looking like a clean run."""
    from typer.testing import CliRunner

    from tidescout.cli import app

    records = [
        salinity_fit.build_site_record(
            "WYSS1", [(date(2026, 5, 1), 6.0)], located=True, distance_km=19.03,
            snap_gap_m=5.0, max_snap_m=500.0,
        ),
        salinity_fit.build_site_record(
            "NIWCBWQ", [], located=True, distance_km=12.88, snap_gap_m=4.0,
            max_snap_m=500.0, off_axis=True,
        ),
    ]
    monkeypatch.setattr(
        salinity_fit,
        "collect_observations",
        lambda *a, **k: salinity_fit.CalibrationInput(
            [(19.03, 4000.0, 6.0), (16.68, 9000.0, 3.0), (19.03, 9000.0, 2.0)],
            [],
            records,
            90,
            (date(2016, 1, 1), date(2026, 8, 22)),
            90,
            n_off_axis=1,
            stem_field_missing=True,
            observation_sources=["nerrs", "nerrs", "nerrs"],
            n_colocated=2,
        ),
    )
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))

    assert "1 station" in out
    assert "stem field not built" in out.lower()
    assert "2 WQP station" in out
    assert "co-located" in out.lower()


def test_calibration_reports_the_memory_window_and_its_exclusions(monkeypatch):
    """A dropped observation must be visible, not inferred from a smaller n."""
    from typer.testing import CliRunner

    from tidescout.cli import app

    monkeypatch.setattr(
        salinity_fit, "collect_observations",
        lambda *a, **k: salinity_fit.CalibrationInput(
            [(19.03, 4000.0, 6.0), (16.68, 9000.0, 3.0), (19.03, 9000.0, 2.0)],
            [], [], 90, (date(2016, 1, 1), date(2026, 8, 22)), 90,
            n_no_discharge_history=17, memory_days=7.0,
        ),
    )
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "17" in out
    # The WINDOW value, not just the exclusion count -- hardcoding the CLI to
    # a fixed "0 day(s)" and ignoring `data.memory_days` entirely would still
    # pass a test that only checked the exclusion count below.
    assert "memory window: 7 day(s)" in out.lower()
    # A phrase unique to the memory-loss line itself. Plain `"history" in
    # out.lower()` is a no-op here: the site-table header's unrelated
    # "NERRS store: full held history)" text already contains "history", so
    # deleting the entire memory-window line would still leave this
    # assertion passing.
    assert (
        "excluded from the composite discharge series for insufficient "
        "preceding history"
    ) in out.lower()


def test_wqp_stations_enter_as_individual_grab_samples_not_daily_means(monkeypatch):
    """WQP samples are single grabs -- a one-sample day fails the 40-reading
    gate in `daily_means_and_swings`, correctly. They must enter the fit
    directly, each paired with its own day's composite discharge, keeping
    the exact timestamp so it resolves to its own tidal phase."""
    from datetime import datetime

    fishery = load_fishery("winyah-bay")
    d1, d2 = date(2026, 5, 1), date(2026, 5, 2)

    class _NerrsStore:
        def salinity_series(self, station):
            return []

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _NerrsStore())
    monkeypatch.setattr(salinity_fit, "_store_distances", lambda slug, fishery, sites: {})
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {d1: 4000.0, d2: 4200.0}, [], {}),
    )
    monkeypatch.setattr(
        salinity_fit, "_wqp_sites",
        lambda slug: {
            "WB-06": [
                (datetime(2026, 5, 1, 15, 0, tzinfo=UTC), 5.0),
                (datetime(2026, 5, 2, 16, 0, tzinfo=UTC), 6.0),
            ],
            "OFFSTA": [(datetime(2026, 5, 1, 12, 0, tzinfo=UTC), 30.0)],
            "NOCOORD": [(datetime(2026, 5, 1, 12, 0, tzinfo=UTC), 9.0)],
        },
    )
    monkeypatch.setattr(
        "tidescout.sources.wqp.station_coords",
        lambda slug: {"WB-06": (-79.30, 33.30), "OFFSTA": (-79.10, 33.50)},
    )
    monkeypatch.setattr(
        salinity_fit, "site_distances_km",
        lambda slug, fishery, sites: {s: (10.28, 5.0) for s in sites},
    )
    stem = {"WB-06": 0.5, "OFFSTA": 9.9}
    monkeypatch.setattr(
        salinity_fit, "station_stem_km",
        lambda slug, fishery, sites: {s: stem.get(s, float("nan")) for s in sites},
    )
    # Bracketing events so both WB-06 grabs resolve a real phase and survive
    # into `data.observations` -- this test is about which STATIONS/rows
    # reach the fit, not about `phase_at` itself (covered in test_tides.py).
    from tidescout.sources import noaa

    events = _bracketing_tide_events(
        datetime(2026, 4, 28, tzinfo=UTC), datetime(2026, 5, 5, tzinfo=UTC)
    )
    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: events)

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    assert sorted(data.observations) == sorted(
        [(10.28, 4000.0, 5.0), (10.28, 4200.0, 6.0)]
    )
    notes = {r.site: r.note for r in data.sites}
    assert "WB-06" in notes and notes["WB-06"] == ""
    assert "axis" in notes["OFFSTA"].lower()
    assert "coordinates" in notes["NOCOORD"].lower()
    assert data.n_wqp_no_discharge_day == 0
    assert data.n_no_phase == 0
    # Both surviving WQP rows carry a real (non-FIT_PHASE-defaulted) phase.
    assert len(data.observation_phases) == len(data.observations)
    # VALUE assertions, not just length/count: each row's phase must equal
    # `phase_at` computed independently against the SAME events for that
    # row's OWN timestamp -- not merely differ from FIT_PHASE by
    # coincidence, and not silently reintroducible by reverting
    # `obs_phases.append(ph)` back to `obs_phases.append(FIT_PHASE)` in
    # `collect_observations` (that regression would leave `n_no_phase`,
    # `n_wqp_no_discharge_day` and the length check above all unchanged).
    from tidescout.engine.tides import phase_at as _phase_at

    phase_by_row = dict(zip(data.observations, data.observation_phases, strict=True))
    assert phase_by_row[(10.28, 4000.0, 5.0)] == pytest.approx(
        _phase_at(events, datetime(2026, 5, 1, 15, 0, tzinfo=UTC))
    )
    assert phase_by_row[(10.28, 4200.0, 6.0)] == pytest.approx(
        _phase_at(events, datetime(2026, 5, 2, 16, 0, tzinfo=UTC))
    )
    assert phase_by_row[(10.28, 4000.0, 5.0)] != pytest.approx(salinity_fit.FIT_PHASE)
    assert phase_by_row[(10.28, 4200.0, 6.0)] != pytest.approx(salinity_fit.FIT_PHASE)
    assert data.n_wqp_phase_resolved == 2


def test_wqp_rows_with_no_composite_discharge_day_are_counted_not_silent(monkeypatch):
    """`composite_discharge_by_day` requires every river gauge to report a
    day before it appears in `by_day` -- the day any gauge's own record
    starts later than a WQP station's earliest sample, that sample's day is
    simply absent from `by_day`. That drop must be counted, not silent (this
    codebase's reject-and-report rule), and the site table must not claim a
    station contributed a row it did not: `n_days` is the count of rows that
    COULD reach the fit, not the raw series length."""
    from datetime import datetime

    fishery = load_fishery("winyah-bay")
    d1 = date(2026, 5, 1)

    class _NerrsStore:
        def salinity_series(self, station):
            return []

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _NerrsStore())
    monkeypatch.setattr(salinity_fit, "_store_distances", lambda slug, fishery, sites: {})
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {d1: 4000.0}, [], {}),
    )
    monkeypatch.setattr(
        salinity_fit, "_wqp_sites",
        lambda slug: {
            "WB-06": [
                (datetime(2026, 5, 1, 15, 0, tzinfo=UTC), 5.0),
                # 2026-05-03 has no composite discharge under the by_day
                # mock above -- simulates a river gauge whose own record
                # starts later than this station's earliest sample.
                (datetime(2026, 5, 3, 16, 0, tzinfo=UTC), 6.0),
            ],
        },
    )
    monkeypatch.setattr(
        "tidescout.sources.wqp.station_coords",
        lambda slug: {"WB-06": (-79.30, 33.30)},
    )
    monkeypatch.setattr(
        salinity_fit, "site_distances_km",
        lambda slug, fishery, sites: {s: (10.28, 5.0) for s in sites},
    )
    monkeypatch.setattr(
        salinity_fit, "station_stem_km",
        lambda slug, fishery, sites: dict.fromkeys(sites, 0.5),
    )
    # The surviving row (2026-05-01) needs a resolvable phase; the other
    # (2026-05-03) is already dropped for lacking a discharge day, before
    # phase is ever consulted.
    from tidescout.sources import noaa

    events = _bracketing_tide_events(
        datetime(2026, 4, 28, tzinfo=UTC), datetime(2026, 5, 5, tzinfo=UTC)
    )
    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: events)

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    assert data.n_wqp_no_discharge_day == 1
    assert data.n_no_phase == 0
    assert data.observations == [(10.28, 4000.0, 5.0)]
    site = next(r for r in data.sites if r.site == "WB-06")
    assert site.used is True
    assert site.n_days == 1, "the undischarged row must not inflate n_days"
    # VALUE assertion, not just a count: the surviving row's phase must
    # equal `phase_at` computed independently for its own timestamp against
    # the same events -- catches a revert of `obs_phases.append(ph)` back to
    # `obs_phases.append(FIT_PHASE)`, which would leave every assertion
    # above this one unchanged (see the sibling test's comment for why).
    from tidescout.engine.tides import phase_at as _phase_at

    assert data.observation_phases == [
        pytest.approx(_phase_at(events, datetime(2026, 5, 1, 15, 0, tzinfo=UTC)))
    ]
    assert data.observation_phases[0] != pytest.approx(salinity_fit.FIT_PHASE)
    assert data.n_wqp_phase_resolved == 1


# -- Cross-store co-location: WQP-to-STORE only, never WQP-to-WQP -----------
# Reviewer-found (2026-08-24): WQP `21SC60WQ_WQX-WB-08` sits 2 m from
# WYSS1's declared coordinates -- the same physical platform, sampled by a
# different agency, not a second site. Its 15 grab rows double-represent
# WYSS1's own daily means on the same dates, at the same distance and the
# same day's discharge. This must NOT generalise to WQP-to-WQP pairs like
# the legacy/WQX ID split, which are also 0 m apart but hold disjoint
# halves of one record (zero shared (timestamp, value) rows measured across
# all 48 such pairs) -- collapsing those would discard real observations.


def test_colocated_wqp_stations_flags_a_station_near_a_declared_store_station(monkeypatch):
    import rasterio.warp

    from tidescout.config import load_fishery

    # Identity transform: treat the raw coordinates as already being in the
    # projected frame, so the metres math is exact and decoupled from the
    # real CRS -- same technique test_estuary.py uses for its synthetic grids.
    monkeypatch.setattr(rasterio.warp, "transform", lambda f, t, lons, lats: (lons, lats))
    fishery = load_fishery("winyah-bay")
    store_coords = {"WYSS1": (500000.0, 3700000.0)}
    wqp_coords = {"21SC60WQ_WQX-WB-08": (500000.0 + 2.0, 3700000.0)}  # 2 m away

    out = salinity_fit._colocated_wqp_stations(fishery, store_coords, wqp_coords)

    assert out == {"21SC60WQ_WQX-WB-08"}


def test_colocated_wqp_stations_leaves_a_genuinely_separate_station_alone(monkeypatch):
    import rasterio.warp

    from tidescout.config import load_fishery

    monkeypatch.setattr(rasterio.warp, "transform", lambda f, t, lons, lats: (lons, lats))
    fishery = load_fishery("winyah-bay")
    store_coords = {"WYSS1": (500000.0, 3700000.0)}
    wqp_coords = {"FAR-STATION": (500000.0 + 5000.0, 3700000.0)}  # 5 km away

    out = salinity_fit._colocated_wqp_stations(fishery, store_coords, wqp_coords)

    assert out == set()


def test_colocated_wqp_stations_never_flags_a_wqp_to_wqp_pair(monkeypatch):
    """The legacy/WQX ID split puts two WQP stations at 0 m of each other
    holding DISJOINT halves of one record -- this function must only ever
    compare WQP coordinates against DECLARED store coordinates, never WQP
    against WQP, or it would collapse exactly the pair Task 5 was told not
    to."""
    import rasterio.warp

    from tidescout.config import load_fishery

    monkeypatch.setattr(rasterio.warp, "transform", lambda f, t, lons, lats: (lons, lats))
    fishery = load_fishery("winyah-bay")
    store_coords = {"WYSS1": (0.0, 0.0)}  # far from the pair below
    wqp_coords = {
        "21SCSHL-05-24": (600000.0, 3800000.0),
        "21SCSHL_WQX-05-24": (600000.0, 3800000.0),  # identical coords, 0 m apart
    }

    out = salinity_fit._colocated_wqp_stations(fishery, store_coords, wqp_coords)

    assert out == set()


def test_colocated_wqp_station_is_excluded_and_not_double_counted(monkeypatch):
    """End to end through `collect_observations`: a WQP station
    `_colocated_wqp_stations` flags must be reported with the colocation
    reason and its observations must NOT enter the fit alongside the
    declared station's own record -- otherwise one site's reading counts
    twice on the same day, at the same distance, against the same
    discharge."""
    from datetime import datetime, timedelta

    fishery = load_fishery("winyah-bay")
    d1 = date(2026, 5, 1)

    class _NerrsStore:
        def salinity_series(self, station):
            if station != "WYSS1":
                return []
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 12.0) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _NerrsStore())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs",
        lambda *a, **k: ({}, {d1: 4000.0}, [], {}),
    )
    monkeypatch.setattr(
        salinity_fit, "_wqp_sites",
        lambda slug: {"WB08": [(datetime(2026, 5, 1, 15, 0, tzinfo=UTC), 11.69)]},
    )
    monkeypatch.setattr(
        "tidescout.sources.wqp.station_coords",
        lambda slug: {"WB08": (-79.30, 33.36)},
    )
    monkeypatch.setattr(
        salinity_fit, "site_distances_km",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    monkeypatch.setattr(
        salinity_fit, "station_stem_km",
        lambda slug, fishery, sites: dict.fromkeys(sites, 0.5),
    )
    monkeypatch.setattr(
        salinity_fit, "_colocated_wqp_stations",
        lambda fishery, store_coords, wqp_coords: (
            {"WB08"} if "WB08" in wqp_coords else set()
        ),
    )
    # WB08 is excluded as co-located before the phase lookup is ever
    # consulted; an empty events list just avoids the real
    # `tide_events_range` hitting `cache=None`.
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    notes = {r.site: r.note for r in data.sites}
    assert "co-located" in notes["WB08"].lower()
    assert data.n_colocated == 1
    # Only WYSS1's own daily mean enters -- WB08's duplicate grab does not.
    assert data.observations == [(19.03, 4000.0, 12.0)]


# -- Task 6: per-cell coverage on SalinityField ------------------------------
# Coverage states how well OBSERVED a cell's along-estuary position is, as a
# separate axis from `extrapolated` (about the DISCHARGE) and `fitted` (about
# the CONFIG). It is per-cell -- the whole reason it exists is that coverage
# varies along the estuary within a single evaluation -- so unlike those two
# scalar flags it must be array-shaped and aligned elementwise with `.ppt`.


def test_coverage_is_per_cell_and_aligned_with_ppt():
    from tidescout.engine.salinity import salinity_field

    d = np.array([2.0, 6.0, 12.0, 30.0])
    f = salinity_field(d, 4000.0, 0.25, CFG)
    assert f.coverage.shape == f.ppt.shape


def test_a_cell_at_an_observation_reads_measured():
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([5.56]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.MEASURED


def test_a_cell_between_observations_reads_interpolated():
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([8.0]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.INTERPOLATED


def test_a_cell_outside_the_observed_span_reads_extrapolated():
    """The 53 features seaward of North Jetty have no WQP station below
    2.58 km and must come out extrapolated -- that band gains nothing from
    this work and must not claim otherwise."""
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([1.0, 30.0]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.EXTRAPOLATED
    assert cov[1] == Coverage.EXTRAPOLATED


def test_no_observations_makes_everything_extrapolated():
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([5.0, 15.0]), observed_km=[], near_km=1.0)
    assert set(cov) == {Coverage.EXTRAPOLATED}


def test_salinity_field_defaults_to_all_extrapolated_with_no_observed_km():
    """Every existing caller of `salinity_field` does not know what has been
    observed, so it must keep working and get the honest, pessimistic answer
    -- not a crash, and not a silently optimistic default."""
    from tidescout.engine.salinity import Coverage, salinity_field

    x = np.array([2.0, 10.0, 30.0])
    f = salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG)
    assert set(f.coverage) == {str(Coverage.EXTRAPOLATED)}


def test_salinity_field_coverage_dtype_holds_the_longest_labels_exactly():
    """'interpolated' and 'extrapolated' are both exactly 12 characters --
    verify a fixed-width dtype does not truncate either."""
    from tidescout.engine.salinity import Coverage, salinity_field

    x = np.array([5.56, 8.0, 30.0])
    f = salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG, observed_km=[5.56, 10.28])
    assert str(f.coverage[0]) == str(Coverage.MEASURED)
    assert str(f.coverage[1]) == str(Coverage.INTERPOLATED)
    assert str(f.coverage[2]) == str(Coverage.EXTRAPOLATED)


# -- Review fix round (2026-08-24): scalar shape, boundary semantics, and the
# coverage-vs-fitted conflation -----------------------------------------


def test_coverage_matches_ppt_shape_for_a_scalar_distance():
    """`salinity_at` returns a 0-d array for a bare scalar `distance_km` (no
    `atleast_1d` anywhere in it). `classify_coverage` used to force
    `atleast_1d` on its own input, so `f.coverage.shape` came back `(1,)`
    while `f.ppt.shape` came back `()` -- a caller zipping the two arrays
    together for a scalar evaluation got silently misaligned results. Pin
    both the bare-Python-float and the 0-d-numpy-array cases."""
    from tidescout.engine.salinity import salinity_field

    f = salinity_field(10.0, 4000.0, 0.25, CFG, observed_km=[5.56, 10.28])
    assert f.coverage.shape == f.ppt.shape == ()

    f2 = salinity_field(np.asarray(10.0), 4000.0, 0.25, CFG, observed_km=[5.56, 10.28])
    assert f2.coverage.shape == f2.ppt.shape == ()


def test_classify_coverage_scalar_output_still_classifies_correctly():
    """The shape fix must not have changed what a scalar call reports --
    only its shape."""
    from tidescout.engine.salinity import Coverage, classify_coverage

    assert classify_coverage(5.56, observed_km=[5.56, 10.28], near_km=1.0) == Coverage.MEASURED
    assert classify_coverage(8.0, observed_km=[5.56, 10.28], near_km=1.0) == Coverage.INTERPOLATED
    assert classify_coverage(30.0, observed_km=[5.56, 10.28], near_km=1.0) == Coverage.EXTRAPOLATED


def test_a_cell_just_outside_the_span_but_near_an_edge_reads_measured():
    """Pinned as intended, not a bug: MEASURED is checked -- and applied --
    AFTER "inside the span", so it is not a subset of it. A cell 1.0 km
    short of the first observation (5.56 km) sits outside [5.56, 10.28] yet
    is still within `near_km` of that observation, so it reads MEASURED, not
    EXTRAPOLATED. Along a single 1-D coordinate, proximity to an observation
    is what carries information -- the alternative would call a cell 0.1 km
    beyond the span EXTRAPOLATED while calling one 0.9 km inside it
    MEASURED, a discontinuity with no real difference in what is known."""
    from tidescout.engine.salinity import Coverage, classify_coverage

    cov = classify_coverage(np.array([4.56]), observed_km=[5.56, 10.28], near_km=1.0)
    assert cov[0] == Coverage.MEASURED


def test_measured_coverage_does_not_imply_the_config_was_fitted():
    """`coverage=MEASURED` and `fitted=False` are meant to coexist -- that is
    Winyah Bay's real, current situation (78.6% of cells MEASURED, 0% of the
    fishery `fitted`). MEASURED says a raw observation sat near this
    position; it says nothing about whether the model producing this number
    was ever calibrated against anything. A caller that reads only
    `coverage` must not be able to mistake one for the other."""
    from tidescout.engine.salinity import Coverage, salinity_field

    assert CFG.fitted is False
    f = salinity_field(
        np.array([5.56]), 4000.0, 0.25, CFG, observed_km=[5.56, 10.28]
    )
    assert f.coverage[0] == Coverage.MEASURED
    assert f.fitted is False


# -- Spec section 4c: the raw nearest-observation distance beside `coverage` -
# `coverage` is an ordinal derived from this number; the gate report needed
# the number itself ("North Jetty -- nearest observation 1.862 km away") and
# had to compute it in an ad-hoc script because it was never exposed.


def test_nearest_observation_km_matches_a_hand_computed_value():
    from tidescout.engine.salinity import nearest_observation_km

    out = nearest_observation_km(np.array([4.442, 13.052, 2.580]), observed_km=[4.442, 13.052])
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.862)


def test_nearest_observation_km_is_nan_with_no_observations():
    """No 'nearest' of an empty set -- NaN is the honest answer, not 0.0
    (which would falsely claim a cell sits exactly on an observation) or inf
    (which would silently poison arithmetic done on the result)."""
    from tidescout.engine.salinity import nearest_observation_km

    out = nearest_observation_km(np.array([5.0, 15.0]), observed_km=[])
    assert np.all(np.isnan(out))


def test_nearest_observation_km_matches_ppt_shape_for_a_scalar_distance():
    """Same 0-d-scalar contract `classify_coverage`/`coverage` already hold,
    so `SalinityField.nearest_observed_km` stays aligned with `.ppt` even
    for a bare scalar evaluation."""
    from tidescout.engine.salinity import nearest_observation_km, salinity_at

    out = nearest_observation_km(10.0, observed_km=[5.56, 10.28])
    assert out.shape == salinity_at(10.0, 4000.0, 0.25, CFG).shape == ()


def test_salinity_field_carries_nearest_observed_km_beside_coverage():
    from tidescout.engine.salinity import salinity_field

    d = np.array([2.58, 5.56, 13.05])
    f = salinity_field(d, 4000.0, 0.25, CFG, observed_km=[5.56, 10.28])
    assert f.nearest_observed_km.shape == f.ppt.shape == f.coverage.shape
    assert f.nearest_observed_km[0] == pytest.approx(abs(2.58 - 5.56))
    assert f.nearest_observed_km[1] == pytest.approx(0.0)


def test_salinity_field_nearest_observed_km_is_nan_with_no_observed_km():
    """Mirrors `test_salinity_field_defaults_to_all_extrapolated_with_no_
    observed_km`: a caller that does not know what has been observed gets
    the honest, empty-set answer for BOTH the ordinal and its raw
    companion, not a crash and not an optimistic default."""
    from tidescout.engine.salinity import salinity_field

    x = np.array([2.0, 10.0, 30.0])
    f = salinity_field(x, cfs=4000.0, phase=0.25, cfg=CFG)
    assert np.all(np.isnan(f.nearest_observed_km))


# -- Pinning the "one tide station suffices" ruling --------------------------
# The spec's Sec 2 rules that no per-location phase-lag model is needed for
# this 4.4-32.9 km estuary, on the strength of a MEASURED lag of <= 0.011
# phase units at 16.68-19.03 km against station 8662549 -- see the spec's
# Sec 2 ("Up-estuary phase lag is negligible -- one tide station suffices",
# `docs/superpowers/specs/2026-08-24-tidal-phase-and-ocean-endmember-design.md`)
# on why the whole fit uses ONE station's phase; the module docstring for
# `pipeline/salinity_fit.py` never discusses tide stations or phase sourcing
# at all. That ruling is what keeps this work small (the alternative was a
# per-cell lag model across a 36 km domain); nothing before this test
# protected it from silently going false.


def test_up_estuary_tidal_lag_stays_negligible():
    """The spec rules that ONE tide station's phase serves the whole
    estuary, because the measured lag at 16.68-19.03 km is <= 0.011 phase
    units against the ~0.25 error the old code made. That ruling is what
    keeps this work small; this pins it against becoming FALSE -- a future
    change to the tide station, the phase convention, or the interpolation
    that pushes the real lag past this bay's tidal-averaging tolerance.

    Measured 2026-08-24 over 2026-07-01..21 against CO-OPS station 8662549,
    from each NERRS station's own `depth_m` record (its high-water time
    compared against the tide station's predicted high, matched within a
    4-hour window, median lag across 37-39 matched highs per station):
        NIWTAWQ  16.68 km   -2.0 min  (-0.003 phase units)
        WYSS1    19.03 km   +4.0 min  (+0.005 phase units)
        NIWWBWQ  19.03 km   +8.0 min  (+0.011 phase units)

    The 0.05 threshold below is deliberately NOT the measured 0.011: it
    guards the ruling from becoming wrong, not the exact measurement, which
    will vary a little with the window chosen. Tightening it to 0.011 would
    make this test fail on ordinary variation and invite deletion.

    Skips cleanly when the NERRS store or the tide predictions are
    unavailable -- a fresh clone has no `data/` directory, and this asserts
    about real measured geography, like the other real-data tests here."""
    from zoneinfo import ZoneInfo

    from tidescout.errors import SourceUnavailable
    from tidescout.sources import ndbc, noaa
    from tidescout.sources.cache import default_cache

    fishery = load_fishery("winyah-bay")
    store = ndbc.default_store("winyah-bay")
    if not store.stations():
        pytest.skip("NERRS store not present")

    zone = ZoneInfo(fishery.timezone)
    try:
        events = noaa.tide_events_range(
            fishery.stations.tide[0], date(2026, 7, 1), date(2026, 7, 21),
            fishery.timezone, default_cache(),
        )
    except SourceUnavailable:
        pytest.skip("tide predictions unavailable")
    highs = sorted(e.time for e in events if e.kind == "H")

    checked = 0
    for station in ("NIWTAWQ", "WYSS1", "NIWWBWQ"):
        rows = [
            (t, r.depth_m)
            for t, r in (
                (o.ts.astimezone(zone), o)
                for o in store.read(
                    station,
                    datetime(2026, 7, 1, tzinfo=zone),
                    datetime(2026, 7, 21, tzinfo=zone),
                )
            )
            if r.depth_m is not None
        ]
        if len(rows) < 500:
            continue  # this station's window is too sparse to trust here; try the others
        times = [t for t, _ in rows]
        depths = np.array([d for _, d in rows], dtype="float64")
        lags = []
        for h in highs:
            idx = [i for i, t in enumerate(times) if abs((t - h).total_seconds()) < 4 * 3600]
            if len(idx) < 20:
                continue
            j = idx[int(np.argmax(depths[idx]))]
            lags.append((times[j] - h).total_seconds() / 60.0)
        if not lags:
            continue
        checked += 1
        phase_units = abs(float(np.median(lags))) / (12.42 * 60)
        assert phase_units <= 0.05, (
            f"{station}'s tidal lag is {phase_units:.3f} phase units -- the spec's "
            "'one station suffices' ruling assumed <= 0.011. If this is real, a "
            "per-location lag model is now needed and the spec must be revisited."
        )
    if checked == 0:
        pytest.skip("no NERRS station had enough depth readings in the 2026-07-01..21 window")


# -- The ocean end-member's provenance --------------------------------------


OCEAN_PPT_TOLERANCE_PPT = 1.0


def test_ocean_ppt_matches_the_north_inlet_marine_measurement():
    """`ocean_ppt` is HELD, not fitted -- so nothing else checks that the value
    in the fishery YAML still matches the data it claims to come from.

    Derived 2026-08-25 from North Inlet's three NERRS stations conditioned on
    genuinely marine conditions: composite discharge at or below its 10th
    percentile (3,263 cfs) AND tidal phase in [0.40, 0.60], near high water.
    18,454 readings over 389 days; the three stations agree within 0.25 ppt;
    mean of daily means 35.47, sd 0.90, standard error ~0.20.

    The tolerance is 1.0 ppt, about 5 standard errors -- loose enough to
    tolerate a re-import shifting the record, tight enough to catch the two
    failures that matter: a silent reversion to the old unsourced 34.0
    (1.47 ppt away), or the ~20.8 ppt a freed, unanchored fit drives it to.

    Skips when the store or the tide predictions are unavailable, like the
    other real-data tests here -- a fresh clone has no `data/` directory.
    """
    import sqlite3
    from collections import defaultdict
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    import numpy as np

    from tidescout.config import load_fishery
    from tidescout.engine.tides import phase_at
    from tidescout.errors import SourceUnavailable
    from tidescout.pipeline.salinity_fit import composite_discharge_by_day
    from tidescout.sources import ndbc, noaa, usgs
    from tidescout.sources.cache import default_cache

    fishery = load_fishery("winyah-bay")
    store = ndbc.default_store("winyah-bay")
    stations = [s for s in ("NIWCBWQ", "NIWOLWQ", "NIWDCWQ") if s in store.stations()]
    if not stations:
        pytest.skip("NERRS store not present")

    cache = default_cache()
    daily = usgs.fetch_daily(
        [r.usgs_site for r in fishery.rivers if r.usgs_site],
        usgs.PARAM_DISCHARGE, "2016-01-01", "2026-08-23", cache,
    )
    by_day = composite_discharge_by_day(fishery, daily)
    if not by_day:
        pytest.skip("no composite discharge available")
    low_flow = float(np.percentile(list(by_day.values()), 10))

    try:
        events = noaa.tide_events_range(
            fishery.stations.tide[0], date(2016, 1, 1), date(2026, 12, 31),
            fishery.timezone, cache,
        )
    except SourceUnavailable:
        pytest.skip("tide predictions unavailable")

    zone = ZoneInfo(fishery.timezone)
    per_day: dict[date, list[float]] = defaultdict(list)
    con = sqlite3.connect(f"file:{store._db_path}?mode=ro", uri=True)
    for station in stations:
        rows = con.execute(
            "SELECT ts, salinity_psu FROM observations "
            "WHERE station = ? AND salinity_psu IS NOT NULL",
            (station,),
        )
        for ts, psu in rows:
            when = datetime.fromisoformat(ts).astimezone(zone)
            if by_day.get(when.date(), float("inf")) > low_flow:
                continue
            phase = phase_at(events, when)
            if phase is not None and 0.40 <= phase <= 0.60:
                per_day[when.date()].append(psu)

    if len(per_day) < 50:
        pytest.skip(f"only {len(per_day)} marine-condition days available")

    measured = float(np.mean([np.mean(v) for v in per_day.values()]))
    assert fishery.salinity.ocean_ppt == pytest.approx(
        measured, abs=OCEAN_PPT_TOLERANCE_PPT
    ), (
        f"ocean_ppt is {fishery.salinity.ocean_ppt} but North Inlet's "
        f"marine-condition record now measures {measured:.2f} ppt over "
        f"{len(per_day)} days. Re-derive the YAML value and update its comment "
        "rather than widening this tolerance."
    )


def test_zero_tau_returns_the_discharge_untouched():
    """The backward-compatibility guarantee: every existing caller and test
    must keep getting exactly today's behaviour."""
    from datetime import date

    from tidescout.pipeline.salinity_fit import smooth_discharge

    raw = {date(2026, 5, d): 1000.0 * d for d in range(1, 11)}
    out, dropped = smooth_discharge(raw, 0.0)
    assert out == raw
    assert dropped == 0


def test_a_constant_series_smooths_to_itself():
    """Any weighted mean of a constant is that constant. Catches a
    normalisation bug, which would otherwise show up only as a scale error in
    the fitted parameters."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 5000.0 for d in range(1, 61)}
    out, dropped = smooth_discharge(raw, 7.0)
    # tau=7, window=28, so a day needs 29 days of history (itself + 28 prior)
    # -- only days 29-60 of the 60-day record qualify.
    assert len(out) == 32
    assert dropped == 28
    assert all(v == pytest.approx(5000.0) for v in out.values())


def test_smoothing_lags_a_step_change():
    """The physical content: a discharge step must reach the model gradually.
    One day after a 10x step, a 7-day memory must have moved well short of
    the new value."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): (1000.0 if d <= 40 else 10000.0) for d in range(1, 61)}
    out, _ = smooth_discharge(raw, 7.0)
    assert out[day(41)] < 4000.0
    assert out[day(41)] > 1000.0
    assert out[day(60)] > out[day(41)]


def test_days_without_enough_history_are_dropped_and_counted():
    """A day smoothed over a SHORT window is not comparable with one smoothed
    over a full window -- its discharge would mean something different. Drop
    and count, never default."""
    from datetime import date

    from tidescout.pipeline.salinity_fit import smooth_discharge

    raw = {date(2026, 5, d): 5000.0 for d in range(1, 31)}
    out, dropped = smooth_discharge(raw, 7.0)
    assert dropped > 0
    assert min(out) > min(raw), "the earliest days cannot have a full window"
    assert dropped + len(out) == len(raw)


def test_a_gap_in_the_discharge_record_drops_the_days_it_covers():
    """A missing gauge day must not be interpolated across -- the composite
    already refuses to sum short when a gauge is dark, and this must not
    quietly undo that."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 5000.0 for d in range(1, 61) if d != 45}
    out, dropped = smooth_discharge(raw, 7.0)
    # 59 days present (60 minus day 45). A day survives when n >= 29 (full
    # 29-day window) and its lookback clears the day-45 hole, i.e. n <= 44
    # -- days 29-44 survive, 16 of them; the other 43 are dropped.
    assert len(out) == 16
    assert dropped == 43
    assert day(46) not in out


def test_a_longer_memory_drops_more_days_for_insufficient_history():
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 5000.0 for d in range(1, 61)}
    # Both taus must leave SURVIVORS -- tau=21 gives a 84-day window against a
    # 60-day record, so it drops everything and the assertion would then hold
    # at saturation rather than because memory length drives exclusions.
    # tau=3 -> window 12 -> 12 dropped; tau=10 -> window 40 -> 40 dropped.
    _, few = smooth_discharge(raw, 3.0)
    _, many = smooth_discharge(raw, 10.0)
    assert many > few
    assert few == 12 and many == 40


# -- Task 3 of this plan: wiring memory into the fit, and profiling tau -----


def _synthetic_calibration_input() -> "salinity_fit.CalibrationInput":
    """A `CalibrationInput` with a DATED discharge span long enough for a
    30-day memory window (needs 121 days of unbroken preceding history: a
    day survives tau=30's 120-day window only from day index 121 onward).

    Deliberately mixes two groups of observation days:

    * day(50)/day(60) -- old enough to survive tau=0 and tau=7's (much
      shorter) windows, but NOT tau=30's 120-day one.
    * day(125)..day(145) -- old enough to survive every tau in the [0, 7, 30]
      grid, tau=30 included.

    A scan that let each tau keep whatever ITS OWN window retained would
    therefore score tau in {0, 7} on 7 rows and tau=30 on only 5 -- three
    DIFFERENT populations. Restricting every candidate to the days tau=30
    retains (the correct behaviour) scores all three on the same 5. This
    mix is what makes `test_memory_scan_scores_every_tau_on_the_same_rows`
    able to fail on the defect it names; a helper whose observations all
    already fell inside every candidate's own window (as an earlier draft
    of this fixture did) could not distinguish the two implementations at
    all -- both would report equal counts, RIGHT AND WRONG ALIKE, and the
    test would pass by accident, not by proof.
    """
    from datetime import timedelta

    def day(n: int) -> date:
        return date(2026, 1, 1) + timedelta(days=n - 1)

    discharge_by_day = {day(n): 3000.0 + 25.0 * n for n in range(1, 151)}
    # The discharge value paired with each in `observations` below is a
    # placeholder -- `_memory_rows_by_tau` replaces it with each candidate
    # tau's own smoothed value at that day, never reading this one.
    obs_days = [day(n) for n in (50, 60, 125, 130, 135, 140, 145)]
    observations = [
        (10.0 + 0.5 * i, discharge_by_day[d], 5.0 + 0.1 * i)
        for i, d in enumerate(obs_days)
    ]
    # A per-row tidal phase, same length/order as `observations` -- lets
    # `profile_memory` tests prove it forwards each retained row's OWN
    # phase to `fit_intrusion` rather than silently defaulting every row to
    # the shared `FIT_PHASE`. Values are arbitrary but distinct from
    # `salinity_fit.FIT_PHASE` (0.25) so a test could tell the two apart.
    observation_phases = [0.1 * i for i in range(len(obs_days))]
    return salinity_fit.CalibrationInput(
        observations, [], [], 150, (day(1), day(150)), 0,
        discharge_by_day=discharge_by_day, observation_days=obs_days,
        observation_phases=observation_phases,
    )


def test_memory_scan_scores_every_tau_on_the_same_rows():
    """Larger tau drops more early days for insufficient history. Scoring each
    tau on whatever it happens to retain would let a tau win by discarding
    the hardest observations rather than by fitting better."""
    from tidescout.pipeline import salinity_fit

    counts = salinity_fit.profile_memory_row_counts(_synthetic_calibration_input(), [0, 7, 30])
    assert len(set(counts)) == 1, f"populations differ across tau: {counts}"


def test_the_fit_path_routes_its_discharge_through_smooth_discharge(monkeypatch):
    """If `collect_observations` inlines its own smoothing, or ignores
    `discharge_memory_days`, every fitted parameter silently describes a
    different quantity than a prediction caller supplies -- and nothing
    errors, because both are floats.

    This watches the fit path itself. Calling `smooth_discharge` twice and
    comparing the two results would NOT catch it: that only proves a pure
    function is deterministic, which is true of any implementation including
    an inlined duplicate.
    """
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")
    monkeypatch.setattr(fishery.salinity, "discharge_memory_days", 7.0)

    class _Store:
        def salinity_series(self, station):
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 10.0 + (i % 8)) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    # A discharge record long enough that a 7-day memory (28-day window) has
    # full history for the observation days above.
    by_day = {date(2026, 3, 1) + timedelta(days=i): 4000.0 + 10.0 * i for i in range(70)}
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs", lambda *a, **k: ({}, by_day, [], {}),
    )
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    real = salinity_fit.smooth_discharge
    seen_taus: list[float] = []

    def spy(by_day, tau_days):
        seen_taus.append(tau_days)
        return real(by_day, tau_days)

    monkeypatch.setattr(salinity_fit, "smooth_discharge", spy)
    salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    assert seen_taus, (
        "collect_observations never called smooth_discharge -- the fit path is "
        "reading raw discharge, or has inlined its own smoothing"
    )
    assert seen_taus == [7.0], f"expected the CONFIGURED tau, saw {seen_taus}"


def test_smoothing_at_the_configured_tau_is_not_a_no_op():
    """Guards the test above: if tau were misread as 0 everywhere, the spy
    would still fire and still see the configured value, while the discharge
    reaching the fit was unchanged."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 1000.0 + 100.0 * d for d in range(1, 61)}
    out, _ = smooth_discharge(raw, 7.0)
    target = day(55)
    assert out[target] != pytest.approx(raw[target])
    assert out[target] < raw[target], "a backward-weighted mean of a rising series must lag it"


# -- Post-review fixes (coordinator review of Task 3, 2026-08-25) -----------


def test_profile_memory_scores_each_tau_on_the_rows_own_resolved_phase(monkeypatch):
    """`profile_memory` must forward each retained row's OWN resolved tidal
    phase to `fit_intrusion`, not silently drop it and let every row default
    to the shared `FIT_PHASE` -- exactly the gap review caught: the real
    Winyah scan measured `n_phase_supplied` at 0 out of 12,204 rows, even
    though 1,860 of that population are WQP grabs whose individually
    resolved phase is worth up to 12.3 ppt at some sites (see
    `fit_intrusion`'s own docstring).

    A spy on `fit_intrusion`, not a rerun-and-compare: calling
    `fit_intrusion` twice with the same phases would only prove it is
    deterministic, which says nothing about whether `profile_memory` passed
    phases in the first place -- the same reasoning the smooth_discharge spy
    test above already applies to `collect_observations`.
    """
    from tidescout.pipeline import salinity_fit

    data = _synthetic_calibration_input()
    real = salinity_fit.fit_intrusion
    seen_phase_lengths: list[int] = []

    def spy(observations, cfg, swings=(), sources=(), phases=()):
        seen_phase_lengths.append(len(phases))
        return real(observations, cfg, swings=swings, sources=sources, phases=phases)

    monkeypatch.setattr(salinity_fit, "fit_intrusion", spy)
    profile = salinity_fit.profile_memory(data, CFG, [0, 7, 30])

    # Every one of the 3 candidate taus scored on the 5 rows the largest
    # (tau=30) retains -- see `_synthetic_calibration_input` -- each
    # carrying its own resolved phase, not an empty (FIT_PHASE-default) tuple.
    assert seen_phase_lengths == [5, 5, 5], (
        f"expected 3 calls, each passed 5 phases (one per retained row) -- "
        f"saw {seen_phase_lengths}"
    )
    # Also closes the coverage gap review named: nothing in this file called
    # `profile_memory` before this test. Adequate synthetic data (>=3 rows,
    # >=2 distinct discharges at every tau) -- confirms real numbers came
    # back, not every candidate silently landing on the thin-data nan branch.
    assert [tau for tau, _ in profile] == [0, 7, 30]
    assert all(np.isfinite(rmse) for _, rmse in profile), profile


def test_profile_memory_reports_nan_not_a_crash_for_a_tau_with_non_finite_rows():
    """`profile_memory`'s thin-data guard must filter for finiteness the
    SAME way `fit_intrusion`'s own `_finite_rows` does, evaluated BEFORE
    checking `len(rows) < 3` -- not after. `smooth_discharge` PROPAGATES a
    NaN gauge reading through `np.dot` rather than dropping it (only a
    MISSING day is dropped, via the `any(h is None ...)` check), so a tau
    whose retained population includes a corrupted discharge value has a
    raw row count that can stay >= 3 while its FINITE row count drops below
    it. Before this fix that raw-count-only guard let such a tau reach
    `fit_intrusion`, which raises `ValueError: need at least 3 finite
    observations...` once its own `_finite_rows` drops the same rows -- and
    now that the blanket `except ValueError` around that call is correctly
    narrowed (review's earlier finding), that exception PROPAGATES instead
    of being swallowed as `nan`, which would abort `salinity calibrate`
    mid-table for a fixture just like this one. Not reachable on today's
    shipped data (`discharge_memory_days` stays 0.0, and `n_dropped: 0` on
    the real Winyah collection), but reachable in principle."""
    from datetime import timedelta

    data = _synthetic_calibration_input()

    def day(n: int) -> date:
        return date(2026, 1, 1) + timedelta(days=n - 1)

    # `_synthetic_calibration_input`'s 5 largest-tau survivors are
    # day(125)/day(130)/day(135)/day(140)/day(145) (see its own docstring).
    # Corrupting the LAST 3 of those to NaN leaves 5 raw rows at every tau
    # (a day with a NaN discharge value is not DROPPED by `smooth_discharge`,
    # only a genuinely MISSING one is) but only 2 FINITE ones -- below
    # fit_intrusion's 3-row minimum, while the raw count (5) is not.
    corrupted = dict(data.discharge_by_day)
    for n in (135, 140, 145):
        corrupted[day(n)] = float("nan")
    data = salinity_fit.CalibrationInput(
        data.observations, [], [], data.days, data.day_span, 0,
        discharge_by_day=corrupted, observation_days=data.observation_days,
        observation_phases=data.observation_phases,
    )

    profile = salinity_fit.profile_memory(data, CFG, [0, 7, 30])

    assert [tau for tau, _ in profile] == [0, 7, 30]
    assert all(np.isnan(rmse) for _, rmse in profile), (
        f"expected every tau to report nan (2 finite rows, below the "
        f"3-row minimum) rather than raise or silently fit on NaN -- "
        f"got {profile}"
    )


def test_calibrate_cli_prints_the_tau_scan_table_when_data_is_dated(monkeypatch):
    """The scan's CLI block -- gated on `data.discharge_by_day` and
    `data.observation_days` both being populated (every OTHER CLI test in
    this file hand-builds a `CalibrationInput` without those two fields, so
    this block never executed under test before this one) -- contains a
    `strict=True` zip over `profile`/`counts` and renders each row's rmse
    with `f"{rmse:.4f}"`.

    Asserted against the SAME `profile_memory`/`profile_memory_row_counts`
    outputs the CLI itself computes internally, not a rerun-and-compare of
    those functions alone: this proves the CLI's OWN rendering loop (the
    zip, the formatting) is faithful to what they returned, one row per
    grid candidate -- a real gap review caught (Minor 6 residual): a
    truncated zip (e.g. `counts[:-1]`, silently dropping the last row) or a
    removed `n/a`-on-nan branch both left this test's PREDECESSOR green,
    because it checked only that the table existed, never that every row in
    it did."""
    from typer.testing import CliRunner

    from tidescout.cli import app
    from tidescout.config import load_fishery

    data = _synthetic_calibration_input()
    monkeypatch.setattr(salinity_fit, "collect_observations", lambda *a, **k: data)
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))

    assert result.exit_code == 0, result.output
    assert "discharge-memory tau scan" in out.lower()
    assert "diagnostic only" in out.lower()
    assert "rows scored" in out.lower()

    fishery = load_fishery("winyah-bay")
    expected_profile = salinity_fit.profile_memory(
        data, fishery.salinity, salinity_fit.MEMORY_GRID_DAYS
    )
    expected_counts = salinity_fit.profile_memory_row_counts(data, salinity_fit.MEMORY_GRID_DAYS)
    assert len(expected_profile) == len(salinity_fit.MEMORY_GRID_DAYS)
    for (tau, rmse), n in zip(expected_profile, expected_counts, strict=True):
        # This fixture is adequate data at every candidate tau (see
        # `_synthetic_calibration_input`), so every rmse here is a real,
        # distinct 4-decimal float, not `nan` -- a row a truncated zip
        # dropped would leave ITS rmse string absent from `out` while every
        # other row's remained, which a weaker "some numbers appear"
        # check cannot distinguish from a correctly rendered table.
        assert not np.isnan(rmse), f"fixture expected to fit at tau={tau:g}"
        assert f"{rmse:.4f}" in out, f"missing rendered row for tau={tau:g} (rmse {rmse:.4f})"
        assert str(n) in out


def test_calibrate_cli_warns_when_the_grid_outruns_the_record(monkeypatch):
    """When the largest tau in the grid cannot find ANY row in common with
    the rest of the collection, every candidate reports 0 rows and `nan`
    rmse. The footer's blanket claim ("every tau scored on the SAME row
    population") is technically still true at that point (0 == 0 == ...),
    but reads as if the scan succeeded on real data. This guard says
    explicitly that nothing survived -- review's Minor 7.

    Also the discriminating home for the `nan` -> `"n/a"` rendering branch
    (review's Minor 6 residual): every rmse in THIS fixture is `nan` by
    construction, so `"n/a"` must appear -- unlike the sibling table test
    above, whose adequate-data fixture never exercises that branch at all."""
    from typer.testing import CliRunner

    from tidescout.cli import app

    def _detached_days_input(*a, **k):
        data = _synthetic_calibration_input()
        return salinity_fit.CalibrationInput(
            data.observations, [], [], data.days, data.day_span, 0,
            discharge_by_day=data.discharge_by_day,
            # None of these dates exist among `discharge_by_day`'s keys at
            # all (which span 2026), so every candidate tau's restricted
            # population is empty -- simulates the grid's largest tau
            # outrunning the record.
            observation_days=[date(1999, 1, 1)] * len(data.observations),
        )

    monkeypatch.setattr(salinity_fit, "collect_observations", _detached_days_input)
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))

    assert result.exit_code == 0, result.output
    assert "outran the record" in out.lower()
    # The rendering branch itself: no test anywhere in this suite previously
    # asserted "n/a" appears anywhere, so removing the
    # `"n/a" if math.isnan(rmse) else f"{rmse:.4f}"` branch entirely (and
    # rendering raw `nan` instead) left every existing test green.
    assert "n/a" in out.lower()


def test_reject_and_report_counts_observations_and_swings_lost_to_smoothing(monkeypatch):
    """A day-level count of days dropped for insufficient history
    (`n_no_discharge_history`) cannot be read as an observation-level count
    -- a single dropped day can carry more than one admitted station's
    reading. Measured on the real Winyah record: 112 days lost at tau=7 cost
    164 observations and 142 swings, and neither number is derivable from
    the other. Review's Important 1: this must be visible directly, not
    inferred from a smaller `len(observations)`."""
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")
    monkeypatch.setattr(fishery.salinity, "discharge_memory_days", 7.0)

    # index 4 (2026-03-05) -- DROPPED by tau=7's 28-day window (needs 28
    # preceding days; only 4 exist). index 35 (2026-04-05) -- SURVIVES.
    early = datetime(2026, 3, 5, 4, 0, tzinfo=UTC)
    late = datetime(2026, 4, 5, 4, 0, tzinfo=UTC)

    class _Store:
        def salinity_series(self, station):
            return [
                (early + timedelta(minutes=15 * i), 10.0 + (i % 8)) for i in range(96)
            ] + [(late + timedelta(minutes=15 * i), 12.0 + (i % 6)) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    # No WQP fixture in THIS test -- see the dedicated WQP test below for
    # that half of the split; keeping this one to NERRS/USGS isolates the
    # observation/swing counters from the WQP-specific counter.
    monkeypatch.setattr(salinity_fit, "_wqp_sites", lambda slug: {})
    raw_by_day = {date(2026, 3, 1) + timedelta(days=i): 4000.0 + 10.0 * i for i in range(40)}
    monkeypatch.setattr(salinity_fit, "_usgs_inputs", lambda *a, **k: ({}, raw_by_day, [], {}))
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    # window = round(4 * 7) = 28 -> days index 0..27 (28 of them) dropped.
    assert data.n_no_discharge_history == 28
    # 3 on-axis NERRS stations (WYSS1, NIWWBWQ, NIWTAWQ) each lose their
    # EARLY-day mean and EARLY-day swing to smoothing -- 3 observations and
    # 3 swings, distinct from the 28-day count above.
    assert data.n_obs_no_discharge_history == 3
    assert data.n_swing_no_discharge_history == 3
    # Unaffected by this failure mode -- no WQP fixture, no gauge-dark days.
    assert data.n_wqp_no_discharge_day == 0
    # The smaller `len` alone is real but was, before this fix, the ONLY
    # place this loss was visible -- exactly what this counter now makes
    # explicit instead.
    assert len(data.observations) == 3
    assert len(data.swings) == 3


def test_wqp_grabs_lost_to_smoothing_are_not_misreported_as_no_discharge_day(monkeypatch):
    """A WQP grab whose day genuinely HAD a composite discharge, but which
    `smooth_discharge` then dropped for insufficient preceding history, must
    not land in `n_wqp_no_discharge_day` -- that counter's own CLI text says
    the cause is a day with NO discharge at all (e.g. a gauge's record
    starting late), which would be FALSE for a row excluded this way. It
    must count into `n_obs_no_discharge_history` instead, and
    `n_wqp_no_discharge_day` must stay reserved for the genuinely-missing
    day this fixture also carries. Review's Important 1, the WQP half."""
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")
    monkeypatch.setattr(fishery.salinity, "discharge_memory_days", 7.0)

    class _NerrsStore:
        def salinity_series(self, station):
            return []

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _NerrsStore())
    monkeypatch.setattr(salinity_fit, "_store_distances", lambda slug, fishery, sites: {})
    # 40 days of unbroken raw discharge starting 2026-03-01. tau=7 -> a
    # 28-day window, so day index < 28 (2026-03-01..2026-03-28) is dropped
    # by smoothing; day index >= 28 (2026-03-29 onward) survives.
    raw_by_day = {date(2026, 3, 1) + timedelta(days=i): 4000.0 + 10.0 * i for i in range(40)}
    monkeypatch.setattr(salinity_fit, "_usgs_inputs", lambda *a, **k: ({}, raw_by_day, [], {}))
    monkeypatch.setattr(
        salinity_fit, "_wqp_sites",
        lambda slug: {
            "WB-06": [
                # index 4 -- HAS raw discharge but tau=7 drops it (needs a
                # 29-day window; only 5 days precede it).
                (datetime(2026, 3, 5, 15, 0, tzinfo=UTC), 5.0),
                # index 35 -- survives tau=7's window.
                (datetime(2026, 4, 5, 15, 0, tzinfo=UTC), 6.0),
                # Outside the raw discharge record entirely (no gauge ever
                # reported this day) -- the ORIGINAL failure mode
                # `n_wqp_no_discharge_day` exists for.
                (datetime(2026, 6, 1, 15, 0, tzinfo=UTC), 7.0),
            ],
        },
    )
    monkeypatch.setattr(
        "tidescout.sources.wqp.station_coords",
        lambda slug: {"WB-06": (-79.30, 33.30)},
    )
    monkeypatch.setattr(
        salinity_fit, "site_distances_km",
        lambda slug, fishery, sites: {s: (10.28, 5.0) for s in sites},
    )
    monkeypatch.setattr(
        salinity_fit, "station_stem_km",
        lambda slug, fishery, sites: dict.fromkeys(sites, 0.5),
    )
    from tidescout.sources import noaa

    events = _bracketing_tide_events(
        datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 6, 5, tzinfo=UTC)
    )
    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: events)

    data = salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    # Only the 2026-04-05 grab survives into the fit.
    assert len(data.observations) == 1
    assert data.observations[0][2] == pytest.approx(6.0)
    # The genuinely-missing day (2026-06-01) is the ONLY thing
    # n_wqp_no_discharge_day counts.
    assert data.n_wqp_no_discharge_day == 1
    # The smoothing-dropped day (2026-03-05) counts into the
    # OBSERVATION-level counter instead, not the day-has-no-discharge one.
    assert data.n_obs_no_discharge_history == 1
