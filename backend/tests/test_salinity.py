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
    This is the property that must never regress."""
    x = np.full(1, 2.58)  # North Jetty
    lo, hi = CFG.calibration_range_cfs
    low = salinity.salinity_at(x, cfs=lo, phase=0.5, cfg=CFG)[0]
    high = salinity.salinity_at(x, cfs=hi, phase=0.5, cfg=CFG)[0]
    assert low != high, "salinity must not be identical across the full discharge range"
    assert high < low, "higher discharge must still freshen the cell, even at high water"


def test_intrusion_length_shrinks_as_a_power_law_in_discharge():
    assert salinity.intrusion_length_km(CFG.q0_cfs, CFG) == pytest.approx(CFG.l0_km)
    doubled = salinity.intrusion_length_km(2 * CFG.q0_cfs, CFG)
    assert doubled == pytest.approx(CFG.l0_km * 2 ** (-CFG.k), rel=1e-6)
    assert doubled < CFG.l0_km


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
