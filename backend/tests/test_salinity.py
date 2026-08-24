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


# -- Assembling real observations -------------------------------------------

from datetime import UTC, date, datetime  # noqa: E402

from tidescout.config import load_fishery  # noqa: E402
from tidescout.pipeline import salinity_fit  # noqa: E402


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
        # old order tested `not rows` first and reported "no 00480 history" --
        # a data-availability claim about a site whose problem is that nobody
        # knows where it is.
        ([], False, NAN, float("inf"), False, "no coordinates"),
        ([], True, 10.0, 12.0, False, "no 00480 history"),
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
        ),
    )
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))

    assert "00480 sensors over the last 90 days" not in out
    assert "NERRS store" in out
    assert "2016-01-01 .. 2026-08-22" in out
