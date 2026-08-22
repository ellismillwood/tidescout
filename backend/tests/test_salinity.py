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
        assert np.all(s <= CFG.ocean_ppt + 1e-9)


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
