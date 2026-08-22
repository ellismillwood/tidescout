"""Intrusion-model behaviour.

These pin SHAPE, not calibrated values -- the constants are fitted in Task 5 and
will move. Every assertion here must survive recalibration.
"""

import numpy as np
import pytest

from tidescout.engine import salinity
from tidescout.models import SalinityConfig

CFG = SalinityConfig(
    ocean_ppt=34.0, l0_km=18.0, q0_cfs=4000.0, k=0.33, excursion_km=7.0,
    calibration_range_cfs=(1232.0, 22996.0),
)


def test_salinity_falls_monotonically_up_the_estuary():
    x = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
    s = salinity.salinity_at(x, cfs=4000.0, phase=0.25, cfg=CFG)
    assert np.all(np.diff(s) < 0)
    assert s[0] == pytest.approx(CFG.ocean_ppt, rel=0.05)


def test_higher_discharge_pushes_the_salt_front_seaward():
    """The whole reason salinity cannot read a three-value bucket."""
    x = np.full(1, 15.0)
    low = salinity.salinity_at(x, cfs=2000.0, phase=0.25, cfg=CFG)[0]
    high = salinity.salinity_at(x, cfs=20000.0, phase=0.25, cfg=CFG)[0]
    assert high < low
    assert low - high > 2.0, "a 10x discharge change must move salinity materially"


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


def test_tidal_swing_is_bounded_by_the_excursion():
    """The salt field slides; it does not teleport."""
    x = np.full(1, 12.0)
    swing = [
        salinity.salinity_at(x, cfs=4000.0, phase=p, cfg=CFG)[0]
        for p in np.linspace(0, 1, 24, endpoint=False)
    ]
    span_km = CFG.excursion_km * 2
    bound = CFG.ocean_ppt * (1 - np.exp(-span_km / CFG.l0_km))
    assert max(swing) - min(swing) <= bound + 1e-9


def test_salinity_never_exceeds_the_ocean_end_member():
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


def test_zero_discharge_does_not_divide_by_zero():
    s = salinity.salinity_at(np.array([10.0]), cfs=0.0, phase=0.25, cfg=CFG)
    assert np.isfinite(s[0])
