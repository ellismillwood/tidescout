"""Empirical salt-intrusion model. Pure -- no I/O, no library lookup.

Winyah Bay is river-dominated, so salt-wedge position is a first-order control
on where fish are (spec section 7). This computes it analytically rather than
simulating it, which has one decisive advantage: the flow library knows only
three discharge values spanning 2,774-6,292 cfs, while the observed record runs
1,232-22,996. An analytic model reads the real number.

    S(x, Q, phase) = S_ocean * exp(-x_eff / L(Q))
    L(Q)           = L0 * (Q / Q0)^-k
    x_eff          = max(0, x + E * cos(2*pi*phase))
"""

from dataclasses import dataclass

import numpy as np

from tidescout.models import SalinityConfig


@dataclass
class SalinityField:
    ppt: np.ndarray
    cfs: float
    extrapolated: bool


def intrusion_length_km(cfs: float, cfg: SalinityConfig) -> float:
    """Distance scale over which salinity decays, shrinking as discharge rises.

    A floor of 1 cfs keeps a dry-gauge zero from dividing by zero; at that flow
    the estuary is tidally dominated and the length scale saturates anyway.
    """
    q = max(float(cfs), 1.0)
    return cfg.l0_km * (q / cfg.q0_cfs) ** (-cfg.k)


def salinity_at(distance_km, cfs: float, phase: float, cfg: SalinityConfig):
    """Salinity in ppt at one or many along-estuary distances.

    The tidal term slides the whole profile: phase 0 is LOW water (spin-up is
    0.4831 of a cycle), so cos(2*pi*phase) is +1 there, pushing x_eff UP and
    making a given cell fresher, and -1 at high water. Reversing that sign
    inverts the tidal salinity swing across the entire bay.

    NaN distances -- cells with no water route to the sea -- stay NaN. Treating
    them as 0 km would make an isolated pond the saltiest water in the model.
    """
    x = np.asarray(distance_km, dtype="float64")
    shifted = x + cfg.excursion_km * np.cos(2.0 * np.pi * phase)
    x_eff = np.clip(shifted, 0.0, None)
    return cfg.ocean_ppt * np.exp(-x_eff / intrusion_length_km(cfs, cfg))


def salinity_field(
    distance_km, cfs: float, phase: float, cfg: SalinityConfig
) -> SalinityField:
    """`salinity_at` plus provenance: what discharge, and was it in range.

    The extrapolation flag exists because this model's characteristic failure
    is not a crash -- it is returning a confident number for a discharge nothing
    was ever fitted against. Spec section 10 requires degraded data to be
    surfaced, not swallowed.
    """
    lo, hi = cfg.calibration_range_cfs
    return SalinityField(
        ppt=salinity_at(distance_km, cfs, phase, cfg),
        cfs=float(cfs),
        extrapolated=not (lo <= float(cfs) <= hi),
    )
