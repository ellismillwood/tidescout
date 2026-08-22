"""Empirical salt-intrusion model. Pure -- no I/O, no library lookup.

Winyah Bay is river-dominated, so salt-wedge position is a first-order control
on where fish are (spec section 7). This computes it analytically rather than
simulating it, which has one decisive advantage: the flow library knows only
three discharge values spanning 2,774-6,292 cfs, while the observed record runs
1,232-22,996. An analytic model reads the real number.

    x_eff = x + E * cos(2*pi*phase)                -- tidal shift, UNCLIPPED
    L(Q)  = L0 * (Q / Q0)^-k                        -- salt front's position
    S(x, Q, phase) = S_ocean * 0.5 * (1 - tanh((x_eff - L(Q)) / W))

W = `front_width_km`, the front's SHARPNESS, independent of L's POSITION.

A first version of this model used a clipped exponential --
S = S_ocean * exp(-max(0, x_eff) / L) -- and a real-data review caught two
structural defects in it (full arithmetic in
.superpowers/sdd/2026-08-16-04-phase2-salinity/task-3-report.md):

  1. Translating x by the tidal shift and then clipping at 0 is
     mathematically IDENTICAL to clipping S at S_ocean. Measured on the real
     587,325-cell distance field at high water: 47.40% of cells read exactly
     34.00 ppt whether discharge was 1,232 or 22,996 cfs -- a 19x change.
     North Jetty was discharge-blind 37.9% of every tidal cycle, Georgetown
     Lighthouse 21.1%. No clip on a translated coordinate avoids this; it has
     to be designed out.
  2. A single length scale is over-constrained: it has to set BOTH how salty
     the mouth stays and how fresh the head gets, and a plain exponential has
     no second knob to keep those apart. Forcing 1 ppt at the real domain's
     31.57 km head drove L down to 8.95 km, which alone cost North Jetty
     (2.58 km) 8.5 of its 34 ppt -- a compromise a least-squares fit would
     have landed on silently.

The bounded logistic (sigmoid) form above fixes both by construction. tanh's
range is [-1, 1], so S sits in [0, S_ocean] with NO clipping anywhere in this
module -- there is no plateau region left to reintroduce, at any phase. And L
(WHERE the front sits) is independent of W (HOW SHARP it is), so satisfying
the head's near-fresh condition no longer has to fight the mouth's near-ocean
one. Re-verified against the real field under this form: 0 exact ties between
the calibration range's endpoints at every phase, and the same 1 ppt head
constraint above now costs North Jetty only 0.01 ppt.
"""

from dataclasses import dataclass

import numpy as np

from tidescout.models import SalinityConfig


@dataclass
class SalinityField:
    ppt: np.ndarray
    cfs: float
    extrapolated: bool


def _effective_cfs(cfs: float) -> float:
    """The discharge the model actually evaluates at, after the divide-by-zero
    guard below.

    `cfs` MUST be `max`'s first argument: `max(nan, 1.0)` returns nan (Python
    keeps its first argument unless a later one compares strictly greater,
    and every comparison against nan is False), so a missing discharge
    reading stays nan the whole way through `salinity_at` instead of being
    silently treated as a real 1 cfs flow. `max(1.0, cfs)` would swallow that
    nan and return 1.0 -- a tidy-looking reorder that breaks this silently.
    """
    return max(float(cfs), 1.0)


def intrusion_length_km(cfs: float, cfg: SalinityConfig) -> float:
    """Distance scale that sets the salt front's position, shrinking as
    discharge rises.

    L does NOT saturate as Q -> 0 -- the power law diverges there (Q^-k grows
    without bound). What keeps it finite is `_effective_cfs`'s 1 cfs floor:
    with this fishery's defaults (l0_km=18, q0_cfs=4000, k=0.33) that floor
    caps L at ~277.9 km, an order of magnitude past the domain's own
    31.57 km extent. That is a guard against the power law blowing up at
    Q=0, not a physical property of the estuary.
    """
    return cfg.l0_km * (_effective_cfs(cfs) / cfg.q0_cfs) ** (-cfg.k)


def salinity_at(distance_km, cfs: float, phase: float, cfg: SalinityConfig):
    """Salinity in ppt at one or many along-estuary distances.

    The tidal term slides the whole profile: phase 0 is LOW water (spin-up is
    0.4831 of a cycle), so cos(2*pi*phase) is +1 there, pushing x_eff UP and
    making a given cell fresher, and -1 at high water. Reversing that sign
    inverts the tidal salinity swing across the entire bay. `x_eff` is used
    UNCLIPPED here -- see the module docstring for why clipping it (as an
    earlier version did) reintroduces the discharge-blind plateau this form
    exists to remove.

    NaN distances -- cells with no water route to the sea -- stay NaN. Treating
    them as 0 km would make an isolated pond the saltiest water in the model.
    """
    x = np.asarray(distance_km, dtype="float64")
    x_eff = x + cfg.excursion_km * np.cos(2.0 * np.pi * phase)
    length = intrusion_length_km(cfs, cfg)
    return cfg.ocean_ppt * 0.5 * (1.0 - np.tanh((x_eff - length) / cfg.front_width_km))


def salinity_field(
    distance_km, cfs: float, phase: float, cfg: SalinityConfig
) -> SalinityField:
    """`salinity_at` plus provenance: what discharge, and was it in range.

    The extrapolation flag exists because this model's characteristic failure
    is not a crash -- it is returning a confident number for a discharge nothing
    was ever fitted against. Spec section 10 requires degraded data to be
    surfaced, not swallowed.

    `.cfs` reports the EFFECTIVE (floored) discharge the model actually
    evaluated at -- not the raw input -- so it matches what was computed, per
    this field's own contract of "carries the discharge it was evaluated at."
    `.extrapolated` is checked against that same effective value for the same
    reason: the two must describe the same run.
    """
    lo, hi = cfg.calibration_range_cfs
    effective_cfs = _effective_cfs(cfs)
    return SalinityField(
        ppt=salinity_at(distance_km, cfs, phase, cfg),
        cfs=effective_cfs,
        extrapolated=not (lo <= effective_cfs <= hi),
    )
