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
     31.57 km head (36.19 km since the 2026-08-23 re-seeding) drove L down
     to 8.95 km, which alone cost North Jetty
     (2.58 km) 8.5 of its 34 ppt -- a compromise a least-squares fit would
     have landed on silently.

The bounded logistic (sigmoid) form above fixes both by construction. tanh's
range is [-1, 1], so S sits in [0, S_ocean] with NO clipping anywhere in this
module -- there is no plateau region left to reintroduce ACROSS THE
CALIBRATION RANGE, at any phase: 0 exact ties between the range's endpoints
at every phase tested, re-verified against the real field.

That scoping is deliberate, not decorative. Below the calibration range,
float64 itself reintroduces a plateau: tanh(z) rounds to exactly +-1 once
|z| exceeds ~19.06, i.e. once |x_eff - L| is at least ~19.06 * front_width_km
(~95 km with the defaults below). `intrusion_length_km`'s 1 cfs floor pushes
L to ~278 km at near-zero discharge, so at cfs <= ~5 -- 38x below the
observed 1,232 cfs minimum -- the whole field reads exactly ocean_ppt
regardless of the exact (sub-floor) discharge. That is the right physical
limit (marine everywhere at zero river flow), and every discharge that low
is already flagged `extrapolated=True`, but it IS a bit-exact tie -- the
same failure mode defect 1 was about, just relocated to where this model
never claimed to be trustworthy.

L (WHERE the front sits) is independent of W (HOW SHARP it is), so
satisfying the head's near-fresh condition no longer has to fight the
mouth's near-ocean one: the same 1 ppt head constraint above now costs
North Jetty only 0.01 ppt.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from tidescout.models import SalinityConfig


class Coverage(StrEnum):
    """How well OBSERVED this cell's along-estuary position is.

    A coverage statement, not a quality score -- deliberately ordinal rather
    than a 0-1 number, because a number invites being multiplied into
    something and this must not silently scale a bite score.

    Distinct from `SalinityField.extrapolated`, which asks only whether the
    DISCHARGE fell inside `calibration_range_cfs`, and from `fitted`, which
    asks whether the config was ever calibrated at all:

    * A caller can have `extrapolated=False` on a cell whose position
      nothing ever observed (coverage=EXTRAPOLATED) -- the discharge is in
      range, but this position is not.
    * The symmetric, currently-live case on Winyah Bay: a caller can just as
      easily have `coverage=MEASURED` -- a raw observation sat right next to
      this position -- on a config with `fitted=False`, meaning nothing ever
      calibrated the MODEL that turned that position into this number.
      MEASURED says a nearby observation exists; it says nothing about
      whether the value computed here was ever checked against it. Today
      78.6% of Winyah's cells are MEASURED while `fitted` is False for the
      whole fishery -- read `coverage` alone and this reads as trustworthy;
      it is not.
    """

    MEASURED = "measured"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"


# Width of the fixed-width string dtype `classify_coverage` and
# `SalinityField.coverage` use. Derived from `Coverage`'s own members rather
# than hardcoded, so a future member cannot silently outgrow it -- numpy
# TRUNCATES on overflow instead of raising (`np.full((1,), "x", dtype="<U3")
# [0] = "abcdef"` reads back as `"abc"`, no error anywhere), so a hardcoded
# width would corrupt values with no signal the day a longer label is added.
_COVERAGE_DTYPE = f"<U{max(len(c.value) for c in Coverage)}"


def classify_coverage(
    distance_km, observed_km: Sequence[float], near_km: float = 1.0
) -> np.ndarray:
    """Per-cell coverage against the along-estuary distances actually observed.

    MEASURED within `near_km` of an observation, INTERPOLATED inside their
    span (but not within `near_km` of one), EXTRAPOLATED outside it. With no
    observations everything is EXTRAPOLATED -- the honest answer, and the one
    Winyah gave before this work.

    MEASURED is checked AFTER, and overrides, "inside the span" -- it is
    *not* a subset of it. A cell just past either end of the observed span
    but within `near_km` of the nearest edge observation reads MEASURED, not
    EXTRAPOLATED: e.g. with `observed_km=[5.56, 10.28], near_km=1.0`, a cell
    at 4.56 km is outside [5.56, 10.28] yet reads MEASURED. This is
    deliberate, not an off-by-one -- see
    `test_a_cell_just_outside_the_span_but_near_an_edge_reads_measured` for
    the rationale: along a single 1-D coordinate, proximity to an
    observation is what carries information, not being bracketed by two of
    them. The alternative (span membership decides MEASURED) would call a
    cell 0.1 km beyond the last observation EXTRAPOLATED while calling one
    0.9 km inside it MEASURED -- a discontinuity reflecting no real
    difference in what is known.

    Output shape matches `distance_km`'s -- including a bare scalar, which
    comes back 0-d, matching `salinity_at`'s own scalar behaviour so
    `SalinityField.coverage` always aligns elementwise with `.ppt`.
    """
    d = np.asarray(distance_km, dtype="float64")
    obs = np.asarray(sorted(observed_km), dtype="float64")
    out = np.full(d.shape, str(Coverage.EXTRAPOLATED), dtype=_COVERAGE_DTYPE)
    if obs.size == 0:
        return out
    inside = (d >= obs[0]) & (d <= obs[-1])
    out[inside] = str(Coverage.INTERPOLATED)
    # `d[..., None]` adds a trailing axis so it broadcasts against `obs`
    # regardless of `d`'s rank (0-d scalar included, where plain `d[:, None]`
    # would raise); `.reshape(d.shape)` then puts the per-cell minimum back
    # in `d`'s own shape rather than whatever shape the reduction produced.
    nearest = np.min(np.abs(d[..., None] - obs), axis=-1).reshape(d.shape)
    out[nearest <= near_km] = str(Coverage.MEASURED)
    return out


@dataclass
class SalinityField:
    """Salinity plus the provenance a caller needs to know what it is worth.

    `extrapolated` and `fitted` answer DIFFERENT questions and a caller
    that reads only the first will be misled:

    * `extrapolated` -- was this DISCHARGE outside `calibration_range_cfs`.
      A per-evaluation property; it changes with the river.
    * `fitted` -- did a calibration ever constrain these parameters AT ALL.
      A property of the config, identical at every cell and every hour. It
      is `False` for Winyah Bay today (see `models.SalinityConfig.fitted`
      for the measured reason), which means a caller can get
      `extrapolated=False` -- "this discharge is in range" -- on a number
      that no observation anywhere ever constrained.
    * `coverage` -- how well OBSERVED each cell's POSITION is (see
      `Coverage`). Per-cell, unlike the two flags above: coverage varies
      ALONG the estuary within a single evaluation, which is the whole
      thing this field exists to express, so a scalar would collapse it.
      `coverage=MEASURED` is NOT a trust signal on the computed VALUE --
      see `Coverage`'s docstring for the concrete, currently-live case
      (78.6% MEASURED coexisting with `fitted=False` on all of Winyah Bay).

    Neither `extrapolated` nor `fitted` changes a computed value, and
    `coverage` changes none either. They ride alongside the numbers.
    """

    ppt: np.ndarray
    cfs: float
    extrapolated: bool
    # Whether cfg's parameters were fitted to observations. Metadata only --
    # see the class docstring for why this cannot be folded into
    # `extrapolated`.
    fitted: bool = False
    # Per-cell coverage, aligned elementwise with `ppt` (same shape, scalar
    # included -- see `classify_coverage`). Defaults to empty, not
    # populated -- callers that build a `SalinityField` directly (as several
    # existing tests do) have no need of it.
    coverage: np.ndarray = field(default_factory=lambda: np.array([], dtype=_COVERAGE_DTYPE))


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
    caps L at ~277.9 km, an order of magnitude past L at either end of
    `calibration_range_cfs` (~26.5 km at 1,232 cfs, ~10.1 km at 22,996 cfs).
    That is a guard against the power law blowing up at Q=0, not a physical
    property of the estuary -- and that gap is exactly what saturates
    `salinity_at`'s tanh outside the calibration range (see the module
    docstring); it is not sized to keep L "reasonable" in any absolute sense.
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
    distance_km,
    cfs: float,
    phase: float,
    cfg: SalinityConfig,
    observed_km: Sequence[float] = (),
) -> SalinityField:
    """`salinity_at` plus provenance: what discharge, was it in range, and how
    well observed each cell's position is.

    The extrapolation flag exists because this model's characteristic failure
    is not a crash -- it is returning a confident number for a discharge nothing
    was ever fitted against. Spec section 10 requires degraded data to be
    surfaced, not swallowed.

    `.cfs` reports the EFFECTIVE (floored) discharge the model actually
    evaluated at -- not the raw input -- so it matches what was computed, per
    this field's own contract of "carries the discharge it was evaluated at."
    `.extrapolated` is checked against that same effective value for the same
    reason: the two must describe the same run.

    `observed_km` is an ARGUMENT, not a `SalinityConfig` field, and that is
    deliberate: coverage is a property of the observations currently held,
    which change every time a store is imported, while `cfg` is authored in
    the fishery YAML and is meant to be stable and reviewable. It defaults to
    `()`, so every existing caller keeps working and gets all-EXTRAPOLATED
    coverage -- the honest answer for a caller that has not said what it
    observed.
    """
    lo, hi = cfg.calibration_range_cfs
    effective_cfs = _effective_cfs(cfs)
    return SalinityField(
        ppt=salinity_at(distance_km, cfs, phase, cfg),
        cfs=effective_cfs,
        extrapolated=not (lo <= effective_cfs <= hi),
        fitted=cfg.fitted,
        coverage=classify_coverage(distance_km, observed_km),
    )
