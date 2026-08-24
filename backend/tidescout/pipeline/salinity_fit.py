"""Fit the intrusion model to observations, and report what the fit can't say.

Least squares always returns numbers. The load-bearing output of this module
is therefore not `SalinityConfig` but the diagnostics beside it: how many
observations, over what discharge span, at how many distinct along-estuary
distances, and whether the parameters are separable from each other at all.

WHAT IS FITTED, AND WHY THE REST IS HELD
----------------------------------------
Free:  `l0_km`, `k`, `front_width_km`, and `excursion_km` -- the last ONLY
when swing observations are supplied (see below).

Held, each for a specific reason rather than convenience:

* `ocean_ppt` -- there is no ocean observation to fit it against. Task 4
  live-verified that no CO-OPS station within 250 km of Winyah Bay serves
  salinity (Springmaid Pier has no conductivity sensor at all), so
  `fetch_ocean_salinity` raises `SourceUnavailable` on every real call.
  Letting the optimizer move a seaward end-member that nothing observes
  would hand it a free scale factor to absorb everything else's error.

* `q0_cfs` -- it is EXACTLY degenerate with `l0_km`, not merely
  ill-conditioned. L(Q) = l0 * (Q/q0)^-k is invariant under
  (l0, q0) -> (l0 * c^k, c * q0) for any c > 0, so fitting both makes the
  problem rank-deficient by construction and the pair lands wherever the
  optimizer's path happens to stop.

`excursion_km` is the subtle one. Observations are daily means, so the fit
evaluates the model at phase 0.25 -- and at phase 0.25 `cos(2*pi*phase)` is
exactly zero, which means the profile residual is IDENTICALLY independent of
`excursion_km`. Handed to the optimizer against a spatial-only objective it
comes back as its starting value with zero gradient, indistinguishable in the
output from a fitted number. It is therefore held unless swing observations
(high-water minus low-water salinity) are supplied, which are the only data
that carry information about it. Task 3 measured why this matters: held at
7.0 km with a climatology-scale profile the model implies a 22-29 ppt tidal
swing at North Jetty and Mud Bay Cut, which is unphysical, and the same is
true of the clipped-exponential form it replaced (20-24 ppt) -- both forms
translate a rigid profile by +-E.

DISTANCE IS AN INPUT, NOT A NUISANCE PARAMETER
----------------------------------------------
No length scale here absorbs an error in the distance field. Task 1 measured
what changing the seed definition does to it: the field warps SPATIALLY (p50
0.75 km, p90 4.28, max 7.94, std 1.72), and a single scalar cannot undo a
spatially structured deformation. `snap_gap_m` on each site record exists so
a site with no honest place in the field is visible rather than quietly
assigned the nearest cell's value.

WHAT THIS FOUND ON WINYAH BAY (2026-08-22)
------------------------------------------
Both of the bay's USGS `00480` sites sit OUTSIDE the model domain -- 1,362 m
and 9,498 m from the nearest in-domain cell -- and both therefore snap to the
same cell, the field's maximum at 31.57 km. Every real observation available
carries one along-estuary distance, at the extreme fresh end; the 0-31.57 km
reach that the scoring layer actually reads has no salinity observation at
all. The one in-domain candidate (Sampit, 02136371, at 20.01 km) has zero
`00480` history over four years, and CO-OPS supplies no ocean end-member.

Fitted anyway, 348 observations over a year and a 9.7x discharge span, four
parameter sets whose rmse differed by 0.016 ppt -- 60x below the data's own
1 ppt quantisation -- predicted 19.8-34.0 ppt at North Jetty and 12.6-33.9 ppt
at Mud Bay Cut. The fit is not constrained where it is read. That is a
statement about the data, not about the optimizer.

EVERY DISTANCE ABOVE PREDATES THE 2026-08-23 RE-SEEDING -- READ THEM AS HISTORY
------------------------------------------------------------------------------
The field those numbers were measured on routed the mid/upper bay east through
Mud Bay and out North Inlet instead of down the bay past the jetties, because
`ocean_seed_mask` seeded from a coastal strip holding BOTH openings. See
`models.ModelDomain.salt_source_boundary_utm_km`. The field now maxes at
36.19 km, not 31.57, and the same USGS pair snaps to 36.19. Distances quoted
anywhere in this repo alongside a date before 2026-08-23 were measured on the
old routing; the conclusions they support still hold -- the re-seeding made
the coverage gap larger, not smaller -- but the kilometre figures moved.

WHAT THE FULL NERRS RECORD THEN SHOWED (2026-08-23)
---------------------------------------------------
10,864 observations and 10,864 tidal swings over 2016-01-01..2026-08-22, a
149x discharge span, from three in-bay stations at 16.68 and 19.03 km. rmse
4.060 ppt against an observation resolution of 0.003 -- 1,353x, where Task 5's
1.719 ppt was only 1.7x its 1.0 ppt USGS quantum. That difference is the whole
result: Task 5 could report only that its data could not tell, and this run
can state that the model does not reproduce the estuary. Condition number
12.2, every 1-sigma below its value, nothing at a bound -- numerically healthy
and reproducing nothing, which is exactly what `_warnings` exists to surface.

The reasons are structural and more data of this kind will not fix them: a
depth-averaged single layer cannot hold a +3.30 ppt median stratification at
one distance; one distance axis cannot carry North Inlet's 25 ppt baseline
offset (hence `WaterSensor.off_axis`); and the bay's own observations span
2.35 km against a 14.48 km fitted front width, entirely ABOVE the 2.58-13.05
km reach where the fishing spots are.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import least_squares

from tidescout.engine.salinity import salinity_at
from tidescout.models import Fishery, SalinityConfig

# (along-estuary km, discharge cfs, ppt). Swing rows carry a high-minus-low
# salinity difference in the third slot instead of a level.
Observation = tuple[float, float, float]

# Daily means average the tidal swing out, so they are evaluated where the
# tidal term vanishes rather than at an arbitrary instant. Fitting a daily
# mean at, say, phase 0.5 would push the excursion's signal into `l0_km`.
FIT_PHASE = 0.25

# An observation is "interior" if it sits between 10% and 90% of the ocean
# value. Anchors at the ends fix the asymptotes and say nothing about the
# shape between them, which is the part the scoring layer reads.
INTERIOR_BAND = (0.10, 0.90)

# Below this ratio between the highest and lowest discharge, the data cannot
# resolve a power law in discharge: a factor of 2 is already a thin lever on
# an exponent, and anything less is arithmetic rather than evidence.
MIN_CFS_RATIO = 2.0

# Readings a day needs before its mean and range are usable. A full day at
# 15-minute cadence is 96; 40 is ~10 hours, most of a 12.42 h tidal cycle.
# Below that the day is not merely thinner, it is BIASED -- see
# `daily_means_and_swings`.
MIN_DAILY_READINGS = 40

# Water-sensor kinds whose history lives in the local NERRS store rather than
# behind a live parameter feed. Both routes write the same table (cdmo.py
# aliases `niwwswq` onto WYSS1), so one reader covers them.
_STORE_KINDS = ("ndbc", "cdmo")

# USGS serves INSTANTANEOUS values for 120 days. Past that the IV endpoint
# answers 301 to a different host, and httpx does not follow redirects by
# default, so it arrives as `SourceUnavailable` -- measured: P120D returns
# 200, P180D returns 301. Daily means have no such limit, so a longer --days
# still lengthens the profile record; only the swing window is capped.
MAX_IV_DAYS = 120

# Scaled-Jacobian condition number above which the parameters are trading off
# against each other rather than being determined separately.
MAX_CONDITION = 1.0e6

# Optimizer bounds. Every one sits strictly inside `SalinityConfig`'s own
# validation, so the fit cannot land on a config the model would reject --
# see `models.SalinityConfig`, where l0_km <= 0 was measured to produce
# salinity ABOVE ocean_ppt with no error anywhere.
_BOUNDS: dict[str, tuple[float, float]] = {
    "l0_km": (0.1, 200.0),
    "k": (0.0, 2.0),
    "front_width_km": (0.1, 100.0),
    "excursion_km": (0.05, 50.0),
}
_SPATIAL_PARAMS = ("l0_km", "k", "front_width_km")


def observation_resolution_ppt(values: Sequence[float]) -> float:
    """The finest distinction the observation set actually draws, in ppt.

    Mean spacing between distinct observed values: (max - min) / (n - 1).
    DERIVED from the data rather than authored, and for uniformly quantised
    data it recovers the quantum exactly -- Winyah's 348 daily means take 11
    distinct values, the integers 0 through 10, giving exactly 1.0000 ppt.
    That is the most those observations can resolve, so a residual below it
    is not measurable and a residual above it is structure the model failed
    to represent, not rounding.

    Degrades sensibly for continuous data, where it becomes the mean level
    spacing: measured on dense synthetic observations spanning 0-34 ppt it
    gives ~1.7 ppt, which passes a legitimately noisy fit (rmse 0.263) and
    still catches a per-site bias the model cannot represent (rmse 3.897).

    NaN when fewer than two distinct values exist -- nothing to derive from,
    and the caller skips the check rather than inventing a threshold.
    """
    v = np.unique(np.asarray(list(values), dtype="float64"))
    if v.size < 2:
        return float("nan")
    return float((v[-1] - v[0]) / (v.size - 1))


def _finite_rows(rows: Sequence[Observation]) -> tuple[list[Observation], int]:
    """Keep rows whose three values are all finite; count what was dropped.

    A dark sensor arrives as NaN. One NaN row makes every residual NaN, and
    `least_squares` then terminates on the first iteration and returns the
    starting guess -- a fit that never happened, reported as one that did.
    """
    clean = [
        (float(d), float(q), float(y))
        for d, q, y in rows
        if np.isfinite(d) and np.isfinite(q) and np.isfinite(y)
    ]
    return clean, len(rows) - len(clean)


def _by_discharge(rows: Sequence[Observation]) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Group rows by discharge: (cfs, row indices, distances).

    `salinity_at` takes one discharge and many distances, so grouping turns
    one engine call per observation into one per distinct flow -- and keeps
    the model's own vectorised path, rather than reimplementing it here.
    """
    groups: dict[float, list[int]] = {}
    for i, (_, q, _) in enumerate(rows):
        groups.setdefault(q, []).append(i)
    return [
        (q, np.array(idx), np.array([rows[i][0] for i in idx], dtype="float64"))
        for q, idx in groups.items()
    ]


def _levels(groups, cfg: SalinityConfig, n: int) -> np.ndarray:
    out = np.empty(n, dtype="float64")
    for q, idx, dist in groups:
        out[idx] = salinity_at(dist, q, FIT_PHASE, cfg)
    return out


def _swing(groups, cfg: SalinityConfig, n: int) -> np.ndarray:
    """High water minus low water. Phase 0 is LOW water in this model."""
    out = np.empty(n, dtype="float64")
    for q, idx, dist in groups:
        out[idx] = salinity_at(dist, q, 0.5, cfg) - salinity_at(dist, q, 0.0, cfg)
    return out


def _at_bounds(fitted_values: dict[str, float]) -> list[str]:
    """Parameters resting on an optimizer bound.

    Measured on the real Winyah fit: `k` came back as exactly 2.0, its upper
    bound, with a 1-sigma of 0.82 and a condition number of 46 -- i.e. every
    numerical health check said "fine". A value at its bound is not a fitted
    value. It means the data pushed the parameter past the range the bounds
    enclose and the BOUND stopped it, not the evidence, and the covariance
    around it is meaningless because the parameter was never free to move.
    """
    out = []
    for name, value in fitted_values.items():
        lo, hi = _BOUNDS[name]
        tol = 1e-6 * max(1.0, abs(lo), abs(hi))
        if abs(value - lo) <= tol or abs(value - hi) <= tol:
            out.append(name)
    return out


def _uncertainty(sol, names: list[str], values: list[float]):
    """1-sigma per parameter, and the scaled Jacobian's condition number.

    Two different questions, deliberately reported separately:

    * The condition number is a property of the DESIGN -- which distances and
      discharges were observed. It is large when parameters trade off against
      each other, whatever the noise. Columns are scaled by their parameter's
      own magnitude first, since km and a dimensionless exponent are not
      comparable raw.
    * `param_sigma` converts that into ppt-denominated uncertainty using the
      fit's own residual, so it is zero for exactly-reproducing synthetic
      data no matter how degenerate the design. Neither alone is enough.

    `None` means "not estimable at all": fewer residuals than parameters, or a
    numerically rank-deficient Jacobian.
    """
    jac = np.asarray(sol.jac, dtype="float64")
    m, n = jac.shape
    scale = np.array([abs(v) if abs(v) > 1e-12 else 1.0 for v in values])
    _, sv, vt = np.linalg.svd(jac * scale, full_matrices=False)
    condition = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")

    dof = m - n
    tol = max(m, n) * float(np.finfo(float).eps) * float(sv[0]) if sv.size else 0.0
    if dof <= 0 or sv.size == 0 or sv[-1] <= tol:
        return dict.fromkeys(names), condition
    resid = np.asarray(sol.fun, dtype="float64")
    resid_var = float(resid @ resid) / dof
    cov = (vt.T * (1.0 / sv**2)) @ vt * resid_var
    sigma = np.sqrt(np.abs(np.diag(cov))) * scale
    return {name: float(v) for name, v in zip(names, sigma, strict=True)}, condition


def fit_intrusion(
    observations: Sequence[Observation],
    cfg: SalinityConfig,
    swings: Sequence[Observation] = (),
    sources: Sequence[str] = (),
) -> tuple[SalinityConfig, dict]:
    """Least-squares fit of the intrusion model to `observations`.

    `observations` is [(distance_km, cfs, ppt)] -- daily means, evaluated at
    `FIT_PHASE`. `swings` is [(distance_km, cfs, ppt)] where the third value
    is an observed high-water-minus-low-water DIFFERENCE; supplying it is what
    frees `excursion_km` (see the module docstring). Both are fitted in the
    same ppt units and enter the objective with equal weight.

    `sources`, when given, is a sequence the SAME LENGTH AND ORDER as
    `observations`, naming where each row came from (e.g. "usgs", "nerrs",
    "wqp" -- see `collect_observations`). It plays no part in the fit
    itself: every row is fitted identically regardless of source, and
    `sources` only controls what `rmse_by_source_ppt` reports back. This
    matters because `collect_observations` as of Task 5 mixes two
    population shapes -- 15-minute daily means and single WQP grab samples
    entered at `FIT_PHASE` as though they were tidal averages (see the
    comment where WQP rows are appended) -- and those populations were
    measured to residualise very differently (NERRS daily means rmse 4.061
    ppt on 10,880 rows vs WQP grabs rmse 6.102 ppt on 1,860, 2026-08-24).
    A single headline rmse hides that split; empty (the default) omits it,
    and `rmse_by_source_ppt` is then `{}`.

    Returns the fitted config -- constructed through validation, not
    `model_copy`, which skips it -- and a diagnostics dict. Read the
    diagnostics before the config, and within them know which numbers bite:

    * `n_distinct_distances` / `distance_span_km` are the load-bearing
      checks. The profile's SHAPE is what the scoring layer reads and only
      these say whether the data spans it.
    * `n_interior_obs` is VALUE-based -- it counts observations falling
      between 10% and 90% of `ocean_ppt` -- and is BLIND TO SPATIAL
      COVERAGE. It read 36 on Winyah's real data, every one of those 36 at
      the same along-estuary distance. A healthy interior count is not
      evidence of a constrained profile; do not read it alone.
    * `condition_number`, `param_sigma` and `at_bounds` describe the design
      that WAS observed and are structurally blind to extrapolation beyond
      it. All three read healthy on Winyah's unusable fit.
    * `fitted` on the returned config is True only if `warning` is empty.
    """
    if sources and len(sources) != len(observations):
        raise ValueError(
            f"sources has {len(sources)} entries but observations has "
            f"{len(observations)} -- they must be the same length, in the same order"
        )
    obs, dropped = _finite_rows(observations)
    swing_obs, swing_dropped = _finite_rows(swings)
    if len(obs) < 3:
        raise ValueError(
            f"need at least 3 finite observations to fit 2+ parameters, got {len(obs)}"
            + (f" ({dropped} dropped as non-finite)" if dropped else "")
        )
    flows = {q for _, q, _ in obs}
    if len(flows) < 2:
        raise ValueError(
            "all observations share one discharge -- k is the model's response TO "
            "discharge and cannot be constrained by a single value. Collect "
            "observations across at least two distinct flows."
        )

    names = [*_SPATIAL_PARAMS] + (["excursion_km"] if swing_obs else [])
    lo = [_BOUNDS[n][0] for n in names]
    hi = [_BOUNDS[n][1] for n in names]
    x0 = [float(np.clip(getattr(cfg, n), *_BOUNDS[n])) for n in names]

    groups = _by_discharge(obs)
    y = np.array([o[2] for o in obs], dtype="float64")
    swing_groups = _by_discharge(swing_obs)
    y_swing = np.array([o[2] for o in swing_obs], dtype="float64")

    def residual(params: np.ndarray) -> np.ndarray:
        trial = cfg.model_copy(update=dict(zip(names, map(float, params), strict=True)))
        r = _levels(groups, trial, len(obs)) - y
        if swing_obs:
            r = np.concatenate([r, _swing(swing_groups, trial, len(swing_obs)) - y_swing])
        return r

    sol = least_squares(residual, x0=x0, bounds=(lo, hi))
    fitted_values = {n: float(v) for n, v in zip(names, sol.x, strict=True)}
    q = np.array([o[1] for o in obs], dtype="float64")
    cfs_span = (float(q.min()), float(q.max()))
    solution = cfg.model_copy(update=fitted_values)

    d = np.array([o[0] for o in obs], dtype="float64")
    band_lo, band_hi = INTERIOR_BAND
    frac = y / cfg.ocean_ppt
    n_interior = int(((frac > band_lo) & (frac < band_hi)).sum())
    n_distinct_d = len({round(v, 6) for v in d})
    distance_span = float(d.max() - d.min())
    level_resid = _levels(groups, solution, len(obs)) - y
    rmse = float(np.sqrt(np.mean(level_resid**2)))
    rmse_by_source: dict[str, float] = {}
    if sources:
        # `observations` (pre-filter, same length as `sources`) and `obs`
        # (post-`_finite_rows`) can differ in length -- rebuild the same
        # finite mask `_finite_rows` applies so `kept_sources` lines up with
        # `obs`/`level_resid` row for row.
        finite_mask = [
            np.isfinite(d) and np.isfinite(q) and np.isfinite(y_)
            for d, q, y_ in observations
        ]
        kept_sources = [s for s, keep in zip(sources, finite_mask, strict=True) if keep]
        by_source: dict[str, list[float]] = {}
        for src, r in zip(kept_sources, level_resid, strict=True):
            by_source.setdefault(src, []).append(float(r))
        rmse_by_source = {
            src: float(np.sqrt(np.mean(np.square(rs)))) for src, rs in sorted(by_source.items())
        }
    swing_rmse = (
        float(np.sqrt(np.mean((_swing(swing_groups, solution, len(swing_obs)) - y_swing) ** 2)))
        if swing_obs
        else None
    )
    param_sigma, condition = _uncertainty(
        sol, names, [fitted_values[n] for n in names]
    )
    at_bounds = _at_bounds(fitted_values)
    warning = _warnings(
        rmse=rmse,
        resolution=observation_resolution_ppt(y),
        swing_rmse=swing_rmse,
        swing_resolution=observation_resolution_ppt(y_swing) if swing_obs else float("nan"),
        n_obs=len(obs),
        n_params=len(names),
        n_interior=n_interior,
        n_distinct_d=n_distinct_d,
        distance_span=distance_span,
        front_width_km=fitted_values["front_width_km"],
        cfs_span=cfs_span,
        fitted_values=fitted_values,
        param_sigma=param_sigma,
        condition=condition,
        has_swings=bool(swing_obs),
        at_bounds=at_bounds,
    )

    # `fitted=True` requires the fit to raise NO warning about its own data.
    # A clean residual is not enough and never was: the run that produced
    # Winyah's 1.719 ppt rmse also reported a healthy condition number and
    # 1-sigmas below every value, on data that could not distinguish 19.8
    # from 34.0 ppt at North Jetty. The warnings are the checks with teeth,
    # so passing all of them is the bar.
    fitted = SalinityConfig(
        **{
            **cfg.model_dump(),
            **fitted_values,
            "calibration_range_cfs": cfs_span,
            "fitted": not warning,
        }
    )

    diagnostics = {
        "rmse_ppt": rmse,
        "rmse_by_source_ppt": rmse_by_source,
        "n_obs": len(obs),
        "n_interior_obs": n_interior,
        "cfs_span": cfs_span,
        "warning": warning,
        "n_dropped": dropped + swing_dropped,
        "n_swing_obs": len(swing_obs),
        "swing_rmse_ppt": swing_rmse,
        "n_distinct_distances": n_distinct_d,
        "distance_span_km": distance_span,
        "n_distinct_discharges": len(flows),
        "fitted_params": list(names),
        "param_sigma": param_sigma,
        "condition_number": condition,
        "at_bounds": at_bounds,
    }
    return fitted, diagnostics


def _warnings(
    *,
    rmse: float,
    resolution: float,
    swing_rmse: float | None,
    swing_resolution: float,
    n_obs: int,
    n_params: int,
    n_interior: int,
    n_distinct_d: int,
    distance_span: float,
    front_width_km: float,
    cfs_span: tuple[float, float],
    fitted_values: dict[str, float],
    param_sigma: dict[str, float | None],
    condition: float,
    has_swings: bool,
    at_bounds: list[str],
) -> str:
    """Every way this data set can fail to constrain the model, stated plainly.

    Deliberately verbose. This string is printed verbatim by the CLI and
    pasted into the fishery YAML; a terse flag would be read as a formality
    and a number that means nothing would go on being used as though it did.
    """
    out: list[str] = []
    poor = []
    if np.isfinite(resolution) and resolution > 0 and rmse > resolution:
        poor.append(f"salinity rmse {rmse:.3f} ppt against a resolution of {resolution:.3f}")
    if (
        swing_rmse is not None
        and np.isfinite(swing_resolution)
        and swing_resolution > 0
        and swing_rmse > swing_resolution
    ):
        poor.append(
            f"tidal-swing rmse {swing_rmse:.3f} ppt against a resolution of "
            f"{swing_resolution:.3f}"
        )
    if poor:
        out.append(
            "POOR FIT: " + "; ".join(poor) + ". The resolution is the mean spacing "
            "between the distinct values the observations actually take -- derived "
            "from them, not authored -- and for quantised data it is the quantum "
            "exactly. Missing by more than the observations can resolve means the "
            "residual is structure the model cannot represent, not rounding. A "
            "config is not calibrated by parameters that reproduce nothing, however "
            "well-conditioned the fit that produced them."
        )
    if n_interior == 0:
        out.append(
            "NO INTERIOR OBSERVATIONS: every point sits at an end of the gradient "
            "(near-ocean or near-fresh), so the fit reproduces its anchors and "
            "constrains the shape between them barely at all -- which is where the "
            "fish are. Treat these parameters as order-of-magnitude, keep the "
            "climatology fallback live, and prefer any mid-bay observation over "
            "more end-member data. Registering the NERR CDMO IP would supply "
            "exactly this."
        )
    if n_distinct_d < 3 or distance_span < front_width_km:
        out.append(
            f"DISTANCE COVERAGE TOO THIN: {n_distinct_d} distinct along-estuary "
            f"distance(s) spanning {distance_span:.2f} km, against a fitted front "
            f"width of {front_width_km:.2f} km. The profile's SHAPE is what the "
            "scoring layer reads, and a shape cannot be fitted from points that do "
            "not span it -- l0_km, front_width_km and k trade off freely against "
            "each other here. "
            + (
                f"Note that n_interior_obs reports {n_interior} observation(s) as "
                "'interior', which is a VALUE test (a fraction of ocean_ppt) and is "
                "blind to this: all of them sit at the distance(s) named above. A "
                "healthy interior count is not evidence of a constrained profile. "
                if n_interior
                else ""
            )
            + "Do not let the numbers above talk you out of this: "
            "condition_number, param_sigma and at_bounds are all properties of the "
            "design that WAS observed and read perfectly healthy on exactly this "
            "failure (measured on Winyah's real 348-observation fit: condition 44.8, "
            "every 1-sigma below its value, nothing at a bound -- while four "
            "parameter sets whose rmse differed by 0.016 ppt predicted anywhere from "
            "19.8 to 34.0 ppt at North Jetty)."
        )
    ratio = cfs_span[1] / cfs_span[0] if cfs_span[0] > 0 else float("inf")
    if ratio < MIN_CFS_RATIO:
        out.append(
            f"DISCHARGE SPAN TOO NARROW: {cfs_span[0]:.0f}-{cfs_span[1]:.0f} cfs is a "
            f"factor of {ratio:.2f}. k is a power-law exponent in discharge and needs "
            f"at least a factor of {MIN_CFS_RATIO:.0f} to be more than an extrapolation."
        )
    if n_obs < 4 * n_params:
        out.append(
            f"FEW OBSERVATIONS: {n_obs} for {n_params} free parameters. The residual "
            "will look small because the model has almost as many knobs as points."
        )
    loose = [
        n
        for n, s in param_sigma.items()
        if s is None or not np.isfinite(s) or s > abs(fitted_values[n])
    ]
    if loose or condition > MAX_CONDITION:
        out.append(
            f"UNCONSTRAINED PARAMETERS: scaled-Jacobian condition number "
            f"{condition:.3g}"
            + (
                f"; 1-sigma exceeds the value itself for {', '.join(sorted(loose))}"
                if loose
                else ""
            )
            + ". The parameters are trading off against each other rather than being "
            "determined separately -- a different combination would fit these data "
            "about as well."
        )
    if at_bounds:
        out.append(
            f"AT THE OPTIMIZER BOUND: {', '.join(sorted(at_bounds))}. The data pushed "
            "these past the physical range the bounds enclose and the BOUND stopped "
            "them, not the evidence. A parameter resting on its bound is not a fitted "
            "value, and its 1-sigma describes a direction it was never free to move "
            "in -- do not read either as calibration."
        )
    if not has_swings:
        out.append(
            "EXCURSION NOT FITTED: no swing observations were supplied, so "
            "excursion_km was HELD at its incoming value, not fitted. At the "
            "daily-mean phase the tidal term is exactly zero, so a spatial-only "
            "objective has identically zero gradient in it -- freeing it here would "
            "return the starting value while looking like a fit. Task 3 measured "
            "that holding it at 7.0 km with a climatology-scale profile implies a "
            "22-29 ppt tidal swing at North Jetty and Mud Bay Cut, which is "
            "unphysical, so this value is unverified rather than merely unfitted."
        )
    return "\n\n".join(out)


# -- Assembling real observations -------------------------------------------
# Pure pairing below, network above it in `collect_observations`.


@dataclass
class SiteRecord:
    """One salinity sensor's place in the fit, and how well it has one."""

    site: str
    distance_km: float
    # Metres from the site to the nearest in-domain library cell. Zero-ish
    # means the site is in the domain; anything large means its distance was
    # borrowed from a cell that is not where the sensor is.
    snap_gap_m: float
    n_days: int
    ppt_range: tuple[float, float]
    used: bool
    note: str = ""


def build_site_record(
    site: str,
    rows: list[tuple[date, float]],
    *,
    located: bool,
    distance_km: float,
    snap_gap_m: float,
    max_snap_m: float,
    off_axis: bool = False,
    colocated: bool = False,
) -> SiteRecord:
    """One sensor's admission decision, and the REASON, which must be the
    real one.

    Order matters and is the whole point of this being a function. A site
    whose coordinates USGS never supplied is never queried for history
    either, so it arrives here with `rows == []` -- and a naive "no rows
    means no history" test would report a data-availability claim about a
    site whose actual problem is that nobody knows where it is. `located` is
    therefore tested FIRST, before anything that depends on having asked.

    `colocated` is tested before EVERYTHING else, including `off_axis`: a
    WQP station sitting on the same physical platform as an already-declared
    NERRS/NDBC/CDMO station (see `WQP_COLOCATION_RADIUS_M`) is redundant with
    that station's own record regardless of which side of the axis screen it
    would otherwise land on -- the reason is "this isn't a second site," not
    "the wrong branch" or "no coordinates."
    """
    ppt = [v for _, v in rows]
    if colocated:
        note = (
            f"co-located with an already-declared station within "
            f"{WQP_COLOCATION_RADIUS_M:.0f} m -- the same physical site's record, "
            "not a second one"
        )
    elif off_axis:
        # Tested BEFORE `located`, and before anything about data, because
        # this exclusion holds however good the station is: it is about
        # which estuary the reading belongs to, not whether it exists. A
        # station reporting perfectly at a well-known position is still
        # unusable here if the coordinate cannot place it -- and reporting
        # "no history" for it would be a false explanation of a real
        # modelling decision.
        note = "off the salt-intrusion axis -- a separate branch the 1-D coordinate cannot place"
    elif not located:
        note = "USGS gave no coordinates for this site"
    elif not np.isfinite(distance_km):
        note = "no water route to the sea from the nearest in-domain cell"
    elif not rows:
        note = "no 00480 history"
    elif snap_gap_m > max_snap_m:
        note = f"outside the domain by {snap_gap_m:.0f} m (limit {max_snap_m:.0f} m)"
    else:
        note = ""
    return SiteRecord(
        site=site,
        distance_km=distance_km,
        snap_gap_m=snap_gap_m,
        n_days=len(rows),
        ppt_range=(min(ppt), max(ppt)) if ppt else (float("nan"), float("nan")),
        used=not note,
        note=note,
    )


def composite_discharge_by_day(
    fishery: Fishery, daily: dict[str, list[tuple[date, float]]]
) -> dict[date, float]:
    """Weighted sum of the river gauges, per day, over days ALL gauges report.

    `weight` (include this gauge in the composite), not `inflow_share` (how to
    split a composite across branches) -- the two mean different things and
    Phase 1 Task 2 exists because they were once conflated. Days where a gauge
    is dark are dropped rather than summed short: a missing Pee Dee reading
    would otherwise arrive as a 78% drop in river flow, which the model reads
    as a salt front driving up the bay.
    """
    weights = {r.usgs_site: r.weight for r in fishery.rivers if r.usgs_site}
    if not weights or any(s not in daily for s in weights):
        return {}
    days = set.intersection(*(set(dict(daily[s])) for s in weights))
    return {
        d: sum(dict(daily[s])[d] * w for s, w in weights.items()) for d in sorted(days)
    }


def pair_daily_means(
    salinity_daily: dict[str, list[tuple[date, float]]],
    discharge_by_day: dict[date, float],
    distance_km: dict[str, float],
) -> list[Observation]:
    """(distance, that day's composite discharge, that day's mean salinity)."""
    return [
        (distance_km[site], discharge_by_day[day], ppt)
        for site, rows in sorted(salinity_daily.items())
        if site in distance_km
        for day, ppt in rows
        if day in discharge_by_day
    ]


def daily_swings(
    salinity_iv: dict[tuple[str, str], list[tuple[datetime, float]]],
    param: str,
    discharge_by_day: dict[date, float],
    distance_km: dict[str, float],
    min_readings: int = 40,
) -> list[Observation]:
    """Observed within-day salinity range, as a tidal-swing target.

    Max minus min of a day's instantaneous readings. It is a PROXY for the
    model's high-minus-low-water swing, not the same quantity: it also
    contains sensor noise and whatever the discharge did within the day. Days
    with fewer than `min_readings` samples are dropped, since a partial day
    understates the range and would drag the fitted excursion down.
    """
    out: list[Observation] = []
    for (site, p), points in sorted(salinity_iv.items()):
        if p != param or site not in distance_km:
            continue
        by_day: dict[date, list[float]] = {}
        for t, v in points:
            by_day.setdefault(t.date(), []).append(v)
        for day, values in sorted(by_day.items()):
            if len(values) < min_readings or day not in discharge_by_day:
                continue
            out.append((distance_km[site], discharge_by_day[day], max(values) - min(values)))
    return out


def is_off_axis(stem_km: float, declared: bool) -> bool:
    """Whether a station sits on a branch the 1-D coordinate cannot place.

    The COMPUTED distance decides; `declared` (the fishery YAML's
    `off_axis: true`) is an override that can only ever EXCLUDE. A hand flag
    able to force a station back IN would reintroduce the hand-marking this
    replaces -- and hand-marking every WQP station by eye is precisely the
    labour this screen exists to remove. One that can only exclude is a
    safety valve for geometry the criterion gets wrong.

    NaN excludes: it means the cell has no water route to the stem at all.
    """
    from tidescout.pipeline.estuary import ON_AXIS_MAX_KM

    if declared:
        return True
    if not np.isfinite(stem_km):
        return True
    return stem_km > ON_AXIS_MAX_KM


def station_stem_km(
    slug: str, fishery: Fishery, sites: Mapping[str, tuple[float, float]]
) -> dict[str, float]:
    """Distance from each {site: (lon, lat)} to the estuary's main stem."""
    from rasterio.warp import transform as warp_transform

    from tidescout.pipeline.estuary import load_stem_distance_field
    from tidescout.pipeline.flowlib import grid_spec

    if not sites:
        return {}
    spec = grid_spec(slug, fishery)
    stem = load_stem_distance_field(slug)
    ids = sorted(sites)
    xs, ys = warp_transform(
        "EPSG:4326",
        f"EPSG:{fishery.bathymetry.epsg}",
        [sites[s][0] for s in ids],
        [sites[s][1] for s in ids],
    )
    out = {}
    for site, x, y in zip(ids, xs, ys, strict=True):
        i = int(np.argmin((spec.xs - x) ** 2 + (spec.ys - y) ** 2))
        out[site] = float(stem[i])
    return out


def site_distances_km(
    slug: str, fishery: Fishery, sites: dict[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Along-estuary distance for each {site: (lon, lat)}, plus its snap gap.

    Nearest in-domain library cell, and how far away that cell actually is.
    The gap is returned rather than swallowed because a site outside the
    domain still gets an answer here -- the nearest cell's -- and that answer
    is not the site's distance. Callers decide what gap they will tolerate;
    this function will not decide it for them silently.
    """
    from rasterio.warp import transform as warp_transform

    from tidescout.pipeline.estuary import load_distance_field
    from tidescout.pipeline.flowlib import grid_spec

    spec = grid_spec(slug, fishery)
    field = load_distance_field(slug)
    ids = sorted(sites)
    xs, ys = warp_transform(
        "EPSG:4326",
        f"EPSG:{fishery.bathymetry.epsg}",
        [sites[s][0] for s in ids],
        [sites[s][1] for s in ids],
    )
    out = {}
    for site, x, y in zip(ids, xs, ys, strict=True):
        d2 = (spec.xs - x) ** 2 + (spec.ys - y) ** 2
        i = int(np.argmin(d2))
        out[site] = (float(field[i]), float(np.sqrt(d2[i])))
    return out


@dataclass
class CalibrationInput:
    observations: list[Observation]
    swings: list[Observation]
    sites: list[SiteRecord]
    days: int
    day_span: tuple[date, date] | None
    # Swings come from instantaneous readings, which USGS retains for a
    # shorter window than daily means -- see MAX_IV_DAYS.
    swing_days: int = 0
    # How many `sites` the stem screen removed -- counted from the site
    # records' own notes rather than tracked separately, so it can never
    # drift from what the table actually says.
    n_off_axis: int = 0
    # True when `station_stem_km` could not run at all because
    # `stem_km.npy` has not been built (`tidescout salinity stem <slug>`).
    # `off_axis` then fell back to whatever declared information existed
    # instead of a computed distance -- see `_stem_km_or_fallback`.
    stem_field_missing: bool = False
    # Same length and order as `observations` -- "usgs", "nerrs" or "wqp",
    # naming which route each row came from. Fed to `fit_intrusion` as
    # `sources=` so it can report `rmse_by_source_ppt`; plays no part in the
    # fit itself. Default `[]` (not populated) rather than an error, since
    # several existing tests hand-build a `CalibrationInput` with an empty
    # `observations` list and have no need of it.
    observation_sources: list[str] = field(default_factory=list)
    # How many WQP stations were excluded as co-located with an
    # already-declared NERRS/NDBC/CDMO station -- see
    # `WQP_COLOCATION_RADIUS_M`. Counted the same way as `n_off_axis`, from
    # the site records rather than tracked separately.
    n_colocated: int = 0


def daily_means_and_swings(
    series: Sequence[tuple[datetime, float]],
    tz: ZoneInfo,
    min_readings: int = MIN_DAILY_READINGS,
) -> tuple[dict[date, float], dict[date, float]]:
    """15-minute readings -> that day's mean and its within-day range.

    LOCAL days, not UTC. The store keeps UTC; grouping on the UTC date would
    push the four hours after 20:00 local into the next day, splitting every
    day's tidal cycle across two means and pairing each half with the wrong
    day's discharge.

    Both outputs are gated on the SAME reading count, for two reasons that
    point the same way. A partial day understates the RANGE, which would drag
    the fitted excursion down. It also biases the MEAN, because these readings
    are tidal: a handful of samples landing on one phase is offset by up to
    the full swing. Winyah's measured daily swing is 11.9 ppt median, so that
    bias is larger than the entire signal the fit is trying to resolve.
    """
    by_day: dict[date, list[float]] = {}
    for t, v in series:
        by_day.setdefault(t.astimezone(tz).date(), []).append(v)
    means, swings = {}, {}
    for day, values in by_day.items():
        if len(values) < min_readings:
            continue
        means[day] = float(np.mean(values))
        swings[day] = float(max(values) - min(values))
    return means, swings


def _open_store(slug: str):
    from tidescout.sources.ndbc import default_store

    return default_store(slug)


def _store_coords(sites: Sequence[str]) -> dict[str, tuple[float, float]]:
    """{station: (lon, lat)} for every store station whose surveyed position
    is known -- no discovery call, because these are not USGS sites and
    nothing serves their coordinates as data. Shared by `_store_distances`
    and `collect_observations`'s stem-distance lookup so the mapping is
    built once, not once per caller."""
    from tidescout.sources import cdmo

    return {
        s: cdmo.NIW_STATION_COORDS_LONLAT[s]
        for s in sites
        if s in cdmo.NIW_STATION_COORDS_LONLAT
    }


def _store_distances(
    slug: str, fishery: Fishery, sites: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Along-estuary distance for each store station, from its own surveyed
    position."""
    known = _store_coords(sites)
    return site_distances_km(slug, fishery, known) if known else {}


def _stem_km_or_fallback(
    slug: str, fishery: Fishery, sites: Mapping[str, tuple[float, float]]
) -> tuple[dict[str, float], bool]:
    """`station_stem_km`, degrading instead of crashing when the stem field
    has not been built.

    `load_stem_distance_field` raises `FileNotFoundError` until
    `tidescout salinity stem <slug>` has run once. Letting that propagate
    would break `salinity calibrate` outright on any machine that has not
    run it -- the spec calls for a derived screen, not a hard dependency.

    Returns `({}, False)` on that path so the caller can fall back to
    whatever `off_axis` information it already has without a computed stem
    distance: the YAML's declared flag for NERRS/NDBC stations (the
    pre-this-task behaviour), or exclusion for WQP stations, which have no
    declared flag to fall back to. The `bool` is carried onto
    `CalibrationInput.stem_field_missing` so the CLI reports the fallback
    rather than it looking like a clean, fully-screened run.
    """
    if not sites:
        return {}, True
    try:
        return station_stem_km(slug, fishery, sites), True
    except FileNotFoundError:
        return {}, False


def _wqp_sites(slug: str) -> dict[str, list[tuple[datetime, float]]]:
    """Every WQP station held for this fishery, with its salinity series.

    Discovered from the store rather than declared in the fishery YAML:
    there are ~130 of them, they arrive from a bbox query, and which ones
    are usable is decided by the stem screen, not by hand. `{}` when this
    fishery has never imported any WQP results -- checked by file existence
    rather than opening the store, which would otherwise create an empty
    `wqp.sqlite` as a side effect of a mere read (the same posture
    `wqp.station_coords` already takes).
    """
    from tidescout.paths import fishery_data_dir
    from tidescout.sources import wqp

    if not (fishery_data_dir(slug) / "wqp.sqlite").exists():
        return {}
    store = wqp.default_store(slug)
    return {s: store.salinity_series(s) for s in sorted(store.stations())}


# A WQP station sitting THIS close to an already-declared NERRS/NDBC/CDMO
# station is the same physical platform, sampled by a different agency, not
# a second site. Measured 2026-08-24 on the real store: WQP
# `21SC60WQ_WQX-WB-08` sits 2 m from WYSS1's declared coordinates, holds 15
# grab rows (2017-05-03..2018-09-05), and on every one of those dates WYSS1
# also has a full 96-reading daily mean at the same along-estuary distance,
# tracking closely (2018-07-18: WYSS1 mean 11.90 ppt vs WB-08 grab 11.69) --
# the same site's record, counted twice, at the same distance and the same
# day's discharge. 50 m sits comfortably above that 2 m measurement and far
# below any genuine separate site on this estuary (site records span
# kilometres, not tens of metres).
#
# WQP-to-WQP co-location is DELIBERATELY not covered by this constant --
# see `_colocated_wqp_stations`'s docstring for why collapsing those would
# discard real data rather than remove a duplicate.
WQP_COLOCATION_RADIUS_M = 50.0


def _colocated_wqp_stations(
    fishery: Fishery,
    store_coords: Mapping[str, tuple[float, float]],
    wqp_coords: Mapping[str, tuple[float, float]],
    radius_m: float = WQP_COLOCATION_RADIUS_M,
) -> set[str]:
    """WQP station ids within `radius_m` of an already-declared
    NERRS/NDBC/CDMO station (`fishery.stations.water`) -- the same physical
    platform's record, not a second site (see `WQP_COLOCATION_RADIUS_M` for
    the WB-08/WYSS1 measurement that sets the radius).

    WQP-to-STORE only, on purpose. Two WQP stations at the same coordinates
    -- e.g. `21SCSHL-05-24` / `21SCSHL_WQX-05-24`, the legacy/WQX ID split
    at WQP's own migration -- are NOT touched here: measured across all 48
    such pairs, zero (timestamp, value) rows are shared between the two IDs
    of a pair (WQP split each site's record at the migration, 1,305 rows
    under the WQX id and 3,584 under the legacy one), so they hold disjoint
    halves of ONE record. Collapsing them on distance alone would discard
    up to 3,584 real observations rather than remove a duplicate. The
    distinction this function draws is about WHAT is co-located with WHAT:
    a WQP station co-located with a declared STORE station is redundant
    with that station's own record; two WQP stations co-located with each
    other are not touched at all.
    """
    from rasterio.warp import transform as warp_transform

    if not store_coords or not wqp_coords:
        return set()
    epsg = fishery.bathymetry.epsg
    store_ids = sorted(store_coords)
    wqp_ids = sorted(wqp_coords)
    sx, sy = warp_transform(
        "EPSG:4326", f"EPSG:{epsg}",
        [store_coords[s][0] for s in store_ids], [store_coords[s][1] for s in store_ids],
    )
    wx, wy = warp_transform(
        "EPSG:4326", f"EPSG:{epsg}",
        [wqp_coords[s][0] for s in wqp_ids], [wqp_coords[s][1] for s in wqp_ids],
    )
    sx_arr, sy_arr = np.asarray(sx), np.asarray(sy)
    out = set()
    for site, x, y in zip(wqp_ids, wx, wy, strict=True):
        if np.any(np.hypot(sx_arr - x, sy_arr - y) <= radius_m):
            out.add(site)
    return out


def _usgs_inputs(slug, fishery, cache, days, start, end, max_snap_m):
    """The USGS half: salinity daily means, composite discharge, the sensors
    and their distances. Unchanged in behaviour from before the store
    existed -- it is still the only source with a live 00480 feed."""
    from tidescout.sources import discovery, usgs

    sensors = [
        w
        for w in fishery.stations.water
        if w.kind == "usgs" and usgs.PARAM_SALINITY in w.params
    ]
    river_sites = [r.usgs_site for r in fishery.rivers if r.usgs_site]
    discharge_daily = usgs.fetch_daily(
        river_sites, usgs.PARAM_DISCHARGE, str(start), str(end), cache
    )
    by_day = composite_discharge_by_day(fishery, discharge_daily)
    if not sensors:
        return {}, by_day, [], {}
    known = {s.id: (s.lon, s.lat) for s in discovery.find_usgs_sites(fishery, cache, "00480")}
    wanted = {w.station: known[w.station] for w in sensors if w.station in known}
    distances = site_distances_km(slug, fishery, wanted) if wanted else {}
    salinity_daily = usgs.fetch_daily(
        sorted(wanted), usgs.PARAM_SALINITY, str(start), str(end), cache
    )
    return salinity_daily, by_day, sensors, distances


def collect_observations(
    slug: str,
    fishery: Fishery,
    cache,
    days: int = 90,
    max_snap_m: float = 500.0,
) -> CalibrationInput:
    """Real observations from both routes, paired with distance and discharge.

    TWO SOURCES, deliberately different in how far back they reach:

    * USGS `00480`, fetched live over the last `days`. Specific conductance
      (`00095`) is a different quantity and is not interchangeable with it.
    * The NERRS store (`sources/ndbc.py`), read in FULL, whatever `days`
      says. That store exists to accumulate permanently -- NDBC's own feed
      is a ~45-day rolling window, so history not captured is gone -- and
      truncating it to a rolling window here would defeat the point of
      having it. The discharge series is then fetched to cover whichever
      route reaches furthest back, so both are paired against the same days.

    CO-OPS supplies no ocean end-member (Task 4 verified this live against
    every station within 250 km), so `ocean_ppt` is held rather than fitted;
    nothing here fetches one, and no default silently stands in for one.

    A THIRD source joins as of Task 5: WQP grab samples (`sources/wqp.py`),
    discovered from that fishery's own store rather than declared in the
    YAML. Unlike the other two these are single samples, not a series, so
    they are NOT run through `daily_means_and_swings` -- a one-sample day
    would fail its 40-reading gate, correctly. Each keeps its own exact
    timestamp and is paired with that day's composite discharge directly,
    which is what resolves it to a known distance, discharge AND tidal
    phase -- see `sources/wqp.py`'s module docstring for why a grab sample
    with all three is a fully-specified observation on its own.

    Four exclusions, all RECORDED rather than silent, because a fit that
    quietly narrows its own inputs looks identical to one that had less data:

    * A site further than `max_snap_m` from any in-domain cell -- its
      along-estuary distance would be the nearest cell's, which is a
      different place on the estuary.
    * A station `is_off_axis` against the COMPUTED distance to the main
      stem -- see that function's docstring. On Winyah that is North
      Inlet's three declared stations plus every WQP station on a branch
      the 1-D coordinate cannot place.
    * A WQP station WQP itself never reported a position for -- reported as
      "no coordinates", same as an unlocated USGS site.
    * A WQP station within `WQP_COLOCATION_RADIUS_M` of an already-declared
      NERRS/NDBC/CDMO station -- the same physical platform's record
      counted twice, not a second site. See `_colocated_wqp_stations`.
    """
    from tidescout.sources.ndbc import PARAM_SALINITY as STORE_SALINITY

    store_sensors = [
        w
        for w in fishery.stations.water
        if w.kind in _STORE_KINDS and STORE_SALINITY in w.params
    ]
    # A station declared off_axis=True is EXCLUDED unconditionally --
    # `is_off_axis` returns True for it no matter what the computed stem
    # distance says (declared can only ever exclude, never admit; see that
    # function's docstring) -- so its series is never worth fetching.
    on_axis = [w for w in store_sensors if not w.off_axis]

    tz = ZoneInfo(fishery.timezone)
    store_means: dict[str, dict[date, float]] = {}
    store_swings: dict[str, dict[date, float]] = {}
    if on_axis:
        store = _open_store(slug)
        for w in on_axis:
            means, swings = daily_means_and_swings(store.salinity_series(w.station), tz)
            store_means[w.station] = means
            store_swings[w.station] = swings

    wqp_series = _wqp_sites(slug)

    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    # The store reaches back further than `days` by design, so the discharge
    # window has to reach back with it or every older reading pairs with
    # nothing and is silently dropped. WQP grabs can reach back further
    # still (2005 was measured live), so their earliest dates widen it too.
    earliest = [min(m) for m in store_means.values() if m]
    earliest += [
        min(ts.astimezone(tz).date() for ts, _ in series)
        for series in wqp_series.values()
        if series
    ]
    if earliest:
        start = min(start, min(earliest))

    salinity_daily, by_day, usgs_sensors, usgs_dist = _usgs_inputs(
        slug, fishery, cache, days, start, end, max_snap_m
    )
    store_dist = _store_distances(slug, fishery, [w.station for w in store_sensors])
    store_coords = _store_coords([w.station for w in store_sensors])
    store_stem, store_stem_ok = _stem_km_or_fallback(slug, fishery, store_coords)

    from tidescout.sources import wqp as wqp_source

    wqp_coords = wqp_source.station_coords(slug)
    wqp_known = {s: wqp_coords[s] for s in wqp_series if s in wqp_coords}
    wqp_dist = site_distances_km(slug, fishery, wqp_known) if wqp_known else {}
    wqp_stem, wqp_stem_ok = _stem_km_or_fallback(slug, fishery, wqp_known)
    # See `WQP_COLOCATION_RADIUS_M`: a WQP station on the same physical
    # platform as an already-declared store station (measured: WB-08 sits
    # 2 m from WYSS1) would otherwise double-represent one site's record.
    colocated_wqp = _colocated_wqp_stations(fishery, store_coords, wqp_known)

    stem_field_missing = not (store_stem_ok and wqp_stem_ok)

    usable: dict[str, float] = {}
    records: list[SiteRecord] = []
    for w in usgs_sensors:
        record = build_site_record(
            w.station,
            salinity_daily.get(w.station, []),
            located=w.station in usgs_dist,
            distance_km=usgs_dist.get(w.station, (float("nan"), float("inf")))[0],
            snap_gap_m=usgs_dist.get(w.station, (float("nan"), float("inf")))[1],
            max_snap_m=max_snap_m,
        )
        if record.used:
            usable[w.station] = record.distance_km
        records.append(record)

    store_usable: dict[str, float] = {}
    for w in store_sensors:
        rows = sorted(store_means.get(w.station, {}).items())
        off_axis = (
            is_off_axis(store_stem.get(w.station, float("nan")), w.off_axis)
            if store_stem_ok
            # No computed distance available -- fall back to exactly the
            # declared flag, the behaviour before this task existed.
            else w.off_axis
        )
        record = build_site_record(
            w.station,
            rows,
            located=w.station in store_dist,
            distance_km=store_dist.get(w.station, (float("nan"), float("inf")))[0],
            snap_gap_m=store_dist.get(w.station, (float("nan"), float("inf")))[1],
            max_snap_m=max_snap_m,
            off_axis=off_axis,
        )
        if record.used:
            store_usable[w.station] = record.distance_km
        records.append(record)

    # WQP stations carry no declared off_axis flag at all -- they are
    # discovered, not authored (see `_wqp_sites`). A station with no known
    # position is never claimed off-axis (that would misreport WHY it is
    # excluded); `located` handles it below instead, same ordering
    # `build_site_record` already enforces for USGS sites.
    if wqp_stem_ok:
        wqp_off_axis = {
            s: is_off_axis(wqp_stem.get(s, float("nan")), False) for s in wqp_known
        }
    else:
        # No declared flag to fall back to and no computed distance either
        # -- admitting an unscreened branch station is exactly the
        # extrapolation this task exists to remove, so every locatable WQP
        # station is excluded rather than let through unscreened.
        wqp_off_axis = dict.fromkeys(wqp_known, True)

    wqp_usable: dict[str, float] = {}
    for site, series in sorted(wqp_series.items()):
        located = site in wqp_known
        dist, gap = wqp_dist.get(site, (float("nan"), float("inf")))
        rows = [(ts.astimezone(tz).date(), ppt) for ts, ppt in series]
        record = build_site_record(
            site,
            rows,
            located=located,
            distance_km=dist,
            snap_gap_m=gap,
            max_snap_m=max_snap_m,
            off_axis=wqp_off_axis.get(site, False),
            colocated=site in colocated_wqp,
        )
        if record.used:
            wqp_usable[site] = record.distance_km
        records.append(record)

    n_off_axis = sum(1 for r in records if "axis" in r.note.lower())
    n_colocated = sum(1 for r in records if "co-located" in r.note.lower())

    observations = pair_daily_means(salinity_daily, by_day, usable)
    sources = ["usgs"] * len(observations)
    store_obs = pair_daily_means(
        {s: sorted(store_means[s].items()) for s in store_usable}, by_day, store_usable
    )
    observations += store_obs
    sources += ["nerrs"] * len(store_obs)

    # WQP grabs are individual observations, not daily means -- each keeps
    # its own timestamp so it resolves to its own tidal phase rather than
    # being averaged into a day that (for most WQP stations) holds only it.
    #
    # KNOWN LIMITATION, not fixed here: `Observation` carries no tidal
    # phase, and `_levels`/`fit_intrusion` evaluate EVERY row -- daily mean
    # or grab alike -- at `FIT_PHASE` (0.25, the phase where the tidal term
    # is exactly zero). That is correct for a daily mean, which already
    # averages the tide out, but wrong for a single instantaneous grab: a
    # sample taken near high or low water is being scored as though it were
    # a tidal average when it is not one. The model's own tidal swing at the
    # distances where WQP grabs sit is 8.3-12.3 ppt, so phase mismatch alone
    # can account for ~4-6 ppt of error on a single grab. This is very
    # likely why the two populations residualise so differently (measured
    # 2026-08-24: NERRS daily means rmse 4.061 ppt on 10,880 rows vs WQP
    # grabs rmse 6.102 ppt on 1,860, mean residuals -0.594 vs -1.250 -- see
    # `rmse_by_source_ppt` in `fit_intrusion`'s diagnostics). The spec's
    # justification for grab samples was that each one "resolves to a known
    # distance, discharge AND tidal phase" -- this interface accepts the
    # first two and drops the third. A real fix needs historical tide
    # predictions spanning 1999-2026 (this repo currently predicts tides
    # forward from CO-OPS harmonics, not backward over WQP's date range) and
    # a phase-carrying change to `Observation`/`fit_intrusion` so each row
    # is scored at ITS OWN phase rather than the daily-mean phase -- a task
    # in its own right, not this one.
    wqp_obs_start = len(observations)
    for site, series in sorted(wqp_series.items()):
        if site not in wqp_usable:
            continue
        dist = wqp_usable[site]
        for ts, ppt in series:
            day = ts.astimezone(tz).date()
            if day in by_day:
                observations.append((dist, by_day[day], ppt))
    sources += ["wqp"] * (len(observations) - wqp_obs_start)

    swing_days = min(days, MAX_IV_DAYS)
    from tidescout.sources import usgs

    iv = (
        usgs.fetch_series(sorted(usable), [usgs.PARAM_SALINITY], swing_days, cache)
        if usable
        else {}
    )
    swings = daily_swings(iv, usgs.PARAM_SALINITY, by_day, usable)
    swings += [
        (store_usable[s], by_day[d], v)
        for s in sorted(store_usable)
        for d, v in sorted(store_swings.get(s, {}).items())
        if d in by_day
    ]
    span = (min(by_day), max(by_day)) if by_day else None
    return CalibrationInput(
        observations, swings, records, days, span, swing_days,
        n_off_axis=n_off_axis, stem_field_missing=stem_field_missing,
        observation_sources=sources, n_colocated=n_colocated,
    )
