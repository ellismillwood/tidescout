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
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

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
) -> tuple[SalinityConfig, dict]:
    """Least-squares fit of the intrusion model to `observations`.

    `observations` is [(distance_km, cfs, ppt)] -- daily means, evaluated at
    `FIT_PHASE`. `swings` is [(distance_km, cfs, ppt)] where the third value
    is an observed high-water-minus-low-water DIFFERENCE; supplying it is what
    frees `excursion_km` (see the module docstring). Both are fitted in the
    same ppt units and enter the objective with equal weight.

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


def collect_observations(
    slug: str,
    fishery: Fishery,
    cache,
    days: int = 90,
    max_snap_m: float = 500.0,
) -> CalibrationInput:
    """Real USGS observations, paired with distance and composite discharge.

    Only USGS `00480` -- specific conductance (`00095`) is a different
    quantity and is not interchangeable with it. CO-OPS supplies no ocean
    end-member (Task 4 verified this live against every station within
    250 km), so `ocean_ppt` is held rather than fitted; nothing here fetches
    one, and no default silently stands in for one.

    A site further than `max_snap_m` from any in-domain cell is recorded and
    EXCLUDED, not quietly snapped: its along-estuary distance would be the
    nearest cell's, which is a different place on the estuary.
    """
    from tidescout.sources import discovery, usgs

    sensors = [
        w
        for w in fishery.stations.water
        if w.kind == "usgs" and usgs.PARAM_SALINITY in w.params
    ]
    if not sensors:
        return CalibrationInput([], [], [], days, None)
    known = {s.id: (s.lon, s.lat) for s in discovery.find_usgs_sites(fishery, cache, "00480")}
    wanted = {w.station: known[w.station] for w in sensors if w.station in known}
    distances = site_distances_km(slug, fishery, wanted) if wanted else {}

    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    sites = sorted(wanted)
    salinity_daily = usgs.fetch_daily(sites, usgs.PARAM_SALINITY, str(start), str(end), cache)
    river_sites = [r.usgs_site for r in fishery.rivers if r.usgs_site]
    discharge_daily = usgs.fetch_daily(
        river_sites, usgs.PARAM_DISCHARGE, str(start), str(end), cache
    )
    by_day = composite_discharge_by_day(fishery, discharge_daily)

    usable, records = {}, []
    for w in sensors:
        site = w.station
        rows = salinity_daily.get(site, [])
        ppt = [v for _, v in rows]
        located = site in distances
        dist, gap = distances.get(site, (float("nan"), float("inf")))
        ok = bool(rows) and located and gap <= max_snap_m and np.isfinite(dist)
        if ok:
            usable[site] = dist
        records.append(
            SiteRecord(
                site=site,
                distance_km=dist,
                snap_gap_m=gap,
                n_days=len(rows),
                ppt_range=(min(ppt), max(ppt)) if ppt else (float("nan"), float("nan")),
                used=ok,
                note=(
                    ""
                    if ok
                    else "no 00480 history"
                    if not rows
                    else "USGS gave no coordinates for this site"
                    if not located
                    else "no water route to the sea from the nearest cell"
                    if not np.isfinite(dist)
                    else f"outside the domain by {gap:.0f} m (limit {max_snap_m:.0f} m)"
                ),
            )
        )

    observations = pair_daily_means(salinity_daily, by_day, usable)
    swing_days = min(days, MAX_IV_DAYS)
    iv = (
        usgs.fetch_series(sorted(usable), [usgs.PARAM_SALINITY], swing_days, cache)
        if usable
        else {}
    )
    swings = daily_swings(iv, usgs.PARAM_SALINITY, by_day, usable)
    span = (min(by_day), max(by_day)) if by_day else None
    return CalibrationInput(observations, swings, records, days, span, swing_days)
