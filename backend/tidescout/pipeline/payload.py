"""Assemble one day's full bite-score payload: 24 hours x 3 species, plus
every in-domain map feature, as a single JSON-serialisable dict.

This is the seam between the engine (Tasks 1-6, pure scoring functions) and
the frontend (spec section 3's API, section 9's map) -- one call produces
everything a day's UI needs so scrubbing the hour slider never refetches.

WHERE THIS FITS: `dayloader.load_day` assembles the day's raw conditions
(weather, tides, currents, sun/moon, water, discharge), wrapping every source
fetch so a single dead source degrades to `missing`, never a crash. This
module takes that `DayConditions`, resolves the flow-library regime blend,
maps each hour onto both tidal-phase conventions the engine needs (see
`_salinity_phase` and `_flow_events` below for why there are two), scores
the fishery-wide hour x species grid, and -- when a flow state is actually
resolvable for the day -- scores every in-domain feature the same way.

TWO PHASE CONVENTIONS, NOT ONE: `engine.phase.library_phase` (low-to-low
elapsed TIME, what the flow library is indexed by) and the salinity model's
own tidal term (`engine.salinity.salinity_at`'s cosine argument, which pins
high water at exactly 0.5) deliberately disagree by up to ~18 minutes -- see
`engine/phase.py`'s module docstring. This module computes each with its OWN
logic rather than reusing one for both.

WHY A SECOND `noaa.tide_events` FETCH: `DayConditions` does not carry the raw
`TideEvent` list `dayloader.load_day` fetched internally (it only keeps the
derived per-hour `tide_phase`/`tide_frac`, via `engine.tides.stage_at`) --
and `library_phase` needs the real events, not an approximation. Fetching
them again here, keyed the same way `load_day` already did, is a cache hit
in production (`Cache.get_or_fetch`'s whole point) and costs nothing extra
over a real run. `cache=None` skips the fetch outright rather than crashing
on a `None` cache or reaching the network, degrading gracefully to "no
flow-library state this run" for the map-feature half; the hourly
FISHERY-WIDE score is unaffected either way (`score_factors`'s own flow
factor already falls back to the CO-OPS current station).

Most tests in `test_payload.py` pass `cache=None` and so exercise the REAL
salinity distance field for `winyah-bay` (static, on-disk, no network) but
NOT the real flow library -- `test_the_feature_path_runs_against_the_real_
flow_library` (on the `synthetic_day_with_flow` fixture, `noaa.tide_events`
stubbed rather than `cache` set to `None`) is the one that does, and is the
only one slow enough (~70s) to notice. 2026-08-26 review, Important 2: an
earlier version of this docstring claimed the flow library was exercised by
every test in the file; `flowlib.load_state`, `activation.structure_fields`
and `activation.sample_features` were called zero times until that fixture
was added.

SALINITY: A SERIES, NOT A LAST-WRITE. The bay-wide (non-feature) salinity
reading is computed once per hour (`_bay_salinity_reading`) because the
tidal shift term in `engine.salinity.salinity_at` genuinely moves it across
the day -- on the modelled path it is not a constant. An earlier version of
this module kept only the LAST hour's reading, silently overwriting the
other 23 and publishing it unlabelled as `salinity.representative_ppt`
(2026-08-26 review, Important 7: measured 13.7 -> 35.3 ppt across a real
modelled day, reported as 23.5 with no hour attached, and hidden entirely on
Winyah today only because the live gauge pins `constrained_share` to 1.0 for
every hour -- see `_bay_salinity_reading`'s own docstring). `payload["salinity"]`
now carries the FULL per-hour `series` plus a `representative_ppt`/
`representative_hour` pair deliberately chosen as the day's MIDPOINT hour,
not whichever hour the loop happened to finish on.
"""

import json
import math
from datetime import UTC, date, datetime

import numpy as np

from tidescout.config import load_fishery, load_species
from tidescout.engine import flow, salinity
from tidescout.engine.activation import FeatureMetrics, _sampling_anchors, sample_features
from tidescout.engine.activation import structure_fields as compute_structure_fields
from tidescout.engine.phase import library_phase
from tidescout.engine.score import (
    SalinityProvenance,
    SalinityReading,
    SubScore,
    _recombine_tide_frac,
    combine,
    score_factors,
    score_feature,
)
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import flowlib
from tidescout.pipeline.estuary import load_distance_field
from tidescout.pipeline.features import load_features
from tidescout.pipeline.schedule import cell_schedule
from tidescout.sources import dayloader, noaa

# No module-level radius constant: `fishery.structure.ambush_radius_m` is
# THE per-fishery-tunable value `activation.structure_fields` itself reads
# (via `StructureThresholds`), and a second, hardcoded 150.0 here would
# silently diverge from it the moment a fishery's config moved away from the
# class default -- the exact class of bug `score_feature`'s own `thresholds`
# parameter (required, never defaulted) exists to prevent one level up
# (2026-08-26 review, Minor). Salinity is sampled over the SAME radius
# structure uses so a feature's "own" reading means one footprint for every
# factor, not a structural signal on one radius and a chemical one on
# another -- see `_feature_distance_km`'s call site in `build_payload`.


def _range_bucket_for_day(moon) -> str:
    """neap / mean / spring from the moon's illuminated fraction.

    No existing function maps a calendar day onto a tidal-RANGE bucket:
    `pipeline.regimes.RANGE_BUCKETS` only names the three buckets a library
    BUILD simulates, and `DayConditions` carries no observed tide-range
    series a payload could measure a real range from -- the shipped station
    is a subordinate one whose hourly heights are interpolated, and (see
    `tests/conftest.py`'s `synthetic_day` fixture) `tide_height_ft` is a flat
    placeholder there, not a range signal.

    `MoonInfo.phase_frac` is ephem's ILLUMINATED FRACTION (0.0 new .. 1.0
    full), not a 0..1 cycle angle -- both new and full moon read at the
    extremes, and quarter moons read near 0.5. Spring tides cluster at
    syzygy (new/full); neap at quadrature (either quarter). Distance from
    quadrature (0.0 at a quarter, 0.5 at new/full) is therefore the
    monotonic quantity to threshold, not `phase_frac` itself.
    """
    if moon is None:
        return "mean"
    dist_from_quadrature = abs(moon.phase_frac - 0.5)
    if dist_from_quadrature <= 0.15:
        return "neap"
    if dist_from_quadrature >= 0.35:
        return "spring"
    return "mean"


def _available_regimes(slug: str) -> set[str]:
    """Regime names with an actually-rasterised grid on disk.

    Deliberately NOT `library.json`'s key set: a regime can finish
    simulation (and so appear there) without `flowlib.rasterise_regime`
    having run for it yet, and `flow.blend_regimes` must not offer a regime
    it cannot actually load a state from.
    """
    base = fishery_data_dir(slug) / "flow"
    if not base.exists():
        return set()
    return {
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / "grid" / "grid.json").exists()
    }


def _flow_events(fishery, day: date, cache) -> tuple[list, bool]:
    """Real hi/lo `TideEvent`s for `library_phase` -- see the module
    docstring's "WHY A SECOND noaa.tide_events FETCH" for why this exists
    and why `cache=None` degrades rather than crashes.
    """
    station = fishery.stations.tide[0] if fishery.stations.tide else None
    if station is None or cache is None:
        return [], False
    try:
        return noaa.tide_events(station, day, fishery.timezone, cache), True
    except Exception:  # noqa: BLE001 -- any failure here degrades, never crashes the payload
        return [], False


def _salinity_phase(hour) -> float | None:
    """0 (low water) .. 1 (next low water), the convention
    `engine.salinity.salinity_at` depends on -- reuses
    `engine.score._recombine_tide_frac` rather than a second, independently
    written copy of the same half-to-full arithmetic (that function's own
    docstring: a second copy is "a second chance to get it wrong the same
    way", which is exactly what happened once already on this branch).
    """
    if hour.tide_frac is None or hour.tide_phase is None:
        return None
    return _recombine_tide_frac(hour)


def _is_extrapolated(effective_cfs: float, cfg) -> bool:
    """Stricter than `SalinityField.extrapolated`'s own boundary-INCLUSIVE
    check (`lo <= x <= hi` reads in-range): a discharge sitting EXACTLY on
    the edge of an unfitted, theoretical calibration range is exactly the
    case a disclosure should flag, not the case the model's own convention
    treats as safely inside. Winyah's `freshet_cfs` (22,996) now equals
    `calibration_range_cfs`'s own upper bound exactly, so the model's
    inclusive check reads it as in-range; this payload-level check does not.
    """
    lo, hi = cfg.calibration_range_cfs
    return not (lo < effective_cfs < hi)


def _bay_salinity_reading(day, fishery, distance_field: np.ndarray, phase: float | None):
    """The fishery-wide (non-feature) salinity reading: a sensor value when
    one exists, else the model evaluated at the domain's median along-estuary
    distance -- "bay-representative", per `score_factors`'s own module
    docstring ("Spatial when scoring a feature, bay-representative
    otherwise"). Per-feature readings never take this path; see
    `_feature_salinity_readings`, which always uses the feature's own
    distance regardless of whether a sensor exists.

    `usgs.water_summary` NEVER returns `salinity_ppt=None` -- it falls back
    to `fishery.climatology.salinity_ppt_by_month` (a monthly-average GUESS,
    not an observation) whenever no USGS sensor reported one. Treating that
    fallback as MEASURED would be exactly the silent-default this project's
    disclosure rules exist to catch, so `water.source == "climatology"` (the
    one signal `WaterSummary` exposes) is checked first. `source` is set
    from whichever sensor supplied TEMPERATURE, not necessarily salinity --
    on a fishery whose sensors disagree about which one reports which
    parameter, `source` could read a real station while salinity itself
    silently fell to climatology. Winyah Bay's own configured water sensors
    do not hit that gap today (021108125, tried first, reports both), so it
    is not exercised in practice; closing it fully would mean widening
    `WaterSummary` itself, which is `sources/usgs.py`, not this task's file.
    """
    water = getattr(day, "water", None)
    if water is not None and water.salinity_ppt is not None and water.source != "climatology":
        return SalinityReading(float(water.salinity_ppt), SalinityProvenance.MEASURED)

    discharge = getattr(day, "discharge", None)
    if discharge is None or discharge.cfs_now is None or phase is None:
        return None
    representative_km = float(np.nanmedian(distance_field))
    if not math.isfinite(representative_km):
        return None
    field = salinity.salinity_field(representative_km, discharge.cfs_now, phase, fishery.salinity)
    return SalinityReading(
        float(field.ppt),
        SalinityProvenance.MODELLED,
        fitted=fishery.salinity.fitted,
        extrapolated=_is_extrapolated(field.cfs, fishery.salinity),
    )


def _feature_distance_km(features: list[dict], spec, distance_grid: np.ndarray, radius_m: float):
    """Each feature's own along-estuary distance: the disc-mean of the
    static distance field over the SAME anchor/radius `sample_features` uses
    for structure. Computed ONCE (the distance field does not change hour to
    hour), so the per-hour salinity cost is just `salinity.salinity_at`
    evaluated over this already-reduced per-feature array -- cheap -- rather
    than repeating an O(features x domain cells) reduction 24 times.

    `_sampling_anchors` is `engine.activation`'s private helper, reused
    rather than re-derived: it is THE point that must match `detect.
    feature_key`'s hash anchor (see its own docstring), and a second,
    independently written reprojection-and-centroid routine is exactly the
    kind of silent-drift risk this codebase's docstrings repeatedly warn
    about paying twice for.
    """
    if not features:
        return np.array([], dtype="float64")
    xs, ys = _sampling_anchors(features, spec, already_projected=False)
    r2 = radius_m**2
    out = np.full(len(xs), np.nan, dtype="float64")
    for i, (fx, fy) in enumerate(zip(xs, ys, strict=True)):
        sel = (spec.xs - fx) ** 2 + (spec.ys - fy) ** 2 <= r2
        if not sel.any():
            continue
        here = distance_grid[sel]
        if np.isfinite(here).any():
            out[i] = float(np.nanmean(here))
    return out


def _feature_salinity_readings(
    feature_distance_km: np.ndarray, cfs: float | None, phase: float | None, fishery
) -> list[SalinityReading | None]:
    """One `SalinityReading` per feature (`None` where undeterminable),
    ALWAYS the spatial model estimate at the feature's own distance -- never
    the bay-wide sensor value, per `score_feature`'s own contract.
    """
    n = feature_distance_km.size
    if cfs is None or phase is None:
        return [None] * n
    ppt = salinity.salinity_at(feature_distance_km, cfs, phase, fishery.salinity)
    effective_cfs = max(float(cfs), 1.0)
    extrapolated = _is_extrapolated(effective_cfs, fishery.salinity)
    out: list[SalinityReading | None] = []
    for v in np.asarray(ppt, dtype="float64"):
        if not math.isfinite(v):
            out.append(None)
        else:
            out.append(
                SalinityReading(
                    float(v),
                    SalinityProvenance.MODELLED,
                    fitted=fishery.salinity.fitted,
                    extrapolated=extrapolated,
                )
            )
    return out


def _sub_to_dict(s: SubScore) -> dict:
    return {
        "factor": s.factor,
        "value": s.value,
        "weight": s.weight,
        "reason": s.reason,
        "missing": s.missing,
        "provisional": s.provisional,
    }


def _sub_to_trimmed_dict(s: SubScore) -> dict:
    """factor/value/reason only -- see the per-feature-hour call site's
    comment for why this is trimmed rather than the full `_sub_to_dict`."""
    return {"factor": s.factor, "value": s.value, "reason": s.reason}


def _hour_to_dict(hour_time: datetime, combined) -> dict:
    return {
        "time": hour_time.isoformat(),
        "score": combined.score,
        "subs": [_sub_to_dict(s) for s in combined.subs],
        "excluded": combined.excluded,
        "confidence": combined.confidence,
        "constrained_share": combined.constrained_share,
        "provisional": combined.provisional,
    }


def _json_safe(obj):
    """Recursively convert NaN/Inf -> None and numpy scalars -> native
    Python, so `json.dumps(..., allow_nan=False)` succeeds and nothing
    upstream has to know a numpy array ever touched this payload.

    NaN specifically, not just non-finite in general, is the documented
    failure mode (spec's JSON boundary forbids it outright); Inf is handled
    the same way since it is equally invalid JSON and equally meaningless as
    a bite score.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def _blended_state(slug: str, regime_phases: dict[str, list[float]], mix, phase: float) -> dict:
    """u/v/depth at `phase`, blended across the (at most two) regimes in
    `mix` -- phase-interpolated within each regime via `flow.bracket_phases`
    / `flow.interpolate_state`, then blended across regimes with the SAME
    function (it is a general two-state linear blend, not phase-specific).
    """
    per_regime = []
    for name, weight in mix:
        phases = regime_phases[name]
        i, j, w = flow.bracket_phases(phases, phase)
        state = flow.interpolate_state(
            flowlib.load_state(slug, name, i), flowlib.load_state(slug, name, j), w
        )
        per_regime.append((state, weight))
    state, _ = per_regime[0]
    total_w = per_regime[0][1]
    for extra_state, w in per_regime[1:]:
        total_w += w
        # Blend toward the running combination in proportion to the new
        # entry's share of the weight accumulated so far -- with exactly two
        # entries (the only case `blend_regimes` ever returns) this reduces
        # to `interpolate_state(state, extra_state, w)`.
        blend_w = w / total_w if total_w else 0.0
        state = flow.interpolate_state(state, extra_state, blend_w)
    return state


def build_payload(slug: str, day: date, model_label: str, cache) -> dict:
    """One (fishery, date, weather model)'s full scored payload.

    See the module docstring for the two-phase-convention design and why a
    second `tide_events` fetch happens here. Everything below is: load the
    day's conditions, resolve the discharge/range regime blend (needs no
    flow state -- it is a pure function of `day.discharge.cfs_now` and the
    fishery's authored buckets), then for every hour compute the two phase
    numbers, the bay-wide flow speed and salinity, the fishery-wide score
    for every species, and -- only when a flow state actually resolves for
    that hour -- every in-domain feature's activation for every species too.
    """
    fishery = load_fishery(slug)
    species = load_species()

    day_conditions = dayloader.load_day(fishery, day, model_label, cache)
    missing: list[str] = list(day_conditions.missing)

    range_bucket = _range_bucket_for_day(getattr(day_conditions, "moon", None))
    available = _available_regimes(slug)
    discharge = getattr(day_conditions, "discharge", None)
    mix: list[tuple[str, float]] = []
    if discharge is not None and discharge.cfs_now is not None and available:
        mix, _ = flow.blend_regimes(
            range_bucket, discharge.cfs_now, fishery.discharge_buckets, available
        )
    # A single-regime pin -- whether the discharge landed exactly on the
    # simulated envelope's edge or genuinely beyond it -- means no genuine
    # two-regime interpolation happened; see `_is_extrapolated`'s docstring
    # for the matching, deliberately boundary-inclusive-as-suspect choice on
    # the salinity side.
    clamped = len(mix) == 1

    flow_events, flow_events_ok = _flow_events(fishery, day, cache)
    if not flow_events_ok:
        missing.append("flow-library-phase")

    regime_phases: dict[str, list[float]] = {}
    for name, _ in mix:
        grid_path = fishery_data_dir(slug) / "flow" / name / "grid" / "grid.json"
        stored = json.loads(grid_path.read_text())["phases"]
        # The library stores a closing snapshot at phase 1.0 (recorded as
        # 0.0, the same value as the first) so every consecutive PAIR of
        # snapshots on disk spans one interpolatable gap, cycle-closing gap
        # included. `flow.bracket_phases` wraps the cycle itself (its own
        # `ordered[(i + 1) % len(ordered)]`), so it wants the phase axis
        # WITHOUT that duplicate -- fed the raw 26-entry list it sees
        # 0.966 -> 0.0 as a DESCENDING step and raises. Dropping the closing
        # duplicate loses no state: index 25's snapshot is bit-identical to
        # index 0's (same phase, same simulated instant one cycle later).
        if len(stored) > 1 and stored[-1] == stored[0]:
            stored = stored[:-1]
        regime_phases[name] = stored

    try:
        distance_field = load_distance_field(slug)
    except FileNotFoundError:
        distance_field = None
        missing.append("along-estuary-distance-field")

    # Structure fields/sample_features are only worth computing per-hour if a
    # flow state can actually be resolved for at least one hour -- see the
    # module docstring's "TWO PHASE CONVENTIONS" / "WHY A SECOND fetch"
    # sections for when that is (and is not) the case.
    spec = None
    feats: list[dict] = []
    feature_distance_km = np.array([], dtype="float64")
    schedule = None
    if mix and flow_events_ok and distance_field is not None:
        spec = flowlib.grid_spec(slug, fishery)
        feats = load_features(slug)["features"]
        feature_distance_km = _feature_distance_km(
            feats, spec, distance_field, fishery.structure.ambush_radius_m
        )
        dominant_regime = max(mix, key=lambda nw: nw[1])[0]
        schedule = cell_schedule(slug, dominant_regime)

    payload: dict = {
        "slug": slug,
        "day": day.isoformat(),
        "model_label": model_label,
        "missing": missing,
        "freshness": {
            "day": day.isoformat(),
            "model_label": model_label,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "flow": {
            "range_bucket": range_bucket,
            "discharge_cfs": discharge.cfs_now if discharge else None,
            "discharge_bucket": discharge.bucket if discharge else None,
            "regimes": [[name, weight] for name, weight in mix],
            "clamped": clamped,
        },
    }

    bay_cfs = discharge.cfs_now if discharge else None
    # Every hour's bay-wide salinity reading, not just the last one -- see
    # "SALINITY: A SERIES, NOT A SILENT LAST-WRITE" above.
    bay_salinity_series: list[dict] = []
    species_hours: dict[str, list] = {name: [] for name in species}
    species_features: dict[str, dict] = {name: {} for name in species}

    for hour in day_conditions.hours:
        lib_phase = library_phase(flow_events, hour.time) if flow_events_ok else None
        sal_phase = _salinity_phase(hour)

        state = None
        if mix and lib_phase is not None:
            state = _blended_state(slug, regime_phases, mix, lib_phase)

        bay_flow_speed = None
        if state is not None:
            sp, _ = flow.speed_direction(state["u"], state["v"])
            if np.isfinite(sp).any():
                bay_flow_speed = float(np.nanmean(sp))

        bay_reading = None
        if distance_field is not None:
            bay_reading = _bay_salinity_reading(day_conditions, fishery, distance_field, sal_phase)
        if bay_reading is not None:
            bay_salinity_series.append({
                "time": hour.time.isoformat(),
                "ppt": bay_reading.ppt,
                "provenance": bay_reading.provenance.value,
            })

        for name, profile in species.items():
            subs = score_factors(
                hour, day_conditions, profile, salinity=bay_reading, flow_speed=bay_flow_speed
            )
            combined = combine(subs)
            species_hours[name].append(_hour_to_dict(hour.time, combined))

        # Per-feature: only when a real flow state resolved for THIS hour --
        # `score_feature` needs `FeatureMetrics`, which needs the derived
        # structure fields, which need a flow state to derive them from.
        if state is not None and spec is not None and feats:
            fields = compute_structure_fields(
                state["u"], state["v"], state["depth"], spec, fishery.structure
            )
            metrics_list: list[FeatureMetrics] = sample_features(
                feats, spec, fields, schedule, fishery.structure.ambush_radius_m
            )
            feature_readings = _feature_salinity_readings(
                feature_distance_km, bay_cfs, sal_phase, fishery
            )
            for metrics, f_reading in zip(metrics_list, feature_readings, strict=True):
                if metrics.n_cells == 0:
                    continue
                for name, profile in species.items():
                    activation = score_feature(
                        metrics, hour, day_conditions, profile, f_reading, fishery.structure
                    )
                    bucket = species_features[name].setdefault(
                        metrics.key, {"type": metrics.type, "hours": []}
                    )
                    bucket["hours"].append(
                        {
                            "time": hour.time.isoformat(),
                            "activation": activation.activation,
                            "reason": activation.reason,
                            "confidence": activation.confidence,
                            "constrained_share": activation.constrained_share,
                            "excluded": activation.excluded,
                            "provisional": activation.provisional,
                            # Trimmed (factor/value/reason only, no
                            # weight/missing/provisional-per-sub -- those
                            # ride on `provisional`/`excluded` above instead)
                            # rather than omitted: spec section 8/9 covers
                            # BOTH the hourly and the per-feature score, and
                            # the marker popover's "what + why active"
                            # cannot be built from `reason` alone -- see the
                            # module docstring and 2026-08-26 review,
                            # Important 5. The full `SubScore` (weight,
                            # missing, per-sub provisional) is already on
                            # every FISHERY-WIDE hour instead; duplicating
                            # all of it across 529 features x 24 hours x 3
                            # species too would roughly double an already
                            # 13.4 MB payload for fields this trimmed form
                            # already covers.
                            "subs": [_sub_to_trimmed_dict(s) for s in activation.subs],
                        }
                    )

    payload["species"] = {
        name: {"hours": species_hours[name], "features": species_features[name]}
        for name in species
    }

    effective_cfs = max(float(bay_cfs), 1.0) if bay_cfs is not None else None
    # A DELIBERATE choice of hour, not whichever one a loop happened to
    # finish on -- see the module docstring's "SALINITY: A SERIES, NOT A
    # LAST-WRITE". The midpoint is not claimed to be more REPRESENTATIVE of
    # the day's fishing than any other hour; it is just a fixed, reproducible
    # index so `representative_ppt` never silently means "hour 23" again.
    # The full `series` is the honest answer for anything that cares about
    # the day's actual range.
    mid_reading = (
        bay_salinity_series[len(bay_salinity_series) // 2] if bay_salinity_series else None
    )
    payload["salinity"] = {
        "cfs": effective_cfs,
        "fitted": fishery.salinity.fitted,
        "extrapolated": (
            _is_extrapolated(effective_cfs, fishery.salinity)
            if effective_cfs is not None
            else False
        ),
        "series": bay_salinity_series,
        "representative_ppt": mid_reading["ppt"] if mid_reading else None,
        "representative_hour": mid_reading["time"] if mid_reading else None,
        "provenance": mid_reading["provenance"] if mid_reading else None,
    }

    return _json_safe(payload)
