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
    FACTORS,
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


def _measured_salinity_in_domain(source: str, fishery) -> bool:
    """Whether the sensor named in `WaterSummary.source` is the kind of
    reading that can stand for the reach the scoring layer actually reads.

    2026-08-26 review, Important 1: `source` alone (e.g. `"usgs:021108125"`)
    said nothing about WHERE that station sits relative to the model domain,
    so a sensor 9,498 m outside it -- snapped, by the along-estuary distance
    field, to the SAME cell as a second station 1,362 m outside it, both at
    the field's extreme fresh end -- was labelled `MEASURED` unconditionally.
    That makes `SalinityReading.constrained` True with no caveat, on the one
    factor this project has spent five PRs learning to disclose honestly.

    `WaterSensor.in_domain` is the authored verdict (see that field's
    docstring for why it is hand-set rather than computed here): this
    function only has to find the declared station matching `source` and
    read it, defaulting True (never disqualify a station this fishery has
    not explicitly flagged) when `source` names no declared USGS sensor at
    all -- a non-USGS or synthetic `source` (a test fixture's `"synthetic"`,
    for instance) is not this check's business.

    CALLERS MUST PASS THE SALINITY STATION, not `WaterSummary.source`.
    `source` names whichever sensor supplied TEMPERATURE, and this function
    cannot tell the difference -- it only looks a station id up. Gating on
    `source` was 2026-09-02 review Finding 5: on the shipped Winyah config
    the one in-domain USGS gauge (02136371) declares `params: ["00010"]`,
    temperature only, while both salinity-capable gauges are
    `in_domain: false`. A climatology-fallback salinity therefore passed
    this gate on the temperature station's credentials and shipped as
    `MEASURED` with no caveat. `WaterSummary.salinity_source` now carries
    the salinity station's own identity and is what `_bay_salinity_reading`
    hands in.
    """
    if not source.startswith("usgs:"):
        return True
    station_id = source.removeprefix("usgs:")
    for w in fishery.stations.water:
        if w.kind == "usgs" and w.station == station_id:
            return w.in_domain
    return True


def _bay_flow_speed(state) -> float | None:
    """The bay-representative current speed for one hour, over WET cells only.

    ANUGA reports a dry cell as u = v = 0, which is numerically
    indistinguishable from genuine slack water --
    `activation.structure_fields` masks exactly this case and its docstring
    says why. Averaged over the whole domain instead, the 17-19% of cells
    that are dry at any given phase drag the bay-wide speed down by the same
    fraction: measured on the real winyah-bay/mean_med library, phase 0 reads
    0.1127 m/s unmasked against 0.1355 wet-only. That is worth ~0.09 of the
    redfish flow sub-score, and enough to move the reason string across the
    0.1 m/s "slack" label boundary.

    Factored out of `build_payload` so the masking is testable without
    building a whole day (the same reason `_recombine_tide_frac` is its own
    function).
    """
    if state is None:
        return None
    sp, _ = flow.speed_direction(state["u"], state["v"])
    sp = np.where(flow.wet_mask(state["depth"]), sp, np.nan)
    if not np.isfinite(sp).any():
        return None
    return float(np.nanmean(sp))


def _bay_salinity_reading(day, fishery, distance_field: np.ndarray | None, phase: float | None):
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

    A second gap closed alongside the first (2026-08-26 review, Important
    1): `source` naming a real, live-reporting station is not by itself
    proof that station's number describes the reach being scored --
    `_measured_salinity_in_domain` also has to agree, or the reading falls
    through to the MODELLED path below exactly as a climatology fallback
    does. That path already carries `fitted`/`extrapolated`, the existing
    provenance machinery this project uses everywhere else to say "included,
    but not confidently" -- no new flag needed.
    """
    water = getattr(day, "water", None)
    # `salinity_source`, NOT `source`: the latter names whichever sensor
    # supplied TEMPERATURE, and the two are resolved by independent
    # per-parameter fallbacks in `usgs.water_summary`. On the shipped Winyah
    # config they routinely differ -- the only in-domain USGS gauge
    # (02136371) declares `params: ["00010"]`, so it can supply temperature
    # and never salinity, while both salinity-capable gauges are
    # `in_domain: false`. Gating on `source` therefore published a
    # CLIMATOLOGY salinity as MEASURED, `provisional=False`,
    # `constrained_share` 1.0, with no caveat in the reason, whenever the
    # two out-of-domain gauges had no data for the day and 02136371 did
    # (2026-09-02 review, Finding 5). `_historical_water_summary` requires
    # that specific calendar day per station, so one gauge gap on a hindcast
    # date was enough to reach it.
    salinity_source = getattr(water, "salinity_source", None) if water is not None else None
    if (
        water is not None
        and water.salinity_ppt is not None
        and salinity_source is not None
        and salinity_source != "climatology"
        and _measured_salinity_in_domain(salinity_source, fishery)
    ):
        # `fitted` is inert here -- `SalinityReading.constrained` short-
        # circuits on `provenance is MEASURED` before ever reading it -- but
        # the field is required (2026-08-26 review, Minor 5), so `True` is
        # passed explicitly rather than leaving a stale default to fall back
        # on. A real sensor reading has no model calibration to be unfitted.
        return SalinityReading(
            float(water.salinity_ppt), SalinityProvenance.MEASURED, fitted=True
        )

    discharge = getattr(day, "discharge", None)
    if discharge is None or discharge.cfs_now is None or phase is None:
        return None
    # Only THIS branch needs the distance field, which is why the caller no
    # longer gates the whole function on it -- a MEASURED in-domain sensor
    # above is answerable without one.
    if distance_field is None:
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


# Which factors are a property of the HOUR (and the species profile) versus
# of the FEATURE being scored. Measured, not assumed: on the real 2026-07-21
# payload, across all 529 in-domain features at every hour, each factor below
# in HOUR_SCOPE produced exactly ONE distinct {value, reason} per hour, while
# flow produced 493, salinity 528 and structure 420.
#
# Nothing in `score_factors` marks a factor as one or the other -- the split
# is a fact about what each factor READS, so it is stated here once and
# published in the payload as `sub_scope` rather than left for the frontend
# to rediscover.
HOUR_SCOPE_SUBS = frozenset(
    {"stage", "light", "solunar", "pressure", "wind", "water_temp", "season"}
)
# DERIVED, never hand-listed. `_feature_scope_subs` filters with the
# denylist above, so writing the allowlist out separately would let the two
# disagree: a newly added factor in neither set would be SHIPPED on every
# feature-hour (the filter keeps it, deliberately -- see that function) while
# `sub_scope` failed to declare it, and the payload would violate the very
# contract it publishes. Deriving both from one rule makes that unreachable.
#
# `structure` is not in `FACTORS`: it is the feature-only sibling scored by
# `score_feature`, not one of the nine hourly factors -- see
# `SpeciesProfile`'s docstring.
FEATURE_SCOPE_SUBS = frozenset({*FACTORS, "structure"}) - HOUR_SCOPE_SUBS


def _sub_to_trimmed_dict(s: SubScore) -> dict:
    """factor/value/reason only -- see the per-feature-hour call site's
    comment for why this is trimmed rather than the full `_sub_to_dict`."""
    return {"factor": s.factor, "value": s.value, "reason": s.reason}


def _feature_scope_subs(subs) -> list[dict]:
    """Only the subs that actually vary per feature.

    The other seven are identical to `species[name].hours[i].subs`' entries
    for the same hour and species -- verified across all 266,616 shared
    values on the real 2026-07-21 payload, zero mismatches -- so shipping
    them on all 529 features x 24 hours x 3 species stated something false
    (that they vary per feature) and cost 21.1 MB of a 47.9 MB payload to
    state it.

    A factor that is neither hour-scope nor feature-scope is kept rather
    than silently dropped: an unrecognised factor is a scope question
    nobody has answered yet, and dropping it would lose data to make a
    number smaller.
    """
    return [
        _sub_to_trimmed_dict(s) for s in subs if s.factor not in HOUR_SCOPE_SUBS
    ]


def _conditions_to_dict(hour) -> dict:
    """The raw hourly values, for the right rail and the tide-curve underlay.

    `_hour_to_dict` deliberately carries only scores and reasons; the raw
    numbers survive nowhere else in the payload, so §9's rail and tide curve
    have no source without this. Emitted ONCE at the top level rather than per
    species: these are fishery-wide hour facts, and duplicating them across
    three species would repeat the modelling error PR #11 corrected.
    """
    return {
        "time": hour.time.isoformat(),
        "air_temp_f": hour.air_temp_f,
        "wind_speed_kn": hour.wind_speed_kn,
        "wind_dir_deg": hour.wind_dir_deg,
        "wind_gust_kn": hour.wind_gust_kn,
        "pressure_mb": hour.pressure_mb,
        "pressure_trend_mb_3h": hour.pressure_trend_mb_3h,
        "cloud_cover_pct": hour.cloud_cover_pct,
        "precip_in": hour.precip_in,
        "tide_height_ft": hour.tide_height_ft,
        "tide_phase": hour.tide_phase,
        "tide_frac": hour.tide_frac,
    }


def _water_to_dict(water) -> dict | None:
    """The day-level water-temperature summary, a sibling of `payload["salinity"]`
    rather than a field inside it.

    Spec §5 asks the rail to carry water temperature alongside air
    temperature; `HourlyConditions` has no per-hour water reading (the
    engine treats water temperature, like salinity, as a slow-moving DAY
    fact rather than something that changes hour to hour), so this reads
    `DayConditions.water` once instead.

    Deliberately narrowed to `temp_f`/`temp_trend_f_3d` -- `salinity_ppt`,
    `source` and `salinity_source` are withheld even though `WaterSummary`
    carries them, because `payload["salinity"]` already publishes that same
    reading WITH its provenance (MEASURED vs MODELLED, fitted, extrapolated
    -- see `_bay_salinity_reading`). A second, provenance-free copy of the
    same number here would give a reader two places to look for one fact
    and no way to tell which one to trust if they ever disagreed.

    `day_conditions.water` is `None` whenever no water sensor and no
    climatology fallback produced a reading at all; returned as `None`
    here too rather than raising, the same degrade-gracefully contract
    every other optional source in this payload follows.
    """
    if water is None:
        return None
    return {"temp_f": water.temp_f, "temp_trend_f_3d": water.temp_trend_f_3d}


def _astro_to_dict(sun, moon) -> dict:
    """The day's sun and moon times, the other sibling spec §5 asks for
    beside `payload["salinity"]` and `payload["water"]`.

    `SunTimes` and `MoonInfo` are both computed once per DAY (see
    `sources.astronomy`), never once per hour, so this reads
    `DayConditions.sun`/`.moon` directly rather than being folded into
    `_conditions_to_dict` -- publishing either in all 24 hourly rows would
    repeat one day fact 24 times, the exact duplication this task's own
    hour/day split exists to avoid (see the module docstring's "SALINITY: A
    SERIES" section for the same reasoning applied to a genuinely hourly
    value, which water/astro are not).

    `sun` and `moon` can independently be `None` -- two different upstream
    sources, wrapped in `dayloader.load_day`'s own single-dead-source
    degrade-gracefully contract -- so each field is guarded on its own
    source rather than assuming both arrived together. `moon.rise`/`.set`
    are independently optional even when `moon` itself is present (a moon
    that neither rises nor sets on a given calendar day is ephem's own
    convention, not a bug here), so `moonrise`/`moonset` guard one level
    deeper than `moon_phase_frac` does.
    """
    return {
        "dawn": sun.dawn.isoformat() if sun else None,
        "sunrise": sun.sunrise.isoformat() if sun else None,
        "sunset": sun.sunset.isoformat() if sun else None,
        "dusk": sun.dusk.isoformat() if sun else None,
        "moon_phase_frac": moon.phase_frac if moon else None,
        "moonrise": moon.rise.isoformat() if moon and moon.rise else None,
        "moonset": moon.set.isoformat() if moon and moon.set else None,
    }


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
        # WHICH FACTORS LIVE WHERE. A feature-hour ships only the subs that
        # vary per feature; the other seven are on `species[name].hours[i]`,
        # once per hour instead of once per feature-hour. A reader wanting a
        # marker popover's full ten-factor breakdown merges the two by
        # POSITION (feature `hours[i]` <-> species `hours[i]`).
        #
        # Published rather than left implicit so the frontend does not
        # hardcode the split and silently break the day a factor changes
        # scope -- e.g. if salinity ever became bay-wide, or a new factor
        # arrived that reads a feature's own geometry.
        "sub_scope": {
            "hour": sorted(HOUR_SCOPE_SUBS),
            "feature": sorted(FEATURE_SCOPE_SUBS),
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

        bay_flow_speed = _bay_flow_speed(state)

        # NOT gated on `distance_field`: only the MODELLED branch of
        # `_bay_salinity_reading` needs it (for `np.nanmedian`), and that
        # branch returns None on its own when it cannot run. Gating the whole
        # call dropped the salinity factor for all 24 hours x every species
        # whenever `estuary_km.npy` was absent, even with a live in-domain
        # sensor whose reading needs no distance field at all.
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
                        metrics, hour, day_conditions, profile, f_reading,
                        fishery.structure, lib_phase,
                    )
                    bucket = species_features[name].setdefault(
                        metrics.key, {"type": metrics.type, "hours": []}
                    )
                    bucket["hours"].append(
                        {
                            # No `time`: a feature-hour is positionally
                            # aligned with `species[name].hours[i]`, whose
                            # `time` it always equalled anyway (verified on
                            # the real payload, all species, all 529
                            # features, zero exceptions). Storing it 38,088
                            # times also implied a feature-hour could carry
                            # its own timestamp, which it cannot.
                            "activation": activation.activation,
                            "reason": activation.reason,
                            "confidence": activation.confidence,
                            "constrained_share": activation.constrained_share,
                            "excluded": activation.excluded,
                            "provisional": activation.provisional,
                            # SCOPED, then trimmed. `_feature_scope_subs`
                            # drops the seven hour-scope factors entirely
                            # (they are on `species[name].hours[i]`, once
                            # per hour rather than once per feature-hour);
                            # what survives is trimmed to factor/value/
                            # reason, without weight/missing/per-sub
                            # provisional, which ride on `provisional`/
                            # `excluded` above.
                            #
                            # Not omitted altogether: spec section 8/9
                            # covers BOTH the hourly and the per-feature
                            # score, and the marker popover's "what + why
                            # active" cannot be built from `reason` alone
                            # (2026-08-26 review, Important 5).
                            #
                            # SIZE, measured on the real 2026-07-21 payload
                            # (2026-09-02, Phase 4). Before scoping: 47.86
                            # MB compact JSON, 2.34 MB gzip -6, 83 ms
                            # JSON.parse, 75.7 MB JS heap. After: 24.59 MB,
                            # 1.67 MB gzipped, 30 ms, 40.1 MB heap. The
                            # frontend contract is the gzipped number, and
                            # the heap is what actually constrains a device
                            # -- desktop is comfortable at 40 MB even with
                            # two or three days resident. Revisit only if
                            # this goes mobile-first: the trigger is a
                            # second day held in memory beside a MapLibre
                            # GL context, not a byte count.
                            "subs": _feature_scope_subs(activation.subs),
                        }
                    )

    payload["species"] = {
        name: {"hours": species_hours[name], "features": species_features[name]}
        for name in species
    }
    payload["conditions"] = [_conditions_to_dict(h) for h in day_conditions.hours]

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
    # Day-level siblings of `salinity` above -- see `_water_to_dict`/
    # `_astro_to_dict` for why these are day facts, not per-hour ones.
    payload["water"] = _water_to_dict(getattr(day_conditions, "water", None))
    payload["astro"] = _astro_to_dict(
        getattr(day_conditions, "sun", None), getattr(day_conditions, "moon", None)
    )

    return _json_safe(payload)
