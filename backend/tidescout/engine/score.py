"""The factor pipeline. Pure -- conditions in, sub-scores and reasons out.

Two consumers share it: the fishery-wide hourly score and the per-feature
activation. They differ only in where `flow_speed` and `salinity` come
from -- the bay's representative values, or one feature's own.

Every factor obeys the same contract:
  - its weight comes from the species profile, never from code;
  - its response shape comes from a YAML curve, never from code;
  - a missing input yields missing=True and NaN, never 0.0, because "no data"
    and "conditions are dead" are different claims and spec section 8 requires
    the first to renormalise rather than score;
  - it always returns a reason, because the UI renders factor bars with text.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from tidescout.engine.activation import FeatureMetrics
from tidescout.engine.curves import evaluate
from tidescout.models import SpeciesProfile, StructureThresholds

FACTORS = (
    "flow", "stage", "light", "solunar", "pressure", "wind",
    "water_temp", "salinity", "season",
)

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


class SalinityProvenance(StrEnum):
    """WHERE a salinity number came from, which is not the same question as
    whether it is in range."""

    MEASURED = "measured"   # a sensor read it -- `day.water.salinity_ppt`
    MODELLED = "modelled"   # `engine.salinity` computed it


@dataclass(frozen=True)
class SalinityReading:
    """A salinity value WITH the provenance a scorer needs to be honest.

    A bare float cannot distinguish a sensor reading from an uncalibrated
    model estimate, and this model is uncalibrated: `SalinityConfig.fitted`
    is False for Winyah Bay. Passing a float here would reproduce, at the
    factor level, exactly the confusion `SalinityField` was built to prevent.
    """

    ppt: float
    provenance: SalinityProvenance
    # Did a calibration EVER constrain these parameters. Config-level, so it
    # is identical at every cell and hour. False for Winyah Bay today.
    fitted: bool = True
    # Was THIS evaluation's discharge outside `calibration_range_cfs`.
    # Per-evaluation; changes with the river. Independent of `fitted`.
    extrapolated: bool = False

    @property
    def constrained(self) -> bool:
        """True only when the number is worth stating without a caveat."""
        return self.provenance is SalinityProvenance.MEASURED or (
            self.fitted and not self.extrapolated
        )


@dataclass
class SubScore:
    factor: str
    value: float
    weight: float
    reason: str
    missing: bool
    # Scored, and counted at full weight, but nothing observed constrains it.
    # DISTINCT from `missing`: missing means excluded and renormalised;
    # provisional means included and disclosed. A UI that renders these the
    # same way is not implementing spec section 10.
    provisional: bool = False


def _missing(factor: str, profile: SpeciesProfile, what: str) -> SubScore:
    return SubScore(factor, float("nan"), profile.weights[factor],
                    f"{factor}: no data ({what})", True)


def _scored(factor: str, profile: SpeciesProfile, x: float, reason: str) -> SubScore:
    return SubScore(factor, evaluate(profile.curves[factor], x),
                    profile.weights[factor], reason, False)


def _hours_from_twilight(t: datetime, sun) -> float | None:
    """Hours to the nearer of sunrise and sunset. Zero at the edge of light."""
    if sun is None or sun.sunrise is None or sun.sunset is None:
        return None
    return min(
        abs((t - sun.sunrise).total_seconds()), abs((t - sun.sunset).total_seconds())
    ) / 3600.0


def _minutes_from_solunar(t: datetime, periods) -> float | None:
    """Minutes to the nearest solunar major or minor."""
    if not periods:
        return None
    return min(abs((t - p.start).total_seconds()) for p in periods) / 60.0


def _recombine_tide_frac(hour) -> float:
    """Half-cycle `tide_frac` -> the FULL 0 (low water) .. 1 (next low
    water) fraction the stage curve is authored against.

    ONLY the arithmetic, factored out of the stage factor below so the
    per-feature flat-wetness gate (`_flat_wet_multiplier`) can read the same
    number the stage factor reads, rather than re-deriving it -- this exact
    half-to-full recombination was the branch's one shipped spec violation
    (2026-08-26 review, Important 1: the stage factor labelled 9 of 19 hours
    on the wrong tidal half before the fix), so a second, independently
    written copy of it is a second chance to get it wrong the same way.

    Callers must confirm `hour.tide_frac` and `hour.tide_phase` are both
    present before calling this -- exactly as the stage factor's own
    `if hour.tide_frac is None: ... elif hour.tide_phase is None: ...` does
    below -- because this function has no missing-data branch of its own to
    stay in sync with that one.
    """
    return hour.tide_frac / 2 if hour.tide_phase == "rising" else 0.5 + hour.tide_frac / 2


def score_factors(
    hour,
    day,
    profile: SpeciesProfile,
    salinity: SalinityReading | None = None,
    flow_speed: float | None = None,
) -> list[SubScore]:
    """All nine sub-scores for one hour. Always nine, some possibly missing."""
    subs: list[SubScore] = []

    # 1. Tidal flow rate. Prefers the flow library's speed; falls back to the
    # CO-OPS current station, which is a single point and cannot describe the
    # bay, but beats nothing.
    speed = flow_speed
    if speed is None and hour.current_speed_kn is not None:
        speed = hour.current_speed_kn * 0.514444
    if speed is None:
        subs.append(_missing("flow", profile, "no flow state or current station"))
    else:
        kind = "slack" if speed < 0.1 else "moving" if speed < 0.8 else "ripping"
        subs.append(_scored("flow", profile, speed, f"flow {speed:.2f} m/s — {kind}"))

    # 2. Tide stage. `stage_at`'s `frac` resets to 0 at every hi/lo turn --
    # it is a HALF-cycle fraction between whichever pair of events brackets
    # this hour, NOT the full 0 (low water) .. 1 (next low water) fraction
    # the YAML curves are authored against (species_weights.yaml: "stage
    # tide_frac (0 = low water, 0.5 = high)"). `tide_phase` says which half
    # we are in, so it recombines the two: half the half-cycle frac while
    # flooding (rising, heading to the next high water), and 0.5 plus half
    # of it while ebbing (falling, heading to the next low). `tide_phase`
    # can be None independently of `tide_frac` (see `stage_at`), so it is
    # guarded on its own rather than folded into one combined check that
    # could silently guess a half.
    # (2026-08-26 review, Important 1 -- this comment previously, and
    # wrongly, claimed `tide_frac` already matched the flow library's phase
    # convention; `library_phase` is a different quantity than
    # `stage_at().frac` and conflating them left the curve seeing the same
    # x on flood and ebb, unable to express any direction bias at all.)
    if hour.tide_frac is None:
        subs.append(_missing("stage", profile, "no tide prediction"))
    elif hour.tide_phase is None:
        subs.append(_missing("stage", profile, "no tide phase"))
    else:
        full = _recombine_tide_frac(hour)
        half = "flooding" if hour.tide_phase == "rising" else "ebbing"
        subs.append(_scored("stage", profile, full,
                            f"tide {full:.2f} of cycle — {half}"))

    # 3. Light. Cloud cover widens the low-light window, so heavy cloud is
    # credited as bringing the hour closer to twilight. The curve is
    # evaluated at `effective`, the cloud-widened value -- the reason quotes
    # THAT number, not the raw `hours_off`, or it would describe a different
    # score than the one actually produced (2026-08-26 review, Minor 3).
    hours_off = _hours_from_twilight(hour.time, getattr(day, "sun", None))
    if hours_off is None:
        subs.append(_missing("light", profile, "no sun times"))
    else:
        # Explicit `is None` check, not `cloud_cover_pct or 0.0`: `or` would
        # collapse a genuine 0.0% reading into the same branch as "no cloud
        # data," which is a different claim (2026-08-26 review, Minor 4).
        # Missing cloud data still degrades to "no widening" rather than
        # marking the whole light factor missing -- cloud is a refinement
        # of this factor, not a precondition for scoring it.
        cloud = hour.cloud_cover_pct if hour.cloud_cover_pct is not None else 0.0
        effective = hours_off * (1.0 - 0.35 * cloud / 100.0)
        # Disclosed whenever the adjustment is non-zero, not only above some
        # threshold -- a silent +0.070 at cloud=50 under the old `> 50`
        # cutoff is exactly the kind of right-value-wrong-justification gap
        # this project has been bitten by before.
        note = f", {cloud:.0f}% cloud widened it from {hours_off:.1f} h" if cloud > 0 else ""
        subs.append(_scored("light", profile, effective,
                            f"{effective:.1f} h from twilight{note}"))

    # 4. Solunar. Smallest default weight of the nine, per spec section 8.
    mins = _minutes_from_solunar(hour.time, getattr(day, "solunar", None))
    if mins is None:
        subs.append(_missing("solunar", profile, "no solunar periods"))
    else:
        subs.append(_scored("solunar", profile, mins,
                            f"{mins:.0f} min from a solunar period"))

    # 5. Pressure trend.
    if hour.pressure_trend_mb_3h is None:
        subs.append(_missing("pressure", profile, "no pressure trend"))
    else:
        p = hour.pressure_trend_mb_3h
        note = ("falling — pre-frontal feeding window" if p < -0.5
                else "rising sharply — post-frontal shutdown" if p > 2.0
                else "steady")
        subs.append(_scored("pressure", profile, p,
                            f"pressure {p:+.1f} mb/3h — {note}"))

    # 6. Wind.
    if hour.wind_speed_kn is None:
        subs.append(_missing("wind", profile, "no wind forecast"))
    else:
        w = hour.wind_speed_kn
        note = ("calm" if w < 5 else "light" if w < 12
                else "building" if w < 18 else "hard — fishability suffers")
        subs.append(_scored("wind", profile, w, f"wind {w:.0f} kn — {note}"))

    # 7. Water temperature.
    water = getattr(day, "water", None)
    temp = getattr(water, "temp_f", None) if water else None
    if temp is None:
        subs.append(_missing("water_temp", profile, "no water sensor or climatology"))
    else:
        trend = getattr(water, "temp_trend_f_3d", None)
        note = ""
        if trend is not None and abs(trend) >= 1.0:
            note = f", {'warming' if trend > 0 else 'cooling'} {abs(trend):.1f}F/3d"
        subs.append(_scored("water_temp", profile, temp, f"water {temp:.0f}F{note}"))

    # 8. Salinity. Spatial when scoring a feature, bay-representative otherwise.
    # The value is scored the same either way; only the REASON changes, and it
    # must change, because an uncalibrated model estimate and a sensor reading
    # are different claims about the world.
    if salinity is None or not math.isfinite(salinity.ppt):
        subs.append(_missing("salinity", profile, "no salinity estimate"))
    else:
        ppt = salinity.ppt
        note = "near-fresh" if ppt < 5 else "brackish" if ppt < 18 else "salty"
        if salinity.constrained:
            reason = f"salinity {ppt:.1f} ppt — {note}"
        else:
            caveats = []
            if not salinity.fitted:
                caveats.append("UNCALIBRATED model estimate, no observation constrains it")
            if salinity.extrapolated:
                caveats.append("discharge outside the calibrated range")
            # "~" on the number as well as the caveat: the tilde survives
            # truncation in a narrow UI column, the parenthetical may not.
            reason = f"salinity ~{ppt:.1f} ppt — {note} ({'; '.join(caveats)})"
        subs.append(SubScore("salinity", evaluate(profile.salinity, ppt),
                             profile.weights["salinity"], reason, False,
                             provisional=not salinity.constrained))

    # 9. Season. A table lookup, not a curve -- months are discrete.
    month = hour.time.month
    subs.append(SubScore("season", float(profile.months[month]),
                         profile.weights["season"],
                         f"{MONTH_NAMES[month - 1]} (month {month}) seasonal modifier",
                         False))

    return subs


# Floor applied to a sub-score before taking its log. log(0) is -inf, which
# would turn the whole day payload into NaN. 1e-3 still tanks the hour -- with
# nine equal weights it pulls a perfect score to about 46 -- so this is a guard
# against a numerical edge, not a rescue from a bad factor.
SCORE_FLOOR = 1e-3


@dataclass
class HourScore:
    score: int              # 0-100
    subs: list[SubScore]
    excluded: list[str]     # factors dropped for missing data
    confidence: float       # share of total authored weight that survived
    # Share of SURVIVING weight that is actually constrained by an observation.
    # Distinct from `confidence`: a provisional factor survives (so it does not
    # move `confidence`) while contributing nothing trustworthy (so it does
    # move this). On Winyah Bay today, a full-data hour reads confidence 1.0
    # and constrained_share well below it, which is the honest pair.
    constrained_share: float
    provisional: list[str]
    # The geometric mean BEFORE the one rounding-and-scaling step that turns
    # it into `score`, in [0, 1]. Exists so a caller that needs to multiply
    # the score by something else (`score_feature`'s flat-wetness gate) can
    # do that multiply in the continuous domain and round ONCE, at the
    # boundary, instead of rounding here and then again after multiplying an
    # already-rounded integer -- the exact double-rounding bug 2026-08-26's
    # review measured differing from the correct answer on 26 of 196 real
    # flats (Finding 9). `score` stays as the public 0-100 display value;
    # `raw` is the number a second stage of arithmetic should actually use.
    raw: float


def combine(subs: list[SubScore]) -> HourScore:
    """Weighted geometric mean of the present factors, as 0-100.

    Geometric, not arithmetic, because spec section 8 requires a near-zero
    critical factor to tank the hour rather than average away: dead slack water
    or a cold shock should not be rescued by a pleasant sky. Arithmetically,
    (0 + 1 + 1)/3 is a respectable 0.67; geometrically it is ~0.

    Missing factors are EXCLUDED and the remaining weights renormalised -- never
    defaulted to a middling value, which would invent data. `confidence` reports
    how much of the authored weight survived, so the UI can show that an hour
    scored 82 on six of nine factors.

    `constrained_share` answers the OTHER question: of the weight that did
    survive, how much rests on something observed. A provisional factor
    (scored, full weight, but unconstrained -- see `SalinityReading`) counts
    toward `confidence` and against this. On Winyah Bay today an all-factors
    hour reads confidence 1.0 with constrained_share well below it, and
    collapsing the two into one number would hide exactly that.
    """
    present = [s for s in subs if not s.missing and math.isfinite(s.value)]
    excluded = [s.factor for s in subs if s.missing or not math.isfinite(s.value)]
    total_weight = sum(s.weight for s in subs) or 1.0
    live_weight = sum(s.weight for s in present)

    if not present or live_weight <= 0:
        return HourScore(
            score=0, subs=subs, excluded=excluded, confidence=0.0,
            constrained_share=0.0, provisional=[], raw=0.0,
        )

    log_sum = sum(s.weight * math.log(max(s.value, SCORE_FLOOR)) for s in present)
    raw = min(max(math.exp(log_sum / live_weight), 0.0), 1.0)
    return HourScore(
        score=int(round(100 * raw)),
        subs=subs,
        excluded=excluded,
        confidence=live_weight / total_weight,
        constrained_share=(
            sum(x.weight for x in present if not x.provisional) / live_weight
        ),
        provisional=[x.factor for x in present if x.provisional],
        raw=raw,
    )


# --- Per-feature activation --------------------------------------------------
#
# The map half of the same pipeline. `score_factors` already takes its flow
# speed and salinity as parameters instead of reaching for one bay-wide
# number, so scoring a single feature is the SAME nine-factor pipeline fed
# that feature's own `FeatureMetrics.speed` and its own location's
# `SalinityReading` -- not a separate code path. `structure` is the one
# factor `score_factors` cannot produce by itself, because it depends on the
# Phase 1 derived-structure fields (`ambush`, `okubo_w`, `eddy_share`,
# `convergence`) that exist only per-feature, never per-hour.
#
# `structure_weight` and the four `structure_*` curves live on
# `SpeciesProfile` as a SIBLING of `weights`/`curves`, not a member of
# either -- see that class's docstring for why "structure" cannot join the
# nine-factor `FACTORS` set (2026-08-26 review, Important 2).


def _structure_subscore(
    metrics: FeatureMetrics, profile: SpeciesProfile, thresholds: StructureThresholds
) -> SubScore:
    """How much this feature's geometry looks like a fishing spot.

    Four independent structural signatures, each its OWN YAML `Curve` under
    `profile.curves["structure_*"]` -- this function no longer picks the
    response shape in Python. A clamped linear ramp with a saturation
    breakpoint, which the previous version of this function was, written in
    Python, IS a response curve, and the module docstring's "response shape
    comes from a YAML curve, never from code" contract applies to it exactly
    as it applies to the other nine factors (2026-08-26 review, Important 2's
    second sentence).

    - `structure_ambush` reads `metrics.ambush`, a raw speed CONTRAST in m/s
      -- see `structure.ambush_contrast`.
    - `structure_seam` reads `metrics.okubo_w` DIRECTLY, with no sign branch.
      `okubo_w` is MAX-reduced per feature (see `FeatureMetrics`'s own
      docstring), which makes it a SEAM detector, W > 0, and STRUCTURALLY
      INCAPABLE of reporting an eddy: of 13,614 real per-feature samples
      measured over the whole winyah-bay `mean_med` library, the most
      negative is -8.8e-7, an order of magnitude inside the quiet band --
      floating-point residue, not a rotation. The curve's own x=0.0
      breakpoint clamps that residue to y=0.0 (values below the authored x
      range clamp to the first y -- see `curves.evaluate`), so no separate
      negative branch is needed in code.
    - `structure_eddy` reads `metrics.eddy_share` -- the channel Phase 1
      built for exactly this case (`FeatureMetrics`'s docstring calls it
      "the eddy channel that leaves `okubo_w` alone"). The previous version
      of this function derived "eddy" from negative `okubo_w` instead, which
      the paragraph above proves never fires in practice: spec section 7's
      headline object IS an eddy, and that version could not recognise one
      (2026-08-26 review, Finding 3).
    - `structure_convergence` reads `metrics.convergence` directly.

    All four YAML breakpoints are anchored to the REAL winyah-bay `mean_med`
    library, not guessed -- measured 2026-08-26 over all 26 phases, every
    feature, `n_cells > 0`, discarding only non-finite samples:

        field          n       p50      p75      p90      p95      p99      max
        ambush       13,742   0.035    0.101    0.233    0.326    0.535    0.832
        okubo_w      13,614   2.6e-7   2.1e-6   1.1e-5   2.6e-5   7.9e-5   3.09e-4
        convergence  13,614   3.3e-4   9.5e-4   2.4e-3   3.7e-3   7.0e-3   1.47e-2
        eddy_share   13,614   0.0      0.0      0.0      0.018    0.076    0.25

    (okubo_w's min was -8.8e-7 -- see above; only 10.6% of samples exceed the
    `quiet_w` seam floor at all.) The PREVIOUS version's free "5x the
    quiet-band threshold" saturation point put convergence's ceiling at
    5.0e-4 -- just above its OWN MEDIAN, so 40.3% of real features saturated
    at exactly 1.0 there -- while clamping ambush at 1.0 m/s flat, which the
    real data never once reaches (measured max 0.832), so ambush could never
    saturate at all. The two were mis-calibrated relative to each other by
    roughly an order of magnitude (2026-08-26 review, Finding 3); the YAML
    breakpoints above instead put each field's own measured max near y=1.0.

    Combined with MAX, not a mean: any ONE strong signature -- an ambush
    pocket, OR a seam, OR an eddy, OR a convergence front -- is what makes a
    spot worth fishing, and averaging four quiet numbers against one loud one
    would dilute the loud one exactly the way an arithmetic mean would at the
    `combine()` level (see that function's docstring for the same argument
    one level up). A feature does not need all four to be worth marking.

    Missing is possible and distinct from a quiet reading: `sample_features`
    can leave any of the four NaN even when `n_cells > 0`, if no sampled cell
    has a finite value for that particular field (an entirely dry disc, for
    instance). Only when ALL FOUR are NaN is there truly nothing to score; if
    even one is finite it is used, and the reason says which.

    `thresholds` no longer SCALES anything here -- the YAML curves do that --
    but it still names the quiet-band floor a seam or convergence reading is
    being compared against in the reason text, so a reader sees not just the
    score but what "structural" meant.
    """
    signals: dict[str, tuple[float, str]] = {}
    if math.isfinite(metrics.ambush):
        v = evaluate(profile.curves["structure_ambush"], metrics.ambush)
        signals["ambush pocket"] = (v, f"ambush contrast {metrics.ambush:.2f} m/s")
    if math.isfinite(metrics.okubo_w):
        v = evaluate(profile.curves["structure_seam"], metrics.okubo_w)
        signals["current seam"] = (
            v, f"okubo_w {metrics.okubo_w:.1e} s^-2 vs a {thresholds.quiet_w:.0e} quiet floor"
        )
    if math.isfinite(metrics.eddy_share):
        v = evaluate(profile.curves["structure_eddy"], metrics.eddy_share)
        signals["eddy"] = (v, f"eddy_share {metrics.eddy_share:.2f} of the wet disc")
    if math.isfinite(metrics.convergence):
        v = evaluate(profile.curves["structure_convergence"], metrics.convergence)
        signals["convergence front"] = (
            v,
            f"convergence {metrics.convergence:.1e} s^-1 vs a "
            f"{thresholds.convergence_min:.0e} quiet floor",
        )

    if not signals:
        return SubScore(
            "structure", float("nan"), profile.structure_weight,
            "structure: no data (ambush, okubo_w, eddy_share and convergence all NaN)", True,
        )

    name, (value, detail) = max(signals.items(), key=lambda kv: kv[1][0])
    return SubScore(
        "structure", value, profile.structure_weight,
        f"structure {value:.2f} — strongest signal: {name} ({detail})", False,
    )


def _flat_wet_multiplier(metrics: FeatureMetrics, hour) -> tuple[float, str]:
    """Is THIS flat holding water at THIS hour's tide -- not merely for its
    average share of the whole cycle.

    2026-08-26 review, Finding 4: `wet_fraction` is a CYCLE MEAN
    (`schedule.schedule_from_depths`'s `wet.mean(axis=0)`) -- identical at
    every hour of the day -- so gating on it alone applies a fixed haircut
    ALL DAY, including at the top of the tide when a flat is actually
    flooded and prime. The median shipped flat (0.735) lost 26% of its
    activation at EVERY hour under that gate, at low water and high water
    alike -- exactly backwards from spec section 8 factor 2's "which flats
    are actually flooded".

    `flood_phase` (also on `FeatureMetrics`, also a Phase 1 field, previously
    unused here) says WHERE in the 0 (low water) .. 1 (next low water) cycle
    this flat's wet window STARTS. `FeatureMetrics` carries no `drain_phase`,
    so the window's END is approximated as `flood_phase + wet_fraction` --
    a KNOWN APPROXIMATION with a MEASURED cost, not a derivation from
    `schedule.py` (2026-08-26 re-review: an earlier version of this comment
    claimed `schedule.py` "notes `wet_fraction` already IS that length"; it
    does not -- `schedule.py:13` defines the wet-window length as
    `(drain_phase - flood_phase) % 1.0`, a DIFFERENT quantity from
    `wet_fraction`, which line 61 documents separately as "share of the
    cycle holding water"). Measured directly, 2026-08-26, over every cell in
    the winyah-bay `mean_med` library with a partial `wet_fraction` and both
    phases finite (51,558 of 587,325 domain cells): median error
    (`wet_fraction` minus the true first-window length) -0.011, 2.38% of
    cells off by more than 0.05 of a cycle, and 3.92% overshooting by more
    than 0.02 into the UNSAFE direction -- reading a cell as wet when its
    true drain has already passed. The mechanism is a residual puddle:
    `wet_fraction` counts every wet snapshot across the whole cycle, while
    `drain_phase` closes only the FIRST wet/dry transition after
    `flood_phase` (`schedule_from_depths`'s own comment: "the first dry
    transition AFTER the flood"), so a cell that dries, re-floods and dries
    again reads as continuously wet straight across the dry gap in between.
    At the feature level, 12 of the 196 in-domain flats have a mean
    cell-level error beyond 0.05 -- materially wrong, though still a small
    minority of the 196. `CellSchedule` already carries `drain_phase` per
    cell; `sample_features` simply does not sample it onto `FeatureMetrics`
    yet. Adding it would remove this approximation entirely, but that is
    Phase 1 code and an improvement to a working gate, not a defect in it --
    out of scope here, recorded as a follow-up rather than fixed.

    Three cases:
      wet_fraction >= 1.0  -- always wet, full credit regardless of hour.
      wet_fraction <= 0.0  -- never wet, zero regardless of hour.
      0 < wet_fraction < 1 -- gate on whether THIS hour's tide fraction
        falls inside [flood_phase, flood_phase + wet_fraction) mod 1.0.

    An UNKNOWN `wet_fraction` (NaN -- no schedule data reached this feature)
    is treated as NOT confirmed wet, never as fully wet: this gate has no
    `provisional` channel to disclose an uncertain number through the way
    `SubScore` does, so rather than defaulting to the optimistic answer it
    gates hard, to the conservative one, and says so in the returned note
    (2026-08-26 review, Finding 9: "a NaN wet_fraction skips the gate and
    scores FULL -- the optimistic default the plan forbids").

    A missing or incomplete TIDE reading for this hour (`hour.tide_frac` or
    `hour.tide_phase` is None) similarly cannot be resolved to a phase, but
    unlike an unknown `wet_fraction` there IS an honest fallback available --
    the cycle average itself -- so that is what is returned, clearly labelled
    as a fallback rather than a real per-hour reading.
    """
    wf = metrics.wet_fraction
    if not math.isfinite(wf):
        return 0.0, "wet_fraction unknown -- treated as dry, not assumed flooded"
    if wf >= 1.0:
        return 1.0, "wet all cycle"
    if wf <= 0.0:
        return 0.0, "dry all cycle"

    if hour.tide_frac is None or hour.tide_phase is None:
        return wf, f"no tide reading this hour -- using the {wf:.2f} cycle average"
    if not math.isfinite(metrics.flood_phase):
        # wet_fraction is strictly between 0 and 1 but flood_phase is NaN --
        # should not happen per schedule.py's "static cells only" NaN rule,
        # but the cycle average is the honest fallback if it ever does.
        return wf, f"no flood phase on record -- using the {wf:.2f} cycle average"

    full = _recombine_tide_frac(hour)
    since_flood = (full - metrics.flood_phase) % 1.0
    wet_now = since_flood < wf
    return (
        1.0 if wet_now else 0.0,
        f"{'flooded' if wet_now else 'dry'} at this hour's tide "
        f"(flood phase {metrics.flood_phase:.2f}, wet window {wf:.2f} of the cycle)",
    )


@dataclass
class FeatureActivation:
    """One feature's bite-worthiness at one hour -- the map's per-marker
    number, next to `HourScore`'s single fishery-wide one.

    Carries the same honesty fields `HourScore` does (2026-08-26 review,
    Finding 5): `confidence`, `constrained_share`, `excluded` and
    `provisional` used to survive only as prose inside `reason`, readable by
    a person but not consumable by code. Spec section 9 (click a marker, see
    what and why) and section 10 (a confidence indicator) both need these
    STRUCTURED, and Task 7's payload consumes this next.
    """

    key: str
    type: str
    activation: int
    subs: list[SubScore]
    confidence: float
    constrained_share: float
    excluded: list[str]
    provisional: list[str]
    reason: str


def score_feature(
    metrics: FeatureMetrics,
    hour,
    day,
    profile: SpeciesProfile,
    salinity: SalinityReading | None,
    thresholds: StructureThresholds,
) -> FeatureActivation:
    """Score one feature at one hour: the map half of the pipeline.

    Reuses `score_factors` for the nine shared factors, feeding it THIS
    feature's own flow (`metrics.speed`) and THIS feature's own salinity
    reading -- not the bay-representative values `score_factors` falls back
    to when it is scoring the fishery-wide `HourScore` instead. Spec section
    7's actual, owner-ratified requirement (2026-08-26 review) is that
    salinity moves a feature's activation DIRECTIONALLY, species by species
    -- NOT that it multiplies with veto power over the whole map, which was
    considered and rejected because it would hand one still-uncalibrated
    factor (Winyah's `fitted=False`) control of every other factor's work.
    See test_the_same_feature_scores_lower_in_fresh_water_for_trout for the
    measured, fully-populated-hour numbers this produces.

    `thresholds` is REQUIRED, not defaulted to `StructureThresholds()`:
    `cli.py`'s `flow_structure` command is careful to load and pass the
    fishery's own (`fishery.structure`), and a silent default here would
    silently diverge from it for any fishery whose thresholds are tuned away
    from the class defaults (2026-08-26 review, Finding 9).

    A tenth, feature-only `structure` sub-score is appended from the Phase 1
    derived-structure fields (see `_structure_subscore`), then `combine` runs
    UNCHANGED over all ten -- the geometric mean, the missing-factor
    renormalisation and the provisional/confidence split all apply exactly
    as they do for the hourly score, because they are the same honesty
    requirements about the same kind of number, just addressed at a point
    instead of a whole fishery.

    `n_cells == 0` means the feature's sampling disc found no library cells
    at all -- outside the model domain -- and every metric on it is NaN by
    construction (`activation.sample_features`). Scoring that would be
    scoring noise with a straight face, so it short-circuits before
    `score_factors` ever runs, returning an explanatory reason and no subs
    to inspect rather than a number computed on NaN.

    `flat`-type features are gated on `_flat_wet_multiplier` AFTER combining
    -- you cannot fish a flat with no water on it right now, however good the
    tide, wind and salinity look elsewhere on the same disc, but a flat that
    IS flooded at this hour must not be penalised for its dry hours too (see
    that function's docstring; 2026-08-26 review, Finding 4). The multiply
    happens on `combined.raw`, the pre-rounding [0, 1] value, with exactly
    ONE rounding at the very end -- not on the already-rounded `combined.
    score` -- because rounding twice measurably diverges from rounding once
    (2026-08-26 review, Finding 9: 26 of 196 real flats differ). Other
    feature types are not gated this way -- a hole or a channel edge holds
    water at low tide by definition, so multiplying by a wetness fraction
    there would penalise them for a quantity that says nothing about their
    fishability.

    The owner's 2026-08-26 ruling (progress.md) is to include an
    uncalibrated salinity at full weight, flagged, rather than exclude or
    discount it -- and that has to hold at the feature boundary too, or the
    map would quietly present an unconstrained per-feature number as a
    confident one. Nothing here strips `SubScore.provisional`: `salinity` is
    forwarded to `score_factors` untouched, so its provisional flag and its
    "UNCALIBRATED" reason text ride along inside `subs` exactly as `combine`
    received them, `reason` restates the provisional list explicitly, and
    `FeatureActivation.provisional`/`.confidence`/`.constrained_share` carry
    the same fields `HourScore` does rather than leaving them as prose only
    (2026-08-26 review, Finding 5).
    """
    if metrics.n_cells == 0:
        return FeatureActivation(
            key=metrics.key, type=metrics.type, activation=0, subs=[],
            confidence=0.0, constrained_share=0.0, excluded=[], provisional=[],
            reason=(
                f"{metrics.key} is outside the model domain — no library cells "
                "fall within the feature's sampling disc"
            ),
        )

    subs = score_factors(hour, day, profile, salinity=salinity, flow_speed=metrics.speed)
    subs.append(_structure_subscore(metrics, profile, thresholds))
    combined = combine(subs)

    raw = combined.raw
    flat_note = ""
    if metrics.type == "flat":
        mult, flat_note = _flat_wet_multiplier(metrics, hour)
        raw = raw * mult
    activation = int(round(100 * raw))

    reason = (
        f"{metrics.type} {metrics.key}: activation {activation}/100, "
        f"confidence {combined.confidence:.2f}"
    )
    if combined.provisional:
        reason += (
            f" -- provisional (scored at full weight, unconstrained): "
            f"{', '.join(combined.provisional)}"
        )
    if flat_note:
        reason += f", {flat_note}"

    return FeatureActivation(
        key=metrics.key, type=metrics.type, activation=activation, subs=combined.subs,
        confidence=combined.confidence, constrained_share=combined.constrained_share,
        excluded=combined.excluded, provisional=combined.provisional, reason=reason,
    )
