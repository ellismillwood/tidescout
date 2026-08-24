"""Pure flow-state lookup. No I/O -- callers hand in loaded arrays.

Tidal flow is quasi-periodic, so the library stores a handful of regimes and
the runtime interpolates between phase snapshots. Everything here is a pure
function of its arguments so it can be property-tested cheaply.
"""

import numpy as np

RANGE_ORDER = ["neap", "mean", "spring"]
# `freshet` is the observed maximum of the composite record, not a fourth
# evenly-spaced step: on Winyah the four flows are 2,774 / 4,533 / 6,292 /
# 22,996 cfs, so the last gap is nine times the first two. It exists because
# the axis used to stop at the p75 while real freshets run 3.65x past it, and
# a 22,996 cfs run differs from `high` by 17.20% of p99 speed -- 22x the
# floor of a change known to be negligible. It is optional per fishery: a
# fishery with no `discharge_buckets.freshet_cfs` has no such regime, and
# `bucket_flows` omits it rather than inventing a flow for it.
DISCHARGE_ORDER = ["low", "med", "high", "freshet"]

# Cost of substituting one bucket step on each axis, when the exact regime is
# missing. These are deliberately NOT equal.
#
# One range step rescales the entire tidal forcing: RANGE_FACTORS are
# 0.72/1.0/1.28 on a 1.10 m mean range, so about 15 cm of amplitude. One
# discharge step moves domain-mean depth by roughly 1 cm -- the whole
# low->high span is +99.6 m3/s, about 4.45e6 m3 over a 12.42 h cycle against a
# domain holding ~6.7e8 m3. Range is the order of fifteen times more
# consequential, so a plain Manhattan distance (which calls them equal and
# then breaks ties alphabetically) will happily swap the range bucket to
# preserve discharge -- exactly backwards.
#
# 3 was the smallest integer that made one range step cost more than the
# widest possible discharge gap, which was 2 on a three-bucket axis: never
# trade range for discharge.
#
# THAT INVARIANT IS NOW A TIE, NOT A WIN, and the honest thing is to say so
# rather than leave the claim above standing. `freshet` makes the widest
# discharge gap 3 (low <-> freshet), which EQUALS RANGE_STEP_COST, so
# select_regime("spring", "low", {"mean_low", "spring_freshet"}) scores both
# candidates at 3 and falls through to the alphabetical tiebreak.
#
# Left at 3 deliberately, on two grounds. First, the tie is only reachable in
# a PARTIAL library -- with all twelve regimes present the exact match always
# wins and no cost is ever computed. Second, and more substantively, the
# premise that made "never trade range for discharge" obviously right no
# longer holds at the top of the axis: it was measured across 2,774-6,292
# cfs, where a discharge step moves domain-mean depth ~1 cm, but a low <->
# freshet step is an 8.3x change in discharge that moves the velocity field
# by 17.20% of p99 speed. Trading three discharge steps to keep the range
# bucket is not clearly the better answer any more, so a tie is arguably the
# correct expression of the uncertainty. Raising this to 4 would assert the
# opposite on evidence that does not exist.
#
# `test_a_three_step_discharge_gap_now_ties_one_range_step` pins the resulting
# behaviour so it cannot change silently.
RANGE_STEP_COST = 3
DISCHARGE_STEP_COST = 1


def speed_direction(u: np.ndarray, v: np.ndarray):
    return np.hypot(u, v), (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0


def select_regime(
    range_bucket: str, discharge_bucket: str, available: set[str]
) -> tuple[str, bool]:
    """Nearest available regime, and whether a substitution happened.

    Spec section 10 requires a missing state to degrade to the nearest with a
    warning rather than fail, so the caller gets the flag and surfaces it.

    Names in `available` that are not valid regimes are skipped rather than
    raising: the set is built by listing the library directory, and a stray
    file there must not take down every lookup.
    """
    if range_bucket not in RANGE_ORDER or discharge_bucket not in DISCHARGE_ORDER:
        raise ValueError(
            f"unknown regime buckets ({range_bucket!r}, {discharge_bucket!r}); "
            f"expected one of {RANGE_ORDER} and {DISCHARGE_ORDER}"
        )
    exact = f"{range_bucket}_{discharge_bucket}"
    if exact in available:
        return exact, False
    if not available:
        raise ValueError("flow library is empty")

    ri = RANGE_ORDER.index(range_bucket)
    di = DISCHARGE_ORDER.index(discharge_bucket)

    candidates = []
    for name in available:
        r, _, d = name.partition("_")
        if r not in RANGE_ORDER or d not in DISCHARGE_ORDER:
            continue
        cost = (
            RANGE_STEP_COST * abs(RANGE_ORDER.index(r) - ri)
            + DISCHARGE_STEP_COST * abs(DISCHARGE_ORDER.index(d) - di)
        )
        # Name breaks genuine ties, so the same library always resolves the
        # same way -- a forecast must not change between runs for no reason.
        candidates.append((cost, name))
    if not candidates:
        raise ValueError(
            f"flow library contains no recognisable regimes: {sorted(available)}"
        )
    return min(candidates)[1], True


def bucket_flows(buckets) -> dict[str, float]:
    """The cfs each simulated discharge bucket actually represents.

    THE single source of truth for these values: `forcing.river_inflow_m3s`
    calls this to decide what to inject at the ANUGA boundary, so the flow the
    library is indexed by and the flow it was built at cannot drift apart.

    These are the values injected, not the bucket EDGES: 'med' is the midpoint
    of low and high, which is 4,533 cfs and not the record's median of 3,866.

    'freshet' is present only when the fishery measured one. It is the
    observed maximum of the composite record rather than a midpoint or an
    edge, so it is read straight off `freshet_cfs`. Keys come out in
    DISCHARGE_ORDER, which callers rely on for a deterministic tiebreak.
    """
    flows = {
        "low": buckets.low_below_cfs,
        "med": 0.5 * (buckets.low_below_cfs + buckets.high_above_cfs),
        "high": buckets.high_above_cfs,
    }
    freshet = getattr(buckets, "freshet_cfs", None)
    if freshet is not None:
        flows["freshet"] = freshet
    return flows


def blend_regimes(
    range_bucket: str, cfs: float, buckets, available: set[str]
) -> tuple[list[tuple[str, float]], bool]:
    """Weights over regimes bracketing `cfs` on the discharge axis.

    Snapping to the nearest bucket throws away most of the axis, so this
    interpolates along it -- justified by Plan 3's measurement that depth rises
    monotonically and near-linearly with discharge at every inflow.

    With `freshet` simulated the four bucket flows span 2,774-22,996 cfs
    against an observed record of 1,232-22,996, so the ramp now covers the
    whole upper record instead of stopping at the p75. The spacing is very
    uneven (1,759 / 1,759 / 16,704 cfs), which is fine here because the
    bracket is chosen by cfs and the weight is linear in cfs -- but it does
    mean the top interval interpolates across a much larger jump in forcing
    than the two below it.

    The RANGE axis is deliberately not blended. One range step rescales the
    entire tidal forcing (~15 cm of amplitude on a 1.10 m mean range) against a
    discharge step's ~1 cm of depth; RANGE_STEP_COST=3 exists to stop range
    being traded away, and a blend that crossed it would be the same mistake.

    Outside the simulated span this CLAMPS and returns True. Extrapolating a
    shallow-water solution past any flow it was run at would be inventing data,
    and the caller needs to know the difference.
    """
    flows = bucket_flows(buckets)
    # A bucket only joins the ramp if BOTH a regime exists for it and the
    # fishery says what flow that regime represents.
    order = [b for b in DISCHARGE_ORDER if b in flows and f"{range_bucket}_{b}" in available]
    # ...and if a regime exists whose flow is NOT configured, that is a
    # library/config mismatch, not something to quietly route around. Dropping
    # it would silently clamp the axis back to `high` and throw away the very
    # regimes a rebuild was spent on -- the failure mode this whole task
    # exists to remove. Junk names in `available` are still ignored, exactly
    # as select_regime ignores them; only a RECOGNISED bucket is an error.
    unpriced = [
        b
        for b in DISCHARGE_ORDER
        if b not in flows and f"{range_bucket}_{b}" in available
    ]
    if unpriced:
        raise ValueError(
            f"the flow library has {range_bucket} regimes at discharge buckets "
            f"{unpriced} but the fishery's discharge_buckets does not say what "
            "flow they were run at -- add the matching *_cfs value (e.g. "
            "freshet_cfs) rather than leaving the regime unusable"
        )
    if not order:
        # No regime at this range at all: fall back to the existing nearest-
        # regime logic, which is allowed to cross the range axis as a last resort.
        # The discharge bucket passed in for costing must be the one nearest
        # `cfs`, not a fixed "med" -- when two range-adjacent candidates tie on
        # range distance, hardcoding "med" scores both against the wrong
        # discharge target and lets the alphabetic tiebreak override the real
        # discharge distance (e.g. cfs=100 with {"neap_high", "spring_low"}
        # available would pick neap_high, the high-discharge regime, over the
        # discharge-correct spring_low).
        # Ranges over `flows`, not DISCHARGE_ORDER: a bucket with no
        # configured flow has no cfs to measure a distance against. `flows`
        # is built in DISCHARGE_ORDER, so the tiebreak stays deterministic.
        nearest = min(flows, key=lambda d: abs(cfs - flows[d]))
        name, _ = select_regime(range_bucket, nearest, available)
        return [(name, 1.0)], True

    lo_b, hi_b = order[0], order[-1]
    if cfs <= flows[lo_b]:
        return [(f"{range_bucket}_{lo_b}", 1.0)], cfs < flows[lo_b]
    if cfs >= flows[hi_b]:
        return [(f"{range_bucket}_{hi_b}", 1.0)], cfs > flows[hi_b]

    for a, b in zip(order, order[1:], strict=False):
        fa, fb = flows[a], flows[b]
        if fa <= cfs <= fb:
            w = (cfs - fa) / (fb - fa) if fb > fa else 0.0
            mix = [(f"{range_bucket}_{a}", 1.0 - w), (f"{range_bucket}_{b}", w)]
            return [(n, x) for n, x in mix if x > 0.0] or [(f"{range_bucket}_{a}", 1.0)], False
    return [(f"{range_bucket}_{hi_b}", 1.0)], True


def bracket_phases(phases, phase: float) -> tuple[int, int, float]:
    """Indices either side of `phase` and the weight toward the second.

    The tidal cycle is periodic, so a phase past the last snapshot wraps to the
    first rather than clamping -- clamping would freeze the flow at the top of
    every cycle. `phase` may be any real number (hour 30 of a forecast is
    phase 2.4); it is reduced into [0, 1) first.
    """
    ordered = list(phases)
    if not ordered:
        raise ValueError("no phases in library")
    # This walks the list in cycle order, so unsorted input does not fail --
    # it silently returns a bracket that does not contain `phase`.
    if any(b <= a for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError(f"phases must be in strictly ascending order, got {ordered}")
    phase = phase % 1.0
    for i in range(len(ordered)):
        lo = ordered[i]
        hi = ordered[(i + 1) % len(ordered)]
        span = (hi - lo) % 1.0
        if span == 0:
            continue  # single-snapshot library: nothing to interpolate across
        offset = (phase - lo) % 1.0
        if offset <= span:
            return i, (i + 1) % len(ordered), offset / span
    return len(ordered) - 1, 0, 0.0


def interpolate_state(a: dict, b: dict, w: float) -> dict:
    """Linear blend of two snapshots. Components only -- never directions.

    Returns fresh arrays; the inputs are the caller's cached library state and
    must not be mutated.
    """
    return {k: (1.0 - w) * a[k] + w * b[k] for k in ("u", "v", "depth")}


def wet_mask(depth: np.ndarray, tol: float = 0.01) -> np.ndarray:
    return depth > tol


def tide_states(stage_bc_m, slack_frac: float = 0.25) -> list[str]:
    """Label every stored phase "flood", "ebb" or "slack" from boundary stage.

    Derived from the recorded stage series rather than from the phase number,
    because phase 0 is NOT high water: `phase` is measured from the end of
    spin-up, and spin_up_h / cycle_h = 6.0 / 12.42 = 0.4831 of a cycle, so
    phase 0 lands near LOW water (measured: -0.547 m, rising to +0.547 m at
    phase 0.5). Anything that assumes phase 0 is high water inverts ebb and
    flood -- which would make Task 13's gate confidently report the opposite of
    the truth for every spot.

    The series is periodic over one cycle, so the rate of change is a cyclic
    central difference. "slack" is the turning region, where the stage is
    changing slower than `slack_frac` of its peak rate -- an eddy or current
    break is defined by that, not by an instantaneous zero crossing that no
    stored snapshot may land on.
    """
    s = np.asarray(stage_bc_m, dtype=float)
    if s.size == 0:
        raise ValueError("no boundary stage recorded for this regime")
    if s.size < 3:
        return ["slack"] * int(s.size)
    rate = 0.5 * (np.roll(s, -1) - np.roll(s, 1))
    peak = float(np.abs(rate).max())
    if peak == 0.0:
        return ["slack"] * int(s.size)
    out = []
    for r in rate:
        if abs(r) < slack_frac * peak:
            out.append("slack")
        else:
            out.append("flood" if r > 0 else "ebb")
    return out
