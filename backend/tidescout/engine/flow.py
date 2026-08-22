"""Pure flow-state lookup. No I/O -- callers hand in loaded arrays.

Tidal flow is quasi-periodic, so the library stores a handful of regimes and
the runtime interpolates between phase snapshots. Everything here is a pure
function of its arguments so it can be property-tested cheaply.
"""

import numpy as np

RANGE_ORDER = ["neap", "mean", "spring"]
DISCHARGE_ORDER = ["low", "med", "high"]

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
# 3 is the smallest integer that makes one range step cost more than the
# widest possible discharge gap (2), which is the property that matters:
# never trade range for discharge.
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

    These are the values `forcing.river_inflow_m3s` injects, not the bucket
    EDGES: 'med' is the midpoint of low and high, which is 4,533 cfs and not
    the record's median of 3,866.
    """
    return {
        "low": buckets.low_below_cfs,
        "med": 0.5 * (buckets.low_below_cfs + buckets.high_above_cfs),
        "high": buckets.high_above_cfs,
    }


def blend_regimes(
    range_bucket: str, cfs: float, buckets, available: set[str]
) -> tuple[list[tuple[str, float]], bool]:
    """Weights over regimes bracketing `cfs` on the discharge axis.

    The library holds three discharge values spanning 2,774-6,292 cfs while the
    observed record runs 1,232-22,996, so snapping to the nearest bucket throws
    away most of the axis. Blending recovers it within the simulated span --
    justified by Plan 3's measurement that depth rises monotonically and
    near-linearly with discharge at every inflow.

    The RANGE axis is deliberately not blended. One range step rescales the
    entire tidal forcing (~15 cm of amplitude on a 1.10 m mean range) against a
    discharge step's ~1 cm of depth; RANGE_STEP_COST=3 exists to stop range
    being traded away, and a blend that crossed it would be the same mistake.

    Outside the simulated span this CLAMPS and returns True. Extrapolating a
    shallow-water solution 3.7x past any flow it was run at would be inventing
    data, and the caller needs to know the difference.
    """
    flows = bucket_flows(buckets)
    order = [b for b in DISCHARGE_ORDER if f"{range_bucket}_{b}" in available]
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
        nearest = min(DISCHARGE_ORDER, key=lambda d: abs(cfs - flows[d]))
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
