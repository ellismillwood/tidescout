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
