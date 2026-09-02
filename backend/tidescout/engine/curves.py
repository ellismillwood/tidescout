"""Piecewise-linear response curves. Pure."""

import numpy as np

from tidescout.models import Curve


def evaluate(curve: Curve, value: float | None) -> float:
    """Curve value at `value`, clamped to the authored range.

    Clamping rather than extrapolating: a curve authored out to 40 knots has
    nothing to say about 60, and a linear extension would run it negative --
    inventing a response shape nobody chose.

    `None` returns NaN, not 0.0. Missing data and terrible conditions are
    different statements, and spec section 8 requires the first to exclude the
    factor and renormalise rather than score it zero.
    """
    if value is None:
        return float("nan")
    v = float(value)
    if not np.isfinite(v):
        return float("nan")
    return float(np.interp(v, curve.x, curve.y))
