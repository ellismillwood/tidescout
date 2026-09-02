import pytest
from pydantic import ValidationError

from tidescout.engine.curves import evaluate
from tidescout.models import Curve


def test_evaluate_interpolates_between_breakpoints():
    c = Curve(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.0])
    assert evaluate(c, 0.5) == pytest.approx(0.5)
    assert evaluate(c, 1.5) == pytest.approx(0.5)
    assert evaluate(c, 1.0) == pytest.approx(1.0)


def test_evaluate_clamps_outside_the_authored_range():
    """A curve authored to 40 knots must not go negative at 60. Extrapolating
    a hand-drawn response curve past its last breakpoint invents a shape
    nobody chose."""
    c = Curve(x=[0.0, 10.0, 40.0], y=[1.0, 0.8, 0.05])
    assert evaluate(c, -5.0) == pytest.approx(1.0)
    assert evaluate(c, 100.0) == pytest.approx(0.05)


def test_curve_rejects_unsorted_breakpoints():
    """np.interp returns silent nonsense for unsorted x rather than raising."""
    with pytest.raises(ValidationError, match="ascending"):
        Curve(x=[0.0, 2.0, 1.0], y=[0.0, 1.0, 0.5])


def test_curve_rejects_mismatched_lengths():
    with pytest.raises(ValidationError, match="same length"):
        Curve(x=[0.0, 1.0], y=[0.0])


def test_curve_rejects_outputs_outside_zero_to_one():
    """Sub-scores are 0-1 by contract; the geometric mean is undefined for
    negatives and a >1 factor would let one input inflate the whole score."""
    with pytest.raises(ValidationError, match="between 0 and 1"):
        Curve(x=[0.0, 1.0], y=[0.0, 1.4])


def test_curve_needs_at_least_two_points():
    with pytest.raises(ValidationError, match="at least two"):
        Curve(x=[1.0], y=[1.0])


def test_evaluate_returns_nan_for_a_missing_input():
    """None means "no data", which must reach the combiner as an exclusion --
    not as 0.0, which means "conditions are terrible"."""
    import math

    c = Curve(x=[0.0, 1.0], y=[0.0, 1.0])
    assert math.isnan(evaluate(c, None))
