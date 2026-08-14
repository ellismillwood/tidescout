from datetime import UTC, datetime, timedelta

import pytest

from tidescout.engine.tides import TideEvent
from tidescout.pipeline import forcing


def _events(start):
    return [
        TideEvent(start, "H", 5.0),
        TideEvent(start + timedelta(hours=6, minutes=13), "L", 1.0),
        TideEvent(start + timedelta(hours=12, minutes=25), "H", 5.0),
    ]


def test_tide_function_converts_feet_to_metres_and_shifts_datum():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    fn = forcing.tide_function(_events(start), datum_offset_m=-0.8, start=start)
    # t=0 is the 5.0 ft high water: 5 ft = 1.524 m, minus 0.8 m datum shift
    assert fn(0.0) == pytest.approx(1.524 - 0.8, abs=1e-3)


def test_tide_function_is_smooth_and_bounded():
    start = datetime(2026, 8, 13, 0, 0, tzinfo=UTC)
    fn = forcing.tide_function(_events(start), datum_offset_m=0.0, start=start)
    vals = [fn(t) for t in range(0, 12 * 3600, 600)]
    assert min(vals) >= 1.0 * forcing.FT_TO_M - 1e-6
    assert max(vals) <= 5.0 * forcing.FT_TO_M + 1e-6
    # No discontinuity much larger than a few cm between 10-minute samples --
    # bounds a broken interpolator (e.g. a step function at the event
    # boundary), not the smooth cosine ramp itself. With the mandated
    # _cosine_height reuse, this fixture's second interval (22320 s, 4 ft
    # range) peaks at ~0.0515 m/step at its steepest point (measured via
    # TDD), so the threshold is set just above that with headroom rather than
    # at the originally-drafted 0.05, which the reference cosine ramp does
    # not itself satisfy.
    assert max(abs(b - a) for a, b in zip(vals, vals[1:], strict=False)) < 0.06


def test_range_buckets_scale_amplitude_about_mean_level():
    neap = forcing.range_scaled_tide(mean_range_m=1.5, bucket="neap")
    spring = forcing.range_scaled_tide(mean_range_m=1.5, bucket="spring")
    assert max(spring(t) for t in range(0, 44712, 600)) > max(
        neap(t) for t in range(0, 44712, 600)
    )


def test_river_inflow_scales_with_discharge_bucket(fishery):
    lo = forcing.river_inflow_m3s(fishery, "low")
    hi = forcing.river_inflow_m3s(fishery, "high")
    assert set(lo) == {r.name for r in fishery.rivers}
    assert sum(hi.values()) > sum(lo.values())
    assert all(v >= 0 for v in lo.values())
