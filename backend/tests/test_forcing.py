from datetime import UTC, datetime, timedelta

import pytest

from tidescout.config import load_fishery
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


def test_inflow_split_follows_inflow_share_not_gauge_weight():
    """A river's share of the composite is its own long-term flow fraction.

    `weight` means "how this gauge contributes to the composite total" (1.0 =
    include it in the sum). `inflow_share` means "what fraction of that total
    enters here". Conflating them injected equal thirds into three rivers whose
    real split is 78/13/8.
    """
    f = load_fishery("winyah-bay")
    inflows = forcing.river_inflow_m3s(f, "med")
    total = sum(inflows.values())

    assert inflows["Pee Dee"] / total == pytest.approx(0.783, abs=0.01)
    assert inflows["Waccamaw"] / total == pytest.approx(0.134, abs=0.01)
    assert inflows["Black"] / total == pytest.approx(0.083, abs=0.01)
    # The Pee Dee must dominate; equal thirds is the bug this test pins.
    assert inflows["Pee Dee"] > 4 * inflows["Black"]


def test_inflow_total_still_matches_the_composite_bucket():
    """Redistributing shares must not change how much water enters overall."""
    f = load_fishery("winyah-bay")
    for bucket, cfs in (
        ("low", f.discharge_buckets.low_below_cfs),
        ("high", f.discharge_buckets.high_above_cfs),
    ):
        total = sum(forcing.river_inflow_m3s(f, bucket).values())
        assert total == pytest.approx(cfs * forcing.CFS_TO_M3S, rel=1e-9)


def test_inflow_shares_are_rejected_when_they_do_not_sum_to_one():
    """A silent renormalisation would hide an authoring mistake."""
    f = load_fishery("winyah-bay")
    f.rivers[0].inflow_share = 0.5  # now sums to ~0.72
    with pytest.raises(ValueError, match="inflow_share"):
        forcing.river_inflow_m3s(f, "med")
