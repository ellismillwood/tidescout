from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest

from tidescout.engine.tides import (
    CurrentHour,
    TideEvent,
    _cosine_height,
    interpolate_current_hours,
    interpolate_tide_hours,
    phase_at,
    stage_at,
)

ET = ZoneInfo("America/New_York")


def test_stage_at_interpolates():
    events = [
        TideEvent(datetime(2026, 8, 15, 3, 0, tzinfo=ET), "H", 5.0),
        TideEvent(datetime(2026, 8, 15, 9, 0, tzinfo=ET), "L", 0.5),
    ]
    stage = stage_at(events, datetime(2026, 8, 15, 6, 0, tzinfo=ET))
    assert stage is not None
    assert stage.phase == "falling"
    assert abs(stage.frac - 0.5) < 0.01
    assert stage.next_event.kind == "L"
    assert stage_at(events, datetime(2026, 8, 15, 1, 0, tzinfo=ET)) is None


def test_cosine_height_at_midpoint_frac():
    # Pure formula check, decoupled from wall-clock alignment: H 5.0ft ->
    # L 0.5ft, frac=0.5 is the temporal midpoint between the two events
    # regardless of what clock time that falls on.
    assert abs(_cosine_height(5.0, 0.5, 0.5) - 2.75) < 0.01
    assert _cosine_height(5.0, 0.5, 0.0) == 5.0
    assert abs(_cosine_height(5.0, 0.5, 1.0) - 0.5) < 1e-9


def test_interpolate_tide_hours_subordinate_station():
    # Winyah Bay (8662549) shape: only hi/lo events available, no harmonic
    # hourly predictions. H@03:00 5.0ft, L@09:12 0.5ft -- the 12-minute
    # offset is deliberate so no top-of-hour grid point lands exactly on
    # the true chronological midpoint (covered separately, above, via the
    # pure formula test).
    events = [
        TideEvent(datetime(2026, 8, 15, 3, 0, tzinfo=ET), "H", 5.0),
        TideEvent(datetime(2026, 8, 15, 9, 12, tzinfo=ET), "L", 0.5),
    ]
    hours = interpolate_tide_hours(events, date(2026, 8, 15), "America/New_York")

    by_time = {h.time: h.height_ft for h in hours}
    assert by_time[datetime(2026, 8, 15, 3, 0, tzinfo=ET)] == 5.0

    # Hours before the first event are absent (unbracketed).
    assert all(t >= events[0].time for t in by_time)
    assert datetime(2026, 8, 15, 2, 0, tzinfo=ET) not in by_time

    # Strictly decreasing across the bracketed H->L stretch.
    bracketed = sorted(
        (t, v) for t, v in by_time.items() if events[0].time <= t <= events[1].time
    )
    assert all(a[1] > b[1] for a, b in pairwise(bracketed))

    assert hours == sorted(hours, key=lambda h: h.time)


def test_interpolate_current_hours_linear_and_direction_flip():
    # ACT6531 shape: subordinate current station, predictions land at
    # irregular slack/max-flood/max-ebb times, not top-of-hour.
    points = [
        CurrentHour(datetime(2026, 8, 15, 0, 20, tzinfo=ET), 1.2, 315.0),
        CurrentHour(datetime(2026, 8, 15, 1, 40, tzinfo=ET), -0.9, 135.0),
    ]
    hours = interpolate_current_hours(points, date(2026, 8, 15), "America/New_York")
    by_time = {h.time: h for h in hours}

    at_one = by_time[datetime(2026, 8, 15, 1, 0, tzinfo=ET)]
    assert abs(at_one.speed_kn - 0.15) < 0.001
    # Interpolated sign (+) still matches the leading point's sign (+), so
    # direction stays the leading point's flood heading.
    assert at_one.dir_deg == 315.0

    # Before the first point: unbracketed, absent.
    assert datetime(2026, 8, 15, 0, 0, tzinfo=ET) not in by_time


def test_interpolate_current_hours_exact_grid_passthrough():
    points = [
        CurrentHour(datetime(2026, 8, 15, 0, 0, tzinfo=ET), 1.0, 310.0),
        CurrentHour(datetime(2026, 8, 15, 1, 0, tzinfo=ET), -0.5, 130.0),
        CurrentHour(datetime(2026, 8, 15, 2, 0, tzinfo=ET), 0.8, 310.0),
    ]
    hours = interpolate_current_hours(points, date(2026, 8, 15), "America/New_York")
    by_time = {h.time: h for h in hours}

    assert by_time[datetime(2026, 8, 15, 0, 0, tzinfo=ET)] == points[0]
    assert by_time[datetime(2026, 8, 15, 1, 0, tzinfo=ET)] == points[1]
    assert by_time[datetime(2026, 8, 15, 2, 0, tzinfo=ET)] == points[2]


# Mapping a timestamp to the salinity model's tidal phase.
#
# The convention is load-bearing and stated in `engine/salinity.py`: phase 0
# is LOW water, 0.5 is high water. Reversing it inverts the tidal salinity
# swing across the entire bay, which no test of the fit itself would catch --
# it would simply fit different parameters to compensate.


def _phase_events():
    """Low at 00:00, high at 06:00, low at 12:00, high at 18:00."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    return [
        TideEvent(base, "L", -0.5),
        TideEvent(base + timedelta(hours=6), "H", 4.0),
        TideEvent(base + timedelta(hours=12), "L", -0.4),
        TideEvent(base + timedelta(hours=18), "H", 4.2),
    ]


def test_low_water_is_phase_zero():
    assert phase_at(_phase_events(), datetime(2026, 5, 1, 0, 0, tzinfo=UTC)) == pytest.approx(0.0)


def test_high_water_is_phase_one_half():
    assert phase_at(_phase_events(), datetime(2026, 5, 1, 6, 0, tzinfo=UTC)) == pytest.approx(0.5)


def test_midway_rising_is_a_quarter():
    assert phase_at(_phase_events(), datetime(2026, 5, 1, 3, 0, tzinfo=UTC)) == pytest.approx(0.25)


def test_midway_falling_is_three_quarters():
    """The falling limb runs 0.5 -> 1.0, so halfway down is 0.75. Getting
    this branch backwards would put ebb water where flood water belongs."""
    assert phase_at(_phase_events(), datetime(2026, 5, 1, 9, 0, tzinfo=UTC)) == pytest.approx(0.75)


def test_the_next_low_wraps_to_zero_not_one():
    """Phase is in [0, 1). A low returning 1.0 would be a different number
    for the same physical state, and cos(2*pi*1.0) == cos(0) makes that
    invisible in the model but visible in any grouping or reporting."""
    p = phase_at(_phase_events(), datetime(2026, 5, 1, 12, 0, tzinfo=UTC))
    assert p == pytest.approx(0.0)


def test_a_time_before_the_first_event_is_undeterminable():
    assert phase_at(_phase_events(), datetime(2026, 4, 30, 23, 0, tzinfo=UTC)) is None


def test_a_time_after_the_last_event_is_undeterminable():
    assert phase_at(_phase_events(), datetime(2026, 5, 1, 19, 0, tzinfo=UTC)) is None


def test_a_gap_in_the_events_is_undeterminable():
    """A missing prediction must not be interpolated across -- a 12-hour
    'interval' spanning a real gap would put phase 0.25 in the middle of
    what was actually a whole missing cycle."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    sparse = [TideEvent(base, "L", -0.5), TideEvent(base + timedelta(hours=30), "H", 4.0)]
    assert phase_at(sparse, base + timedelta(hours=15)) is None


def test_unsorted_events_are_handled():
    """CO-OPS returns sorted data, but a caller concatenating yearly
    fetches can produce unsorted input at the seams."""
    ev = list(reversed(_phase_events()))
    assert phase_at(ev, datetime(2026, 5, 1, 6, 0, tzinfo=UTC)) == pytest.approx(0.5)


def test_two_consecutive_events_of_the_same_kind_are_undeterminable():
    """A missing intervening event: two lows in a row means the high
    between them was not predicted, and the interval is not half a cycle."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    bad = [TideEvent(base, "L", -0.5), TideEvent(base + timedelta(hours=12), "L", -0.4)]
    assert phase_at(bad, base + timedelta(hours=6)) is None


def test_exact_hit_on_interior_event_prefers_valid_side_over_a_gap():
    """A `t` equal to an interior event's own timestamp matches two adjacent
    pairs (inclusive bounds on both ends): the one ending there and the one
    starting there. Here the earlier pair (L@00:00 -> H@20:00) is a 20h span,
    well past MAX_HALF_CYCLE_H -- a real gap, not a long half-cycle -- but
    the later pair (H@20:00 -> L@26:00) is a valid 6h half-cycle and t IS
    that high water. Returning None because the invalid pair happened to be
    checked first would silently drop a determinable grab sample."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    events = [
        TideEvent(base, "L", -0.5),
        TideEvent(base + timedelta(hours=20), "H", 4.0),
        TideEvent(base + timedelta(hours=26), "L", -0.4),
    ]
    assert phase_at(events, base + timedelta(hours=20)) == pytest.approx(0.5)


def test_exact_hit_on_interior_event_prefers_valid_side_over_same_kind_pair():
    """Same ambiguity as above, but the invalid neighbour is a same-kind
    pair (missing intervening high) instead of a gap: L@00:00 -> L@06:00 is
    unusable, but L@06:00 -> H@12:00 is a valid half-cycle and t IS that low
    water, so the answer should be 0.0, not None."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    events = [
        TideEvent(base, "L", -0.5),
        TideEvent(base + timedelta(hours=6), "L", -0.3),
        TideEvent(base + timedelta(hours=12), "H", 4.0),
    ]
    assert phase_at(events, base + timedelta(hours=6)) == pytest.approx(0.0)
