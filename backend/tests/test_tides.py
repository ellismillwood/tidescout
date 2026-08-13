from datetime import date, datetime
from itertools import pairwise
from zoneinfo import ZoneInfo

from tidescout.engine.tides import (
    CurrentHour,
    TideEvent,
    _cosine_height,
    interpolate_current_hours,
    interpolate_tide_hours,
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
