"""Phase mapping. Phase 0 is LOW water -- the convention every part of the
flow library depends on, and the one Plan 3 documented as the easiest to
invert without noticing."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tidescout.engine.phase import library_phase
from tidescout.engine.tides import TideEvent

TZ = ZoneInfo("America/New_York")
LOW = datetime(2026, 8, 16, 6, 0, tzinfo=TZ)
HIGH = LOW + timedelta(hours=6, minutes=13)     # half of 12.42 h
NEXT_LOW = LOW + timedelta(hours=12, minutes=25)


def _events():
    return [
        TideEvent(time=LOW, height_ft=0.2, kind="L"),
        TideEvent(time=HIGH, height_ft=4.8, kind="H"),
        TideEvent(time=NEXT_LOW, height_ft=0.3, kind="L"),
    ]


def test_phase_is_zero_at_low_water():
    assert library_phase(_events(), LOW) == pytest.approx(0.0, abs=0.01)


def test_phase_is_half_at_high_water():
    assert library_phase(_events(), HIGH) == pytest.approx(0.5, abs=0.02)


def test_phase_is_a_quarter_at_mid_flood():
    """Flood is the FIRST half of the cycle. If this returns 0.75 the ebb/flood
    convention has been inverted and every flow lookup is reading the wrong
    half of the tide."""
    mid_flood = LOW + timedelta(hours=3, minutes=6)
    assert library_phase(_events(), mid_flood) == pytest.approx(0.25, abs=0.02)


def test_phase_is_three_quarters_at_mid_ebb():
    mid_ebb = HIGH + timedelta(hours=3, minutes=6)
    assert library_phase(_events(), mid_ebb) == pytest.approx(0.75, abs=0.02)


def test_phase_wraps_into_the_next_cycle():
    assert library_phase(_events(), NEXT_LOW) == pytest.approx(0.0, abs=0.02)


def test_phase_is_none_before_the_first_low_water():
    """Rather than guessing backwards past the record and silently returning a
    plausible number for an hour we have no tide data for."""
    assert library_phase(_events(), LOW - timedelta(hours=3)) is None


def test_phase_is_none_when_there_are_no_lows():
    events = [TideEvent(time=HIGH, height_ft=4.8, kind="H")]
    assert library_phase(events, HIGH) is None


def test_library_phase_deliberately_disagrees_with_the_salinity_models_phase_at():
    """Both return "tidal phase in [0, 1)" with phase 0 at low water, and they
    are NOT interchangeable -- see this task's note above. `phase_at` brackets
    on EVENTS so high water is pinned at exactly 0.5; this one runs LOW TO LOW
    so high water floats with the real diurnal inequality.

    Pinned so anyone "unifying" the two has to come here and read why they
    differ. Measured 2026-08-26 on a 6.5 h flood / 5.9 h ebb -- deliberately
    NOT `_events()`, which is symmetric and would make the two agree, hiding
    the whole point.
    """
    from tidescout.engine.tides import phase_at

    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=TZ)
    events = [
        TideEvent(time=t0, height_ft=0.0, kind="L"),
        TideEvent(time=t0 + timedelta(hours=6.5), height_ft=5.0, kind="H"),
        TideEvent(time=t0 + timedelta(hours=12.4), height_ft=0.2, kind="L"),
    ]

    for hours, want_event, want_library in (
        (0.00, 0.0000, 0.0000),
        (3.25, 0.2500, 0.2621),
        (6.50, 0.5000, 0.5242),
        (9.45, 0.7500, 0.7621),
    ):
        t = t0 + timedelta(hours=hours)
        assert phase_at(events, t) == pytest.approx(want_event, abs=1e-4)
        assert library_phase(events, t) == pytest.approx(want_library, abs=1e-4)

    # The load-bearing assertion: they must actually DIFFER away from low
    # water. Without it this test would still pass if someone made
    # library_phase delegate straight to phase_at.
    mid = t0 + timedelta(hours=6.5)
    assert abs(library_phase(events, mid) - phase_at(events, mid)) > 0.02


def test_library_phase_differences_in_utc_across_a_dst_change():
    """Subtracting two aware datetimes that share a tzinfo OBJECT is
    documented Python behaviour to resolve as naive wall-clock arithmetic,
    silently dropping any DST offset change between them. `ZoneInfo` interns
    by key, so the station-local events `noaa._parse_t` builds all share one
    instance -- exactly the pair at risk.

    `engine/tides.py:phase_at` converts to UTC first and says why at length.
    This function is its twin and did not, so on a spring-forward day it
    stretched a 12.42 h cycle into a 13.42 h wall-clock one: measured
    0.4845 where true elapsed time gives 0.4430. That 0.042-cycle error is
    larger than the 0.024 this module's own docstring calls "enough to
    select the wrong library snapshot", and wider than one snapshot spacing
    (1/26 = 0.0385) on the shipped library.
    """
    lows = ZoneInfo("America/New_York")
    a = datetime(2026, 3, 8, 0, 30, tzinfo=lows)    # EST
    b = datetime(2026, 3, 8, 13, 55, tzinfo=lows)   # EDT -- 12.42 h later in REAL time
    events = [
        TideEvent(time=a, height_ft=0.2, kind="L"),
        TideEvent(time=a + timedelta(hours=6, minutes=13), height_ft=4.8, kind="H"),
        TideEvent(time=b, height_ft=0.3, kind="L"),
    ]
    t = datetime(2026, 3, 8, 7, 0, tzinfo=lows)
    assert library_phase(events, t) == pytest.approx(0.4430, abs=1e-3)


def test_library_phase_refuses_a_gap_wider_than_a_whole_cycle():
    """The docstring promises "None -- never a guess". Between two `L`
    events it kept that promise only for spans it could bracket, not for
    spans that are physically impossible as one cycle.

    `tides.py` documents that CO-OPS year-seam chunking drops the extremum
    nearest midnight, and that every resulting >9 h gap is an L -> L pair --
    `phase_at` rejects those via `MAX_HALF_CYCLE_H`. Stretched across a
    dropped low, a ~24.8 h L -> L span here returned a confident, plausible,
    wrong fraction for every hour in it, and nothing was added to `missing`.
    """
    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=TZ)
    two_cycles = [
        TideEvent(time=t0, height_ft=0.2, kind="L"),
        TideEvent(time=t0 + timedelta(hours=24, minutes=50), height_ft=0.3, kind="L"),
    ]
    assert library_phase(two_cycles, t0 + timedelta(hours=12)) is None
    # A genuine single cycle in the same shape still resolves.
    one_cycle = [
        TideEvent(time=t0, height_ft=0.2, kind="L"),
        TideEvent(time=t0 + timedelta(hours=12, minutes=25), height_ft=0.3, kind="L"),
    ]
    assert library_phase(one_cycle, t0 + timedelta(hours=6)) == pytest.approx(0.4832, abs=1e-3)
