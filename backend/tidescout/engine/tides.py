import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo


@dataclass
class TideHour:
    time: datetime
    height_ft: float


@dataclass
class TideEvent:
    time: datetime
    kind: str  # "H" | "L"
    height_ft: float


@dataclass
class TideStage:
    phase: str  # "rising" | "falling"
    frac: float
    next_event: TideEvent


@dataclass
class CurrentHour:
    time: datetime
    speed_kn: float  # signed: + flood, - ebb
    dir_deg: float


def stage_at(events: list[TideEvent], t: datetime) -> TideStage | None:
    ordered = sorted(events, key=lambda e: e.time)
    for prev, nxt in pairwise(ordered):
        if prev.time <= t <= nxt.time:
            span = (nxt.time - prev.time).total_seconds()
            frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
            phase = "rising" if nxt.kind == "H" else "falling"
            return TideStage(phase, frac, nxt)
    return None


def _cosine_height(h0: float, h1: float, frac: float) -> float:
    return h0 + (h1 - h0) * (1 - math.cos(math.pi * frac)) / 2


def interpolate_tide_hours(events: list[TideEvent], day: date, tz: str) -> list[TideHour]:
    """Cosine interpolation of hourly heights between consecutive hi/lo events.

    Standard tide-clock approximation for subordinate stations, which reject
    interval=h hourly predictions.
    """
    zone = ZoneInfo(tz)
    start = datetime.combine(day - timedelta(days=1), datetime.min.time(), zone)
    end = datetime.combine(day + timedelta(days=2), datetime.min.time(), zone)
    pairs = list(pairwise(sorted(events, key=lambda e: e.time)))

    hours: list[TideHour] = []
    t = start
    while t < end:
        for prev, nxt in pairs:
            if prev.time <= t <= nxt.time:
                span = (nxt.time - prev.time).total_seconds()
                frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
                hours.append(TideHour(t, _cosine_height(prev.height_ft, nxt.height_ft, frac)))
                break
        t += timedelta(hours=1)
    return hours


# The longest interval between consecutive hi/lo predictions that can still
# be treated as half a tidal cycle. Measured across all 28 cached yearly
# chunks CONCATENATED (39,519 consecutive hi/lo intervals -- measuring each
# year separately, as an earlier pass of this comment did, hides exactly the
# seams a concatenated series actually has to cross) at station 8662549 --
# SOUTH ISLAND FERRY, INTRACOASTAL WATERWAY (see `fisheries/winyah-bay.yaml`
# for the station-to-name mapping), NOT Springmaid Pier, which is a
# different station (8661070) used elsewhere in this repo: min 4.42 h,
# median 6.28 h, max 12.13 h, with 8 intervals over 8 h and 3 over 9 h.
#
# Of those 3 over 9 h, all are year-seam gaps (2004->2005, 2008->2009,
# 2012->2013) where CO-OPS's yearly chunking drops the extremum nearest
# midnight -- and all three are L -> L pairs, so `phase_at`'s same-kind
# guard already rejects them before this threshold is ever reached. They
# are a chunking artefact, not evidence that a genuine half-cycle runs this
# long.
#
# The remaining 5 (8.05-8.37 h) were a DST spring-forward artefact: `span`
# below used to be computed by subtracting two datetimes that shared one
# ZoneInfo object, which Python resolves as naive wall-clock arithmetic
# (silently dropping the offset change) rather than true elapsed time. Fixed
# by differencing in UTC (see `phase_at`); their true elapsed times are
# 7.05-7.37 h, so they no longer land in this band. (Both ranges above were
# reconstructed from the real cached payloads: the five are 8.05, 8.15,
# 8.233, 8.317 and 8.367 h under the old arithmetic, on 2003-04-05,
# 2023-03-11, 2007-03-11, 2011-03-13 and 2001-04-01 -- every one an H -> L
# pair spanning a spring-forward transition.)
#
# 9.0 sits between the observed genuine maximum and the ~12.4 h gap a
# single missing event would produce, so it clears real half-cycles with
# room to spare while still catching a dropped prediction as a GAP, not a
# long half-cycle. Interpolating across a real gap would place phase 0.25
# in the middle of a cycle that was never predicted.
MAX_HALF_CYCLE_H = 9.0


def _is_ascending(events: Sequence[TideEvent]) -> bool:
    """O(n) check `phase_at` uses to skip re-sorting an already-sorted list.

    Far cheaper than `sorted()`: no comparison-sort machinery, no merge
    steps, and it can bail out on the first out-of-order pair instead of
    always touching every element.
    """
    prev = None
    for e in events:
        if prev is not None and e.time < prev:
            return False
        prev = e.time
    return True


def phase_at(events: Sequence[TideEvent], t: datetime) -> float | None:
    """The salinity model's tidal phase at `t`, or None if undeterminable.

    Phase 0 is LOW water and 0.5 is high water -- the convention
    `engine/salinity.py:salinity_at` documents and depends on. Reversing it
    inverts the tidal salinity swing across the whole bay, and no test of
    the FIT would catch that: least squares would simply choose different
    parameters to compensate.

    Interpolated LINEARLY IN TIME between the bracketing events, the same
    approximation `interpolate_tide_hours` already rests on. Returns a value
    in [0, 1): a low is always 0.0, never 1.0, so one physical state has one
    number.

    None -- never a default -- when `t` is outside the events, falls in a
    gap, or sits between two events of the same kind (which means the event
    between them was not predicted). A fabricated phase is exactly the error
    this codebase already refuses at parse time when it rejects rows with no
    usable timestamp.

    Invariant when `t` lands exactly on an interior event's own timestamp:
    bracketing intervals are inclusive on both ends (`before.time <= t <=
    after.time`), so such a `t` matches two adjacent pairs -- the one ending
    there and the one starting there. This prefers the first VALID pair over
    the first MATCHING one, so an invalid neighbour (a prediction gap or a
    same-kind pair) on one side can't shadow a valid determination from the
    other side. `None` is returned only when every matching pair is
    unusable.

    CONTRACT: `events` should already be ascending by `.time` --
    `tide_events_range` returns them that way. Unsorted input is still
    handled correctly (`test_unsorted_events_are_handled` pins this): a
    cheap O(n) ascending check runs first (`_is_ascending`), and only an
    out-of-order list pays for a real `sorted()`. This is not pedantry --
    a calibration run calls `phase_at` thousands of times over the SAME
    events list, and re-sorting from scratch every call, plus the O(n)
    linear scan the search used to do, measured at 21.45 s for 1,860 calls
    over 39,520 events. The bracketing pair is now found by bisection
    (O(log n)) instead of a linear scan, on top of skipping the sort.
    """
    if t.tzinfo is None:
        raise ValueError("phase_at needs a tz-aware timestamp; a naive one silently shifts phase")
    ordered = events if _is_ascending(events) else sorted(events, key=lambda e: e.time)
    n = len(ordered)
    # Only a pair (i, i+1) with ordered[i].time <= t <= ordered[i+1].time can
    # bracket t. For an ascending sequence that is exactly the contiguous
    # index range [lo-1, hi-1], where `lo`/`hi` are `t`'s left/right
    # insertion points -- a single pair unless `t` exactly equals one or
    # more event times, in which case the range widens by one per tie,
    # reproducing the "matches two adjacent pairs" invariant documented
    # above. Trying them in ascending `i` order reproduces the original
    # left-to-right scan's preference for the first VALID pair.
    lo = bisect_left(ordered, t, key=lambda e: e.time)
    hi = bisect_right(ordered, t, key=lambda e: e.time)
    for i in range(lo - 1, hi):
        if i < 0 or i + 1 >= n:
            continue
        before, after = ordered[i], ordered[i + 1]
        if before.kind == after.kind:
            continue  # the event between them was not predicted -- try the other matching pair
        # Differenced in UTC, not in the (possibly shared) local ZoneInfo:
        # subtracting two aware datetimes that carry the SAME tzinfo object
        # is documented Python behaviour to resolve as naive wall-clock
        # arithmetic, silently dropping any DST offset change between them.
        # `before`/`after` are both station-local events and so are exactly
        # the pair at risk of sharing one ZoneInfo; converting both to UTC
        # first forces true elapsed time regardless.
        before_utc = before.time.astimezone(UTC)
        after_utc = after.time.astimezone(UTC)
        span = (after_utc - before_utc).total_seconds()
        if span <= 0 or span > MAX_HALF_CYCLE_H * 3600.0:
            continue  # a gap, not a half-cycle -- try the other matching pair
        frac = (t.astimezone(UTC) - before_utc).total_seconds() / span
        # low -> high covers 0.0..0.5; high -> low covers 0.5..1.0
        phase = frac * 0.5 if before.kind == "L" else 0.5 + frac * 0.5
        return phase % 1.0
    return None


def interpolate_current_hours(points: list[CurrentHour], day: date, tz: str) -> list[CurrentHour]:
    """Linear interpolation of signed current speed onto the top-of-hour grid.

    Subordinate current stations predict at irregular times (slack/max ebb/max
    flood); the hourly display grid interpolates between them.
    """
    zone = ZoneInfo(tz)
    start = datetime.combine(day - timedelta(days=1), datetime.min.time(), zone)
    end = datetime.combine(day + timedelta(days=2), datetime.min.time(), zone)
    pairs = list(pairwise(sorted(points, key=lambda p: p.time)))

    hours: list[CurrentHour] = []
    t = start
    while t < end:
        for prev, nxt in pairs:
            if prev.time <= t <= nxt.time:
                span = (nxt.time - prev.time).total_seconds()
                frac = 0.0 if span == 0 else (t - prev.time).total_seconds() / span
                speed = prev.speed_kn + (nxt.speed_kn - prev.speed_kn) * frac
                # Direction is a step function of flood/ebb regime, not a
                # continuous quantity: keep the leading point's heading until
                # the interpolated sign no longer matches it, then switch.
                same_regime = (speed >= 0) == (prev.speed_kn >= 0)
                dir_deg = prev.dir_deg if same_regime else nxt.dir_deg
                hours.append(CurrentHour(t, speed, dir_deg))
                break
        t += timedelta(hours=1)
    return hours
