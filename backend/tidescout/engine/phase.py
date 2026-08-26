"""Map a wall-clock instant onto the flow library's tide phase.

The library's phase 0 is LOW water. That is not a convention anyone would guess
-- it falls out of spin_up_h / cycle_h = 6.0 / 12.42 = 0.4831 of a cycle -- and
assuming phase 0 is high water inverts flood and ebb everywhere. Measured
against the shipped config: phase 0.000 -> -0.547 m, 0.250 -> -0.058,
0.500 -> +0.547, 0.750 -> +0.058. Flood is the FIRST half.

`engine/tides.py:phase_at` looks like a twin of this function -- identical
signature, identical return type, identical "phase 0 is LOW water" convention,
identical None-rather-than-guess policy -- but it is NOT interchangeable with
`library_phase` and neither may be deleted:

* `phase_at` serves `engine/salinity.py:salinity_at`, which evaluates
  `cos(2*pi*phase)`. It interpolates between BRACKETING EVENTS, pinning high
  water at exactly phase 0.5 so the cosine's extreme lands on the real tidal
  extreme.
* `library_phase` (this function) serves the flow library. It interpolates
  LOW TO LOW, letting high water float with the real diurnal inequality,
  because the library's 26 snapshots are spaced uniformly in TIME through a
  simulated cycle -- selecting one is a question of elapsed time, not of
  tidal state.

Measured on a 6.5 h flood / 5.9 h ebb, the two disagree by up to 0.024 of a
cycle (about 18 minutes) away from low water -- enough to select the wrong
library snapshot. This is deliberate, not a bug: see
`test_library_phase_deliberately_disagrees_with_the_salinity_models_phase_at`
in `tests/test_phase.py`. Do not make this function delegate to `phase_at`,
and do not adjust either to make them agree.
"""

from datetime import datetime

from tidescout.engine.tides import TideEvent


def library_phase(events: list[TideEvent], t: datetime) -> float | None:
    """Fraction of the tidal cycle elapsed since the preceding low water.

    Interpolates between the bracketing low waters rather than assuming a fixed
    12.42 h period: real successive cycles differ by tens of minutes, and over a
    24-hour day a fixed period drifts far enough to select the wrong snapshot.

    Returns None -- never a guess -- when `t` falls outside the tide record.
    A plausible wrong phase would be read as fact by everything downstream.
    """
    lows = sorted(e.time for e in events if e.kind == "L")
    if len(lows) < 2:
        return None
    # Redundant with the loop's trailing `return None` TODAY -- `lows` is
    # freshly sorted above, so a `t` outside it can never satisfy `a <= t <= b`
    # for any pair and the loop always falls through. Proved on review over 16
    # boundary cases (duplicate lows, zero-length spans, unsorted input, naive
    # datetimes) and by that argument. Kept as an explicit statement of the
    # domain: it is what still refuses an out-of-record `t` if the loop below
    # is ever changed to wrap or extrapolate. No test can distinguish it.
    if t < lows[0] or t > lows[-1]:
        return None
    for a, b in zip(lows, lows[1:], strict=False):
        if a <= t <= b:
            span = (b - a).total_seconds()
            if span <= 0:
                return None
            return ((t - a).total_seconds() / span) % 1.0
    return None
