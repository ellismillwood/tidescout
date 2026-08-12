from datetime import date, timedelta

from tidescout.config import load_fishery
from tidescout.sources.astronomy import moon_info, solunar_periods, sun_times

DAY = date(2026, 8, 15)


def _fishery():
    return load_fishery("winyah-bay")


def test_sun_times_ordered_and_plausible():
    s = sun_times(_fishery(), DAY)
    assert s.dawn < s.sunrise < s.sunset < s.dusk
    assert s.sunrise.date() == DAY
    # Mid-August sunrise on the SC coast is between 6:15 and 7:15 AM ET.
    assert 6 <= s.sunrise.hour <= 7


def test_moon_info_shape():
    m = moon_info(_fishery(), DAY)
    assert 0.0 <= m.phase_frac <= 1.0
    assert 1 <= len(m.transits) <= 3
    for t in m.transits:
        assert t.date() == DAY


def test_solunar_periods():
    periods = solunar_periods(_fishery(), DAY)
    kinds = {p.kind for p in periods}
    assert kinds <= {"major", "minor"}
    majors = [p for p in periods if p.kind == "major"]
    assert majors, "at least one lunar transit per day"
    for p in periods:
        expected = timedelta(hours=2) if p.kind == "major" else timedelta(hours=1)
        assert p.end - p.start == expected
    assert periods == sorted(periods, key=lambda p: p.start)
