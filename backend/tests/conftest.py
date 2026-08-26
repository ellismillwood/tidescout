from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache


@pytest.fixture
def fishery():
    return load_fishery("winyah-bay")


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "c.sqlite")


TZ = ZoneInfo("America/New_York")
_MID = datetime(2026, 8, 16, tzinfo=TZ)
_EVENTS = [(0, 0.2, "L"), (6.2, 4.8, "H"), (12.42, 0.3, "L"),
           (18.6, 4.9, "H"), (24.84, 0.2, "L")]


def _day_conditions(cfs: float, bucket: str):
    """A full 24-hour day with EVERY factor live.

    Verified 2026-08-26 against the real `score_factors`/`combine`: nothing is
    excluded on any hour, and the score varies 68..84 across the day. A fixture
    on which the score never moves would pass most payload assertions while
    testing nothing.
    """
    from tidescout.engine.conditions import DayConditions, HourlyConditions
    from tidescout.engine.tides import TideEvent, stage_at
    from tidescout.sources.astronomy import MoonInfo, SolunarPeriod, SunTimes
    from tidescout.sources.usgs import DischargeSummary, WaterSummary

    events = [TideEvent(time=_MID + timedelta(hours=o), height_ft=h, kind=k)
              for o, h, k in _EVENTS]
    hours = []
    for i in range(24):
        t = _MID + timedelta(hours=i)
        st = stage_at(events, t)
        hours.append(HourlyConditions(
            time=t, air_temp_f=82.0, wind_speed_kn=8.0, wind_dir_deg=180.0,
            pressure_mb=1014.0, pressure_trend_mb_3h=-0.8, cloud_cover_pct=30.0,
            tide_height_ft=2.5,
            tide_phase=(st.phase if st else None),
            tide_frac=(round(st.frac, 3) if st else None),
            current_speed_kn=1.2))
    return DayConditions(
        fishery_slug="winyah-bay", day=date(2026, 8, 16), model_label="gfs_seamless",
        hours=hours,
        sun=SunTimes(dawn=_MID + timedelta(hours=6), sunrise=_MID + timedelta(hours=6.5),
                     sunset=_MID + timedelta(hours=20), dusk=_MID + timedelta(hours=20.5)),
        moon=MoonInfo(phase_frac=0.5, rise=_MID + timedelta(hours=19),
                      set=_MID + timedelta(hours=7), transits=[_MID + timedelta(hours=13)]),
        solunar=[SolunarPeriod(kind="major", start=_MID + timedelta(hours=12.5),
                               end=_MID + timedelta(hours=14.5))],
        water=WaterSummary(temp_f=84.0, temp_trend_f_3d=0.4,
                           salinity_ppt=None, source="synthetic"),
        discharge=DischargeSummary(
            cfs_now=cfs, cfs_lagged=cfs * 0.95, bucket=bucket, sites=["02135200"],
            contributing=["02135200"], stale=[], trend=1.05, limb="steady"),
        missing=[])


def _payload_kwargs(monkeypatch, cfs: float, bucket: str):
    from tidescout.sources import dayloader

    monkeypatch.setattr(dayloader, "load_day", lambda *a, **k: _day_conditions(cfs, bucket))
    return dict(slug="winyah-bay", day=date(2026, 8, 16),
                model_label="gfs_seamless", cache=None)


@pytest.fixture
def synthetic_day(monkeypatch):
    """Median flow: 4,200 cfs, inside `calibration_range_cfs` (1232-22996)."""
    return _payload_kwargs(monkeypatch, 4_200.0, "med")


@pytest.fixture
def synthetic_day_freshet(monkeypatch):
    """22,996 cfs -- the TOP of the calibrated range and 3.7x the highest flow
    ever simulated, so this is the fixture that must surface both an
    extrapolated salinity and a clamped regime blend."""
    return _payload_kwargs(monkeypatch, 22_996.0, "freshet")
