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


def _day_conditions(cfs: float, bucket: str, water=None):
    """A full 24-hour day with EVERY factor live.

    Verified 2026-08-26 against the real `score_factors`/`combine`: nothing is
    excluded on any hour, and the score varies 68..84 across the day. A fixture
    on which the score never moves would pass most payload assertions while
    testing nothing.

    `water` defaults to the synthetic `WaterSummary` below; a caller wanting
    to exercise a REAL declared sensor's `source` (see
    `synthetic_day_out_of_domain_gauge`) passes its own instead.
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
        water=water or WaterSummary(temp_f=84.0, temp_trend_f_3d=0.4,
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
def synthetic_day_out_of_domain_gauge(monkeypatch):
    """The SAME day `synthetic_day` builds, but `water` names a REAL Winyah
    Bay sensor declared `in_domain: false` in `fisheries/winyah-bay.yaml` --
    station 021108125, 9,498 m outside the model domain, snapped to the
    along-estuary distance field's own extreme fresh end (2026-08-26 review,
    Important 1). Exists to prove `_bay_salinity_reading` never labels this
    reading MEASURED just because a real, live-reporting station supplied a
    real number -- see `test_an_out_of_domain_gauge_is_never_labelled_
    measured` in `test_payload.py`.
    """
    from tidescout.sources import dayloader
    from tidescout.sources.usgs import WaterSummary

    water = WaterSummary(temp_f=84.0, temp_trend_f_3d=0.4,
                          salinity_ppt=0.0, source="usgs:021108125")
    monkeypatch.setattr(
        dayloader, "load_day", lambda *a, **k: _day_conditions(4_200.0, "med", water=water)
    )
    return dict(slug="winyah-bay", day=date(2026, 8, 16), model_label="gfs_seamless", cache=None)


@pytest.fixture
def synthetic_day_freshet(monkeypatch):
    """22,996 cfs -- the TOP of the calibrated range, so this is the fixture
    that must surface both an extrapolated salinity and a clamped regime
    blend.

    NOT "3.7x the highest flow ever simulated" (this docstring's original
    claim, corrected 2026-08-26 review): all twelve regimes, including
    `freshet` at every tidal-range bucket, are now rasterised on disk at
    EXACTLY 22,996 cfs (`fisheries/winyah-bay.yaml`'s `freshet_cfs`) --
    `flow.blend_regimes` reads that as an exact, in-range match, not an
    extrapolation, by its own boundary-inclusive convention. `payload.py`'s
    `clamped`/`_is_extrapolated` deliberately read the exact edge of an
    unfitted, theoretical range as suspect anyway (see their docstrings) --
    the two flags below are true because of that payload-level choice, not
    because this discharge is beyond what was ever simulated.
    """
    return _payload_kwargs(monkeypatch, 22_996.0, "freshet")


def _no_weather_day_conditions(cfs: float, bucket: str):
    """The SAME day `_day_conditions` builds, with every weather-sourced
    field `None` -- `weather` going dark (a single dead upstream source) is
    the ordinary, routine failure `dayloader.load_day`'s `attempt()` wrapper
    exists for, not a hypothetical.

    `_day_conditions` deliberately keeps every factor live (see its own
    docstring) precisely so the fixture's SHAPE is well-tested elsewhere;
    that also means it is the only fixture in this file, and NONE of it
    exercises a genuine NaN sub-score value. `score_factors`'s `_missing`
    helper sets `SubScore.value = float("nan")` for `pressure` and `wind`
    when their inputs are `None` -- this is the fixture that actually
    produces one, so `_json_safe` has something real to sanitise. 2026-08-26
    review, Important 1: deleting both branches of `_json_safe` left all
    eight ORIGINAL payload tests green, because none of their fixtures ever
    gave it a NaN to catch.
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
            time=t,
            # air_temp_f/wind_*/pressure_*/cloud_cover_pct all left at their
            # None default -- weather is the dead source this fixture models.
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
        missing=["weather"])


@pytest.fixture
def synthetic_day_no_weather(monkeypatch):
    """Median flow, `weather` dead -- see `_no_weather_day_conditions`."""
    from tidescout.sources import dayloader

    monkeypatch.setattr(
        dayloader, "load_day", lambda *a, **k: _no_weather_day_conditions(4_200.0, "med")
    )
    return dict(slug="winyah-bay", day=date(2026, 8, 16), model_label="gfs_seamless", cache=None)


@pytest.fixture
def synthetic_day_with_flow(monkeypatch, cache):
    """The SAME day `synthetic_day` builds, but with `noaa.tide_events`
    monkeypatched to return the fixture's own `_EVENTS` (as real `TideEvent`s)
    and a real (empty, tmp_path-backed) `Cache` instead of `None`.

    Every other fixture in this file passes `cache=None`, which
    `payload._flow_events` treats as "skip the second tide-events fetch
    entirely" (see that function's docstring) -- correct for keeping this
    suite hermetic and fast, but it also means `library_phase` never
    resolves under any of them, so `_blended_state`, `flowlib.load_state`,
    `activation.structure_fields` and `activation.sample_features` never
    ran under ANY existing test (2026-08-26 review, Important 2). This
    fixture activates that path against winyah-bay's REAL rasterised flow
    library and REAL along-estuary distance field, without any network
    call (`tide_events` itself is stubbed, so the real `Cache` is never
    actually queried) -- slow (~70s), the one test in this file meant to
    prove the feature path runs at all.
    """
    from tidescout.engine.tides import TideEvent
    from tidescout.sources import dayloader, noaa

    events = [TideEvent(time=_MID + timedelta(hours=o), height_ft=h, kind=k)
              for o, h, k in _EVENTS]
    monkeypatch.setattr(noaa, "tide_events", lambda *a, **k: events)
    monkeypatch.setattr(dayloader, "load_day", lambda *a, **k: _day_conditions(4_200.0, "med"))
    return dict(slug="winyah-bay", day=date(2026, 8, 16), model_label="gfs_seamless", cache=cache)
