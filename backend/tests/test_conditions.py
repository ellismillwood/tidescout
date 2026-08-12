from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tidescout.config import load_fishery
from tidescout.engine.conditions import assemble_day
from tidescout.sources.noaa import TideEvent, TideHour
from tidescout.sources.weather import WeatherHour

ET = ZoneInfo("America/New_York")
DAY = date(2026, 8, 15)
START = datetime(2026, 8, 14, 0, 0, tzinfo=ET)


def _weather_48h() -> list[WeatherHour]:
    hours = []
    for i in range(48):
        t = START + timedelta(hours=i)
        hours.append(
            WeatherHour(
                time=t, air_temp_f=80.0, wind_speed_kn=9.0, wind_dir_deg=220.0,
                wind_gust_kn=14.0, pressure_mb=1013.0 + 0.5 * i, cloud_cover_pct=40.0,
                precip_in=0.0,
            )
        )
    return hours


def _tides() -> list[TideHour]:
    return [
        TideHour(START + timedelta(hours=i), 2.0 + (i % 12) / 6) for i in range(72)
    ]


def _events() -> list[TideEvent]:
    out = []
    t = datetime(2026, 8, 14, 3, 0, tzinfo=ET)
    kind = "H"
    while t < datetime(2026, 8, 16, 12, 0, tzinfo=ET):
        out.append(TideEvent(t, kind, 5.0 if kind == "H" else 0.5))
        kind = "L" if kind == "H" else "H"
        t += timedelta(hours=6, minutes=12)
    return out


def test_assemble_day_shape_and_trend():
    f = load_fishery("winyah-bay")
    result = assemble_day(
        fishery=f, day=DAY, model_label="gfs", weather_48h=_weather_48h(),
        tides=_tides(), events=_events(), currents=[], sun=None, moon=None,
        solunar=[], water=None, discharge=None, missing=["currents"],
    )
    assert result.day == DAY
    assert len(result.hours) == 24
    assert all(h.time.date() == DAY for h in result.hours)
    # pressure rises 0.5 mb/hour in the fixture -> 3h trend == 1.5
    assert abs(result.hours[0].pressure_trend_mb_3h - 1.5) < 0.01
    assert result.hours[0].tide_phase in ("rising", "falling")
    assert result.hours[0].current_speed_kn is None
    assert result.missing == ["currents"]


def test_assemble_day_solunar_tags():
    from tidescout.sources.astronomy import SolunarPeriod

    f = load_fishery("winyah-bay")
    noon = datetime(2026, 8, 15, 12, 0, tzinfo=ET)
    periods = [SolunarPeriod("major", noon - timedelta(hours=1), noon + timedelta(hours=1))]
    result = assemble_day(
        fishery=f, day=DAY, model_label="gfs", weather_48h=_weather_48h(),
        tides=_tides(), events=_events(), currents=[], sun=None, moon=None,
        solunar=periods, water=None, discharge=None, missing=[],
    )
    tagged = [h.time.hour for h in result.hours if "major" in h.solunar]
    assert 11 in tagged and 12 in tagged


def test_assemble_day_dst_spring_forward_always_24_rows():
    # 2026-03-08: US DST start (02:00 -> 03:00 local). assemble_day walks
    # wall-clock hours, not elapsed real time, so the nonexistent 02:00
    # local hour still gets a row -- the day is always exactly 24 rows.
    f = load_fishery("winyah-bay")
    day = date(2026, 3, 8)
    result = assemble_day(
        fishery=f, day=day, model_label="gfs", weather_48h=[],
        tides=[], events=[], currents=[], sun=None, moon=None,
        solunar=[], water=None, discharge=None, missing=[],
    )
    assert len(result.hours) == 24
    assert result.hours[0].time == datetime(2026, 3, 8, 0, 0, tzinfo=ET)
    assert result.hours[-1].time == datetime(2026, 3, 8, 23, 0, tzinfo=ET)


def test_assemble_day_dst_fall_back_always_24_rows():
    # 2026-11-01: US DST end (02:00 -> 01:00 local). The repeated 01:00
    # local hour appears once, not twice -- still exactly 24 rows.
    f = load_fishery("winyah-bay")
    day = date(2026, 11, 1)
    result = assemble_day(
        fishery=f, day=day, model_label="gfs", weather_48h=[],
        tides=[], events=[], currents=[], sun=None, moon=None,
        solunar=[], water=None, discharge=None, missing=[],
    )
    assert len(result.hours) == 24
    assert result.hours[0].time == datetime(2026, 11, 1, 0, 0, tzinfo=ET)
    assert result.hours[-1].time == datetime(2026, 11, 1, 23, 0, tzinfo=ET)
