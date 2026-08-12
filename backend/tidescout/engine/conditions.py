from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from tidescout.models import Fishery
    from tidescout.sources.astronomy import MoonInfo, SolunarPeriod, SunTimes
    from tidescout.sources.noaa import CurrentHour, TideEvent, TideHour
    from tidescout.sources.usgs import DischargeSummary, WaterSummary
    from tidescout.sources.weather import WeatherHour


@dataclass
class HourlyConditions:
    time: datetime
    air_temp_f: float | None = None
    wind_speed_kn: float | None = None
    wind_dir_deg: float | None = None
    wind_gust_kn: float | None = None
    pressure_mb: float | None = None
    pressure_trend_mb_3h: float | None = None
    cloud_cover_pct: float | None = None
    precip_in: float | None = None
    tide_height_ft: float | None = None
    tide_phase: str | None = None
    tide_frac: float | None = None
    current_speed_kn: float | None = None
    current_dir_deg: float | None = None
    solunar: list[str] = field(default_factory=list)


@dataclass
class DayConditions:
    fishery_slug: str
    day: date
    model_label: str
    # Always exactly 24 wall-clock-labeled rows (00:00 through 23:00 local).
    # On DST-transition days the loop below walks wall-clock hours, not
    # elapsed real time, so the nonexistent local hour still appears as a
    # label on the spring-forward day, and the repeated local hour appears
    # once, not twice, on the fall-back day -- see
    # test_assemble_day_dst_spring_forward/fall_back in test_conditions.py.
    hours: list[HourlyConditions]
    sun: SunTimes | None
    moon: MoonInfo | None
    solunar: list[SolunarPeriod]
    water: WaterSummary | None
    discharge: DischargeSummary | None
    missing: list[str]


def assemble_day(
    fishery: Fishery,
    day: date,
    model_label: str,
    weather_48h: list[WeatherHour],
    tides: list[TideHour],
    events: list[TideEvent],
    currents: list[CurrentHour],
    sun: SunTimes | None,
    moon: MoonInfo | None,
    solunar: list[SolunarPeriod],
    water: WaterSummary | None,
    discharge: DischargeSummary | None,
    missing: list[str],
) -> DayConditions:
    from tidescout.sources.noaa import stage_at  # pure function, no I/O

    tz = ZoneInfo(fishery.timezone)
    day_start = datetime.combine(day, datetime.min.time(), tz)
    day_end = day_start + timedelta(days=1)

    weather_by_time = {w.time: w for w in weather_48h}
    tide_by_time = {t.time: t for t in tides}
    current_by_time = {c.time: c for c in currents}

    hours: list[HourlyConditions] = []
    t = day_start
    while t < day_end:
        h = HourlyConditions(time=t)
        w = weather_by_time.get(t)
        if w:
            h.air_temp_f = w.air_temp_f
            h.wind_speed_kn = w.wind_speed_kn
            h.wind_dir_deg = w.wind_dir_deg
            h.wind_gust_kn = w.wind_gust_kn
            h.pressure_mb = w.pressure_mb
            h.cloud_cover_pct = w.cloud_cover_pct
            h.precip_in = w.precip_in
            w3 = weather_by_time.get(t - timedelta(hours=3))
            if w3 and w.pressure_mb is not None and w3.pressure_mb is not None:
                h.pressure_trend_mb_3h = round(w.pressure_mb - w3.pressure_mb, 2)
        tide = tide_by_time.get(t)
        if tide:
            h.tide_height_ft = tide.height_ft
        stage = stage_at(events, t) if events else None
        if stage:
            h.tide_phase = stage.phase
            h.tide_frac = round(stage.frac, 3)
        cur = current_by_time.get(t)
        if cur:
            h.current_speed_kn = cur.speed_kn
            h.current_dir_deg = cur.dir_deg
        hour_end = t + timedelta(hours=1)
        h.solunar = sorted(
            {p.kind for p in solunar if p.start < hour_end and p.end > t}
        )
        hours.append(h)
        t = hour_end

    return DayConditions(
        fishery_slug=fishery.slug, day=day, model_label=model_label, hours=hours,
        sun=sun, moon=moon, solunar=solunar, water=water, discharge=discharge,
        missing=missing,
    )
