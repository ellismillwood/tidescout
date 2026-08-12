from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import ephem
from astral import LocationInfo
from astral.sun import sun as astral_sun

from tidescout.models import Fishery


@dataclass
class SunTimes:
    dawn: datetime
    sunrise: datetime
    sunset: datetime
    dusk: datetime


@dataclass
class MoonInfo:
    phase_frac: float
    rise: datetime | None
    set: datetime | None
    transits: list[datetime]


@dataclass(frozen=True)
class SolunarPeriod:
    kind: str  # "major" | "minor"
    start: datetime
    end: datetime


def _tz(fishery: Fishery) -> ZoneInfo:
    return ZoneInfo(fishery.timezone)


def sun_times(fishery: Fishery, day: date) -> SunTimes:
    lon, lat = fishery.center
    loc = LocationInfo("fishery", "SC", fishery.timezone, lat, lon)
    s = astral_sun(loc.observer, date=day, tzinfo=_tz(fishery))
    return SunTimes(dawn=s["dawn"], sunrise=s["sunrise"], sunset=s["sunset"], dusk=s["dusk"])


def _observer(fishery: Fishery, start_utc: datetime) -> ephem.Observer:
    obs = ephem.Observer()
    lon, lat = fishery.center
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.date = ephem.Date(start_utc.replace(tzinfo=None))
    return obs


def _to_local(edate: ephem.Date, tz: ZoneInfo) -> datetime:
    return edate.datetime().replace(tzinfo=UTC).astimezone(tz)


def moon_info(fishery: Fishery, day: date) -> MoonInfo:
    tz = _tz(fishery)
    day_start = datetime.combine(day, datetime.min.time(), tz)
    day_end = day_start + timedelta(days=1)
    start_utc = day_start.astimezone(UTC)

    moon = ephem.Moon()
    obs = _observer(fishery, start_utc)
    moon.compute(obs)
    phase_frac = float(moon.moon_phase)

    def first_in_day(method_name: str) -> datetime | None:
        o = _observer(fishery, start_utc)
        try:
            t = _to_local(getattr(o, method_name)(ephem.Moon()), tz)
        except (ephem.CircumpolarError, ephem.NeverUpError):
            return None
        return t if day_start <= t < day_end else None

    transits: list[datetime] = []
    for method in ("next_transit", "next_antitransit"):
        o = _observer(fishery, start_utc)
        cursor = start_utc
        while True:
            o.date = ephem.Date(cursor.replace(tzinfo=None))
            t = _to_local(getattr(o, method)(ephem.Moon()), tz)
            if t >= day_end:
                break
            if t >= day_start:
                transits.append(t)
            cursor = t.astimezone(UTC) + timedelta(minutes=1)

    return MoonInfo(
        phase_frac=phase_frac,
        rise=first_in_day("next_rising"),
        set=first_in_day("next_setting"),
        transits=sorted(transits),
    )


def solunar_periods(fishery: Fishery, day: date) -> list[SolunarPeriod]:
    info = moon_info(fishery, day)
    periods = [
        SolunarPeriod("major", t - timedelta(hours=1), t + timedelta(hours=1))
        for t in info.transits
    ]
    for edge in (info.rise, info.set):
        if edge is not None:
            periods.append(
                SolunarPeriod("minor", edge - timedelta(minutes=30), edge + timedelta(minutes=30))
            )
    return sorted(periods, key=lambda p: p.start)
