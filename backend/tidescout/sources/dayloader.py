from datetime import date

from tidescout.engine import tides
from tidescout.engine.conditions import DayConditions, assemble_day
from tidescout.errors import SourceUnavailable
from tidescout.models import Fishery
from tidescout.sources import astronomy, noaa, usgs, weather
from tidescout.sources.cache import Cache


def load_day(fishery: Fishery, day: date, model_key: str, cache: Cache) -> DayConditions:
    missing: list[str] = []

    def attempt(name: str, fn, default):
        try:
            return fn()
        except SourceUnavailable:
            missing.append(name)
            return default
        # Any other unexpected error from a single source must not crash the
        # whole `conditions` command; record it and keep going.
        except Exception:  # noqa: BLE001
            missing.append(name)
            return default

    weather_48h, label = attempt(
        "weather", lambda: weather.fetch_weather(fishery, day, model_key, cache), ([], model_key)
    )

    tide_station = fishery.stations.tide[0] if fishery.stations.tide else None
    if tide_station:
        # Fetch hi/lo events first: some subordinate stations (e.g. Winyah
        # Bay 8662549) reject interval=h hourly predictions outright but do
        # serve interval=hilo, so events are needed both for tide_phase/frac
        # (via stage_at) and as the input to the interpolation fallback below.
        events = attempt(
            "tide-events", lambda: noaa.tide_events(tide_station, day, fishery.timezone, cache), []
        )
        try:
            tide_hours = noaa.tide_hours(tide_station, day, fishery.timezone, cache)
        # Winyah Bay 8662549 always lands here (CO-OPS rejects interval=h for
        # this subordinate station); any other unexpected error falls back
        # the same way rather than crashing the command.
        except Exception:  # noqa: BLE001
            tide_hours = []
        # The fallback also applies when tide_hours "succeeds" with a
        # genuinely empty list (no exception at all) -- not just when it
        # raises -- so both failure shapes land here the same way.
        if not tide_hours and events:
            tide_hours = tides.interpolate_tide_hours(events, day, fishery.timezone)
        if not tide_hours:
            missing.append("tides")
    else:
        missing.append("tides")
        events = []
        tide_hours = []

    current_station = fishery.stations.currents[0] if fishery.stations.currents else None
    if current_station:
        # Subordinate current stations (e.g. ACT6531 at the Winyah Bay
        # entrance) predict at irregular slack/max-flood/max-ebb times, not
        # top-of-hour, so raw points are interpolated onto the hour grid
        # before assembly -- same shape as the tide fallback above.
        try:
            points = noaa.current_hours(current_station, day, fishery.timezone, cache)
        except Exception:  # noqa: BLE001
            points = []
        currents = tides.interpolate_current_hours(points, day, fishery.timezone) if points else []
        if not currents:
            missing.append("currents")
    else:
        missing.append("currents")
        currents = []
    sun = attempt("sun", lambda: astronomy.sun_times(fishery, day), None)
    moon = attempt("moon", lambda: astronomy.moon_info(fishery, day), None)
    solunar = attempt("solunar", lambda: astronomy.solunar_periods(fishery, day), [])
    water = attempt("water", lambda: usgs.water_summary(fishery, cache, day.month), None)
    discharge = attempt("discharge", lambda: usgs.discharge_summary(fishery, cache), None)

    return assemble_day(
        fishery=fishery, day=day, model_label=label, weather_48h=weather_48h,
        tides=tide_hours, events=events, currents=currents, sun=sun, moon=moon,
        solunar=solunar, water=water, discharge=discharge, missing=missing,
    )
