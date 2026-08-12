from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tidescout.config import load_fishery
from tidescout.errors import SourceUnavailable
from tidescout.sources import noaa, usgs, weather
from tidescout.sources.cache import Cache
from tidescout.sources.dayloader import load_day
from tidescout.sources.noaa import CurrentHour, TideEvent, TideHour

ET = ZoneInfo("America/New_York")
DAY = date(2026, 8, 15)


def _events() -> list[TideEvent]:
    # Continuous H/L coverage spanning the whole assembled day so every
    # hour of DAY is bracketed by some consecutive pair.
    out = []
    t = datetime(2026, 8, 14, 0, 0, tzinfo=ET)
    kind = "H"
    while t < datetime(2026, 8, 17, 0, 0, tzinfo=ET):
        out.append(TideEvent(t, kind, 5.0 if kind == "H" else 0.5))
        kind = "L" if kind == "H" else "H"
        t += timedelta(hours=6, minutes=12)
    return out


def _current_points() -> list[CurrentHour]:
    # ACT6531 shape: off-hour slack/max-flood/max-ebb points, continuous
    # coverage across the 3-day interpolation window so every hour of DAY
    # is bracketed.
    out = []
    t = datetime(2026, 8, 14, 0, 6, tzinfo=ET)
    speed, flood_dir, ebb_dir = -2.5, 320.0, 140.0
    while t < datetime(2026, 8, 17, 0, 0, tzinfo=ET):
        dir_deg = flood_dir if speed >= 0 else ebb_dir
        out.append(CurrentHour(t, speed, dir_deg))
        speed = -speed
        t += timedelta(hours=3, minutes=10)
    return out


def _unavailable(*_args, **_kwargs):
    raise SourceUnavailable("test", "stubbed failure")


def test_load_day_falls_back_to_interpolated_tides_when_hourly_unavailable(monkeypatch, tmp_path):
    # Winyah Bay shape: CO-OPS rejects interval=h for this subordinate
    # station (always raises), but hi/lo events are available -> dayloader
    # should recover hourly heights via cosine interpolation instead of
    # reporting tides as missing.
    fishery = load_fishery("winyah-bay")

    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: ([], "gfs"))
    monkeypatch.setattr(noaa, "tide_events", lambda *a, **k: _events())
    monkeypatch.setattr(noaa, "tide_hours", _unavailable)
    monkeypatch.setattr(noaa, "current_hours", _unavailable)
    monkeypatch.setattr(usgs, "water_summary", _unavailable)
    monkeypatch.setattr(usgs, "discharge_summary", _unavailable)

    cache = Cache(tmp_path / "cache.sqlite")
    result = load_day(fishery, DAY, "gfs", cache)

    assert "tides" not in result.missing
    assert "tide-events" not in result.missing
    assert all(h.tide_height_ft is not None for h in result.hours)
    # sources that really did fail are still reported
    assert "currents" in result.missing
    assert "water" in result.missing
    assert "discharge" in result.missing


def test_load_day_falls_back_to_interpolated_tides_when_hourly_succeeds_empty(
    monkeypatch, tmp_path
):
    # tide_hours can also "succeed" with a genuinely empty list -- no
    # exception at all -- and the fallback must still trigger on that path,
    # not just on SourceUnavailable.
    fishery = load_fishery("winyah-bay")

    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: ([], "gfs"))
    monkeypatch.setattr(noaa, "tide_events", lambda *a, **k: _events())
    monkeypatch.setattr(noaa, "tide_hours", lambda *a, **k: [])
    monkeypatch.setattr(noaa, "current_hours", _unavailable)
    monkeypatch.setattr(usgs, "water_summary", _unavailable)
    monkeypatch.setattr(usgs, "discharge_summary", _unavailable)

    cache = Cache(tmp_path / "cache.sqlite")
    result = load_day(fishery, DAY, "gfs", cache)

    assert "tides" not in result.missing
    assert all(h.tide_height_ft is not None for h in result.hours)


def test_load_day_marks_tides_missing_when_both_paths_empty(monkeypatch, tmp_path):
    fishery = load_fishery("winyah-bay")

    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: ([], "gfs"))
    monkeypatch.setattr(noaa, "tide_events", _unavailable)
    monkeypatch.setattr(noaa, "tide_hours", _unavailable)
    monkeypatch.setattr(noaa, "current_hours", _unavailable)
    monkeypatch.setattr(usgs, "water_summary", _unavailable)
    monkeypatch.setattr(usgs, "discharge_summary", _unavailable)

    cache = Cache(tmp_path / "cache.sqlite")
    result = load_day(fishery, DAY, "gfs", cache)

    assert "tides" in result.missing
    assert "tide-events" in result.missing
    assert all(h.tide_height_ft is None for h in result.hours)


def test_load_day_uses_direct_tide_hours_when_available(monkeypatch, tmp_path):
    # Regression guard: stations that DO support interval=h should keep the
    # brief's original behavior untouched -- no fallback, direct heights.
    fishery = load_fishery("winyah-bay")
    direct_hours = [TideHour(datetime(2026, 8, 15, h, 0, tzinfo=ET), float(h)) for h in range(24)]

    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: ([], "gfs"))
    monkeypatch.setattr(noaa, "tide_events", lambda *a, **k: _events())
    monkeypatch.setattr(noaa, "tide_hours", lambda *a, **k: direct_hours)
    monkeypatch.setattr(noaa, "current_hours", _unavailable)
    monkeypatch.setattr(usgs, "water_summary", _unavailable)
    monkeypatch.setattr(usgs, "discharge_summary", _unavailable)

    cache = Cache(tmp_path / "cache.sqlite")
    result = load_day(fishery, DAY, "gfs", cache)

    assert "tides" not in result.missing
    assert result.hours[5].tide_height_ft == 5.0


def test_load_day_interpolates_currents_when_off_hour(monkeypatch, tmp_path):
    # Addition #2: ACT6531 shape (subordinate current station, irregular
    # slack/max-flood/max-ebb timestamps) should no longer leave the
    # currents column silently blank -- dayloader interpolates onto the
    # hour grid before handing off to the assembler.
    fishery = load_fishery("winyah-bay")

    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: ([], "gfs"))
    monkeypatch.setattr(noaa, "tide_events", lambda *a, **k: _events())
    monkeypatch.setattr(noaa, "tide_hours", _unavailable)
    monkeypatch.setattr(noaa, "current_hours", lambda *a, **k: _current_points())
    monkeypatch.setattr(usgs, "water_summary", _unavailable)
    monkeypatch.setattr(usgs, "discharge_summary", _unavailable)

    cache = Cache(tmp_path / "cache.sqlite")
    result = load_day(fishery, DAY, "gfs", cache)

    assert "currents" not in result.missing
    assert all(h.current_speed_kn is not None for h in result.hours)
    assert all(h.current_dir_deg is not None for h in result.hours)


def test_load_day_marks_currents_missing_when_station_yields_no_points(monkeypatch, tmp_path):
    # Regression guard: a current station that exists but returns nothing
    # (or raises) must still land "currents" in missing exactly once, not
    # be silently dropped now that there's a post-processing step.
    fishery = load_fishery("winyah-bay")

    monkeypatch.setattr(weather, "fetch_weather", lambda *a, **k: ([], "gfs"))
    monkeypatch.setattr(noaa, "tide_events", lambda *a, **k: _events())
    monkeypatch.setattr(noaa, "tide_hours", _unavailable)
    monkeypatch.setattr(noaa, "current_hours", lambda *a, **k: [])
    monkeypatch.setattr(usgs, "water_summary", _unavailable)
    monkeypatch.setattr(usgs, "discharge_summary", _unavailable)

    cache = Cache(tmp_path / "cache.sqlite")
    result = load_day(fishery, DAY, "gfs", cache)

    assert result.missing.count("currents") == 1
    assert all(h.current_speed_kn is None for h in result.hours)
