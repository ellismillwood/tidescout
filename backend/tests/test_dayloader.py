from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tidescout.config import load_fishery
from tidescout.errors import SourceUnavailable
from tidescout.sources import noaa, usgs, weather
from tidescout.sources.cache import Cache
from tidescout.sources.dayloader import load_day
from tidescout.sources.noaa import TideEvent, TideHour

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
