import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="TideScout: SC inshore fishing decision support.")
console = Console()


def _snap_zero(v: float) -> float:
    """Round values that display as zero at 1 decimal to plain 0.0.

    Prevents "-0.0" from printing under a signed (`:+.1f`) format at slack
    water / a flat pressure trend, where the true value is a tiny negative
    float that rounds to zero but keeps its sign.
    """
    return 0.0 if abs(v) < 0.05 else v


@app.callback()
def _root() -> None:
    """TideScout CLI."""


@app.command()
def stations(slug: str) -> None:
    """Discover NOAA and USGS stations for a fishery's area."""
    from tidescout.config import load_fishery
    from tidescout.sources import discovery
    from tidescout.sources.cache import default_cache

    fishery = load_fishery(slug)
    cache = default_cache()
    table = Table(title=f"Stations near {fishery.name}")
    for col in ("kind", "id", "name", "lat", "lon"):
        table.add_column(col)
    rows = (
        discovery.find_tide_stations(fishery, cache)
        + discovery.find_current_stations(fishery, cache)
        + [
            s
            for param in ("00060", "00010", "00480")
            for s in discovery.find_usgs_sites(fishery, cache, param)
        ]
    )
    seen = set()
    for s in rows:
        if (s.kind, s.id) in seen:
            continue
        seen.add((s.kind, s.id))
        table.add_row(s.kind, s.id, s.name, f"{s.lat:.4f}", f"{s.lon:.4f}")
    console.print(table)


@app.command()
def conditions(
    slug: str,
    date_str: str = typer.Option(None, "--date", help="YYYY-MM-DD (default: today)"),
    model: str = typer.Option("best", "--model", help="best|gfs|ecmwf|icon|hrrr|nbm"),
) -> None:
    """Print a day of hourly conditions for a fishery."""
    from datetime import date as date_cls
    from datetime import datetime as datetime_cls
    from zoneinfo import ZoneInfo

    from tidescout.config import load_fishery
    from tidescout.sources.cache import default_cache
    from tidescout.sources.dayloader import load_day
    from tidescout.sources.weather import WEATHER_MODELS

    if model not in WEATHER_MODELS:
        raise typer.BadParameter(f"model must be one of {sorted(WEATHER_MODELS)}")
    fishery = load_fishery(slug)
    # Fishery-local "today", not the operator's machine timezone -- same
    # reasoning as weather._today(): near midnight ET those can disagree.
    day = (
        date_cls.fromisoformat(date_str)
        if date_str
        else datetime_cls.now(ZoneInfo(fishery.timezone)).date()
    )
    result = load_day(fishery, day, model, default_cache())

    table = Table(title=f"{fishery.name} — {day} — model: {result.model_label}")
    for col in ("hour", "tide ft", "stage", "cur kn", "wind", "gust", "press", "trend", "cloud", "air°F", "solunar"):
        table.add_column(col, justify="right")
    for h in result.hours:
        arrow = {"rising": "↑", "falling": "↓"}.get(h.tide_phase or "", "")
        wind = (
            f"{h.wind_speed_kn:.0f}@{h.wind_dir_deg:.0f}"
            if h.wind_speed_kn is not None and h.wind_dir_deg is not None
            else "—"
        )
        table.add_row(
            h.time.strftime("%H:%M"),
            f"{h.tide_height_ft:.1f}" if h.tide_height_ft is not None else "—",
            f"{arrow}{h.tide_frac:.0%}" if h.tide_frac is not None else "—",
            f"{_snap_zero(h.current_speed_kn):+.1f}" if h.current_speed_kn is not None else "—",
            wind,
            f"{h.wind_gust_kn:.0f}" if h.wind_gust_kn is not None else "—",
            f"{h.pressure_mb:.1f}" if h.pressure_mb is not None else "—",
            f"{_snap_zero(h.pressure_trend_mb_3h):+.1f}" if h.pressure_trend_mb_3h is not None else "—",
            f"{h.cloud_cover_pct:.0f}%" if h.cloud_cover_pct is not None else "—",
            f"{h.air_temp_f:.0f}" if h.air_temp_f is not None else "—",
            ",".join(h.solunar) or "—",
        )
    console.print(table)
    if result.sun:
        console.print(
            f"sun: dawn {result.sun.dawn:%H:%M} rise {result.sun.sunrise:%H:%M} "
            f"set {result.sun.sunset:%H:%M} dusk {result.sun.dusk:%H:%M}"
        )
    if result.moon:
        console.print(f"moon: {result.moon.phase_frac:.0%} illuminated, transits {[t.strftime('%H:%M') for t in result.moon.transits]}")
    if result.water:
        trend = f" ({result.water.temp_trend_f_3d:+.1f}°F/3d)" if result.water.temp_trend_f_3d is not None else ""
        console.print(f"water: {result.water.temp_f:.0f}°F{trend}, salinity {result.water.salinity_ppt:.0f} ppt [{result.water.source}]")
    if result.discharge:
        console.print(f"discharge: {result.discharge.bucket} ({result.discharge.cfs_now:,.0f} cfs now)" if result.discharge.cfs_now else f"discharge: {result.discharge.bucket}")
    if result.missing:
        console.print(f"[yellow]missing sources: {', '.join(result.missing)}[/yellow]")


def main() -> None:
    app()
