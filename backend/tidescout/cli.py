import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="TideScout: SC inshore fishing decision support.")
console = Console()


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


def main() -> None:
    app()
