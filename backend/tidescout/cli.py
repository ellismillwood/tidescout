import json
import time

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="TideScout: SC inshore fishing decision support.")
bathy_app = typer.Typer(no_args_is_help=True, help="Bathymetry tile discovery and processing.")
app.add_typer(bathy_app, name="bathy")
flow_app = typer.Typer(no_args_is_help=True, help="ANUGA flow-state library.")
app.add_typer(flow_app, name="flow")
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
    columns = (
        "hour", "tide ft", "stage", "cur kn", "wind", "gust", "press",
        "trend", "cloud", "air°F", "solunar",
    )
    for col in columns:
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
            (
                f"{_snap_zero(h.pressure_trend_mb_3h):+.1f}"
                if h.pressure_trend_mb_3h is not None
                else "—"
            ),
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
        transits = [t.strftime("%H:%M") for t in result.moon.transits]
        console.print(f"moon: {result.moon.phase_frac:.0%} illuminated, transits {transits}")
    if result.water:
        trend = (
            f" ({result.water.temp_trend_f_3d:+.1f}°F/3d)"
            if result.water.temp_trend_f_3d is not None
            else ""
        )
        console.print(
            f"water: {result.water.temp_f:.0f}°F{trend}, "
            f"salinity {result.water.salinity_ppt:.0f} ppt [{result.water.source}]"
        )
    if result.discharge:
        console.print(
            f"discharge: {result.discharge.bucket} ({result.discharge.cfs_now:,.0f} cfs now)"
            if result.discharge.cfs_now
            else f"discharge: {result.discharge.bucket}"
        )
    if result.missing:
        console.print(f"[yellow]missing sources: {', '.join(result.missing)}[/yellow]")


@app.command()
def features(slug: str, rebuild: bool = typer.Option(False, "--rebuild")) -> None:
    """Build (if needed) and summarize the ambush-feature inventory."""
    from shapely.geometry import shape

    from tidescout.config import load_fishery
    from tidescout.paths import fishery_data_dir
    from tidescout.pipeline.features import build_features, load_features

    fishery = load_fishery(slug)
    path = fishery_data_dir(slug) / "features.geojson"
    if rebuild or not path.exists():
        build_features(slug, fishery)
    fc = load_features(slug)
    counts: dict[str, int] = {}
    for f in fc["features"]:
        counts[f["properties"]["type"]] = counts.get(f["properties"]["type"], 0) + 1
    table = Table(title=f"{fishery.name} — ambush features")
    table.add_column("type")
    table.add_column("count", justify="right")
    for k in sorted(counts):
        table.add_row(k, str(counts[k]))
    console.print(table)
    sized = [f for f in fc["features"] if "area_m2" in f["properties"]]
    for f in sorted(sized, key=lambda x: -x["properties"]["area_m2"])[:5]:
        # Deviation from a literal "take the exterior ring's first vertex"
        # reading: that point can sit arbitrarily far from the feature's
        # visual location for an elongated/irregular polygon (a channel-
        # hugging wall, an oxbow bar), which would make the printed lat/lon
        # misleading rather than merely imprecise. shapely's centroid is the
        # actual area-weighted center the docstring/brief text calls for,
        # and geometry is already in EPSG:4326 here so no extra transform.
        lon, lat = shape(f["geometry"]).centroid.coords[0]
        console.print(
            f"  {f['id']}: {f['properties']['area_m2']:,.0f} m² near ({lat:.4f}, {lon:.4f})"
        )


@app.command()
def spots(slug: str) -> None:
    """Show each known spot's nearest detected feature (static validation aid)."""
    from rasterio.warp import transform as warp_transform
    from shapely.geometry import LineString, Point, Polygon, shape

    from tidescout.config import load_fishery, load_known_spots
    from tidescout.paths import fishery_data_dir
    from tidescout.pipeline.features import load_features

    fishery = load_fishery(slug)
    known = load_known_spots(slug)
    if not known:
        console.print(f"no spots in fisheries/{slug}.known-spots.yaml — add some!")
        return
    path = fishery_data_dir(slug) / "features.geojson"
    if not path.exists():
        console.print(
            f"no features.geojson for {slug} — run `tidescout features {slug} --rebuild` first"
        )
        return
    fc = load_features(slug)
    epsg = fishery.bathymetry.epsg

    def to_utm(lons, lats):
        return warp_transform("EPSG:4326", f"EPSG:{epsg}", lons, lats)

    def to_utm_geom(g):
        """Reproject a feature's full geometry to UTM for exact `distance`."""

        def tx(coords):
            xs, ys = to_utm(*zip(*coords, strict=True))
            return list(zip(xs, ys, strict=True))

        if g.geom_type == "Point":
            return Point(tx([(g.x, g.y)])[0])
        if g.geom_type == "LineString":
            return LineString(tx(list(g.coords)))
        if g.geom_type == "Polygon":
            return Polygon(tx(list(g.exterior.coords)), [tx(list(r.coords)) for r in g.interiors])
        raise TypeError(f"unsupported geometry: {g.geom_type}")

    feats = [
        (f["id"], f["properties"]["type"], to_utm_geom(shape(f["geometry"])))
        for f in fc["features"]
    ]
    table = Table(title=f"{fishery.name} — known spots vs detected features")
    for col in ("spot", "nearest feature", "type", "distance m"):
        table.add_column(col)
    for s in known:
        (sx,), (sy,) = to_utm([s.lon], [s.lat])
        p = Point(sx, sy)
        best = min(
            ((fid, ftype, p.distance(geom)) for fid, ftype, geom in feats),
            key=lambda r: r[2],
        )
        table.add_row(s.name, best[0], best[1], f"{best[2]:,.0f}")
    console.print(table)


@bathy_app.command()
def discover(
    slug: str,
    catalog_url: str = typer.Option(
        "https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_19as/catalog.html",
        "--catalog-url",
    ),
) -> None:
    """Find CUDEM tiles intersecting the fishery bbox; verify and record a manifest.

    Live source is NCEI THREDDS, not S3: no public S3/GeoTIFF bucket exists for
    CUDEM (verified against the AWS Open Data Registry and direct bucket-name
    probes -- see the cudem.py module comment and task-4-report.md for the full
    resolution-ladder log). cudem.py still carries the brief's original
    list_s3_keys/candidate_tiles S3 path, tested and intact, for a genuinely
    S3-hosted source if one is ever used.
    """
    import rasterio

    from tidescout.config import load_fishery
    from tidescout.sources import cudem

    fishery = load_fishery(slug)
    keys = cudem.list_thredds_keys(catalog_url)
    candidate_keys = cudem.thredds_candidate_keys(fishery, keys)
    console.print(f"{len(keys)} keys under catalog; {len(candidate_keys)} intersect bbox")
    entries = []
    for key in candidate_keys:
        # rasterio-only DAP connection string -- never persisted (see
        # thredds_tile_url's docstring); the manifest gets thredds_file_url()'s
        # plain downloadable URL instead, via thredds_manifest_entry() below.
        dap_url = cudem.thredds_tile_url(key)
        with rasterio.open(dap_url) as src:  # OPeNDAP metadata read only, no full download
            b = src.bounds
            entry = cudem.thredds_manifest_entry(
                key, (b.left, b.bottom, b.right, b.top), str(src.crs)
            )
            entries.append(entry)
        console.print(f"  ok {key} bounds={entry['bounds']}")
    path = cudem.write_manifest(slug, entries)
    console.print(f"manifest written: {path} ({len(entries)} tiles)")


@bathy_app.command()
def build(slug: str) -> None:
    """Download manifest tiles (cached), build the UTM analysis raster, and
    derive slope/curvature/zones rasters from it."""
    from tidescout.config import load_fishery
    from tidescout.pipeline.bathy import build_bathy, ensure_tiles
    from tidescout.pipeline.derivatives import build_derivatives
    from tidescout.sources.cudem import load_manifest

    fishery = load_fishery(slug)
    entries = load_manifest(slug)
    if not entries:
        raise typer.BadParameter(f"no tile manifest — run `tidescout bathy discover {slug}` first")
    tile_paths = ensure_tiles(slug, entries)
    out = build_bathy(fishery, tile_paths)
    meta = json.loads((out.parent / "bathy_meta.json").read_text())
    console.print(f"built {out}: {meta['width']}x{meta['height']} @10m, stats={meta['stats']}")
    deriv_paths = build_derivatives(slug, fishery)
    for name, path in deriv_paths.items():
        console.print(f"  derived {name}: {path}")


@bathy_app.command()
def artifacts(slug: str) -> None:
    """Render hillshade, quicklook PNG, and contour GeoJSON."""
    from tidescout.config import load_fishery
    from tidescout.pipeline.artifacts import build_artifacts

    fishery = load_fishery(slug)
    for name, path in build_artifacts(slug, fishery).items():
        console.print(f"{name}: {path} ({path.stat().st_size:,} bytes)")


@bathy_app.command("wetlands")
def bathy_wetlands(slug: str) -> None:
    """Fetch USFWS NWI wetlands intersecting the fishery bbox.

    Independent of CUDEM: bathymetry-threshold land/water is biased in
    vegetated marsh (lidar returns off canopy), so this is a second,
    unauthenticated public source for where marsh actually is. Filed under
    `bathy`, not `flow` -- it is bathymetry-adjacent land/water reference data,
    not part of the ANUGA mesh/regime pipeline, and is not wired into either
    yet (see sources/nwi.py module docstring).
    """
    from tidescout.config import load_fishery
    from tidescout.sources import nwi
    from tidescout.sources.cache import default_cache

    fishery = load_fishery(slug)
    fc = nwi.fetch_wetlands(fishery, default_cache())
    path = nwi.wetlands_path(slug)
    console.print(
        f"{len(fc['features'])} wetland features -> {path} ({path.stat().st_size:,} bytes)"
    )


@flow_app.command("mesh")
def flow_mesh(slug: str) -> None:
    """Build the mesh and report its size without simulating."""
    from tidescout.config import load_fishery
    from tidescout.pipeline import mesh as meshmod

    fishery = load_fishery(slug)
    domain = meshmod.build_mesh(slug, fishery)
    console.print(
        f"{len(domain.triangles):,} triangles "
        f"(base {fishery.anuga.base_edge_m:.0f} m, jetty {fishery.anuga.jetty_edge_m:.0f} m)"
    )


@flow_app.command("run")
def flow_run(
    slug: str,
    workers: int = typer.Option(0, "--workers", help="0 = use anuga.max_workers"),
    sim_hours: float = typer.Option(0.0, "--sim-hours", help="0 = full spin-up + cycle"),
) -> None:
    """Run the full regime matrix as parallel processes."""
    from tidescout.pipeline.regimes import build_library

    # A nine-regime build is five to six hours. Print each regime the moment
    # it lands rather than only the table at the end -- build #1 sat with six
    # dead regimes for over an hour because the first output of any kind came
    # after all nine had finished.
    started = time.monotonic()

    def report(name: str, meta: dict) -> None:
        ok = meta.get("status") == "ok"
        mins = (time.monotonic() - started) / 60.0
        if ok:
            console.print(
                f"[green]ok[/green]   {name:12s} "
                f"{meta.get('wall_seconds', 0) / 60.0:6.1f} min  "
                f"resid {meta.get('mass_residual', float('nan')):.1e}  "
                f"reversed={meta.get('reversal', {}).get('reversed', '-')}  "
                f"[dim](+{mins:.0f} min elapsed)[/dim]"
            )
        else:
            console.print(
                f"[red]FAIL[/red] {name:12s} {meta.get('error', '?')}  "
                f"[dim](+{mins:.0f} min elapsed)[/dim]"
            )

    results = build_library(
        slug,
        max_workers=workers or None,
        sim_hours=sim_hours or None,
        on_result=report,
    )
    table = Table(title=f"{slug} — regime library")
    for col in ("regime", "status", "triangles", "wall s", "mass resid", "reversed"):
        table.add_column(col)
    for name in sorted(results):
        m = results[name]
        triangles = m.get("triangles")
        table.add_row(
            name, m.get("status", "?"),
            f"{triangles:,}" if triangles is not None else "-",
            str(m.get("wall_seconds", "-")),
            f"{m.get('mass_residual', float('nan')):.2e}",
            str(m.get("reversal", {}).get("reversed", "-")),
        )
    console.print(table)


@flow_app.command("validate")
def flow_validate(
    slug: str,
    regime: str = typer.Option("mean_med", "--regime"),
    radius_m: float = typer.Option(150.0, "--radius"),
) -> None:
    """Compare the flow library against Ellis's known spots (the spec's gate)."""
    import numpy as np
    from rasterio.warp import transform as warp_transform

    from tidescout.config import load_fishery, load_known_spots
    from tidescout.engine import flow
    from tidescout.paths import fishery_data_dir
    from tidescout.pipeline import flowlib

    fishery = load_fishery(slug)
    spots = load_known_spots(slug)
    if not spots:
        console.print(f"no spots in fisheries/{slug}.known-spots.yaml — add some!")
        raise typer.Exit(1)

    grid_path = fishery_data_dir(slug) / "flow" / regime / "grid" / "grid.json"
    if not grid_path.exists():
        console.print(
            f"[red]no rasterised library for regime {regime!r}[/red] "
            f"(expected {grid_path}).\nRun the build, then `flowlib.rasterise_regime`."
        )
        raise typer.Exit(1)
    grid_meta = json.loads(grid_path.read_text())
    phases = grid_meta["phases"]
    states = flow.tide_states(grid_meta["stage_bc_m"])

    spec = flowlib.grid_spec(slug, fishery)
    xs, ys = warp_transform(
        "EPSG:4326",
        f"EPSG:{fishery.bathymetry.epsg}",
        [s.lon for s in spots],
        [s.lat for s in spots],
    )

    # Load each phase ONCE and evaluate every spot against it. The obvious
    # loop order (spot, then phase) re-reads the whole grid file per spot.
    near = {}
    for spot, sx, sy in zip(spots, xs, ys, strict=True):
        near[spot.name] = (spec.xs - sx) ** 2 + (spec.ys - sy) ** 2 <= radius_m**2
    peak_speed = {s.name: [] for s in spots}
    spread = {s.name: [] for s in spots}
    for i in range(len(phases)):
        st = flowlib.load_state(slug, regime, i)
        sp, _ = flow.speed_direction(st["u"], st["v"])
        for spot in spots:
            sel = near[spot.name]
            if not sel.any():
                continue
            here = sp[sel]
            peak_speed[spot.name].append(float(np.nanmax(here)))
            # Within-radius max minus min: a proxy for a seam. Deliberately NOT
            # called shear -- that is the velocity gradient
            # `engine.structure.strain_rate` computes, and conflating the two
            # would hide which one was used.
            spread[spot.name].append(float(np.nanmax(here) - np.nanmin(here)))

    table = Table(title=f"{fishery.name} — known spots vs flow library ({regime})")
    for col in ("spot", "expects", "peak speed", "at phase", "peak state",
                "speed spread", "slack min", "agrees"):
        table.add_column(col)

    for spot in spots:
        if not near[spot.name].any():
            table.add_row(spot.name, spot.works_on or "-", "OUTSIDE DOMAIN",
                          "-", "-", "-", "-", "-")
            continue
        speeds = peak_speed[spot.name]
        peak_i = int(np.argmax(speeds))
        peak_state = states[peak_i]
        if not spot.works_on:
            agrees = "[dim]no hint[/dim]"
        elif spot.works_on == "slack":
            # An eddy or current break is defined by CONTRAST, not by a peak:
            # fast water beside a low-speed pocket. Judge it on the spread.
            agrees = "[green]yes[/green]" if max(spread[spot.name]) > min(speeds) else "?"
        else:
            agrees = "[green]yes[/green]" if peak_state == spot.works_on else "[red]no[/red]"
        table.add_row(
            spot.name, spot.works_on or "-", f"{max(speeds):.2f} m/s",
            f"{phases[peak_i]:.2f}", peak_state, f"{max(spread[spot.name]):.2f}",
            f"{min(speeds):.2f}", agrees,
        )
    console.print(table)
    console.print(
        "\n[bold]This is a judgement call, not a pass/fail.[/bold] `agrees` only asks "
        "whether the peak landed on the half of the cycle the notes describe.\n"
        "Ebb/flood spots should peak on their stated half; a slack/eddy spot shows its "
        "signature as a low minimum beside a high peak — the contrast, not the value.\n"
        "Phase 0 is LOW water here (spin-up is 0.483 of a cycle), so flood runs "
        "phase 0→0.5 and ebb 0.5→1.0.\n"
        "[dim]Do not tune thresholds to make this pass. If the model does not "
        "reproduce a spot, that is the finding.[/dim]"
    )
    for spot in spots:
        if spot.notes:
            console.print(f"  [dim]{spot.name}: {spot.notes}[/dim]")


@flow_app.command("structure")
def flow_structure(
    slug: str,
    regime: str = typer.Option("mean_med", "--regime"),
    phase: int = typer.Option(-1, "--phase", help="phase index; -1 = every phase"),
    top: int = typer.Option(15, "--top", help="features to list"),
) -> None:
    """Derived structure at the feature inventory: seams, eddies, ambush points."""
    import numpy as np

    from tidescout.config import load_fishery
    from tidescout.engine import activation, flow
    from tidescout.paths import fishery_data_dir
    from tidescout.pipeline import flowlib, schedule
    from tidescout.pipeline.features import load_features

    fishery = load_fishery(slug)
    # Every prerequisite is checked before anything reads it. `grid_spec` calls
    # `read_bathy` and `load_features` opens features.geojson, and both used to
    # run BEFORE the library guard below -- so a fresh checkout that had not run
    # `tidescout bathy build` or `tidescout features` got a raw
    # FileNotFoundError traceback out of this command's third line.
    data_dir = fishery_data_dir(slug)
    for path, remedy in (
        (data_dir / "bathy_meta.json", f"tidescout bathy build {slug}"),
        (data_dir / "bathy_utm.tif", f"tidescout bathy build {slug}"),
        (data_dir / "features.geojson", f"tidescout features {slug} --rebuild"),
    ):
        if not path.exists():
            console.print(
                f"[red]no {path.name} for {slug}[/red] (expected {path}).\n"
                f"Run `{remedy}` first."
            )
            raise typer.Exit(1)
    grid_path = data_dir / "flow" / regime / "grid" / "grid.json"
    if not grid_path.exists():
        console.print(
            f"[red]no rasterised library for regime {regime!r}[/red] "
            f"(expected {grid_path}).\nRun the build, then `flowlib.rasterise_regime`."
        )
        raise typer.Exit(1)
    grid_meta = json.loads(grid_path.read_text())

    # Validate at the boundary. `phase < 0` means "every phase", so an
    # unvalidated `--phase -2` silently ran the whole sweep instead of erroring,
    # and an out-of-range positive failed deep inside `load_state`.
    n_phases = len(grid_meta["phases"])
    if phase < -1 or phase >= n_phases:
        console.print(
            f"[red]--phase {phase} is out of range[/red]: this library has "
            f"{n_phases} phases (0-{n_phases - 1}), and -1 means every phase."
        )
        raise typer.Exit(1)

    spec = flowlib.grid_spec(slug, fishery)
    feats = load_features(slug)["features"]
    states = flow.tide_states(grid_meta["stage_bc_m"])
    sched = schedule.cell_schedule(slug, regime)

    idxs = range(n_phases) if phase < 0 else [phase]
    best: dict[str, activation.FeatureMetrics] = {}
    best_state: dict[str, str] = {}
    for i in idxs:
        st = flowlib.load_state(slug, regime, i)
        fields = activation.structure_fields(
            st["u"], st["v"], st["depth"], spec, fishery.structure
        )
        for m in activation.sample_features(
            feats, spec, fields, sched, fishery.structure.ambush_radius_m
        ):
            prev = best.get(m.key)
            if prev is None or (np.nan_to_num(m.ambush) > np.nan_to_num(prev.ambush)):
                best[m.key] = m
                best_state[m.key] = states[i]

    ranked = sorted(
        (m for m in best.values() if m.n_cells),
        key=lambda m: np.nan_to_num(m.ambush),
        reverse=True,
    )[:top]

    def num(value: float, spec_: str) -> str:
        """NaN prints as an em dash, not as `nan`/`+nan`. Every one of these
        columns can legitimately be NaN -- a disc that is entirely dry at its
        best phase has no speed, no strain and no tensor at all -- and only
        `wet frac` used to say so."""
        return "-" if np.isnan(value) else format(value, spec_)

    table = Table(title=f"{fishery.name} — derived structure ({regime})")
    for col in ("feature", "type", "best state", "ambush m/s", "speed",
                "strain 1/s", "okubo W", "eddy %", "converg.", "wet frac"):
        table.add_column(col)
    for m in ranked:
        table.add_row(
            m.key, m.type, best_state[m.key],
            num(m.ambush, ".3f"), num(m.speed, ".3f"),
            num(m.strain, ".2e"), num(m.okubo_w, "+.2e"),
            num(m.eddy_share * 100.0, ".1f"),
            num(m.convergence, "+.2e"),
            num(m.wet_fraction, ".2f"),
        )
    console.print(table)
    console.print(
        "\n[dim]`okubo W` is MAX-reduced over the feature's disc, so it reports the "
        "strongest SEAM in it (W > 0, strain-dominated) and structurally cannot report "
        "an eddy: a max over a mixed disc returns its most seam-like cell. `eddy %` is "
        "the eddy channel — the share of the feature's WET cells that classify as "
        f"rotation-dominated (W < -{fishery.structure.quiet_w:g}). `ambush` is how much "
        f"faster the water within {fishery.structure.ambush_radius_m:.0f} m is than the "
        "feature itself — the current-shadow signal.[/dim]"
    )


def main() -> None:
    app()
