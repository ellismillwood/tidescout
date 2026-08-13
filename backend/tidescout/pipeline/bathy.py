"""Download, mosaic, clip, and reproject CUDEM bathymetry tiles into the
per-fishery UTM analysis raster.

Transport note (task 4 -> task 5 handoff; see task-5-brief.md "APPROVED
ADAPTATIONS" and task-4-report.md): CUDEM has no public GeoTIFF/S3
distribution. The live source is NCEI THREDDS, and the manifest's `url`
field is a plain HTTPS fileServer URL (verified live serving real bytes,
HTTP 206 partial content) -- httpx streams it exactly like a GeoTIFF would
be streamed, no special handling needed for *download*.

What differs from the brief's original GeoTIFF assumption is *local
opening*: the downloaded files are NetCDF (.nc). `_open_tile` below is
where that's handled, so the mosaic/clip/reproject math never has to know
which format it got.
"""

import json
import subprocess
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReader
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir, tiles_dir

NODATA = -9999.0

# Deviation from the brief's "size within 1% of the server's Content-Length"
# skip rule (approved adaptation #3, task-5-brief.md): the THREDDS
# fileServer origin omits Content-Length on HEAD responses entirely (only
# GET responses carry a real size, via `content-range` -- verified live in
# task 4). A HEAD-based precheck is therefore not possible without doing
# most of the work of a real download first. Every real tile is 79-139 MB,
# so "exists and > 10 MB" safely distinguishes a completed prior download
# from a missing/truncated one without an extra network round trip.
MIN_CACHED_TILE_BYTES = 10 * 1024 * 1024

# Live discovery (task 5, 2026-08-13): the THREDDS fileServer origin can
# stall mid-stream on a long-lived httpx connection -- observed both as a
# fast `RemoteProtocolError` (peer closed connection, incomplete chunked
# read) and, worse, as a silent hang with no exception at all (a `.part`
# file stopped growing and the process just sat there for over an hour).
# httpx timeouts don't reliably catch the second case. HTTPX_ATTEMPTS gives
# a few fresh-connection retries; if those are all exhausted, `_download_tile`
# falls back to `curl` (separate process, its own connection/timeout
# handling, resumable via `-C -`, which suits a server that stalls
# mid-transfer more than one that outright refuses).
HTTPX_ATTEMPTS = 3
CURL_TIMEOUT_S = 900


def ensure_tiles(slug: str, entries: list[dict]) -> list[Path]:
    """Download each manifest entry's tile to tiles_dir(slug), skipping files
    that already look like a complete prior download (see
    MIN_CACHED_TILE_BYTES above for why size-based, not Content-Length-based).
    """
    out = []
    for e in entries:
        dest = tiles_dir(slug) / e["url"].rsplit("/", 1)[-1]
        if not (dest.exists() and dest.stat().st_size > MIN_CACHED_TILE_BYTES):
            _download_tile(e["url"], dest)
        out.append(dest)
    return out


def _download_tile(url: str, dest: Path, attempts: int = HTTPX_ATTEMPTS) -> None:
    """Download one tile to `dest`, retrying transient httpx failures with a
    fresh connection each time (default 3 attempts) before falling back to
    `curl` for a final attempt. Writes to a `.part` sibling and atomically
    renames on success, so a killed/interrupted run never leaves a file that
    looks complete. See HTTPX_ATTEMPTS' comment above for why both layers
    exist.
    """
    tmp = dest.with_suffix(".part")
    last_exc: httpx.HTTPError | None = None
    for _ in range(attempts):
        try:
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            tmp.rename(dest)
            return
        except httpx.HTTPError as exc:
            last_exc = exc
            continue

    # httpx exhausted its attempts -- fall back to curl. Deliberately leaves
    # `tmp` (the last httpx attempt's partial bytes, if any) in place: curl's
    # `-C -` resumes from whatever is already there instead of restarting.
    try:
        result = subprocess.run(
            ["curl", "-fL", "--retry", "3", "--retry-delay", "5", "-C", "-", "-o", str(tmp), url],
            capture_output=True,
            timeout=CURL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"failed to download {url}: {attempts} httpx attempts failed ({last_exc!r}), "
            f"curl fallback also timed out after {CURL_TIMEOUT_S}s"
        ) from exc
    if result.returncode != 0 or not tmp.exists():
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        raise RuntimeError(
            f"failed to download {url}: {attempts} httpx attempts failed ({last_exc!r}), "
            f"curl fallback also failed (exit {result.returncode}): {stderr[:500]}"
        )
    tmp.rename(dest)


def _open_tile(path: Path) -> DatasetReader:
    """Open one tile file for pixel access.

    Handles both the synthetic tests' small GeoTIFFs and CUDEM's real
    NetCDF (.nc) tiles. Ladder: try a direct open first -- this always
    works for GeoTIFF, and also works for CUDEM's NetCDF whenever the file
    is single-variable, which the brief's live discovery (task 4) reports
    as the common case and which was confirmed live for winyah-bay's first
    downloaded tile (see task-5-report.md). If that open yields no readable
    band or no CRS, fall back to GDAL's subdataset syntax
    (`NETCDF:<path>:<variable>`), picking a likely elevation variable name
    from whatever `rasterio.open(path).subdatasets` actually reports rather
    than guessing blind. A direct open that raises outright is left to
    propagate -- there is no reliable way to recover a subdataset list from
    a failed open, and silently guessing a connection string in that case
    would hide a real transport/format problem instead of surfacing it.
    """
    ds = rasterio.open(path)
    if ds.count >= 1 and ds.crs is not None:
        return ds
    subdatasets = ds.subdatasets
    ds.close()
    if not subdatasets:
        raise RasterioIOError(f"{path}: opened with no bands/crs and no subdatasets")
    preferred_names = ("elevation", "Elevation", "z", "Band1", "topography")
    chosen = next(
        (sub for name in preferred_names for sub in subdatasets if sub.rsplit(":", 1)[-1] == name),
        subdatasets[0],
    )
    return rasterio.open(chosen)


def build_bathy(fishery: Fishery, tile_paths: list[Path]) -> Path:
    west, south, east, north = fishery.bbox
    sources = [_open_tile(p) for p in tile_paths]
    try:
        mosaic, src_transform = merge(sources, bounds=(west, south, east, north), nodata=NODATA)
        src_crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    band = mosaic[0]
    dst_crs = f"EPSG:{fishery.bathymetry.epsg}"
    cell = fishery.bathymetry.cell_m
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, band.shape[1], band.shape[0],
        left=west, bottom=south, right=east, top=north, resolution=cell,
    )
    dst = np.full((height, width), NODATA, dtype="float32")
    reproject(
        band, dst, src_transform=src_transform, src_crs=src_crs,
        dst_transform=transform, dst_crs=dst_crs,
        src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.bilinear,
    )
    out_dir = fishery_data_dir(fishery.slug)
    out = out_dir / "bathy_utm.tif"
    with rasterio.open(
        out, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32",
        crs=dst_crs, transform=transform, nodata=NODATA, compress="lzw",
    ) as dst_file:
        dst_file.write(dst, 1)
    valid = dst[dst != NODATA]
    meta = {
        "crs": dst_crs,
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "width": width, "height": height,
        "stats": {
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "pct_nodata": float((dst == NODATA).mean() * 100.0),
        },
    }
    (out_dir / "bathy_meta.json").write_text(json.dumps(meta, indent=1))
    return out


def read_bathy(slug: str) -> tuple[np.ndarray, Affine, dict]:
    out_dir = fishery_data_dir(slug)
    meta = json.loads((out_dir / "bathy_meta.json").read_text())
    with rasterio.open(out_dir / "bathy_utm.tif") as src:
        arr = src.read(1).astype("float32")
    arr[arr == NODATA] = np.nan
    t = meta["transform"]
    return arr, Affine(t[0], t[1], t[2], t[3], t[4], t[5]), meta
