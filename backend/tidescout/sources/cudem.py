import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from tidescout.models import Fishery
from tidescout.paths import FISHERIES_DIR

TILE_SPAN_DEG = 0.25
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
# Extension widened to (tif|nc): the brief's assumed public S3/GeoTIFF bucket for
# CUDEM does not exist (verified live 2026-08-13 -- see thredds_* below). The real
# distribution is NCEI THREDDS, serving NetCDF (.nc), not GeoTIFF. Kept both so this
# regex still matches the brief's original .tif fixtures/tests unchanged.
NAME_RE = re.compile(r"ncei19_n(\d+)x(\d+)_w(\d+)x(\d+).*\.(?:tif|nc)$")

# --- Real live source (NCEI THREDDS), used by `discover` -------------------------
# No public S3 bucket exists for CUDEM ninth arc-second tiles: neither the brief's
# guessed `noaa-nos-cudem-pds` nor several other plausible NODD-style names
# (noaa-ncei-cudem-pds, noaa-nodd-cudem-pds, cudem-pds, noaa-cudem-pds) resolve, and
# a full-text search of the AWS Open Data Registry's dataset index
# (github.com/awslabs/open-data-registry) turns up no "cudem" entry at all. NCEI
# instead serves this data exclusively via THREDDS as NetCDF (see task-4-report.md
# for the full resolution-ladder log).
THREDDS_CATALOG_URL = "https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_19as/catalog.html"
THREDDS_DODS_BASE = "https://www.ngdc.noaa.gov/thredds/dodsC/tiles/tiled_19as/"
THREDDS_DATASET_RE = re.compile(r'dataset=tilesDatasetScan/tiled_19as/([\w.\-]+\.nc)"')


@dataclass
class TileRef:
    key: str
    url: str


def list_s3_keys(bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        params: dict[str, str] = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        resp = httpx.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        keys += [el.text for el in root.iter(f"{S3_NS}Key") if el.text]
        truncated = (root.findtext(f"{S3_NS}IsTruncated") or "false") == "true"
        if not truncated:
            return keys
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            return keys


def parse_ninth_arc_name(key: str) -> tuple[float, float] | None:
    m = NAME_RE.search(key)
    if not m:
        return None
    lat = int(m.group(1)) + int(m.group(2)) / 100.0
    lon = -(int(m.group(3)) + int(m.group(4)) / 100.0)
    return lat, lon


def _tile_intersects_bbox(
    tile_n: float, tile_w: float, bbox: tuple[float, float, float, float]
) -> bool:
    """0.25 deg tile anchored at its NW corner (tile_n, tile_w), extending south+east.

    Corner convention verified live 2026-08-13 against rasterio-reported bounds for
    ncei19_n33x50_w079x50_2019v1.nc: real bounds top=33.500185/left=-79.500185,
    bottom=33.249815/right=-79.249815 -- i.e. north/west edges at the parsed
    (33.50, -79.50) corner, spanning south+east by ~0.25 deg (the ~0.0002 deg
    excess is a small inter-tile overlap buffer, not a parsing error). No fix
    needed; convention confirmed as originally specified.
    """
    west, south, east, north = bbox
    tile_s, tile_e = tile_n - TILE_SPAN_DEG, tile_w + TILE_SPAN_DEG
    return tile_w < east and tile_e > west and tile_s < north and tile_n > south


def candidate_tiles(fishery: Fishery, keys: list[str], bucket: str) -> list[TileRef]:
    out = []
    for key in keys:
        parsed = parse_ninth_arc_name(key)
        if parsed is None:
            continue
        tile_n, tile_w = parsed
        if _tile_intersects_bbox(tile_n, tile_w, fishery.bbox):
            out.append(TileRef(key, f"https://{bucket}.s3.amazonaws.com/{key}"))
    return out


def list_thredds_keys(catalog_url: str = THREDDS_CATALOG_URL) -> list[str]:
    """List ninth arc-second CUDEM tile filenames from the NCEI THREDDS DatasetScan
    catalog (the real live source; see module docstring comment above)."""
    resp = httpx.get(catalog_url, timeout=60)
    resp.raise_for_status()
    return sorted(set(THREDDS_DATASET_RE.findall(resp.text)))


def thredds_tile_url(key: str) -> str:
    """Build a GDAL netCDF/OPeNDAP URL for a THREDDS-hosted CUDEM tile.

    Plain https/vsicurl access to the .nc file fails on this platform: GDAL's
    netCDF driver requires Linux userfaultfd for remote/vsicurl reads (verified
    live: RasterioIOError). Forcing the HDF5 fallback driver (GDAL_SKIP=netCDF)
    opens the file but loses all CF georeferencing (crs=None, bounds=raw pixel
    indices). Routing through THREDDS's OpenDAP endpoint with the netCDF driver's
    `NETCDF:"..."` prefix gives GDAL a DAP2 metadata/data channel it can read
    remotely without downloading the full grid (verified live: <1s open, real
    bounds/crs, for an ~83MB source file) and CF georeferencing intact.
    """
    return f'NETCDF:"{THREDDS_DODS_BASE}{key}"'


def thredds_candidate_keys(fishery: Fishery, keys: list[str]) -> list[str]:
    """Filenames (not TileRefs -- THREDDS keys carry no bucket) whose tile
    intersects the fishery bbox. Pairs with thredds_tile_url() to build TileRefs."""
    return [
        key
        for key in keys
        if (parsed := parse_ninth_arc_name(key)) is not None
        and _tile_intersects_bbox(parsed[0], parsed[1], fishery.bbox)
    ]


def manifest_path(slug: str) -> Path:
    return FISHERIES_DIR / f"{slug}.tiles.yaml"


def load_manifest(slug: str) -> list[dict]:
    p = manifest_path(slug)
    if not p.exists():
        return []
    return yaml.safe_load(p.read_text()) or []


def write_manifest(slug: str, entries: list[dict]) -> Path:
    p = manifest_path(slug)
    p.write_text(yaml.safe_dump(entries, sort_keys=False))
    return p
