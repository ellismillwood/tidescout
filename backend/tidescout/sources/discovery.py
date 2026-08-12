from dataclasses import dataclass
from datetime import timedelta

import httpx

from tidescout.models import Fishery
from tidescout.sources.cache import Cache

MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
USGS_SITE = "https://waterservices.usgs.gov/nwis/site/"
TTL = timedelta(days=30)
# Upstream river gauges sit outside the bay bbox; pad generously for USGS search.
USGS_BBOX_PAD_DEG = 1.0


@dataclass
class StationInfo:
    id: str
    name: str
    lat: float
    lon: float
    kind: str


def _get_json(url: str, params: dict) -> dict:
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east


def _noaa_stations(fishery: Fishery, cache: Cache, type_: str, kind: str) -> list[StationInfo]:
    cached = cache.get_or_fetch(
        "noaa-mdapi", type_, TTL, lambda: _get_json(MDAPI, {"type": type_})
    )
    out = []
    for s in cached.payload.get("stations", []):
        lat, lon = float(s["lat"]), float(s["lng"])
        if _in_bbox(lat, lon, fishery.bbox):
            out.append(StationInfo(str(s["id"]), s["name"], lat, lon, kind))
    return out


def find_tide_stations(fishery: Fishery, cache: Cache) -> list[StationInfo]:
    return _noaa_stations(fishery, cache, "tidepredictions", "tide")


def find_current_stations(fishery: Fishery, cache: Cache) -> list[StationInfo]:
    return _noaa_stations(fishery, cache, "currentpredictions", "current")


def find_usgs_sites(fishery: Fishery, cache: Cache, param: str) -> list[StationInfo]:
    west, south, east, north = fishery.bbox
    bbox = (
        f"{west - USGS_BBOX_PAD_DEG:.4f},{south - 0.1:.4f},"
        f"{east + 0.1:.4f},{north + USGS_BBOX_PAD_DEG:.4f}"
    )
    params = {
        "format": "rdb",
        "bBox": bbox,
        "parameterCd": param,
        "siteStatus": "active",
        "hasDataTypeCd": "iv",
    }

    def fetch() -> str:
        # USGS site service is slow for combined bBox+parameterCd+hasDataTypeCd
        # queries (observed 10-60s in practice); allow more headroom than the
        # mdapi calls, which reliably return in well under a second.
        resp = httpx.get(USGS_SITE, params=params, timeout=60)
        resp.raise_for_status()
        return resp.text

    cached = cache.get_or_fetch("usgs-site", f"{bbox}:{param}", TTL, fetch)
    lines = [ln for ln in cached.payload.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    out = []
    for ln in lines[2:]:  # line after header is the rdb format row (e.g. "5s")
        cols = ln.split("\t")
        try:
            out.append(
                StationInfo(
                    id=cols[idx["site_no"]],
                    name=cols[idx["station_nm"]],
                    lat=float(cols[idx["dec_lat_va"]]),
                    lon=float(cols[idx["dec_long_va"]]),
                    kind="usgs",
                )
            )
        except (KeyError, ValueError, IndexError):
            continue
    return out
