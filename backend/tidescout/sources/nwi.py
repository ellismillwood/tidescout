"""USFWS National Wetlands Inventory -- a land/water source independent of CUDEM.

We currently infer land/water purely by thresholding CUDEM bathymetry at
`wet_level_m`. That is a weak proxy: CUDEM in vegetated marsh is biased (lidar
returns off Spartina canopy), so marsh islands visible on satellite imagery can
get modelled as open water. NWI is public, unauthenticated, and classifies
wetlands from aerial-photo interpretation against the Cowardin classification
system -- a second, decorrelated opinion on where marsh actually is.

Verified live 2026-08-14 against the ArcGIS REST service backing NWI:
  https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0
  maxRecordCount 1000; WETLAND_TYPE='Estuarine and Marine Wetland' count for the
  Winyah bbox = 1070.

This module only fetches and caches the raw wetlands GeoJSON to
data/<slug>/nwi_wetlands.geojson. It does NOT feed into mesh/domain logic --
using it to correct the land/water boundary is a follow-on decision, made
after a visual comparison against the CUDEM-derived mask.
"""

import json
from pathlib import Path

import httpx

from tidescout.errors import SourceUnavailable
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.sources.cache import Cache

LAYER_URL = (
    "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/"
    "Wetlands/MapServer/0/query"
)
# The layer joins Wetlands to NWI_Wetland_Codes on ATTRIBUTE, so both the WHERE
# clause and outFields must be table-qualified: an unqualified "ATTRIBUTE" is
# ambiguous between the two tables, and an unqualified WHERE clause fails
# outright. Verified live -- both failure modes return HTTP 200 with a JSON
# {"error": {"code": 400, ...}} body, not an HTTP error status, so fetch_page
# checks the payload for an "error" key rather than relying on raise_for_status.
WETLAND_TYPE_FIELD = "Wetlands.WETLAND_TYPE"
ATTRIBUTE_FIELD = "Wetlands.ATTRIBUTE"
# E2EM = estuarine emergent marsh (Cowardin code, e.g. E2EM1P) -- exactly what
# distinguishes a marsh island from open water.
DEFAULT_WETLAND_TYPE = "Estuarine and Marine Wetland"
PAGE_SIZE = 1000  # matches the live service's maxRecordCount


def _page_params(
    bbox: tuple[float, float, float, float], wetland_type: str, offset: int, page_size: int
) -> dict:
    west, south, east, north = bbox
    return {
        "where": f"{WETLAND_TYPE_FIELD}='{wetland_type}'",
        "outFields": f"{WETLAND_TYPE_FIELD},{ATTRIBUTE_FIELD}",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": page_size,
    }


def _normalize_properties(props: dict) -> dict:
    """Strip the service's `Wetlands.` table-qualification prefix so callers
    see plain WETLAND_TYPE/ATTRIBUTE keys instead of the join's internal names."""
    return {k.rsplit(".", 1)[-1]: v for k, v in props.items()}


def fetch_page(
    bbox: tuple[float, float, float, float],
    offset: int,
    cache: Cache,
    wetland_type: str = DEFAULT_WETLAND_TYPE,
    page_size: int = PAGE_SIZE,
) -> dict:
    """One page of the NWI wetlands query, cached by bbox/type/offset/page_size."""
    params = _page_params(bbox, wetland_type, offset, page_size)

    def fetch() -> dict:
        resp = httpx.get(LAYER_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise SourceUnavailable("nwi", str(payload["error"]))
        return payload

    key = f"{','.join(str(v) for v in bbox)}:{wetland_type}:{offset}:{page_size}"
    # Published NWI polygons do not change on a timescale this app cares about
    # (same reasoning as usgs.fetch_daily's immutable-once-published TTL=None),
    # so a page is cached forever once fetched.
    cached = cache.get_or_fetch("nwi", key, None, fetch)
    return cached.payload


def fetch_wetlands(
    fishery: Fishery,
    cache: Cache,
    wetland_type: str = DEFAULT_WETLAND_TYPE,
    page_size: int = PAGE_SIZE,
) -> dict:
    """Page the NWI wetlands query across `fishery.bbox`, merge into one GeoJSON
    FeatureCollection, and write it to data/<slug>/nwi_wetlands.geojson.

    Pagination follows the Esri REST convention: keep requesting resultOffset +
    len(features) while the previous page reports exceededTransferLimit=True.
    The final (possibly empty) page returns fewer than page_size features and
    exceededTransferLimit False/absent, which ends the loop -- relying on that
    flag rather than `len(features) == page_size` correctly stops on a page
    that happens to be exactly full without truly having more behind it.
    """
    features: list[dict] = []
    offset = 0
    while True:
        page = fetch_page(fishery.bbox, offset, cache, wetland_type, page_size)
        page_features = page.get("features", [])
        for f in page_features:
            features.append(
                {
                    "type": "Feature",
                    "geometry": f["geometry"],
                    "properties": _normalize_properties(f.get("properties", {})),
                }
            )
        if not page_features or not page.get("exceededTransferLimit"):
            break
        offset += len(page_features)
    fc = {"type": "FeatureCollection", "features": features}
    write_wetlands(fishery.slug, fc)
    return fc


def wetlands_path(slug: str) -> Path:
    return fishery_data_dir(slug) / "nwi_wetlands.geojson"


def write_wetlands(slug: str, fc: dict) -> Path:
    path = wetlands_path(slug)
    path.write_text(json.dumps(fc))
    return path
