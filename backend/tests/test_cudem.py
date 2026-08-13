import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cudem import (
    candidate_tiles,
    list_s3_keys,
    list_thredds_keys,
    parse_ninth_arc_name,
    thredds_candidate_keys,
    thredds_tile_url,
)

S3_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>CUDEM_9as/ncei19_n33x50_w079x50_2018v1.tif</Key></Contents>
  <Contents><Key>CUDEM_9as/ncei19_n33x25_w079x25_2018v1.tif</Key></Contents>
  <Contents><Key>CUDEM_9as/ncei19_n41x00_w074x00_2018v1.tif</Key></Contents>
  <Contents><Key>CUDEM_9as/readme.txt</Key></Contents>
</ListBucketResult>"""

# Trimmed real markup shape from the live NCEI THREDDS DatasetScan catalog
# (https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_19as/catalog.html,
# captured 2026-08-13) -- the actual live source for CUDEM ninth arc-second
# tiles. See cudem.py module comment and task-4-report.md for why: no public
# S3/GeoTIFF bucket exists for this dataset.
THREDDS_CATALOG_PAGE = """<html><body><table>
<tr><td><a href="catalog.html?dataset=tilesDatasetScan/tiled_19as/ncei19_n33x50_w079x50_2019v1.nc">
<code>ncei19_n33x50_w079x50_2019v1.nc</code></a></td></tr>
<tr><td><a href="catalog.html?dataset=tilesDatasetScan/tiled_19as/ncei19_n33x25_w079x25_2019v1.nc">
<code>ncei19_n33x25_w079x25_2019v1.nc</code></a></td></tr>
<tr><td><a href="catalog.html?dataset=tilesDatasetScan/tiled_19as/ncei19_n41x00_w074x00_2019v1.nc">
<code>ncei19_n41x00_w074x00_2019v1.nc</code></a></td></tr>
</table></body></html>"""


def test_parse_ninth_arc_name():
    assert parse_ninth_arc_name("x/ncei19_n33x50_w079x25_2018v1.tif") == (33.50, -79.25)
    assert parse_ninth_arc_name("x/ncei19_n41x00_w074x00_v3.tif") == (41.00, -74.00)
    assert parse_ninth_arc_name("x/readme.txt") is None


@respx.mock
def test_list_and_filter_candidates():
    respx.get(url__regex=r"https://testbucket\.s3\.amazonaws\.com/.*").mock(
        return_value=Response(200, text=S3_PAGE)
    )
    keys = list_s3_keys("testbucket", "CUDEM_9as/")
    assert len(keys) == 4
    f = load_fishery("winyah-bay")
    tiles = candidate_tiles(f, keys, "testbucket")
    got = {t.key.rsplit("/", 1)[-1] for t in tiles}
    # n33x50/w079x50 spans lat 33.25..33.50, lon -79.50..-79.25 -> intersects bbox
    # n33x25/w079x25 spans lat 33.00..33.25, lon -79.25..-79.00 -> intersects (south edge)
    # n41/w074 does not; readme.txt is not a tif
    assert got == {"ncei19_n33x50_w079x50_2018v1.tif", "ncei19_n33x25_w079x25_2018v1.tif"}
    assert all(t.url.startswith("https://testbucket.s3.amazonaws.com/") for t in tiles)


def test_parse_ninth_arc_name_accepts_nc_extension():
    # Live-verified 2026-08-13: CUDEM has no public S3/GeoTIFF bucket (see
    # cudem.py module comment). The real NCEI THREDDS source serves NetCDF
    # (.nc), so NAME_RE was widened from .tif-only to (tif|nc). This is the
    # actual observed tile covering Winyah Bay's NW corner; its rasterio-read
    # bounds confirmed the NW-corner-extends-south+east convention is correct
    # as originally written (see _tile_intersects_bbox docstring) -- only the
    # extension assumption was wrong, not the corner math.
    assert parse_ninth_arc_name("ncei19_n33x50_w079x50_2019v1.nc") == (33.50, -79.50)


@respx.mock
def test_list_thredds_keys_parses_catalog_html():
    respx.get("https://example.test/thredds/catalog/tiles/tiled_19as/catalog.html").mock(
        return_value=Response(200, text=THREDDS_CATALOG_PAGE)
    )
    keys = list_thredds_keys("https://example.test/thredds/catalog/tiles/tiled_19as/catalog.html")
    # sorted + deduped, unlike list_s3_keys' insertion order
    assert keys == [
        "ncei19_n33x25_w079x25_2019v1.nc",
        "ncei19_n33x50_w079x50_2019v1.nc",
        "ncei19_n41x00_w074x00_2019v1.nc",
    ]


def test_thredds_tile_url_builds_opendap_netcdf_url():
    # NETCDF:"..." driver prefix over the dodsC (OpenDAP) endpoint -- plain
    # https/vsicurl access to the raw .nc fails under GDAL's netCDF driver on
    # this platform (needs Linux userfaultfd), verified live 2026-08-13.
    url = thredds_tile_url("ncei19_n33x50_w079x50_2019v1.nc")
    assert url == (
        'NETCDF:"https://www.ngdc.noaa.gov/thredds/dodsC/tiles/tiled_19as/'
        'ncei19_n33x50_w079x50_2019v1.nc"'
    )


def test_thredds_candidate_keys_filters_to_bbox():
    keys = [
        "ncei19_n33x50_w079x50_2019v1.nc",
        "ncei19_n33x25_w079x25_2019v1.nc",
        "ncei19_n41x00_w074x00_2019v1.nc",
        "readme.txt",
    ]
    f = load_fishery("winyah-bay")
    got = thredds_candidate_keys(f, keys)
    assert got == ["ncei19_n33x50_w079x50_2019v1.nc", "ncei19_n33x25_w079x25_2019v1.nc"]
