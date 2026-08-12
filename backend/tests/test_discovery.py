import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache
from tidescout.sources.discovery import (
    find_current_stations,
    find_tide_stations,
    find_usgs_sites,
)

TIDE_FIXTURE = {
    "stations": [
        {"id": "8662245", "name": "South Island Ferry", "lat": 33.2517, "lng": -79.2533},
        {"id": "8661070", "name": "Springmaid Pier", "lat": 33.655, "lng": -78.9183},
    ]
}

CURRENT_FIXTURE = {
    "stations": [
        {"id": "WIN1201", "name": "Winyah Bay Entrance", "lat": 33.21, "lng": -79.17},
        {"id": "CHA0001", "name": "Charleston Harbor", "lat": 32.77, "lng": -79.92},
    ]
}

USGS_RDB = (
    "# comment\n"
    "agency_cd\tsite_no\tstation_nm\tsite_tp_cd\tdec_lat_va\tdec_long_va\n"
    "5s\t15s\t50s\t7s\t16s\t16s\n"
    "USGS\t02131000\tPEE DEE RIVER AT PEEDEE, SC\tST\t34.2043\t-79.5495\n"
    "USGS\t02136000\tBLACK RIVER AT KINGSTREE, SC\tST\t33.6654\t-79.8309\n"
)


@respx.mock
def test_find_tide_stations_filters_to_bbox(tmp_path):
    respx.get(url__regex=r"https://api\.tidesandcurrents\.noaa\.gov/mdapi/.*tidepredictions.*").mock(
        return_value=Response(200, json=TIDE_FIXTURE)
    )
    f = load_fishery("winyah-bay")
    cache = Cache(tmp_path / "c.sqlite")
    stations = find_tide_stations(f, cache)
    assert [s.id for s in stations] == ["8662245"]
    assert stations[0].kind == "tide"


@respx.mock
def test_find_current_stations_filters_to_bbox(tmp_path):
    respx.get(
        url__regex=r"https://api\.tidesandcurrents\.noaa\.gov/mdapi/.*currentpredictions.*"
    ).mock(return_value=Response(200, json=CURRENT_FIXTURE))
    f = load_fishery("winyah-bay")
    cache = Cache(tmp_path / "c.sqlite")
    stations = find_current_stations(f, cache)
    assert [s.id for s in stations] == ["WIN1201"]


@respx.mock
def test_find_usgs_sites_parses_rdb(tmp_path):
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/nwis/site/.*").mock(
        return_value=Response(200, text=USGS_RDB)
    )
    f = load_fishery("winyah-bay")
    cache = Cache(tmp_path / "c.sqlite")
    sites = find_usgs_sites(f, cache, "00060")
    assert {s.id for s in sites} == {"02131000", "02136000"}
    assert all(s.kind == "usgs" for s in sites)
