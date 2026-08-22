import json

import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources import nwi
from tidescout.sources.cache import Cache


def _feature(attribute: str) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-79.2, 33.3], [-79.2, 33.31],
                                                           [-79.19, 33.31], [-79.2, 33.3]]]},
        "properties": {"Wetlands.WETLAND_TYPE": "Estuarine and Marine Wetland",
                        "Wetlands.ATTRIBUTE": attribute},
    }


def _fc(features: list[dict], exceeded: bool) -> dict:
    return {
        "type": "FeatureCollection",
        "features": features,
        "exceededTransferLimit": exceeded,
    }


@respx.mock
def test_fetch_wetlands_pages_until_transfer_limit_not_exceeded(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    fishery = load_fishery("winyah-bay")

    page1 = _fc([_feature("E2EM1P"), _feature("E2EM1N")], exceeded=True)
    page2 = _fc([_feature("E1UBL")], exceeded=False)
    route = respx.get(url__regex=r"https://fwspublicservices\.wim\.usgs\.gov/.*").mock(
        side_effect=[Response(200, json=page1), Response(200, json=page2)]
    )

    fc = nwi.fetch_wetlands(fishery, Cache(tmp_path / "c.sqlite"), page_size=2)

    assert route.call_count == 2
    # first call's resultOffset=0, second call picks up at len(page1.features)=2
    first_params = dict(route.calls[0].request.url.params)
    second_params = dict(route.calls[1].request.url.params)
    assert first_params["resultOffset"] == "0"
    assert second_params["resultOffset"] == "2"

    assert len(fc["features"]) == 3
    attrs = {f["properties"]["ATTRIBUTE"] for f in fc["features"]}
    assert attrs == {"E2EM1P", "E2EM1N", "E1UBL"}
    # prefix stripped, not merely renamed alongside the original
    assert all("Wetlands.ATTRIBUTE" not in f["properties"] for f in fc["features"])
    assert all(f["properties"]["WETLAND_TYPE"] == "Estuarine and Marine Wetland"
               for f in fc["features"])

    written = json.loads(nwi.wetlands_path("winyah-bay").read_text())
    assert written == fc


@respx.mock
def test_fetch_wetlands_empty_result_writes_empty_feature_collection(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    fishery = load_fishery("winyah-bay")

    respx.get(url__regex=r"https://fwspublicservices\.wim\.usgs\.gov/.*").mock(
        return_value=Response(200, json=_fc([], exceeded=False))
    )

    fc = nwi.fetch_wetlands(fishery, Cache(tmp_path / "c.sqlite"))

    assert fc == {"type": "FeatureCollection", "features": []}
    written = json.loads(nwi.wetlands_path("winyah-bay").read_text())
    assert written == fc


@respx.mock
def test_fetch_wetlands_single_page_does_not_over_request(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    fishery = load_fishery("winyah-bay")

    route = respx.get(url__regex=r"https://fwspublicservices\.wim\.usgs\.gov/.*").mock(
        return_value=Response(200, json=_fc([_feature("E2EM1P")], exceeded=False))
    )

    fc = nwi.fetch_wetlands(fishery, Cache(tmp_path / "c.sqlite"))

    assert route.call_count == 1
    assert len(fc["features"]) == 1


@respx.mock
def test_fetch_page_uses_table_qualified_where_and_fields(tmp_path):
    fishery = load_fishery("winyah-bay")
    route = respx.get(url__regex=r"https://fwspublicservices\.wim\.usgs\.gov/.*").mock(
        return_value=Response(200, json=_fc([], exceeded=False))
    )
    nwi.fetch_page(fishery.bbox, 0, Cache(tmp_path / "c.sqlite"))
    params = dict(route.calls[0].request.url.params)
    assert params["where"] == "Wetlands.WETLAND_TYPE='Estuarine and Marine Wetland'"
    assert params["outFields"] == "Wetlands.WETLAND_TYPE,Wetlands.ATTRIBUTE"
    assert params["f"] == "geojson"
    assert params["outSR"] == "4326"


@respx.mock
def test_fetch_page_raises_on_esri_error_payload_despite_http_200(tmp_path):
    from tidescout.errors import SourceUnavailable

    fishery = load_fishery("winyah-bay")
    respx.get(url__regex=r"https://fwspublicservices\.wim\.usgs\.gov/.*").mock(
        return_value=Response(200, json={"error": {"code": 400, "message": "Failed"}})
    )
    try:
        nwi.fetch_page(fishery.bbox, 0, Cache(tmp_path / "c.sqlite"))
    except SourceUnavailable as exc:
        assert exc.source == "nwi"
    else:
        raise AssertionError("expected SourceUnavailable")
