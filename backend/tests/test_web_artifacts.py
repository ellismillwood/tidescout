"""Browser-ready versions of two artifacts the frontend cannot use as-is."""

import json

import numpy as np
import pytest

from tidescout.pipeline import webartifacts


def _fc(coords):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"objectid": 1, "photo_year": 2019, "calcgeo_ac": 0.4},
             "geometry": {"type": "Polygon", "coordinates": [coords]}}
        ],
    }


def test_simplify_keeps_every_reef(tmp_path):
    """8451 reefs at 37.6 MB is a rendering problem, not a data problem --
    dropping reefs to hit a byte target would change what the map claims
    exists. Simplification must preserve the COUNT."""
    src, dst = tmp_path / "in.geojson", tmp_path / "out.geojson"
    ring = [[-79.0 + i * 1e-6, 33.5 + i * 1e-6] for i in range(200)] + [[-79.0, 33.5]]
    fc = _fc(ring)
    fc["features"] = fc["features"] * 50
    src.write_text(json.dumps(fc))

    stats = webartifacts.simplify_oysters(src, dst)
    out = json.loads(dst.read_text())
    assert stats["features"] == 50
    assert len(out["features"]) == 50


def test_simplify_trims_coordinate_precision_and_drops_unused_properties(tmp_path):
    """34.7 of 37.6 MB was geometry at FOURTEEN decimal places -- roughly
    nanometre precision on a shellfish bed. 6 dp is ~11 cm at this latitude.
    The only properties present are objectid/photo_year/calcgeo_ac, none of
    which the UI reads."""
    src, dst = tmp_path / "in.geojson", tmp_path / "out.geojson"
    ring = [[-79.05985689076714, 33.51795163424045],
            [-79.05986496225968, 33.51795176070612],
            [-79.05987000000001, 33.51796000000001],
            [-79.05985689076714, 33.51795163424045]]
    src.write_text(json.dumps(_fc(ring)))

    webartifacts.simplify_oysters(src, dst)
    out = json.loads(dst.read_text())
    feat = out["features"][0]
    assert feat["properties"] == {}
    for x, y in feat["geometry"]["coordinates"][0]:
        assert len(str(x).split(".")[-1]) <= 6, x
        assert len(str(y).split(".")[-1]) <= 6, y


def test_simplified_output_is_materially_smaller(tmp_path):
    src, dst = tmp_path / "in.geojson", tmp_path / "out.geojson"
    ring = [[-79.0 + i * 1e-7, 33.5 + i * 1e-7] for i in range(500)] + [[-79.0, 33.5]]
    src.write_text(json.dumps(_fc(ring)))
    stats = webartifacts.simplify_oysters(src, dst)
    assert stats["bytes"] < src.stat().st_size / 2


def test_hillshade_is_reprojected_to_web_mercator_and_written_as_png(tmp_path):
    """`hillshade.tif` is a single-band GeoTIFF in EPSG:26917. A browser can
    render neither the format nor the projection (spec §4.1)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    src = tmp_path / "hillshade.tif"
    data = np.linspace(0, 255, 64 * 64, dtype="uint8").reshape(64, 64)
    with rasterio.open(
        src, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8",
        crs="EPSG:26917", transform=from_origin(600000, 3700000, 30, 30),
    ) as dst_ds:
        dst_ds.write(data, 1)

    png, bounds_json = tmp_path / "hillshade.png", tmp_path / "hillshade.bounds.json"
    meta = webartifacts.hillshade_png(src, png, bounds_json)

    assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    b = json.loads(bounds_json.read_text())["bounds"]
    assert len(b) == 4
    # WGS84 degrees, not UTM metres -- the whole point of the conversion.
    assert -180 <= b[0] <= 180 and -90 <= b[1] <= 90, b
    assert meta["width"] > 0 and meta["height"] > 0
