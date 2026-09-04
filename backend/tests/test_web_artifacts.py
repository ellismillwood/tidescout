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
    # ...and in [west, south, east, north] ORDER. At this latitude a swapped
    # west/south pair (-79, 33) still satisfies the range check above, so the
    # range check alone cannot catch an axis swap -- which would put the
    # hillshade underlay in the wrong hemisphere with every number in bounds.
    assert b[0] < b[2] and b[1] < b[3], b
    assert meta["width"] > 0 and meta["height"] > 0


def test_hillshade_makes_nodata_transparent_and_keeps_real_relief_opaque(tmp_path):
    """A single-band hillshade PNG cannot say "nothing here".

    Two independent sources of nodata land in this raster. `render.hillshade`
    writes 0 wherever the DEM is NaN and `build_artifacts` stamps nodata=0 on
    the GeoTIFF; and warping a north-up UTM grid onto web mercator rotates it,
    leaving a margin in the destination rectangle that no source pixel reaches.
    Written single-band, both painted opaque BLACK -- on the real Winyah
    raster 6.7% of pixels, a ~74 px hard dark frame around the fishery that no
    frontend paint property can remove.

    Both cases and their opposite live in one test so the relationship is
    visible: a regression that made the whole layer transparent would pass a
    nodata-only assertion.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    src = tmp_path / "hillshade.tif"
    data = np.zeros((64, 64), dtype="uint8")
    data[32:, :] = 200  # real terrain; the top half stays at the nodata value
    with rasterio.open(
        src, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8",
        crs="EPSG:26917", transform=from_origin(600000, 3700000, 30, 30),
        nodata=0,
    ) as ds:
        ds.write(data, 1)

    png, bounds_json = tmp_path / "hillshade.png", tmp_path / "hillshade.bounds.json"
    webartifacts.hillshade_png(src, png, bounds_json)

    with rasterio.open(png) as out:
        assert out.count == 4, "must be RGBA so nodata can be transparent"
        alpha = out.read(4)
        rgb = out.read([1, 2, 3])

    assert alpha[8, 32] == 0, "a nodata cell must be fully transparent"
    assert alpha[55, 32] == 255, "real terrain must be fully opaque"
    # The warp margin is a DIFFERENT code path from a source nodata value --
    # it is the destination array's fill, never touched by a masked read -- so
    # it needs its own case. Sampled in the bottom-left corner, where every
    # source cell is terrain: a transparent pixel there can only be margin.
    assert alpha[-1, 0] == 0, "the reprojection margin must be transparent too"
    # Relief itself must survive: neutral grey (the depth tint under it owns
    # the colour) at the source's own value, not flattened to black.
    assert tuple(rgb[:, 55, 32]) == (200, 200, 200), rgb[:, 55, 32]


def test_depth_tint_renders_a_colour_ramp_and_makes_nodata_transparent(tmp_path):
    """§9 asks for "bathy hillshade + depth tint + contours". The hillshade
    shipped in PR #12 and contours already exist; the tint had no artifact.

    Two properties, both load-bearing. The output must be RGBA -- a
    single-band PNG cannot express "no data here", so the raster's -9999 fill
    would paint an opaque block over the basemap outside the survey. And
    deeper water must differ from shallow, or the layer conveys nothing.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    src = tmp_path / "bathy.tif"
    data = np.full((64, 64), -9999.0, dtype="float32")
    data[:32, :] = -2.0    # shallow
    data[32:, :] = -14.0   # deep
    data[0, 0] = -9999.0   # a nodata cell inside the covered area
    with rasterio.open(
        src, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
        crs="EPSG:26917", transform=from_origin(600000, 3700000, 30, 30),
        nodata=-9999.0,
    ) as ds:
        ds.write(data, 1)

    png = tmp_path / "depth_tint.png"
    meta = webartifacts.depth_tint_png(src, png)

    assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert meta["width"] > 0 and meta["height"] > 0

    with rasterio.open(png) as out:
        assert out.count == 4, "must be RGBA so nodata can be transparent"
        alpha = out.read(4)
        rgb = out.read([1, 2, 3])

    assert alpha.min() == 0, "nodata must be fully transparent"
    assert alpha.max() == 255, "surveyed water must be fully opaque"
    # Shallow and deep must not paint the same colour...
    top = rgb[:, 8, 8]
    bottom = rgb[:, 55, 8]
    assert tuple(top) != tuple(bottom), (top, bottom)
    # ...and deep must be DARKER than shallow. `np.interp` with a descending
    # xp returns garbage without raising, so "they differ" alone would pass
    # against a scrambled ramp. This pins the direction.
    assert int(bottom.sum()) < int(top.sum()), (top, bottom)


def test_depth_tint_is_registered_as_a_servable_layer():
    """The API's allowlist is a dict-key check, so an artifact absent from it
    cannot be fetched even once it exists on disk."""
    from tidescout.api.layers import LAYERS

    assert LAYERS["depth-tint"] == "depth_tint.png"


def test_depth_tint_treats_land_as_transparent_not_shallow_water(tmp_path):
    """`bathy_utm.tif` is a topobathy DEM, not a water-only grid: cells above
    datum are dry land, not water. Measured on the real raster, 70.2% of its
    non-nodata cells are positive elevation. Painting that as opaque pale
    "shallow water" would draw fake water over dry ground across most of the
    bay -- worse than shipping no depth layer at all, because it actively
    misinforms. Only elevation <= 0 (at or below datum) may render; a cell
    above datum must be as transparent as true nodata.

    All three cases live in one test so the relationship is visible: a fix
    that made EVERYTHING transparent would pass a land-only assertion.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    src = tmp_path / "bathy.tif"
    data = np.full((64, 64), -9999.0, dtype="float32")
    data[:21, :] = -9999.0   # nodata: outside the survey
    data[21:42, :] = 5.0     # land: above datum, inside the survey
    data[42:, :] = -8.0      # water: below datum
    with rasterio.open(
        src, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
        crs="EPSG:26917", transform=from_origin(600000, 3700000, 30, 30),
        nodata=-9999.0,
    ) as ds:
        ds.write(data, 1)

    png = tmp_path / "depth_tint.png"
    webartifacts.depth_tint_png(src, png)

    with rasterio.open(png) as out:
        alpha = out.read(4)
        rgb = out.read([1, 2, 3])

    assert alpha[10, 8] == 0, "true nodata must be transparent"
    assert alpha[30, 8] == 0, "land above datum must be transparent, not shallow water"
    assert alpha[55, 8] == 255, "water at or below datum must be opaque"
    assert tuple(rgb[:, 55, 8]) != (0, 0, 0), "water must get a real ramp colour"
