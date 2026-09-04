"""Browser-ready versions of two artifacts the frontend cannot consume as-is.

Spec §4.1. Both are OFFLINE pipeline steps -- the API only serves their output.

`oyster_reefs.web.geojson` is a DISPLAY artifact ONLY. Feature scoring reads
real oyster geometry (`features.geojson` carries `oyster_area_m2`,
`oyster_density`, `oyster_nearest_m`, derived from the full-precision source).
Nothing in `pipeline/` may read the web version, or ambush scoring silently
degrades to match a rendering optimisation.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from shapely.geometry import mapping, shape

# ~11 cm at 33 degrees N. The source carries FOURTEEN decimal places, which is
# sub-millimetre precision on a shellfish bed, and 34.7 of its 37.6 MB is that.
OYSTER_DECIMALS = 6
# ~2 m. Below the resolution of any zoom level a person fishes from.
OYSTER_TOLERANCE_DEG = 0.00002


def _round_coords(obj, decimals: int):
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_coords(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_coords(x, decimals) for x in obj]
    return obj


def simplify_oysters(
    src: Path,
    dst: Path,
    decimals: int = OYSTER_DECIMALS,
    tolerance_deg: float = OYSTER_TOLERANCE_DEG,
) -> dict:
    """Trim precision, drop unused properties, simplify -- keeping every reef.

    Measured on the real 8451-reef source: 37.6 MB -> 1.9 MB raw, 0.3 MB
    gzipped, with all 8451 retained. Dropping reefs to hit a byte target would
    change what the map claims exists, so the count is preserved and asserted.
    """
    raw = json.loads(Path(src).read_text())
    out = []
    for feat in raw["features"]:
        geom = shape(feat["geometry"])
        simplified = geom.simplify(tolerance_deg, preserve_topology=True)
        # A reef that simplifies to nothing keeps its ORIGINAL geometry rather
        # than vanishing from the map.
        chosen = geom if simplified.is_empty else simplified
        out.append({
            "type": "Feature",
            "properties": {},
            "geometry": _round_coords(mapping(chosen), decimals),
        })
    payload = {"type": "FeatureCollection", "features": out}
    text = json.dumps(payload, separators=(",", ":"))
    Path(dst).write_text(text)
    return {"features": len(out), "bytes": len(text.encode())}


def hillshade_png(src: Path, png: Path, bounds_json: Path) -> dict:
    """Reproject a UTM GeoTIFF hillshade to web mercator and write an RGBA PNG.

    MapLibre needs raster tiles or an image plus bounds; it can consume neither
    a GeoTIFF nor EPSG:26917. The bounds sidecar is written in WGS84 degrees
    because that is what an image overlay takes.

    RGBA rather than single-band, for exactly the reason `depth_tint_png` below
    is. Two independent sources of "nothing here" land in this raster: the
    source declares 0 as its nodata (`render.hillshade` writes 0 wherever the
    DEM is NaN, and `build_artifacts` stamps nodata=0 on the GeoTIFF), and
    warping a north-up UTM grid onto web mercator rotates it slightly, so the
    destination rectangle always carries a margin no source pixel reaches. On
    the real Winyah raster that was 6.7% of all pixels, a ~74 px border. A
    single-band PNG has no way to say "nothing here", so every one of those
    pixels painted opaque BLACK -- a hard dark frame around the fishery. No
    frontend paint property can remove it either: lowering the layer's white
    point (`raster-brightness-max`) makes already-black pixels RELATIVELY more
    prominent, not less. Alpha 0 on nodata is what lets the relief sit inside
    the survey's real outline.
    """
    with rasterio.open(src) as ds:
        transform, width, height = calculate_default_transform(
            ds.crs, "EPSG:3857", ds.width, ds.height, *ds.bounds
        )
        # float32/NaN rather than the uint8 the source is, purely so nodata
        # has somewhere to live: in uint8 the fill value would be an ordinary
        # shade and indistinguishable from real terrain that happens to match.
        shade = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(ds, 1),
            destination=shade,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            src_nodata=ds.nodata,
            dst_nodata=np.nan,
        )
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)

    lit = np.isfinite(shade)
    rgba = np.zeros((4, height, width), dtype="uint8")
    grey = np.zeros((height, width), dtype="uint8")
    grey[lit] = shade[lit].astype("uint8")
    # Neutral grey: relief is a luminance signal, and the depth tint under it
    # owns the colour. Alpha carries the whole "is there terrain here" claim.
    for band in range(3):
        rgba[band] = grey
    rgba[3][lit] = 255

    with rasterio.open(
        png, "w", driver="PNG", height=height, width=width, count=4, dtype="uint8"
    ) as out:
        out.write(rgba)

    Path(bounds_json).write_text(
        json.dumps({"bounds": [west, south, east, north], "crs": "EPSG:4326"})
    )
    return {"width": width, "height": height, "bounds": [west, south, east, north]}


# Depth breakpoints in metres (negative = below datum) and their RGB colours,
# shallow to deep. Chosen to read as water rather than as a heatmap: the
# hillshade above supplies relief, so this layer only has to say "how deep".
DEPTH_STOPS_M = (-0.5, -2.0, -5.0, -10.0, -20.0)
DEPTH_COLOURS = (
    (198, 232, 240),
    (140, 200, 224),
    (86, 158, 204),
    (44, 108, 173),
    (18, 62, 128),
)


def depth_tint_png(src: Path, png: Path) -> dict:
    """Colour-ramp a bathymetry GeoTIFF into a web-mercator RGBA PNG.

    RGBA, not single-band: the source fills unsurveyed cells with its nodata
    value, and a PNG without an alpha channel has no way to say "nothing
    here" -- it would paint an opaque rectangle over the basemap wherever the
    survey stops. Alpha 0 on nodata is what lets the tint sit inside the
    survey's real outline.

    Shares `hillshade_png`'s bounds sidecar rather than writing its own: both
    are warps of the same grid, so a second bounds file would be a second
    thing to keep in sync.
    """
    with rasterio.open(src) as ds:
        transform, width, height = calculate_default_transform(
            ds.crs, "EPSG:3857", ds.width, ds.height, *ds.bounds
        )
        depth = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(ds, 1),
            destination=depth,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            src_nodata=ds.nodata,
            dst_nodata=np.nan,
        )

    stops = np.array(DEPTH_STOPS_M, dtype="float32")
    cols = np.array(DEPTH_COLOURS, dtype="float32")
    flat = depth.ravel()
    # `bathy_utm.tif` is a topobathy DEM, not a water-only grid: on the real
    # source, 70.2% of non-nodata cells are positive elevation -- dry land
    # and marsh, not water. Only cells AT OR BELOW datum (<= 0) are water;
    # painting land as opaque pale "shallow water" would draw fake water
    # over dry ground across most of the survey, actively misinforming
    # rather than just being blank. Land gets alpha 0 alongside true nodata.
    water = np.isfinite(flat) & (flat <= 0)
    # `np.interp` REQUIRES ascending xp and returns garbage silently -- no
    # error -- when given a descending one. DEPTH_STOPS_M is negative and
    # descending (-0.5 .. -20), so negating it gives 0.5 .. 20, already
    # ascending. Do NOT also reverse: negate-and-reverse yields
    # [20, 10, 5, 2, 0.5], which is descending again and paints noise.
    xs = -stops
    rgba = np.zeros((flat.size, 4), dtype="uint8")
    for band in range(3):
        rgba[water, band] = np.interp(-flat[water], xs, cols[:, band]).astype("uint8")
    rgba[water, 3] = 255

    out = rgba.reshape(height, width, 4).transpose(2, 0, 1)
    with rasterio.open(
        png, "w", driver="PNG", height=height, width=width, count=4, dtype="uint8"
    ) as dst:
        dst.write(out)
    return {"width": width, "height": height}
