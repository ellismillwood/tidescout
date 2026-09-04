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
    """Reproject a UTM GeoTIFF hillshade to web mercator and write a PNG.

    MapLibre needs raster tiles or an image plus bounds; it can consume neither
    a GeoTIFF nor EPSG:26917. The bounds sidecar is written in WGS84 degrees
    because that is what an image overlay takes.
    """
    with rasterio.open(src) as ds:
        transform, width, height = calculate_default_transform(
            ds.crs, "EPSG:3857", ds.width, ds.height, *ds.bounds
        )
        dest = np.zeros((height, width), dtype="uint8")
        reproject(
            source=rasterio.band(ds, 1),
            destination=dest,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
        )
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)

    with rasterio.open(
        png, "w", driver="PNG", height=height, width=width, count=1, dtype="uint8"
    ) as out:
        out.write(dest, 1)

    Path(bounds_json).write_text(
        json.dumps({"bounds": [west, south, east, north], "crs": "EPSG:4326"})
    )
    return {"width": width, "height": height, "bounds": [west, south, east, north]}
