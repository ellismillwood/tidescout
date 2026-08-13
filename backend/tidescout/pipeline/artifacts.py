"""Build human-facing map artifacts (hillshade GeoTIFF, depth quicklook PNG,
contour GeoJSON) from the per-fishery UTM analysis raster."""

import json
from pathlib import Path

import rasterio

from tidescout.engine import render
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy
from tidescout.pipeline.derivatives import _write


def build_artifacts(slug: str, fishery: Fishery) -> dict[str, Path]:
    z, transform, meta = read_bathy(slug)
    d = fishery_data_dir(slug)
    hs = render.hillshade(z, fishery.bathymetry.cell_m)
    _write(d / "hillshade.tif", hs, meta, "uint8", 0)

    rgba = render.depth_rgba(z, fishery.features.deep_min_m, fishery.bathymetry.land_elev_m)
    shade = (hs.astype("float32") / 255.0) * 0.6 + 0.4
    for band in range(3):
        rgba[..., band] = (rgba[..., band] * shade).astype("uint8")
    ql = d / "quicklook.png"
    with rasterio.open(
        ql, "w", driver="PNG", height=rgba.shape[0], width=rgba.shape[1], count=4, dtype="uint8"
    ) as dst:
        for band in range(4):
            dst.write(rgba[..., band], band + 1)

    lines = render.contour_lines(
        z, transform, fishery.bathymetry.epsg, fishery.bathymetry.contour_depths_m
    )
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"depth_m": li["depth_m"]},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in li["coords"]],
                },
            }
            for li in lines
        ],
    }
    (d / "contours.geojson").write_text(json.dumps(fc))
    return {"hillshade": d / "hillshade.tif", "quicklook": ql, "contours": d / "contours.geojson"}
