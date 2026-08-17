"""Ambush-feature inventory: run all engine.detect detectors against the
per-fishery UTM analysis raster, reproject geometries to EPSG:4326, and write
a single `features.geojson` FeatureCollection."""

import json
from pathlib import Path

from rasterio.warp import transform as warp_transform
from shapely.geometry import LineString, Point, Polygon, mapping

from tidescout.engine import detect
from tidescout.engine.terrain import slope_deg
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy


def _to4326(geom, epsg: int):
    src = f"EPSG:{epsg}"

    def tx(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        lons, lats = warp_transform(src, "EPSG:4326", xs, ys)
        return list(zip(lons, lats, strict=True))

    if isinstance(geom, Point):
        (lonlat,) = tx([(geom.x, geom.y)])
        return Point(lonlat)
    if isinstance(geom, LineString):
        return LineString(tx(list(geom.coords)))
    if isinstance(geom, Polygon):
        return Polygon(
            tx(list(geom.exterior.coords)),
            [tx(list(r.coords)) for r in geom.interiors],
        )
    raise TypeError(f"unsupported geometry: {geom.geom_type}")


def build_features(slug: str, fishery: Fishery) -> Path:
    z, transform, meta = read_bathy(slug)
    cell = fishery.bathymetry.cell_m
    epsg = fishery.bathymetry.epsg
    t = fishery.features
    wet_level_m = fishery.bathymetry.static_wet_level_m
    slope = slope_deg(z, cell)

    def lonlat_to_grid(lons, lats):
        return warp_transform("EPSG:4326", f"EPSG:{epsg}", lons, lats)

    feats = (
        detect.detect_dropoffs(z, slope, t, transform, wet_level_m)
        + detect.detect_holes(z, t, cell, transform, wet_level_m)
        + detect.detect_flats(z, slope, t, transform)
        + detect.detect_creek_mouths(z, t, cell, transform, wet_level_m)
        + detect.detect_bars(z, t, cell, transform)
        + detect.seed_jetties(fishery, lonlat_to_grid)
    )
    out = []
    for f in feats:
        props = {"type": f.type}
        for k, v in f.attrs.items():
            props[k] = round(v, 2) if isinstance(v, float) else v
        out.append(
            {
                "type": "Feature",
                # Hash of type + quantised UTM centroid, computed BEFORE
                # reprojection: _to4326 would put the centroid in degrees,
                # where a 1 m quantum is ~100 km.
                "id": detect.feature_key(f),
                "properties": props,
                "geometry": mapping(_to4326(f.geometry, epsg)),
            }
        )
    path = fishery_data_dir(slug) / "features.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": out}))
    return path


def load_features(slug: str) -> dict:
    return json.loads((fishery_data_dir(slug) / "features.geojson").read_text())
