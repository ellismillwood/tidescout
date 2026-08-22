"""Ambush-feature inventory: run all engine.detect detectors against the
per-fishery UTM analysis raster, reproject geometries to EPSG:4326, and write
a single `features.geojson` FeatureCollection."""

import json
import math
from pathlib import Path

from rasterio.warp import transform as warp_transform
from shapely.geometry import LineString, Point, Polygon, mapping

from tidescout.engine import detect
from tidescout.engine.terrain import slope_deg
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy
from tidescout.pipeline.oysters import (
    load_reefs_utm,
    nearest_reef_m,
    reef_area_m2_within,
    reef_density_within,
)

# Float properties are rounded before they are written, to keep the GeoJSON
# compact and to stop float64 noise in the last digits from churning the file
# between rebuilds. Two decimals suits every property measured in metres or
# square metres; `oyster_density` is the exception and needs its own entry.
#
# `oyster_density` is a DIMENSIONLESS FRACTION of the buffered search area
# (see `oysters.reef_density_within`), with a real range of roughly
# 0.001-0.14 on the shipped inventory. At 2 dp, 25 of the 146 reef-carrying
# features (17%) rounded to exactly 0.0 -- indistinguishable from "no reef at
# all" -- and the whole 2,162-feature inventory held only 13 distinct values.
# The field exists solely so Phase 3 can normalise away the extent confound in
# `oyster_area_m2` (oysters.py's module docstring has the full argument), and
# it cannot do that job at 2 dp. Six decimals resolves ~0.02 m2 of reef inside
# a Point feature's 75 m buffer -- far finer than the SCDNR layer's own 24 m2
# median reef polygon -- while still costing at most six characters per
# feature in the file.
_PROP_DECIMALS = {"oyster_density": 6}
_DEFAULT_DECIMALS = 2


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

    # Oysters hold bait/structure independently of the tide, so this is a
    # static attribute computed once here rather than per-hour derived
    # structure. Both layers must still be in the same UTM CRS for metre
    # distances to mean anything, so this runs before _to4326 below.
    #
    # oyster_area_m2/oyster_nearest_m buffer each feature's *whole* geometry,
    # not a fixed neighbourhood -- for a large Polygon feature (2,026 of the
    # real inventory's 2,162 are Polygon; flats run 1.6-2.0 million m^2) this
    # counts reef anywhere in its interior and reports 0.0 distance whenever
    # any reef falls anywhere inside it, regardless of feature size.
    # oyster_density corrects for that by normalising to the buffered search
    # area itself; see oysters.py's module docstring for the full rationale.
    reefs_utm = load_reefs_utm(slug, epsg)
    geoms = [f.geometry for f in feats]
    reef_areas = reef_area_m2_within(geoms, reefs_utm)
    reef_dists = nearest_reef_m(geoms, reefs_utm)
    # Reuse reef_areas rather than letting reef_density_within recompute it:
    # that would be a second full STRtree build + buffer/intersection pass
    # over every feature, not just a second tree construction.
    reef_density = reef_density_within(geoms, reefs_utm, areas=reef_areas)
    for f, area, dist, density in zip(feats, reef_areas, reef_dists, reef_density, strict=True):
        f.attrs["oyster_area_m2"] = area
        f.attrs["oyster_nearest_m"] = dist
        f.attrs["oyster_density"] = density

    out = []
    for f in feats:
        props = {"type": f.type}
        for k, v in f.attrs.items():
            if isinstance(v, float) and not math.isfinite(v):
                # nearest_reef_m's documented no-reef-layer value is inf; NaN
                # is caught defensively too, for the same reason. JSON has no
                # infinity/NaN literal -- json.dumps emits the bare tokens
                # `Infinity`/`NaN`, both forbidden by RFC 8259 and rejected
                # outright by a browser's JSON.parse, so either must become
                # null here.
                props[k] = None
            elif isinstance(v, float):
                props[k] = round(v, _PROP_DECIMALS.get(k, _DEFAULT_DECIMALS))
            else:
                props[k] = v
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
