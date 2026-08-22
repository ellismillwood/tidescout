"""SCDNR oyster reefs as a habitat attribute of the existing features.

Carryover item 6 is explicit that this is a scoring/habitat layer, not an
ambush-feature class and not mesh geometry: 8,451 polygons with a median area
of 24 m^2 would add thousands of markers to the map and resolve nothing, but
"this drop-off has 400 m2 of reef within a cast" is a real signal that holds
independently of the tide.

Static: reefs do not move with the tide, so this is computed once into
features.geojson rather than per hour.

CAUTION -- extent confound in `oyster_area_m2`/`oyster_nearest_m`: both
buffer the feature's *whole* geometry, not a fixed-size neighbourhood. For a
Point that is exactly "reef within radius_m of this point," but 2,026 of the
real inventory's 2,162 features are Polygon, and flats alone run 1.6-2.0
million m^2 -- four of the five largest `oyster_area_m2` values in the real
inventory are flats, purely because the detector drew a big polygon there,
not because their reef is denser than a small drop-off's. `oyster_nearest_m`
is 0.0 whenever any reef falls anywhere inside a Polygon feature, however
large that polygon is. Consumers that want a scale-free "how reef-dense is
this spot" answer should use `reef_density_within`/`oyster_density` instead,
which normalises by the buffered search area itself.

`oyster_area_m2` and `oyster_nearest_m` are kept exactly as specified even
so -- they are the plan's named cross-phase interface, and the seemingly
obvious fix (buffer the centroid instead of the whole geometry) trades this
bias for a worse one: a flat's fishable oyster is typically along its edge,
so centroid-buffering would systematically miss the reef that matters most
on precisely the features this note is about.
"""

import json
import logging
import math

from rasterio.warp import transform as warp_transform
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.strtree import STRtree

from tidescout.paths import fishery_data_dir

log = logging.getLogger(__name__)

# Roughly a cast. Wider and every feature in the bay picks up some reef;
# narrower and the median 24 m^2 reef is missed by features that fish it.
DEFAULT_RADIUS_M = 75.0


def reef_area_m2_within(features, reefs, radius_m: float = DEFAULT_RADIUS_M) -> list[float]:
    """Total reef area within `radius_m` of each feature, in m^2.

    Zero, not NaN, when nothing is nearby: most of the bay has no reef, and that
    is a real answer. NaN would make Phase 3 exclude the factor and renormalise,
    converting "no oysters here" into "we have no oyster data" -- opposite
    claims with opposite consequences for the score.

    STRtree-indexed: the layer is 8,451 reefs against thousands of features, and
    the naive nested loop is 30M+ intersection tests.

    Extent confound (see module docstring): buffers the feature's whole
    geometry, not a fixed neighbourhood -- a large Polygon feature's total
    includes reef anywhere in its interior, not just near an edge or within a
    cast's length of one. Prefer `reef_density_within` when comparing feature
    sizes that vary a lot, e.g. a Point against a multi-hectare flat.
    """
    if not reefs:
        return [0.0] * len(features)
    tree = STRtree(reefs)
    out = []
    for geom in features:
        buffered = geom.buffer(radius_m)
        total = 0.0
        for idx in tree.query(buffered):
            reef = reefs[idx]
            if buffered.intersects(reef):
                total += buffered.intersection(reef).area
        out.append(total)
    return out


def nearest_reef_m(features, reefs) -> list[float]:
    """Distance to the closest reef, 0.0 when the feature overlaps one.

    Infinity when there are no reefs at all -- a distance, unlike an area, has
    no meaningful zero-substitute, and inf makes any downstream curve clamp to
    its worst authored value rather than its best. Callers that serialise this
    to JSON must convert inf to null themselves: JSON has no infinity literal.

    Extent confound (see module docstring): 0.0 for a Polygon feature means
    "some reef is somewhere inside this polygon," which can be anywhere in a
    multi-hectare flat -- not necessarily near any particular edge of it.
    """
    if not reefs:
        return [math.inf] * len(features)
    tree = STRtree(reefs)
    return [float(geom.distance(reefs[tree.nearest(geom)])) for geom in features]


def reef_density_within(
    features,
    reefs,
    radius_m: float = DEFAULT_RADIUS_M,
    *,
    areas: list[float] | None = None,
) -> list[float]:
    """Reef area within `radius_m` of each feature, as a fraction of the
    buffered search area itself: `reef_area_m2_within / buffer_area_m2`.

    Dimensionless and scale-free -- comparable between a 20 m point feature
    and a 2 km^2 flat, where `reef_area_m2_within` alone is dominated by how
    big the detector's polygon happened to be (see module docstring). A
    small feature fully ringed by reef and a huge one fully ringed by reef
    both come out near 1.0 here, even though their `reef_area_m2_within`
    values differ by orders of magnitude.

    Pass `areas` -- `reef_area_m2_within`'s own output -- when the caller has
    already computed it, as `build_features` does: reusing it skips a second
    full `STRtree` build (8,451 reefs) and a second buffer/intersection pass
    over every feature, which is real cost, not just tree-construction
    overhead. Left unset, this calls `reef_area_m2_within` itself so the
    function stays standalone and pure for direct callers, including this
    module's own tests -- that call *does* redo the full computation; there
    is no way to get the area without it when the caller has nothing cached.
    """
    if areas is None:
        areas = reef_area_m2_within(features, reefs, radius_m)
    return [
        area / geom.buffer(radius_m).area for area, geom in zip(areas, features, strict=True)
    ]


def reef_to_utm(geom, epsg: int):
    """Reproject one reef geometry from the layer's native EPSG:4326 to the
    fishery's analysis UTM CRS.

    8,409 of the 8,451 real SCDNR reefs are Polygon; 42 are MultiPolygon (a
    single reef id traced as disjoint patches). `features._to4326` only runs
    UTM->4326 and raises TypeError on MultiPolygon, so this is its own
    function rather than a reuse: both the direction and the geometry
    coverage differ.
    """

    def tx(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        us, vs = warp_transform("EPSG:4326", f"EPSG:{epsg}", xs, ys)
        return list(zip(us, vs, strict=True))

    if isinstance(geom, Polygon):
        return Polygon(
            tx(list(geom.exterior.coords)),
            [tx(list(r.coords)) for r in geom.interiors],
        )
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([reef_to_utm(part, epsg) for part in geom.geoms])
    raise TypeError(f"unsupported reef geometry: {geom.geom_type}")


def load_reefs_utm(slug: str, epsg: int) -> list:
    """Load `data/<slug>/oyster_reefs.geojson` (SCDNR reefs, EPSG:4326) and
    reproject every reef into the fishery's analysis UTM CRS.

    A missing file is not an error -- it is the spec's documented contingency
    ("SCDNR oyster layer unavailable -> feature class is optional"). Logged
    and an empty list returned: `reef_area_m2_within`/`nearest_reef_m` already
    turn `reefs=[]` into 0.0 area / inf distance for every feature, so the
    fallback needs no extra branching here.
    """
    path = fishery_data_dir(slug) / "oyster_reefs.geojson"
    if not path.exists():
        log.warning(
            "no oyster reef layer at %s -- every feature gets oyster_area_m2=0.0, "
            "oyster_nearest_m=inf",
            path,
        )
        return []
    fc = json.loads(path.read_text())
    return [reef_to_utm(shape(f["geometry"]), epsg) for f in fc["features"]]
