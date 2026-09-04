/**
 * The ground the chart sits on.
 *
 * Spec §1.1: swapping OSM for satellite must be ONE constant, not a rewrite.
 * So both definitions carry the same shape, `MapView` reads only
 * `ACTIVE_BASEMAP`, and nothing downstream knows which one it got.
 */
import type {
  RasterLayerSpecification,
  RasterSourceSpecification,
} from "@maplibre/maplibre-gl-style-spec";

export interface Basemap {
  /** The style's source id. The layer id matches it -- separate namespaces. */
  id: string;
  /** Human name, for a future basemap switcher. */
  label: string;
  /**
   * `attribution` is REQUIRED here, not optional.
   *
   * `RasterSourceSpecification` declares it `attribution?: string`, and an
   * optional legal notice is one that eventually ships empty. Intersecting
   * the required field on makes a basemap without attribution a type error.
   * It lives on the source rather than beside it because that is the field
   * MapLibre's AttributionControl actually reads.
   */
  source: RasterSourceSpecification & { attribution: string };
  layer: RasterLayerSpecification;
}

/**
 * OSM raster.
 *
 * Knocked back on purpose. Street cartography is tuned to be the subject of
 * its own map; here it is the ground under a chart, and at full saturation
 * its road casings and landuse fills compete with the depth tint and the
 * markers -- the two things a person is actually reading. Desaturating and
 * darkening it turns it into terrain and shoreline context, which is all it
 * is being asked for.
 */
export const OSM: Basemap = {
  id: "basemap",
  label: "OpenStreetMap",
  source: {
    type: "raster",
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    tileSize: 256,
    maxzoom: 19,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  layer: {
    id: "basemap",
    type: "raster",
    source: "basemap",
    paint: {
      "raster-saturation": -0.5,
      "raster-brightness-max": 0.4,
      "raster-contrast": -0.05,
      // Raster cross-fading holds the previous zoom level on screen while the
      // next one loads. On a still chart that reads as smearing, not motion.
      "raster-fade-duration": 0,
    },
  },
};

/**
 * Esri World Imagery. Unused today (spec §1.1: no UI toggle yet) and kept
 * beside OSM so the swap is `ACTIVE_BASEMAP = SATELLITE` and nothing else.
 *
 * No desaturation: aerial imagery of a marsh is already the muted green-brown
 * this palette wants, and pulling colour out of it would only make the water
 * and the land harder to tell apart.
 */
export const SATELLITE: Basemap = {
  id: "basemap",
  label: "Satellite",
  source: {
    type: "raster",
    tiles: [
      "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    tileSize: 256,
    maxzoom: 19,
    attribution:
      'Imagery &copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics',
  },
  layer: {
    id: "basemap",
    type: "raster",
    source: "basemap",
    paint: {
      "raster-brightness-max": 0.9,
      "raster-fade-duration": 0,
    },
  },
};

/** The one line that chooses. */
export const ACTIVE_BASEMAP: Basemap = OSM;
