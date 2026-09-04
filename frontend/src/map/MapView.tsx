/**
 * The chart. Layer stack, one join, and two paint properties per scrub tick.
 *
 * The layer order (spec §4.1) is bottom to top: basemap, depth tint,
 * hillshade, contours, oyster reefs, then markers. Layers arrive over the
 * network in whatever order the server answers, so `addInOrder` places each
 * one against the stack rather than trusting arrival order.
 *
 * Every fetch here is independent and every failure is logged and skipped
 * (spec §7). A missing depth tint is a degraded chart, not a broken one --
 * and the sounding key says which layer is missing rather than leaving a
 * person to wonder why the water is flat.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import {
  MapLibreMap,
  NavigationControl,
  ScaleControl,
  setWorkerUrl,
  type GeoJSONSource,
  type LngLatBoundsLike,
} from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import type {
  ExpressionSpecification,
  StyleSpecification,
} from "@maplibre/maplibre-gl-style-spec";
import "maplibre-gl/dist/maplibre-gl.css";

import { layerUrl } from "../api/client";
import { useDay } from "../state/DayContext";
import { activationKey, joinActivations } from "./join";
import { ACTIVE_BASEMAP } from "./basemap";
import { colorExpr, MARKER_LAYER_ID, radiusExpr } from "./layers";
import "./MapView.css";

/**
 * MapLibre parses tiles in a worker it loads from a file SIBLING to its own
 * module -- `new URL("./maplibre-gl-worker.mjs", import.meta.url)`, computed
 * at runtime, so no bundler can follow it. Neither Vite's dep pre-bundling
 * (which rewrites the module into .vite/deps/) nor the production build
 * (which hashes it into assets/) leaves that sibling where the URL points,
 * and the miss is SILENT: raster tiles still draw because they decode on the
 * main thread, while every GeoJSON layer stays permanently unloaded and
 * renders zero features. Naming the worker through Vite's `?worker&url`
 * makes it a real build input in both modes.
 */
setWorkerUrl(maplibreWorkerUrl);

/**
 * The API validates a layer name as a dict KEY, so anything not on this list
 * 404s -- see backend/tidescout/api/layers.py. Typed here so a typo is a
 * compile error rather than a layer that silently never appears.
 */
type LayerName =
  | "features"
  | "contours"
  | "oysters"
  | "hillshade"
  | "hillshade-bounds"
  | "depth-tint";

const DEPTH_TINT = "depth-tint";
const HILLSHADE = "hillshade";
const CONTOURS = "contours";
const CONTOUR_LABELS = "contour-labels";
const OYSTERS = "oyster-reefs";
const MARKER_SOURCE = "marker-points";

/**
 * Bottom to top. `addInOrder` inserts a layer before the lowest layer already
 * present that belongs above it, so a slow fetch cannot land its layer on top
 * of one that should cover it.
 *
 * Between the oysters and the markers sit spec §4.1's slots 6 (salinity
 * field) and 7 (flow arrows). They are Task 13's, and they are left empty
 * here rather than stubbed -- an id in this array with no layer behind it
 * would be a promise this task cannot keep.
 */
const STACK = [DEPTH_TINT, HILLSHADE, CONTOURS, CONTOUR_LABELS, OYSTERS, MARKER_LAYER_ID];

const IMAGE_LAYERS = [
  [DEPTH_TINT, { "raster-opacity": 0.82, "raster-fade-duration": 0 }],
  [
    HILLSHADE,
    {
      // `hillshade.png` is a single-band greyscale with NO alpha channel --
      // `webartifacts.hillshade_png` never got the transparency treatment
      // `depth_tint_png` did -- so it is an opaque mid-grey (mean 168/255)
      // across its whole rectangle, nodata included. Laid over the basemap at
      // any real opacity it draws a visible grey BOX around the fishery.
      // Pulling its white point down to 0.5 lands its mid-tone on top of the
      // knocked-back basemap's own tone, so what survives the blend is the
      // relief variation rather than the rectangle. The proper fix is an
      // alpha channel on the artifact, which is backend work.
      "raster-opacity": 0.35,
      "raster-brightness-max": 0.5,
      "raster-fade-duration": 0,
    },
  ],
] as const;

/**
 * Glyphs for the contour labels. MapLibre renders text from font PBFs, which
 * the TideScout API does not serve; this is MapLibre's own public endpoint,
 * the same third-party posture as the OSM tiles above. If it fails, the
 * labels do not draw and every other layer is unaffected.
 */
const GLYPHS = "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf";
const LABEL_FONT = ["Noto Sans Regular"];

/** What a person calls each layer when the key has to admit one is missing. */
const LAYER_LABELS: Record<string, string> = {
  [DEPTH_TINT]: "Depth tint",
  [HILLSHADE]: "Hillshade",
  [CONTOURS]: "Depth contours",
  [OYSTERS]: "Oyster reefs",
  features: "Ambush features",
};

/**
 * What MapLibre's GeoJSON source accepts. Our collections are structurally
 * GeoJSON but typed loosely on purpose -- `join.ts` models `geometry` as
 * `unknown` so it can stay free of any map toolkit -- so they cross into
 * MapLibre through this one named seam instead of an `as never` per call.
 */
type SourceData = Parameters<GeoJSONSource["setData"]>[0];

type Position = [number, number];
type MarkerFeature = {
  type: "Feature";
  id?: string | number;
  properties: Record<string, unknown>;
  geometry: unknown;
};
type MarkerCollection = { type: "FeatureCollection"; features: MarkerFeature[] };

/** Every [lng, lat] pair in a geometry, at any nesting depth. */
function positions(coords: unknown, out: Position[]): void {
  if (!Array.isArray(coords)) return;
  const [first, second] = coords;
  if (typeof first === "number" && typeof second === "number") {
    out.push([first, second]);
    return;
  }
  for (const child of coords) positions(child, out);
}

/**
 * Collapse each feature to the centre of its bounding box.
 *
 * This is not cosmetic. `features.geojson` is 2026 polygons, 134 points and
 * 2 line strings, and MapLibre's circle bucket draws a circle at EVERY vertex
 * of a geometry -- so a circle layer over the raw collection paints five
 * overlapping blobs per dropoff and a dotted string along each jetty, at five
 * times the vertex cost. One ambush feature is one mark.
 *
 * Bounding-box centre rather than a vertex mean: a mean is pulled by wherever
 * the tracer happened to put more vertices, and polygon rings repeat their
 * first vertex to close, which would tug every marker toward it.
 */
export function toMarkerPoints(collection: MarkerCollection): MarkerCollection {
  const features: MarkerFeature[] = [];
  for (const feature of collection.features) {
    const geometry = feature.geometry as { coordinates?: unknown } | null;
    const pts: Position[] = [];
    positions(geometry?.coordinates, pts);
    if (pts.length === 0) continue;
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    for (const [lng, lat] of pts) {
      if (lng < west) west = lng;
      if (lng > east) east = lng;
      if (lat < south) south = lat;
      if (lat > north) north = lat;
    }
    features.push({
      type: "Feature",
      id: feature.id,
      // The join key is the top-level `id`, and MapLibre does not hand a
      // non-numeric id back through queryRenderedFeatures. Copying it into a
      // property is what will let a click find its feature block (Task 11).
      properties: { ...feature.properties, key: feature.id },
      geometry: { type: "Point", coordinates: [(west + east) / 2, (south + north) / 2] },
    });
  }
  return { type: "FeatureCollection", features };
}

/** The bounding box of a whole collection, as MapLibre wants it. */
function collectionBounds(collection: MarkerCollection): LngLatBoundsLike | null {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const feature of collection.features) {
    const geometry = feature.geometry as { coordinates?: unknown } | null;
    const pts: Position[] = [];
    positions(geometry?.coordinates, pts);
    for (const [lng, lat] of pts) {
      if (lng < west) west = lng;
      if (lng > east) east = lng;
      if (lat < south) south = lat;
      if (lat > north) north = lat;
    }
  }
  if (!Number.isFinite(west) || !Number.isFinite(south)) return null;
  return [west, south, east, north];
}

function isBounds(value: unknown): value is [number, number, number, number] {
  return (
    Array.isArray(value) && value.length === 4 && value.every((n) => typeof n === "number")
  );
}

function addInOrder(map: MapLibreMap, layer: Parameters<MapLibreMap["addLayer"]>[0]): void {
  const at = STACK.indexOf(layer.id);
  const above = STACK.slice(at + 1).find((id) => map.getLayer(id) !== undefined);
  map.addLayer(layer, above);
}

export function MapView() {
  const { slug, payload, state, species, hour } = useDay();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  // The join runs on payload arrival, not on a scrub, so it cannot take the
  // hour from a dependency array. It reads the live selection from here to
  // paint the layer correctly the first time.
  const selectionRef = useRef({ species, hour });
  const [ready, setReady] = useState<MapLibreMap | null>(null);
  const [geojson, setGeojson] = useState<MarkerCollection | null>(null);
  const [degraded, setDegraded] = useState<string[]>([]);
  const [unsupported, setUnsupported] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    selectionRef.current = { species, hour };
  }, [species, hour]);

  // --- the map, and every layer that does not depend on the payload --------
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const style: StyleSpecification = {
      version: 8,
      glyphs: GLYPHS,
      sources: { [ACTIVE_BASEMAP.id]: ACTIVE_BASEMAP.source },
      layers: [
        // Only visible where tiles have not landed yet: deep water, so the
        // chart resolves out of the dark instead of flashing white.
        { id: "ground", type: "background", paint: { "background-color": "#06131c" } },
        ACTIVE_BASEMAP.layer,
      ],
    };

    let map: MapLibreMap;
    try {
      map = new MapLibreMap({
        container,
        style,
        center: [0, 0],
        zoom: 1,
        renderWorldCopies: false,
        fadeDuration: 0,
        attributionControl: { compact: false },
      });
    } catch (err) {
      // No WebGL (a headless test environment, a locked-down browser). Say so
      // rather than leaving an empty rectangle.
      setUnsupported(err instanceof Error ? err.message : String(err));
      return;
    }
    mapRef.current = map;
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    // Nautical miles: the scale bar on the chart this is imitating, and the
    // unit the person reading it already thinks in.
    map.addControl(new ScaleControl({ unit: "nautical", maxWidth: 120 }), "bottom-right");

    const controller = new AbortController();
    const objectUrls: string[] = [];
    let cancelled = false;
    let fitted = false;

    // Tile 404s, a glyph endpoint that is down, an image that will not
    // decode: all logged, none fatal.
    map.on("error", (event) => console.warn("[map]", event.error?.message ?? event));

    const url = (name: LayerName) => layerUrl(slug, name);
    const skip = (name: string, err: unknown) => {
      if (cancelled) return;
      const label = LAYER_LABELS[name] ?? name;
      console.warn(`[map] ${label} unavailable, rendering without it:`, err);
      setDegraded((current) => (current.includes(label) ? current : [...current, label]));
    };

    async function getJson<T>(name: LayerName): Promise<T> {
      const res = await fetch(url(name), { signal: controller.signal });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return (await res.json()) as T;
    }

    /**
     * Fetch an image layer ourselves and hand MapLibre a blob URL.
     *
     * An image source pointed straight at the API would swallow a 404 inside
     * the renderer -- no exception to catch, just a layer that never draws --
     * and the key would claim a depth tint that is not there. Fetching it
     * here puts the status in our hands. A HEAD probe would be cheaper but
     * the API does not answer one: FastAPI's `@app.get` registers GET only,
     * so HEAD returns 405, which is not the "missing layer" this is asking
     * about. The blob costs no extra download -- MapLibre reads the bytes we
     * already have instead of requesting them again.
     */
    async function imageUrl(name: LayerName): Promise<string> {
      const res = await fetch(url(name), { signal: controller.signal });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const objectUrl = URL.createObjectURL(await res.blob());
      objectUrls.push(objectUrl);
      return objectUrl;
    }

    function fitTo(bounds: LngLatBoundsLike) {
      if (cancelled || fitted) return;
      fitted = true;
      map.fitBounds(bounds, { padding: 40, duration: 0 });
      setRevealed(true);
    }

    async function imagery() {
      const sidecar = await getJson<{ bounds?: unknown }>("hillshade-bounds");
      if (cancelled) return;
      if (!isBounds(sidecar.bounds)) throw new Error("bounds sidecar is not [w,s,e,n]");
      const [west, south, east, north] = sidecar.bounds;
      // Both PNGs are warps of the same grid onto EPSG:3857 and share this
      // one sidecar, which is why they take identical corners.
      const corners: [Position, Position, Position, Position] = [
        [west, north],
        [east, north],
        [east, south],
        [west, south],
      ];
      fitTo([west, south, east, north]);

      for (const [name, paint] of IMAGE_LAYERS) {
        try {
          const src = await imageUrl(name);
          if (cancelled) return;
          map.addSource(name, { type: "image", url: src, coordinates: corners });
          addInOrder(map, { id: name, type: "raster", source: name, paint });
        } catch (err) {
          skip(name, err);
        }
      }
    }

    async function contours() {
      const data = await getJson<MarkerCollection>("contours");
      if (cancelled) return;
      map.addSource(CONTOURS, { type: "geojson", data: data as unknown as SourceData });
      addInOrder(map, {
        id: CONTOURS,
        type: "line",
        source: CONTOURS,
        paint: {
          // Four levels only (-2, -5, -10, -15 m). The 2-metre line is the one
          // that matters to a shallow-draft boat, so it carries the weight;
          // the deep ones recede to context.
          "line-color": "rgba(196,226,240,0.42)",
          "line-width": ["match", ["get", "depth_m"], -2, 0.9, -5, 0.7, -10, 0.6, 0.5],
        },
      });
      addInOrder(map, {
        id: CONTOUR_LABELS,
        type: "symbol",
        source: CONTOURS,
        layout: {
          "symbol-placement": "line",
          "text-field": ["concat", ["to-string", ["abs", ["get", "depth_m"]]], " m"],
          "text-font": LABEL_FONT,
          "text-size": 10,
          "text-letter-spacing": 0.08,
          "symbol-spacing": 240,
          // Contours traced off a real DEM wander; the default 45 degrees is
          // already the loosest MapLibre will place text through, and
          // tightening it drops nearly every label on a line like these.
          "text-max-angle": 45,
        },
        paint: {
          "text-color": "rgba(213,232,240,0.8)",
          "text-halo-color": "rgba(6,19,28,0.85)",
          "text-halo-width": 1.1,
        },
      });
    }

    async function oysters() {
      const data = await getJson<MarkerCollection>("oysters");
      if (cancelled) return;
      map.addSource(OYSTERS, { type: "geojson", data: data as unknown as SourceData });
      addInOrder(map, {
        id: OYSTERS,
        type: "fill",
        source: OYSTERS,
        paint: {
          // Sage, not the gold at the top of the activation ramp: at bay zoom
          // a reef is a speck, and a warm speck beside a warm marker reads as
          // a second hot spot. Oysters are texture, not score.
          "fill-color": "rgba(163,181,146,0.5)",
          "fill-outline-color": "rgba(163,181,146,0.75)",
        },
      });
    }

    async function features() {
      const data = await getJson<MarkerCollection>("features");
      if (cancelled) return;
      setGeojson(data);
      const bounds = collectionBounds(data);
      if (bounds) fitTo(bounds);
    }

    map.on("load", () => {
      if (cancelled) return;
      setReady(map);
      void Promise.allSettled([
        // The bounds sidecar feeds BOTH image layers, so losing it loses
        // both -- named separately because that is what the key must say.
        imagery().catch((err) => {
          skip(DEPTH_TINT, err);
          skip(HILLSHADE, err);
        }),
        contours().catch((err) => skip(CONTOURS, err)),
        oysters().catch((err) => skip(OYSTERS, err)),
        features().catch((err) => skip("features", err)),
      ]).then(() => {
        // Nothing placed the view -- show the map anyway rather than holding
        // a blank panel forever.
        if (!cancelled) setRevealed(true);
      });
    });

    return () => {
      cancelled = true;
      controller.abort();
      mapRef.current = null;
      setReady(null);
      setRevealed(false);
      map.remove();
      for (const objectUrl of objectUrls) URL.revokeObjectURL(objectUrl);
    };
  }, [slug]);

  // --- the join: once, when the payload and the geometry have both arrived --
  useEffect(() => {
    if (!ready || !geojson || !payload) return;
    const joined = joinActivations(geojson, payload);
    const points = toMarkerPoints(joined);
    const source = ready.getSource(MARKER_SOURCE) as GeoJSONSource | undefined;
    if (source) {
      // Reached only when the DAY changes -- a new payload is new numbers on
      // the same geometry. Never on a scrub.
      source.setData(points as unknown as SourceData);
      return;
    }
    ready.addSource(MARKER_SOURCE, { type: "geojson", data: points as unknown as SourceData });
    const { species: sp, hour: hr } = selectionRef.current;
    addInOrder(ready, {
      id: MARKER_LAYER_ID,
      type: "circle",
      source: MARKER_SOURCE,
      paint: {
        "circle-radius": radiusExpr(sp, hr) as ExpressionSpecification,
        "circle-color": colorExpr(sp, hr) as ExpressionSpecification,
        // Constant, all three of them: anything data-driven by activation
        // would have to be re-set on every scrub tick, which is the cost this
        // whole design exists to avoid.
        "circle-opacity": 0.92,
        "circle-stroke-width": 0.8,
        "circle-stroke-color": "rgba(6,19,28,0.6)",
      },
    });

    // The 1,633 features outside the flow-model domain are not clickable, and
    // the cursor is where that becomes visible before a click proves it.
    const canvas = ready.getCanvas();
    ready.on("mousemove", MARKER_LAYER_ID, (event) => {
      const props = event.features?.[0]?.properties;
      const { species: s, hour: h } = selectionRef.current;
      const scored = props ? props[activationKey(s, h)] !== undefined : false;
      canvas.style.cursor = scored ? "pointer" : "";
    });
    ready.on("mouseleave", MARKER_LAYER_ID, () => {
      canvas.style.cursor = "";
    });
    // `species` and `hour` are deliberately absent: this must not re-run on a
    // scrub. The two setPaintProperty calls below are the whole tick.
  }, [ready, geojson, payload]);

  // --- the scrub loop ------------------------------------------------------
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !species || !map.getLayer(MARKER_LAYER_ID)) return;
    // The cast is the seam between a toolkit-free `layers.ts` (it returns
    // plain arrays, which is what lets it be unit-tested without a map) and
    // MapLibre's typed setter. tests/layers.test.ts parses both expressions
    // against the real style spec, so the cast is backed by a test.
    map.setPaintProperty(
      MARKER_LAYER_ID,
      "circle-radius",
      radiusExpr(species, hour) as ExpressionSpecification,
    );
    map.setPaintProperty(
      MARKER_LAYER_ID,
      "circle-color",
      colorExpr(species, hour) as ExpressionSpecification,
    );
  }, [species, hour]);

  const counts = useMemo(() => {
    const total = geojson?.features.length ?? 0;
    const block = species ? payload?.species[species] : undefined;
    const scored = block ? Object.keys(block.features).length : 0;
    return { total, unscored: Math.max(total - scored, 0) };
  }, [geojson, payload, species]);

  if (unsupported) {
    return (
      <div className="map" data-testid="map">
        <div className="map-unsupported">
          <p className="eyebrow">Chart unavailable</p>
          <p>This map needs WebGL. {unsupported}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="map" data-testid="map">
      <div className="map-canvas" ref={containerRef} data-revealed={revealed} />
      <figure className="key" data-testid="map-key">
        <figcaption className="eyebrow">
          Activation
          <span className="key-scope">
            {species ? species.replace(/_/g, " ") : "—"} ·{" "}
            {String(hour).padStart(2, "0")}:00
          </span>
        </figcaption>
        <div className="key-ramp" />
        <div className="key-ticks">
          <span>0</span>
          <span>50</span>
          <span>100</span>
        </div>
        <ul className="key-notes">
          <li>
            <i className="dot dot-ghost" />
            {counts.unscored.toLocaleString()} of {counts.total.toLocaleString()} features sit
            outside the flow model — detected, not scored
          </li>
          {state === "building" && (
            <li>
              <i className="dot dot-pending" />
              Scoring this day. Markers appear when it finishes.
            </li>
          )}
          {degraded.map((label) => (
            <li key={label}>
              <i className="dot dot-missing" />
              {label} did not load — the rest of the chart is drawn without it
            </li>
          ))}
        </ul>
      </figure>
    </div>
  );
}

export default MapView;
