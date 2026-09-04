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
import type { CSSProperties, ReactNode } from "react";

import {
  MapLibreMap,
  NavigationControl,
  ScaleControl,
  setWorkerUrl,
  type GeoJSONSource,
  type LngLatBoundsLike,
  type MapLayerMouseEvent,
  type MapMouseEvent,
} from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import type {
  ExpressionSpecification,
  StyleSpecification,
} from "@maplibre/maplibre-gl-style-spec";
import "maplibre-gl/dist/maplibre-gl.css";

import { layerUrl } from "../api/client";
import { FeaturePopover } from "../rail/FeaturePopover";
import { useDay } from "../state/DayContext";
import { activationKey, joinActivations } from "./join";
import { ACTIVE_BASEMAP } from "./basemap";
import {
  ACTIVATION_STOPS,
  colorExpr,
  MARKER_LAYER_ID,
  radiusExpr,
  UNSCORED_COLOR,
} from "./layers";
import {
  arrowCollection,
  classRange,
  fetchFlowVectors,
  fetchSalinityField,
  KNOTS_PER_MS,
  REFERENCE_SPEED_MS,
  SALINITY_CLASS_COUNT,
  SALINITY_CLASS_PPT,
  useOverlay,
  type Overlay,
  type SalinitySection,
} from "./overlays";
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
const FLOW_SOURCE = "flow-field";
const FLOW_CASING = "flow-arrows-casing";
const FLOW_ARROWS = "flow-arrows";

/**
 * Bottom to top. `addInOrder` inserts a layer before the lowest layer already
 * present that belongs above it, so a slow fetch cannot land its layer on top
 * of one that should cover it.
 *
 * Slot 6, the flow arrows, sits between the oysters and the markers, and it
 * is two layers rather than one: a dark casing under a pale shaft. The tint
 * beneath ramps pale cyan (198,232,240) in the shallows to deep navy
 * (18,62,128) in the channel, so ONE ink cannot carry the field -- a pale
 * arrow vanishes on the flats and a dark one vanishes in the deep. The casing
 * is what a paper chart does when it prints over its own soundings.
 *
 * The salinity field is deliberately NOT a layer here. It has no geometry to
 * be one with: the endpoint returns a 1-D profile binned by along-estuary
 * distance, and the per-cell distance field it would be painted through
 * (`estuary_km.npy`) is not on the layer allowlist, so no URL serves it.
 * Painting the bay anyway would mean inventing an estuary axis on the client
 * -- the discounting-by-overclaiming spec §1.1 forbids, applied to a model
 * that is already falsified (`fitted: false`). It renders as a section inset
 * in the chart's margin instead -- see `SalinityInset`. Spec §4.1 listed it
 * as layer 6 and is amended to match what shipped, including what serving the
 * distance field would take.
 */
const STACK = [
  DEPTH_TINT,
  HILLSHADE,
  CONTOURS,
  CONTOUR_LABELS,
  OYSTERS,
  FLOW_CASING,
  FLOW_ARROWS,
  MARKER_LAYER_ID,
];

const IMAGE_LAYERS = [
  [DEPTH_TINT, { "raster-opacity": 0.82, "raster-fade-duration": 0 }],
  [
    HILLSHADE,
    {
      // The nodata frame is GONE, and it was fixed in the backend, not here:
      // `hillshade.png` is RGBA now (`webartifacts.hillshade_png`, alpha 0 on
      // nodata), so the 6.7% margin that used to paint an opaque black border
      // around the fishery is simply not painted. No paint property could
      // have fixed that -- lowering the white point makes already-black
      // pixels RELATIVELY more prominent.
      //
      // The white point stays, doing the OTHER job it turned out to be doing.
      // A hillshade is relief, i.e. variation, and it must not also be a
      // brightness lift. This artifact's opaque area means 180/255 while the
      // deliberately knocked-back basemap under it sits at ~83, so at full
      // white point the layer washes its whole rectangle +34 levels brighter
      // than the world around it -- measured in the running app, and a slab
      // that obvious is just the old box in a lighter colour. Mapping the
      // white point to 0.5 lands the mean at ~90 against the ground's 83
      // (the exactly-seamless value is 83/180 = 0.46; 0.5 measures a +3.0
      // step, at the noise floor) and what survives the blend is the relief
      // variation alone. Relief amplitude is unchanged: sd 10.7 vs 11.1.
      "raster-opacity": 0.35,
      "raster-brightness-max": 0.5,
      "raster-fade-duration": 0,
    },
  ],
] as const;

/**
 * The key's ramp bar, built from `layers.ts`'s stops rather than a hand copy.
 * Activation is 0-100, so a stop's value IS its percentage along the bar --
 * which is what makes the bar show the real, unevenly spaced mapping.
 */
const KEY_RAMP = `linear-gradient(to right, ${ACTIVATION_STOPS.map(
  ([at, colour]) => `${colour} ${at}%`,
).join(", ")})`;

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

/** The marker a click picked, kept in map coordinates so it can be reprojected. */
type Picked = { key: string; lng: number; lat: number };

/**
 * Where the popover hangs off that marker, in pixels, plus which way it
 * flips. `.map` clips its overflow, so a card anchored near an edge has to
 * flip rather than hang off it.
 */
type Anchor = {
  x: number;
  y: number;
  vertical: "above" | "below";
  horizontal: "start" | "center" | "end";
  /** How tall the card may be before it would run out of chart to sit in. */
  room: number;
};

/**
 * The current arrows, in the chart's own ink.
 *
 * NO COLOUR RAMP. `layers.ts` spends the entire hue channel on activation --
 * "nothing in this palette but a marker is magenta or gold" -- and a
 * second ramp beside it would make the chart ask which colour meant what.
 * Speed rides on length and line width instead, both of which the reference
 * arrow in the overlay panel gives a scale for.
 */
const ARROW_INK = "rgba(236,231,218,0.88)";
const ARROW_CASING = "rgba(6,19,28,0.72)";

/** Width from speed, at the same reference the arrow's length uses. */
const arrowWidth = (extra: number): ExpressionSpecification =>
  [
    "interpolate",
    ["linear"],
    ["get", "speed"],
    0,
    0.55 + extra,
    REFERENCE_SPEED_MS,
    1.5 + extra,
  ] as ExpressionSpecification;

/** Dimmed while a later hour is in flight, so stale arrows never look current. */
const ARROW_OPACITY = 0.92;
const ARROW_OPACITY_STALE = 0.4;

/** Half the popover's width plus its offset -- the edge it must not cross. */
const POPOVER_MARGIN = 190;
/** The gap between the marker and the card, and between the card and the edge. */
const POPOVER_GAP = 14;
/** Below this the card is not worth flipping for -- it scrolls instead. */
const POPOVER_MIN = 190;

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

/**
 * The reference arrow. Length means speed, so the key has to draw one at full
 * length and say what full length is -- otherwise the field is a direction
 * map with a decorative second variable.
 */
function ArrowScale() {
  return (
    <p className="ov-scale">
      <svg viewBox="0 0 46 10" aria-hidden="true" focusable="false">
        <path d="M2 5 H41 M33.6 1.4 L41 5 L33.6 8.6" />
      </svg>
      <span className="num">{(REFERENCE_SPEED_MS * KNOTS_PER_MS).toFixed(1)} kn</span>
      <span>or faster, at full length</span>
    </p>
  );
}

/** What the section says about itself under the badge. Never empty. */
function salinityNote(section: SalinitySection): string {
  if (!section.fitted) {
    return "The model behind this is unfitted: no observation constrains it, at any distance.";
  }
  if (section.extrapolated) {
    return "Today's discharge sits outside the range this model was fit over.";
  }
  return "Modelled from today's discharge and the tide phase of this hour.";
}

/**
 * The salinity field, as a section through the estuary.
 *
 * Spec §1.1 is "flagged, not discounted", and both halves of that are load
 * bearing. It IS drawn. And every choice in this drawing exists so that it
 * cannot be read as measurement:
 *
 *   - It is a SECTION, not a tint on the water. The endpoint returns one
 *     value per kilometre of along-estuary distance, which is a curve, not a
 *     field; painting it over the bay would need a distance field the client
 *     does not have and would claim a two-dimensional structure the model
 *     never produced.
 *   - It is STEPPED, into 2.5 ppt classes, and the steps are the only heights
 *     available -- `SalinityBand` carries an integer class and no ppt, so
 *     there is nothing here to draw a smooth curve or a crisp isoline from.
 *     See the header of `overlays.ts`.
 *   - It wears the WEAVE this app already uses for "modelled, nothing
 *     observed constrains it" (FactorBars, the strip, the disclosure band), so
 *     a reader who has learnt the mark once reads it here without being told.
 *   - The badge is PERMANENT. It is a required field of the section, typed as
 *     a union of three non-empty literals, so no state of this component
 *     renders without one.
 */
export function SalinityInset({
  section,
  busy = false,
}: {
  section: SalinitySection;
  busy?: boolean;
}) {
  let peak = 0;
  for (const band of section.bands) if (band.klass > peak) peak = band.klass;
  const [saltLow, saltHigh] = classRange(peak);
  return (
    <figure
      className="sal"
      data-testid="salinity-section"
      data-busy={busy}
      aria-busy={busy}
    >
      <figcaption>
        <span className="eyebrow">Salinity section</span>
        <span className="sal-badge" data-testid="salinity-badge">
          {section.badge}
        </span>
      </figcaption>
      <div
        className="sal-plot"
        role="img"
        aria-label={
          `Modelled salinity along ${Math.round(section.kmMax - section.kmMin)} km of ` +
          `estuary, stepped into ${SALINITY_CLASS_PPT} ppt classes. The saltiest class ` +
          `reached is ${saltLow} to ${saltHigh} ppt. ${section.badge}.`
        }
        style={{ "--sal-classes": SALINITY_CLASS_COUNT } as CSSProperties}
      >
        {section.bands.map((band) => (
          <span
            key={band.km}
            className="sal-band"
            data-klass={band.klass}
            style={{ "--klass": band.klass } as CSSProperties}
          />
        ))}
      </div>
      <p className="sal-axis">
        <span>{Math.round(section.kmMin)} km · mouth</span>
        <span>{Math.round(section.kmMax)} km · head</span>
      </p>
      <p className="sal-note">
        Each step is a {SALINITY_CLASS_PPT} ppt class, not a value, and no contour is
        drawn through them. {salinityNote(section)}
      </p>
    </figure>
  );
}

/** One overlay's row: its switch, what it is doing, and why it stopped. */
function OverlayRow({
  overlay,
  name,
  children,
}: {
  overlay: Overlay<unknown>;
  name: string;
  children?: ReactNode;
}) {
  const testId = name.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="ov-row">
      <label className="ov-switch">
        <input
          type="checkbox"
          checked={overlay.on}
          data-testid={`toggle-${testId}`}
          onChange={(event) => overlay.enable(event.target.checked)}
        />
        <span className="ov-name">{name}</span>
        <span className="ov-state num" aria-hidden={!overlay.busy}>
          {overlay.busy ? "…" : ""}
        </span>
      </label>
      {overlay.on && children}
      {overlay.error && (
        <p className="ov-error" role="status" data-testid={`error-${testId}`}>
          {name} did not load — {overlay.error}. Switch it back on to try again.
        </p>
      )}
    </div>
  );
}

export function MapView() {
  const { slug, payload, state, species, hour, date, model } = useDay();
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
  const [picked, setPicked] = useState<Picked | null>(null);
  const [anchor, setAnchor] = useState<Anchor | null>(null);

  /**
   * The two optional overlays. Both off on mount, both per-hour, and both
   * debounced -- the ONE place in this app where a scrub reaches the network.
   * `useOverlay` owns the toggle as well as the fetch, so a failure reverts
   * its own switch and the chart under it is never touched.
   */
  const request = { slug, date, hour, model };
  const flow = useOverlay(fetchFlowVectors, request);
  const salinity = useOverlay(fetchSalinityField, request);

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
      // All four pieces of per-map state die with the map. `degraded` and
      // `geojson` in particular: `setDegraded` only ever APPENDS, so a
      // fishery whose depth tint failed would keep saying so on the next
      // fishery, whose tint is fine -- and a stale `geojson` would let the
      // join paint fishery A's markers over fishery B's basemap if B's
      // `features` fetch failed. `slug` is a constant today; Task 12's
      // fishery picker is what makes this reachable.
      setReady(null);
      setRevealed(false);
      setDegraded([]);
      setGeojson(null);
      // A popover keyed to a feature of the OLD fishery would outlive its
      // payload and read as a feature of the new one.
      setPicked(null);
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
      // the same geometry. Never on a scrub. The source and layer are already
      // there; the handlers below are (re-)registered either way, and the
      // cleanup at the end of this effect is what makes that safe.
      source.setData(points as unknown as SourceData);
    } else {
      ready.addSource(MARKER_SOURCE, {
        type: "geojson",
        data: points as unknown as SourceData,
      });
      const { species: sp, hour: hr } = selectionRef.current;
      addInOrder(ready, {
        id: MARKER_LAYER_ID,
        type: "circle",
        source: MARKER_SOURCE,
        paint: {
          "circle-radius": radiusExpr(sp, hr) as ExpressionSpecification,
          "circle-color": colorExpr(sp, hr) as ExpressionSpecification,
          // Constant, all three of them: anything data-driven by activation
          // would have to be re-set on every scrub tick, which is the cost
          // this whole design exists to avoid.
          "circle-opacity": 0.92,
          "circle-stroke-width": 0.8,
          "circle-stroke-color": "rgba(6,19,28,0.6)",
        },
      });
    }

    /*
     * THE THREE HANDLERS, AND WHY THEY ARE NAMED AND CLEANED UP.
     *
     * They register on every run of this effect and come off in its cleanup,
     * so "exactly one of each is attached" is a property of the effect's own
     * shape. It used to rest on the `setData` branch above returning early,
     * three statements away: correct, but only as long as nobody moved the
     * return -- and no test can catch that, because jsdom has no WebGL and
     * this effect never runs under vitest. (The review confirmed the
     * accumulation this guards against does NOT happen today. This is
     * structure, not a bug fix.)
     */
    // The 1,633 features outside the flow-model domain are not clickable, and
    // the cursor is where that becomes visible before a click proves it.
    const canvas = ready.getCanvas();
    const onMarkerMove = (event: MapLayerMouseEvent) => {
      const props = event.features?.[0]?.properties;
      const { species: s, hour: h } = selectionRef.current;
      const scored = props ? props[activationKey(s, h)] !== undefined : false;
      canvas.style.cursor = scored ? "pointer" : "";
    };
    const onMarkerLeave = () => {
      canvas.style.cursor = "";
    };
    ready.on("mousemove", MARKER_LAYER_ID, onMarkerMove);
    ready.on("mouseleave", MARKER_LAYER_ID, onMarkerLeave);

    /**
     * The click that opens the popover.
     *
     * ONE handler on the map, querying the marker layer itself, rather than a
     * layer-scoped `on("click", MARKER_LAYER_ID, ...)` plus a second one for
     * "clicked the water": both fire for a click on a marker, and the pair
     * would open the popover and immediately close it depending on the order
     * MapLibre happens to dispatch them.
     */
    const onMapClick = (event: MapMouseEvent) => {
      const hit = ready.queryRenderedFeatures(event.point, {
        layers: [MARKER_LAYER_ID],
      })[0];
      if (!hit) {
        setPicked(null);
        return;
      }
      // `key` is the join key copied into properties by `toMarkerPoints` --
      // MapLibre does not hand a non-numeric feature id back through a query.
      const key = hit.properties?.["key"];
      const coords = (hit.geometry as { coordinates?: unknown }).coordinates;
      if (typeof key !== "string" || !Array.isArray(coords)) return;
      const [lng, lat] = coords;
      if (typeof lng !== "number" || typeof lat !== "number") return;
      // The marker's own point, not the click point: the card stays pinned to
      // the feature rather than to wherever the pointer happened to land.
      setPicked({ key, lng, lat });
    };
    ready.on("click", onMapClick);
    return () => {
      ready.off("mousemove", MARKER_LAYER_ID, onMarkerMove);
      ready.off("mouseleave", MARKER_LAYER_ID, onMarkerLeave);
      ready.off("click", onMapClick);
      // The cursor is set by the handler that just went away; leaving a
      // "pointer" behind would point at a layer nothing is listening on.
      canvas.style.cursor = "";
    };
    // DO NOT ADD `species` OR `hour` TO THIS ARRAY. This effect re-joins
    // 2,162 features and rebuilds the whole marker source; running it on a
    // scrub tick is the single cost this design exists to avoid (measured:
    // 24 hour changes in 143 ms, median frame 6.0 ms, zero network). The
    // scrub is the two setPaintProperty calls below and nothing else.
    //
    // The array is exhaustive as written -- the effect reads the live
    // selection from `selectionRef`, not from `species`/`hour`, purely to
    // paint the layer correctly the FIRST time -- so the rule is quiet today.
    // The suppression is for the day someone inlines that ref read: the rule
    // would then demand both names and an autofix would supply them,
    // reintroducing the per-tick re-join with no test to catch it.
    //
    // Cost of this line, measured: oxlint 1.79 does not scope an
    // `exhaustive-deps` directive to the line it sits on. It silently drops
    // `react/set-state-in-effect` for the ENTIRE enclosing component -- here
    // that is the accepted `setUnsupported` in the WebGL catch above, and it
    // would hide a genuine one too. `react/rules-of-hooks` is unaffected, and
    // no rule config prevents it. Reproduced against every spelling of the
    // directive; naming any other rule behaves correctly.
    // oxlint-disable-next-line react/exhaustive-deps
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

  // --- slot 6: the current arrows ------------------------------------------
  // Adds on first arrival, `setData`s on every later hour, and REMOVES itself
  // when the toggle goes off or a fetch fails. Removing the layer and its
  // source rather than hiding them is what makes "toggling it off restores the
  // base map" true of the map's own state and not just of what is on screen.
  useEffect(() => {
    if (!ready) return;
    const field = flow.data;
    if (!field) {
      if (ready.getLayer(FLOW_ARROWS)) ready.removeLayer(FLOW_ARROWS);
      if (ready.getLayer(FLOW_CASING)) ready.removeLayer(FLOW_CASING);
      if (ready.getSource(FLOW_SOURCE)) ready.removeSource(FLOW_SOURCE);
      return;
    }
    const arrows = arrowCollection(field) as unknown as SourceData;
    const source = ready.getSource(FLOW_SOURCE) as GeoJSONSource | undefined;
    if (source) {
      source.setData(arrows);
      return;
    }
    ready.addSource(FLOW_SOURCE, { type: "geojson", data: arrows });
    for (const [id, colour, extra] of [
      [FLOW_CASING, ARROW_CASING, 1.1],
      [FLOW_ARROWS, ARROW_INK, 0],
    ] as const) {
      addInOrder(ready, {
        id,
        type: "line",
        source: FLOW_SOURCE,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": colour,
          "line-width": arrowWidth(extra),
          "line-opacity": ARROW_OPACITY,
        },
      });
    }
  }, [ready, flow.data]);

  // Stale arrows are dimmed rather than dropped: during a drag the previous
  // hour's field is the best answer available, and blanking the layer for
  // 220 ms would strobe. Dimming says "this is not the hour you are on" while
  // keeping the shape of the channel visible.
  useEffect(() => {
    if (!ready || !ready.getLayer(FLOW_ARROWS)) return;
    const opacity = flow.busy ? ARROW_OPACITY_STALE : ARROW_OPACITY;
    ready.setPaintProperty(FLOW_ARROWS, "line-opacity", opacity);
    ready.setPaintProperty(FLOW_CASING, "line-opacity", opacity);
  }, [ready, flow.busy, flow.data]);

  // --- the popover's anchor: reprojected on every map move ------------------
  // The card is a DOM overlay, so nothing keeps it over its marker while the
  // map pans or zooms except projecting the marker's lng/lat again. Closing
  // the popover on the first pan would be the cheaper answer and the wrong
  // one: comparing two features means moving between them.
  useEffect(() => {
    if (!ready || !picked) {
      setAnchor(null);
      return;
    }
    const update = () => {
      const point = ready.project([picked.lng, picked.lat]);
      const box = ready.getContainer();
      // `.map` clips its overflow, so the card takes whichever side of the
      // marker has more room AND is told how much room that is. Choosing the
      // side by a fixed fraction of the height (the first version of this)
      // put a 520px card in 438px of space and the chart quietly ate its
      // header -- measured in the running app.
      const above = point.y - POPOVER_GAP * 2;
      const below = box.clientHeight - point.y - POPOVER_GAP * 2;
      setAnchor({
        x: point.x,
        y: point.y,
        vertical: above >= below ? "above" : "below",
        room: Math.max(Math.max(above, below), POPOVER_MIN),
        horizontal:
          point.x < POPOVER_MARGIN
            ? "start"
            : point.x > box.clientWidth - POPOVER_MARGIN
              ? "end"
              : "center",
      });
    };
    update();
    ready.on("move", update);
    ready.on("resize", update);
    return () => {
      ready.off("move", update);
      ready.off("resize", update);
    };
  }, [ready, picked]);

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
      {picked && anchor && (
        <div
          className="popover-anchor"
          data-testid="popover-anchor"
          data-vertical={anchor.vertical}
          data-horizontal={anchor.horizontal}
          style={{
            insetInlineStart: `${anchor.x}px`,
            insetBlockStart: `${anchor.y}px`,
            // The card reads this as its ceiling; past it, it scrolls.
            "--popover-room": `${Math.round(anchor.room)}px`,
          } as CSSProperties}
        >
          <FeaturePopover featureKey={picked.key} onClose={() => setPicked(null)} />
        </div>
      )}
      <div className="overlays" data-testid="overlays">
        <p className="eyebrow">Overlays</p>
        <OverlayRow overlay={flow} name="Current arrows">
          <ArrowScale />
        </OverlayRow>
        <OverlayRow overlay={salinity} name="Salinity section">
          {salinity.data && <SalinityInset section={salinity.data} busy={salinity.busy} />}
        </OverlayRow>
      </div>
      <figure className="key" data-testid="map-key">
        <figcaption className="eyebrow">
          Activation
          <span className="key-scope">
            {species ? species.replace(/_/g, " ") : "—"} ·{" "}
            {String(hour).padStart(2, "0")}:00
          </span>
        </figcaption>
        <div className="key-ramp" style={{ background: KEY_RAMP }} />
        <div className="key-ticks">
          <span>0</span>
          <span>50</span>
          <span>100</span>
        </div>
        <ul className="key-notes">
          <li>
            <i className="dot dot-ghost" style={{ background: UNSCORED_COLOR }} />
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
