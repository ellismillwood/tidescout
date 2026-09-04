/**
 * The two optional overlays: the current field, and the salinity section.
 *
 * Both are OFF by default and both are per-hour requests, which makes them the
 * one deliberate exception to "scrubbing never refetches" (spec §3). The
 * exception was accepted rather than prefetching 24 hours x 2 overlays into
 * the day payload, which would have undone a 49% payload reduction -- so the
 * cost is paid here instead, as a debounce: a fast drag issues one request
 * when it comes to rest, not one per frame.
 *
 * ------------------------------------------------------------------------
 * THE SALINITY LAYER CANNOT BE DRAWN AS A SMOOTH FIELD, AND THAT IS ENFORCED
 * BY THE SHAPE OF WHAT THIS MODULE RETURNS, NOT BY THE STYLESHEET.
 *
 * Winyah Bay's salinity fit is FALSIFIED: `fitted: false`, with a residual
 * roughly 1,159x the resolution of the observations it was fit against. Spec
 * §1.1 ratified "flagged, not discounted" -- the layer IS drawn, because
 * dropping it would be the discounting that decision rejected -- but it must
 * never be mistakable for measurement.
 *
 * A stylesheet can be edited. A type cannot be edited by accident. So
 * `toSalinitySection` QUANTISES on the way in and the continuous `ppt` values
 * never leave this module: `SalinityBand` carries an integer class index and
 * nothing else. There is no exported function anywhere in this codebase that
 * maps a salinity to a colour, an opacity or a coordinate, because there is no
 * continuous salinity to map. A future renderer that wanted a smooth gradient,
 * or a crisp isoline, would have no numbers to build one from -- it would have
 * to reach past `fetchSalinityField` to the raw response first, which is a
 * deliberate act rather than a styling slip.
 *
 * `badge` is the same idea applied to the disclosure: it is a non-optional
 * field of a union of three non-empty string literals, so no code path
 * produces a section without one, and TypeScript rejects "".
 * ------------------------------------------------------------------------
 */
import { useCallback, useEffect, useRef, useState } from "react";

// --- the request both overlays take --------------------------------------

export interface OverlayRequest {
  slug: string;
  date: string;
  hour: number;
  model: string;
}

/** The API's own error envelope, same as `api/client.ts` reads. */
async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

function overlayUrl(kind: string, { slug, date, hour, model }: OverlayRequest): string {
  return (
    `/api/fisheries/${encodeURIComponent(slug)}/${kind}/${encodeURIComponent(date)}` +
    `?hour=${hour}&model=${encodeURIComponent(model)}`
  );
}

// --- the current field ---------------------------------------------------

/**
 * `u`/`v` are row-major over `rows` x `cols`, in metres per second in the
 * flow library's own projected CRS (`xmomentum / depth` at the ANUGA
 * centroids -- see `pipeline/regimes._centroid_speed`). Dry cells and cells
 * outside the model domain are exactly 0.
 *
 * `bbox` is `[west, south, east, north]` in WGS84 degrees and describes the
 * extent of THE GRID: cell (0, 0) is the north-west corner. Row 0 is north,
 * matching the raster the library scatters back onto (confirmed against
 * `GridSpec`: the row holding `ys.max()` is the lowest row index).
 */
export interface FlowField {
  hour: number;
  rows: number;
  cols: number;
  bbox: [number, number, number, number];
  u: number[];
  v: number[];
}

export async function fetchFlowVectors(
  request: OverlayRequest,
  signal?: AbortSignal,
): Promise<FlowField> {
  return json<FlowField>(await fetch(overlayUrl("flow-vectors", request), { signal }));
}

/**
 * The speed a full-length arrow means, in m/s.
 *
 * FIXED, not a per-hour percentile. Comparing a flood hour with an ebb hour is
 * the whole reason someone turns this on, and a scale renormalised per hour
 * would draw slack water at the same length as peak ebb -- the one comparison
 * the overlay exists to support, silently destroyed. Winyah's blended library
 * peaks near 0.40 m/s, so 0.35 puts the strongest arrows at full length
 * without clipping most of the field to the cap.
 */
export const REFERENCE_SPEED_MS = 0.35;

/** m/s to knots, for the reference arrow's label. */
export const KNOTS_PER_MS = 1.943_844;

/**
 * Below this a "direction" is numerical noise, not a current. Dry and
 * out-of-domain cells are exactly zero, so this only trims near-slack water.
 */
const SPEED_FLOOR_MS = 0.005;

/** Even the slowest drawn arrow has to be long enough to read a heading off. */
const MIN_ARROW_FRACTION = 0.2;

/** Of the cell spacing. Leaves a hairline gutter so a dense field stays legible. */
const MAX_ARROW_FRACTION = 0.9;

/** Barb half-angle from the shaft, and barb length as a fraction of the shaft. */
const BARB_DEGREES = 148;
const BARB_FRACTION = 0.34;

const M_PER_DEG_LAT = 110_574;
const M_PER_DEG_LNG_EQUATOR = 111_320;

export interface ArrowFeature {
  type: "Feature";
  properties: { speed: number };
  geometry: { type: "MultiLineString"; coordinates: [number, number][][] };
}

export interface ArrowCollection {
  type: "FeatureCollection";
  features: ArrowFeature[];
}

function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/**
 * The field as drawable arrows: a shaft and a two-stroke barb per wet cell.
 *
 * A line layer rather than a rotated symbol on purpose. A symbol layer needs
 * either a sprite (the API serves none) or a glyph the label font is not
 * guaranteed to carry, and a missing arrow glyph fails the way MapLibre
 * failures usually do here -- silently, as a layer that draws nothing. Two
 * polylines per cell need neither, and they are the mark a paper chart
 * actually prints for a current.
 *
 * Length carries speed and so does line width, which is the redundancy that
 * keeps the field readable where arrows overlap: a short fat arrow and a long
 * thin one never mean the same thing.
 */
export function arrowCollection(field: FlowField): ArrowCollection {
  const { rows, cols, u, v } = field;
  const [west, south, east, north] = field.bbox;
  const features: ArrowFeature[] = [];
  if (rows < 2 || cols < 2) return { type: "FeatureCollection", features };

  const stepLng = (east - west) / (cols - 1);
  const stepLat = (north - south) / (rows - 1);

  for (let row = 0; row < rows; row += 1) {
    const lat = north - row * stepLat;
    const mPerDegLng = M_PER_DEG_LNG_EQUATOR * Math.cos((lat * Math.PI) / 180);
    if (mPerDegLng <= 0) continue;
    // The cell is a rectangle in degrees but a near-square in metres; the
    // shorter side is what an arrow must not overrun.
    const span = Math.min(Math.abs(stepLng) * mPerDegLng, Math.abs(stepLat) * M_PER_DEG_LAT);
    for (let col = 0; col < cols; col += 1) {
      const at = row * cols + col;
      const east_ = u[at];
      const north_ = v[at];
      if (east_ === undefined || north_ === undefined) continue;
      const speed = Math.hypot(east_, north_);
      if (speed < SPEED_FLOOR_MS) continue;

      const scale = Math.min(speed / REFERENCE_SPEED_MS, 1);
      const length =
        span * MAX_ARROW_FRACTION * (MIN_ARROW_FRACTION + (1 - MIN_ARROW_FRACTION) * scale);
      const dirX = east_ / speed;
      const dirY = north_ / speed;

      const lng = west + col * stepLng;
      // Metres to degrees at THIS latitude, so an arrow's drawn heading is its
      // real heading. Dividing both components by the same constant would
      // shear every arrow toward east-west by ~1.2 at Winyah's latitude.
      const toLng = (metres: number) => metres / mPerDegLng;
      const toLat = (metres: number) => metres / M_PER_DEG_LAT;
      const point = (mx: number, my: number): [number, number] => [
        round6(lng + toLng(mx)),
        round6(lat + toLat(my)),
      ];

      const half = length / 2;
      const tail = point(-dirX * half, -dirY * half);
      const tip = point(dirX * half, dirY * half);
      const barbLength = length * BARB_FRACTION;
      const barbs: [number, number][] = [];
      for (const sign of [1, -1]) {
        const angle = (sign * BARB_DEGREES * Math.PI) / 180;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        barbs.push(
          point(
            dirX * half + (dirX * cos - dirY * sin) * barbLength,
            dirY * half + (dirX * sin + dirY * cos) * barbLength,
          ),
        );
      }
      const [left, right] = barbs;
      if (!left || !right) continue;

      features.push({
        type: "Feature",
        properties: { speed },
        geometry: { type: "MultiLineString", coordinates: [[tail, tip], [left, tip, right]] },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

// --- the salinity section ------------------------------------------------

/** What the endpoint sends. Never handed on: `toSalinitySection` quantises it. */
interface RawSalinityField {
  hour: number;
  fitted: boolean;
  extrapolated: boolean;
  cells: { km: number; ppt: number }[];
}

/**
 * The width of one salinity class, in ppt.
 *
 * Coarse deliberately. This is the resolution the drawing claims, and the
 * model behind it is falsified -- a finer step would be a finer claim.
 */
export const SALINITY_CLASS_PPT = 2.5;

/** 0 to 37.5 ppt: fresh river water through fully oceanic, with headroom. */
export const SALINITY_CLASS_COUNT = 15;

/**
 * One kilometre bin of the along-estuary section.
 *
 * `ppt` IS ABSENT ON PURPOSE. See the module header: the class index is the
 * only salinity that leaves this module, so nothing downstream has the
 * numbers a smooth field or an isoline would need.
 */
export interface SalinityBand {
  readonly km: number;
  /** Integer, 0 to SALINITY_CLASS_COUNT - 1. Class n spans [n, n+1) * 2.5 ppt. */
  readonly klass: number;
}

/**
 * Three words, none of them empty, and the field carrying one is required.
 * "UNCALIBRATED" is Winyah's permanent state and outranks extrapolation: a
 * model no observation constrains cannot be said to be inside or outside its
 * fitted range in any meaningful way.
 */
export type SalinityBadge = "UNCALIBRATED" | "EXTRAPOLATED" | "MODELLED";

export interface SalinitySection {
  readonly hour: number;
  readonly fitted: boolean;
  readonly extrapolated: boolean;
  readonly badge: SalinityBadge;
  readonly bands: readonly SalinityBand[];
  readonly kmMin: number;
  readonly kmMax: number;
}

export function badgeFor(fitted: boolean, extrapolated: boolean): SalinityBadge {
  if (!fitted) return "UNCALIBRATED";
  return extrapolated ? "EXTRAPOLATED" : "MODELLED";
}

/** The ppt range class `klass` stands for, for a label. Never for geometry. */
export function classRange(klass: number): [number, number] {
  return [klass * SALINITY_CLASS_PPT, (klass + 1) * SALINITY_CLASS_PPT];
}

function toSalinitySection(raw: RawSalinityField): SalinitySection {
  const bands: SalinityBand[] = [];
  let kmMin = Infinity;
  let kmMax = -Infinity;
  for (const cell of raw.cells) {
    if (!Number.isFinite(cell.km) || !Number.isFinite(cell.ppt)) continue;
    const klass = Math.min(
      SALINITY_CLASS_COUNT - 1,
      Math.max(0, Math.floor(cell.ppt / SALINITY_CLASS_PPT)),
    );
    bands.push({ km: cell.km, klass });
    if (cell.km < kmMin) kmMin = cell.km;
    if (cell.km > kmMax) kmMax = cell.km;
  }
  if (bands.length === 0) throw new Error("salinity field has no cells to draw");
  return {
    hour: raw.hour,
    fitted: raw.fitted,
    extrapolated: raw.extrapolated,
    badge: badgeFor(raw.fitted, raw.extrapolated),
    bands,
    kmMin,
    kmMax,
  };
}

/**
 * The only way to get a salinity section. There is no exported path from the
 * response to a continuous ppt, which is what makes the smooth field the
 * module header forbids unreachable rather than merely unfashionable.
 */
export async function fetchSalinityField(
  request: OverlayRequest,
  signal?: AbortSignal,
): Promise<SalinitySection> {
  const raw = await json<RawSalinityField>(
    await fetch(overlayUrl("salinity-field", request), { signal }),
  );
  return toSalinitySection(raw);
}

// --- the toggle, the debounce and the revert -----------------------------

/**
 * Long enough that a drag across the strip settles into one request, short
 * enough that a single arrow-key step feels like a direct answer. Measured
 * against the endpoints at ~1.5s each, so a per-frame refetch would queue
 * dozens of requests behind a one-second drag.
 */
export const OVERLAY_DEBOUNCE_MS = 220;

/**
 * `value`, but only after it has held still for `ms`.
 *
 * Seeded with `value`, so switching an overlay ON is answered immediately for
 * the hour already selected -- the debounce is for CHANGES, and making the
 * first paint wait would read as a broken toggle.
 */
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    // Each new `value` REPLACES the pending timer rather than queueing behind
    // it, which is the whole mechanism: 24 hour changes in a drag schedule 24
    // timers and cancel 23, leaving one request carrying the LAST hour.
    const id = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return settled;
}

export interface Overlay<T> {
  /** Whether the layer is asked for. False on mount, and false after a failure. */
  on: boolean;
  /** The field on screen, or null. Survives an hour change; not a day change. */
  data: T | null;
  /** `data` is not the hour the strip is on -- either absent, or one behind. */
  busy: boolean;
  /** Why the toggle reverted. Cleared when it is switched on again. */
  error: string | null;
  enable: (on: boolean) => void;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** What arrived, and which request it answers. */
interface Loaded<T> {
  day: string;
  hour: number;
  value: T;
}

/**
 * One optional overlay: its toggle, its debounced per-hour fetch, and the
 * revert that a failure owes it.
 *
 * The toggle lives HERE rather than at the call site because reverting it is
 * the failure path's job -- spec §7 requires a failed optional layer to leave
 * the base map untouched, and a toggle that stayed on after its fetch died
 * would sit there claiming a layer nobody can see. Owning both means the
 * revert cannot be forgotten by a caller.
 *
 * `data` survives an hour change but not a day, model or fishery change: a
 * neighbouring hour's field is a fair thing to keep on screen for 220 ms while
 * the next one loads (and `busy` says so), whereas another fishery's field is
 * a different place entirely.
 *
 * ONE piece of state, holding what arrived and which request it answers;
 * `data` and `busy` are DERIVED from it during render. The obvious shape --
 * three useStates kept in step by effects that null one and flag another --
 * has a window on every day change where `data` is last week's field and
 * `busy` is false, i.e. where the panel says "current" over something that is
 * not. Deriving both from one record closes it by construction, and it is
 * also why `busy` compares against `request.hour` rather than the debounced
 * one: what is on screen either IS the hour the strip is on or it is not, and
 * the 220 ms of deliberate delay is part of "is not".
 */
export function useOverlay<T>(
  fetcher: (request: OverlayRequest, signal?: AbortSignal) => Promise<T>,
  request: OverlayRequest,
  debounceMs: number = OVERLAY_DEBOUNCE_MS,
): Overlay<T> {
  const [on, setOn] = useState(false);
  const [loaded, setLoaded] = useState<Loaded<T> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hour = useDebounced(request.hour, debounceMs);
  const { slug, date, model } = request;
  const day = `${slug} ${date} ${model}`;

  // The fetcher is typically a module function, but taking it as an argument
  // means a test can pass its own. Held in a ref so a caller that passes an
  // inline lambda does not re-fetch on every render.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  }, [fetcher]);

  useEffect(() => {
    if (!on) return;
    const controller = new AbortController();
    let cancelled = false;
    fetcherRef
      .current({ slug, date, hour, model }, controller.signal)
      .then((value) => {
        if (cancelled) return;
        setLoaded({ day, hour, value });
      })
      .catch((err: unknown) => {
        if (cancelled || controller.signal.aborted) return;
        // Revert. The layer is optional; the chart under it is not.
        setOn(false);
        setLoaded(null);
        setError(message(err));
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [on, day, slug, date, hour, model]);

  const enable = useCallback((next: boolean) => {
    setOn(next);
    setError(null);
  }, []);

  const fresh = loaded !== null && loaded.day === day;
  return {
    on,
    data: on && fresh ? loaded.value : null,
    busy: on && !(fresh && loaded.hour === request.hour),
    error,
    enable,
  };
}
