/**
 * The day, all at once: 24 score bars over a tide curve, with a playhead.
 *
 * Form (dataviz): two single-series panels stacked on ONE shared x-axis --
 * columns for the score, a line for the tide -- never one plot with two
 * y-scales. Score is 0-100 and tide is feet; overlaying them would fix an
 * arbitrary alignment between the two scales and invent a correlation the
 * data does not contain. Sharing the x and separating the y is the sanctioned
 * form for "two measures, same time base", and it is also the honest one:
 * the reader compares SHAPES against a common clock, which is the actual
 * question (does the bite follow the tide?), and never reads a false
 * crossing.
 *
 * Colour (dataviz, validated -- not eyeballed). Two series, so two hues, run
 * through `scripts/validate_palette.js` against this app's panel surface
 * #0a1e29 in dark mode:
 *
 *   node validate_palette.js "#b8822f,#2f9ac6" --mode dark \
 *       --surface "#0a1e29" --pairs all      -> ALL CHECKS PASS
 *   node validate_palette.js "#b8822f,#ffc247" --mode dark \
 *       --surface "#0a1e29" --ordinal        -> ALL CHECKS PASS
 *
 * The first run is the two series (amber score, cyan tide): worst pair CVD
 * dE 20.0 protan, 22.6 normal-vision, both clear of the 8/15 floors, both
 * inside the dark lightness band and over the chroma floor and 3:1 on the
 * surface. The second is the selected bar, which is not a third series but
 * an ORDINAL step of the bars' own hue up to the app's one accent -- the
 * same `--hot` that means "top of the activation ramp" on the map -- so it
 * is validated as a ramp, not as a category.
 *
 * The bars deliberately do NOT wear the map's activation ramp. That ramp
 * encodes per-FEATURE activation; these bars are the species' hour score.
 * Painting them with the same ramp would assert the two numbers are the same
 * quantity, which they are not -- and it would spend the hue channel
 * re-encoding what bar height already says, leaving nothing to carry the one
 * fact height cannot: which hours are only partly observed.
 */
import { useCallback, useMemo, useRef, useState } from "react";

import { useDay } from "../state/DayContext";
import "./HourStrip.css";

import type { PointerEvent as ReactPointerEvent, KeyboardEvent as ReactKeyboardEvent } from "react";

const HOURS = 24;
/** Not all 24: a tick per hour is a picket fence. The reader gets the rest
 *  from the readout, which follows the pointer and the keyboard alike. */
const AXIS_TICKS = [0, 6, 12, 18, 23];
const PAGE_STEP = 6;

/** One hour's worth of everything the strip draws. */
export interface HourSlot {
  hour: number;
  score: number | null;
  confidence: number | null;
  tide: number | null;
}

function clampHour(value: number): number {
  return Math.min(HOURS - 1, Math.max(0, value));
}

function isReading(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * The runs of consecutive hours that actually carry a tide reading.
 *
 * This is the whole "do not interpolate across a gap" rule, expressed as
 * data rather than as drawing code: a null hour ENDS a run, so no path ever
 * spans one. A smooth line through a missing hour would draw a tide reading
 * that was never measured, and this project discloses gaps instead.
 */
export function tideSegments(values: readonly (number | null)[]): number[][] {
  const runs: number[][] = [];
  let run: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    if (isReading(values[i])) {
      run.push(i);
    } else if (run.length > 0) {
      runs.push(run);
      run = [];
    }
  }
  if (run.length > 0) runs.push(run);
  return runs;
}

/**
 * The tide band's y-domain, padded so the extremes are not welded to the
 * band's edges. Null when nothing was measured all day.
 */
export function tideDomain(values: readonly (number | null)[]): [number, number] | null {
  const seen = values.filter(isReading);
  if (seen.length === 0) return null;
  const lo = Math.min(...seen);
  const hi = Math.max(...seen);
  // A flat day (or a single reading) still needs a band with height, or the
  // line lands on a division by zero.
  const pad = hi - lo < 0.2 ? 0.5 : (hi - lo) * 0.12;
  return [lo - pad, hi + pad];
}

const round = (n: number) => Math.round(n * 100) / 100;

/** The tide band is drawn in a 24x100 user space: x is the hour, y the level. */
function project(index: number, value: number, domain: [number, number]): [number, number] {
  const [lo, hi] = domain;
  return [index + 0.5, round(100 - ((value - lo) / (hi - lo)) * 100)];
}

function linePath(points: readonly (readonly [number, number])[]): string {
  const first = points[0];
  if (!first) return "";
  // A lone reading between two gaps is a real measurement and has to show.
  // A zero-length path with a round cap is that point, and it stays round
  // because the stroke is non-scaling -- see the `preserveAspectRatio` note.
  if (points.length === 1) return `M ${first[0]} ${first[1]} L ${first[0]} ${first[1]}`;
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
}

function areaPath(points: readonly (readonly [number, number])[]): string | null {
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last || points.length < 2) return null;
  const ridge = points.map(([x, y]) => `${x} ${y}`).join(" L ");
  return `M ${first[0]} 100 L ${ridge} L ${last[0]} 100 Z`;
}

const pct = (n: number) => `${round(n * 100)}%`;

export function HourStrip() {
  const { payload, species, hour, setHour } = useDay();
  const plotRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const [hovered, setHovered] = useState<number | null>(null);

  const slots = useMemo<HourSlot[]>(() => {
    const block = species ? payload?.species[species] : undefined;
    return Array.from({ length: HOURS }, (_, i) => {
      // `noUncheckedIndexedAccess` is doing real work here: a payload whose
      // arrays are short renders empty hours rather than throwing, and the
      // strip keeps its 24 columns either way.
      const scored = block?.hours[i];
      const conditions = payload?.conditions[i];
      return {
        hour: i,
        score: isReading(scored?.score) ? scored.score : null,
        confidence: isReading(scored?.confidence) ? scored.confidence : null,
        tide: isReading(conditions?.tide_height_ft) ? conditions.tide_height_ft : null,
      };
    });
  }, [payload, species]);

  const tides = useMemo(() => slots.map((slot) => slot.tide), [slots]);
  const domain = useMemo(() => tideDomain(tides), [tides]);
  const segments = useMemo(() => tideSegments(tides), [tides]);
  /**
   * The best hour of the day, and the only bar that gets a direct label.
   *
   * The bars keep a true zero baseline, which is non-negotiable for columns
   * -- and on a day whose scores span 62-78 that draws an almost flat row.
   * The row is CORRECT; it is just quiet about the question a person opened
   * this for. Labelling the extreme is the sanctioned fix (never a number on
   * every bar), and it costs the axis nothing. First index wins a tie: the
   * earlier of two equal hours is the one you can still get to.
   */
  const peak = useMemo(() => {
    let best: number | null = null;
    for (const slot of slots) {
      if (slot.score === null) continue;
      if (best === null || slot.score > (slots[best]?.score ?? -1)) best = slot.hour;
    }
    return best;
  }, [slots]);

  /** What was actually measured, unpadded -- the key states real readings. */
  const observed = useMemo<[number, number] | null>(() => {
    const seen = tides.filter(isReading);
    return seen.length === 0 ? null : [Math.min(...seen), Math.max(...seen)];
  }, [tides]);
  const gaps = useMemo(
    () => slots.filter((slot) => slot.tide === null).map((slot) => slot.hour),
    [slots],
  );
  const partial = useMemo(
    () => slots.filter((slot) => slot.confidence !== null && slot.confidence < 1).length,
    [slots],
  );

  const hasData = payload !== null && species !== "";
  const shown = slots[hovered ?? hour] ?? slots[0]!;

  const scrub = useCallback(
    (next: number) => {
      const clamped = clampHour(next);
      // The clamp is what makes 23 -> ArrowRight a no-op instead of a wrap to
      // 00:00, and the equality guard is what keeps a drag from firing a
      // state set per pixel.
      if (clamped !== hour) setHour(clamped);
    },
    [hour, setHour],
  );

  /** Which hour the pointer is over. Null when the box has no width yet. */
  const hourAt = useCallback((clientX: number): number | null => {
    const element = plotRef.current;
    if (!element) return null;
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0) return null;
    return clampHour(Math.floor(((clientX - rect.left) / rect.width) * HOURS));
  }, []);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!hasData || event.button !== 0) return;
      draggingRef.current = true;
      // jsdom has PointerEvent but not pointer capture, and a browser without
      // it still works -- it just stops tracking if the pointer leaves the
      // strip mid-drag.
      const target = event.currentTarget;
      if (typeof target.setPointerCapture === "function") {
        try {
          target.setPointerCapture(event.pointerId);
        } catch {
          /* capture is an optimisation, never a requirement */
        }
      }
      const next = hourAt(event.clientX);
      if (next !== null) scrub(next);
    },
    [hasData, hourAt, scrub],
  );

  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const next = hourAt(event.clientX);
      if (draggingRef.current) {
        if (next !== null) scrub(next);
        return;
      }
      // Hover feeds the readout. Keyboard focus feeds the same readout, so a
      // value is never reachable by pointer alone.
      setHovered((current) => (current === next ? current : next));
    },
    [hourAt, scrub],
  );

  const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    draggingRef.current = false;
    const target = event.currentTarget;
    if (typeof target.releasePointerCapture === "function") {
      try {
        target.releasePointerCapture(event.pointerId);
      } catch {
        /* nothing was captured */
      }
    }
  }, []);

  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (!hasData) return;
      let next: number;
      switch (event.key) {
        case "ArrowRight":
        case "ArrowUp":
          next = hour + 1;
          break;
        case "ArrowLeft":
        case "ArrowDown":
          next = hour - 1;
          break;
        case "PageUp":
          next = hour + PAGE_STEP;
          break;
        case "PageDown":
          next = hour - PAGE_STEP;
          break;
        case "Home":
          next = 0;
          break;
        case "End":
          next = HOURS - 1;
          break;
        default:
          return;
      }
      event.preventDefault();
      scrub(next);
    },
    [hasData, hour, scrub],
  );

  return (
    <section className="strip" data-testid="hour-strip" aria-label="The day, hour by hour">
      <header className="strip-head">
        <p className="eyebrow">
          Bite score{species ? ` · ${species.replace(/_/g, " ")}` : ""}
          <span className="strip-scale num">0–100</span>
        </p>
        <p className="strip-readout" data-testid="strip-readout">
          <span className="num">{String(shown.hour).padStart(2, "0")}:00</span>
          <span className="strip-sep" aria-hidden="true">
            ·
          </span>
          <span className="num">{shown.score === null ? "—" : shown.score}</span>
          <span className="strip-unit">score</span>
          <span className="strip-sep" aria-hidden="true">
            ·
          </span>
          <span className="num">{shown.tide === null ? "—" : shown.tide.toFixed(2)}</span>
          <span className="strip-unit">ft tide</span>
          {shown.confidence !== null && shown.confidence < 1 && (
            <span className="strip-flag">
              {Math.round(shown.confidence * 100)}% of factors resolved
            </span>
          )}
        </p>
      </header>

      <div
        className="strip-plot"
        data-testid="strip-plot"
        data-active={hasData}
        ref={plotRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={(event) => {
          endDrag(event);
          setHovered(null);
        }}
      >
        {/* Score, 0-100. No gridlines and no tick column: direct labels come
            before gridlines, the band's ceiling is named in the header, and
            every individual value is in the readout above (which follows
            keyboard focus as well as the pointer) and in each bar's own
            aria-label. A tick gutter would also have to sit INSIDE this box,
            where it would offset the hour grid from the pointer maths. */}
        <div className="strip-band strip-band-score">
          <div
            className="strip-bars"
            data-testid="hour-bars"
            role="radiogroup"
            aria-label="Hour of day"
            onKeyDown={onKeyDown}
          >
            {slots.map((slot) => {
              const selected = slot.hour === hour;
              const isPartial = slot.confidence !== null && slot.confidence < 1;
              return (
                <button
                  key={slot.hour}
                  type="button"
                  role="radio"
                  data-testid="hour-bar"
                  className="hour-bar"
                  aria-checked={selected}
                  data-selected={selected}
                  data-confidence={isPartial ? "partial" : "full"}
                  // APG's roving tabindex: one stop for the whole strip, and
                  // the arrow keys move within it.
                  tabIndex={selected ? 0 : -1}
                  disabled={!hasData}
                  aria-label={
                    `${String(slot.hour).padStart(2, "0")}:00 — ` +
                    (slot.score === null ? "not scored" : `score ${slot.score}`) +
                    (isPartial && slot.confidence !== null
                      ? `, ${Math.round(slot.confidence * 100)}% of factors resolved`
                      : "") +
                    (slot.tide === null
                      ? ", no tide reading"
                      : `, tide ${slot.tide.toFixed(2)} ft`)
                  }
                  onClick={() => scrub(slot.hour)}
                  onFocus={() => setHovered(slot.hour)}
                >
                  <span
                    className="hour-bar-fill"
                    style={{ blockSize: `${slot.score ?? 0}%` }}
                  >
                    {/* ...unless the playhead is parked on it: measured in
                        the running app, the grip and its rule cut straight
                        through the label, and the readout above is already
                        printing that exact score. */}
                    {slot.hour === peak && slot.score !== null && !selected && (
                      <span className="hour-bar-peak num" data-testid="hour-peak">
                        {slot.score}
                      </span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Tide, in feet. Its own band and its own scale -- see the header
            note on why this is not underlaid on the bars' axis. */}
        <div className="strip-band strip-band-tide">
          <svg
            className="strip-tide"
            // The band stretches with the window while the hour grid stays
            // exactly 24 wide, which is what keeps the curve's x aligned to
            // the bars' x. `non-scaling-stroke` below is the price: without
            // it the stretch would smear the line's width.
            viewBox={`0 0 ${HOURS} 100`}
            preserveAspectRatio="none"
            aria-hidden="true"
            focusable="false"
          >
            {domain &&
              segments.map((run) => {
                const points = run.map((i) => project(i, tides[i] as number, domain));
                const area = areaPath(points);
                const label = `${run[0]}-${run[run.length - 1]}`;
                return (
                  <g key={label}>
                    {area && <path className="strip-tide-area" d={area} />}
                    <path
                      className="strip-tide-line"
                      data-testid="tide-segment"
                      data-hours={label}
                      d={linePath(points)}
                    />
                  </g>
                );
              })}
          </svg>
          {gaps.map((index) => (
            <span
              key={index}
              className="strip-tide-gap"
              data-testid="tide-gap"
              data-hour={index}
              title={`${String(index).padStart(2, "0")}:00 — no tide reading`}
              style={{ insetInlineStart: pct(index / HOURS), inlineSize: pct(1 / HOURS) }}
            />
          ))}
        </div>

        <div
          className="strip-playhead"
          data-testid="playhead"
          data-hour={hour}
          aria-hidden="true"
          style={{ insetInlineStart: pct((hour + 0.5) / HOURS) }}
        >
          <span className="strip-playhead-grip" />
        </div>
      </div>

      <footer className="strip-foot">
        <div className="strip-axis">
          {AXIS_TICKS.map((tick) => (
            <span
              key={tick}
              className="strip-axis-tick num"
              style={{ insetInlineStart: pct((tick + 0.5) / HOURS) }}
            >
              {String(tick).padStart(2, "0")}
            </span>
          ))}
        </div>
        <ul className="strip-key">
          {/* The tide's key is permanent: it is the second series, and it is
              the only place its colour and its scale are named. */}
          <li>
            <i className="strip-swatch strip-swatch-tide" />
            Tide
            {observed && (
              <span className="num">
                {observed[0].toFixed(2)}–{observed[1].toFixed(2)} ft
              </span>
            )}
          </li>
          {/* The two disclosure rows admit only what this day actually
              contains. A row for a condition no hour is in would be noise;
              one that stayed silent when the condition IS present would be
              the omission this project exists not to make. */}
          {partial > 0 && (
            <li>
              <i className="strip-swatch strip-swatch-partial" />
              {partial} {partial === 1 ? "hour" : "hours"} scored with some factors
              unresolved
            </li>
          )}
          {gaps.length > 0 && (
            <li>
              <i className="strip-swatch strip-swatch-gap" />
              {gaps.length} {gaps.length === 1 ? "hour" : "hours"} with no tide reading —
              the curve breaks rather than guessing across it
            </li>
          )}
        </ul>
      </footer>
    </section>
  );
}

export default HourStrip;
