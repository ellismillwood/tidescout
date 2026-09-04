/** Formatting shared by the rail and the popover. Nothing here invents data. */

/** A number the payload actually carries -- `null` and NaN are not readings. */
export function isReading(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** A 0..1 fraction as a whole percent, or `null` if there is no reading. */
export function percent(value: number | null | undefined): string | null {
  return isReading(value) ? `${Math.round(value * 100)}%` : null;
}

/** A fixed-decimal reading, or `null` -- never a 0 standing in for "unknown". */
export function fixed(value: number | null | undefined, digits: number): string | null {
  return isReading(value) ? value.toFixed(digits) : null;
}

const ISO_CLOCK = /^\d{4}-\d{2}-\d{2}T(\d{2}:\d{2})/;

/**
 * The wall clock out of a payload timestamp, e.g. "06:52".
 *
 * Read out of the STRING, deliberately, and not through `Date`. Every time in
 * the payload is already in the fishery's own zone and carries its offset
 * ("2026-09-03T06:52:48.458034-04:00"); parsing it to a `Date` and formatting
 * that would re-render it in the BROWSER's zone, so a person checking Winyah
 * Bay's sunrise from a laptop still on Pacific time would be told 03:52. The
 * fishery's local time is the only time this app means.
 */
export function clock(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const match = ISO_CLOCK.exec(iso);
  return match ? (match[1] ?? null) : null;
}

const COMPASS = [
  "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
] as const;

/** Meteorological degrees (the direction wind comes FROM) to a 16-point name. */
export function compass(deg: number | null | undefined): string | null {
  if (!isReading(deg)) return null;
  const index = Math.round((((deg % 360) + 360) % 360) / 22.5) % 16;
  return COMPASS[index] ?? null;
}

/** A signed trend, so "+0.1" and "-0.1" both read as a direction. */
export function signed(value: number | null | undefined, digits: number): string | null {
  if (!isReading(value)) return null;
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return `±${text}`;
  return `${value < 0 ? "−" : "+"}${text}`;
}
