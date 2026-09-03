import { activationKey } from "./join";

export const MARKER_LAYER_ID = "markers";

/**
 * The value an unscored feature's activation coalesces to.
 *
 * Negative on purpose: activations are 0-100, so no scored feature can collide
 * with it, and a `["<", value, 0]` branch cleanly separates "outside the model
 * domain" from "scored, and the answer is zero". Those are different claims.
 */
export const UNSCORED_SENTINEL = -1;

function activation(species: string, hour: number) {
  // `coalesce` is load-bearing: unscored features carry no a_* property, and
  // feeding interpolate a null throws at style-parse time -- which would take
  // down the whole marker layer, scored features included.
  return ["coalesce", ["get", activationKey(species, hour)], UNSCORED_SENTINEL];
}

/** Marker radius from activation. Unscored features render small and flat. */
export function radiusExpr(species: string, hour: number): unknown[] {
  return [
    "case",
    ["<", activation(species, hour), 0],
    3,
    ["interpolate", ["linear"], activation(species, hour), 0, 4, 50, 9, 100, 16],
  ];
}

/** Marker colour from activation. Unscored features render muted grey. */
export function colorExpr(species: string, hour: number): unknown[] {
  return [
    "case",
    ["<", activation(species, hour), 0],
    "#9aa3ad",
    [
      "interpolate",
      ["linear"],
      activation(species, hour),
      0, "#2b3a4a",
      50, "#2f7fb8",
      100, "#f2c14e",
    ],
  ];
}
