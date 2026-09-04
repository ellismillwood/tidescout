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

/**
 * The colour an unscored feature paints, and the ONE place it is written.
 *
 * The key's ghost swatch shows the same colour; letting the stylesheet keep
 * its own copy meant the key could go on describing a mapping the map no
 * longer used.
 */
export const UNSCORED_COLOR = "rgba(154,170,182,0.32)";

/**
 * The activation colour ramp: `[activation 0-100, colour]`, ascending.
 *
 * The single source for both the paint expression below and the key's ramp
 * bar in `MapView`. The bar reads these stops at their real positions, so it
 * shows the ACTUAL mapping -- uneven spacing included -- rather than a
 * tidied version, and changing the ramp cannot leave the key lying about it.
 */
export const ACTIVATION_STOPS: readonly (readonly [number, string])[] = [
  [0, "#5f3168"],
  [35, "#8f356f"],
  [60, "#cf5560"],
  [80, "#ef8a3c"],
  [100, "#ffc247"],
];

function activation(species: string, hour: number) {
  // `coalesce` is load-bearing: unscored features carry no a_* property, and
  // feeding interpolate a null throws at style-parse time -- which would take
  // down the whole marker layer, scored features included.
  return ["coalesce", ["get", activationKey(species, hour)], UNSCORED_SENTINEL];
}

/**
 * Marker radius from activation. Unscored features render as a small ghost.
 *
 * The stops are not evenly spaced, and that is the considered part. Winyah's
 * real activations on 2026-09-03 run 0-87 with the mass between 38 and 78
 * (p25 47, p50 56, p75 65 for redfish), so an evenly spaced 0/50/100 ramp
 * spends most of its dynamic range on values the data never takes and
 * flattens the band a person is actually choosing between. Bunching the
 * stops through 40-80 puts the visual difference where the decision is.
 *
 * The cap is 12px, not 16: at bay zoom 529 markers already crowd, and a
 * 32px-wide hot marker swallows its neighbours instead of standing out
 * among them.
 */
export function radiusExpr(species: string, hour: number): unknown[] {
  return [
    "case",
    ["<", activation(species, hour), 0],
    2,
    [
      "interpolate",
      ["linear"],
      activation(species, hour),
      0, 3.5,
      40, 5,
      60, 7.5,
      80, 10,
      100, 12,
    ],
  ];
}

/**
 * Marker colour from activation. Unscored features render as a grey ghost.
 *
 * The ramp runs aubergine -> mulberry -> coral -> sodium gold, and the choice
 * of hue family is forced by what it sits on. `depth_tint.png` ramps pale
 * cyan (198,232,240) to deep navy (18,62,128), so the water OWNS blue: the
 * placeholder ramp's mid-value #2f7fb8 is within a few units of the tint's
 * 5-metre colour (86,158,204), which would have made a middling marker
 * disappear into middling water. Nothing in this palette but a marker is
 * magenta or gold.
 *
 * Lightness climbs monotonically with activation (L* roughly 30 -> 38 -> 52
 * -> 66 -> 82), so the ramp still reads in order for a colour-blind viewer
 * and in a greyscale screenshot -- the hue is the second channel, not the
 * only one. Stops match `radiusExpr`'s for the same reason they are uneven.
 *
 * The zero end stops at a plum rather than going near-black on purpose. More
 * than 5% of feature-hours score exactly 0, and "scored, and the answer is
 * zero" has to stay tellable from "outside the model domain" at 3 px -- the
 * distinction the sentinel exists to keep. Scored zero is a saturated plum;
 * unscored is a pale translucent grey.
 */
export function colorExpr(species: string, hour: number): unknown[] {
  return [
    "case",
    ["<", activation(species, hour), 0],
    UNSCORED_COLOR,
    [
      "interpolate",
      ["linear"],
      activation(species, hour),
      ...ACTIVATION_STOPS.flatMap(([at, colour]) => [at, colour]),
    ],
  ];
}
