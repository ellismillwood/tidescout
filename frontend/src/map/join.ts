import type { DayPayload } from "../api/types";

/**
 * The MapLibre property name holding one species' activation at one hour.
 *
 * Species ids contain underscores (`speckled_trout`), so `a_speckled_trout_11`
 * cannot be split back apart unambiguously. These keys are only ever
 * CONSTRUCTED, never parsed -- species and hour are always known from state.
 */
export function activationKey(species: string, hour: number): string {
  return `a_${species}_${hour}`;
}

type Feature = {
  type: "Feature";
  id?: string | number;
  properties: Record<string, unknown>;
  geometry: unknown;
};
type FeatureCollection = { type: "FeatureCollection"; features: Feature[] };

/**
 * Bake every hour's activation onto the feature geometry, once, at load.
 *
 * This is what makes scrubbing free: with 529 x 3 x 24 activations already on
 * the features, changing the hour is a paint-expression swap rather than 529
 * JavaScript writes per frame. See spec §4.3 for the rejected alternatives.
 *
 * An unscored feature gets NO `a_*` properties -- not zeroes. The map's muted
 * styling branch keys off their absence (§4.4), so writing defaults here would
 * silently render 1,633 undomained features as genuinely dead water.
 *
 * The join key is the GeoJSON Feature's top-level `id`, NOT a
 * `properties.key` field -- `data/winyah-bay/features.geojson` carries no such
 * property (verified against the real 8 MB file: properties are area_m2,
 * max_slope_deg, max_z, mean_slope_deg, min_z, orientation_deg,
 * oyster_area_m2, oyster_density, oyster_nearest_m, p90_slope_deg, type).
 * The payload's `species[x].features` dict is keyed by that same top-level
 * `id` -- see backend/tidescout/engine/activation.py, which builds
 * `FeatureMetrics(key=f["id"], ...)`.
 */
export function joinActivations(
  geojson: FeatureCollection,
  payload: DayPayload,
): FeatureCollection {
  const bySpecies = Object.entries(payload.species);
  return {
    type: "FeatureCollection",
    features: geojson.features.map((feature) => {
      const key = feature.id;
      const extra: Record<string, number> = {};
      if (typeof key === "string") {
        for (const [species, blob] of bySpecies) {
          const block = blob.features[key];
          if (!block) continue;
          block.hours.forEach((fh, hour) => {
            extra[activationKey(species, hour)] = fh.activation;
          });
        }
      }
      return { ...feature, properties: { ...feature.properties, ...extra } };
    }),
  };
}
