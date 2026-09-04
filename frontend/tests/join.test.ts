import { describe, expect, it } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { DayPayload } from "../src/api/types";
import { activationKey, joinActivations } from "../src/map/join";

const payload = fixture as unknown as DayPayload;

// NOTE: data/winyah-bay/features.geojson carries no `properties.key` field
// (verified against the real 8 MB file: properties are area_m2,
// max_slope_deg, max_z, mean_slope_deg, min_z, orientation_deg,
// oyster_area_m2, oyster_density, oyster_nearest_m, p90_slope_deg, type). The
// join key is the GeoJSON Feature's top-level `id`, which is what the
// payload's species[x].features dict is keyed by -- see
// backend/tidescout/engine/activation.py: FeatureMetrics(key=f["id"], ...).
// This fixture builder matches that reality.
function featureCollection(keys: string[]) {
  return {
    type: "FeatureCollection" as const,
    features: keys.map((key) => ({
      type: "Feature" as const,
      id: key,
      properties: { type: "dropoff" },
      geometry: { type: "Point" as const, coordinates: [-79.2, 33.3] },
    })),
  };
}

describe("joinActivations", () => {
  const scored = Object.keys(payload.species.redfish!.features);

  it("writes every species x hour activation onto a scored feature", () => {
    const out = joinActivations(featureCollection(scored), payload);
    const props = out.features[0]!.properties as Record<string, unknown>;
    const speciesNames = Object.keys(payload.species);
    for (const name of speciesNames) {
      for (let h = 0; h < 24; h += 1) {
        expect(props[activationKey(name, h)]).toBeTypeOf("number");
      }
    }
    expect(speciesNames.length * 24).toBeGreaterThan(0);
  });

  it("leaves an UNSCORED feature with no activation properties at all", () => {
    // features.geojson holds 2162 features; the payload scores only the 529
    // inside the flow-model domain. The muted, non-interactive styling branch
    // is reachable only because these keys are ABSENT -- a join that wrote
    // zeroes or nulls would make every unscored feature render as dead water.
    const out = joinActivations(featureCollection(["not-a-real-key"]), payload);
    const props = out.features[0]!.properties as Record<string, unknown>;
    const activationProps = Object.keys(props).filter((k) => k.startsWith("a_"));
    expect(activationProps).toEqual([]);
  });

  it("matches the payload's own activation values, hour for hour", () => {
    const key = scored[0]!;
    const out = joinActivations(featureCollection([key]), payload);
    const props = out.features[0]!.properties as Record<string, number>;
    const hours = payload.species.redfish!.features[key]!.hours;
    hours.forEach((fh, h) => {
      expect(props[activationKey("redfish", h)]).toBe(fh.activation);
    });
    expect(hours.length).toBe(24);
  });

  it("does not mutate the input collection", () => {
    const input = featureCollection([scored[0]!]);
    joinActivations(input, payload);
    expect(Object.keys(input.features[0]!.properties)).toEqual(["type"]);
  });
});
