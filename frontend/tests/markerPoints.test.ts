import { describe, expect, it } from "vitest";

import { toMarkerPoints } from "../src/map/MapView";

// The real features.geojson is mixed: 2026 Polygons, 134 Points and 2
// LineStrings. MapLibre's circle bucket draws a circle at every VERTEX of a
// geometry, so this collection is the shape of the bug being prevented -- a
// polygon whose ring would paint five circles, and a line that would paint
// a dotted string.
const collection = {
  type: "FeatureCollection" as const,
  features: [
    {
      type: "Feature" as const,
      id: "dropoff-619d21e1f694",
      properties: { type: "dropoff", a_redfish_11: 62 },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [-79.11, 33.59],
            [-79.11, 33.6],
            [-79.1, 33.6],
            [-79.1, 33.59],
            [-79.11, 33.59],
          ],
        ],
      },
    },
    {
      type: "Feature" as const,
      id: "creek_mouth-abc",
      properties: { type: "creek_mouth" },
      geometry: { type: "Point", coordinates: [-79.2, 33.3] },
    },
    {
      type: "Feature" as const,
      id: "jetty-1",
      properties: { type: "jetty" },
      geometry: {
        type: "LineString",
        coordinates: [
          [-79.18, 33.2],
          [-79.16, 33.22],
        ],
      },
    },
  ],
};

describe("toMarkerPoints", () => {
  it("emits one point per feature, not one per vertex", () => {
    const out = toMarkerPoints(collection);
    // 3 in, 3 out -- the polygon's ring holds 5 positions and the line 2, so
    // a vertex-per-circle layer would have shown 8 marks for 3 features.
    expect(out.features).toHaveLength(3);
    const types = out.features.map((f) => (f.geometry as { type: string }).type);
    expect(types).toEqual(["Point", "Point", "Point"]);
  });

  it("places a polygon's mark at its bounding-box centre", () => {
    const [first] = toMarkerPoints(collection).features;
    const [lng, lat] = (first!.geometry as { coordinates: [number, number] }).coordinates;
    expect(lng).toBeCloseTo(-79.105, 9);
    expect(lat).toBeCloseTo(33.595, 9);
    // A vertex mean would land here instead, tugged by the ring's repeated
    // closing vertex. Asserting the difference is what makes the choice real.
    expect(lng).not.toBeCloseTo(-79.106, 9);
    expect(lat).not.toBeCloseTo(33.594, 9);
  });

  it("carries the joined activations and the join key through", () => {
    const [first] = toMarkerPoints(collection).features;
    expect(first!.properties["a_redfish_11"]).toBe(62);
    expect(first!.properties["type"]).toBe("dropoff");
    // MapLibre does not hand a non-numeric feature id back from
    // queryRenderedFeatures, so the key has to travel as a property.
    expect(first!.properties["key"]).toBe("dropoff-619d21e1f694");
    expect(first!.id).toBe("dropoff-619d21e1f694");
  });
});
