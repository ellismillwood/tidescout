import { describe, expect, it } from "vitest";

import { activationKey } from "../src/map/join";
import { colorExpr, radiusExpr, UNSCORED_SENTINEL } from "../src/map/layers";

function flatten(expr: unknown): string[] {
  if (typeof expr === "string") return [expr];
  if (!Array.isArray(expr)) return [];
  return expr.flatMap(flatten);
}

describe("paint expressions", () => {
  it("reads the key for the species and hour it was asked for", () => {
    // A builder that ignored its arguments and always returned hour 0 would
    // pass a "returns an object" check while freezing the map on one hour.
    const tokens = flatten(radiusExpr("speckled_trout", 17));
    expect(tokens).toContain(activationKey("speckled_trout", 17));
    expect(tokens).not.toContain(activationKey("speckled_trout", 0));
    expect(tokens).not.toContain(activationKey("redfish", 17));
  });

  it("changes with the hour and with the species", () => {
    expect(radiusExpr("redfish", 3)).not.toEqual(radiusExpr("redfish", 4));
    expect(colorExpr("redfish", 3)).not.toEqual(colorExpr("speckled_trout", 3));
  });

  it("coalesces a missing activation to the unscored sentinel", () => {
    // Unscored features have no a_* property. Without the coalesce, `get`
    // returns null, interpolate throws, and the whole layer fails to paint --
    // taking the 529 scored markers down with the 1,633 unscored ones.
    for (const expr of [radiusExpr("redfish", 0), colorExpr("redfish", 0)]) {
      const json = JSON.stringify(expr);
      expect(json).toContain("coalesce");
      expect(json).toContain(String(UNSCORED_SENTINEL));
    }
  });

  it("produces expressions MapLibre can parse", async () => {
    // From @maplibre/maplibre-gl-style-spec, NOT from maplibre-gl -- the
    // latter's package exports are only ".", "./dist/*" and "./package.json",
    // so `expression` is not reachable from it.
    const { expression } = await import("@maplibre/maplibre-gl-style-spec");
    // createExpression's real signature is (expression, rootKey: string,
    // propertySpec, globalState) -- rootKey just identifies the expression's
    // location for error messages ("rootKey must identify the location of
    // the expression in the style JSON"), it is not part of the spec object.
    // propertySpec is a discriminated union keyed on `type`; each call is
    // written out with a literal `type` rather than looped over a shared
    // `kind` variable, which would widen `type` to "number" | "color" and
    // stop TS from resolving which union member (and which required fields,
    // e.g. `overridable` on "color" only) applies.
    const radiusParsed = expression.createExpression(radiusExpr("redfish", 12), "layers.test", {
      type: "number",
      "property-type": "data-driven",
      expression: { interpolated: true, parameters: ["zoom", "feature"] },
      transition: false,
    });
    expect(radiusParsed.result).toBe("success");

    const colorParsed = expression.createExpression(colorExpr("redfish", 12), "layers.test", {
      type: "color",
      "property-type": "data-driven",
      expression: { interpolated: true, parameters: ["zoom", "feature"] },
      transition: false,
      overridable: false,
    });
    expect(colorParsed.result).toBe("success");
  });
});
