import { describe, expect, it } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { DayPayload } from "../src/api/types";

const TOP_LEVEL = [
  "slug", "day", "model_label", "missing", "freshness",
  "sub_scope", "flow", "species", "salinity", "conditions",
  // Day-level siblings, added in Task 1. They are NOT inside `conditions`:
  // sun, moon and water temperature are properties of the DAY, and putting
  // them in each of 24 hourly rows would duplicate a day fact 24 times --
  // the same error the conditions block itself exists to avoid.
  "water", "astro",
] as const;

const HOUR_KEYS = [
  "time", "score", "subs", "excluded", "confidence",
  "constrained_share", "provisional",
] as const;

const FEATURE_HOUR_KEYS = [
  "activation", "reason", "confidence", "constrained_share",
  "excluded", "provisional", "subs",
] as const;

describe("payload contract", () => {
  const payload = fixture as unknown as DayPayload;

  it("carries exactly the top-level keys the types declare", () => {
    expect(Object.keys(payload).sort()).toEqual([...TOP_LEVEL].sort());
  });

  it("declares sub_scope rather than leaving the split to be hardcoded", () => {
    expect(payload.sub_scope.hour.length).toBe(7);
    expect(payload.sub_scope.feature.sort()).toEqual(["flow", "salinity", "structure"]);
  });

  it("has 24 hours and 24 conditions on one aligned axis", () => {
    expect(payload.conditions.length).toBe(24);
    for (const blob of Object.values(payload.species)) {
      expect(blob.hours.length).toBe(24);
      blob.hours.forEach((h, i) => expect(h.time).toBe(payload.conditions[i]!.time));
    }
  });

  it("gives every hour the declared keys", () => {
    for (const blob of Object.values(payload.species)) {
      for (const hour of blob.hours) {
        expect(Object.keys(hour).sort()).toEqual([...HOUR_KEYS].sort());
      }
    }
  });

  it("gives every feature-hour the declared keys, and NO time", () => {
    let checked = 0;
    for (const blob of Object.values(payload.species)) {
      for (const feature of Object.values(blob.features)) {
        expect(feature.hours.length).toBe(24);
        for (const fh of feature.hours) {
          expect(Object.keys(fh).sort()).toEqual([...FEATURE_HOUR_KEYS].sort());
          checked += 1;
        }
      }
    }
    // Without this the four loops above pass vacuously on an empty fixture.
    expect(checked).toBeGreaterThan(0);
  });

  it("ships only feature-scope factors on a feature-hour", () => {
    const allowed = new Set(payload.sub_scope.feature);
    let checked = 0;
    for (const blob of Object.values(payload.species)) {
      for (const feature of Object.values(blob.features)) {
        for (const fh of feature.hours) {
          for (const sub of fh.subs) {
            expect(allowed.has(sub.factor)).toBe(true);
            checked += 1;
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(0);
  });
});
