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

// A HourScore's subs are the full SubScore -- factor/value/weight/reason/
// missing/provisional. Distinct from TrimmedSub below.
const SUB_SCORE_KEYS = [
  "factor", "value", "weight", "reason", "missing", "provisional",
] as const;

// A feature-hour's subs are TRIMMED to factor/value/reason. A previous PR
// shrank the payload 49% by trimming these; a regression that re-fattens
// them back toward SubScore would double the payload with nothing here
// noticing unless this exact key set is pinned.
const TRIMMED_SUB_KEYS = ["factor", "value", "reason"] as const;

const CONDITIONS_KEYS = [
  "time", "air_temp_f", "wind_speed_kn", "wind_dir_deg", "wind_gust_kn",
  "pressure_mb", "pressure_trend_mb_3h", "cloud_cover_pct", "precip_in",
  "tide_height_ft", "tide_phase", "tide_frac",
] as const;

const FLOW_KEYS = [
  "range_bucket", "discharge_cfs", "discharge_bucket", "regimes", "clamped",
] as const;

const SALINITY_KEYS = [
  "cfs", "fitted", "extrapolated", "series", "representative_ppt",
  "representative_hour", "provenance",
] as const;

const FRESHNESS_KEYS = ["day", "model_label", "generated_at"] as const;

// water/astro may legitimately be null (no sensor / no astronomy for the
// day) -- the key set is asserted only when the block is present.
const WATER_KEYS = ["temp_f", "temp_trend_f_3d"] as const;

const ASTRO_KEYS = [
  "dawn", "sunrise", "sunset", "dusk", "moon_phase_frac", "moonrise",
  "moonset",
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

  it("gives every conditions entry the declared key set", () => {
    let checked = 0;
    for (const c of payload.conditions) {
      expect(Object.keys(c).sort()).toEqual([...CONDITIONS_KEYS].sort());
      checked += 1;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("gives flow, salinity and freshness the declared key sets", () => {
    expect(Object.keys(payload.flow).sort()).toEqual([...FLOW_KEYS].sort());
    expect(Object.keys(payload.salinity).sort()).toEqual([...SALINITY_KEYS].sort());
    expect(Object.keys(payload.freshness).sort()).toEqual([...FRESHNESS_KEYS].sort());
  });

  it("gives water and astro the declared key sets when present, else allows null", () => {
    if (payload.water === null) {
      expect(payload.water).toBeNull();
    } else {
      expect(Object.keys(payload.water).sort()).toEqual([...WATER_KEYS].sort());
    }
    if (payload.astro === null) {
      expect(payload.astro).toBeNull();
    } else {
      expect(Object.keys(payload.astro).sort()).toEqual([...ASTRO_KEYS].sort());
    }
  });

  it("gives every hour the declared keys", () => {
    for (const blob of Object.values(payload.species)) {
      for (const hour of blob.hours) {
        expect(Object.keys(hour).sort()).toEqual([...HOUR_KEYS].sort());
      }
    }
  });

  it("gives every hour's subs the full SubScore key set", () => {
    let checked = 0;
    for (const blob of Object.values(payload.species)) {
      for (const hour of blob.hours) {
        for (const sub of hour.subs) {
          expect(Object.keys(sub).sort()).toEqual([...SUB_SCORE_KEYS].sort());
          checked += 1;
        }
      }
    }
    expect(checked).toBeGreaterThan(0);
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

  it("ships only feature-scope factors on a feature-hour, trimmed to TrimmedSub", () => {
    const allowed = new Set(payload.sub_scope.feature);
    let checked = 0;
    for (const blob of Object.values(payload.species)) {
      for (const feature of Object.values(blob.features)) {
        for (const fh of feature.hours) {
          for (const sub of fh.subs) {
            expect(allowed.has(sub.factor)).toBe(true);
            // The whole point of TrimmedSub: no weight/missing/provisional.
            // A regression that re-fattens this back toward SubScore would
            // silently double the payload without this catching it.
            expect(Object.keys(sub).sort()).toEqual([...TRIMMED_SUB_KEYS].sort());
            checked += 1;
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(0);
  });
});
