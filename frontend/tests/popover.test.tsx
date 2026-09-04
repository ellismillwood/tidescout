import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { DayPayload } from "../src/api/types";
import type { DayContextValue } from "../src/state/DayContext";
import { ConditionsRail } from "../src/rail/ConditionsRail";
import { FactorBars } from "../src/rail/FactorBars";
import { FeaturePopover, mergeFeatureSubs } from "../src/rail/FeaturePopover";
import { HourStrip } from "../src/strip/HourStrip";

const payload = fixture as unknown as DayPayload;

describe("mergeFeatureSubs", () => {
  const key = Object.keys(payload.species.redfish!.features)[0]!;

  it("reconstructs all ten factors from the two scopes", () => {
    const merged = mergeFeatureSubs(payload, "redfish", key, 12);
    expect(merged).toHaveLength(10);
  });

  it("lets the FEATURE's own value win over the bay-wide one", () => {
    // flow and salinity exist at both scopes: the hour carries the bay-wide
    // reading, the feature carries its own. A merge that let the hour win
    // would show every marker the same flow -- the exact thing per-feature
    // scoring exists to avoid.
    const merged = mergeFeatureSubs(payload, "redfish", key, 12);
    const featureFlow = payload.species.redfish!.features[key]!.hours[12]!.subs
      .find((s) => s.factor === "flow");
    expect(merged.find((s) => s.factor === "flow")?.value).toBe(featureFlow?.value);
  });

  it("restores the flags a trimmed feature sub cannot carry", () => {
    // A feature-hour's subs are factor/value/reason only, but the feature-hour
    // itself lists which of them are provisional. Losing that would drop the
    // one disclosure that matters most on this bay: salinity is modelled and
    // nothing observed constrains it.
    const featureHour = payload.species.redfish!.features[key]!.hours[12]!;
    expect(featureHour.provisional).toContain("salinity");
    const merged = mergeFeatureSubs(payload, "redfish", key, 12);
    expect(merged.find((s) => s.factor === "salinity")?.provisional).toBe(true);
    expect(merged.find((s) => s.factor === "structure")?.provisional).toBe(false);
  });

  it("reads sub_scope.feature from the payload rather than a hardcoded list", () => {
    // A frontend with the split hardcoded breaks silently the day a factor
    // changes scope -- which is precisely why the payload publishes it.
    const shuffled = {
      ...payload,
      sub_scope: { hour: [...payload.sub_scope.hour], feature: ["flow"] },
    } as DayPayload;
    const merged = mergeFeatureSubs(shuffled, "redfish", key, 12);
    expect(merged.length).not.toBe(10);
  });

  it("reads sub_scope.hour from the payload rather than a hardcoded list", () => {
    // The same guard, the OTHER scope. The check above shuffles
    // sub_scope.feature and leaves sub_scope.hour untouched, so a merge that
    // reads sub_scope.feature correctly but hardcodes the hour-scope split
    // would still pass it -- and did: a hardcoded
    // new Set(["light","pressure","season","solunar","stage","water_temp","wind"])
    // in place of `payload.sub_scope.hour` left all 66 tests green. Shuffle
    // hour down to a single factor (feature scope left at the payload's own
    // three) and assert the exact factor set that implies.
    const shuffled = {
      ...payload,
      sub_scope: { hour: ["stage"], feature: [...payload.sub_scope.feature] },
    } as DayPayload;
    const merged = mergeFeatureSubs(shuffled, "redfish", key, 12);
    expect(merged.map((sub) => sub.factor).sort()).toEqual([
      "flow",
      "salinity",
      "stage",
      "structure",
    ]);
  });
});

// ---------------------------------------------------------------------------
// The rest of this file renders the components. They read the day through
// `useDay`, so the whole context is mocked -- a real provider would only add a
// fetch to stub, and this is about what the panels DRAW.
// ---------------------------------------------------------------------------

const holder = vi.hoisted(() => ({ value: null as DayContextValue | null }));

vi.mock("../src/state/DayContext", () => ({
  useDay: () => {
    if (!holder.value) throw new Error("the test did not install a day context");
    return holder.value;
  },
}));

const SPECIES = "redfish";
const KEY = Object.keys(payload.species.redfish!.features)[0]!;

function payloadWith(mutate?: (payload: DayPayload) => void): DayPayload {
  const copy = structuredClone(fixture) as unknown as DayPayload;
  mutate?.(copy);
  return copy;
}

function install(options: { hour?: number; payload?: DayPayload | null } = {}) {
  const next = options.payload === undefined ? payloadWith() : options.payload;
  holder.value = {
    state: next ? "ready" : "loading",
    payload: next,
    error: null,
    stale: false,
    species: next ? SPECIES : "",
    hour: options.hour ?? 0,
    slug: "winyah-bay",
    date: "2026-09-03",
    model: "best",
    setHour: vi.fn(),
    setSpecies: vi.fn(),
    setDate: vi.fn(),
    setModel: vi.fn(),
  };
  return next;
}

/**
 * The strip and the rail, sharing ONE context whose `hour` is real state.
 *
 * The previous task's harness froze `hour`, so every assertion about what
 * happens after the hour changes could not fail. Here `setHour` genuinely
 * moves the selection, and the strip -- the real control -- is what calls it,
 * so "scrub the strip and the rail follows" is asserted end to end rather
 * than by poking the rail directly.
 */
function mountLive(startHour: number) {
  const day = payloadWith();
  function Harness() {
    const [hour, setHour] = useState(startHour);
    holder.value = {
      state: "ready",
      payload: day,
      error: null,
      stale: false,
      species: SPECIES,
      hour,
      slug: "winyah-bay",
      date: "2026-09-03",
      model: "best",
      setHour,
      setSpecies: vi.fn(),
      setDate: vi.fn(),
      setModel: vi.fn(),
    };
    return (
      <>
        <HourStrip />
        <ConditionsRail />
      </>
    );
  }
  render(<Harness />);
}

const rows = () => screen.getAllByTestId("factor-row");
const reasons = () =>
  screen.getAllByTestId("factor-reason").map((node) => node.textContent);

afterEach(() => {
  holder.value = null;
  vi.restoreAllMocks();
});

describe("FactorBars", () => {
  const subs = payload.species.redfish!.hours[12]!.subs;

  it("draws one row per sub and prints every reason verbatim", () => {
    render(<FactorBars subs={subs} caption="Factors" />);
    expect(rows()).toHaveLength(subs.length);
    // Verbatim, in payload order, with nothing truncated or reworded -- these
    // strings are where "UNCALIBRATED model estimate, no observation
    // constrains it" reaches a person.
    expect(reasons()).toEqual(subs.map((sub) => sub.reason));
  });

  it("draws a bar for a value and NO bar for an absent one", () => {
    const [first, second] = subs;
    render(
      <FactorBars
        subs={[
          { ...first!, value: 0.5 },
          { ...second!, value: null, missing: true },
        ]}
      />,
    );
    const drawn = rows();
    expect(drawn).toHaveLength(2);
    expect(drawn[0]?.getAttribute("data-scored")).toBe("true");
    expect(drawn[0]?.querySelector(".factor-fill")).not.toBeNull();
    // An absence is an absence, never a zero-length bar: that would read as
    // "scored, and the answer is zero", which is a different claim.
    expect(drawn[1]?.getAttribute("data-scored")).toBe("false");
    expect(drawn[1]?.querySelector(".factor-fill")).toBeNull();
    expect(screen.getAllByTestId("factor-flag-missing")).toHaveLength(1);
  });
});

describe("ConditionsRail", () => {
  it("shows the selected hour's conditions and the hour's score", () => {
    const day = install({ hour: 12 });
    render(<ConditionsRail />);
    const now = day!.conditions[12]!;
    expect(screen.getByTestId("rail-clock")).toHaveTextContent("12:00");
    expect(screen.getByTestId("rail-score")).toHaveTextContent(
      String(day!.species[SPECIES]!.hours[12]!.score),
    );
    const tide = screen.getByTestId("cond-row-Tide");
    expect(tide).toHaveTextContent(`${now.tide_height_ft!.toFixed(2)} ft`);
    expect(tide).toHaveTextContent(now.tide_phase!);
  });

  it("follows the strip: scrubbing to another hour redraws the rail", () => {
    mountLive(12);
    const day = payload;
    const before = day.conditions[12]!.tide_height_ft!.toFixed(2);
    const after = day.conditions[18]!.tide_height_ft!.toFixed(2);
    // The pair is only a test if the two differ -- otherwise a rail that
    // never redrew would pass.
    expect(before).not.toBe(after);

    expect(screen.getByTestId("rail-clock")).toHaveTextContent("12:00");
    expect(screen.getByTestId("cond-row-Tide")).toHaveTextContent(`${before} ft`);

    fireEvent.click(screen.getAllByTestId("hour-bar")[18]!);

    expect(screen.getByTestId("rail-clock")).toHaveTextContent("18:00");
    expect(screen.getByTestId("cond-row-Tide")).toHaveTextContent(`${after} ft`);
    expect(screen.getByTestId("rail-score")).toHaveTextContent(
      String(day.species[SPECIES]!.hours[18]!.score),
    );
  });

  it("tags the hour's per-feature factors from sub_scope, not a hardcoded list", () => {
    // The rail shows the HOUR's subs, which carry the bay-wide flow and
    // salinity. Which of them vary per feature comes from the payload.
    install({ hour: 12 });
    const { unmount } = render(<ConditionsRail />);
    expect(screen.getAllByTestId("factor-flag-note")).toHaveLength(
      payload.sub_scope.feature.filter((factor) =>
        payload.species[SPECIES]!.hours[12]!.subs.some((sub) => sub.factor === factor),
      ).length,
    );
    unmount();

    install({ hour: 12, payload: payloadWith((p) => (p.sub_scope.feature = [])) });
    render(<ConditionsRail />);
    expect(screen.queryAllByTestId("factor-flag-note")).toHaveLength(0);
  });

  it("says 'no reading' for a payload that carries NO conditions block at all", () => {
    // Same pre-branch payload the strip test describes: the field is required
    // by the type and absent from anything cached before it existed, and
    // indexing `undefined` throws through a rail with no error boundary above
    // it. Degrade, do not crash -- and the pair says which: the conditions
    // block reports its absence AND the rest of the rail (the hour's score,
    // its factor bars) still renders, which a caught-and-blank rail would not.
    install({
      hour: 12,
      payload: payloadWith((p) => {
        delete (p as Partial<DayPayload>).conditions;
      }),
    });
    render(<ConditionsRail />);
    expect(screen.getByTestId("cond-absent")).toHaveTextContent(/no conditions row/i);
    expect(screen.getByTestId("rail-score")).toHaveTextContent(
      String(payload.species[SPECIES]!.hours[12]!.score),
    );
    expect(screen.getAllByTestId("factor-row").length).toBeGreaterThan(0);
  });

  it("says so when the day carries no water or astronomy block", () => {
    install({
      hour: 12,
      payload: payloadWith((p) => {
        p.water = null;
        p.astro = null;
      }),
    });
    render(<ConditionsRail />);
    expect(screen.getByTestId("cond-row-Water")).toHaveTextContent(
      "no water temperature for this day",
    );
    expect(screen.getByTestId("cond-row-Sun & moon")).toHaveTextContent(
      "no astronomy for this day",
    );
  });
});

describe("FeaturePopover", () => {
  it("shows all ten merged factors, each with its reason verbatim", () => {
    const day = install({ hour: 12 });
    render(<FeaturePopover featureKey={KEY} onClose={vi.fn()} />);
    expect(rows()).toHaveLength(10);
    expect(reasons()).toEqual(
      mergeFeatureSubs(day!, SPECIES, KEY, 12).map((sub) => sub.reason),
    );
    expect(screen.getByTestId("popover-activation")).toHaveTextContent(
      String(day!.species[SPECIES]!.features[KEY]!.hours[12]!.activation),
    );
    expect(screen.getByTestId("popover-reason")).toHaveTextContent(
      day!.species[SPECIES]!.features[KEY]!.hours[12]!.reason,
    );
  });

  it("prints the FEATURE's flow reason, not the bay-wide one", () => {
    const day = install({ hour: 12 });
    render(<FeaturePopover featureKey={KEY} onClose={vi.fn()} />);
    const featureFlow = day!.species[SPECIES]!.features[KEY]!.hours[12]!.subs.find(
      (sub) => sub.factor === "flow",
    )!;
    const hourFlow = day!.species[SPECIES]!.hours[12]!.subs.find(
      (sub) => sub.factor === "flow",
    )!;
    expect(featureFlow.reason).not.toBe(hourFlow.reason);
    const row = screen.getByTestId("factor-bars-feature").querySelector('[data-factor="flow"]');
    expect(row).toHaveTextContent(featureFlow.reason);
    expect(reasons()).not.toContain(hourFlow.reason);
  });

  it("groups the ten by the scope they came from", () => {
    install({ hour: 12 });
    render(<FeaturePopover featureKey={KEY} onClose={vi.fn()} />);
    expect(
      screen.getByTestId("factor-bars-hour").querySelectorAll("[data-factor]"),
    ).toHaveLength(payload.sub_scope.hour.length);
    expect(
      screen.getByTestId("factor-bars-feature").querySelectorAll("[data-factor]"),
    ).toHaveLength(payload.sub_scope.feature.length);
  });

  it("shows the excluded-from-the-score treatment for a feature-scope factor the payload excludes", () => {
    // A feature-hour's subs are trimmed to factor/value/reason -- they carry
    // no flags of their own. `missing` (rendered as "no reading -- excluded
    // from the score") is read back from the feature-hour's own `excluded`
    // list, the same way `provisional` is read back from its `provisional`
    // list (pinned above). The fixture ships zero excluded factors across
    // all 144 feature-hours, so that readback line has no fixture coverage;
    // mutate one locally. Salinity is the factor this project flags as
    // uncalibrated everywhere else, so it is the one worth pinning here too.
    // The sub's own value is left untouched (a real number, not null) so
    // this test can only pass because of the `missing` readback -- not
    // because `value === null` independently triggers the same flag.
    const day = install({
      hour: 12,
      payload: payloadWith((p) => {
        const featureHour = p.species[SPECIES]!.features[KEY]!.hours[12]!;
        featureHour.excluded = ["salinity"];
        featureHour.provisional = [];
      }),
    });
    render(<FeaturePopover featureKey={KEY} onClose={vi.fn()} />);
    const salinitySub = day!.species[SPECIES]!.features[KEY]!.hours[12]!.subs.find(
      (sub) => sub.factor === "salinity",
    )!;
    expect(salinitySub.value).not.toBeNull();
    const row = screen
      .getByTestId("factor-bars-feature")
      .querySelector('[data-factor="salinity"]');
    expect(row).not.toBeNull();
    expect(row?.querySelector('[data-testid="factor-flag-missing"]')).toHaveTextContent(
      "no reading — excluded from the score",
    );
  });

  it("admits an unscored feature instead of drawing an empty card", () => {
    install({ hour: 12 });
    render(<FeaturePopover featureKey="not-a-feature" onClose={vi.fn()} />);
    expect(screen.getByTestId("popover-unscored")).toBeInTheDocument();
    // The hour's own factors still apply and are still shown; only the three
    // that need the feature are missing.
    expect(rows()).toHaveLength(payload.sub_scope.hour.length);
    expect(screen.getByTestId("factors-empty")).toBeInTheDocument();
  });

  it("closes on the button and on Escape", () => {
    install({ hour: 12 });
    const onClose = vi.fn();
    render(<FeaturePopover featureKey={KEY} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: /close this feature/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
