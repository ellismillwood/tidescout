import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Mock } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { DayPayload } from "../src/api/types";
import type { DayContextValue } from "../src/state/DayContext";
import { HourStrip, tideSegments } from "../src/strip/HourStrip";

// The strip reads the day through `useDay`, so the whole context is mocked
// here rather than driven through a provider: this file is about what the
// strip DRAWS and what it CALLS, and a real provider would only add a fetch
// to stub.
const holder = vi.hoisted(() => ({ value: null as DayContextValue | null }));

vi.mock("../src/state/DayContext", () => ({
  useDay: () => {
    if (!holder.value) throw new Error("the test did not install a day context");
    return holder.value;
  },
}));

const SPECIES = "redfish";

/** The committed payload, deep-copied so a mutation cannot leak between tests. */
function payloadWith(mutate?: (payload: DayPayload) => void): DayPayload {
  const payload = structuredClone(fixture) as unknown as DayPayload;
  mutate?.(payload);
  return payload;
}

function hours(payload: DayPayload) {
  const block = payload.species[SPECIES];
  if (!block) throw new Error(`fixture has no ${SPECIES}`);
  return block.hours;
}

interface Spies {
  setHour: Mock<(hour: number) => void>;
  setSpecies: Mock<(species: string) => void>;
  setDate: Mock<(date: string) => void>;
  setModel: Mock<(model: string) => void>;
}

function mount(options: { hour?: number; payload?: DayPayload | null } = {}): Spies {
  const spies: Spies = {
    setHour: vi.fn<(hour: number) => void>(),
    setSpecies: vi.fn<(species: string) => void>(),
    setDate: vi.fn<(date: string) => void>(),
    setModel: vi.fn<(model: string) => void>(),
  };
  const payload = options.payload === undefined ? payloadWith() : options.payload;
  holder.value = {
    state: payload ? "ready" : "loading",
    payload,
    error: null,
    species: payload ? SPECIES : "",
    hour: options.hour ?? 0,
    slug: "winyah-bay",
    date: "2026-09-03",
    model: "best",
    ...spies,
  };
  render(<HourStrip />);
  return spies;
}

/**
 * The same strip, but with `hour` held in real state so `setHour` actually
 * MOVES the selection.
 *
 * `mount` above deliberately freezes `hour`, which is right for asserting
 * what the strip CALLS. It is wrong for asserting what the strip does after
 * the hour has changed -- and that blind spot is exactly what hid the focus
 * bug this harness exists to catch: every keyboard test fired `keyDown` at a
 * component whose hour could never move.
 */
function mountLive(startHour: number): Spies {
  const spies: Spies = {
    setHour: vi.fn<(hour: number) => void>(),
    setSpecies: vi.fn<(species: string) => void>(),
    setDate: vi.fn<(date: string) => void>(),
    setModel: vi.fn<(model: string) => void>(),
  };
  const payload = payloadWith();
  function Harness() {
    const [hour, setHour] = useState(startHour);
    holder.value = {
      state: "ready",
      payload,
      error: null,
      species: SPECIES,
      hour,
      slug: "winyah-bay",
      date: "2026-09-03",
      model: "best",
      ...spies,
      setHour: (next: number) => {
        spies.setHour(next);
        setHour(next);
      },
    };
    return <HourStrip />;
  }
  render(<Harness />);
  return spies;
}

const bars = () => screen.getAllByTestId("hour-bar");
const isSelected = (bar: HTMLElement) => bar.getAttribute("aria-checked") === "true";
const group = () => screen.getByTestId("hour-bars");

afterEach(() => {
  holder.value = null;
  vi.restoreAllMocks();
});

describe("HourStrip", () => {
  it("renders exactly 24 bars", () => {
    mount();
    // Not `toBeDefined` -- a strip that rendered zero bars would pass that.
    expect(bars()).toHaveLength(24);
  });

  it("still renders 24 bars, disabled, before the day has loaded", () => {
    // The frame is the same shape whether or not the numbers arrived, so the
    // strip cannot cause a layout jump when the payload lands. Both halves
    // asserted: a strip that rendered nothing while loading, and one that let
    // a person scrub a day it does not have, are each a failure.
    const spies = mount({ payload: null });
    const all = bars();
    expect(all).toHaveLength(24);
    expect(all.filter((bar) => bar.hasAttribute("disabled"))).toHaveLength(24);
    fireEvent.click(all[9]!);
    expect(spies.setHour).not.toHaveBeenCalled();
  });

  it("marks the selected hour and only that hour", () => {
    mount({ hour: 7 });
    const all = bars();
    expect(all.filter(isSelected)).toHaveLength(1);
    // ...and it is bar SEVEN. A component that always marked bar 0 would pass
    // the count on its own.
    expect(all.findIndex(isSelected)).toBe(7);
    expect(screen.getByTestId("playhead")).toHaveAttribute("data-hour", "7");
  });

  it("calls setHour with the bar's own index when clicked", () => {
    const spies = mount();
    fireEvent.click(bars()[17]!);
    expect(spies.setHour).toHaveBeenCalledWith(17);
    // A second, different bar: a handler wired to a constant passes on one.
    fireEvent.click(bars()[3]!);
    expect(spies.setHour).toHaveBeenLastCalledWith(3);
    expect(spies.setHour).toHaveBeenCalledTimes(2);
  });

  it("moves the hour with arrow keys", () => {
    // §9 requires arrow-key scrubbing. The positive control for the two clamp
    // tests below: without it, a strip with NO key handler at all would pass
    // both of them.
    const spies = mount({ hour: 12 });
    fireEvent.keyDown(group(), { key: "ArrowRight" });
    expect(spies.setHour).toHaveBeenLastCalledWith(13);
    fireEvent.keyDown(group(), { key: "ArrowLeft" });
    expect(spies.setHour).toHaveBeenLastCalledWith(11);
    expect(spies.setHour).toHaveBeenCalledTimes(2);
  });

  it("clamps at the end of the day instead of wrapping to hour 0", () => {
    // A clamp that only held at one end is the bug you find at 23:00.
    const spies = mount({ hour: 23 });
    fireEvent.keyDown(group(), { key: "ArrowRight" });
    expect(spies.setHour).not.toHaveBeenCalled();
    // ...and the other direction still works from the same end.
    fireEvent.keyDown(group(), { key: "ArrowLeft" });
    expect(spies.setHour).toHaveBeenCalledTimes(1);
    expect(spies.setHour).toHaveBeenCalledWith(22);
  });

  it("clamps at the start of the day instead of wrapping to hour 23", () => {
    const spies = mount({ hour: 0 });
    fireEvent.keyDown(group(), { key: "ArrowLeft" });
    expect(spies.setHour).not.toHaveBeenCalled();
    fireEvent.keyDown(group(), { key: "ArrowRight" });
    expect(spies.setHour).toHaveBeenCalledTimes(1);
    expect(spies.setHour).toHaveBeenCalledWith(1);
  });

  it("moves DOM focus with the arrow keys, and the readout follows it", () => {
    // Both halves, asserted together after every press. Focus alone would
    // pass against a readout still wired to the old hour; the readout alone
    // would pass against a component that never moves focus at all -- which
    // is what a roving tabindex that only sets `tabIndex` actually is.
    const spies = mountLive(5);
    bars()[5]!.focus();
    expect(document.activeElement).toBe(bars()[5]!);
    expect(screen.getByTestId("strip-readout")).toHaveTextContent("05:00");

    fireEvent.keyDown(document.activeElement!, { key: "ArrowRight" });
    expect(spies.setHour).toHaveBeenLastCalledWith(6);
    expect(document.activeElement).toBe(bars()[6]!);
    expect(screen.getByTestId("strip-readout")).toHaveTextContent("06:00");

    // Twice more: a component that moves focus once and then freezes on the
    // new bar would pass the single-press version.
    fireEvent.keyDown(document.activeElement!, { key: "ArrowRight" });
    fireEvent.keyDown(document.activeElement!, { key: "ArrowRight" });
    expect(document.activeElement).toBe(bars()[8]!);
    expect(screen.getByTestId("strip-readout")).toHaveTextContent("08:00");
    // The tab stop moved with it, or the strip has two of them.
    expect(bars().filter((bar) => bar.tabIndex === 0)).toHaveLength(1);
    expect(bars()[8]!.tabIndex).toBe(0);
  });

  it("never takes focus that was not already inside the strip", () => {
    // On mount, and on an hour change driven from anywhere else, the strip
    // must not yank the caret out of whatever the person is using.
    const outside = document.createElement("button");
    document.body.appendChild(outside);
    outside.focus();

    mountLive(5);
    expect(document.activeElement).toBe(outside);

    // A click moves the hour without focusing a bar (jsdom does not focus on
    // click), which is the same shape as a scrub driven from another
    // component. The pair: the hour DID move, and focus did NOT.
    fireEvent.click(bars()[9]!);
    expect(bars()[9]!).toHaveAttribute("aria-checked", "true");
    expect(document.activeElement).toBe(outside);

    outside.remove();
  });

  it("jumps to the ends with Home and End, still inside 0-23", () => {
    const spies = mount({ hour: 9 });
    fireEvent.keyDown(group(), { key: "End" });
    expect(spies.setHour).toHaveBeenLastCalledWith(23);
    fireEvent.keyDown(group(), { key: "Home" });
    expect(spies.setHour).toHaveBeenLastCalledWith(0);
  });

  it("marks hours with confidence below 1.0 as distinct from full-data hours", () => {
    // The strip is the only view of a whole day, so it carries disclosure.
    const payload = payloadWith((p) => {
      hours(p)[5]!.confidence = 0.62;
      hours(p)[6]!.confidence = 0.9;
    });
    mount({ payload });
    const all = bars();
    const partial = all.filter((bar) => bar.dataset.confidence === "partial");
    const full = all.filter((bar) => bar.dataset.confidence === "full");
    // Both sides counted: a component that marked EVERY bar partial would
    // pass an assertion that only looked at hours 5 and 6.
    expect(partial).toHaveLength(2);
    expect(full).toHaveLength(22);
    expect(all[5]!.dataset.confidence).toBe("partial");
    expect(all[7]!.dataset.confidence).toBe("full");
  });

  it("direct-labels exactly one bar, the day's best hour", () => {
    // The one direct label in the chart. Exactly one, and on the RIGHT bar:
    // "some label rendered" would pass against a label on every bar.
    const payload = payloadWith((p) => {
      hours(p)[13]!.score = 96;
    });
    mount({ payload, hour: 2 });
    const labels = screen.getAllByTestId("hour-peak");
    expect(labels).toHaveLength(1);
    expect(labels[0]!).toHaveTextContent("96");
    expect(bars()[13]!).toContainElement(labels[0]!);
  });

  it("hides the peak label when the playhead is parked on the peak", () => {
    const payload = payloadWith((p) => {
      hours(p)[13]!.score = 96;
    });
    // Same payload, two selections: the label is there for one and not the
    // other, so the assertion cannot pass against a strip with no label at all.
    mount({ payload, hour: 13 });
    expect(screen.queryAllByTestId("hour-peak")).toHaveLength(0);
    expect(screen.getByTestId("strip-readout")).toHaveTextContent("96");
  });

  it("draws one unbroken tide line when every hour has a reading", () => {
    mount();
    expect(screen.getAllByTestId("tide-segment")).toHaveLength(1);
    expect(screen.queryAllByTestId("tide-gap")).toHaveLength(0);
  });

  it("breaks the tide line at a null hour rather than interpolating across it", () => {
    const payload = payloadWith((p) => {
      p.conditions[9]!.tide_height_ft = null;
    });
    mount({ payload });
    const segments = screen.getAllByTestId("tide-segment");
    // Two segments, not one line drawn straight through the missing hour.
    expect(segments).toHaveLength(2);
    expect(segments[0]!.dataset.hours).toBe("0-8");
    expect(segments[1]!.dataset.hours).toBe("10-23");
    // And the gap is disclosed rather than silently closed.
    const gaps = screen.getAllByTestId("tide-gap");
    expect(gaps).toHaveLength(1);
    expect(gaps[0]!).toHaveAttribute("data-hour", "9");
  });

  it("scrubs the hour and nothing else -- no species, date or model change", () => {
    const spies = mount({ hour: 4 });
    fireEvent.keyDown(group(), { key: "ArrowRight" });
    fireEvent.click(bars()[20]!);
    expect(spies.setHour).toHaveBeenCalledTimes(2);
    expect(spies.setSpecies).not.toHaveBeenCalled();
    expect(spies.setDate).not.toHaveBeenCalled();
    expect(spies.setModel).not.toHaveBeenCalled();
  });

  it("drags the playhead to the hour under the pointer", () => {
    const spies = mount({ hour: 0 });
    const plot = screen.getByTestId("strip-plot");
    // jsdom does no layout, so the plot's box is supplied here. 240px wide
    // over 24 hours makes each hour exactly 10px.
    vi.spyOn(plot, "getBoundingClientRect").mockReturnValue({
      left: 0, top: 0, right: 240, bottom: 60, width: 240, height: 60, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    fireEvent.pointerDown(plot, { clientX: 65, pointerId: 1, button: 0 });
    expect(spies.setHour).toHaveBeenLastCalledWith(6);
    fireEvent.pointerMove(plot, { clientX: 185, pointerId: 1 });
    expect(spies.setHour).toHaveBeenLastCalledWith(18);
    fireEvent.pointerUp(plot, { clientX: 185, pointerId: 1 });
    // After the drag ends, moving the pointer no longer scrubs.
    fireEvent.pointerMove(plot, { clientX: 15, pointerId: 1 });
    expect(spies.setHour).toHaveBeenCalledTimes(2);
    expect(spies.setHour).not.toHaveBeenCalledWith(1);
  });

  it("clamps a drag past either edge to hour 0 and hour 23", () => {
    const spies = mount({ hour: 12 });
    const plot = screen.getByTestId("strip-plot");
    vi.spyOn(plot, "getBoundingClientRect").mockReturnValue({
      left: 0, top: 0, right: 240, bottom: 60, width: 240, height: 60, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    fireEvent.pointerDown(plot, { clientX: -400, pointerId: 1, button: 0 });
    expect(spies.setHour).toHaveBeenLastCalledWith(0);
    fireEvent.pointerMove(plot, { clientX: 9999, pointerId: 1 });
    expect(spies.setHour).toHaveBeenLastCalledWith(23);
  });
});

describe("tideSegments", () => {
  it("returns one run for a complete series and two around a hole", () => {
    const whole = [1, 2, 3, 4].map((n) => n as number | null);
    expect(tideSegments(whole)).toEqual([[0, 1, 2, 3]]);
    expect(tideSegments([1, 2, null, 4])).toEqual([[0, 1], [3]]);
  });

  it("drops leading, trailing and repeated holes without inventing points", () => {
    expect(tideSegments([null, 1, null, null, 2, null])).toEqual([[1], [4]]);
    expect(tideSegments([null, null])).toEqual([]);
    // A NaN is as absent as a null: neither is a reading.
    expect(tideSegments([1, Number.NaN, 3])).toEqual([[0], [2]]);
  });
});
