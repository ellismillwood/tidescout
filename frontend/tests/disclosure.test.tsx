import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { DayPayload, HourScore } from "../src/api/types";
import { Disclosure, age } from "../src/ui/Disclosure";

/**
 * The hour under test is a REAL scored hour out of the committed payload,
 * patched per test. A hand-built literal would let a component that renders
 * only the fields the literal happens to carry pass every test here.
 */
function hourWith(patch: Partial<HourScore> = {}): HourScore {
  const payload = structuredClone(fixture) as unknown as DayPayload;
  const hour = payload.species["redfish"]?.hours[0];
  if (!hour) throw new Error("the fixture has no redfish hour 0");
  return { ...hour, ...patch };
}

/** Winyah's real hour 0 is provisional on salinity; this is its opposite. */
function fullyConstrained(): HourScore {
  const hour = hourWith();
  return {
    ...hour,
    confidence: 1,
    constrained_share: 1,
    provisional: [],
    excluded: [],
    subs: hour.subs.map((sub) => ({ ...sub, missing: false, provisional: false })),
  };
}

const FRESHNESS = {
  day: "2026-09-03",
  model_label: "best",
  generated_at: "2026-09-03T20:00:00+00:00",
};

describe("Disclosure", () => {
  it("shows confidence and constrained_share as SEPARATE values", () => {
    // They answer different questions -- how much of the authored weight
    // resolved, versus how much of THAT weight rests on measurement rather
    // than on a model. Merging them into one "quality" number would erase a
    // distinction five PRs went into establishing.
    render(<Disclosure hour={hourWith({ confidence: 0.79, constrained_share: 0.92 })} />);
    expect(screen.getByTestId("confidence")).toHaveTextContent("0.79");
    expect(screen.getByTestId("constrained-share")).toHaveTextContent("0.92");
    // ...and neither cell is quietly showing the other's number as well,
    // which is what a merged "quality" score printed twice would look like.
    expect(screen.getByTestId("confidence")).not.toHaveTextContent("0.92");
    expect(screen.getByTestId("constrained-share")).not.toHaveTextContent("0.79");
  });

  it("moves the two numbers independently when the hour changes", () => {
    // The pair the test above cannot state on its own: a component wired to
    // ONE source for both slots passes it whenever the two happen to differ
    // by construction. Swapping the values must swap what is on screen.
    const { rerender } = render(
      <Disclosure hour={hourWith({ confidence: 0.79, constrained_share: 0.92 })} />,
    );
    rerender(<Disclosure hour={hourWith({ confidence: 0.92, constrained_share: 0.79 })} />);
    expect(screen.getByTestId("confidence")).toHaveTextContent("0.92");
    expect(screen.getByTestId("constrained-share")).toHaveTextContent("0.79");
  });

  it("names WHICH factors are provisional, not merely that some are", () => {
    render(<Disclosure hour={hourWith({ provisional: ["salinity"] })} />);
    expect(screen.getByTestId("provisional")).toHaveTextContent("salinity");
    // The flag says it in words too, naming the same factor: "some factors
    // are provisional" is a different, less useful claim.
    expect(screen.getByTestId("disclosure-flag")).toHaveTextContent(/salinity/i);
  });

  it("names a DIFFERENT set when a different set is unconstrained", () => {
    // A hardcoded "salinity" passes the test above. Two names, both present,
    // and the one from the previous case absent.
    render(<Disclosure hour={hourWith({ provisional: ["flow", "water_temp"] })} />);
    const cell = screen.getByTestId("provisional");
    expect(cell).toHaveTextContent("flow");
    // `water_temp` is the payload key; "water temp" is what a person reads --
    // the same `humanizeFactor` the factor bars label their rows with.
    expect(cell).toHaveTextContent("water temp");
    expect(cell).not.toHaveTextContent("salinity");
  });

  it("keeps 'left out' separate from 'modelled'", () => {
    // Excluded and provisional are not the same fact: an excluded factor had
    // no reading at all and moved `confidence`; a provisional one scored at
    // full weight and moved `constrained_share`. Each is named in its own
    // cell, or the disclosure has merged two states of knowledge.
    render(
      <Disclosure
        hour={hourWith({
          confidence: 0.78,
          constrained_share: 0.9,
          excluded: ["water_temp"],
          provisional: ["salinity"],
        })}
      />,
    );
    expect(screen.getByTestId("excluded")).toHaveTextContent("water temp");
    expect(screen.getByTestId("excluded")).not.toHaveTextContent("salinity");
    expect(screen.getByTestId("provisional")).toHaveTextContent("salinity");
    expect(screen.getByTestId("provisional")).not.toHaveTextContent("water temp");
  });

  it("renders nothing alarming when everything is fully constrained", () => {
    // The inverse half: a component that always warned would pass every test
    // above and cry wolf on a fully-observed hour. Both states are asserted
    // here, in one test, so neither branch can rot unnoticed.
    const { rerender } = render(<Disclosure hour={fullyConstrained()} />);
    expect(screen.getByTestId("disclosure")).toHaveAttribute("data-tone", "clear");
    expect(screen.queryByTestId("disclosure-flag")).toBeNull();
    expect(screen.getByTestId("provisional")).toHaveTextContent(/none/i);
    // ...and the numbers are still there. "Clear" is not "silent".
    expect(screen.getByTestId("confidence")).toHaveTextContent("1.00");
    expect(screen.getByTestId("constrained-share")).toHaveTextContent("1.00");

    // The positive control, same component, same render: the real hour DOES
    // raise its flag.
    rerender(<Disclosure hour={hourWith({ constrained_share: 0.92, provisional: ["salinity"] })} />);
    expect(screen.getByTestId("disclosure")).toHaveAttribute("data-tone", "flagged");
    expect(screen.getByTestId("disclosure-flag")).toHaveTextContent(/salinity/i);
  });

  it("reports how old the run is, and says so when it is stale", () => {
    const now = new Date("2026-09-03T23:00:00+00:00");
    const { rerender } = render(
      <Disclosure hour={fullyConstrained()} freshness={FRESHNESS} now={now} />,
    );
    // Freshness is the run's age -- not a fourth 0..1 number, and not
    // anything the other three cells can stand in for.
    expect(screen.getByTestId("freshness")).toHaveTextContent("3 h ago");
    expect(screen.queryByTestId("disclosure-flag")).toBeNull();

    // Stale is a fact about the RUN, so it flags even though every factor in
    // this hour resolved and every one of them is measured.
    rerender(
      <Disclosure hour={fullyConstrained()} freshness={FRESHNESS} now={now} stale />,
    );
    expect(screen.getByTestId("disclosure")).toHaveAttribute("data-tone", "flagged");
    expect(screen.getByTestId("disclosure-flag")).toHaveTextContent(/rebuild/i);
  });

  it("renders the frame with no hour, and never invents a number for it", () => {
    render(<Disclosure hour={null} freshness={null} />);
    expect(screen.getByTestId("disclosure")).toHaveAttribute("data-tone", "waiting");
    expect(screen.getByTestId("confidence")).toHaveTextContent("—");
    expect(screen.getByTestId("confidence")).not.toHaveTextContent("0");
    expect(screen.queryByTestId("disclosure-flag")).toBeNull();
  });
});

describe("age", () => {
  const now = new Date("2026-09-03T12:00:00+00:00");
  const at = (iso: string) => age(iso, now);

  it("counts in the unit a person would use, at each boundary", () => {
    expect(at("2026-09-03T11:59:40+00:00")).toBe("just now");
    expect(at("2026-09-03T11:18:00+00:00")).toBe("42 min ago");
    expect(at("2026-09-03T09:00:00+00:00")).toBe("3 h ago");
    expect(at("2026-08-31T12:00:00+00:00")).toBe("3 d ago");
  });

  it("returns null rather than a fake age for a missing or unparseable stamp", () => {
    expect(at("not a timestamp")).toBeNull();
    expect(age(null, now)).toBeNull();
    expect(age(undefined, now)).toBeNull();
  });

  it("says 'just now' rather than a negative age when the clocks disagree", () => {
    // The stamp is the SERVER's; a browser clock a few minutes behind it
    // must not produce "in 4 minutes".
    expect(at("2026-09-03T12:04:00+00:00")).toBe("just now");
  });
});
