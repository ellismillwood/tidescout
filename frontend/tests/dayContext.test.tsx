import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { DayPayload } from "../src/api/types";
import { DayProvider, useDay } from "../src/state/DayContext";
import type { DayContextValue } from "../src/state/DayContext";
import { TopBar } from "../src/ui/TopBar";

afterEach(() => vi.restoreAllMocks());

function Probe() {
  const { state, payload, hour, species, stale } = useDay();
  return (
    <div>
      <span data-testid="state">{state}</span>
      <span data-testid="hour">{hour}</span>
      <span data-testid="species">{species}</span>
      <span data-testid="missing">{payload?.missing.join(",") ?? ""}</span>
      <span data-testid="stale">{String(stale)}</span>
      {/* WHICH payload is in state, not merely that one is: the stale run and
          the rebuilt one differ only by this stamp. */}
      <span data-testid="generated">{payload?.freshness.generated_at ?? ""}</span>
    </div>
  );
}

/** The committed payload with a given run stamp on it. */
function runStamped(generatedAt: string): DayPayload {
  const payload = structuredClone(fixture) as unknown as DayPayload;
  payload.freshness = { ...payload.freshness, generated_at: generatedAt };
  return payload;
}

function renderWith() {
  return render(
    <DayProvider slug="winyah-bay" initialDate="2026-09-01">
      <Probe />
    </DayProvider>,
  );
}

describe("DayProvider", () => {
  it("goes building -> ready when the first call is a 202", async () => {
    // Both halves matter: a provider that never left "building" and one that
    // never entered it would each pass a single-state assertion.
    const client = await import("../src/api/client");
    vi.spyOn(client, "fetchDay")
      .mockResolvedValueOnce({ kind: "building" })
      .mockResolvedValue({ kind: "ready", payload: fixture as never });
    vi.spyOn(client, "fetchStatus").mockResolvedValue({
      status: "ready", generated_at: "x", stale: false,
    });

    renderWith();
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("building"));
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("ready"), {
      timeout: 5000,
    });
  });

  it("surfaces a failed build with its message instead of hanging", async () => {
    const client = await import("../src/api/client");
    vi.spyOn(client, "fetchDay").mockResolvedValue({ kind: "building" });
    vi.spyOn(client, "fetchStatus").mockResolvedValue({
      status: "failed", error: "RuntimeError: USGS timed out",
    });

    renderWith();
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("failed"), {
      timeout: 5000,
    });
  });

  it("treats a degraded payload as READY, keeping its disclosure", async () => {
    const client = await import("../src/api/client");
    const degraded = { ...(fixture as never as object), missing: ["weather"] };
    vi.spyOn(client, "fetchDay").mockResolvedValue({
      kind: "ready", payload: degraded as never,
    });

    renderWith();
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("ready"));
    expect(screen.getByTestId("missing")).toHaveTextContent("weather");
  });

  it("defaults to hour 0 and the first species, and both are settable", async () => {
    const client = await import("../src/api/client");
    vi.spyOn(client, "fetchDay").mockResolvedValue({
      kind: "ready", payload: fixture as never,
    });

    renderWith();
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("ready"));
    expect(screen.getByTestId("hour")).toHaveTextContent("0");
    expect(screen.getByTestId("species")).toHaveTextContent("redfish");
  });

  it("picks the background rebuild up behind a stale cache hit, and clears the flag", async () => {
    // BOTH halves, in one test, deliberately. `/day` serves a stale payload
    // and rebuilds behind it, and the two answers are indistinguishable from
    // this side -- so a context that never asked `/status` again would show
    // yesterday's run forever. Asserting only the flag would pass against a
    // UI that reports staleness forever and never refreshes; asserting only
    // the swap would pass against one that swaps in silence, telling nobody
    // the numbers they were reading were out of date.
    const OLD = "2026-09-01T02:00:00+00:00";
    const NEW = "2026-09-03T18:30:00+00:00";
    const client = await import("../src/api/client");
    const fetchDay = vi
      .spyOn(client, "fetchDay")
      .mockResolvedValueOnce({ kind: "ready", payload: runStamped(OLD) })
      .mockResolvedValue({ kind: "ready", payload: runStamped(NEW) });
    vi.spyOn(client, "fetchStatus")
      .mockResolvedValueOnce({ status: "ready", generated_at: OLD, stale: true })
      .mockResolvedValue({ status: "ready", generated_at: NEW, stale: false });
    vi.spyOn(client, "fetchFisheries").mockResolvedValue([]);

    // The REAL TopBar, so this covers the wiring too: `stale` on the context
    // is worth nothing if nothing passes it to the disclosure.
    render(
      <DayProvider slug="winyah-bay" initialDate="2026-09-01">
        <TopBar onFisheryChange={() => {}} />
        <Probe />
      </DayProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("stale")).toHaveTextContent("true"));
    expect(screen.getByTestId("generated")).toHaveTextContent(OLD);
    expect(screen.getByTestId("disclosure-flag")).toHaveTextContent(/rebuild/i);

    // The rebuild lands: the NEW payload is what is in state afterwards...
    await waitFor(() => expect(screen.getByTestId("generated")).toHaveTextContent(NEW), {
      timeout: 5000,
    });
    // ...and the flag is gone. (The band still flags this hour's provisional
    // salinity -- that is a different sentence, and it is not this one.)
    expect(screen.getByTestId("stale")).toHaveTextContent("false");
    expect(screen.queryByTestId("disclosure-flag")?.textContent ?? "").not.toMatch(
      /rebuild/i,
    );
    expect(fetchDay.mock.calls.length).toBe(2);
  });

  it("asks /status once on a FRESH cache hit and refetches nothing", async () => {
    // The other side of the call above: the confirming check must not become
    // a second download of the 1.67 MB payload that just arrived.
    const client = await import("../src/api/client");
    const fetchDay = vi.spyOn(client, "fetchDay").mockResolvedValue({
      kind: "ready", payload: fixture as never,
    });
    const fetchStatus = vi.spyOn(client, "fetchStatus").mockResolvedValue({
      status: "ready", generated_at: "x", stale: false,
    });

    renderWith();
    await waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("stale")).toHaveTextContent("false");
    expect(fetchDay).toHaveBeenCalledTimes(1);
  });

  it("keeps the hour and the fish across a MODEL change, and resets on a DATE change", async () => {
    // The whole point of the model picker is reading the same hour under two
    // models -- Winyah's own hours go confidence 1.00 / share 0.92 under one
    // and 0.92 / 1.00 under another. A reset to hour 0 and the first species
    // makes that comparison impossible to see. A date change is the opposite
    // case: different hours, different day, start at the top. Both directions
    // are asserted here, because a context that never reset and one that
    // always reset would each pass half of this test.
    const client = await import("../src/api/client");
    vi.spyOn(client, "fetchDay").mockResolvedValue({
      kind: "ready", payload: fixture as never,
    });
    vi.spyOn(client, "fetchStatus").mockResolvedValue({
      status: "ready", generated_at: "x", stale: false,
    });

    let ctx: DayContextValue | null = null;
    function Capture() {
      ctx = useDay();
      return null;
    }
    render(
      <DayProvider slug="winyah-bay" initialDate="2026-09-01">
        <Capture />
        <Probe />
      </DayProvider>,
    );
    await waitFor(() => expect(ctx?.state).toBe("ready"));

    act(() => ctx?.setHour(14));
    act(() => ctx?.setSpecies("speckled_trout"));

    act(() => ctx?.setModel("ecmwf"));
    await waitFor(() => expect(ctx?.model).toBe("ecmwf"));
    // The payload landed (a new object in state) and the selection survived
    // it -- not merely "nothing re-rendered".
    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("ready"));
    expect(screen.getByTestId("hour")).toHaveTextContent("14");
    expect(screen.getByTestId("species")).toHaveTextContent("speckled_trout");

    act(() => ctx?.setDate("2026-09-02"));
    await waitFor(() => expect(screen.getByTestId("hour")).toHaveTextContent("0"));
    expect(screen.getByTestId("species")).toHaveTextContent("redfish");
  });

  it("falls back to the first species when a preserved one is not in the new payload", async () => {
    // A model change preserves the species BY NAME, and nothing guarantees
    // the next run scores the same set. Left alone, `species` would name
    // something the payload has no block for and the rail, the map and the
    // strip would all read empty for a day that scored perfectly well.
    const client = await import("../src/api/client");
    const narrowed = structuredClone(fixture) as unknown as DayPayload;
    for (const name of Object.keys(narrowed.species)) {
      if (name !== "redfish") delete narrowed.species[name];
    }
    vi.spyOn(client, "fetchDay")
      .mockResolvedValueOnce({ kind: "ready", payload: fixture as never })
      .mockResolvedValue({ kind: "ready", payload: narrowed });
    vi.spyOn(client, "fetchStatus").mockResolvedValue({
      status: "ready", generated_at: "x", stale: false,
    });

    let ctx: DayContextValue | null = null;
    function Capture() {
      ctx = useDay();
      return null;
    }
    render(
      <DayProvider slug="winyah-bay" initialDate="2026-09-01">
        <Capture />
        <Probe />
      </DayProvider>,
    );
    await waitFor(() => expect(ctx?.state).toBe("ready"));
    act(() => ctx?.setHour(9));
    act(() => ctx?.setSpecies("speckled_trout"));

    act(() => ctx?.setModel("ecmwf"));
    await waitFor(() => expect(screen.getByTestId("species")).toHaveTextContent("redfish"));
    // The hour is still preserved: only the unanswerable half was dropped.
    expect(screen.getByTestId("hour")).toHaveTextContent("9");
  });

  it("does not refetch on hour/species scrubs, but does refetch on a model change", async () => {
    // The contrast is the point: a context that never fetched at all would
    // pass a test asserting only the first two, and one that refetches on
    // every state change would pass a test asserting only the third. Call
    // count (not a boolean) also tells "fetched once more" apart from
    // "fetched five more times", which a boolean would blur together.
    const client = await import("../src/api/client");
    const fetchDaySpy = vi.spyOn(client, "fetchDay").mockResolvedValue({
      kind: "ready", payload: fixture as never,
    });
    vi.spyOn(client, "fetchStatus").mockResolvedValue({
      status: "ready", generated_at: "x", stale: false,
    });

    let ctx: DayContextValue | null = null;
    function Capture() {
      ctx = useDay();
      return null;
    }

    render(
      <DayProvider slug="winyah-bay" initialDate="2026-09-01">
        <Capture />
      </DayProvider>,
    );

    await waitFor(() => expect(ctx?.state).toBe("ready"));
    const callsAfterLoad = fetchDaySpy.mock.calls.length;

    act(() => ctx?.setHour(5));
    expect(fetchDaySpy.mock.calls.length).toBe(callsAfterLoad);

    act(() => ctx?.setSpecies("speckled_trout"));
    expect(fetchDaySpy.mock.calls.length).toBe(callsAfterLoad);

    act(() => ctx?.setModel("fallback"));
    await waitFor(() =>
      expect(fetchDaySpy.mock.calls.length).toBeGreaterThan(callsAfterLoad),
    );
  });
});
