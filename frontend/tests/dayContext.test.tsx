import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "../fixtures/day-payload.json";
import { DayProvider, useDay } from "../src/state/DayContext";
import type { DayContextValue } from "../src/state/DayContext";

afterEach(() => vi.restoreAllMocks());

function Probe() {
  const { state, payload, hour, species } = useDay();
  return (
    <div>
      <span data-testid="state">{state}</span>
      <span data-testid="hour">{hour}</span>
      <span data-testid="species">{species}</span>
      <span data-testid="missing">{payload?.missing.join(",") ?? ""}</span>
    </div>
  );
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
