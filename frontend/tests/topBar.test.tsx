import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "../fixtures/day-payload.json";
import type { FisherySummary } from "../src/api/types";
import { DayProvider } from "../src/state/DayContext";
import { TopBar, WEATHER_MODELS, dayRange } from "../src/ui/TopBar";

const FISHERIES: FisherySummary[] = [
  {
    slug: "winyah-bay",
    name: "Winyah Bay",
    center: [-79.2, 33.2],
    timezone: "America/New_York",
    ready: true,
  },
  {
    slug: "charleston-harbor",
    name: "Charleston Harbor",
    center: [-79.9, 32.7],
    timezone: "America/New_York",
    ready: true,
  },
  {
    slug: "st-helena",
    name: "St Helena Sound",
    center: [-80.5, 32.5],
    timezone: "America/New_York",
    ready: false,
    reason: "no flow library",
  },
];

/** Every fetch the bar or the provider can make, stubbed. */
async function stubApi() {
  const client = await import("../src/api/client");
  const fetchDay = vi.spyOn(client, "fetchDay").mockResolvedValue({
    kind: "ready",
    payload: fixture as never,
  });
  vi.spyOn(client, "fetchStatus").mockResolvedValue({
    status: "ready",
    generated_at: "2026-09-03T23:00:27.675191+00:00",
    stale: false,
  });
  const fetchFisheries = vi.spyOn(client, "fetchFisheries").mockResolvedValue(FISHERIES);
  return { fetchDay, fetchFisheries };
}

/** The bar inside a REAL provider: the refetch contract is the provider's. */
async function mount(onFisheryChange = vi.fn<(slug: string) => void>()) {
  const api = await stubApi();
  render(
    <DayProvider slug="winyah-bay" initialDate="2026-09-03">
      <TopBar onFisheryChange={onFisheryChange} />
    </DayProvider>,
  );
  // The day and the fishery list both land before anything is asserted.
  await waitFor(() => expect(screen.getAllByTestId("species-option")).toHaveLength(3));
  await waitFor(() => expect(screen.getAllByTestId("fishery-option")).toHaveLength(3));
  return { ...api, onFisheryChange };
}

afterEach(() => vi.restoreAllMocks());

describe("TopBar", () => {
  it("disables an unready fishery and names what it is missing", async () => {
    await mount();
    const options = screen.getAllByTestId("fishery-option") as HTMLOptionElement[];
    const unready = options.find((option) => option.value === "st-helena");
    const ready = options.find((option) => option.value === "charleston-harbor");

    expect(unready).toBeDefined();
    expect(unready).toBeDisabled();
    // The `reason` field names what is missing -- a flow library, a distance
    // field, a feature inventory -- and it is the whole point of greying the
    // row. Both the visible label and the title carry it.
    expect(unready).toHaveTextContent(/no flow library/i);
    expect(unready).toHaveAttribute("title", expect.stringContaining("no flow library"));

    // The other half: a picker that disabled EVERY option would pass the
    // assertions above and let nobody switch fisheries.
    expect(ready).toBeDefined();
    expect(ready).not.toBeDisabled();
  });

  it("reports the chosen fishery's slug, once, when it changes", async () => {
    const { onFisheryChange } = await mount();
    fireEvent.change(screen.getByTestId("fishery-picker"), {
      target: { value: "charleston-harbor" },
    });
    expect(onFisheryChange).toHaveBeenCalledTimes(1);
    expect(onFisheryChange).toHaveBeenCalledWith("charleston-harbor");
  });

  it("still offers the current fishery when the list cannot be fetched", async () => {
    // Fail soft: losing /api/fisheries must not strip the bar of the fishery
    // it is already showing.
    const client = await import("../src/api/client");
    vi.spyOn(client, "fetchDay").mockResolvedValue({ kind: "ready", payload: fixture as never });
    vi.spyOn(client, "fetchFisheries").mockRejectedValue(new Error("network down"));

    render(
      <DayProvider slug="winyah-bay" initialDate="2026-09-03">
        <TopBar onFisheryChange={vi.fn()} />
      </DayProvider>,
    );

    await waitFor(() => expect(screen.getAllByTestId("fishery-option")).toHaveLength(1));
    expect(screen.getByTestId("fishery-picker")).toHaveValue("winyah-bay");
  });

  it("offers exactly the models the API accepts, and no invented ones", async () => {
    // `?model=` becomes part of a filename on the backend; anything outside
    // this set is a 422, and the validation that produces it closed a real
    // path-traversal. A picker that offered "gefs" would 422 every request.
    await mount();
    const values = (screen.getAllByTestId("model-option") as HTMLOptionElement[]).map(
      (option) => option.value,
    );
    expect(values.slice().sort()).toEqual(["best", "ecmwf", "gfs", "hrrr", "icon", "nbm"]);
    expect(WEATHER_MODELS.map((model) => model.value).slice().sort()).toEqual(
      ["best", "ecmwf", "gfs", "hrrr", "icon", "nbm"],
    );
  });

  it("constrains the date picker to the usable range up front", async () => {
    // Spec §7: the 422 is the BACKSTOP. A picker with no bounds hands a
    // person a date the API will refuse.
    await mount();
    const picker = screen.getByTestId("date-picker");
    const range = dayRange(new Date());
    expect(picker).toHaveValue("2026-09-03");
    expect(picker).toHaveAttribute("max", range.max);
    expect(picker).toHaveAttribute("min", range.min);
  });

  it("bounds the range at the forecast horizon, not at some other number", async () => {
    // The horizon mirrors the backend's FORECAST_HORIZON_DAYS = 16, measured
    // from the local day. Both ends stated: a range open at either end is
    // the defect this replaces.
    const range = dayRange(new Date(2026, 8, 3, 12, 0, 0));
    expect(range.max).toBe("2026-09-19");
    expect(range.min).toBe("2025-09-03");
    // Month and year rollover, where naive date arithmetic breaks.
    expect(dayRange(new Date(2026, 11, 28, 23, 30, 0)).max).toBe("2027-01-13");
  });

  it("switches species without refetching, but refetches on a model change", async () => {
    // The asymmetry is the point. Every species is already scored in the
    // payload, so a species change is a state set; a model change is a
    // different scoring run and must go to the API. A test asserting only
    // one half passes against a bar that refetches on everything, or on
    // nothing.
    const { fetchDay } = await mount();
    const afterLoad = fetchDay.mock.calls.length;

    const trout = screen
      .getAllByTestId("species-option")
      .find((button) => button.textContent?.includes("speckled trout"));
    expect(trout).toBeDefined();
    fireEvent.click(trout!);

    // The selection genuinely moved -- otherwise "no refetch" is trivially
    // true because nothing happened at all.
    await waitFor(() => expect(trout).toHaveAttribute("aria-pressed", "true"));
    expect(fetchDay.mock.calls.length).toBe(afterLoad);

    fireEvent.change(screen.getByTestId("model-picker"), { target: { value: "hrrr" } });
    await waitFor(() => expect(fetchDay.mock.calls.length).toBe(afterLoad + 1));
    expect(fetchDay).toHaveBeenLastCalledWith("winyah-bay", "2026-09-03", "hrrr");
  });

  it("refetches on a date change, for the date that was picked", async () => {
    const { fetchDay } = await mount();
    const afterLoad = fetchDay.mock.calls.length;
    fireEvent.change(screen.getByTestId("date-picker"), { target: { value: "2026-09-05" } });
    await waitFor(() => expect(fetchDay.mock.calls.length).toBe(afterLoad + 1));
    expect(fetchDay).toHaveBeenLastCalledWith("winyah-bay", "2026-09-05", "best");
  });

  it("shows the hour's disclosure beside the pickers", async () => {
    // The bar is where a person reads how much to trust what they are
    // looking at, so the four signals ride with it rather than living in a
    // badge someone has to remember to check.
    await mount();
    expect(screen.getByTestId("disclosure")).toHaveAttribute("data-tone", "flagged");
    expect(screen.getByTestId("confidence")).toHaveTextContent("1.00");
    expect(screen.getByTestId("constrained-share")).toHaveTextContent("0.92");
    expect(screen.getByTestId("provisional")).toHaveTextContent("salinity");
  });
});
