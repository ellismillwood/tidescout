import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchDay, layerUrl } from "../src/api/client";

afterEach(() => vi.unstubAllGlobals());

function stub(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  );
}

describe("fetchDay", () => {
  it("distinguishes a ready payload from a build in progress", async () => {
    stub(200, { slug: "winyah-bay" });
    const ready = await fetchDay("winyah-bay", "2026-09-01", "best");
    expect(ready.kind).toBe("ready");

    stub(202, { status: "building", started_at: "x", key: "y" });
    const building = await fetchDay("winyah-bay", "2026-09-01", "best");
    expect(building.kind).toBe("building");
  });

  it("does NOT treat a degraded payload as an error", async () => {
    // The rule the whole stack is built around: `missing` and a lowered
    // `confidence` are DATA. A client that threw here, or stripped them,
    // would undo the disclosure the backend was forbidden from breaking.
    stub(200, { slug: "winyah-bay", missing: ["weather"], conditions: [] });
    const res = await fetchDay("winyah-bay", "2026-09-01", "best");
    expect(res.kind).toBe("ready");
    if (res.kind !== "ready") throw new Error("unreachable");
    expect(res.payload.missing).toEqual(["weather"]);
  });

  it("throws on a real error status", async () => {
    stub(404, { detail: "unknown fishery" });
    await expect(fetchDay("nope", "2026-09-01", "best")).rejects.toThrow(/unknown fishery/);
  });
});

describe("layerUrl", () => {
  it("builds a path the API's allowlist accepts", () => {
    expect(layerUrl("winyah-bay", "oysters")).toBe(
      "/api/fisheries/winyah-bay/layers/oysters",
    );
  });
});
