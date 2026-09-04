/**
 * The two optional overlays.
 *
 * Every case here is a PAIR. "The arrows point somewhere" is satisfied by a
 * field that points the same way on flood and on ebb, which is the exact
 * defect that makes an arrow overlay worse than none; "flood and ebb point
 * opposite ways" is not. Likewise the salinity cases: a single render that
 * shows a badge proves nothing about the render that does not, so each one
 * asserts the fitted case AND the unfitted case together.
 */
import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SalinityInset } from "../src/map/MapView";
import {
  arrowCollection,
  badgeFor,
  classRange,
  fetchSalinityField,
  REFERENCE_SPEED_MS,
  SALINITY_CLASS_COUNT,
  SALINITY_CLASS_PPT,
  useOverlay,
  type FlowField,
  type OverlayRequest,
  type SalinitySection,
} from "../src/map/overlays";

// --- the current field ---------------------------------------------------

/** A 2x2 field over one degree of the Winyah latitudes. */
function field(u: number[], v: number[], rows = 2, cols = 2): FlowField {
  return { hour: 12, rows, cols, bbox: [-79.4, 33.1, -79.2, 33.4], u, v };
}

function shaft(feature: { geometry: { coordinates: [number, number][][] } }) {
  const line = feature.geometry.coordinates[0];
  if (!line) throw new Error("no shaft");
  const [tail, tip] = line;
  if (!tail || !tip) throw new Error("shaft is not a segment");
  return { tail, tip };
}

/** Compass bearing of a shaft, in degrees, correcting for the meridian squeeze. */
function bearing(feature: { geometry: { coordinates: [number, number][][] } }): number {
  const { tail, tip } = shaft(feature);
  const lat = ((tail[1] + tip[1]) / 2) * (Math.PI / 180);
  const east = (tip[0] - tail[0]) * Math.cos(lat);
  const north = tip[1] - tail[1];
  return (((Math.atan2(east, north) * 180) / Math.PI) + 360) % 360;
}

/** The cell an arrow is drawn on: the shaft's midpoint, since arrows are centred. */
function centre(feature: { geometry: { coordinates: [number, number][][] } }): [number, number] {
  const { tail, tip } = shaft(feature);
  return [(tail[0] + tip[0]) / 2, (tail[1] + tip[1]) / 2];
}

/** Ground length of a shaft, in metres. */
function metres(feature: { geometry: { coordinates: [number, number][][] } }): number {
  const { tail, tip } = shaft(feature);
  const lat = ((tail[1] + tip[1]) / 2) * (Math.PI / 180);
  return Math.hypot(
    (tip[0] - tail[0]) * Math.cos(lat) * 111_320,
    (tip[1] - tail[1]) * 110_574,
  );
}

describe("arrowCollection", () => {
  it("reverses every arrow when the field reverses", () => {
    // THE case this overlay exists for. A reader turns it on to see the bay
    // fill and then empty; arrows that pointed the same way at both hours
    // would pass every "does it draw?" check and be actively misleading.
    const flood = arrowCollection(field([0.2, 0.1, 0.05, 0.3], [0.3, 0.25, 0.2, 0.1]));
    const ebb = arrowCollection(field([-0.2, -0.1, -0.05, -0.3], [-0.3, -0.25, -0.2, -0.1]));
    expect(flood.features).toHaveLength(4);
    expect(ebb.features).toHaveLength(4);
    for (let i = 0; i < 4; i += 1) {
      const a = flood.features[i];
      const b = ebb.features[i];
      if (!a || !b) throw new Error("missing arrow");
      const turned = (((bearing(a) - bearing(b)) % 360) + 360) % 360;
      expect(Math.abs(turned - 180)).toBeLessThan(0.5);
    }
  });

  it("puts row 0 at the north edge and the last row at the south", () => {
    // A flipped row axis draws a field that is a mirror of the estuary: every
    // arrow lands on the wrong side of the channel while the picture still
    // looks like a plausible current. The pair is what catches it -- one
    // corner alone is satisfied by a grid read upside down.
    const grid = arrowCollection(field([0.3, 0, 0, 0.3], [0, 0, 0, 0]));
    expect(grid.features).toHaveLength(2);
    const [first, last] = grid.features;
    if (!first || !last) throw new Error("missing corner");
    expect(centre(first)[1]).toBeGreaterThan(centre(last)[1]);
    // ...and against the bbox itself, not just against each other. The bbox
    // corners are cell CENTRES, so a centred arrow sits exactly on them.
    expect(centre(first)).toEqual([expect.closeTo(-79.4, 4), expect.closeTo(33.4, 4)]);
    expect(centre(last)).toEqual([expect.closeTo(-79.2, 4), expect.closeTo(33.1, 4)]);
  });

  it("draws a fast arrow longer than a slow one, and caps it at the cell", () => {
    const grid = arrowCollection(field([0.02, REFERENCE_SPEED_MS * 4, 0, 0], [0, 0, 0, 0]));
    const [slow, fast] = grid.features;
    if (!slow || !fast) throw new Error("missing arrow");
    expect(metres(fast)).toBeGreaterThan(metres(slow) * 2);
    // The cap is what keeps a dense field readable: at four times the
    // reference speed the arrow is still shorter than the cell it sits in.
    const cellMetres = Math.min(0.2 * 111_320 * Math.cos(0.58), 0.3 * 110_574);
    expect(metres(fast)).toBeLessThan(cellMetres);
  });

  it("measures length on the ground, not in degrees", () => {
    // A degree of longitude at 33 N is 16% shorter than a degree of latitude.
    // Scaling both components by the same number draws every arrow sheared
    // toward east-west -- a heading error of up to 8 degrees, which is the
    // difference between "along the channel" and "across it".
    const east = arrowCollection(field([0.3, 0, 0, 0], [0, 0, 0, 0])).features[0];
    const north = arrowCollection(field([0, 0, 0, 0], [0.3, 0, 0, 0])).features[0];
    if (!east || !north) throw new Error("missing arrow");
    expect(bearing(east)).toBeCloseTo(90, 1);
    expect(bearing(north)).toBeCloseTo(0, 1);
    expect(metres(east)).toBeCloseTo(metres(north), 0);
  });

  it("draws a wet cell and skips a dry one", () => {
    // Dry and out-of-domain cells are exactly zero. Drawing them would put a
    // minimum-length arrow on every acre of marsh in the bounding box.
    const mixed = arrowCollection(field([0.25, 0, 0, 0], [0, 0, 0, 0]));
    expect(mixed.features).toHaveLength(1);
    expect(arrowCollection(field([0, 0, 0, 0], [0, 0, 0, 0])).features).toHaveLength(0);
  });
});

// --- the salinity section ------------------------------------------------

function salinityResponse(
  ppt: number[],
  flags: { fitted: boolean; extrapolated: boolean },
): unknown {
  return {
    hour: 12,
    fitted: flags.fitted,
    extrapolated: flags.extrapolated,
    cells: ppt.map((value, km) => ({ km, ppt: value })),
  };
}

function stubFetch(body: unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  );
}

const request: OverlayRequest = {
  slug: "winyah-bay",
  date: "2026-09-02",
  hour: 12,
  model: "best",
};

afterEach(() => vi.unstubAllGlobals());

describe("fetchSalinityField", () => {
  it("hands on a class index and never a salinity", async () => {
    // THE structural guarantee. A smooth field and a crisp isoline both need
    // continuous values; this asserts there are none to be had. A future
    // renderer cannot slip into painting one, because the numbers it would
    // need did not survive the fetch.
    stubFetch(salinityResponse([35.49, 26.921, 0.06], { fitted: false, extrapolated: false }));
    const section = await fetchSalinityField(request);
    for (const band of section.bands) {
      expect(Object.keys(band).sort()).toEqual(["klass", "km"]);
      expect(Number.isInteger(band.klass)).toBe(true);
      expect(band.klass).toBeGreaterThanOrEqual(0);
      expect(band.klass).toBeLessThan(SALINITY_CLASS_COUNT);
    }
    expect(JSON.stringify(section)).not.toContain("35.49");
    expect(JSON.stringify(section)).not.toContain("26.92");
  });

  it("collapses a within-class difference and keeps a between-class one", () => {
    // Quantisation has to be real in both directions. Rounding that never
    // merges anything is a smooth field wearing an integer's name.
    const near = async () =>
      fetchSalinityField(request).then((s) => s.bands.map((b) => b.klass));
    stubFetch(salinityResponse([30.0], { fitted: false, extrapolated: false }));
    const a = near();
    stubFetch(salinityResponse([30.0 + SALINITY_CLASS_PPT * 0.4], { fitted: false, extrapolated: false }));
    const b = near();
    stubFetch(salinityResponse([30.0 + SALINITY_CLASS_PPT * 1.4], { fitted: false, extrapolated: false }));
    const c = near();
    return Promise.all([a, b, c]).then(([one, two, three]) => {
      expect(two).toEqual(one);
      expect(three).not.toEqual(one);
    });
  });

  it("keeps far more input values than it has classes to put them in", async () => {
    // Winyah's real section is 38 bins of continuous ppt. If the output had
    // as many distinct heights as the input had values, nothing was quantised.
    const ppt = Array.from({ length: 38 }, (_, i) => 35.5 * Math.exp(-i / 8));
    stubFetch(salinityResponse(ppt, { fitted: false, extrapolated: false }));
    const section = await fetchSalinityField(request);
    expect(new Set(ppt).size).toBe(38);
    expect(new Set(section.bands.map((b) => b.klass)).size).toBeLessThanOrEqual(
      SALINITY_CLASS_COUNT,
    );
  });

  it("steps the section identically whether or not the model is fitted", async () => {
    // The pair that closes the defect this constraint exists to prevent. If a
    // `fitted: true` response could take a different, smoother path, this is
    // where it would show -- there is no such path, so both are stepped and
    // only the badge differs.
    const ppt = [35.4, 30.1, 12.6, 0.4];
    stubFetch(salinityResponse(ppt, { fitted: false, extrapolated: false }));
    const unfitted = await fetchSalinityField(request);
    stubFetch(salinityResponse(ppt, { fitted: true, extrapolated: false }));
    const fitted = await fetchSalinityField(request);
    expect(fitted.bands).toEqual(unfitted.bands);
    expect(unfitted.badge).toBe("UNCALIBRATED");
    expect(fitted.badge).toBe("MODELLED");
  });

  it("throws rather than returning an undrawable section", async () => {
    stubFetch(salinityResponse([], { fitted: false, extrapolated: false }));
    await expect(fetchSalinityField(request)).rejects.toThrow(/no cells/);
    stubFetch({ detail: "no along-estuary distance field" }, 404);
    await expect(fetchSalinityField(request)).rejects.toThrow(/distance field/);
  });
});

describe("badgeFor", () => {
  it("names the falsified model before it names the range", () => {
    // Extrapolation is a claim about a fitted model's domain. For a model
    // nothing constrains, "outside the fitted range" is the smaller of two
    // problems and must not be the one on the badge.
    expect(badgeFor(false, false)).toBe("UNCALIBRATED");
    expect(badgeFor(false, true)).toBe("UNCALIBRATED");
    expect(badgeFor(true, true)).toBe("EXTRAPOLATED");
    expect(badgeFor(true, false)).toBe("MODELLED");
  });
});

// --- the section, drawn --------------------------------------------------

function section(over: Partial<SalinitySection> = {}): SalinitySection {
  return {
    hour: 12,
    fitted: false,
    extrapolated: false,
    badge: "UNCALIBRATED",
    bands: [0, 4, 9, 14].map((klass, km) => ({ km, klass })),
    kmMin: 0,
    kmMax: 3,
    ...over,
  };
}

describe("SalinityInset", () => {
  it("carries its badge whether the model is fitted or not", () => {
    const { unmount } = render(<SalinityInset section={section()} />);
    expect(screen.getByTestId("salinity-badge")).toHaveTextContent("UNCALIBRATED");
    unmount();
    render(<SalinityInset section={section({ fitted: true, badge: "MODELLED" })} />);
    // Still badged, still stepped -- the fitted branch is not a smooth one.
    expect(screen.getByTestId("salinity-badge")).toHaveTextContent("MODELLED");
    expect(
      screen.getByTestId("salinity-section").querySelectorAll("[data-klass]"),
    ).toHaveLength(4);
  });

  it("draws every band from a class index, so no band can hold a salinity", () => {
    render(<SalinityInset section={section()} />);
    const bands = screen.getByTestId("salinity-section").querySelectorAll("[data-klass]");
    expect(bands).toHaveLength(4);
    bands.forEach((band) => {
      const klass = Number(band.getAttribute("data-klass"));
      expect(Number.isInteger(klass)).toBe(true);
      // The height is the class, expressed as a fraction of the class count.
      // There is no other number on this element to draw a curve through.
      expect(band.getAttribute("style")).toContain(`--klass: ${klass}`);
    });
  });

  it("says what the model is, not only that it is a model", () => {
    render(<SalinityInset section={section()} />);
    expect(screen.getByTestId("salinity-section")).toHaveTextContent(/unfitted/i);
    expect(screen.getByTestId("salinity-section")).toHaveTextContent(
      new RegExp(`${SALINITY_CLASS_PPT} ppt class`),
    );
    expect(screen.getByTestId("salinity-section")).toHaveTextContent(/no contour/i);
  });

  it("labels the saltiest class as a range, never as a value", () => {
    render(<SalinityInset section={section()} />);
    const [low, high] = classRange(14);
    const label = screen.getByRole("img").getAttribute("aria-label") ?? "";
    expect(label).toContain(`${low} to ${high} ppt`);
    expect(label).toContain("UNCALIBRATED");
  });
});

// --- the toggle, the debounce and the revert -----------------------------

describe("useOverlay", () => {
  beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
  afterEach(() => vi.useRealTimers());

  /** A hook harness whose hour really moves -- freezing it would test nothing. */
  function harness(fetcher: (r: OverlayRequest) => Promise<string>) {
    return renderHook(
      ({ hour, date }: { hour: number; date: string }) =>
        useOverlay<string>(fetcher, { slug: "winyah-bay", date, hour, model: "best" }, 200),
      { initialProps: { hour: 0, date: "2026-09-02" } },
    );
  }

  it("issues nothing while it is off, and one request when it goes on", async () => {
    const calls: number[] = [];
    const fetcher = vi.fn(async (r: OverlayRequest) => {
      calls.push(r.hour);
      return `h${r.hour}`;
    });
    const { result, rerender } = harness(fetcher);

    for (let hour = 1; hour <= 6; hour += 1) rerender({ hour, date: "2026-09-02" });
    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    expect(calls).toEqual([]);

    await act(async () => {
      result.current.enable(true);
    });
    await waitFor(() => expect(result.current.data).toBe("h6"));
    expect(calls).toEqual([6]);
  });

  it("answers a fast drag with ONE request, and it carries the last hour", async () => {
    // The exception to "scrubbing never refetches" was bought with this
    // debounce. Asserting only the count would pass for a leading-edge
    // debounce that fetched hour 1 and drew it under a strip reading 12.
    const calls: number[] = [];
    const fetcher = vi.fn(async (r: OverlayRequest) => {
      calls.push(r.hour);
      return `h${r.hour}`;
    });
    const { result, rerender } = harness(fetcher);
    await act(async () => {
      result.current.enable(true);
    });
    await waitFor(() => expect(result.current.data).toBe("h0"));
    calls.length = 0;

    for (let hour = 1; hour <= 12; hour += 1) {
      rerender({ hour, date: "2026-09-02" });
      await act(async () => {
        vi.advanceTimersByTime(30);
      });
    }
    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    await waitFor(() => expect(result.current.data).toBe("h12"));
    expect(calls).toEqual([12]);
  });

  it("marks itself busy from the scrub until the answer, not from the request", async () => {
    const fetcher = vi.fn(async (r: OverlayRequest) => `h${r.hour}`);
    const { result, rerender } = harness(fetcher);
    await act(async () => {
      result.current.enable(true);
    });
    await waitFor(() => expect(result.current.busy).toBe(false));

    // Inside the debounce window nothing has been asked for yet -- and the
    // arrows on screen are still last hour's, which is what `busy` is for.
    rerender({ hour: 7, date: "2026-09-02" });
    expect(result.current.busy).toBe(true);
    expect(result.current.data).toBe("h0");
    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    await waitFor(() => expect(result.current.data).toBe("h7"));
    expect(result.current.busy).toBe(false);
  });

  it("reverts its own toggle when the fetch fails, and can be switched back on", async () => {
    let attempt = 0;
    const fetcher = vi.fn(async (r: OverlayRequest) => {
      attempt += 1;
      if (attempt === 1) throw new Error("no regime resolves for this day");
      return `h${r.hour}`;
    });
    const { result } = harness(fetcher);

    await act(async () => {
      result.current.enable(true);
    });
    await waitFor(() => expect(result.current.error).toMatch(/no regime/));
    expect(result.current.on).toBe(false);
    expect(result.current.data).toBeNull();
    expect(result.current.busy).toBe(false);

    await act(async () => {
      result.current.enable(true);
    });
    await waitFor(() => expect(result.current.data).toBe("h0"));
    expect(result.current.on).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it("drops another day's field instead of drawing it under the new date", async () => {
    const fetcher = vi.fn(async (r: OverlayRequest) => `${r.date}#${r.hour}`);
    const { result, rerender } = harness(fetcher);
    await act(async () => {
      result.current.enable(true);
    });
    await waitFor(() => expect(result.current.data).toBe("2026-09-02#0"));

    rerender({ hour: 0, date: "2026-09-03" });
    // The pair: the stale field is gone AND the panel says a request is out.
    // Keeping it would draw the 2nd's currents over the 3rd's chart.
    expect(result.current.data).toBeNull();
    expect(result.current.busy).toBe(true);
    await act(async () => {
      vi.advanceTimersByTime(400);
    });
    await waitFor(() => expect(result.current.data).toBe("2026-09-03#0"));
  });
});
