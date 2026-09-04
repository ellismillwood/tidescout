/**
 * The end-to-end smoke: a real API, a real browser, a real map, a real day.
 *
 * Every other test in this repo is structurally blind to one thing. The
 * backend suite's client never mounts the frontend (which is how two routes
 * shipped unreachable, shadowed by the SPA catch-all). The Vitest suite runs
 * in jsdom, which has no WebGL, so `layers.ts` gets its expressions parsed
 * against the style spec but nothing ever draws them (which is how a 404'ing
 * worker URL left every GeoJSON layer rendering zero features while the raster
 * tiles still painted, and the app still looked like a map). This file exists
 * for that class and only that class, so it mocks nothing: no stubbed fetch,
 * no fixture payload, no fake map.
 *
 * THREE THINGS EVERY ASSERTION HERE OBEYS.
 *
 * 1. PAIRS, NOT POINTS. "A screenshot was taken" passes against a frozen map.
 *    So the map assertions are round trips: scrub 00 -> 18 and the pixels MUST
 *    change; scrub back to 00 and they MUST return byte-identical. The second
 *    half is what makes the first half mean something -- it proves the change
 *    came from the hour and not from a basemap tile landing between the two
 *    shots. Same shape for the species switch.
 *
 * 2. WAITS ARE ON OBSERVABLE STATE. `retries: 0` in the config is a promise
 *    that a failure here is real, and that is only true if nothing waits on a
 *    fixed sleep and hopes. The one loop that does sleep -- `settle()` --
 *    sleeps between OBSERVATIONS and exits on a condition (two identical
 *    frames), which is the difference between waiting for the map to stop and
 *    guessing how long it takes.
 *
 * 3. NOTHING PASSES VACUOUSLY. A warmed day and a building day both render 24
 *    bars, so `toHaveCount(24)` alone would pass against an empty strip:
 *    `loadWarmedDay` also requires the bars to be ENABLED, which is the
 *    component's own "payload and species are here" condition. The
 *    no-refetch test records the day endpoint's hit count BEFORE the species
 *    switch and requires it to be non-zero -- a URL pattern that matched
 *    nothing would report zero at both ends and pass while testing nothing.
 */
import { expect, test, type Page } from "@playwright/test";

/** `/api/fisheries/{slug}/day/{date}` -- the payload itself, not its status. */
const DAY_PATH = /^\/api\/fisheries\/[^/]+\/day\/[^/]+$/;
/** The same, plus `/status`: everything the brief means by `day/*`. */
const ANY_DAY_PATH = /^\/api\/fisheries\/[^/]+\/day\//;

/** Ten: seven that vary only by hour, three that vary by feature. */
const TOTAL_FACTORS = 10;

/**
 * The app, showing a day that is actually scored.
 *
 * The enabled check is the load-bearing one. `HourStrip` renders its 24
 * columns unconditionally -- a day still building draws 24 disabled bars over
 * an empty tide band -- so counting bars proves the strip mounted and nothing
 * more. `disabled={!hasData}` is the component's own statement that a payload
 * AND a species arrived, which is the thing this suite needs to be true before
 * any of it means anything.
 */
async function loadWarmedDay(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByTestId("hour-bar")).toHaveCount(24);
  await expect(page.getByTestId("hour-bar").first()).toBeEnabled({ timeout: 90_000 });
  // The map has placed its view and shown itself: `data-revealed` flips only
  // after the bounds sidecar (or the feature collection) has landed.
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-revealed", "true");
  // Every tile, image layer and GeoJSON layer for this fixed view has been
  // fetched. The view never moves in this suite, so MapLibre never asks for
  // another tile after this point -- which is what makes the round-trip pixel
  // comparisons below attributable to the selection and nothing else.
  await page.waitForLoadState("networkidle");
}

/**
 * The DOM that sits ON TOP of the map canvas, masked out of every shot below.
 *
 * THIS LIST IS THE DIFFERENCE BETWEEN THIS SUITE TESTING THE MAP AND TESTING
 * NOTHING. `locator.screenshot()` does not read the element's own pixels -- it
 * takes a viewport shot and clips it to the element's box -- so the map key,
 * the overlay switches and MapLibre's own controls all land inside a shot of
 * the canvas. The key prints the selected species and hour as TEXT ("redfish ·
 * 00:00"), which means an unmasked before/after comparison changes on every
 * scrub whether or not a single marker was repainted.
 *
 * Verified by mutation, not by reading: with the two `setPaintProperty` calls
 * in `MapView`'s scrub loop commented out -- the entire mechanism this suite
 * exists to protect -- the unmasked version of these tests still passed, on
 * the strength of a label. Masked, they fail.
 */
const mapChrome = (page: Page) => [
  page.getByTestId("map-key"),
  page.getByTestId("overlays"),
  page.locator(".maplibregl-control-container"),
];

const mapCanvas = (page: Page) => page.locator("[data-testid=map] canvas");

/** One frame of the map's own pixels, with every DOM overlay masked flat. */
const frame = (page: Page) => mapCanvas(page).screenshot({ mask: mapChrome(page) });

/**
 * The map's pixels, once it has stopped changing them.
 *
 * Two consecutive identical frames, not a sleep: a fixed wait long enough for
 * a cold tile cache is dead time on every other run, and a fixed wait tuned to
 * a warm one is a coin flip. Throws rather than returning a moving target,
 * because a map that never settles is a finding, not something to paper over.
 */
async function settle(page: Page): Promise<Buffer> {
  let previous = await frame(page);
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await page.waitForTimeout(250);
    const next = await frame(page);
    if (Buffer.compare(previous, next) === 0) return next;
    previous = next;
  }
  throw new Error("map canvas never settled: 40 frames, no two consecutive alike");
}

/**
 * Where on the canvas a SCORED marker is, found by hit-testing the real layer.
 *
 * This is a probe, not the interaction under test -- the click that follows is
 * a real `page.mouse.click`. It sweeps synthetic `mousemove`s across the
 * canvas and watches for the cursor to turn into a pointer, which `MapView`
 * sets only when `queryRenderedFeatures` returns a marker that HAS an
 * activation for the current species and hour. Three things fall out of that:
 * it runs entirely in-page (8,840 points in ~180 ms, versus a round trip per
 * point from Node), it cannot land on one of the 1,633 unscored features, and
 * -- because `queryRenderedFeatures` only ever returns geometry that actually
 * RENDERED -- a hit is direct proof the marker layer drew. That is the exact
 * assertion the worker-URL regression needed and jsdom cannot make.
 */
async function findScoredMarker(page: Page): Promise<{ x: number; y: number }> {
  const found = await page.evaluate(() => {
    const canvas = document.querySelector("[data-testid=map] canvas");
    if (!(canvas instanceof HTMLCanvasElement)) return null;
    const rect = canvas.getBoundingClientRect();
    // 5 px: markers run 3.5-12 px in radius, so a step under the smallest
    // diameter cannot step over one.
    const STEP = 5;
    for (let y = STEP; y < rect.height - STEP; y += STEP) {
      for (let x = STEP; x < rect.width - STEP; x += STEP) {
        const clientX = rect.left + x;
        const clientY = rect.top + y;
        canvas.dispatchEvent(
          new MouseEvent("mousemove", {
            clientX,
            clientY,
            bubbles: true,
            cancelable: true,
            view: window,
          }),
        );
        if (canvas.style.cursor === "pointer") return { x: clientX, y: clientY };
      }
    }
    return null;
  });
  expect(found, "no scored marker is hit-testable on the map canvas").not.toBeNull();
  return found as { x: number; y: number };
}

/**
 * `sub_scope` out of the live payload, without parsing 25 MB of JSON.
 *
 * Read from the payload rather than hardcoded here for the same reason
 * `FeaturePopover` reads it rather than keeping its own copy: the split
 * between the two scopes is the backend's to declare, and a list frozen in a
 * test would keep passing after the backend moved a factor -- while the
 * popover, which follows `sub_scope`, moved with it.
 */
function subScope(raw: string): { hour: string[]; feature: string[] } {
  const at = raw.indexOf('"sub_scope":');
  if (at < 0) throw new Error("payload has no sub_scope");
  const open = raw.indexOf("{", at);
  let depth = 0;
  for (let i = open; i < raw.length; i += 1) {
    if (raw[i] === "{") depth += 1;
    else if (raw[i] === "}") {
      depth -= 1;
      if (depth === 0) {
        return JSON.parse(raw.slice(open, i + 1)) as { hour: string[]; feature: string[] };
      }
    }
  }
  throw new Error("payload's sub_scope object never closes");
}

/** The `data-factor` of every row in one of the popover's two groups. */
async function factorsIn(page: Page, group: string): Promise<string[]> {
  return page
    .getByTestId(group)
    .getByTestId("factor-row")
    .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-factor") ?? ""));
}

test("loads a warmed day and scrubs", async ({ page }) => {
  await loadWarmedDay(page);
  await expect(page.getByTestId("hour-bar")).toHaveCount(24);

  // Before any pixel comparison: prove the marker layer actually drew. Without
  // this the round trip below would still pass on a map showing raster tiles
  // and nothing else -- there would simply be no scrub-driven change to find,
  // and `at18 === at00` would fail for a reason no one could read.
  await findScoredMarker(page);

  const at00 = await settle(page);

  await page.getByTestId("hour-bar").nth(18).click();
  // Observable state, not a sleep: the selection has moved before any repaint
  // is asked about.
  await expect(page.getByTestId("playhead")).toHaveAttribute("data-hour", "18");
  await expect(page.getByTestId("strip-readout")).toContainText("18:00");
  const at18 = await settle(page);

  // THE assertion. Identical pixels here mean `setPaintProperty` never ran --
  // the two expressions in the scrub loop are the entire mechanism this app is
  // built on, and a map that ignores them is a map that shows one hour forever.
  expect(
    Buffer.compare(at00, at18),
    "the map painted 18:00 exactly as it painted 00:00",
  ).not.toBe(0);

  // The other half of the pair. A basemap tile landing between the two shots
  // would also make them differ; only a return to byte-identical pixels shows
  // the difference tracked the HOUR.
  await page.getByTestId("hour-bar").nth(0).click();
  await expect(page.getByTestId("playhead")).toHaveAttribute("data-hour", "0");
  const back = await settle(page);
  expect(
    Buffer.compare(at00, back),
    "scrubbing back to 00:00 did not restore the original frame",
  ).toBe(0);
});

test("a marker popover shows all ten factors", async ({ page, request }) => {
  const dayRequests: string[] = [];
  page.on("request", (event) => {
    if (DAY_PATH.test(new URL(event.url()).pathname)) dayRequests.push(event.url());
  });

  await loadWarmedDay(page);
  const dayUrl = dayRequests[0];
  expect(dayUrl, "the day payload was never fetched").toBeDefined();

  // The same URL the app just used, asked again for one small field.
  const scope = subScope(await (await request.get(dayUrl as string)).text());
  expect(scope.hour.length + scope.feature.length).toBe(TOTAL_FACTORS);

  const marker = await findScoredMarker(page);
  await page.mouse.click(marker.x, marker.y);
  const popover = page.getByTestId("feature-popover");
  await expect(popover).toBeVisible();

  const hourFactors = await factorsIn(page, "factor-bars-hour");
  const featureFactors = await factorsIn(page, "factor-bars-feature");

  // The merge, checked against the payload's own declaration of it -- seven
  // from the hour scope, three from the feature scope, and the popover puts
  // them back together.
  expect(hourFactors.sort()).toEqual([...scope.hour].sort());
  expect(featureFactors.sort()).toEqual([...scope.feature].sort());
  expect(new Set([...hourFactors, ...featureFactors]).size).toBe(TOTAL_FACTORS);
  await expect(popover.getByTestId("factor-row")).toHaveCount(TOTAL_FACTORS);

  // Every factor shipped by this payload is placed at one of the two scopes.
  // The popover says so out loud when one is not, and that note must be absent.
  await expect(page.getByTestId("popover-undeclared")).toHaveCount(0);
  // The feature's three arrive trimmed to value and reason, so they carry no
  // weight -- and the popover explains the gap rather than leaving a column of
  // numbers with holes in it.
  await expect(page.getByTestId("popover-trimmed")).toBeVisible();
});

test("switching species re-colours without refetching", async ({ page }) => {
  const dayRequests: string[] = [];
  page.on("request", (event) => {
    if (ANY_DAY_PATH.test(new URL(event.url()).pathname)) dayRequests.push(event.url());
  });

  await loadWarmedDay(page);
  await findScoredMarker(page);

  const species = page.getByTestId("species-option");
  await expect(species).toHaveCount(3);
  const first = species.nth(0);
  const second = species.nth(1);
  await expect(first).toHaveAttribute("aria-pressed", "true");

  /**
   * The count BEFORE the switch is what makes the count after it meaningful:
   * a URL pattern that matched nothing would read zero at both ends and this
   * test would be asserting that nothing happened, which is not the claim.
   *
   * Deliberately "at least one" rather than "exactly one". Under the Vite dev
   * server this is TWO, and that is React's `StrictMode` doing its job:
   * `main.tsx` wraps the app in it, so in development every effect is mounted,
   * torn down and mounted again, and `DayProvider`'s load effect fetches the
   * day on each pass. Measured against the built bundle served by `tidescout
   * serve` -- same page, same payload, StrictMode's double-invoke compiled
   * out -- it is exactly one. Pinning 1 here would fail on the dev server;
   * pinning 2 would encode a development artifact as a requirement. What this
   * test is actually about is the delta below.
   */
  const beforeSwitch = dayRequests.length;
  expect(beforeSwitch, "loading the day never hit the day endpoint").toBeGreaterThan(0);

  const asRedfish = await settle(page);

  await second.click();
  await expect(second).toHaveAttribute("aria-pressed", "true");
  await expect(first).toHaveAttribute("aria-pressed", "false");
  const asOther = await settle(page);

  // Every species is scored in the one payload (spec §9) -- which is why that
  // payload is 25 MB. If this ever fires, the size is being paid for nothing.
  expect(
    dayRequests.slice(beforeSwitch),
    "switching species refetched the day",
  ).toEqual([]);

  // Re-COLOURS: same geometry, new activation expression. Both halves again --
  // the pixels move, and switching back restores the exact frame, which no
  // refetch-and-rejoin would guarantee.
  expect(
    Buffer.compare(asRedfish, asOther),
    "the map painted the second species exactly as it painted the first",
  ).not.toBe(0);

  await first.click();
  await expect(first).toHaveAttribute("aria-pressed", "true");
  const backToRedfish = await settle(page);
  expect(
    Buffer.compare(asRedfish, backToRedfish),
    "switching back to the first species did not restore its frame",
  ).toBe(0);
  expect(dayRequests, "a later species switch refetched the day").toHaveLength(beforeSwitch);
});
