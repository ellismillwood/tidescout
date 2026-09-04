/**
 * The one test that drives the real stack.
 *
 * Every other suite in this repo stubs something: the backend's tests use a
 * client that never mounts the frontend, and the frontend's Vitest suite runs
 * in jsdom, which has no WebGL and therefore no map. Three real defects got
 * through both -- a MapLibre worker URL that 404'd so every GeoJSON layer drew
 * zero features while the raster tiles still painted, an axis tick row 873px
 * wide inside a 1408px plot, and two API routes made unreachable by
 * registration order. None of them are reachable without a browser, a real
 * server and a real day. Hence: no mocks here, no fixtures, no stubbed fetch.
 *
 * WHAT THIS CONFIG DOES NOT START: the API. `webServer` brings up Vite (which
 * proxies /api to 127.0.0.1:8000), and the suite's first act is to assert the
 * API is up with a warmed day, because a smoke test that silently mocks its
 * way past a missing backend is worth less than no smoke test. Bring it up
 * with:
 *
 *   ~/.venvs/tidescout/bin/tidescout warm winyah-bay --days 1   # ~70 s, once
 *   ~/.venvs/tidescout/bin/tidescout serve
 *
 * Vite rather than `tidescout serve`'s static mount is deliberate: `serve`
 * happily hands out a stale `frontend/dist`, so pointing the browser at it
 * would let this suite pass against a bundle that is not the source in the
 * tree.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  // One worker, no parallelism: every test loads the same 25 MB day payload
  // and a WebGL context, and three of those at once on a laptop turns a
  // deterministic suite into a timing experiment.
  fullyParallel: false,
  workers: 1,
  /**
   * ZERO. A retry here would hide exactly the failure this suite exists to
   * catch: the map is asynchronous, so a flaky pass is indistinguishable from
   * a real regression that happens to lose a race. Every wait below is on
   * observable state, never a fixed sleep, so a failure means something is
   * actually wrong.
   */
  retries: 0,
  forbidOnly: !!process.env["CI"],
  reporter: [["list"]],
  // Generous, and for one reason: against a warmed day a test here takes ~3 s,
  // but if the day is NOT warm the app polls while the backend spends ~70 s
  // scoring it. Timing out on that would report "the frontend is broken" for
  // a backend that is merely busy.
  timeout: 180_000,
  expect: { timeout: 20_000 },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Fixed, and wide enough that the bay fills a real chart rather than
        // a postage stamp -- the marker hit-test below scans this box.
        viewport: { width: 1440, height: 900 },
        launchOptions: {
          // MapLibre needs a WebGL context. Headless Chromium falls back to
          // SwiftShader, which recent builds gate behind this flag; without
          // it `new Map()` throws and MapView renders its "needs WebGL" panel
          // instead of a chart.
          args: ["--enable-unsafe-swiftshader"],
        },
      },
    },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173 --strictPort",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env["CI"],
    timeout: 60_000,
  },
});
