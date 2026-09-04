/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev runs two servers: Vite here, the API on :8000. In production the
    // frontend is same-origin (the API mounts frontend/dist), so this proxy
    // exists only so relative /api paths work identically in both.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    // Vitest's default glob would also collect `e2e/smoke.spec.ts`, which is a
    // Playwright spec: it needs a browser and a live API, and under jsdom it
    // fails on import. The two suites answer different questions and neither
    // runner can run the other's file, so the boundary is drawn here rather
    // than left to a filename convention.
    include: ["tests/**/*.{test,spec}.{ts,tsx}"],
  },
});
