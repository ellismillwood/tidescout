/// <reference types="node" />
// File-scoped rather than a `types` entry in `tsconfig.app.json`: this is the
// one test that reads a file off disk, and the app project has no business
// gaining the node globals for it. `?raw` would be tidier and does not work --
// vitest runs with `css: false`, which stubs every `.css` request to the empty
// string, query included.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

/**
 * The one thing a jsdom test CAN say about forced colors.
 *
 * jsdom has no media-query engine and no forced-colors mode, so nothing here
 * can render the fallback and read it back. What it can do is hold the
 * stylesheets to the claim their own comments make: four files tell a reader
 * that the modelled-vs-measured weave survives forced colors, and every one
 * of those weaves is a `background-image`, which that mode strips. Before
 * this, the claim was false in five places and no test could tell.
 *
 * This is a source assertion, and it is deliberately BOTH halves: a file that
 * draws a weave must carry a forced-colors rule, and the rule must reinstate
 * the distinction through a border -- the one channel the mode preserves. A
 * check for the media query alone would pass against an empty block.
 */
const SHEETS = [
  "src/ui/Disclosure.css",
  "src/rail/FactorBars.css",
  "src/strip/HourStrip.css",
  "src/map/MapView.css",
] as const;

// `new URL(rel, import.meta.url)` would NOT do: under jsdom the global URL
// resolves against the document's http://localhost base, not the file's, and
// silently hands back an http URL. Off the resolved file path instead.
const HERE = dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(join(HERE, "..", rel), "utf8");
}

describe("the weave's forced-colors fallback", () => {
  for (const file of SHEETS) {
    it(`${file} draws a weave and reinstates it under forced colors`, () => {
      const css = read(file);
      // The premise: this file really does encode a state as a weave.
      expect(css).toMatch(/repeating-linear-gradient/);

      const block = css.match(/@media \(forced-colors: active\) \{[\s\S]*\n\}/);
      expect(
        block,
        `${file} has no @media (forced-colors: active) rule`,
      ).not.toBeNull();
      // ...and the block says something forced colors actually keeps. A border
      // style is the channel; a background-image inside it would be stripped
      // exactly like the weave it stands in for.
      expect(block![0]).toMatch(/border(-inline|-style)?:/);
      expect(block![0]).toMatch(/dashed|dotted/);
    });
  }
});
