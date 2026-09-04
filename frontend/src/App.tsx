import { useState } from "react";

import { DayProvider } from "./state/DayContext";
import { MapView } from "./map/MapView";
import { ConditionsRail } from "./rail/ConditionsRail";
import { HourStrip } from "./strip/HourStrip";
import { TopBar } from "./ui/TopBar";

/** Where the app opens. The picker moves off it; nothing else depends on it. */
const DEFAULT_SLUG = "winyah-bay";

function today(): string {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

/**
 * The bar and its disclosure, the chart and its rail, and the day's strip.
 *
 * `slug` is the one selection that lives ABOVE the day context rather than
 * inside it: `DayProvider` takes it as a prop and rebuilds the whole day when
 * it changes, which is exactly right -- a different fishery is a different
 * day. Date, model, hour and species all live in the context, and only the
 * first two of those refetch.
 */
export default function App() {
  const [slug, setSlug] = useState(DEFAULT_SLUG);

  return (
    <DayProvider slug={slug} initialDate={today()}>
      <TopBar onFisheryChange={setSlug} />
      {/* The chart and the rail share the middle row: the map answers
          "where", the rail answers "why", and both read the same `hour` from
          the context the strip below sets. */}
      <main>
        <MapView />
        <ConditionsRail />
      </main>
      <HourStrip />
    </DayProvider>
  );
}
