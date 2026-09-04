import { DayProvider, useDay } from "./state/DayContext";
import { MapView } from "./map/MapView";
import { ConditionsRail } from "./rail/ConditionsRail";
import { HourStrip } from "./strip/HourStrip";

// Until Task 12's picker exists there is one fishery and one day. The picker
// replaces both of these; nothing else here depends on them.
const SLUG = "winyah-bay";

function today(): string {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

const STATUS: Record<string, string> = {
  loading: "Loading the day",
  building: "Scoring the day — about 70 seconds",
  failed: "Could not score this day",
  ready: "",
};

/**
 * The chart margin above, the chart and its rail, and the day's strip below.
 *
 * The species buttons are TEMPORARY -- Task 12 replaces them with the real
 * pickers. The hour slider that used to sit beside them is GONE: it existed
 * only so Task 9 had something to drag while verifying the scrub loop, and
 * `HourStrip` is now the real control. Both it and `MapView` read `hour` from
 * the same context, so scrubbing the strip moves the markers with no wiring
 * between the two.
 */
function Shell() {
  const { state, error, payload, species, setSpecies, date } = useDay();
  const names = payload ? Object.keys(payload.species) : [];

  return (
    <>
      <header className="margin-bar">
        <h1 className="wordmark">TideScout</h1>
        <div className="place">
          <span className="num">{payload?.slug ?? SLUG}</span>
          <span className="num">{payload?.day ?? date}</span>
        </div>
        <div className="species">
          {names.map((name) => (
            <button
              key={name}
              type="button"
              aria-pressed={name === species}
              onClick={() => setSpecies(name)}
            >
              {name.replace(/_/g, " ")}
            </button>
          ))}
        </div>
        <span className="fill" />
        <span className="status" data-tone={state}>
          {state === "failed" ? (error ?? STATUS.failed) : STATUS[state]}
        </span>
      </header>
      {/* The chart and the rail share the middle row: the map answers
          "where", the rail answers "why", and both read the same `hour` from
          the context the strip below sets. */}
      <main>
        <MapView />
        <ConditionsRail />
      </main>
      <HourStrip />
    </>
  );
}

export default function App() {
  return (
    <DayProvider slug={SLUG} initialDate={today()}>
      <Shell />
    </DayProvider>
  );
}
