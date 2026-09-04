import { DayProvider, useDay } from "./state/DayContext";
import { MapView } from "./map/MapView";

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
 * The chart margin.
 *
 * The species buttons and the hour slider are TEMPORARY: Task 10 replaces the
 * slider with the 24-hour strip and Task 12 replaces the rest with the real
 * pickers. They are here because Task 9's own verification step is "drag the
 * hour and watch the markers", and that needs something to drag.
 */
function Shell() {
  const { state, error, payload, species, setSpecies, hour, setHour, date } = useDay();
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
        <label className="scrub">
          <span className="eyebrow">Hour</span>
          <input
            type="range"
            min={0}
            max={23}
            step={1}
            value={hour}
            onChange={(event) => setHour(Number(event.target.value))}
            disabled={state !== "ready"}
          />
          <span className="num">{String(hour).padStart(2, "0")}:00</span>
        </label>
        <span className="fill" />
        <span className="status" data-tone={state}>
          {state === "failed" ? (error ?? STATUS.failed) : STATUS[state]}
        </span>
      </header>
      <main>
        <MapView />
      </main>
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
