import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";

import { fetchDay, fetchStatus } from "../api/client";
import type { DayPayload } from "../api/types";

// Computing one day's scores costs the backend ~70s. A cache miss returns 202
// and builds in the background, so this context polls a cheap status
// endpoint until the build finishes, then fetches the real payload.
//
// A cache HIT needs the same poller for a different reason. `/day` serves a
// STALE payload immediately and kicks off a rebuild behind it (app.py:
// `if store.is_stale(...): coord.ensure(...)`), and only `/day/.../status`
// carries the `stale` flag. So a cache hit is checked once against `/status`,
// and if the answer is stale the poller runs until the rebuild lands and
// replaces the payload on screen. Without that, the rebuild finishes on disk
// and nothing ever picks it up: the reader keeps yesterday's scoring run
// until they happen to reload.
const POLL_INTERVAL_MS = 2000;
const DEFAULT_MODEL = "best";

export type DayState = "loading" | "building" | "ready" | "failed";

export interface DayContextValue {
  state: DayState;
  // Stored WHOLE. A payload carrying `missing: [...]` and a lowered
  // `confidence` is a successful, degraded result -- never filtered here,
  // never treated as `failed`. See spec on payload disclosure.
  payload: DayPayload | null;
  /**
   * This payload is older than the data behind it and a rebuild is running.
   *
   * The BACKEND's judgement (`store.is_stale`), read off `/status` -- never
   * inferred here from `generated_at`, which would be a second, disagreeing
   * definition. True only between noticing the staleness and the rebuilt
   * payload arriving, because that arrival clears it.
   */
  stale: boolean;
  error: string | null;
  species: string;
  hour: number;
  setSpecies: (species: string) => void;
  setHour: (hour: number) => void;
  slug: string;
  date: string;
  setDate: (date: string) => void;
  model: string;
  setModel: (model: string) => void;
}

const DayContext = createContext<DayContextValue | null>(null);

export interface DayProviderProps {
  slug: string;
  initialDate: string;
  initialModel?: string;
  children: ReactNode;
}

export function DayProvider({
  slug,
  initialDate,
  initialModel = DEFAULT_MODEL,
  children,
}: DayProviderProps) {
  const [date, setDate] = useState(initialDate);
  const [model, setModel] = useState(initialModel);
  const [state, setState] = useState<DayState>("loading");
  const [payload, setPayload] = useState<DayPayload | null>(null);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hour, setHour] = useState(0);
  const [species, setSpecies] = useState("");
  /**
   * The (fishery, day) the payload in state was applied for, or null before
   * the first one lands.
   *
   * WHAT THE SELECTION IS ABOUT. `hour` and `species` are a question -- "how
   * does 15:00 look for redfish?" -- and a MODEL change does not change the
   * question, it changes who is answering it. Resetting to hour 0 and the
   * first species there would defeat the one comparison the model picker
   * exists for: Winyah's own hours read confidence 1.00 / constrained_share
   * 0.92 under one model and 0.92 / 1.00 under another, and a reader who has
   * to re-find their hour after every switch cannot see that.
   *
   * A DATE or FISHERY change is the opposite: the question itself moved, the
   * hours are different hours, and hour 0 with the first species is the right
   * place to start. So the reset is keyed on this pair and not on "a new
   * payload arrived" -- which is also what keeps a background rebuild
   * (`stale`, same day, same model) from yanking a reader back to midnight.
   */
  const appliedFor = useRef<string | null>(null);

  // This effect owns the whole load-or-build lifecycle for one (slug, date,
  // model) triple. `hour`/`species` are deliberately NOT in the dependency
  // array -- scrubbing is a plain state set, never a refetch, because every
  // species is already scored in the payload.
  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | undefined;
    // A payload is on screen. A background status check that fails after this
    // point must not blank a day that is already rendering -- the day is
    // fine; what was lost is the watch on its staleness.
    let showing = false;
    // `/status` said this payload is stale, so the next NOT-stale answer is a
    // rebuilt payload to go and fetch. Without this flag the one status call
    // a cache hit makes would re-fetch the 1.67 MB day it just fetched.
    let awaitingRebuild = false;

    function clearPoll() {
      if (intervalId !== undefined) {
        clearInterval(intervalId);
        intervalId = undefined;
      }
    }

    function startPoll() {
      if (intervalId !== undefined) return;
      intervalId = setInterval(() => {
        void pollStatus();
      }, POLL_INTERVAL_MS);
    }

    function applyReady(next: DayPayload) {
      if (cancelled) return;
      showing = true;
      awaitingRebuild = false;
      setPayload(next);
      // Whatever this payload is, it is the newest the backend has: the
      // rebuild it replaces is exactly what `stale` was reporting.
      setStale(false);

      // First key of `species` -- an insertion-ordered Record, so the default
      // is deterministic for a given payload.
      const names = Object.keys(next.species);
      const subject = `${slug}\u0000${date}`;
      const sameSubject = appliedFor.current === subject;
      appliedFor.current = subject;
      if (sameSubject) {
        // Same water, same day: keep the hour and the fish. `hour` is left
        // exactly as it is -- every payload carries all 24 -- and the species
        // is kept only if this payload actually scores it. A fishery whose
        // species list differs between runs would otherwise leave a name in
        // state that nothing in the payload answers to, and the rail, the map
        // and the strip would all read empty for a day that scored fine.
        setSpecies((current) => (names.includes(current) ? current : (names[0] ?? "")));
      } else {
        setHour(0);
        setSpecies(names[0] ?? "");
      }
      setError(null);
      setState("ready");
    }

    async function pollStatus() {
      let status;
      try {
        status = await fetchStatus(slug, date, model);
      } catch (err) {
        if (cancelled) return;
        clearPoll();
        // Losing the status endpoint while a day is on screen degrades the
        // staleness watch, not the day. Failing here would replace a
        // perfectly good scoring run with an error message.
        if (showing) return;
        setError(err instanceof Error ? err.message : String(err));
        setState("failed");
        return;
      }
      if (cancelled) return;

      if (status.status === "ready") {
        // Collect the payload when there is nothing on screen, or when the
        // rebuild this poller was waiting for has landed. A STALE payload is
        // still collected when nothing is on screen: yesterday's numbers,
        // flagged as such, beat a blank panel. Otherwise the payload showing
        // is already the newest there is and re-fetching 1.67 MB to learn
        // that would be the cost this whole status endpoint exists to avoid.
        // `awaitingRebuild` alone re-arms on every still-stale tick (it is
        // set unconditionally below whenever `status.stale` is true), so
        // pairing it with this tick's OWN stale reading is what makes this
        // fire once -- on the tick the rebuild actually lands -- rather than
        // on every poll for the whole rebuild. Without `!status.stale` this
        // re-downloads the 1.67 MB payload every 2s for as long as the
        // rebuild takes, which is exactly the cost cited above.
        if (!showing || (awaitingRebuild && !status.stale)) {
          clearPoll();
          try {
            const result = await fetchDay(slug, date, model);
            if (cancelled) return;
            if (result.kind === "ready") {
              applyReady(result.payload);
            } else {
              // Status said ready but the day-endpoint still 202s (a race
              // with the backend cache) -- keep polling rather than hang.
              setState("building");
              startPoll();
              return;
            }
          } catch (err) {
            if (cancelled) return;
            if (showing) {
              // The day on screen is still fine -- only this collection
              // attempt failed. `clearPoll()` already ran above, so without
              // restarting it here a single transient network blip pins
              // `stale` true permanently: nothing is left to notice the
              // rebuild ever lands.
              startPoll();
              return;
            }
            setError(err instanceof Error ? err.message : String(err));
            setState("failed");
            return;
          }
        }
        if (status.stale) {
          // The bytes served are older than the data behind them and the
          // backend is rebuilding (`/day` kicked that off when it served
          // them). Say so, and keep polling: this poll is the only thing
          // that will ever pick the rebuild up.
          awaitingRebuild = true;
          setStale(true);
          startPoll();
        } else {
          clearPoll();
        }
      } else if (status.status === "failed") {
        clearPoll();
        if (showing) return;
        setError(status.error);
        setState("failed");
      }
      // "building" or "absent": keep polling, nothing to do this tick.
    }

    async function start() {
      setState("loading");
      setError(null);
      setStale(false);
      try {
        const result = await fetchDay(slug, date, model);
        if (cancelled) return;
        if (result.kind === "ready") {
          applyReady(result.payload);
          // The one status call a cache hit makes. `/day` answers 200 for a
          // stale payload exactly as it does for a current one -- the two are
          // indistinguishable from this side -- so without this the rebuild
          // running behind it is never noticed and never collected.
          void pollStatus();
        } else {
          setState("building");
          startPoll();
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setState("failed");
      }
    }

    void start();

    // Clear the poller on unmount AND whenever slug/date/model change --
    // otherwise a leaked interval keeps hitting the API for a day nobody is
    // looking at anymore.
    return () => {
      cancelled = true;
      clearPoll();
    };
  }, [slug, date, model]);

  const value = useMemo<DayContextValue>(
    () => ({
      state,
      payload,
      stale,
      error,
      species,
      hour,
      setSpecies,
      setHour,
      slug,
      date,
      setDate,
      model,
      setModel,
    }),
    [state, payload, stale, error, species, hour, slug, date, model],
  );

  return <DayContext.Provider value={value}>{children}</DayContext.Provider>;
}

export function useDay(): DayContextValue {
  const ctx = useContext(DayContext);
  if (!ctx) throw new Error("useDay must be used within a DayProvider");
  return ctx;
}
