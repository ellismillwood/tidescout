import {
  createContext,
  useContext,
  useEffect,
  useMemo,
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
      setHour(0);
      // First key of `species` -- an insertion-ordered Record, so this is
      // deterministic for a given payload.
      setSpecies(Object.keys(next.species)[0] ?? "");
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
        if (status.stale) {
          // The bytes served are older than the data behind them and the
          // backend is rebuilding. Keep the stale day on screen -- flagged,
          // by `stale` -- and keep polling: this poll is the only thing that
          // will ever pick the rebuild up.
          awaitingRebuild = true;
          setStale(true);
          startPoll();
          return;
        }
        clearPoll();
        // A fresh payload is already on screen and no rebuild was pending --
        // this was the one confirming check a cache hit makes.
        if (showing && !awaitingRebuild) return;
        try {
          const result = await fetchDay(slug, date, model);
          if (cancelled) return;
          if (result.kind === "ready") {
            applyReady(result.payload);
          } else {
            // Status said ready but the day-endpoint still 202s (a race with
            // the backend cache) -- keep polling rather than hang forever.
            setState("building");
            startPoll();
          }
        } catch (err) {
          if (cancelled) return;
          if (showing) return;
          setError(err instanceof Error ? err.message : String(err));
          setState("failed");
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
