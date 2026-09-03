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
const POLL_INTERVAL_MS = 2000;
const DEFAULT_MODEL = "best";

export type DayState = "loading" | "building" | "ready" | "failed";

export interface DayContextValue {
  state: DayState;
  // Stored WHOLE. A payload carrying `missing: [...]` and a lowered
  // `confidence` is a successful, degraded result -- never filtered here,
  // never treated as `failed`. See spec on payload disclosure.
  payload: DayPayload | null;
  error: string | null;
  species: string;
  hour: number;
  setSpecies: (species: string) => void;
  setHour: (hour: number) => void;
  slug: string;
  date: string;
  setDate: (date: string) => void;
}

const DayContext = createContext<DayContextValue | null>(null);

export interface DayProviderProps {
  slug: string;
  initialDate: string;
  model?: string;
  children: ReactNode;
}

export function DayProvider({
  slug,
  initialDate,
  model = DEFAULT_MODEL,
  children,
}: DayProviderProps) {
  const [date, setDate] = useState(initialDate);
  const [state, setState] = useState<DayState>("loading");
  const [payload, setPayload] = useState<DayPayload | null>(null);
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
      setPayload(next);
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
        setError(err instanceof Error ? err.message : String(err));
        setState("failed");
        return;
      }
      if (cancelled) return;

      if (status.status === "ready") {
        clearPoll();
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
          setError(err instanceof Error ? err.message : String(err));
          setState("failed");
        }
      } else if (status.status === "failed") {
        clearPoll();
        setError(status.error);
        setState("failed");
      }
      // "building" or "absent": keep polling, nothing to do this tick.
    }

    async function start() {
      setState("loading");
      setError(null);
      try {
        const result = await fetchDay(slug, date, model);
        if (cancelled) return;
        if (result.kind === "ready") {
          applyReady(result.payload);
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
      error,
      species,
      hour,
      setSpecies,
      setHour,
      slug,
      date,
      setDate,
    }),
    [state, payload, error, species, hour, slug, date],
  );

  return <DayContext.Provider value={value}>{children}</DayContext.Provider>;
}

export function useDay(): DayContextValue {
  const ctx = useContext(DayContext);
  if (!ctx) throw new Error("useDay must be used within a DayProvider");
  return ctx;
}
