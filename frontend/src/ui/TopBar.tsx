/**
 * The margin bar: which water, which day, which weather model, which fish --
 * and, on the line below it, how much to trust the answer.
 *
 * Four pickers and a disclosure band, in that order, because that is the
 * order the questions come in: a person picks the place and the day before
 * they can ask anything about them, and the band under the pickers reports on
 * whatever they landed on.
 *
 * THREE THINGS THIS FILE IS RESPONSIBLE FOR NOT GETTING WRONG:
 *
 *   1. An unprocessed fishery is DISABLED and says why. `readiness.reason`
 *      names what is missing -- a flow library, a distance field, a feature
 *      inventory -- and that sentence is the only thing standing between a
 *      person and a 409 they cannot interpret.
 *   2. The date picker is bounded UP FRONT. The API's 422 past the +16-day
 *      horizon is the backstop (spec §7), not the defence: a picker that
 *      offers a date the API will refuse has already failed.
 *   3. Species changes state and NOTHING else. Every species is scored in the
 *      payload already, so switching one is a re-render, never a refetch --
 *      the asymmetry `DayProvider`'s effect encodes and this bar must not
 *      break. Model and date changes DO refetch, and that is correct: each
 *      names a different scoring run.
 */
import { useEffect, useMemo, useState } from "react";

import { fetchFisheries } from "../api/client";
import type { FisherySummary } from "../api/types";
import { useDay } from "../state/DayContext";
import { Disclosure } from "./Disclosure";
import "./TopBar.css";

export interface WeatherModelOption {
  value: string;
  label: string;
}

/**
 * The six models the API accepts, spelled exactly as `weather.WEATHER_MODELS`
 * spells them.
 *
 * `?model=` becomes part of a FILENAME on the backend, so the API validates
 * the value before it reaches the filesystem -- that check closed a real
 * path-traversal (`?model=../../../x` wrote outside `data/`). Anything not on
 * this list is a 422. Inventing a plausible-looking name here ("gefs",
 * "icon-eu") would produce a picker whose every request fails.
 */
export const WEATHER_MODELS: readonly WeatherModelOption[] = [
  { value: "best", label: "best — blended" },
  { value: "gfs", label: "gfs — NOAA GFS" },
  { value: "ecmwf", label: "ecmwf — ECMWF IFS" },
  { value: "icon", label: "icon — DWD ICON" },
  { value: "hrrr", label: "hrrr — NOAA HRRR" },
  { value: "nbm", label: "nbm — NOAA NBM" },
];

/** Mirrors `weather.FORECAST_HORIZON_DAYS`; the API 422s past it. */
const FORECAST_HORIZON_DAYS = 16;
/**
 * How far back the picker offers.
 *
 * The API sets no lower bound and Open-Meteo's ERA5 archive reaches back
 * years, so this is a UI choice rather than a limit: a year covers "how did
 * it fish this week last season", which is the question people actually ask
 * backwards, without implying the archive is endless. Widening it is a
 * one-line change and costs the backend nothing.
 */
const PAST_WINDOW_DAYS = 365;

function isoDay(when: Date): string {
  const month = String(when.getMonth() + 1).padStart(2, "0");
  const day = String(when.getDate()).padStart(2, "0");
  return `${when.getFullYear()}-${month}-${day}`;
}

/**
 * The dates the picker will offer, as `min`/`max` for `<input type="date">`.
 *
 * Built by adding days to a local Y/M/D triple rather than by adding
 * milliseconds: `new Date(2026, 11, 28 + 16)` rolls the month and the year
 * correctly and is immune to the DST hour that makes `+ n * 86_400_000` land
 * on the wrong calendar day twice a year.
 */
export function dayRange(now: Date): { min: string; max: string } {
  const shift = (days: number) =>
    isoDay(new Date(now.getFullYear(), now.getMonth(), now.getDate() + days));
  return { min: shift(-PAST_WINDOW_DAYS), max: shift(FORECAST_HORIZON_DAYS) };
}

const STATUS: Record<string, string> = {
  loading: "Loading the day",
  building: "Scoring the day — about 70 seconds",
  failed: "Could not score this day",
  ready: "",
};

/** What the picker needs. Narrower than `FisherySummary` on purpose: the bar
 *  never has a centre or a timezone to show, so a placeholder row for the
 *  current fishery does not have to invent one. */
interface PickerOption {
  slug: string;
  name: string;
  ready: boolean;
  reason?: string;
}

export interface TopBarProps {
  /** The fishery lives above the day context, which takes `slug` as a prop. */
  onFisheryChange: (slug: string) => void;
}

export function TopBar({ onFisheryChange }: TopBarProps) {
  const {
    state,
    error,
    payload,
    species,
    setSpecies,
    hour,
    slug,
    date,
    setDate,
    model,
    setModel,
  } = useDay();
  const [fisheries, setFisheries] = useState<FisherySummary[] | null>(null);

  // The list is fetched once. It describes what is on disk, which does not
  // change while someone is looking at a day.
  useEffect(() => {
    let cancelled = false;
    fetchFisheries()
      .then((list) => {
        if (!cancelled) setFisheries(list);
      })
      .catch(() => {
        // Fail soft, and deliberately: losing /api/fisheries must not strip
        // the bar of the fishery it is already showing, and the day itself
        // is unaffected. The fallback below covers it.
        if (!cancelled) setFisheries(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const options = useMemo<PickerOption[]>(() => {
    const list: PickerOption[] = (fisheries ?? []).map((fishery) => ({
      slug: fishery.slug,
      name: fishery.name,
      ready: fishery.ready,
      ...(fishery.reason === undefined ? {} : { reason: fishery.reason }),
    }));
    if (list.some((option) => option.slug === slug)) return list;
    // Before the list lands -- or if it never does -- the select still shows
    // the fishery on screen rather than an empty box or someone else's.
    return [{ slug, name: slug.replace(/-/g, " "), ready: true }, ...list];
  }, [fisheries, slug]);

  const range = dayRange(new Date());
  const names = payload ? Object.keys(payload.species) : [];
  const scored = payload && species ? (payload.species[species]?.hours[hour] ?? null) : null;

  return (
    <header className="topbar">
      <div className="margin-bar">
        <h1 className="wordmark">TideScout</h1>

        <div className="pickers">
          <label className="field">
            <span className="field-label">Fishery</span>
            <select
              className="control"
              data-testid="fishery-picker"
              value={slug}
              onChange={(event) => onFisheryChange(event.target.value)}
            >
              {options.map((option) => (
                <option
                  key={option.slug}
                  value={option.slug}
                  data-testid="fishery-option"
                  disabled={!option.ready}
                  {...(option.ready
                    ? {}
                    : { title: `not processed — missing: ${option.reason ?? "unknown"}` })}
                >
                  {option.ready
                    ? option.name
                    : `${option.name} — ${option.reason ?? "not processed"}`}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span className="field-label">Day</span>
            {/* Bounded up front. A value typed past `max` still reaches the
                API, which answers with the exact horizon it will serve --
                a better sentence than anything this component could guess. */}
            <input
              className="control num"
              data-testid="date-picker"
              type="date"
              value={date}
              min={range.min}
              max={range.max}
              onChange={(event) => {
                // A cleared input is not a date. Refetching for "" would ask
                // the API for /day/ and get a 404 for no reason.
                if (event.target.value) setDate(event.target.value);
              }}
            />
          </label>

          <label className="field">
            <span className="field-label">Model</span>
            <select
              className="control"
              data-testid="model-picker"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              {WEATHER_MODELS.map((option) => (
                <option key={option.value} value={option.value} data-testid="model-option">
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* Species is the one control here that does NOT refetch: buttons
            rather than a select, because the whole set is short, already
            scored, and switching between them is the cheapest thing this app
            does. */}
        <div className="species" role="group" aria-label="Species">
          {names.map((name) => (
            <button
              key={name}
              type="button"
              data-testid="species-option"
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
      </div>

      {/* The four signals, for whichever hour the strip is parked on. */}
      <Disclosure hour={scored} freshness={payload?.freshness ?? null} />
    </header>
  );
}

export default TopBar;
