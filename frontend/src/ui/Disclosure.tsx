/**
 * Disclosure: four signals, four questions, three marks.
 *
 * This is the component that tells a person how much to trust what they are
 * looking at, and it exists as a component precisely so the answer is on
 * screen rather than in someone's memory of a caveat they read once.
 *
 * THE FOUR SIGNALS ARE FOUR DIFFERENT QUESTIONS. They are rendered as four
 * cells, never as one "quality" number:
 *
 *   freshness          how old this scoring run is        payload.freshness
 *   confidence         how much authored weight resolved  hour.confidence
 *   constrained share  how much of THAT rests on          hour.constrained_share
 *                      measurement rather than a model
 *   provisional        WHICH factors are unconstrained    hour.provisional[]
 *
 * `confidence` and `constrained_share` are separate numbers in the payload on
 * purpose -- `engine.score.combine` computes them from different denominators
 * (authored weight, then surviving weight), and Winyah Bay's own hours read
 * confidence 1.00 with constrained_share 0.92: every factor resolved, and a
 * twelfth of the score still rests on an uncalibrated salinity model. One
 * merged number cannot say that, which is why five PRs went into keeping them
 * apart. They are printed as two values, on two rows, with two different
 * marks, and the note under each states its own denominator.
 *
 * `provisional` NAMES the unconstrained factors. "Some factors are
 * provisional" and "salinity is provisional" are different claims and only
 * the second is actionable -- a person who knows it is salinity knows to
 * trust the tide and stage factors in the same hour.
 *
 * THE MARKS ARE THE ONES THIS APP ALREADY USES (FactorBars, and the strip's
 * partly-resolved hour), so a reader learns the vocabulary once:
 *
 *   solid   measured
 *   weave   modelled -- scored at full weight, nothing observed constrains it
 *   void    no reading at all, excluded from the score
 *
 * The 45-degree weave survives greyscale, CVD and forced-colors, which is why
 * this file spends no colour on the distinction: the one warm accent in this
 * interface means "the fish are here", and disclosure must never borrow it.
 *
 * AND IT IS QUIET WHEN THERE IS NOTHING TO SAY. A component that warned on
 * every hour would pass every "does it warn?" test and be ignored by week
 * two. The flag line renders only when a signal is actually short, and
 * `data-tone` is "clear" when nothing is.
 */
import type { ReactNode } from "react";

import type { DayPayload, HourScore } from "../api/types";
import { humanizeFactor } from "../rail/FactorBars";
import { clock, fixed, isReading } from "../rail/format";
import "./Disclosure.css";

export interface DisclosureProps {
  /** The selected hour, or null before the day has loaded. */
  hour: HourScore | null;
  /** The run that produced this payload. Null renders the cell as unknown. */
  freshness?: DayPayload["freshness"] | null;
  /**
   * The run is older than the data behind it and a rebuild is running.
   *
   * Only `/day/{date}/status` carries this, so `DayContext` asks it once on a
   * cache hit and keeps asking while the answer is stale -- `stale` on the
   * context, passed down by `TopBar`, and cleared by the arrival of the
   * rebuilt payload. It is a prop rather than an inference because staleness
   * is the backend's judgement -- `store.is_stale` -- and guessing at it here
   * from `generated_at` would be a second, disagreeing definition.
   */
  stale?: boolean;
  /** Injectable clock, so the run's age is testable without wall-clock races. */
  now?: Date;
}

const MINUTE = 60_000;

/**
 * How long ago the run happened, in the unit a person would use.
 *
 * Relative, not absolute, and deliberately so: `generated_at` is UTC while
 * every other time in this app is the fishery's own zone, and printing
 * "23:00" from a UTC stamp beside a fishery-local hour would invite exactly
 * the misreading `rail/format.clock` exists to prevent. "3 h ago" is true in
 * every zone. Returns null for a missing or unparseable stamp -- never an
 * invented age.
 */
export function age(generatedAt: string | null | undefined, now: Date): string | null {
  if (!generatedAt) return null;
  const then = Date.parse(generatedAt);
  if (!Number.isFinite(then)) return null;
  const minutes = Math.floor((now.getTime() - then) / MINUTE);
  // A browser clock behind the server's must not read "in 4 minutes".
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}

/** "flow", "flow and salinity", "flow, salinity and water temp". */
function list(factors: readonly string[]): string {
  const names = factors.map(humanizeFactor);
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

interface CellProps {
  signal: string;
  label: string;
  /** The value, already formatted. Never a placeholder standing in for 0. */
  value: string;
  note: string;
  testId: string;
  title?: string;
  /** The mark under the value: a meter, drawn before the note explains it. */
  meter?: ReactNode;
  /** Names, under the note: which factors this cell is actually about. */
  names?: ReactNode;
}

/**
 * One signal. The order is fixed -- label, value, mark, note, names -- so the
 * four cells scan as one row: what it is, what it reads, the mark, what the
 * number is a share OF, and which factors it names.
 */
function Cell({ signal, label, value, note, testId, title, meter, names }: CellProps) {
  return (
    <div className="disc-cell" data-signal={signal} {...(title ? { title } : {})}>
      <p className="disc-label">{label}</p>
      <p className="disc-value" data-testid={testId}>
        {value}
      </p>
      {meter}
      <p className="disc-note">{note}</p>
      {names}
    </div>
  );
}

/**
 * A 0..1 share as a bar. `rest` says what the UNFILLED part means, which is
 * the whole difference between the two meters here: the weight confidence
 * lost is weight that produced no reading (void), and the share
 * constrained_share lost is weight that scored off a model (weave).
 */
function Share({ value, rest }: { value: number | null; rest: "void" | "weave" }) {
  const width = value === null ? 0 : Math.min(Math.max(value, 0), 1) * 100;
  return (
    <span className="disc-share" data-rest={rest} aria-hidden="true">
      <span className="disc-share-fill" style={{ inlineSize: `${width}%` }} />
    </span>
  );
}

export function Disclosure({ hour, freshness = null, stale = false, now }: DisclosureProps) {
  const at = now ?? new Date();
  const runAge = age(freshness?.generated_at, at);
  const confidence = hour && isReading(hour.confidence) ? hour.confidence : null;
  const share = hour && isReading(hour.constrained_share) ? hour.constrained_share : null;
  const provisional = hour?.provisional ?? [];
  const excluded = hour?.excluded ?? [];

  // Every flag names the thing it is about. A flag that said "some factors
  // are unconstrained" would be true, useless, and indistinguishable from the
  // next hour's.
  const flags: string[] = [];
  if (stale) {
    flags.push("This scoring run is out of date — a rebuild is running.");
  }
  if (excluded.length > 0) {
    flags.push(
      `No reading for ${list(excluded)}, so ${excluded.length === 1 ? "it is" : "they are"} left out of this hour's score.`,
    );
  } else if (confidence !== null && confidence < 1) {
    flags.push("Part of the authored weight did not resolve for this hour.");
  }
  if (provisional.length > 0) {
    flags.push(
      `No observation constrains ${list(provisional)} — ${provisional.length === 1 ? "it is a model estimate" : "they are model estimates"}.`,
    );
  } else if (share !== null && share < 1) {
    flags.push("Part of this hour's score rests on a model rather than on an observation.");
  }

  const tone = hour === null && !stale ? "waiting" : flags.length > 0 ? "flagged" : "clear";

  return (
    <section
      className="disclosure"
      data-testid="disclosure"
      data-tone={tone}
      aria-label="How much to trust this hour"
    >
      <p className="disc-eyebrow eyebrow">
        This hour
        <span className="num">{clock(hour?.time) ?? "—"}</span>
      </p>

      <Cell
        signal="freshness"
        label="Scored"
        value={runAge ?? "—"}
        note={
          freshness
            ? `${freshness.model_label} · for ${freshness.day}`
            : "no scoring run yet"
        }
        testId="freshness"
        {...(freshness ? { title: `generated_at ${freshness.generated_at}` } : {})}
      />

      <Cell
        signal="confidence"
        label="Confidence"
        value={fixed(confidence, 2) ?? "—"}
        note="of the authored weight resolved"
        testId="confidence"
        meter={<Share value={confidence} rest="void" />}
        // Named only when there is something to name. At confidence 1.00 the
        // value and the solid meter already say "every factor resolved", and
        // a line repeating it in the same small grey as the note above reads
        // as a second clause of that note.
        names={
          excluded.length > 0 ? (
            <p className="disc-names" data-testid="excluded">
              no reading: {list(excluded)}
            </p>
          ) : undefined
        }
      />

      <Cell
        signal="constrained-share"
        label="Constrained share"
        value={fixed(share, 2) ?? "—"}
        note="of that weight rests on an observation"
        testId="constrained-share"
        meter={<Share value={share} rest="weave" />}
      />

      <div className="disc-cell" data-signal="provisional">
        <p className="disc-label">Provisional</p>
        <p className="disc-value disc-chips" data-testid="provisional">
          {provisional.length === 0 ? (
            <span className="disc-none">none</span>
          ) : (
            provisional.map((factor) => (
              <span className="disc-chip" key={factor}>
                {humanizeFactor(factor)}
              </span>
            ))
          )}
        </p>
        <p className="disc-note">
          {provisional.length === 0
            ? "every factor here is constrained by an observation"
            : "scored at full weight, and modelled — not measured"}
        </p>
      </div>

      {flags.length > 0 && (
        <p className="disc-flag" data-testid="disclosure-flag">
          {flags.join(" ")}
        </p>
      )}
    </section>
  );
}

export default Disclosure;
