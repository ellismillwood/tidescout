/**
 * One bar per scoring factor: the breakdown behind "why is 15:00 a 71?".
 *
 * FORM (dataviz). The job is "compare magnitude across ~10 named items on one
 * measure", and the names are long and the reasons longer, so this is a
 * HORIZONTAL bar list, not a column chart -- a column chart would set ten
 * labels sideways and leave the reason strings nowhere to go. Each bar is a
 * METER: the measure has a fixed, meaningful ceiling (a sub-score is 0..1, and
 * 1 is a perfect factor), so the unfilled track is drawn as a lighter step of
 * the bar's own ramp. Without the track a 0.46 and a 0.46-out-of-0.5 look the
 * same; with it the ceiling is visible on every row.
 *
 * COLOUR (dataviz, validated -- not eyeballed). ONE hue, because this is one
 * measure. It is deliberately the hour strip's score amber rather than a new
 * colour: these sub-scores are the COMPONENTS of the score those bars draw
 * (`engine.score.combine` is a weighted geometric mean of exactly these
 * values), so wearing the score's hue asserts something true, and a fourth
 * hue in a palette whose whole premise is a chart margin at 4am would assert
 * something false -- that this is a different quantity. Validated against both
 * surfaces it renders on, the rail panel and the raised popover:
 *
 *   node validate_palette.js "#b8822f" --mode dark --surface "#0a1e29"
 *   node validate_palette.js "#b8822f" --mode dark --surface "#102a37"
 *       -> ALL CHECKS PASS (lightness band, chroma floor, 3:1 on surface)
 *
 * Hue is therefore free to carry nothing at all here, which is what lets the
 * two facts that are NOT magnitude ride the second channel instead:
 *
 *   - provisional (scored at full weight, but no observation constrains it)
 *     -> the 45-degree open weave, the same "modelled, not measured" grammar
 *        the strip uses for a partly resolved hour. It survives greyscale
 *        and CVD, which a colour change would not -- and forced-colors by way
 *        of the `@media (forced-colors: active)` block in `FactorBars.css`,
 *        since that mode strips the background image the weave is drawn with.
 *   - missing / no value -> no fill at all, just the track and a flag. An
 *     absent factor is an absence, never a bar of length zero: a zero-length
 *     bar reads as "scored, and the answer is zero", which is a different
 *     claim (the same distinction the map's unscored sentinel exists to keep).
 *
 * ROW ORDER IS THE PAYLOAD'S, never sorted by value. Sorting would make every
 * row jump on every scrub tick, and the rail's whole job is to be read while
 * the hour moves.
 *
 * THE REASON STRING IS RENDERED VERBATIM, and this is the one rule in this
 * file that is not a matter of taste. "salinity ~34.9 ppt -- salty
 * (UNCALIBRATED model estimate, no observation constrains it)" is where the
 * model's honesty about its own uncertainty reaches a person. It is never
 * truncated (no line clamp, no ellipsis, no `overflow: hidden` anywhere near
 * it) and never reworded here.
 */
import { isReading } from "./format";
import "./FactorBars.css";

/**
 * What a bar needs. Deliberately WIDER than `SubScore`: a feature-hour's subs
 * arrive trimmed to factor/value/reason (the payload ships them that way on
 * purpose -- see `mergeFeatureSubs`), so `weight`, `missing` and `provisional`
 * are optional and their absence means "the payload did not say", which is
 * rendered as silence rather than as a default. A full `SubScore[]` satisfies
 * this structurally, so the rail passes the hour's subs straight in.
 */
export interface FactorSub {
  factor: string;
  value: number | null;
  reason: string;
  weight?: number;
  missing?: boolean;
  provisional?: boolean;
  /** A short scope note from the caller, e.g. "bay-wide". Never invented here. */
  note?: string;
}

export interface FactorBarsProps {
  subs: readonly FactorSub[];
  /** Names what this group of bars is. Doubles as the group's aria label. */
  caption?: string;
  /** Shown in place of the list when it is empty -- never an empty box. */
  empty?: string;
  /** Distinguishes two groups of bars in one panel, e.g. the popover's. */
  testId?: string;
}

/** `water_temp` is a payload key; "water temp" is what a person reads. */
export function humanizeFactor(factor: string): string {
  return factor.replace(/_/g, " ");
}

/**
 * Bar length. Sub-scores are 0..1 by contract, and the clamp is only so a
 * value outside that range cannot draw past the track -- the printed number
 * beside it is always the raw one, so a clamp can never hide a bad value.
 */
function fillWidth(value: number): string {
  return `${Math.min(Math.max(value, 0), 1) * 100}%`;
}

export function FactorBars({ subs, caption, empty, testId }: FactorBarsProps) {
  if (subs.length === 0) {
    return empty ? (
      <p className="factors-empty" data-testid="factors-empty">
        {empty}
      </p>
    ) : null;
  }

  return (
    <section className="factors-group">
      {caption && (
        <p className="eyebrow factors-caption">
          {caption}
          <span className="factors-count num">{subs.length}</span>
        </p>
      )}
      <ol
        className="factors"
        data-testid={testId ?? "factor-bars"}
        aria-label={caption ?? "Factors"}
      >
        {subs.map((sub) => {
          // Bound to a const so the null check narrows: `sub.value` is
          // `number | null`, and a bar is drawn only for a real reading.
          const value = isReading(sub.value) ? sub.value : null;
          return (
            <li
              className="factor"
              key={sub.factor}
              data-testid="factor-row"
              data-factor={sub.factor}
              data-provisional={sub.provisional === true}
              data-scored={value !== null}
            >
              <p className="factor-head">
                <span className="factor-name">{humanizeFactor(sub.factor)}</span>
                <span className="factor-value num" data-testid="factor-value">
                  {value === null ? "—" : value.toFixed(2)}
                </span>
                {/* Weight is the OTHER number that answers "why is this a 71" --
                    how much this factor counted. Absent on a trimmed sub, and
                    absent is printed as nothing rather than as 1.0. */}
                {isReading(sub.weight) && (
                  <span className="factor-weight">weight {sub.weight.toFixed(2)}</span>
                )}
              </p>
              {/* The value and the reason are both text in the DOM already, so
                  the meter is decoration for a screen reader and says so. */}
              <div className="factor-meter" aria-hidden="true">
                {value !== null && (
                  <span className="factor-fill" style={{ inlineSize: fillWidth(value) }} />
                )}
              </div>
              <p className="factor-reason" data-testid="factor-reason">
                {sub.reason}
              </p>
              {(sub.missing === true || value === null || sub.provisional === true || sub.note) && (
                <p className="factor-flags">
                  {(sub.missing === true || value === null) && (
                    <span className="factor-flag" data-testid="factor-flag-missing">
                      no reading — excluded from the score
                    </span>
                  )}
                  {sub.provisional === true && (
                    <span className="factor-flag" data-testid="factor-flag-provisional">
                      scored at full weight, unconstrained
                    </span>
                  )}
                  {sub.note && (
                    <span className="factor-flag" data-testid="factor-flag-note">
                      {sub.note}
                    </span>
                  )}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

export default FactorBars;
