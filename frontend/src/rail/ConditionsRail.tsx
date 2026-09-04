/**
 * The rail: why THIS hour scores what it does.
 *
 * The map answers "where" and the strip answers "when". This column answers
 * the third question -- "why" -- for whichever hour the strip is parked on,
 * and it is the only place in the app that shows the raw conditions the
 * scorer read. It follows `hour` and `species` from the same context the map
 * and the strip read, so scrubbing moves all three with no wiring between
 * them.
 *
 * Two rules run through the whole file:
 *
 *   1. A missing reading is printed as a missing reading. Every value here is
 *      nullable in the payload, and none of them gets a default -- "no
 *      reading" is a fact about the day, and a 0 in its place would be a
 *      fabricated one.
 *   2. `payload.water` and `payload.astro` are DAY-level and either may be
 *      null. They are rendered in their own block, labelled as day facts, so
 *      nothing here implies a water temperature was measured at 15:00.
 *
 * The factor bars at the bottom are the hour's own subs, which include the
 * BAY-WIDE flow and salinity. Which of them vary per feature is not hardcoded
 * here: it is read from `payload.sub_scope`, the same field the popover's
 * merge is driven by, and those rows are tagged so the number a person reads
 * in the rail is never mistaken for the number under a particular marker.
 */
import { useDay } from "../state/DayContext";
import { FactorBars, type FactorSub } from "./FactorBars";
import { clock, compass, fixed, isReading, percent, signed } from "./format";
import "./ConditionsRail.css";

/** Joins the parts that exist. Empty in, `null` out -- never a stray "·". */
function parts(...values: (string | null | undefined)[]): string | null {
  const kept = values.filter((value): value is string => Boolean(value));
  return kept.length > 0 ? kept.join(" · ") : null;
}

interface RowProps {
  label: string;
  value: string | null;
  note?: string | null;
}

function Row({ label, value, note }: RowProps) {
  return (
    <div className="cond-row" data-testid={`cond-row-${label}`}>
      <dt className="cond-label">{label}</dt>
      <dd className={value === null ? "cond-value cond-absent" : "cond-value num"}>
        {value ?? "no reading"}
      </dd>
      {note && <dd className="cond-note">{note}</dd>}
    </div>
  );
}

export function ConditionsRail() {
  const { payload, species, hour, state, error } = useDay();

  if (!payload || !species) {
    return (
      <aside className="rail" data-testid="rail" aria-label="This hour">
        <p className="eyebrow">This hour</p>
        <p className="rail-empty" data-testid="rail-empty">
          {state === "failed"
            ? (error ?? "This day could not be scored.")
            : "Waiting for the day's scores."}
        </p>
      </aside>
    );
  }

  const scored = payload.species[species]?.hours[hour];
  const now = payload.conditions[hour];
  const water = payload.water;
  const astro = payload.astro;

  // Which factors vary per feature -- read, never assumed. The rail shows the
  // hour's own subs, so the ones on this list are bay-wide readings and the
  // rows say so.
  const perFeature = new Set(payload.sub_scope.feature);
  const subs: FactorSub[] = (scored?.subs ?? []).map((sub) => ({
    ...sub,
    ...(perFeature.has(sub.factor)
      ? { note: "bay-wide — this factor varies per feature" }
      : {}),
  }));

  const tidePhase = now?.tide_phase;
  const tideTarget = tidePhase === "rising" ? "high" : tidePhase === "falling" ? "low" : null;

  return (
    <aside className="rail" data-testid="rail" aria-label="This hour">
      <header className="rail-head">
        <p className="eyebrow">
          This hour
          <span className="rail-clock num" data-testid="rail-clock">
            {clock(now?.time) ?? `${String(hour).padStart(2, "0")}:00`}
          </span>
        </p>
        <p className="rail-score">
          <span className="num" data-testid="rail-score">
            {scored ? scored.score : "—"}
          </span>
          <span className="rail-score-unit">
            bite score · {species.replace(/_/g, " ")}
          </span>
        </p>
        {scored && (
          <p className="rail-disclosure" data-testid="rail-disclosure">
            <span className="num">{percent(scored.confidence) ?? "—"}</span> of the authored
            weight resolved · <span className="num">
              {percent(scored.constrained_share) ?? "—"}
            </span>{" "}
            of that rests on an observation
          </p>
        )}
        {scored && scored.excluded.length > 0 && (
          <p className="rail-disclosure" data-testid="rail-excluded">
            excluded from this hour: {scored.excluded.join(", ")}
          </p>
        )}
      </header>

      <section className="rail-block">
        <p className="eyebrow">Conditions</p>
        {now ? (
          <dl className="cond">
            <Row
              label="Tide"
              value={fixed(now.tide_height_ft, 2) === null ? null : `${fixed(now.tide_height_ft, 2)} ft`}
              note={parts(
                tidePhase,
                // `tide_frac` is the fraction through the CURRENT half-cycle
                // (0 at the last turn, 1 at the next), so it is stated as
                // distance to the NEXT turn rather than as "of the cycle" --
                // which is a different number the stage factor uses.
                isReading(now.tide_frac) && tideTarget
                  ? `${percent(now.tide_frac)} of the way to ${tideTarget}`
                  : null,
              )}
            />
            <Row
              label="Wind"
              value={
                fixed(now.wind_speed_kn, 1) === null ? null : `${fixed(now.wind_speed_kn, 1)} kn`
              }
              note={parts(
                compass(now.wind_dir_deg)
                  ? `from ${compass(now.wind_dir_deg)} (${fixed(now.wind_dir_deg, 0)}°)`
                  : null,
                fixed(now.wind_gust_kn, 1) ? `gusting ${fixed(now.wind_gust_kn, 1)} kn` : null,
              )}
            />
            <Row
              label="Pressure"
              value={fixed(now.pressure_mb, 1) === null ? null : `${fixed(now.pressure_mb, 1)} mb`}
              note={
                signed(now.pressure_trend_mb_3h, 1)
                  ? `${signed(now.pressure_trend_mb_3h, 1)} mb over 3 h`
                  : null
              }
            />
            <Row
              label="Air"
              value={fixed(now.air_temp_f, 1) === null ? null : `${fixed(now.air_temp_f, 1)} °F`}
            />
            {/* Already a percentage in the payload, so it is printed, not
                converted -- unlike the 0..1 fractions elsewhere here. */}
            <Row
              label="Cloud"
              value={fixed(now.cloud_cover_pct, 0) === null ? null : `${fixed(now.cloud_cover_pct, 0)}%`}
            />
            <Row
              label="Rain"
              value={fixed(now.precip_in, 2) === null ? null : `${fixed(now.precip_in, 2)} in`}
            />
          </dl>
        ) : (
          <p className="rail-empty" data-testid="cond-absent">
            This day carries no conditions row for hour {String(hour).padStart(2, "0")}.
          </p>
        )}
      </section>

      <section className="rail-block">
        {/* Day-level, and labelled as such: water temperature and the sun and
            moon times are properties of the DAY, not of this hour. */}
        <p className="eyebrow">The day</p>
        <dl className="cond">
          {water ? (
            <Row
              label="Water"
              value={fixed(water.temp_f, 2) === null ? null : `${fixed(water.temp_f, 2)} °F`}
              note={
                signed(water.temp_trend_f_3d, 2)
                  ? `${signed(water.temp_trend_f_3d, 2)} °F over 3 days`
                  : null
              }
            />
          ) : (
            <Row label="Water" value={null} note="no water temperature for this day" />
          )}
          {astro ? (
            <>
              <Row
                label="Sun"
                value={
                  clock(astro.sunrise) && clock(astro.sunset)
                    ? `${clock(astro.sunrise)} → ${clock(astro.sunset)}`
                    : null
                }
                note={parts(
                  clock(astro.dawn) ? `first light ${clock(astro.dawn)}` : null,
                  clock(astro.dusk) ? `last light ${clock(astro.dusk)}` : null,
                )}
              />
              <Row
                label="Moon"
                value={
                  percent(astro.moon_phase_frac)
                    ? `${percent(astro.moon_phase_frac)} lit`
                    : null
                }
                note={parts(
                  clock(astro.moonrise) ? `rise ${clock(astro.moonrise)}` : null,
                  clock(astro.moonset) ? `set ${clock(astro.moonset)}` : null,
                )}
              />
            </>
          ) : (
            <Row label="Sun & moon" value={null} note="no astronomy for this day" />
          )}
        </dl>
      </section>

      <section className="rail-block">
        <FactorBars
          caption="Factors"
          subs={subs}
          empty="This hour carries no scored factors."
        />
      </section>

      <footer className="rail-foot">
        <p>
          Click a marker for that feature's own flow, salinity and structure — the three
          factors that change from one piece of water to the next.
        </p>
        {payload.missing.length > 0 && (
          <p data-testid="rail-missing">
            Sources unavailable for this day: {payload.missing.join(", ")}.
          </p>
        )}
      </footer>
    </aside>
  );
}

export default ConditionsRail;
