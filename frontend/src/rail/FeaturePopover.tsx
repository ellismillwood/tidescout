/**
 * What one marker says when you click it -- and the two-scope merge behind it.
 *
 * THE MERGE IS THE POINT OF THIS FILE. The payload splits a feature-hour's ten
 * factors across two places on purpose:
 *
 *   species[name].hours[i].subs               -- the factors that depend only
 *                                                on the hour and the species
 *   species[name].features[key].hours[i].subs -- the factors that genuinely
 *                                                vary per feature
 *
 * They live apart because shipping all ten on every feature-hour asserted
 * something false (that all ten vary per feature -- measured: seven produce
 * exactly one distinct value per hour across all 529 in-domain features) and
 * cost 21.1 MB of a 47.9 MB payload to assert it. Display has to put them back
 * together, and that is `mergeFeatureSubs`.
 *
 * `payload.sub_scope` says which factor lives at which scope, and this file
 * READS it rather than keeping its own copy of the split. That field exists
 * precisely so a factor changing scope -- salinity becoming bay-wide, a new
 * factor arriving that reads a feature's own geometry -- cannot break the
 * frontend silently. A hardcoded list here would fail a test that shuffles
 * `sub_scope` and expects the merge to move with it.
 *
 * THE FEATURE'S OWN VALUE WINS. `flow` and `salinity` appear at BOTH scopes:
 * the hour carries the bay-wide reading, the feature carries its own. If the
 * hour's value won, every marker on the chart would show the same flow, which
 * is the exact thing per-feature scoring exists to avoid.
 */
import { useEffect } from "react";

import type { DayPayload } from "../api/types";
import { useDay } from "../state/DayContext";
import { FactorBars, type FactorSub } from "./FactorBars";
import { percent } from "./format";
import "./FeaturePopover.css";

/** A merged sub, plus WHERE it came from -- the popover groups by that. */
export interface MergedSub extends FactorSub {
  scope: "hour" | "feature";
}

export interface MergeResult {
  subs: MergedSub[];
  /**
   * Factors the payload shipped but placed at NEITHER scope.
   *
   * Always empty against a correct payload: the backend derives both halves of
   * `sub_scope` from one rule, so every factor it ships is declared somewhere.
   * Reported rather than dropped in silence because the merge is scope-driven
   * -- if that invariant ever broke, a factor would vanish from this panel with
   * nothing saying so, and the popover names the count instead.
   */
  undeclared: string[];
}

/**
 * The two scopes, put back together for one feature-hour.
 *
 * Hour-scope subs come from `species[name].hours[i]` (full `SubScore`s, so
 * they carry weight/missing/provisional); feature-scope subs come from the
 * feature's own `hours[i]`, which the payload trims to factor/value/reason,
 * so the merged sub carries no weight and this file does not invent one.
 * Feature-hours are aligned by POSITION -- a feature-hour has no `time` of its
 * own, by design.
 */
export function mergeScopes(
  payload: DayPayload,
  species: string,
  featureKey: string,
  hour: number,
): MergeResult {
  const block = payload.species[species];
  const hourSubs = block?.hours[hour]?.subs ?? [];
  const featureHour = block?.features[featureKey]?.hours[hour];
  const featureSubs = featureHour?.subs ?? [];
  // A trimmed sub carries no flags of its own, but the feature-hour publishes
  // both lists beside them. Reading those back is what keeps the popover's
  // salinity bar wearing the "modelled, not measured" weave -- without it the
  // ONE factor on Winyah Bay that nothing observed constrains would render as
  // an ordinary solid bar, which is the disclosure this project exists to
  // make. Not invention: these are the payload's own words about this hour.
  const featureProvisional = new Set(featureHour?.provisional ?? []);
  const featureExcluded = new Set(featureHour?.excluded ?? []);

  const hourScope = new Set(payload.sub_scope.hour);
  const featureScope = new Set(payload.sub_scope.feature);

  const merged = new Map<string, MergedSub>();
  for (const sub of hourSubs) {
    // The hour's copy of a FEATURE-scope factor is the bay-wide reading, and
    // it is skipped here -- the feature's own value replaces it below. This
    // is the line that stops every marker showing the same flow.
    if (!hourScope.has(sub.factor)) continue;
    merged.set(sub.factor, {
      factor: sub.factor,
      value: sub.value,
      reason: sub.reason,
      weight: sub.weight,
      missing: sub.missing,
      provisional: sub.provisional,
      scope: "hour",
    });
  }
  for (const sub of featureSubs) {
    if (!featureScope.has(sub.factor)) continue;
    merged.set(sub.factor, {
      factor: sub.factor,
      value: sub.value,
      reason: sub.reason,
      missing: featureExcluded.has(sub.factor),
      provisional: featureProvisional.has(sub.factor),
      scope: "feature",
    });
  }

  const undeclared: string[] = [];
  for (const factor of new Set([...hourSubs, ...featureSubs].map((s) => s.factor))) {
    if (!hourScope.has(factor) && !featureScope.has(factor)) undeclared.push(factor);
  }
  return { subs: [...merged.values()], undeclared };
}

/** The ten factors for one feature-hour, hour-scope first, feature-scope last. */
export function mergeFeatureSubs(
  payload: DayPayload,
  species: string,
  featureKey: string,
  hour: number,
): MergedSub[] {
  return mergeScopes(payload, species, featureKey, hour).subs;
}

export interface FeaturePopoverProps {
  featureKey: string;
  onClose: () => void;
}

export function FeaturePopover({ featureKey, onClose }: FeaturePopoverProps) {
  const { payload, species, hour } = useDay();

  // Escape closes it. A popover that can only be dismissed with a mouse is a
  // popover a keyboard user cannot get out of.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!payload || !species) return null;

  const block = payload.species[species]?.features[featureKey];
  const featureHour = block?.hours[hour];
  const { subs, undeclared } = mergeScopes(payload, species, featureKey, hour);
  const hourScoped = subs.filter((sub) => sub.scope === "hour");
  const featureScoped = subs.filter((sub) => sub.scope === "feature");

  return (
    <aside
      className="popover"
      data-testid="feature-popover"
      aria-label={`Feature ${featureKey}`}
    >
      <header className="popover-head">
        <div className="popover-title">
          <p className="eyebrow">{block?.type ?? "feature"}</p>
          <p className="popover-key num">{featureKey}</p>
        </div>
        <p className="popover-activation">
          <span className="num" data-testid="popover-activation">
            {featureHour ? featureHour.activation : "—"}
          </span>
          <span className="popover-unit">
            activation · {String(hour).padStart(2, "0")}:00
          </span>
        </p>
        <button
          type="button"
          className="popover-close"
          onClick={onClose}
          aria-label="Close this feature"
        >
          ×
        </button>
      </header>

      {featureHour ? (
        <>
          {/* The backend's own sentence about this feature-hour, verbatim. */}
          <p className="popover-reason" data-testid="popover-reason">
            {featureHour.reason}
          </p>
          <p className="popover-meta">
            <span className="popover-meta-item">
              <span className="num">{percent(featureHour.confidence) ?? "—"}</span> of the
              authored weight resolved
            </span>
            <span className="popover-meta-item">
              <span className="num">{percent(featureHour.constrained_share) ?? "—"}</span> of
              that rests on an observation
            </span>
          </p>
        </>
      ) : (
        <p className="popover-reason" data-testid="popover-unscored">
          This feature was detected but not scored for {species.replace(/_/g, " ")} at{" "}
          {String(hour).padStart(2, "0")}:00 — it sits outside the flow model's domain. The
          hour's own factors below still apply; its flow, salinity and structure are
          unknown.
        </p>
      )}

      <FactorBars
        testId="factor-bars-hour"
        caption="This hour — the same at every feature"
        subs={hourScoped}
        empty="The payload places no factor at hour scope."
      />
      <FactorBars
        testId="factor-bars-feature"
        caption="This feature — its own water"
        subs={featureScoped}
        empty="No feature-scope factors: this feature is not scored at this hour."
      />

      {/* Why the three below carry no weight while the seven above do. An
          unexplained gap in a column of numbers reads as a bug; this is the
          payload's trim, and saying so costs one line. */}
      {featureScoped.some((sub) => sub.weight === undefined) && (
        <p className="popover-note" data-testid="popover-trimmed">
          A feature's own factors arrive trimmed to value and reason, so no weight is shown
          for them. They are weighted: the number just is not repeated on every feature-hour.
        </p>
      )}

      {undeclared.length > 0 && (
        <p className="popover-note" data-testid="popover-undeclared">
          {undeclared.length}{" "}
          {undeclared.length === 1 ? "factor is" : "factors are"} not shown —{" "}
          {undeclared.join(", ")} {undeclared.length === 1 ? "is" : "are"} shipped by this
          payload but placed at neither scope by its own `sub_scope`.
        </p>
      )}
    </aside>
  );
}

export default FeaturePopover;
