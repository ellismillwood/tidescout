// Hand-written, and pinned by tests/contract.test.ts against a REAL recorded
// payload. Not generated: `get_day` returns a bare JSONResponse so OpenAPI
// describes it as untyped, and giving it a Pydantic response_model to fix
// that would let FastAPI FILTER the response -- which is exactly why
// `missing` and `confidence` reach us verbatim today. See spec §2.

export interface SubScore {
  factor: string;
  value: number | null;
  weight: number;
  reason: string;
  missing: boolean;
  provisional: boolean;
}

/** A feature-hour's subs are trimmed: factor/value/reason only. */
export interface TrimmedSub {
  factor: string;
  value: number | null;
  reason: string;
}

export interface HourScore {
  time: string;
  score: number;
  subs: SubScore[];
  excluded: string[];
  confidence: number;
  constrained_share: number;
  provisional: string[];
}

/** No `time`: a feature-hour is positionally aligned with species.hours[i]. */
export interface FeatureHour {
  activation: number;
  reason: string;
  confidence: number;
  constrained_share: number;
  excluded: string[];
  provisional: string[];
  subs: TrimmedSub[];
}

export interface FeatureBlock {
  type: string;
  hours: FeatureHour[];
}

export interface SpeciesBlock {
  hours: HourScore[];
  features: Record<string, FeatureBlock>;
}

export interface Conditions {
  time: string;
  air_temp_f: number | null;
  wind_speed_kn: number | null;
  wind_dir_deg: number | null;
  wind_gust_kn: number | null;
  pressure_mb: number | null;
  pressure_trend_mb_3h: number | null;
  cloud_cover_pct: number | null;
  precip_in: number | null;
  tide_height_ft: number | null;
  tide_phase: string | null;
  tide_frac: number | null;
}

export interface SubScope {
  hour: string[];
  feature: string[];
}

/** Day-level, not hourly -- see the note on TOP_LEVEL in the contract test. */
export interface WaterSummary {
  temp_f: number | null;
  temp_trend_f_3d: number | null;
}

export interface Astro {
  dawn: string | null;
  sunrise: string | null;
  sunset: string | null;
  dusk: string | null;
  moon_phase_frac: number | null;
  moonrise: string | null;
  moonset: string | null;
}

export interface DayPayload {
  slug: string;
  day: string;
  model_label: string;
  missing: string[];
  freshness: { day: string; model_label: string; generated_at: string };
  sub_scope: SubScope;
  flow: {
    range_bucket: string;
    discharge_cfs: number | null;
    discharge_bucket: string | null;
    regimes: [string, number][];
    clamped: boolean;
  };
  species: Record<string, SpeciesBlock>;
  salinity: {
    cfs: number | null;
    fitted: boolean;
    extrapolated: boolean;
    series: { time: string; ppt: number; provenance: string }[];
    representative_ppt: number | null;
    representative_hour: string | null;
    provenance: string | null;
  };
  conditions: Conditions[];
  water: WaterSummary | null;
  astro: Astro | null;
}

export interface FisherySummary {
  slug: string;
  name: string;
  center: [number, number];
  timezone: string;
  ready: boolean;
  reason?: string;
}

export type DayStatus =
  | { status: "ready"; generated_at: string | null; stale: boolean }
  | { status: "building"; error?: null }
  | { status: "failed"; error: string }
  | { status: "absent" };
