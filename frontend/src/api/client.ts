import type { DayPayload, DayStatus, FisherySummary } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function fetchFisheries(): Promise<FisherySummary[]> {
  return json<FisherySummary[]>(await fetch("/api/fisheries"));
}

export type DayResult =
  | { kind: "ready"; payload: DayPayload }
  | { kind: "building" };

export async function fetchDay(
  slug: string,
  date: string,
  model: string,
): Promise<DayResult> {
  const res = await fetch(`/api/fisheries/${slug}/day/${date}?model=${model}`);
  // 202 is not an error -- the day is being built. Anything else non-ok is.
  if (res.status === 202) return { kind: "building" };
  return { kind: "ready", payload: await json<DayPayload>(res) };
}

export async function fetchStatus(
  slug: string,
  date: string,
  model: string,
): Promise<DayStatus> {
  return json<DayStatus>(
    await fetch(`/api/fisheries/${slug}/day/${date}/status?model=${model}`),
  );
}

export function layerUrl(slug: string, name: string): string {
  return `/api/fisheries/${slug}/layers/${name}`;
}
