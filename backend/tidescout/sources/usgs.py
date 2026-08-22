from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from statistics import fmean

import httpx

from tidescout.models import Fishery
from tidescout.sources.cache import Cache

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
OBS_TTL = timedelta(minutes=15)

PARAM_DISCHARGE = "00060"
PARAM_TEMP_C = "00010"
PARAM_SALINITY = "00480"

FRESHNESS_CUTOFF = timedelta(hours=6)


@dataclass
class DischargeSummary:
    cfs_now: float | None
    cfs_lagged: float | None
    bucket: str
    sites: list[str]
    contributing: list[str]
    stale: list[str]
    trend: float | None = None  # cfs_now / cfs_lagged
    limb: str = "unknown"


# Fractional change between today's flow and the 24-48 h lagged mean, below
# which the river is called steady. Gauge noise and the diurnal cycle move a
# coastal-plain river a few percent on any quiet day; 15% is comfortably above
# that and comfortably below any real rain event.
LIMB_DEAD_BAND = 0.15


def classify_limb(summary: DischargeSummary) -> str:
    """"rising" / "falling" / "steady" / "unknown".

    The salt front lags discharge by days, so the same flow means different
    things on the way up and on the way down: a rising limb is a freshet
    arriving and fish moving down-bay ahead of it; a falling limb is the slower
    recovery as salt creeps back. The level alone cannot distinguish them.
    """
    if summary.cfs_now is None or summary.cfs_lagged is None or not summary.cfs_lagged:
        return "unknown"
    change = (summary.cfs_now - summary.cfs_lagged) / summary.cfs_lagged
    if change > LIMB_DEAD_BAND:
        return "rising"
    if change < -LIMB_DEAD_BAND:
        return "falling"
    return "steady"


def branch_discharge_cfs(fishery: Fishery, summary: DischargeSummary) -> dict[str, float]:
    """Composite discharge split across rivers by their measured share.

    Uses `cfs_lagged` in preference to `cfs_now`: the bay's salinity today
    reflects the last day or two of river flow, not this instant's reading.

    Splits by `inflow_share`, not gauge `weight` -- see Phase 1 Task 2. This is
    the runtime twin of the ANUGA forcing correction, and it matters more here:
    intrusion length is a strong function of the discharge on each branch, so a
    78/13/8 river system modelled as equal thirds puts the salt front in the
    wrong place on all three.
    """
    basis = summary.cfs_lagged if summary.cfs_lagged is not None else summary.cfs_now
    if basis is None:
        return {}
    shares = [r.inflow_share for r in fishery.rivers]
    if any(s is None for s in shares):
        n = len(fishery.rivers) or 1
        shares = [1.0 / n] * len(fishery.rivers)
    return {r.name: basis * s for r, s in zip(fishery.rivers, shares, strict=True)}


@dataclass
class WaterSummary:
    temp_f: float | None
    temp_trend_f_3d: float | None
    salinity_ppt: float | None
    source: str


def fetch_series(
    sites: list[str], params: list[str], period_days: int, cache: Cache
) -> dict[tuple[str, str], list[tuple[datetime, float]]]:
    sites = [s for s in sites if s]
    if not sites:
        return {}
    query = {
        "format": "json",
        "sites": ",".join(sites),
        "parameterCd": ",".join(params),
        "period": f"P{period_days}D",
        "siteStatus": "all",
    }

    def fetch() -> dict:
        resp = httpx.get(IV_URL, params=query, timeout=30)
        resp.raise_for_status()
        return resp.json()

    key = f"{query['sites']}:{query['parameterCd']}:{period_days}"
    cached = cache.get_or_fetch("usgs-iv", key, OBS_TTL, fetch)
    # Accumulate per (site, param) across every timeSeries entry and every
    # values[] block within it: USGS can split one site/param pair across
    # multiple entries (e.g. provisional vs. approved) and can repeat the same
    # instantaneous timestamp across those splits. Keying the accumulator by
    # timestamp merges all of that into one series and turns a repeat into a
    # dedupe (the later occurrence in payload order wins) instead of either a
    # silent overwrite of a whole entry or a duplicate-timestamp row surviving
    # into the result.
    by_key: dict[tuple[str, str], dict[datetime, float]] = {}
    for ts in cached.payload.get("value", {}).get("timeSeries", []):
        try:
            site = ts["sourceInfo"]["siteCode"][0]["value"]
            param = ts["variable"]["variableCode"][0]["value"]
        except (KeyError, IndexError, TypeError):
            continue  # malformed entry; skip it, keep processing the rest
        points = by_key.setdefault((site, param), {})
        for block in ts.get("values", []):
            for p in block.get("value", []):
                try:
                    v = float(p["value"])
                    t = datetime.fromisoformat(p["dateTime"]).astimezone(UTC)
                except (KeyError, TypeError, ValueError):
                    continue  # malformed point: missing/non-ISO dateTime or value
                if v <= -999:  # USGS sentinel for missing
                    continue
                points[t] = v
    return {pair: sorted(pts.items()) for pair, pts in by_key.items() if pts}


def fetch_daily(
    sites: list[str], param: str, start: str, end: str, cache: Cache
) -> dict[str, list[tuple[date, float]]]:
    """Daily mean values (NWIS dv service). Immutable once published, so cached
    with no TTL -- calibration reads a year of history and must not refetch."""
    sites = [s for s in sites if s]
    if not sites:
        return {}
    query = {
        "format": "json",
        "sites": ",".join(sites),
        "parameterCd": param,
        "startDT": start,
        "endDT": end,
        "statCd": "00003",  # daily mean
        "siteStatus": "all",
    }

    def fetch() -> dict:
        resp = httpx.get(DV_URL, params=query, timeout=60)
        resp.raise_for_status()
        return resp.json()

    key = f"dv:{query['sites']}:{param}:{start}:{end}"
    cached = cache.get_or_fetch("usgs-dv", key, None, fetch)
    out: dict[str, list[tuple[date, float]]] = {}
    for ts in cached.payload.get("value", {}).get("timeSeries", []):
        try:
            site = ts["sourceInfo"]["siteCode"][0]["value"]
        except (KeyError, IndexError, TypeError):
            continue
        rows: list[tuple[date, float]] = []
        for block in ts.get("values", []):
            for p in block.get("value", []):
                try:
                    v = float(p["value"])
                    d = date.fromisoformat(p["dateTime"][:10])
                except (KeyError, TypeError, ValueError):
                    continue
                if v <= -999:
                    continue
                rows.append((d, v))
        if rows:
            out[site] = sorted(rows)
    return out


def discharge_summary(fishery: Fishery, cache: Cache) -> DischargeSummary:
    sites = [r.usgs_site for r in fishery.rivers if r.usgs_site]
    weights = {r.usgs_site: r.weight for r in fishery.rivers if r.usgs_site}
    series = fetch_series(sites, [PARAM_DISCHARGE], 4, cache)
    now = datetime.now(UTC)
    total_now = 0.0
    total_lagged = 0.0
    got_now = got_lagged = False
    contributing: list[str] = []
    stale: list[str] = []
    for site in sites:
        points = series.get((site, PARAM_DISCHARGE), [])
        if not points:
            stale.append(site)
            continue
        w = weights.get(site, 1.0)
        last_t, last_v = points[-1]
        if now - last_t > FRESHNESS_CUTOFF:
            stale.append(site)  # dark gauge: do not let a 4-day-old value in
        else:
            total_now += last_v * w
            got_now = True
            contributing.append(site)
        lag_window = [v for t, v in points if timedelta(hours=24) <= now - t <= timedelta(hours=48)]
        if lag_window:
            total_lagged += fmean(lag_window) * w
            got_lagged = True
    cfs_now = total_now if got_now else None
    cfs_lagged = total_lagged if got_lagged else None
    basis = cfs_lagged if cfs_lagged is not None else cfs_now
    if basis is None:
        bucket = "med"
    elif basis < fishery.discharge_buckets.low_below_cfs:
        bucket = "low"
    elif basis > fishery.discharge_buckets.high_above_cfs:
        bucket = "high"
    else:
        bucket = "med"
    summary = DischargeSummary(cfs_now, cfs_lagged, bucket, sites, contributing, stale)
    summary.trend = (
        cfs_now / cfs_lagged if cfs_now is not None and cfs_lagged not in (None, 0.0) else None
    )
    summary.limb = classify_limb(summary)
    return summary


def _daily_means(points: list[tuple[datetime, float]]) -> dict:
    days: dict = {}
    for t, v in points:
        days.setdefault(t.date(), []).append(v)
    return {d: fmean(vs) for d, vs in days.items()}


def water_summary(fishery: Fishery, cache: Cache, month: int) -> WaterSummary:
    usgs_sensors = [w for w in fishery.stations.water if w.kind == "usgs"]
    temp_f = trend = salinity = None
    source = "climatology"
    if usgs_sensors:
        sites = [w.station for w in usgs_sensors]
        series = fetch_series(sites, [PARAM_TEMP_C, PARAM_SALINITY], 7, cache)
        for w in usgs_sensors:
            temp_points = series.get((w.station, PARAM_TEMP_C), [])
            if temp_points and temp_f is None:
                temp_f = temp_points[-1][1] * 9 / 5 + 32
                means = _daily_means(temp_points)
                days = sorted(means)
                if len(days) >= 4:
                    latest, prior = days[-1], days[-4:-1]
                    trend = (means[latest] - fmean([means[d] for d in prior])) * 9 / 5
                source = f"usgs:{w.station}"
            sal_points = series.get((w.station, PARAM_SALINITY), [])
            if sal_points and salinity is None:
                salinity = sal_points[-1][1]
    if temp_f is None:
        temp_f = fishery.climatology.water_temp_f_by_month[month]
        source = "climatology"
    if salinity is None:
        salinity = fishery.climatology.salinity_ppt_by_month[month]
    return WaterSummary(temp_f, trend, salinity, source)
