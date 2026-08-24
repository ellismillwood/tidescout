"""Water Quality Portal salinity results -- the anchors the fit never had.

WHY THIS SOURCE EXISTS
----------------------
Phase 2 fitted the intrusion model against the full NERRS record and the
model was FALSIFIED, not merely unconstrained: rmse 4.060 ppt against an
observation resolution of 0.003, a factor of 1,353, with a healthy condition
number. The binding cause is coverage -- only 0.4% of the 2,162-feature
inventory sat where observations bracketed it, and the 2.58-13.05 km reach
holding 35% of features had no salinity observation at all.

WQP (waterqualitydata.us, the EPA/USGS/state aggregator) serves 208 salinity
stations in this bbox, 132 in-domain, 55 of them in that reach -- including
Winyah Bay MAIN CHANNEL stations at 5.56, 10.28 and 12.17 km. Measured live
2026-08-24. It is public and unauthenticated, unlike CDMO.

GRAB SAMPLES, NOT A FEED, AND WHY THAT IS FINE HERE
---------------------------------------------------
These are discrete samples -- WB-06 has 40 over four years, not 40 per day.
That would be useless to a model needing a time series, but this codebase
already holds the tide model and 10.6 years of composite discharge, so each
sample resolves to a known DISTANCE, DISCHARGE and TIDAL PHASE. One grab
sample with all three is a fully-specified observation.

That is also why a row with no usable time is REJECTED rather than defaulted
to noon: the phase is the point, and a fabricated time is a fabricated phase.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone

# The WQP characteristic this module reads, and the ONLY one it will read.
# WQP also serves "Specific conductance" and "Conductivity"; `sources/usgs.py`
# already holds the line that specific conductance is a different quantity and
# is not interchangeable with salinity. Mixing them here would be silent.
CHARACTERISTIC = "Salinity"

# Unit code -> multiplier onto psu. `0/00` is per-mille, numerically identical
# to ppt (81 ppt rows and 42 `0/00` rows in one real response). Anything not
# in here is REJECTED AND COUNTED: WQP serves mg/l and uS/cm under neighbouring
# characteristics and coercing one would inject nonsense at full confidence.
ACCEPTED_UNITS: dict[str, float] = {"ppt": 1.0, "0/00": 1.0, "psu": 1.0, "PSU": 1.0}

# `ResultStatusIdentifier` values admitted. "Final", "Accepted" and "Validated"
# are reviewed; "Preliminary" and "Provisional" are not, and are blocked --
# the same posture cdmo.py takes toward unvetted QAQC flags.
ACCEPTED_STATUSES = frozenset({"Final", "Accepted", "Validated", "Historical"})

# `ActivityTypeCode` prefixes that are NOT estuary measurements: field blanks,
# lab replicates, spikes. They pass every other filter and would enter the fit
# as real readings.
_QC_ACTIVITY_PREFIX = "Quality Control"

# WQP reports local clock time plus a US timezone abbreviation. There is no
# stdlib mapping from those abbreviations to offsets, and getting it wrong
# shifts a sample by up to 4 h -- most of a quarter tidal cycle at 12.42 h.
_TZ_OFFSETS: dict[str, int] = {
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
}


@dataclass(frozen=True)
class Sample:
    """One admitted grab sample."""

    station: str
    ts: datetime  # UTC, tz-aware
    salinity_psu: float
    # None means WQP recorded no depth -- true for 120 of 123 real rows.
    # NEVER 0.0, which would claim "surface".
    depth_m: float | None


@dataclass
class ParseReport:
    """Admitted samples plus a full account of what was not admitted.

    Every rejection path has a counter, and `test_counters_account_for_every_row`
    pins their sum to `n_rows`. A rejection with no counter is a silent drop,
    and silent drops are how a fit quietly narrows its own inputs while
    looking complete.
    """

    samples: list[Sample] = field(default_factory=list)
    n_rows: int = 0
    n_admitted: int = 0
    n_no_time: int = 0
    n_bad_unit: int = 0
    n_bad_status: int = 0
    n_qc_activity: int = 0
    n_no_value: int = 0
    unknown_units: dict[str, int] = field(default_factory=dict)
    unknown_statuses: dict[str, int] = field(default_factory=dict)


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _timestamp(date_s: str, time_s: str, tz_s: str) -> datetime | None:
    """Local clock time + US tz abbreviation -> UTC. None if unusable."""
    date_s, time_s, tz_s = date_s.strip(), time_s.strip(), tz_s.strip().upper()
    if not date_s or not time_s or tz_s not in _TZ_OFFSETS:
        return None
    tz = timezone(timedelta(hours=_TZ_OFFSETS[tz_s]))
    try:
        naive = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    except ValueError:
        return None
    return naive.astimezone(UTC)


def _depth_m(value: str, unit: str) -> float | None:
    value, unit = value.strip(), unit.strip().lower()
    if not value:
        return None
    try:
        d = float(value)
    except ValueError:
        return None
    if unit in ("m", "meters", "meter"):
        return d
    if unit in ("ft", "feet", "foot"):
        return d * 0.3048
    return None


def parse_results(fh: Iterable[str]) -> ParseReport:
    """WQP Result CSV -> admitted salinity samples, plus why the rest went."""
    report = ParseReport()
    for row in csv.DictReader(fh):
        if (row.get("CharacteristicName") or "").strip() != CHARACTERISTIC:
            continue  # a different parameter entirely; not this module's business
        report.n_rows += 1

        activity = (row.get("ActivityTypeCode") or "").strip()
        if activity.startswith(_QC_ACTIVITY_PREFIX):
            report.n_qc_activity += 1
            continue

        status = (row.get("ResultStatusIdentifier") or "").strip()
        if status and status not in ACCEPTED_STATUSES:
            report.n_bad_status += 1
            _bump(report.unknown_statuses, status)
            continue

        raw = (row.get("ResultMeasureValue") or "").strip()
        if not raw or (row.get("ResultDetectionConditionText") or "").strip():
            report.n_no_value += 1
            continue
        try:
            value = float(raw)
        except ValueError:
            report.n_no_value += 1
            continue

        unit = (row.get("ResultMeasure/MeasureUnitCode") or "").strip()
        if unit not in ACCEPTED_UNITS:
            report.n_bad_unit += 1
            _bump(report.unknown_units, unit or "<blank>")
            continue

        ts = _timestamp(
            row.get("ActivityStartDate", ""),
            row.get("ActivityStartTime/Time", ""),
            row.get("ActivityStartTime/TimeZoneCode", ""),
        )
        if ts is None:
            report.n_no_time += 1
            continue

        report.samples.append(
            Sample(
                station=(row.get("MonitoringLocationIdentifier") or "").strip(),
                ts=ts,
                salinity_psu=value * ACCEPTED_UNITS[unit],
                depth_m=_depth_m(
                    row.get("ActivityDepthHeightMeasure/MeasureValue", ""),
                    row.get("ActivityDepthHeightMeasure/MeasureUnitCode", ""),
                ),
            )
        )
        report.n_admitted += 1
    return report
