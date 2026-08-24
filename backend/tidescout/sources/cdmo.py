"""CDMO historical water-quality import: the NERRS record NDBC cannot serve.

Task 8 gave Phase 2's salinity fit a second along-estuary distance (WYSS1,
15.02 km) by polling NDBC's `realtime2/WYSS1.ocean` -- but that feed is a
ROLLING ~45-day WINDOW (see `sources/ndbc.py`'s module docstring). Everything
older, and every other North Inlet-Winyah Bay (NIW) station besides WYSS1,
only exists in NERRS's Centralized Data Management Office (CDMO) archive.
This module imports a file the user downloads by hand from CDMO's query
interface (there is no public unauthenticated HTTP endpoint to poll the way
`sources/ndbc.py` polls NDBC -- the fishery YAML already records that CDMO's
SOAP web service returns "Invalid ip" without IP registration; the query
interface / "zip downloads" feature does not have that restriction and is
what a human uses instead).

WHY A SIBLING MODULE, NOT AN EXTENSION OF `sources/ndbc.py`
-------------------------------------------------------------
The STORE is reused untouched: `NdbcStore`, `default_store`, and the
`(station, ts)`-keyed, whole-batch-atomic `append` this module writes
through are exactly Task 8's, imported and called, not reimplemented (see
`import_file` below). What is NOT shared is the parsing: CDMO's export is
comma-delimited with a per-parameter QA/QC flag column (`F_<param>`), its
own missing-value convention (blank cells, not `"MM"`), its own station-code
scheme (`niw` + 2-letter site + `wq`, not a five-character NDBC buoy ID),
and -- most consequentially -- its own timestamp convention (fixed Eastern
STANDARD time, never NDBC's UTC; see below). `parse_ocean`'s per-row
`ValueError`-on-malformed-shape contract does not transfer either: a CDMO
export can run to hundreds of thousands of rows across years, and a single
QAQC-flagged or malformed cell deep inside it is the EXPECTED shape of real
environmental data, not evidence the whole file is corrupt (see "QAQC
FLAGS" below). Folding this into `ndbc.py` would mean two incompatible
parsing philosophies sharing one file for no reason; a sibling module that
imports the store keeps the accumulation contract shared while keeping the
formats' actual differences visible instead of papered over.

WHAT I ESTABLISHED ABOUT THE FORMAT, AND FROM WHERE
-------------------------------------------------------------
No real CDMO export was available when this was written (see "IF THE FILE
ISN'T HERE YET" in `import_path`). Established instead from:

1. The reserve's own metadata PDF, fetched live 2026-08-23:
   https://cdmo.baruch.sc.edu/waf/YearlyFiles/North%20Inlet%20Winyah%20Bay/water%20quality/metadata/niwwq01-12.22m.pdf
   -- Section 10 (station codes), 11 (QAQC flag definitions), 12 (QAQC error
   codes), and the deployment-log tables ("Deployment dates and times (in
   Eastern Standard Time) for 2022 follow").
2. A REAL example CDMO "zip downloads" export, linked directly from the
   `import_local()` function's own documentation in the SWMPr R package
   (the standard tool researchers use to read CDMO data), fetched live from
   https://s3.amazonaws.com/swmpexdata/zip_ex.zip -- 33.7 MB of real
   Apalachicola (APA) water-quality CSVs. This is what pinned down the exact
   column layout, quoting, and flag-cell syntax below; it is a different
   reserve than Winyah Bay's NIW, but the CDMO export code (SWMPr, CDMO's
   own NERRQAQC macro) is shared across every reserve, so the FORMAT
   generalises even though the station codes and values do not.
3. SWMPr's own source (github.com/fawda123/SWMPr, `R/time_vec.R` and
   `R/qaqc.R`), fetched live, for the timezone table and the flag-matching
   convention respectively.

The confirmed CSV shape (header, verbatim from the real zip_ex file):

    StationCode,isSWMP,DateTimeStamp,Historical,ProvisionalPlus,F_Record,
    Temp,F_Temp,SpCond,F_SpCond,Sal,F_Sal,DO_Pct,F_DO_Pct,DO_mgl,F_DO_mgl,
    Depth,F_Depth,cDepth,F_cDepth,Level,F_Level,cLevel,F_cLevel,
    pH,F_pH,Turb,F_Turb,ChlFluor,F_ChlFluor,

One row:

    "apacpwq   ","P",01/01/2012 0:00,0,1,"",16.9,<0> ,45.97,<0> ,29.9,<0> ,...

`StationCode` and a few string-typed cells are quoted; numeric values and
flag cells are not. Every parameter has a paired `F_<param>` flag column
holding the numeric QAQC flag inside angle brackets (`<0>`, `<-3>`), with
room in the same bracket group for an alphanumeric QAQC CODE per Section 12
(e.g. `SCF`, `CRE`) -- none appeared in the real 35,000-row file fetched, so
the code-capturing group below is exercised only by the synthetic fixture,
not (yet) by a real one. `csv.DictReader` parses this correctly as-is (the
unquoted numeric and flag cells are still valid CSV fields); no custom
tokenizer was needed. `StationCode` is present on EVERY row, so a station is
read from the row, not assumed from a filename -- CDMO's "Custom Query"
export mode explicitly combines multiple stations into one file, unlike the
one-station-per-file "zip download" mode the real example above uses; this
parser groups by the row's own `StationCode` so either shape is handled the
same way.

TIMESTAMP CONVENTION: FIXED EASTERN STANDARD TIME, NO DST -- NOT UTC
-------------------------------------------------------------
Established from two independent primary sources, not assumed:

* The NIW metadata PDF states outright: "Deployment dates and times (in
  Eastern Standard Time) for 2022 follow" ahead of its per-station
  deployment logs.
* SWMPr's `time_vec.R` hard-codes a per-reserve GMT-offset table with the
  comment "no DST!" and maps `niw` to a GMT offset of exactly -5,
  represented internally as the tz `America/Jamaica` -- a real IANA zone
  chosen BECAUSE it sits at a permanent UTC-5 with no DST rule, unlike
  `America/New_York` which would silently shift the data by an hour every
  March and November.

So `DateTimeStamp` is `datetime(..., tzinfo=CDMO_TZ)` where `CDMO_TZ` is a
fixed `timezone(timedelta(hours=-5))` -- built directly, not via a DST-aware
zone that happens to read UTC-5 today. Getting this wrong would smear the
tidal signal (a DST-aware conversion silently shifts every summer
timestamp by exactly one hour, i.e. by more than a third of the M2 tide's
~12.4 h period) -- exactly the failure this task's dispatch warned against.
`NdbcStore.append` immediately converts to UTC on write
(`r.ts.astimezone(UTC)`), so once `parse_cdmo_csv` hands back a correctly
tz-aware datetime, the rest of the pipeline (including cross-source dedupe
against NDBC's UTC timestamps -- see `test_cdmo.py`) is unaffected by which
input convention produced it.

QAQC FLAGS: LOAD-BEARING, NOT DECORATION
-------------------------------------------------------------
Full vocabulary (Section 11 of the metadata PDF, `FLAG_MEANINGS` below):

    -5 Outside High Sensor Range        0 Data Passed Initial QAQC Checks
    -4 Outside Low Sensor Range         1 Suspect Data
    -3 Data Rejected due to QAQC        2 Depth from surface/near-surface sonde
    -2 Missing Data                     3 Calculated (e.g. barometric-corrected depth)
    -1 Optional parameter not collected 4 Historical: Pre-Auto QAQC
                                         5 Corrected Data

`ACCEPTED_FLAGS = {0, 2, 3, 5}`. Flags 0 and 5 are the two flags that mean
"this is a real, reviewed number" (initial pass, or explicitly corrected).
2 and 3 are INFORMATIONAL rather than a quality judgement -- they apply only
to Depth/cDepth/Level/cLevel (2: this sonde is mounted at/near the surface,
which is WYSS1's actual deployment per the metadata's site description; 3:
this is the barometric-pressure-corrected calculated value, the NORMAL state
for `cDepth`/`cLevel` in the real sample, not an exception). Everything else
is excluded: -5/-4 (out of sensor range), -3 (rejected), -2/-1 (nothing was
collected), 1 (suspect -- named explicitly in this task's dispatch as data
that "must not silently enter a calibration"), and 4. Flag 4 is a genuine
judgement call: "pre-auto-QAQC" means the point predates the automated
primary check entirely, i.e. it has been vetted by NOTHING, not merely
flagged suspect. SWMPr's own `qaqc()` function -- the standard tool
researchers use to read this exact data -- defaults `qaqc_keep = '0'` ONLY,
with the comment "Generally, only data with the '0' QAQC flag should be
used." Admitting 2/3/5 here is already looser than that default; extending
it to unvetted historical data on top would weaken the one guardrail this
task's dispatch called "load-bearing." Excluded until there is a concrete
reason to trust it.

A cell whose value is non-blank but whose flag is missing entirely (should
not happen in a well-formed export -- every real parameter column is paired
with an `F_` column) is treated as REJECTED, not admitted: a value with no
accompanying quality judgement gets the same non-benefit-of-the-doubt as an
explicitly bad one. A flag cell that does not match the documented
`<int>` shape at all is also never admitted, and is counted separately
(`StationParse.n_flag_unparseable`) so a real format disagreement is visible
in the import report rather than silently swallowed.

A non-blank VALUE cell that fails `float()` (corrupt cell, not a QAQC
question) is dropped to `None` and counted
(`StationParse.n_value_unparseable`) rather than raising. This is a
deliberate departure from `parse_ocean`'s per-row `ValueError`: that
module had a REAL captured file to be strict against (Task 8's 4,235-row
capture); this one has only a documented format and a synthetic fixture,
against a real file that can run to hundreds of thousands of rows over many
years. One corrupt cell aborting an entire multi-year, multi-station import
would be worse than dropping that one reading and reporting it -- the
dispatch's own instruction ("report clearly ... rather than guessing")
reads as asking for exactly this trade-off. A structurally wrong file (no
`StationCode`/`DateTimeStamp` header at all) still raises `ValueError`
immediately, before any row is read and before the store is touched --
see `parse_cdmo_csv`.

STATION-CODE ALIASING: WYSS1 IS niwwswq
-------------------------------------------------------------
CDMO's station code for "Winyah Bay surface" is `niwwswq`. Its documented
position (metadata PDF Section 5/10: 33 deg 18'33.88" N, 79 deg 17'19.57" W
= 33.309411, -79.288769 decimal) sits within ~30 m of WYSS1's independently
documented position (33.309, -79.289 -- see `sources/ndbc.py`'s module
docstring) -- the same physical sonde, reported through two federal feeds
under two different naming conventions. `STATION_ALIASES` maps
`niwwswq -> "WYSS1"` so this importer writes into the SAME store key Task 8
already uses, which is what makes `NdbcStore`'s `(station, ts)` primary key
actually unify the two feeds into one series, per this task's dispatch:
"the union must be stored exactly once." Every other NIW station has no
NDBC mirror and keeps its own CDMO code, uppercased for the same display
convention WYSS1 already uses (`canonical_station`).

WHAT THE DISPATCH GOT WRONG: NIW RUNS SIX STATIONS, NOT FOUR
-------------------------------------------------------------
The dispatch states "each NERRS reserve operates four water quality
stations." The NIW metadata PDF's own "SWMP Station Timeline" table lists
SIX currently-active water-quality stations -- Clambank Landing (CB),
Debidue Creek (DC), Oyster Landing (OL), Thousand Acre (TA), Winyah Bay
surface (WS/WYSS1), and Winyah Bay bottom (WB) -- plus one historical,
long-decommissioned site (Caledonia, 1995 only, outside the reserve
boundary). `NIW_STATION_COORDS_LONLAT` below carries all six current
stations, not four.

MET FILE SUPPORT (Task 10)
-------------------------------------------------------------
No real CDMO MET export was available either (same situation as WQ above).
Established instead from two independent primary sources, both fetched
live 2026-08-23:

1. The reserve's own METEOROLOGICAL metadata PDF (a SEPARATE document from
   the water-quality one above):
   https://cdmo.baruch.sc.edu/waf/YearlyFiles/North%20Inlet%20Winyah%20Bay/
   meteorological/metadata/niwmet01-12.24m.pdf -- Section 4 (parameter
   list and units), Section 5 (station location/coordinates), Section 10
   (station code), and Section 11 (QAQC flag definitions, MET-specific).
2. SWMPr's own `R/param_names.R` (github.com/fawda123/SWMPr), which lists
   the MET parameter set the package expects CDMO to export, in order:
   `atemp, rh, bp, wspd, maxwspd, wdir, sdwdir, totpar, totprcp, cumprcp,
   totsorad`. This is what pins down the exact column set and order below
   (title-cased to CDMO's export convention, e.g. `ATemp`/`F_ATemp`,
   exactly like `Temp`/`F_Temp` above) -- the same authority Task 9 used
   for the WQ column layout.

ONE STATION, NOT SIX: each NERRS reserve runs exactly one weather station
system-wide, not one per water-quality site. NIW's is at Oyster Landing
(the met PDF's own words: "The weather station is located at Oyster
Landing (OL) pier"), station code `niwolmet` -> canonical `NIWOLMET` (no
alias needed -- `canonical_station`'s existing uppercase fallback already
produces this). `NIW_MET_STATION_COORDS_LONLAT` below carries that one
entry, independently sourced from the met PDF's own DMS coordinates (33
20'57.85"N, 79 11'20.03"W) -- a few feet from the WQ PDF's Oyster Landing
water-quality coordinates (33 20'57.70"N, 79 11'19.97"W), which is exactly
what "same pier, independently surveyed sensor mount" should look like;
collapsing the two onto one shared position would be less honest than
keeping the small, real discrepancy visible.

WHY A SIBLING TABLE, NOT A WIDENED `observations`
-------------------------------------------------------------
Water-quality and meteorological parameters are two genuinely disjoint
sets -- temperature/salinity/DO/turbidity/pH/chlorophyll vs. air-temp/
wind/pressure/humidity/PAR/precipitation -- with disjoint station
namespaces (WYSS1 and five siblings vs. one `NIWOLMET`). Cramming both
into one row shape would mean every WQ row carries eight-plus always-NULL
weather columns and vice versa: the sparse-schema smell this codebase
already rejects elsewhere (see `ndbc.py`'s MM-to-None reasoning) and a
direct conflict with this task's own "keep units distinct and labelled"
instruction. `NdbcStore.met_observations` (see `ndbc.py`'s "A SECOND TABLE
FOR METEOROLOGICAL DATA") is a second table in the SAME file, not a new
store class or a new file, because the atomic-transaction crash-safety
`observations` already has (one `sqlite3` connection, one context manager)
applies to a second table for free -- no new mechanism to build or prove.
It stays in `ndbc.py`, not a third module, because that module already
owns the store's SQLite lifecycle (`NdbcStore.__init__`) and the WQ table
it parallels; this module (`cdmo.py`) stays the ONE place that parses
CDMO's CSV/zip shape, for both WQ and MET files, because the low-level
cell/flag parsing (`_parse_value`, `_parse_flag`, `_resolve`) and the
station/timestamp/zip-vs-directory plumbing are IDENTICAL between the two
formats -- only the column map and the accepted-flags set differ (see
below), and forking that shared plumbing into a second parsing module
would duplicate it for no reason, the same argument this module's own
opening section makes against folding WQ parsing into `ndbc.py`.

MET-SPECIFIC QAQC: FLAGS 2 AND 3 ARE RESERVED, NOT ADMITTED
-------------------------------------------------------------
`MET_ACCEPTED_FLAGS = {0, 5}` -- narrower than WQ's `{0, 2, 3, 5}`, and
this is a REAL, documented difference, not an oversight. WQ's flags 2
("Depth from surface/near-surface sonde") and 3 ("Calculated data, e.g.
barometric-corrected depth") have specific, load-bearing meanings for
depth/level parameters. The MET metadata PDF's own Section 11 defines the
SAME numeric flags 2 and 3 as "Open - reserved for later flag" for
meteorological parameters -- CDMO has not assigned them any meaning yet.
Admitting them here would silently invent an interpretation CDMO itself
has not made; excluding them costs nothing in practice (no real MET
export should ever carry them while they remain reserved), and keeps this
importer correct if CDMO ever does define them, since a real future 2/3
value would then need a real decision, not an inherited WQ one.

COLUMNS DELIBERATELY DROPPED
-------------------------------------------------------------
`CumPrcp` (cumulative precipitation) is documented in SWMPr's parameter
list but the met metadata PDF says outright (remark 13.d): "Cumulative
precipitation is no longer available via export from the CDMO." Dropped
via `_MET_UNMAPPED_PARAM_COLUMNS`, same treatment as WQ's cDepth/Level/
cLevel -- a deliberate omission, excluded from `unknown_columns` so
dropping it doesn't read as "could not parse." Battery voltage (mentioned
in the met PDF's prose as an internally-averaged quantity) is NOT in
SWMPr's exported parameter list at all, so it isn't in `_MET_COLUMN_MAP`
either -- if a real export ever does carry a `Battery` column, this
parser's `unknown_columns` reporting will surface it rather than silently
drop it, the same safety net WQ's unmapped-column detection already
provides. A time-of-max-wind-speed column (the met PDF's prose mentions
"Maximum Wind Speed (m/s) and time") is likewise absent from SWMPr's
authoritative exported-parameter list, so no column for it is expected --
the "and time" appears to describe an internal 5-second-data product, not
something that reaches the 15-minute CDMO export.
"""

import csv
import io
import re
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tidescout.sources.ndbc import MetObservation, NdbcStore, Observation

__all__ = [
    "ACCEPTED_FLAGS",
    "BATCH_ROWS",
    "CDMO_TZ",
    "FLAG_MEANINGS",
    "MET_ACCEPTED_FLAGS",
    "MET_FLAG_MEANINGS",
    "NIW_MET_STATION_COORDS_LONLAT",
    "NIW_STATION_COORDS_LONLAT",
    "QAQC_CODE_MEANINGS",
    "SOURCE_CDMO_MET",
    "SOURCE_CDMO_WQ",
    "STATION_ALIASES",
    "STATION_METADATA_FILENAME",
    "ImportReport",
    "ParseResult",
    "StationImport",
    "StationMeta",
    "StationParse",
    "canonical_station",
    "find_station_metadata",
    "import_file",
    "import_path",
    "load_station_metadata",
    "read_station_metadata",
    "parse_cdmo_csv",
    "parse_cdmo_met_csv",
    "station_kind",
]

# Provenance `source` labels for the two CDMO parameter families -- see
# `ndbc.py`'s "PROVENANCE AND CITATION" and `SOURCE_NDBC_REALTIME2` there.
# One interleaved file records BOTH, because it contains both.
SOURCE_CDMO_WQ = "cdmo:water_quality"
SOURCE_CDMO_MET = "cdmo:meteorological"

# Rows buffered per table before they are handed to SQLite. Peak memory for
# an import is set by this, not by the file: the real 494 MB export streams
# through in a bounded working set. See `ndbc.BulkWriter`.
BATCH_ROWS = 20_000

# Fixed UTC-5, year-round -- built directly rather than via a DST-aware zone.
# See the module docstring's "TIMESTAMP CONVENTION". This is the FALLBACK
# used when no station metadata is available; when it is, each station's own
# `GMT Offset` column is read instead (see `StationMeta.tz`).
CDMO_TZ = timezone(timedelta(hours=-5))

FLAG_MEANINGS: dict[int, str] = {
    -5: "Outside High Sensor Range",
    -4: "Outside Low Sensor Range",
    -3: "Data Rejected due to QAQC",
    -2: "Missing Data",
    -1: "Optional SWMP Supported Parameter (not collected)",
    0: "Data Passed Initial QAQC Checks",
    1: "Suspect Data",
    2: "Depth collected from surface or near-surface sonde",
    3: "Calculated data (e.g. barometric-corrected depth/level)",
    4: "Historical Data: Pre-Auto QAQC",
    5: "Corrected Data",
}

# See the module docstring's "QAQC FLAGS" section for why each flag is or
# isn't here.
ACCEPTED_FLAGS = frozenset({0, 2, 3, 5})

# MET metadata PDF Section 11, verbatim -- differs from FLAG_MEANINGS only
# at 2 and 3 ("Open - reserved for later flag" instead of WQ's specific
# depth/barometric-correction meanings). See the module docstring's
# "MET-SPECIFIC QAQC" section.
MET_FLAG_MEANINGS: dict[int, str] = {
    -5: "Outside High Sensor Range",
    -4: "Outside Low Sensor Range",
    -3: "Data Rejected due to QAQC",
    -2: "Missing Data",
    -1: "Optional SWMP supported parameter",
    0: "Passed Initial QAQC Checks",
    1: "Suspect Data",
    2: "Open - reserved for later flag",
    3: "Open - reserved for later flag",
    4: "Historical Data: Pre-Auto QAQC",
    5: "Corrected Data",
}

# Narrower than ACCEPTED_FLAGS -- 2 and 3 are undefined for MET parameters,
# not a WQ-style real judgement. See "MET-SPECIFIC QAQC" above.
MET_ACCEPTED_FLAGS = frozenset({0, 5})

# The alphanumeric QAQC codes that accompany a flag, from CDMO's own QAQC
# page (https://cdmo.baruch.sc.edu/data/qaqc.cfm, fetched live 2026-08-23).
# G = general error, S = sensor error, C = comment. These EXPLAIN a flag;
# they never override it -- see "QUALIFIER CODES" in the module docstring.
QAQC_CODE_MEANINGS: dict[str, str] = {
    # -- general errors -------------------------------------------------
    "GCC": "Calculated value could not be determined; missing correction",
    "GCM": "Calculated value could not be determined due to missing data",
    "GCR": "Calculated value could not be determined due to rejected data",
    "GCS": "Calculated value suspect due to questionable data",
    "GCU": "Calculated value could not be determined due to unavailable data",
    "GDM": "Data missing",
    "GIC": "Incorrect calibration",
    "GIM": "Instrument malfunction",
    "GIT": "Instrument recording error; recovered telemetry data",
    "GMC": "No instrument deployed due to maintenance/calibration",
    "GMT": "Instrument maintenance",
    "GNF": "Data not found",
    "GOW": "Out of water event",
    "GPD": "Power down",
    "GPF": "Power failure / low battery",
    "GPR": "Program reload",
    "GQD": "Data rejected due to QAQC (deployment)",
    "GQR": "Data rejected due to QAQC checks",
    "GQS": "Data suspect due to QAQC checks",
    "GSM": "See metadata",
    # -- sensor errors --------------------------------------------------
    "SBL": "Sensor blocked",
    "SBO": "Blocked optic",
    "SCB": "Sensor cable failure",
    "SCC": "Sensor cap failure",
    "SCF": "Conductivity sensor failure",
    "SCS": "Sensor conditions suspect",
    "SDF": "Depth sensor failure",
    "SDG": "Suspect due to sensor diagnostics",
    "SDO": "DO suspect",
    "SDP": "Depth port blocked",
    "SFD": "Fouled sensor / drift",
    "SIC": "Incorrect calibration / contaminated standard",
    "SIW": "Sensor intermittently out of water",
    "SMA": "Sensor malfunction, adjusted",
    "SMT": "Sensor maintenance",
    "SNV": "Negative value",
    "SOC": "Out of calibration",
    "SOW": "Sensor out of water",
    "SPC": "Post calibration out of range",
    "SQR": "Sensor data rejected",
    "SRD": "Sensor reading drift",
    "SSD": "Sensor drift",
    "SSM": "Sensor malfunction",
    "SSN": "Sensor noise",
    "SSR": "Sensor out of range",
    "STF": "Catastrophic temperature sensor failure",
    "STS": "Temperature sensor suspect",
    "SUL": "Sensor unreliable",
    "SWM": "Wiper malfunction / loss",
    "SXD": "Depth from surface/near-surface sonde at fixed depth",
    # -- comments -------------------------------------------------------
    "CAB": "Algal bloom",
    "CAF": "Acceptable calibration/accuracy error of sensor",
    "CAP": "Depth/level sensor affected by atmospheric pressure",
    "CBF": "Biofouling",
    "CCU": "Cause unknown",
    "CDA": "DO hypoxia (<3 mg/L)",
    "CDB": "Disturbed bottom",
    "CDF": "Data appear to fit conditions",
    "CDR": "Data reviewed",
    "CFK": "Fish kill",
    "CHB": "Heavy biofouling",
    "CIF": "Instrument fouled",
    "CIP": "Incorrect prefix",
    "CLE": "Low estuary flow",
    "CLT": "Low tide",
    "CMC": "In field maintenance/cleaning",
    "CMD": "Mud in probe guard",
    "CML": "Snow melt from previous snowfall event",
    "CND": "New deployment begins",
    "CRE": "Significant rain event",
    "CSM": "See metadata",
    "CTS": "Turbidity spike",
    "CUS": "Unusual conditions",
    "CVT": "Value exceeds sensor specifications",
    "CWD": "Data collected at wrong depth",
    "CWE": "Significant weather event",
}

# niwwswq is WYSS1 -- see "STATION-CODE ALIASING" above.
STATION_ALIASES: dict[str, str] = {"niwwswq": "WYSS1"}

# CDMO's station-code convention: <3-letter reserve><2-letter site><type>.
# Verified against the real `sampling_stations.csv`: across all 367 stations
# system-wide the suffix agrees with the file's own `Station Type` column
# with ZERO exceptions (wq->1, met->0, nut->2). See "ROUTING BY STATION
# CODE" in the module docstring.
_KIND_BY_SUFFIX: dict[str, str] = {"wq": "wq", "met": "met", "nut": "nut"}
_STATION_TYPE_TO_KIND: dict[str, str] = {"0": "met", "1": "wq", "2": "nut"}

STATION_METADATA_FILENAME = "sampling_stations.csv"


def canonical_station(cdmo_code: str) -> str:
    """The `NdbcStore` key a CDMO station code should be written under."""
    code = cdmo_code.strip().lower()
    return STATION_ALIASES.get(code, code.upper())


def station_kind(cdmo_code: str) -> str:
    """"wq", "met", "nut" or "unknown", from the code's own suffix.

    This is what routes a row of the real interleaved export to the right
    table. It reads the code the ROW carries, so it works on a file whose
    header cannot decide the question (the real export's header carries both
    parameter families) and on a station the metadata file has never heard
    of.
    """
    code = cdmo_code.strip().lower()
    for suffix, kind in _KIND_BY_SUFFIX.items():
        if code.endswith(suffix):
            return kind
    return "unknown"


def _dms(deg: float, mins: float, secs: float) -> float:
    return deg + mins / 60.0 + secs / 3600.0


# (lon, lat) WGS84, decimal-converted from the DMS coordinates published in
# the reserve's own metadata PDFs. Kept as a documented FALLBACK for when no
# `sampling_stations.csv` accompanies an export -- `load_station_metadata`
# is the primary source now (see "GEOLOCATION" in the module docstring).
# `test_cdmo_real_export.py` cross-checks the two against each other, so a
# real disagreement is loud rather than silent.
NIW_STATION_COORDS_LONLAT: dict[str, tuple[float, float]] = {
    "NIWCBWQ": (-_dms(79, 11, 34.62), _dms(33, 20, 2.05)),  # Clambank Landing
    "NIWDCWQ": (-_dms(79, 10, 2.81), _dms(33, 21, 36.49)),  # Debidue Creek
    "NIWOLWQ": (-_dms(79, 11, 19.97), _dms(33, 20, 57.70)),  # Oyster Landing
    "NIWTAWQ": (-_dms(79, 15, 21.75), _dms(33, 17, 57.03)),  # Thousand Acre
    "WYSS1": (-_dms(79, 17, 19.57), _dms(33, 18, 33.88)),  # Winyah Bay surface (niwwswq)
    "NIWWBWQ": (-_dms(79, 17, 19.58), _dms(33, 18, 33.94)),  # Winyah Bay bottom
}

NIW_MET_STATION_COORDS_LONLAT: dict[str, tuple[float, float]] = {
    "NIWOLMET": (-_dms(79, 11, 20.03), _dms(33, 20, 57.85)),  # Oyster Landing (niwolmet)
}


@dataclass(frozen=True)
class StationMeta:
    """One row of CDMO's `sampling_stations.csv`, made usable.

    `lon` is NEGATED on the way in -- see "GEOLOCATION: THE LONGITUDE SIGN
    TRAP" in the module docstring. `tz` comes from the row's own `GMT
    Offset`, so an export from a reserve that is not on UTC-5 is not
    silently shifted.
    """

    code: str  # lowercase, whitespace-stripped
    canonical: str
    name: str
    lon: float  # WEST-NEGATIVE decimal degrees
    lat: float
    gmt_offset_hours: int
    kind: str  # "wq" | "met" | "nut" | "unknown"
    status: str
    active_dates: str

    @property
    def tz(self) -> timezone:
        return timezone(timedelta(hours=self.gmt_offset_hours))

    @property
    def lonlat(self) -> tuple[float, float]:
        return (self.lon, self.lat)


_STATION_META_REQUIRED = frozenset({"station code", "latitude", "longitude"})


def looks_like_station_metadata(header: Sequence[str]) -> bool:
    """True if this header is CDMO's station table, not an observation
    export. Both are `.csv` and live in the same download directory; the
    station table has no `DateTimeStamp` and would otherwise be fed to the
    observation parser and abort the whole import."""
    keys = {h.strip().lower() for h in header}
    return _STATION_META_REQUIRED <= keys and "datetimestamp" not in keys


def load_station_metadata(path: Path) -> dict[str, StationMeta]:
    """Every station in CDMO's `sampling_stations.csv`, keyed by lowercase code.

    Reads the FILE, never a hardcoded table: the real one carries all 367
    NERRS stations system-wide, so a future export that adds a station is
    geolocated with no code change.

    Two things are corrected on the way in, both of which are silent
    disasters if missed -- see the module docstring's "GEOLOCATION" section:
    the longitude sign, and the assumption that every reserve is on UTC-5.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return read_station_metadata(fh, source=str(path))


def read_station_metadata(fh: Iterable[str], source: str = "<stream>") -> dict[str, StationMeta]:
    """`load_station_metadata` against an already-open stream -- what a zip
    member goes through, and what the path version delegates to."""
    out: dict[str, StationMeta] = {}
    reader = csv.DictReader(fh)
    if not reader.fieldnames or not looks_like_station_metadata(reader.fieldnames):
        raise ValueError(
            f"{source} does not look like CDMO's station table -- expected columns "
            f"{sorted(_STATION_META_REQUIRED)}, got {reader.fieldnames}"
        )
    reader.fieldnames = [f.strip() for f in reader.fieldnames]
    for row in reader:
        code = (row.get("Station Code") or "").strip().lower()
        if not code:
            continue
        lat_raw = (row.get("Latitude") or "").strip()
        lon_raw = (row.get("Longitude") or "").strip()
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except ValueError:
            continue  # a station with no surveyed position; reported as unlocated
        try:
            offset = int(float((row.get("GMT Offset") or "").strip()))
        except ValueError:
            offset = -5
        kind = _STATION_TYPE_TO_KIND.get(
            (row.get("Station Type") or "").strip(), station_kind(code)
        )
        out[code] = StationMeta(
            code=code,
            canonical=canonical_station(code),
            name=(row.get("Station Name") or "").strip(),
            # NEGATED: CDMO stores west longitudes as positive magnitudes.
            lon=-abs(lon),
            lat=lat,
            gmt_offset_hours=offset,
            kind=kind,
            status=(row.get("Status") or "").strip(),
            active_dates=(row.get("Active Dates") or "").strip(),
        )
    return out


def find_station_metadata(near: Path) -> Path | None:
    """`sampling_stations.csv` beside an export, if the user downloaded it."""
    directory = near if near.is_dir() else near.parent
    for candidate in (directory / STATION_METADATA_FILENAME,):
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob("*.csv"))
    for candidate in matches:
        try:
            with candidate.open(newline="", encoding="utf-8", errors="replace") as fh:
                header = next(csv.reader(fh), [])
        except OSError:
            continue
        if looks_like_station_metadata(header):
            return candidate
    return None


# CDMO column name (lowercased) -> `Observation` field. `eh_mv` has no CDMO
# counterpart (the SWMP parameter set carries no ORP sensor) and is always
# `None` for a CDMO-sourced row -- the same "not measured" convention
# `ndbc.py` uses for CLCON/EH on WYSS1's real feed.
_COLUMN_MAP: dict[str, str] = {
    "temp": "water_temp_c",
    "spcond": "cond_ms_cm",  # specific conductance, mS/cm -- see NOTE below
    "sal": "salinity_psu",
    "do_pct": "o2_pct",
    "do_mgl": "o2_ppm",
    "depth": "depth_m",
    "ph": "ph",
    "turb": "turbidity_ftu",
    "chlfluor": "chlorophyll_ug_l",
}
# NOTE on SpCond -> cond_ms_cm: CDMO's `SpCond` is SPECIFIC conductance
# (temperature-normalised to 25 C), not the raw conductivity NDBC's `COND`
# reports -- a real but second-order distinction (both are mS/cm, both are
# still NOT salinity). Mapped into the same column Task 8 established
# because the load-bearing conflation this codebase guards against is
# conductivity-vs-salinity, not raw-vs-specific conductance, and the two
# columns are never in the same row (a row is CDMO-sourced or NDBC-sourced,
# never both) so nothing here ever mixes the two conductivity conventions
# within one series.

_UNMAPPED_PARAM_COLUMNS = frozenset({"cdepth", "level", "clevel"})

# CDMO MET column name (lowercased) -> `MetObservation` field.
_MET_COLUMN_MAP: dict[str, str] = {
    "atemp": "air_temp_c",
    "rh": "rh_pct",
    "bp": "bp_mb",
    "wspd": "wind_speed_ms",
    "maxwspd": "max_wind_speed_ms",
    "wdir": "wind_dir_deg",
    "sdwdir": "wind_dir_sd_deg",
    "totpar": "par_mmol_m2",
    "totprcp": "precip_mm",
    "totsorad": "solar_rad_wm2",
}

_MET_UNMAPPED_PARAM_COLUMNS = frozenset({"cumprcp"})

# Non-parameter columns a real export carries. `""` covers the trailing
# empty-named column produced by CDMO's own trailing comma. `historical`
# and `provisionalplus` appear in the SWMPr "zip download" shape but NOT in
# the real query-interface export -- both are tolerated.
_METADATA_COLUMNS = frozenset(
    {"stationcode", "isswmp", "datetimestamp", "historical", "provisionalplus", "f_record", ""}
)

_REQUIRED_COLUMNS = frozenset({"stationcode", "datetimestamp"})

_TS_FORMAT = "%m/%d/%Y %H:%M"

# `<flag>` optionally followed by qualifier codes. The real export puts the
# codes AFTER the closing bracket, in square brackets (QAQC error codes) and
# parentheses (comment codes): `<-3> [SSM] (CSM)`. Group 2 also tolerates a
# code INSIDE the bracket (`<1 SDG>`), the shape Task 9 inferred from the
# documentation, so both parse.
_FLAG_RE = re.compile(r"^<\s*(-?\d+)\s*([^>]*)>\s*(.*)$")

# Three-character alphanumeric codes, wherever they sit in the qualifier
# text -- `[GSM] (CWD)` yields GSM and CWD without caring which bracket
# each came from.
_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2}\b")


def _parse_flag(cell: str) -> tuple[int | None, bool, tuple[str, ...]]:
    """(flag, was_parseable, qualifier_codes).

    `flag` is `None` for a blank cell (nothing flagged) or an unparseable
    one; `was_parseable` tells the two apart so the caller can report a real
    format disagreement rather than treat it as ordinary missingness.
    """
    cell = cell.strip()
    if not cell:
        return None, True, ()
    m = _FLAG_RE.match(cell)
    if not m:
        return None, False, ()
    qualifiers = f"{m.group(2)} {m.group(3)}"
    return int(m.group(1)), True, tuple(_CODE_RE.findall(qualifiers))


def _parse_value(cell: str) -> tuple[float | None, bool]:
    """(value, was_parseable). Blank is a valid, parseable "no value"."""
    cell = cell.strip()
    if not cell:
        return None, True
    try:
        return float(cell), True
    except ValueError:
        return None, False


def _resolve(
    value_cell: str, flag_cell: str, accepted_flags: frozenset[int] = ACCEPTED_FLAGS
) -> tuple[float | None, str, tuple[str, ...]]:
    """One (value, flag) column pair -> the value if QAQC accepts it, else
    `None` -- plus an outcome tag ("ok", "blank", "rejected",
    "unparseable_value", "unparseable_flag") and the qualifier codes the
    flag cell carried.

    Order matters: an unparseable flag is reported as such even if the value
    itself parsed fine, since a flag CDMO's own documented syntax cannot be
    read is the more serious problem to surface.

    The NUMERIC FLAG alone decides admission. The qualifier codes are
    returned so they can be counted and reported, never to gate -- see
    "QUALIFIER CODES" in the module docstring for the two independent
    sources that say so.
    """
    value, value_ok = _parse_value(value_cell)
    flag, flag_ok, codes = _parse_flag(flag_cell)
    if not flag_ok:
        return None, "unparseable_flag", ()
    if not value_ok:
        return None, "unparseable_value", codes
    if value is None:
        return None, "blank", codes
    if flag is None or flag not in accepted_flags:
        return None, "rejected", codes
    return value, "ok", codes


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


@dataclass
class StationParse:
    """Everything one CDMO station's rows amounted to, and every way some of
    them could not be used, counted rather than silently dropped.

    `observations` is populated ONLY by `parse_cdmo_csv`/`parse_cdmo_met_csv`
    -- the whole-file, in-memory reading path. `import_file` streams, and
    deliberately never materialises it; read `n_admitted` and `span` there.
    """

    raw_code: str
    kind: str  # "wq" | "met"
    canonical: str
    observations: list = field(default_factory=list)
    n_rows: int = 0
    n_bad_timestamp: int = 0
    # Rows that parsed but yielded no admitted value at all -- CDMO rejected
    # or never collected every parameter. Stored (the `(station, ts)` key is
    # a real "we looked, there was nothing" record, matching `parse_ocean`'s
    # all-MM rows) but counted, because a station that is mostly these is a
    # station whose history is thinner than its row count suggests.
    n_empty: int = 0
    n_admitted: dict[str, int] = field(default_factory=dict)
    n_rejected_by_flag: dict[str, int] = field(default_factory=dict)
    n_value_unparseable: dict[str, int] = field(default_factory=dict)
    n_flag_unparseable: dict[str, int] = field(default_factory=dict)
    # Every qualifier code seen on this station's cells, and separately the
    # subset carried by cells the flag ADMITTED -- the only ones that can
    # let a caveat into calibration data unnoticed.
    qualifier_codes: dict[str, int] = field(default_factory=dict)
    qualifier_codes_admitted: dict[str, int] = field(default_factory=dict)
    unknown_qualifier_codes: dict[str, int] = field(default_factory=dict)
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    @property
    def span(self) -> tuple[datetime, datetime] | None:
        if self.first_ts is None or self.last_ts is None:
            return None
        return (self.first_ts, self.last_ts)

    def _note_ts(self, ts: datetime) -> None:
        if self.first_ts is None or ts < self.first_ts:
            self.first_ts = ts
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts


@dataclass
class ParseResult:
    """One file's worth of parsing. `stations` holds water-quality stations,
    `met_stations` meteorological ones -- both come out of the SAME pass over
    the SAME file, because the real export interleaves them."""

    stations: dict[str, StationParse] = field(default_factory=dict)
    met_stations: dict[str, StationParse] = field(default_factory=dict)
    unknown_columns: list[str] = field(default_factory=list)
    # Station codes that are neither `wq` nor `met` (CDMO's `nut` nutrient
    # stations, or something new) -- skipped rather than written as empty
    # rows, and counted so the skip is visible.
    skipped_stations: dict[str, int] = field(default_factory=dict)
    n_rows: int = 0


class _Layout:
    """Column positions for one CDMO header, resolved once per file.

    Built from `csv.reader` output (a list) rather than `DictReader` (a dict
    per row): at 2.5 million rows x 45 columns the dict construction is the
    single largest avoidable cost, and integer indexing into the row list is
    exactly as clear.
    """

    def __init__(self, header: Sequence[str]):
        names = [h.strip() for h in header]
        lower: dict[str, int] = {}
        for i, name in enumerate(names):
            lower.setdefault(name.lower(), i)

        missing = _REQUIRED_COLUMNS - set(lower)
        if missing:
            raise ValueError(
                f"missing required column(s) {sorted(missing)} -- this does not look like a "
                f"CDMO export; header was {list(header)}"
            )

        self.station_i = lower["stationcode"]
        self.ts_i = lower["datetimestamp"]
        self.wq_pairs = self._pairs(lower, _COLUMN_MAP)
        self.met_pairs = self._pairs(lower, _MET_COLUMN_MAP)
        self.wq_absent = [f for c, f in _COLUMN_MAP.items() if c not in lower]
        self.met_absent = [f for c, f in _MET_COLUMN_MAP.items() if c not in lower]

        known = (
            _METADATA_COLUMNS
            | set(_COLUMN_MAP)
            | set(_MET_COLUMN_MAP)
            | _UNMAPPED_PARAM_COLUMNS
            | _MET_UNMAPPED_PARAM_COLUMNS
        )
        self.unknown_columns = sorted(
            names[i] for key, i in lower.items() if key not in known and not key.startswith("f_")
        )

    @staticmethod
    def _pairs(
        lower: dict[str, int], column_map: dict[str, str]
    ) -> list[tuple[str, str, int, int | None]]:
        return [
            (col, obs_field, lower[col], lower.get(f"f_{col}"))
            for col, obs_field in column_map.items()
            if col in lower
        ]


def _cell(row: Sequence[str], i: int | None) -> str:
    """A row shorter than the header is not fatal, and a row LONGER than it
    is ordinary: 63 rows in the real 2.5 M-row export carry one extra
    trailing field."""
    if i is None or i >= len(row):
        return ""
    return row[i]


class _Reader:
    """One streaming pass over a CDMO export.

    Routes every row by the station code the ROW carries (`station_kind`) --
    never by the file's header or its name. The real export's single header
    carries both parameter families and cannot decide the question, and the
    row is the only thing that can.
    """

    def __init__(
        self,
        layout: _Layout,
        stations: Mapping[str, StationMeta] | None = None,
        collect: bool = False,
    ):
        self.layout = layout
        self.stations = stations or {}
        self.collect = collect
        self.result = ParseResult(unknown_columns=list(layout.unknown_columns))

    def _station(self, raw_code: str, kind: str) -> StationParse:
        bucket = self.result.stations if kind == "wq" else self.result.met_stations
        sp = bucket.get(raw_code)
        if sp is None:
            sp = StationParse(
                raw_code=raw_code, kind=kind, canonical=canonical_station(raw_code)
            )
            bucket[raw_code] = sp
        return sp

    def _tz(self, raw_code: str) -> timezone:
        meta = self.stations.get(raw_code)
        return meta.tz if meta is not None else CDMO_TZ

    def row(self, row: Sequence[str]) -> tuple[str, str, Observation | MetObservation] | None:
        """Parse one row. `None` means it produced no observation -- a blank
        trailer, an unreadable timestamp, or a station kind this store has
        no table for. Never raises on cell content."""
        raw_code = _cell(row, self.layout.station_i).strip().lower()
        if not raw_code:
            return None  # a wholly blank trailer row some exports append
        self.result.n_rows += 1

        kind = station_kind(raw_code)
        if kind not in ("wq", "met"):
            _bump(self.result.skipped_stations, raw_code)
            return None

        sp = self._station(raw_code, kind)
        sp.n_rows += 1

        ts_raw = _cell(row, self.layout.ts_i).strip()
        try:
            ts = datetime.strptime(ts_raw, _TS_FORMAT).replace(tzinfo=self._tz(raw_code))
        except ValueError:
            sp.n_bad_timestamp += 1
            return None
        sp._note_ts(ts)

        if kind == "wq":
            pairs, absent = self.layout.wq_pairs, self.layout.wq_absent
            accepted = ACCEPTED_FLAGS
        else:
            pairs, absent = self.layout.met_pairs, self.layout.met_absent
            accepted = MET_ACCEPTED_FLAGS

        resolved: dict[str, float | None] = dict.fromkeys(absent)
        for col, obs_field, value_i, flag_i in pairs:
            value, outcome, codes = _resolve(
                _cell(row, value_i), _cell(row, flag_i), accepted
            )
            resolved[obs_field] = value
            if outcome == "ok":
                _bump(sp.n_admitted, col)
            elif outcome == "rejected":
                _bump(sp.n_rejected_by_flag, col)
            elif outcome == "unparseable_value":
                _bump(sp.n_value_unparseable, col)
            elif outcome == "unparseable_flag":
                _bump(sp.n_flag_unparseable, col)
            for code in codes:
                _bump(sp.qualifier_codes, code)
                if outcome == "ok":
                    _bump(sp.qualifier_codes_admitted, code)
                if code not in QAQC_CODE_MEANINGS:
                    _bump(sp.unknown_qualifier_codes, code)

        if not any(v is not None for v in resolved.values()):
            sp.n_empty += 1

        obs: Observation | MetObservation
        if kind == "wq":
            obs = Observation(ts=ts, eh_mv=None, **resolved)
        else:
            obs = MetObservation(ts=ts, **resolved)
        if self.collect:
            sp.observations.append(obs)
        return kind, sp.canonical, obs


def _open_rows(fh: Iterable[str]) -> Iterator[list[str]]:
    return csv.reader(fh)


def _read(
    fh: Iterable[str],
    stations: Mapping[str, StationMeta] | None = None,
    collect: bool = False,
) -> tuple[_Reader, Iterator[tuple[str, str, Observation | MetObservation]]]:
    """(reader, generator of parsed rows). Raises `ValueError` on a header
    that is not a CDMO export, BEFORE any row is read."""
    rows = _open_rows(fh)
    header = next(rows, None)
    if not header:
        raise ValueError("empty CDMO export -- no header row")
    reader = _Reader(_Layout(header), stations=stations, collect=collect)

    def parsed() -> Iterator[tuple[str, str, Observation | MetObservation]]:
        for row in rows:
            out = reader.row(row)
            if out is not None:
                yield out

    return reader, parsed()


def parse_cdmo_csv(
    text: str, stations: Mapping[str, StationMeta] | None = None
) -> ParseResult:
    """Parse one CDMO export held in memory, returning its WATER-QUALITY view.

    For small files and tests. `import_file` is the streaming equivalent and
    is what a real export goes through -- the real one is 494 MB, which this
    would need roughly a gigabyte to hold before parsing a single row.

    `result.stations` holds only water-quality stations; meteorological rows
    in the same file are parsed too (see `parse_cdmo_met_csv` for that view)
    but never appear here, so a MET station can never manufacture an
    all-empty water-quality row.

    Raises `ValueError` if the header is missing `StationCode` or
    `DateTimeStamp` entirely. Nothing else here raises: a bad cell is
    dropped and counted (see `StationParse`), never fatal to the rest.
    """
    reader, rows = _read(io.StringIO(text), stations=stations, collect=True)
    for _ in rows:
        pass
    return reader.result


def parse_cdmo_met_csv(
    text: str, stations: Mapping[str, StationMeta] | None = None
) -> ParseResult:
    """`parse_cdmo_csv`'s METEOROLOGICAL view of the same file.

    Returns a `ParseResult` whose `.stations` are the MET stations, so a
    caller that only wants weather reads it the same way it reads the WQ
    one. Same parse, same pass, different projection.
    """
    result = parse_cdmo_csv(text, stations=stations)
    return ParseResult(
        stations=result.met_stations,
        met_stations={},
        unknown_columns=result.unknown_columns,
        skipped_stations=result.skipped_stations,
        n_rows=result.n_rows,
    )


@dataclass
class StationImport:
    """What one station contributed to the store, from one file."""

    raw_code: str
    canonical: str
    kind: str  # "wq" | "met"
    n_rows: int
    n_parsed: int
    n_new: int
    n_bad_timestamp: int
    n_empty: int
    n_admitted: dict[str, int]
    n_rejected_by_flag: dict[str, int]
    n_value_unparseable: dict[str, int]
    n_flag_unparseable: dict[str, int]
    qualifier_codes: dict[str, int]
    qualifier_codes_admitted: dict[str, int]
    unknown_qualifier_codes: dict[str, int]
    span: tuple[datetime, datetime] | None
    meta: StationMeta | None


@dataclass
class ImportReport:
    """One FILE's import. The real export is a single file holding both
    parameter families, so one report carries both lists."""

    source: str  # display label: a file path, or "archive.zip!member.csv"
    stations: list[StationImport] = field(default_factory=list)  # water quality
    met_stations: list[StationImport] = field(default_factory=list)  # meteorological
    unknown_columns: list[str] = field(default_factory=list)
    skipped_stations: dict[str, int] = field(default_factory=dict)
    n_rows: int = 0
    station_metadata_source: Path | None = None


def _station_import(
    sp: StationParse,
    n_new: int,
    stations: Mapping[str, StationMeta],
) -> StationImport:
    return StationImport(
        raw_code=sp.raw_code,
        canonical=sp.canonical,
        kind=sp.kind,
        n_rows=sp.n_rows,
        n_parsed=sp.n_rows - sp.n_bad_timestamp,
        n_new=n_new,
        n_bad_timestamp=sp.n_bad_timestamp,
        n_empty=sp.n_empty,
        n_admitted=sp.n_admitted,
        n_rejected_by_flag=sp.n_rejected_by_flag,
        n_value_unparseable=sp.n_value_unparseable,
        n_flag_unparseable=sp.n_flag_unparseable,
        qualifier_codes=sp.qualifier_codes,
        qualifier_codes_admitted=sp.qualifier_codes_admitted,
        unknown_qualifier_codes=sp.unknown_qualifier_codes,
        span=sp.span,
        meta=stations.get(sp.raw_code),
    )


def _import_stream(
    fh: Iterable[str],
    store: NdbcStore,
    source: str,
    stations: Mapping[str, StationMeta] | None = None,
    metadata_source: Path | None = None,
) -> ImportReport:
    """Stream one CDMO export into the store, atomically.

    The header is validated BEFORE the transaction opens, so a file that is
    not a CDMO export raises without SQLite ever being asked to start one.
    Everything after that -- every station, both tables, and the two
    provenance rows -- lives inside ONE `bulk_writer` transaction: the whole
    file lands or none of it does, at 2.5 million rows exactly as at three.
    """
    stations = stations or {}
    reader, rows = _read(fh, stations=stations, collect=False)

    with store.bulk_writer(batch_rows=BATCH_ROWS) as writer:
        for kind, canonical, obs in rows:
            if kind == "wq":
                writer.add(canonical, obs)
            else:
                writer.add_met(canonical, obs)
        writer.flush()

        result = reader.result
        report = ImportReport(
            source=source,
            unknown_columns=result.unknown_columns,
            skipped_stations=dict(result.skipped_stations),
            n_rows=result.n_rows,
            station_metadata_source=metadata_source,
        )
        for raw_code in sorted(result.stations):
            sp = result.stations[raw_code]
            report.stations.append(
                _station_import(sp, writer.n_new(sp.canonical), stations)
            )
        for raw_code in sorted(result.met_stations):
            sp = result.met_stations[raw_code]
            report.met_stations.append(
                _station_import(sp, writer.n_new_met(sp.canonical), stations)
            )

        # One provenance row per parameter family that actually got data --
        # inside this transaction, so a rollback takes it too. See
        # `ndbc.BulkWriter.record_provenance`.
        for source_label, imports in (
            (SOURCE_CDMO_WQ, report.stations),
            (SOURCE_CDMO_MET, report.met_stations),
        ):
            if not imports:
                continue
            spans = [s.span for s in imports if s.span is not None]
            span = (
                (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else None
            )
            writer.record_provenance(
                source_label,
                [s.canonical for s in imports],
                span,
                sum(s.n_new for s in imports),
            )
    return report


def import_file(
    path: Path,
    store: NdbcStore,
    stations: Mapping[str, StationMeta] | None = None,
    metadata_source: Path | None = None,
) -> ImportReport:
    """Parse and store one CDMO CSV file -- water quality and meteorological
    together, each row routed by its own station code.

    `stations` is the parsed `sampling_stations.csv`; when omitted, one
    beside `path` is used if present (`find_station_metadata`). It supplies
    positions and per-station GMT offsets; without it the import still runs,
    on the documented UTC-5 fallback and with no positions.

    `utf-8-sig` strips a leading BOM if Excel-exported without corrupting
    plain UTF-8/ASCII, which has no BOM to strip. `newline=""` is required
    by `csv`: the real export is CRLF-terminated.
    """
    if stations is None:
        metadata_source = find_station_metadata(path)
        if metadata_source is not None and metadata_source != path:
            stations = load_station_metadata(metadata_source)
        else:
            metadata_source = None
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return _import_stream(fh, store, str(path), stations, metadata_source)


def import_path(path: Path, store: NdbcStore) -> list[ImportReport]:
    """Import a CDMO export at `path`: a single CSV, a directory of CSVs, or
    a `.zip` archive (CDMO's "zip downloads" feature, read directly, matching
    what `SWMPr::import_local` accepts).

    A directory may hold the observation export and `sampling_stations.csv`
    side by side -- which is exactly what CDMO's query interface hands a
    user. The station table is recognised by its header and used as METADATA,
    not fed to the observation parser (where its missing `DateTimeStamp`
    would abort the whole import).

    IF THE FILE ISN'T HERE YET: raises `FileNotFoundError` naming the exact
    path expected, before touching the store at all.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no CDMO export at {path} -- place the downloaded file, the "
            "unzipped directory of .csv files, or the .zip itself there, "
            "or pass a different path"
        )
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(f"{path} is a directory but contains no .csv files")
        metadata_source = find_station_metadata(path)
        stations = (
            load_station_metadata(metadata_source) if metadata_source is not None else None
        )
        data_files = [f for f in files if f != metadata_source]
        if not data_files:
            raise FileNotFoundError(
                f"{path} contains only a station table ({metadata_source}) and no "
                "observation export"
            )
        return [
            import_file(f, store, stations=stations, metadata_source=metadata_source)
            for f in data_files
        ]
    if path.suffix.lower() == ".zip":
        return _import_zip(path, store)
    return [import_file(path, store)]


def _import_zip(path: Path, store: NdbcStore) -> list[ImportReport]:
    reports: list[ImportReport] = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not names:
            raise FileNotFoundError(f"{path} contains no .csv files")

        stations: Mapping[str, StationMeta] | None = None
        metadata_name: str | None = None
        for name in names:
            with zf.open(name) as raw:
                header = next(
                    csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")), []
                )
            if looks_like_station_metadata(header):
                metadata_name = name
                break
        if metadata_name is not None:
            with zf.open(metadata_name) as raw:
                # `load_station_metadata` takes a path; the archive member is
                # small (367 rows) so reading it in full is not the 494 MB
                # problem this module streams to avoid.
                text = raw.read().decode("utf-8", errors="replace")
            stations = read_station_metadata(io.StringIO(text), source=f"{path}!{metadata_name}")

        for name in names:
            if name == metadata_name:
                continue
            with zf.open(name) as raw:
                fh = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reports.append(
                    _import_stream(fh, store, f"{path}!{name}", stations)
                )
    if not reports:
        raise FileNotFoundError(
            f"{path} contains only a station table and no observation export"
        )
    return reports
