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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tidescout.sources.ndbc import MetObservation, NdbcStore, Observation

__all__ = [
    "ACCEPTED_FLAGS",
    "CDMO_TZ",
    "FLAG_MEANINGS",
    "MET_ACCEPTED_FLAGS",
    "MET_FLAG_MEANINGS",
    "NIW_MET_STATION_COORDS_LONLAT",
    "NIW_STATION_COORDS_LONLAT",
    "SOURCE_CDMO_MET",
    "SOURCE_CDMO_WQ",
    "STATION_ALIASES",
    "ImportReport",
    "MetImportReport",
    "MetParseResult",
    "MetStationImport",
    "MetStationParse",
    "ParseResult",
    "StationImport",
    "StationParse",
    "canonical_station",
    "import_file",
    "import_path",
    "parse_cdmo_csv",
    "parse_cdmo_met_csv",
]

# Provenance `source` labels for the two CDMO import routes -- see
# `ndbc.py`'s "PROVENANCE AND CITATION" and `SOURCE_NDBC_REALTIME2` there.
SOURCE_CDMO_WQ = "cdmo:water_quality"
SOURCE_CDMO_MET = "cdmo:meteorological"

# Fixed UTC-5, year-round -- built directly rather than via a DST-aware zone.
# See the module docstring's "TIMESTAMP CONVENTION" section for the two
# independent primary sources this was established from.
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

# niwwswq is WYSS1 -- see "STATION-CODE ALIASING" above.
STATION_ALIASES: dict[str, str] = {"niwwswq": "WYSS1"}


def canonical_station(cdmo_code: str) -> str:
    """The `NdbcStore` key a CDMO station code should be written under."""
    code = cdmo_code.strip().lower()
    return STATION_ALIASES.get(code, code.upper())


def _dms(deg: float, mins: float, secs: float) -> float:
    return deg + mins / 60.0 + secs / 3600.0


# (lon, lat) WGS84, decimal-converted from the DMS coordinates published in
# the reserve's own metadata PDF (Sections 5 and 10 -- see module
# docstring). Keyed by the CANONICAL station id (post `canonical_station`),
# the same key `NdbcStore` uses, so a caller can go from a store row
# straight to a position with no second lookup. All six of NIW's current
# stations -- see "WHAT THE DISPATCH GOT WRONG" above for why this is six,
# not four.
NIW_STATION_COORDS_LONLAT: dict[str, tuple[float, float]] = {
    "NIWCBWQ": (-_dms(79, 11, 34.62), _dms(33, 20, 2.05)),  # Clambank Landing
    "NIWDCWQ": (-_dms(79, 10, 2.81), _dms(33, 21, 36.49)),  # Debidue Creek
    "NIWOLWQ": (-_dms(79, 11, 19.97), _dms(33, 20, 57.70)),  # Oyster Landing
    "NIWTAWQ": (-_dms(79, 15, 21.75), _dms(33, 17, 57.03)),  # Thousand Acre
    "WYSS1": (-_dms(79, 17, 19.57), _dms(33, 18, 33.88)),  # Winyah Bay surface (niwwswq)
    "NIWWBWQ": (-_dms(79, 17, 19.58), _dms(33, 18, 33.94)),  # Winyah Bay bottom
}

# One entry: NIW runs exactly one weather station, system-wide, at Oyster
# Landing -- see "MET FILE SUPPORT" above. Independently sourced from the
# MET metadata PDF's own DMS coordinates, not copied from NIW_STATION_
# COORDS_LONLAT["NIWOLWQ"] -- the two differ by a few feet (same pier,
# different sensor mount), which is worth keeping visible rather than
# collapsing onto one shared position.
NIW_MET_STATION_COORDS_LONLAT: dict[str, tuple[float, float]] = {
    "NIWOLMET": (-_dms(79, 11, 20.03), _dms(33, 20, 57.85)),  # Oyster Landing (niwolmet)
}

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

# Columns CDMO documents but that have no destination field in the shared
# NDBC/CDMO schema (Task 8 never needed vertical datum or barometric
# corrections) -- deliberately dropped, and excluded from
# `ParseResult.unknown_columns` so dropping them doesn't read as "could not
# parse."
_UNMAPPED_PARAM_COLUMNS = frozenset({"cdepth", "level", "clevel"})

# CDMO MET column name (lowercased) -> `MetObservation` field, per SWMPr's
# `param_names.R` MET list -- see module docstring's "MET FILE SUPPORT".
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

# `cumprcp` is documented by SWMPr but no longer exported by CDMO (met
# metadata PDF remark 13.d) -- deliberately dropped, same treatment as
# `_UNMAPPED_PARAM_COLUMNS` above. See module docstring's "COLUMNS
# DELIBERATELY DROPPED".
_MET_UNMAPPED_PARAM_COLUMNS = frozenset({"cumprcp"})

# Diagnostic columns used only to tell a MET file apart from a WQ file by
# its header (see `_looks_like_met_header`) -- never both present in a real
# export, since "ATemp"/"WSpd"/"BP" are MET-only and CDMO issues one file
# type per parameter family.
_MET_DIAGNOSTIC_COLUMNS = frozenset({"atemp", "wspd", "bp", "rh"})

# Non-parameter columns every real export carries. `""` covers the trailing
# empty-named column produced by CDMO's own trailing comma (confirmed on
# the real zip_ex file).
_METADATA_COLUMNS = frozenset(
    {"stationcode", "isswmp", "datetimestamp", "historical", "provisionalplus", "f_record", ""}
)

_REQUIRED_COLUMNS = frozenset({"stationcode", "datetimestamp"})

_TS_FORMAT = "%m/%d/%Y %H:%M"

# `<flag>` or `<flag CODE>` -- the numeric flag first, an optional QAQC code
# after it in the same bracket. See module docstring point 2: no real file
# seen so far actually carries a code, so the second group is exercised
# only by the synthetic fixture.
_FLAG_RE = re.compile(r"<\s*(-?\d+)\s*([^>]*)>")


def _parse_flag(cell: str) -> tuple[int | None, bool]:
    """(flag, was_parseable). `flag` is `None` for a blank cell (nothing
    flagged) or an unparseable one; `was_parseable` tells the two apart so
    the caller can report a real format disagreement rather than treat it
    as ordinary missingness."""
    cell = cell.strip()
    if not cell:
        return None, True
    m = _FLAG_RE.match(cell)
    if not m:
        return None, False
    return int(m.group(1)), True


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
) -> tuple[float | None, str]:
    """One (value, flag) column pair -> the value if QAQC accepts it, else
    `None` -- plus an outcome tag: "ok", "blank", "rejected",
    "unparseable_value", "unparseable_flag". Order matters: an unparseable
    flag is reported as such even if the value itself parsed fine, since a
    flag CDMO's own documented syntax can't be read is the more serious
    problem to surface.

    `accepted_flags` defaults to WQ's set; `parse_cdmo_met_csv` passes
    `MET_ACCEPTED_FLAGS` instead -- see module docstring's "MET-SPECIFIC
    QAQC". The cell-level parsing itself (`_parse_value`/`_parse_flag`) is
    identical for both formats; only which flags count as "admitted"
    differs.
    """
    value, value_ok = _parse_value(value_cell)
    flag, flag_ok = _parse_flag(flag_cell)
    if not flag_ok:
        return None, "unparseable_flag"
    if not value_ok:
        return None, "unparseable_value"
    if value is None:
        return None, "blank"
    if flag is None or flag not in accepted_flags:
        return None, "rejected"
    return value, "ok"


@dataclass
class StationParse:
    """Everything one CDMO station's rows, within one file, parsed to --
    and every way some of them could not be used, counted rather than
    silently dropped."""

    raw_code: str
    observations: list[Observation] = field(default_factory=list)
    n_rows: int = 0
    n_bad_timestamp: int = 0
    n_rejected_by_flag: dict[str, int] = field(default_factory=dict)
    n_value_unparseable: dict[str, int] = field(default_factory=dict)
    n_flag_unparseable: dict[str, int] = field(default_factory=dict)


@dataclass
class ParseResult:
    stations: dict[str, StationParse]  # raw lowercase CDMO code -> parse
    unknown_columns: list[str]  # header columns this parser does not recognise at all


def parse_cdmo_csv(text: str) -> ParseResult:
    """Parse one CDMO CSV export -- one or more stations (see module
    docstring: "Custom Query" exports combine stations; "zip download"
    exports do not, and this handles either the same way, grouping by the
    row's own `StationCode`, never a filename).

    Raises `ValueError` if the header is missing `StationCode` or
    `DateTimeStamp` entirely -- this does not look like a CDMO export at
    all, and nothing downstream (dedupe key, timestamp) can proceed without
    them. Nothing else here raises: a bad cell is dropped and counted (see
    `StationParse`), never fatal to the rest of the file.
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("empty CDMO export -- no header row")

    fieldnames_lower = {name.strip().lower(): name for name in reader.fieldnames if name}
    missing = _REQUIRED_COLUMNS - set(fieldnames_lower)
    if missing:
        raise ValueError(
            f"missing required column(s) {sorted(missing)} -- this does not look like a "
            f"CDMO export; header was {reader.fieldnames}"
        )

    unknown_columns = sorted(
        orig
        for key, orig in fieldnames_lower.items()
        if not key.startswith("f_")
        and key not in _METADATA_COLUMNS
        and key not in _COLUMN_MAP
        and key not in _UNMAPPED_PARAM_COLUMNS
    )
    flag_col_for = {col: fieldnames_lower.get(f"f_{col}") for col in _COLUMN_MAP}
    stationcode_col = fieldnames_lower["stationcode"]
    datetimestamp_col = fieldnames_lower["datetimestamp"]

    stations: dict[str, StationParse] = {}
    for row in reader:
        raw_code = (row.get(stationcode_col) or "").strip().lower()
        if not raw_code:
            continue  # a wholly blank trailer row some exports append
        sp = stations.setdefault(raw_code, StationParse(raw_code=raw_code))
        sp.n_rows += 1

        ts_raw = (row.get(datetimestamp_col) or "").strip()
        try:
            ts = datetime.strptime(ts_raw, _TS_FORMAT).replace(tzinfo=CDMO_TZ)
        except ValueError:
            sp.n_bad_timestamp += 1
            continue

        resolved: dict[str, float | None] = {}
        for col, obs_field in _COLUMN_MAP.items():
            value_col = fieldnames_lower.get(col)
            if value_col is None:
                resolved[obs_field] = None
                continue
            flag_col = flag_col_for[col]
            value, outcome = _resolve(
                row.get(value_col, ""), row.get(flag_col, "") if flag_col else ""
            )
            resolved[obs_field] = value
            if outcome == "rejected":
                sp.n_rejected_by_flag[col] = sp.n_rejected_by_flag.get(col, 0) + 1
            elif outcome == "unparseable_value":
                sp.n_value_unparseable[col] = sp.n_value_unparseable.get(col, 0) + 1
            elif outcome == "unparseable_flag":
                sp.n_flag_unparseable[col] = sp.n_flag_unparseable.get(col, 0) + 1

        sp.observations.append(Observation(ts=ts, eh_mv=None, **resolved))

    return ParseResult(stations=stations, unknown_columns=unknown_columns)


def _looks_like_met_header(fieldnames_lower: dict[str, str]) -> bool:
    """True if this header belongs to a MET export, not a WQ one. CDMO
    issues one file per parameter family, so a real file never carries
    both a MET-only column (e.g. `ATemp`) and would ambiguously also be a
    WQ file -- see module docstring's "MET FILE SUPPORT". Used by
    `import_file`/`_import_zip` to route each file to the right parser
    without relying on a filename convention, matching `parse_cdmo_csv`'s
    own "group by the row's own StationCode, never a filename" philosophy.
    """
    return bool(_MET_DIAGNOSTIC_COLUMNS & set(fieldnames_lower))


@dataclass
class MetStationParse:
    """`StationParse`'s exact shape, for one CDMO MET station's rows."""

    raw_code: str
    observations: list[MetObservation] = field(default_factory=list)
    n_rows: int = 0
    n_bad_timestamp: int = 0
    n_rejected_by_flag: dict[str, int] = field(default_factory=dict)
    n_value_unparseable: dict[str, int] = field(default_factory=dict)
    n_flag_unparseable: dict[str, int] = field(default_factory=dict)


@dataclass
class MetParseResult:
    stations: dict[str, MetStationParse]
    unknown_columns: list[str]


def parse_cdmo_met_csv(text: str) -> MetParseResult:
    """`parse_cdmo_csv`'s exact contract, against the MET column set
    (`_MET_COLUMN_MAP`) and MET's narrower accepted-flags set
    (`MET_ACCEPTED_FLAGS`) instead of WQ's. Same required columns
    (`StationCode`, `DateTimeStamp`), same non-fatal-per-cell philosophy,
    same station-grouping-by-row rule. See module docstring's "MET FILE
    SUPPORT" and "MET-SPECIFIC QAQC".
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("empty CDMO export -- no header row")

    fieldnames_lower = {name.strip().lower(): name for name in reader.fieldnames if name}
    missing = _REQUIRED_COLUMNS - set(fieldnames_lower)
    if missing:
        raise ValueError(
            f"missing required column(s) {sorted(missing)} -- this does not look like a "
            f"CDMO export; header was {reader.fieldnames}"
        )

    unknown_columns = sorted(
        orig
        for key, orig in fieldnames_lower.items()
        if not key.startswith("f_")
        and key not in _METADATA_COLUMNS
        and key not in _MET_COLUMN_MAP
        and key not in _MET_UNMAPPED_PARAM_COLUMNS
    )
    flag_col_for = {col: fieldnames_lower.get(f"f_{col}") for col in _MET_COLUMN_MAP}
    stationcode_col = fieldnames_lower["stationcode"]
    datetimestamp_col = fieldnames_lower["datetimestamp"]

    stations: dict[str, MetStationParse] = {}
    for row in reader:
        raw_code = (row.get(stationcode_col) or "").strip().lower()
        if not raw_code:
            continue
        sp = stations.setdefault(raw_code, MetStationParse(raw_code=raw_code))
        sp.n_rows += 1

        ts_raw = (row.get(datetimestamp_col) or "").strip()
        try:
            ts = datetime.strptime(ts_raw, _TS_FORMAT).replace(tzinfo=CDMO_TZ)
        except ValueError:
            sp.n_bad_timestamp += 1
            continue

        resolved: dict[str, float | None] = {}
        for col, obs_field in _MET_COLUMN_MAP.items():
            value_col = fieldnames_lower.get(col)
            if value_col is None:
                resolved[obs_field] = None
                continue
            flag_col = flag_col_for[col]
            value, outcome = _resolve(
                row.get(value_col, ""),
                row.get(flag_col, "") if flag_col else "",
                MET_ACCEPTED_FLAGS,
            )
            resolved[obs_field] = value
            if outcome == "rejected":
                sp.n_rejected_by_flag[col] = sp.n_rejected_by_flag.get(col, 0) + 1
            elif outcome == "unparseable_value":
                sp.n_value_unparseable[col] = sp.n_value_unparseable.get(col, 0) + 1
            elif outcome == "unparseable_flag":
                sp.n_flag_unparseable[col] = sp.n_flag_unparseable.get(col, 0) + 1

        sp.observations.append(MetObservation(ts=ts, **resolved))

    return MetParseResult(stations=stations, unknown_columns=unknown_columns)


@dataclass
class StationImport:
    raw_code: str
    canonical: str
    n_parsed: int
    n_new: int
    n_bad_timestamp: int
    n_rejected_by_flag: dict[str, int]
    n_value_unparseable: dict[str, int]
    n_flag_unparseable: dict[str, int]


@dataclass
class ImportReport:
    source: str  # display label: a file path, or "archive.zip!member.csv"
    stations: list[StationImport]
    unknown_columns: list[str]


def _apply(result: ParseResult, store: NdbcStore, source: str) -> ImportReport:
    """Write every parsed station's rows through `NdbcStore.append` --
    Task 8's whole-batch-atomic, `(station, ts)`-deduplicated write,
    reused exactly, not reimplemented. Called only after `parse_cdmo_csv`
    has already finished parsing the WHOLE file in memory (see
    `import_file`), so a structural parse failure never reaches here and
    never touches the store.

    Also records ONE provenance row for this whole file (see `ndbc.py`'s
    "PROVENANCE AND CITATION") -- after every station's `append` has
    already succeeded, covering every station this file touched and the
    combined span of what it contributed. Nothing is recorded if the file
    parsed to zero stations (header-only / all-blank-trailer file): there
    is nothing real to attribute an access to.
    """
    stations = []
    all_ts: list[datetime] = []
    touched: list[str] = []
    for raw_code in sorted(result.stations):
        sp = result.stations[raw_code]
        canonical = canonical_station(raw_code)
        n_new = store.append(canonical, sp.observations)
        stations.append(
            StationImport(
                raw_code=raw_code,
                canonical=canonical,
                n_parsed=len(sp.observations),
                n_new=n_new,
                n_bad_timestamp=sp.n_bad_timestamp,
                n_rejected_by_flag=sp.n_rejected_by_flag,
                n_value_unparseable=sp.n_value_unparseable,
                n_flag_unparseable=sp.n_flag_unparseable,
            )
        )
        touched.append(canonical)
        all_ts.extend(o.ts for o in sp.observations)
    if touched:
        span = (min(all_ts), max(all_ts)) if all_ts else None
        store.record_provenance(SOURCE_CDMO_WQ, touched, span, sum(s.n_new for s in stations))
    return ImportReport(source=source, stations=stations, unknown_columns=result.unknown_columns)


@dataclass
class MetStationImport:
    raw_code: str
    canonical: str
    n_parsed: int
    n_new: int
    n_bad_timestamp: int
    n_rejected_by_flag: dict[str, int]
    n_value_unparseable: dict[str, int]
    n_flag_unparseable: dict[str, int]


@dataclass
class MetImportReport:
    source: str
    stations: list[MetStationImport]
    unknown_columns: list[str]


def _apply_met(result: MetParseResult, store: NdbcStore, source: str) -> MetImportReport:
    """`_apply`'s exact contract, against `NdbcStore.append_met` and
    `SOURCE_CDMO_MET` instead."""
    stations = []
    all_ts: list[datetime] = []
    touched: list[str] = []
    for raw_code in sorted(result.stations):
        sp = result.stations[raw_code]
        canonical = canonical_station(raw_code)
        n_new = store.append_met(canonical, sp.observations)
        stations.append(
            MetStationImport(
                raw_code=raw_code,
                canonical=canonical,
                n_parsed=len(sp.observations),
                n_new=n_new,
                n_bad_timestamp=sp.n_bad_timestamp,
                n_rejected_by_flag=sp.n_rejected_by_flag,
                n_value_unparseable=sp.n_value_unparseable,
                n_flag_unparseable=sp.n_flag_unparseable,
            )
        )
        touched.append(canonical)
        all_ts.extend(o.ts for o in sp.observations)
    if touched:
        span = (min(all_ts), max(all_ts)) if all_ts else None
        store.record_provenance(SOURCE_CDMO_MET, touched, span, sum(s.n_new for s in stations))
    return MetImportReport(
        source=source, stations=stations, unknown_columns=result.unknown_columns
    )


def _detect_format(text: str) -> str:
    """"met" or "wq", by peeking the header only -- see
    `_looks_like_met_header`. A malformed/headerless file reads as "wq"
    (the default) and is left for `parse_cdmo_csv` to raise its own clear
    `ValueError` on, rather than this function guessing or raising first.
    """
    reader = csv.DictReader(io.StringIO(text))
    fieldnames_lower = {name.strip().lower(): name for name in (reader.fieldnames or []) if name}
    return "met" if _looks_like_met_header(fieldnames_lower) else "wq"


def import_file(path: Path, store: NdbcStore) -> ImportReport | MetImportReport:
    """Parse and store one CDMO CSV file -- water-quality or meteorological,
    detected from the header (`_detect_format`), never from the filename.

    `utf-8-sig` strips a leading BOM if Excel-exported (the real zip_ex
    file has none, but a user's own AQS export might) without corrupting
    plain UTF-8/ASCII, which has no BOM to strip.
    """
    text = path.read_text(encoding="utf-8-sig")
    if _detect_format(text) == "met":
        return _apply_met(parse_cdmo_met_csv(text), store, str(path))
    return _apply(parse_cdmo_csv(text), store, str(path))


def import_path(path: Path, store: NdbcStore) -> list[ImportReport | MetImportReport]:
    """Import a CDMO export at `path`: a single CSV, a directory of CSVs
    (CDMO's "zip downloads" feature, once unzipped), or a `.zip` archive
    (the same feature, not yet unzipped -- read directly via `zipfile`,
    matching what `SWMPr::import_local` accepts).

    A directory or zip may mix water-quality and meteorological files
    freely -- this task's dispatch asked for exactly that ("the CDMO export
    will contain meteorological files alongside water quality files"), and
    each file is routed to the right store table on its own, by its own
    header.

    IF THE FILE ISN'T HERE YET: raises `FileNotFoundError` naming the exact
    path expected, before touching the store at all. The CLI (`tidescout
    salinity import-cdmo`) catches this and prints it plainly -- see that
    command's docstring for where to put the download and what to run.
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
        return [import_file(f, store) for f in files]
    if path.suffix.lower() == ".zip":
        return _import_zip(path, store)
    return [import_file(path, store)]


def _import_zip(path: Path, store: NdbcStore) -> list[ImportReport | MetImportReport]:
    reports: list[ImportReport | MetImportReport] = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not names:
            raise FileNotFoundError(f"{path} contains no .csv files")
        for name in names:
            text = zf.read(name).decode("utf-8-sig", errors="replace")
            if _detect_format(text) == "met":
                reports.append(_apply_met(parse_cdmo_met_csv(text), store, f"{path}!{name}"))
            else:
                reports.append(_apply(parse_cdmo_csv(text), store, f"{path}!{name}"))
    return reports
