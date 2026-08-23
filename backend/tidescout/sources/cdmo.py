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
"""

import csv
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tidescout.sources.ndbc import NdbcStore, Observation

__all__ = [
    "ACCEPTED_FLAGS",
    "CDMO_TZ",
    "FLAG_MEANINGS",
    "NIW_STATION_COORDS_LONLAT",
    "STATION_ALIASES",
    "ImportReport",
    "ParseResult",
    "StationImport",
    "StationParse",
    "canonical_station",
    "import_file",
    "import_path",
    "parse_cdmo_csv",
]

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


def _resolve(value_cell: str, flag_cell: str) -> tuple[float | None, str]:
    """One (value, flag) column pair -> the value if QAQC accepts it, else
    `None` -- plus an outcome tag: "ok", "blank", "rejected",
    "unparseable_value", "unparseable_flag". Order matters: an unparseable
    flag is reported as such even if the value itself parsed fine, since a
    flag CDMO's own documented syntax can't be read is the more serious
    problem to surface.
    """
    value, value_ok = _parse_value(value_cell)
    flag, flag_ok = _parse_flag(flag_cell)
    if not flag_ok:
        return None, "unparseable_flag"
    if not value_ok:
        return None, "unparseable_value"
    if value is None:
        return None, "blank"
    if flag is None or flag not in ACCEPTED_FLAGS:
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
    """
    stations = []
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
    return ImportReport(source=source, stations=stations, unknown_columns=result.unknown_columns)


def import_file(path: Path, store: NdbcStore) -> ImportReport:
    """Parse and store one CDMO CSV file.

    `utf-8-sig` strips a leading BOM if Excel-exported (the real zip_ex
    file has none, but a user's own AQS export might) without corrupting
    plain UTF-8/ASCII, which has no BOM to strip.
    """
    text = path.read_text(encoding="utf-8-sig")
    result = parse_cdmo_csv(text)
    return _apply(result, store, str(path))


def import_path(path: Path, store: NdbcStore) -> list[ImportReport]:
    """Import a CDMO export at `path`: a single CSV, a directory of CSVs
    (CDMO's "zip downloads" feature, once unzipped), or a `.zip` archive
    (the same feature, not yet unzipped -- read directly via `zipfile`,
    matching what `SWMPr::import_local` accepts).

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


def _import_zip(path: Path, store: NdbcStore) -> list[ImportReport]:
    reports = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not names:
            raise FileNotFoundError(f"{path} contains no .csv files")
        for name in names:
            text = zf.read(name).decode("utf-8-sig", errors="replace")
            result = parse_cdmo_csv(text)
            reports.append(_apply(result, store, f"{path}!{name}"))
    return reports
