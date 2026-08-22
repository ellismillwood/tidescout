"""CO-OPS physical-oceanography salinity: the model's ocean end-member.

Springmaid Pier (8661070), ~42.8 km NE of Winyah Bay on the open coast, is the
closest CO-OPS station tagged with a "Physical Oceanography" product page.
It measures shelf water rather than anything inside the bay, which is exactly
the role it plays here: S_ocean, the boundary value the intrusion profile
decays away from.

CORRECTION to the "sole physocean station within 100 km" premise this station
was originally selected under: live-checked 2026-08-22 against the CO-OPS
mdapi station list, Charleston (8665530, 92.5 km away) carries the same
"physocean" tag and is also within 100 km -- Springmaid Pier is the CLOSEST,
not the only, one. Immaterial to the outcome below (neither reports
salinity), but the original justification was inaccurate.

FINDING (2026-08-22, live probe, not a fixture): despite the "physocean" tag,
Springmaid Pier does NOT serve the salinity product. Its sensors.json lists
only Wind / Air Temperature / Water Temperature / Barometric Pressure /
Tsunami WL / Microwave WL -- no Conductivity or Salinity sensor -- and
`product=salinity` returns `{"error": "No data was found..."}` for every
window tried: 2026-08-16, a recent 2-day window, "latest", and 2015-01-01
through 2015-01-07 (no historical data either). "physocean" is a broad tag
(239 CO-OPS stations nationwide carry it, evidently bundling wind/temp/
pressure into one page) and is NOT evidence a station reports salinity
specifically. For comparison, station 8637689 (Yorktown, VA), which DOES list
a Conductivity sensor, returns real salinity data for the same query shape --
confirming this module's request format is correct and the gap is Springmaid
Pier's data, not the client.

None of the other 4 CO-OPS stations within 250 km of Winyah Bay (Charleston
8665530, Wilmington 8658120, Wrightsville Beach 8658163, Fort Pulaski 8670870)
carry a Conductivity sensor either, so there is currently no substitute
station reachable via CO-OPS. In practice `fetch_ocean_salinity` therefore
RAISES `SourceUnavailable` against the configured station today -- CO-OPS's
"No data was found" response is a `{"error": ...}` payload, which is a real
failure under this module's contract below, not a "responded with nothing
usable" case. That exception is meant to propagate to
`sources.dayloader.load_day`'s `attempt()` helper, the app's one place that
catches source failures and records them by name in `missing`, so Springmaid
Pier shows up there rather than silently supplying
`SalinityConfig.ocean_ppt`'s static default with no trace. Recorded here so
this is not mistaken for a live feed.

Reuses noaa.py's CO-OPS client (`_get_json`) rather than re-implementing an
HTTP layer for the same datagetter endpoint noaa.py already owns.
"""

from datetime import date

from tidescout.sources.cache import Cache
from tidescout.sources.noaa import OBS_TTL, _get_json

__all__ = ["fetch_ocean_salinity"]

# Open shelf water off South Carolina runs ~30-36 ppt. Anything outside this is
# a stuck or miscalibrated sensor, and S_ocean is the single most influential
# constant in the model -- every cell's salinity scales linearly with it.
PLAUSIBLE_PPT = (25.0, 40.0)


def fetch_ocean_salinity(station: str, day: date, cache: Cache) -> float | None:
    """Mean of `day`'s valid salinity readings in ppt.

    `None` and `SourceUnavailable` mean two different things, deliberately
    kept apart:

    - Returns `None` only when the station RESPONDED and had nothing usable:
      every reading blank (CO-OPS's dark-sensor marker), malformed, or
      outside PLAUSIBLE_PPT (a stuck/miscalibrated sensor).
    - Raises `tidescout.errors.SourceUnavailable` for everything that is
      actually a failure: a bad station id, a CO-OPS `{"error": ...}`
      payload (this is how "No data was found" arrives -- see the module
      docstring's FINDING), a network fault, or a cache miss with no stale
      fallback to serve. This is NOT caught here on purpose: catching it
      would make a station typo, an API-shape change, and a transient
      network blip return the identical silent `None` as "the station has
      no salinity today," and -- since Springmaid Pier never succeeds --
      nothing would ever reveal which case actually happened. Letting it
      propagate routes the failure through the app's one place that already
      handles this (`sources.dayloader.load_day`'s `attempt()`), which
      records the source by name instead of it vanishing into a constant.

    Not defensive beyond CO-OPS's documented response shape: if `data` were
    ever something other than a list of `{"t", "s", ...}` dicts, iterating it
    raises whatever built-in exception that produces (e.g. `AttributeError`
    on a non-dict row's `.get`) rather than being caught and turned into
    `None` or `SourceUnavailable` -- left to propagate to `attempt()` too.
    """
    begin_end = day.strftime("%Y%m%d")
    params = {
        "product": "salinity",
        "application": "tidescout",
        "station": station,
        "begin_date": begin_end,
        "end_date": begin_end,
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
    }
    key = f"salinity:{station}:{begin_end}"
    # Same "coops" cache bucket as noaa.py's own key prefixes
    # (pred:/hilo:/cur:/wtemp:) -- one bucket per API family, not per module.
    # SourceUnavailable propagates from here -- see docstring above.
    cached = cache.get_or_fetch("coops", key, OBS_TTL, lambda: _get_json(params))

    values = []
    for row in cached.payload.get("data", []):
        raw = (row.get("s") or "").strip()
        if not raw:
            continue  # CO-OPS sends "" for a dark sensor; float() would not reject it
        try:
            values.append(float(raw))
        except ValueError:
            continue
    lo, hi = PLAUSIBLE_PPT
    values = [v for v in values if lo <= v <= hi]
    if not values:
        return None
    return sum(values) / len(values)
