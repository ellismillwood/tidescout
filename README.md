# TideScout

Personal fishing decision-support app for SC inshore waters (Winyah Bay first; Charleston, Awendaw/Cape Romain, and Murrells Inlet to follow). Simulates tidal flow over real bathymetry with ANUGA to find ambush points — eddies, seams, slack pockets next to moving water — and scores every hour of a chosen day per species (redfish, speckled trout, flounder) with a transparent, tunable factor model.

- **Design spec:** `docs/superpowers/specs/2026-08-11-tidescout-design.md`
- **Implementation plans:** `docs/superpowers/plans/`
- **Stack:** Python 3.12 + FastAPI backend, React + Vite + TypeScript + MapLibre GL frontend. Runs locally in the browser.

## Running (current state: conditions CLI)

    # one-time setup
    uv venv ~/.venvs/tidescout --python 3.12
    make install

    # discover/refresh station IDs for a fishery
    ~/.venvs/tidescout/bin/tidescout stations winyah-bay

    # a day of hourly conditions (tide, currents, wind, pressure, solunar...)
    ~/.venvs/tidescout/bin/tidescout conditions winyah-bay --date 2026-08-15 --model ecmwf

    # quality gate
    make check

    # all cached API data lives in data/cache.sqlite — delete it to force a full refresh

    # bathymetry pipeline (Plan 2): discover tiles once, then build + detect
    ~/.venvs/tidescout/bin/tidescout bathy discover winyah-bay
    ~/.venvs/tidescout/bin/tidescout bathy build winyah-bay      # ~1-2 GB of tiles cached in data/
    ~/.venvs/tidescout/bin/tidescout bathy artifacts winyah-bay  # open data/winyah-bay/quicklook.png
    ~/.venvs/tidescout/bin/tidescout features winyah-bay         # ambush-feature inventory
    ~/.venvs/tidescout/bin/tidescout spots winyah-bay            # your known spots vs detections

    # salinity / weather data: NDBC (live) + CDMO (historical) accumulating store
    ~/.venvs/tidescout/bin/tidescout salinity import-cdmo winyah-bay --path /path/to/export
    ~/.venvs/tidescout/bin/tidescout salinity citation winyah-bay  # attribution for what's held

## Data attribution

Winyah Bay water-quality and meteorological data (`sources/ndbc.py`,
`sources/cdmo.py`, `data/<slug>/ndbc.sqlite`) originates with the **NOAA
National Estuarine Research Reserve System (NERRS) System-wide Monitoring
Program**, collected and processed by the **North Inlet-Winyah Bay NERR**
(Baruch Marine Field Laboratory, University of South Carolina) and
distributed via the NERRS Centralized Data Management Office
(nerrsdata.org) — some of it redistributed through NOAA's NDBC buoy
network (station WYSS1), which does not detach the citation obligation
from the data. Run `tidescout salinity citation <slug>` for the exact
citation, acknowledgement, and disclaimer this store's held data currently
earns, generated from the store itself rather than hardcoded here (see
`sources/ndbc.py`'s "PROVENANCE AND CITATION"). The disclaimer applies in
full: this project bears all responsibility for its use of the data, and
NOAA/the Federal government assumes no liability for it.
