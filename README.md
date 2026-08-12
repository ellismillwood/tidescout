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
