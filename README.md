# TideScout

Personal fishing decision-support app for SC inshore waters (Winyah Bay first; Charleston, Awendaw/Cape Romain, and Murrells Inlet to follow). Simulates tidal flow over real bathymetry with ANUGA to find ambush points — eddies, seams, slack pockets next to moving water — and scores every hour of a chosen day per species (redfish, speckled trout, flounder) with a transparent, tunable factor model.

- **Design spec:** `docs/superpowers/specs/2026-08-11-tidescout-design.md`
- **Implementation plans:** `docs/superpowers/plans/`
- **Stack:** Python 3.12 + FastAPI backend, React + Vite + TypeScript + MapLibre GL frontend. Runs locally in the browser.
