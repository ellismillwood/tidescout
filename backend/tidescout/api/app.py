"""The FastAPI app. Routes only -- the work lives in the sibling modules."""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from tidescout.api import layers as layers_mod
from tidescout.api import readiness, store
from tidescout.api.builds import BuildCoordinator
from tidescout.config import fishery_now, load_fishery
from tidescout.engine import flow as flow_engine
from tidescout.engine import salinity as salinity_engine
from tidescout.engine import structure
from tidescout.engine.phase import library_phase
from tidescout.paths import DATA_DIR
from tidescout.pipeline import flowlib
from tidescout.pipeline import payload as payload_mod
from tidescout.pipeline.estuary import load_distance_field
from tidescout.sources import dayloader
from tidescout.sources.cache import Cache
from tidescout.sources.weather import FORECAST_HORIZON_DAYS, WEATHER_MODELS


def _parse_day(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise HTTPException(422, f"not a date: {raw!r} (expected YYYY-MM-DD)") from None


def _check_model(model: str) -> None:
    """`model` is a query parameter that becomes part of a FILENAME.

    Without this it reaches `store.payload_path` unvalidated, and
    `write_payload` mkdirs the parent, so `?model=../../../x` writes outside
    `data/` and a second request serves any gzipped JSON on disk. Nothing
    upstream stops it: `weather.fetch_weather` raises KeyError on an unknown
    model, but `dayloader.load_day`'s `attempt()` catches bare `Exception` and
    degrades to `missing: ['weather']`, so the build SUCCEEDS and writes.
    """
    if model not in WEATHER_MODELS:
        raise HTTPException(
            422,
            f"unknown weather model: {model!r} "
            f"(expected one of: {', '.join(sorted(WEATHER_MODELS))})",
        )


def _require_ready(slug: str) -> None:
    r = readiness.readiness(slug)
    if r.missing == ("unknown fishery",):
        raise HTTPException(404, f"unknown fishery: {slug!r}")
    if not r.ready:
        raise HTTPException(409, f"fishery not processed -- missing: {', '.join(r.missing)}")


def _check_range(day: date, now: datetime) -> None:
    horizon = now.date() + timedelta(days=FORECAST_HORIZON_DAYS)
    if day > horizon:
        raise HTTPException(
            422,
            f"{day.isoformat()} is beyond the forecast horizon of "
            f"{FORECAST_HORIZON_DAYS} days (through {horizon.isoformat()})",
        )


def create_app(
    coordinator: BuildCoordinator | None = None,
    frontend_dist: Path | None = None,
    dev_cors: bool = False,
) -> FastAPI:
    app = FastAPI(title="TideScout", docs_url="/api/docs", openapi_url="/api/openapi.json")
    coord = coordinator if coordinator is not None else BuildCoordinator()

    @app.get("/api/fisheries")
    def list_fisheries() -> list[dict]:
        return readiness.fishery_summaries()

    @app.get("/api/fisheries/{slug}/day/{raw_day}")
    def get_day(slug: str, raw_day: str, model: str = "best"):
        _check_model(model)
        day = _parse_day(raw_day)
        _require_ready(slug)
        # Fishery-local, NOT UTC: `config.fishery_now` carries the reason.
        # Both users of `now` below gate on a calendar day, and Winyah Bay's
        # day boundary is five hours off UTC's.
        now = fishery_now(slug)
        _check_range(day, now)

        got = store.read_payload_gz(slug, day, model)
        if got is not None:
            raw, payload = got
            # A stale hit is served IMMEDIATELY and rebuilt in the background:
            # the user never waits on a rebuild for data already on disk.
            if store.is_stale(payload, day, now):
                coord.ensure(slug, day, model)
            # The STORED BYTES, verbatim. Gunzipping only to re-serialise costs
            # ~270 ms of CPU per request at the spec's measured 24.59 MB and
            # puts 24.59 MB on the wire instead of the 1.67 MB gzipped figure
            # §5 sizes the frontend against. The payload itself is untouched --
            # `missing: ['weather']` and a lowered `confidence` reach the
            # client exactly as written (§6).
            return Response(
                raw, media_type="application/json", headers={"Content-Encoding": "gzip"}
            )

        state = coord.ensure(slug, day, model)
        return JSONResponse(
            {"status": "building", "started_at": state.started_at.isoformat(),
             "key": f"{slug}/{day.isoformat()}/{model}"},
            status_code=202,
        )

    @app.get("/api/fisheries/{slug}/day/{raw_day}/status")
    def get_status(slug: str, raw_day: str, model: str = "best"):
        _check_model(model)
        day = _parse_day(raw_day)
        _require_ready(slug)
        now = fishery_now(slug)

        payload = store.read_payload(slug, day, model)
        if payload is not None:
            return {
                "status": "ready",
                "generated_at": (payload.get("freshness") or {}).get("generated_at"),
                "stale": store.is_stale(payload, day, now),
            }
        state = coord.state(slug, day, model)
        if state is None:
            return {"status": "absent"}
        return {"status": state.status, "error": state.error}

    @app.get("/api/fisheries/{slug}/layers/{name}")
    def get_layer(slug: str, name: str, request: Request):
        try:
            path = layers_mod.layer_path(slug, name)
        except (KeyError, ValueError):
            # One 404 for both cases on purpose: an unlisted layer name and an
            # unknown fishery should be indistinguishable from outside.
            raise HTTPException(404, "no such layer") from None
        if not path.exists():
            raise HTTPException(404, f"layer {name!r} has not been built for {slug!r}")
        # FileResponse sets a strong ETag from (mtime, size), but -- unlike
        # starlette.staticfiles.StaticFiles -- it does not itself compare that
        # ETag against If-None-Match, so the 304 has to be done here. Passing
        # stat_result up front (the same trick StaticFiles uses) makes the
        # ETag available synchronously instead of only during the ASGI call.
        # `no-cache` means "revalidate every time", NOT "do not store". It is
        # deliberately not `immutable`: `immutable` is only sound for
        # content-addressed URLs, and `/layers/oysters` is a fixed path that
        # `tidescout bathy artifacts` rewrites IN PLACE. With `immutable` a
        # browser would serve a year-old copy and never send the
        # `If-None-Match` the conditional block below exists to answer, so a
        # regenerated layer would be invisible until a hard reload -- and that
        # block would only ever run from the test suite. With the strong ETag,
        # revalidating costs one ~200-byte 304 instead of re-sending 8 MB.
        headers = {"Cache-Control": "no-cache"}
        response = FileResponse(path, headers=headers, stat_result=path.stat())
        if_none_match = request.headers.get("if-none-match")
        if if_none_match:
            etag = response.headers["etag"]
            if etag in (tag.strip().removeprefix("W/") for tag in if_none_match.split(",")):
                return Response(status_code=304, headers={"ETag": etag, **headers})
        return response

    def _check_hour(hour: int) -> int:
        if not 0 <= hour <= 23:
            raise HTTPException(422, f"hour must be 0-23, got {hour}")
        return hour

    def _regime_mix(slug: str, day, model: str):
        """The regime blend and phase axis for a day, without scoring it.

        Reuses `build_payload`'s own helpers so the overlay cannot drift from
        what the markers were scored against -- an arrow field derived from a
        different regime blend than the activations would be quietly wrong.
        """
        fishery = load_fishery(slug)
        # A FRESH Cache, never `sources.cache.default_cache()`. That is a
        # module-level singleton wrapping one sqlite3 connection, and FastAPI
        # runs sync `def` routes in a threadpool -- so the singleton would
        # cross threads and raise ProgrammingError on the first request. No
        # existing route touches it; these would be the first.
        day_conditions = dayloader.load_day(
            fishery, day, model, Cache(DATA_DIR / "cache.sqlite")
        )
        rb = payload_mod._range_bucket_for_day(getattr(day_conditions, "moon", None))
        available = payload_mod._available_regimes(slug)
        disch = getattr(day_conditions, "discharge", None)
        if disch is None or disch.cfs_now is None or not available:
            raise HTTPException(404, "no regime resolves for this day")
        mix, _ = flow_engine.blend_regimes(
            rb, disch.cfs_now, fishery.discharge_buckets, available
        )
        return fishery, day_conditions, mix

    @app.get("/api/fisheries/{slug}/flow-vectors/{raw_day}")
    def get_flow_vectors(slug: str, raw_day: str, hour: int, model: str = "best"):
        _check_model(model)
        _check_hour(hour)
        day = _parse_day(raw_day)
        _require_ready(slug)

        fishery, day_conditions, mix = _regime_mix(slug, day, model)
        events, ok = payload_mod._flow_events(fishery, day, Cache(DATA_DIR / "cache.sqlite"))
        if not ok:
            raise HTTPException(404, "no tide events for this day")
        hour_obj = day_conditions.hours[hour]
        lib_phase = library_phase(events, hour_obj.time)
        if lib_phase is None:
            raise HTTPException(404, "no library phase for this hour")

        regime_phases = payload_mod._regime_phase_axes(slug, mix)
        state = payload_mod._blended_state(slug, regime_phases, mix, lib_phase)
        spec = flowlib.grid_spec(slug, fishery)

        # Decimate to a drawable arrow density. 587,325 cells would be both
        # unreadable and larger than the day payload this exists to stay out of.
        step = max(1, int((spec.shape[0] * spec.shape[1] / 2500) ** 0.5))
        ug = structure.to_grid(state["u"], spec.flat_index, spec.shape)
        vg = structure.to_grid(state["v"], spec.flat_index, spec.shape)
        wet = flow_engine.wet_mask(state["depth"])
        wg = structure.to_grid(wet.astype("float64"), spec.flat_index, spec.shape, fill=0.0)
        ug = np.where(wg > 0.5, ug, 0.0)[::step, ::step]
        vg = np.where(wg > 0.5, vg, 0.0)[::step, ::step]

        # `GridSpec` has NO `bbox` attribute. `xs`/`ys` are the IN-DOMAIN cell
        # centres only -- a strictly smaller extent than the grid `ug`/`vg`
        # actually span, because `ug`/`vg` come from the FULL `spec.shape`
        # raster (out-of-domain cells zeroed, not cropped) before decimation.
        # Deriving bbox from xs/ys min/max shrinks it relative to the shipped
        # u/v grid -- measured ~2.1x too narrow in x and ~1.7x in y, up to
        # ~18 km of positional error once a client maps the array onto it.
        # `array_bounds` on `spec.shape`/`spec.transform` instead describes
        # the full raster -- the same grid `ug`/`vg` are a decimation of --
        # and the result is reprojected to WGS84 degrees for MapLibre, the
        # same way `webartifacts.hillshade_png` converts its bounds.
        from rasterio.transform import array_bounds
        from rasterio.warp import transform_bounds

        west, south, east, north = transform_bounds(
            spec.crs or "EPSG:26917",
            "EPSG:4326",
            *array_bounds(spec.shape[0], spec.shape[1], spec.transform),
        )
        return {
            "hour": hour,
            "rows": int(ug.shape[0]),
            "cols": int(ug.shape[1]),
            "bbox": [west, south, east, north],
            "u": [float(x) for x in np.nan_to_num(ug).ravel()],
            "v": [float(x) for x in np.nan_to_num(vg).ravel()],
        }

    @app.get("/api/fisheries/{slug}/salinity-field/{raw_day}")
    def get_salinity_field(slug: str, raw_day: str, hour: int, model: str = "best"):
        _check_model(model)
        _check_hour(hour)
        day = _parse_day(raw_day)
        _require_ready(slug)

        fishery, day_conditions, _ = _regime_mix(slug, day, model)
        try:
            km = load_distance_field(slug)
        except FileNotFoundError:
            raise HTTPException(404, "no along-estuary distance field") from None

        hour_obj = day_conditions.hours[hour]
        phase = payload_mod._salinity_phase(hour_obj)
        disch = day_conditions.discharge
        if phase is None or disch is None or disch.cfs_now is None:
            raise HTTPException(404, "no phase or discharge for this hour")

        # Bin by along-estuary distance: salinity is a function of km, so one
        # value per kilometre bin describes the whole field, and the client
        # paints it by joining back to the distance field it already has.
        finite = km[np.isfinite(km)]
        edges = np.arange(np.floor(finite.min()), np.ceil(finite.max()) + 1.0, 1.0)
        cells = []
        for left in edges:
            field = salinity_engine.salinity_field(
                float(left), disch.cfs_now, phase, fishery.salinity
            )
            cells.append({"km": float(left), "ppt": float(field.ppt)})

        return {
            "hour": hour,
            "fitted": fishery.salinity.fitted,
            "extrapolated": payload_mod._is_extrapolated(disch.cfs_now, fishery.salinity),
            "cells": cells,
        }

    if dev_cors:
        # Development runs two servers -- Vite on :5173, this on :8000. In
        # production the frontend is same-origin and no CORS headers are
        # emitted at all.
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    if frontend_dist is not None and frontend_dist.is_dir():
        # Spec §4.3: any non-/api path that does not match a file returns
        # index.html, so client-side routes survive a page reload.
        #
        # `StaticFiles(html=True)` does NOT do that. It serves index.html only
        # for a path that resolves to a DIRECTORY; a genuine miss looks for
        # `404.html` and then raises 404 -- measured, `/day/2026-09-03` was a
        # 404. And a `Mount("/")` matches every path FULLY, so a catch-all
        # route registered after it would never be reached. Hence one explicit
        # catch-all that delegates real files to StaticFiles (keeping its safe
        # path resolution, content types, ranges and 304s) and falls back to
        # the shell. This block is the LAST thing registered on `app` --
        # every route above it, including flow-vectors and salinity-field,
        # is defined and therefore matched first. A route added below this
        # point would be shadowed by the catch-all exactly as these two once
        # were; new routes belong above this block, not after it.
        from starlette.exceptions import HTTPException as StarletteHTTPException
        from starlette.staticfiles import StaticFiles

        static = StaticFiles(directory=frontend_dist)
        index = frontend_dist / "index.html"

        @app.get("/{spa_path:path}", include_in_schema=False)
        async def spa(spa_path: str, request: Request) -> Response:
            if spa_path == "api" or spa_path.startswith("api/"):
                # An unknown /api path is a bug, not a client-side route.
                # Returning the HTML shell there would turn a typo'd endpoint
                # into a 200 that no fetch() could parse.
                raise HTTPException(404, "no such endpoint")
            try:
                return await static.get_response(spa_path or "index.html", request.scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404:
                    raise
                return FileResponse(index)

    return app
