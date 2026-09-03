"""The FastAPI app. Routes only -- the work lives in the sibling modules."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from tidescout.api import layers as layers_mod
from tidescout.api import readiness, store
from tidescout.api.builds import BuildCoordinator
from tidescout.sources.weather import WEATHER_MODELS

# Open-Meteo's forecast endpoint allowed through +16 days when measured
# (2026-09-03); anything older than `weather.ARCHIVE_CUTOFF_DAYS` routes to the
# ERA5 archive, which reaches back years. The bound that bites is the future
# one, so only it is enforced here.
FORECAST_HORIZON_DAYS = 16


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
        now = datetime.now(UTC)
        _check_range(day, now)

        payload = store.read_payload(slug, day, model)
        if payload is not None:
            # A stale hit is served IMMEDIATELY and rebuilt in the background:
            # the user never waits on a rebuild for data already on disk.
            if store.is_stale(payload, day, now):
                coord.ensure(slug, day, model)
            return JSONResponse(payload)

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
        now = datetime.now(UTC)

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
        # the shell. Registered LAST, so every /api route still matches first.
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
