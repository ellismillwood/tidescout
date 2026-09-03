"""The FastAPI app. Routes only -- the work lives in the sibling modules."""

from datetime import UTC, date, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from tidescout.api import layers as layers_mod
from tidescout.api import readiness, store
from tidescout.api.builds import BuildCoordinator

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


def create_app(coordinator: BuildCoordinator | None = None) -> FastAPI:
    app = FastAPI(title="TideScout", docs_url="/api/docs", openapi_url="/api/openapi.json")
    coord = coordinator if coordinator is not None else BuildCoordinator()

    @app.get("/api/fisheries")
    def list_fisheries() -> list[dict]:
        return readiness.fishery_summaries()

    @app.get("/api/fisheries/{slug}/day/{raw_day}")
    def get_day(slug: str, raw_day: str, model: str = "best"):
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
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        response = FileResponse(path, headers=headers, stat_result=path.stat())
        if_none_match = request.headers.get("if-none-match")
        if if_none_match:
            etag = response.headers["etag"]
            if etag in (tag.strip().removeprefix("W/") for tag in if_none_match.split(",")):
                return Response(status_code=304, headers={"ETag": etag, **headers})
        return response

    return app
