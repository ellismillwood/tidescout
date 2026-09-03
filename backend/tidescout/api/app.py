"""The FastAPI app. Routes only -- the work lives in the sibling modules."""

from fastapi import FastAPI

from tidescout.api import readiness


def create_app() -> FastAPI:
    app = FastAPI(title="TideScout", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/fisheries")
    def list_fisheries() -> list[dict]:
        return readiness.fishery_summaries()

    return app
