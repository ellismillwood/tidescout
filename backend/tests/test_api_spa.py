"""Serving the built frontend, and the SPA fallback spec §4.3 promises.

`StaticFiles(html=True)` was assumed to provide this and does not: it serves
`index.html` only for a path resolving to a DIRECTORY, so a genuine miss like
`/day/2026-09-03` -- exactly the reload the fallback exists for -- returned a
404. These tests pin both halves, because a catch-all that swallowed `/api`
would "fix" the reload by breaking every fetch the page makes.
"""

import pytest
from fastapi.testclient import TestClient

from tidescout.api.app import create_app

SHELL = "<!doctype html><title>TideScout</title><div id=root></div>"


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(SHELL)
    (d / "assets" / "app.js").write_text("console.log('real asset')")
    return d


@pytest.fixture
def client(dist):
    return TestClient(create_app(frontend_dist=dist))


@pytest.mark.parametrize(
    "path", ["/", "/day/2026-09-03", "/fisheries/winyah-bay/day/2026-09-03?model=gfs", "/settings"]
)
def test_a_deep_client_side_route_returns_the_app_shell(client, path):
    """The reload case. Measured before the fix: `/day/2026-09-03` was a 404."""
    r = client.get(path)
    assert r.status_code == 200, r.text
    assert r.text == SHELL


def test_a_real_static_file_is_served_as_itself_not_as_the_shell(client):
    """The fallback must not shadow the assets the shell then loads -- a
    catch-all that returned index.html for everything would leave the page
    blank while every one of these tests' status codes stayed 200."""
    r = client.get("/assets/app.js")
    assert r.status_code == 200
    assert r.text == "console.log('real asset')"


def test_api_routes_still_match_first_and_return_json_not_the_shell(client):
    """The other half. The catch-all is registered last, so `/api/fisheries`
    must still be answered by its own route."""
    r = client.get("/api/fisheries")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert isinstance(body, list) and body[0]["slug"] == "winyah-bay"


def test_an_unknown_api_path_is_404_not_the_shell(client):
    """A typo'd endpoint must stay an error. Serving the HTML shell there would
    turn it into a 200 that no `fetch().json()` could parse."""
    r = client.get("/api/no-such-endpoint")
    assert r.status_code == 404
    assert SHELL not in r.text


def test_no_dist_means_no_fallback_at_all(tmp_path):
    """With no built frontend the API must not invent one: a missing
    `frontend/dist` is the normal state before the §9 frontend exists."""
    c = TestClient(create_app(frontend_dist=tmp_path / "nope"))
    assert c.get("/api/fisheries").status_code == 200
    assert c.get("/day/2026-09-03").status_code == 404
