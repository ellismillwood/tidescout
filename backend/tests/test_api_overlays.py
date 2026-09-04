"""The two overlays §9 asks for and the day payload cannot supply."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tidescout.api import store
from tidescout.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _day() -> str:
    return (datetime.now(UTC).date() - timedelta(days=2)).isoformat()


@pytest.mark.parametrize("endpoint", ["flow-vectors", "salinity-field"])
@pytest.mark.parametrize(
    "bad_model", ["../../../etc/passwd", "../..", "gfs;rm -rf", "nonesuch"]
)
def test_overlays_reject_a_bad_model_exactly_as_the_day_endpoint_does(
    client, endpoint, bad_model
):
    """PR #12 closed a path-traversal where `?model=../../..` created
    directories and wrote files outside `data/`. These endpoints take the same
    parameter, so they must call the same `_check_model` -- a new endpoint with
    its own ad-hoc check, or none, reintroduces it.
    """
    r = client.get(
        f"/api/fisheries/winyah-bay/{endpoint}/{_day()}",
        params={"model": bad_model, "hour": 12},
    )
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("endpoint", ["flow-vectors", "salinity-field"])
@pytest.mark.parametrize("bad_hour", [-1, 24, 99])
def test_overlays_reject_an_out_of_range_hour(client, endpoint, bad_hour):
    r = client.get(
        f"/api/fisheries/winyah-bay/{endpoint}/{_day()}",
        params={"hour": bad_hour},
    )
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("endpoint", ["flow-vectors", "salinity-field"])
def test_overlays_404_an_unknown_fishery(client, endpoint):
    r = client.get(f"/api/fisheries/no-such-bay/{endpoint}/{_day()}?hour=12")
    assert r.status_code == 404


def test_flow_vectors_returns_a_decimated_grid_not_every_cell(client):
    """The library has 587,325 cells. Shipping all of them as arrows would be
    both unusable and larger than the day payload; the endpoint decimates.

    Asserts the shape AND that the vectors are not all zero -- a grid of
    zeroes would satisfy every structural check while drawing nothing.
    """
    r = client.get(f"/api/fisheries/winyah-bay/flow-vectors/{_day()}?hour=12")
    if r.status_code == 404:
        pytest.skip("no flow library for this date in this checkout")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hour"] == 12
    assert len(body["u"]) == len(body["v"]) == body["cols"] * body["rows"]
    assert len(body["u"]) < 10_000, "not decimated"
    assert any(abs(x) > 1e-6 for x in body["u"] + body["v"]), "all-zero field"
    # WGS84 degrees, not the grid's native UTM metres. GridSpec carries
    # xs/ys in EPSG:26917; shipping those raw would place every arrow
    # somewhere off the coast of Africa with no error anywhere.
    w, s_, e, n = body["bbox"]
    assert -82 < w < -78 and -82 < e < -78, body["bbox"]
    assert 32 < s_ < 34 and 32 < n < 34, body["bbox"]
    assert w < e and s_ < n


def test_salinity_field_cannot_be_rendered_without_its_disclosure(client):
    """Winyah's salinity model is falsified (`fitted: false`). Spec §1.1 draws
    it anyway -- "flagged, not discounted" -- but only because the response
    carries the flag. A client cannot honour a disclosure it was never sent.
    """
    r = client.get(f"/api/fisheries/winyah-bay/salinity-field/{_day()}?hour=12")
    if r.status_code == 404:
        pytest.skip("no distance field in this checkout")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "fitted" in body and "extrapolated" in body
    assert body["fitted"] is False, "winyah's model is unfitted; see spec §1.1"
    assert body["cells"], "no field to draw"


def test_flow_vectors_bbox_describes_the_full_raster_not_the_indomain_subset(client):
    """`u`/`v` are decimated from the FULL `(2527, 1903)` bathymetry raster.
    `bbox` used to be derived from `spec.xs`/`spec.ys`, which are the
    IN-DOMAIN cell CENTRES only -- a strictly smaller extent, because the
    raster carries hundreds of rows/cols of out-of-domain padding the mask
    excludes but `u`/`v` still span (out-of-domain cells are zeroed, not
    cropped).

    Measured directly against `bathy_utm.tif` via `array_bounds` (2026-09-03,
    task-13 report §2b): the full extent is west -79.458, south 33.144, east
    -79.040, north 33.606. The in-domain xs/ys subset the endpoint used to
    ship instead measured west -79.331, south 33.163, east -79.134, north
    33.437 -- ~2.1x narrower in x and ~1.7x narrower in y, a difference of
    more than 0.1 degree in each axis' total span. Asserting the real numbers
    (not just "a bbox came back") is what makes this discriminate a
    regression back to xs/ys -- a bbox that shrank back to the shipped subset
    would still be four in-range floats with w < e and s < n.
    """
    r = client.get(f"/api/fisheries/winyah-bay/flow-vectors/{_day()}?hour=12")
    if r.status_code == 404:
        pytest.skip("no flow library for this date in this checkout")
    assert r.status_code == 200, r.text
    w, s_, e, n = r.json()["bbox"]

    assert w == pytest.approx(-79.458, abs=0.01)
    assert s_ == pytest.approx(33.144, abs=0.01)
    assert e == pytest.approx(-79.040, abs=0.01)
    assert n == pytest.approx(33.606, abs=0.01)

    # ...and the pair: the shipped extent's SPAN, not just its corners, must
    # be wider than the in-domain subset's was -- by more than 0.1 degree in
    # both axes.
    shipped_subset_w, shipped_subset_s, shipped_subset_e, shipped_subset_n = (
        -79.33062, 33.16283, -79.13362, 33.43675,
    )
    assert (e - w) - (shipped_subset_e - shipped_subset_w) > 0.1, "x-span not widened"
    assert (n - s_) - (shipped_subset_n - shipped_subset_s) > 0.1, "y-span not widened"


class _FakeCoordinator:
    """A stand-in for `BuildCoordinator` that never touches the network.

    Only `get_day`/`get_status` call `coord` at all, and the real
    `BuildCoordinator.ensure` submits a REAL background build -- weather,
    tide and USGS fetches -- to a thread pool. This test only cares whether
    the route resolves at all, not what it builds, so it fakes the
    coordinator the same way `test_api_day.py` does.
    """

    def ensure(self, slug, day, model):
        from tidescout.api.builds import BuildState

        return BuildState("building", datetime.now(UTC))

    def state(self, slug, day, model):
        return None


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text(
        "<!doctype html><title>TideScout</title><div id=root></div>"
    )
    return d


def test_every_api_route_still_resolves_with_the_spa_catch_all_registered(
    dist, monkeypatch, tmp_path
):
    """Closes the blind spot this file's own `client` fixture leaves open.

    `TestClient(create_app())` never passes `frontend_dist`, so the SPA
    catch-all in `app.py` never registers at all -- every other test in this
    file runs against an app that structurally CANNOT exhibit a route-
    ordering conflict, because there is no catch-all route to conflict with.
    `tidescout serve` (`cli.py`) always passes `frontend_dist`, so that is the
    shape that has to work, and only building the app that way here exercises
    it.

    `flow-vectors` and `salinity-field` were once defined AFTER the catch-all,
    so Starlette matched the catch-all first and both 404'd with the
    catch-all's OWN message, `no such endpoint` -- verified against a real
    `tidescout serve` (task-13 report §2a). `_not_swallowed` below checks for
    that EXACT message rather than "not a shell page", because a legitimate
    domain 404 (missing fishery data, no regime for the day, ...) is also a
    non-swallowed response and must not be mistaken for the bug.

    Every `/api/*` route the app defines is checked, paired in the same test
    with the other half: an unrecognised, non-API path must still fall
    through to the SPA shell. Either half alone would pass against a broken
    app -- checking only /api/* would pass even if the catch-all were deleted
    outright (nothing left to swallow /api/*, but also nothing serving the
    shell), and checking only the shell would pass even with these two routes
    shadowed exactly as they once were.
    """
    # `store.DATA_DIR` only, not `tidescout.paths.DATA_DIR`: flow-vectors and
    # salinity-field read real winyah-bay data through their OWN DATA_DIR
    # import in app.py and must keep doing so, but the day/status endpoints'
    # payload cache is redirected to an empty dir so a miss is deterministic
    # rather than depending on whatever a previous local `tidescout serve`
    # happened to leave on disk.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "no-payloads-here")
    client = TestClient(create_app(coordinator=_FakeCoordinator(), frontend_dist=dist))
    day = _day()

    def _not_swallowed(r):
        if r.status_code != 404:
            return True
        try:
            return r.json().get("detail") != "no such endpoint"
        except ValueError:
            return True

    fisheries = client.get("/api/fisheries")
    assert fisheries.status_code == 200, fisheries.text
    assert any(f["slug"] == "winyah-bay" for f in fisheries.json())

    day_r = client.get(f"/api/fisheries/winyah-bay/day/{day}")
    assert _not_swallowed(day_r), day_r.text
    assert day_r.status_code == 202, day_r.text  # no cached payload; fake coord says "building"

    status_r = client.get(f"/api/fisheries/winyah-bay/day/{day}/status")
    assert _not_swallowed(status_r), status_r.text
    assert status_r.json()["status"] == "absent"

    layers_r = client.get("/api/fisheries/winyah-bay/layers/features")
    assert _not_swallowed(layers_r), layers_r.text
    assert layers_r.status_code == 200, layers_r.text  # features.geojson ships with the repo

    flow_r = client.get(f"/api/fisheries/winyah-bay/flow-vectors/{day}?hour=12")
    assert _not_swallowed(flow_r), flow_r.text

    sal_r = client.get(f"/api/fisheries/winyah-bay/salinity-field/{day}?hour=12")
    assert _not_swallowed(sal_r), sal_r.text

    # The pair this test exists to check together: an unrecognised, non-API
    # path must still return the SPA shell, not a 404.
    shell = client.get("/some/unknown/client-route")
    assert shell.status_code == 200, shell.text
    assert "TideScout" in shell.text
