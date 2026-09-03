"""The two overlays §9 asks for and the day payload cannot supply."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

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
