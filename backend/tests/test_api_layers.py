"""Static per-fishery layers, and the allowlist that guards them."""

import pytest
from fastapi.testclient import TestClient

from tidescout.api import layers
from tidescout.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.mark.parametrize(
    "name",
    ["../../../etc/passwd", "..%2f..%2fetc", "features/../../secret", "", "cache.sqlite"],
)
def test_traversing_or_unlisted_layer_names_are_rejected(name):
    """`slug` and `name` both reach the filesystem. The allowlist is what stops
    that, so it gets a test rather than a code comment (spec §4).

    `cache.sqlite` is in this list deliberately: it is a REAL file under
    `data/`, so a check that merely verified existence would serve it.
    """
    with pytest.raises((KeyError, ValueError)):
        layers.layer_path("winyah-bay", name)


def test_an_unknown_slug_is_rejected_before_any_path_is_built(tmp_path, monkeypatch):
    """`paths.fishery_data_dir` mkdirs its argument, so slug validation has to
    come first -- see the Global Constraints. Asserted by pointing DATA_DIR at
    an empty tmp dir and proving NOTHING was created in it, rather than by
    trusting the raise alone."""
    monkeypatch.setattr(layers, "DATA_DIR", tmp_path)
    for bad in ("../../etc", "no-such-bay", ""):
        with pytest.raises(ValueError):
            layers.layer_path(bad, "features")
    assert list(tmp_path.iterdir()) == [], "slug validation must precede any path work"


def test_every_allowlisted_name_maps_to_a_filename():
    assert set(layers.LAYERS) == {
        "features", "contours", "oysters", "hillshade", "hillshade-bounds",
    }


def test_features_layer_is_served_with_a_strong_etag(client):
    r = client.get("/api/fisheries/winyah-bay/layers/features")
    assert r.status_code == 200
    assert r.headers.get("etag"), "immutable artifacts must carry an ETag"
    assert "immutable" in r.headers.get("cache-control", "")


def test_a_conditional_request_gets_304(client):
    r1 = client.get("/api/fisheries/winyah-bay/layers/features")
    r2 = client.get(
        "/api/fisheries/winyah-bay/layers/features",
        headers={"If-None-Match": r1.headers["etag"]},
    )
    assert r2.status_code == 304


def test_an_unlisted_layer_name_is_404(client):
    r = client.get("/api/fisheries/winyah-bay/layers/nonesuch")
    assert r.status_code == 404


def test_an_allowlisted_but_unbuilt_layer_is_404_not_500(client, monkeypatch, tmp_path):
    """A DIFFERENT branch from the one above: the name is valid, the file is
    simply not built yet. Pointing DATA_DIR at an empty dir exercises it
    without depending on which artifacts happen to exist in this checkout --
    `oyster_reefs.web.geojson` is absent before Task 6 and present after, so a
    test written against that would flip.
    """
    monkeypatch.setattr(layers, "DATA_DIR", tmp_path)
    (tmp_path / "winyah-bay").mkdir()
    r = client.get("/api/fisheries/winyah-bay/layers/oysters")
    assert r.status_code == 404
    assert "has not been built" in r.json()["detail"]
