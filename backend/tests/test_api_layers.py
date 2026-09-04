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


def test_features_layer_is_served_with_a_strong_etag_and_must_revalidate(client):
    """`no-cache` means "revalidate every time", not "do not store".

    NOT `immutable`: that is only sound for content-addressed URLs, and
    `/layers/<name>` is a fixed path that `tidescout bathy artifacts` rewrites
    IN PLACE. Under `immutable` the browser would serve a year-old copy and
    never send the `If-None-Match` that the conditional branch below exists to
    answer -- making that branch reachable only from this test file, and a
    regenerated layer invisible until a hard reload.
    """
    r = client.get("/api/fisheries/winyah-bay/layers/features")
    assert r.status_code == 200
    assert r.headers.get("etag"), "revalidation needs an ETag to revalidate against"
    assert r.headers["cache-control"] == "no-cache"


def test_a_matching_conditional_request_gets_304_and_a_wrong_one_gets_the_body(client):
    """Both halves. A handler that returned 304 for ANY `If-None-Match` would
    pass a match-only test while wedging every client that ever cached a
    superseded copy.
    """
    r1 = client.get("/api/fisheries/winyah-bay/layers/features")
    match = client.get(
        "/api/fisheries/winyah-bay/layers/features",
        headers={"If-None-Match": r1.headers["etag"]},
    )
    assert match.status_code == 304
    assert match.content == b""

    stale = client.get(
        "/api/fisheries/winyah-bay/layers/features",
        headers={"If-None-Match": '"not-the-current-tag"'},
    )
    assert stale.status_code == 200
    assert stale.content == r1.content, "a non-matching tag must get the real body"


def test_regenerating_a_layer_changes_its_etag_and_supersedes_the_old_one(
    client, monkeypatch, tmp_path
):
    """The reason `immutable` was wrong, pinned as behaviour rather than as a
    header string. These artifacts are rewritten IN PLACE by
    `tidescout bathy artifacts`, so the ETag has to move with the file and a
    request still carrying the previous tag has to get the NEW bytes -- not a
    304 promising the old ones are still current.
    """
    import os

    monkeypatch.setattr(layers, "DATA_DIR", tmp_path)
    (tmp_path / "winyah-bay").mkdir()
    target = tmp_path / "winyah-bay" / "features.geojson"
    target.write_text('{"type":"FeatureCollection","features":[]}')

    url = "/api/fisheries/winyah-bay/layers/features"
    first = client.get(url)
    assert first.status_code == 200
    old_tag = first.headers["etag"]
    assert client.get(url, headers={"If-None-Match": old_tag}).status_code == 304

    # Regenerate in place. The ETag is derived from (mtime, size), and a
    # rewrite inside the same mtime tick is exactly the case a size-only or
    # mtime-only tag would miss, so both are moved.
    target.write_text('{"type":"FeatureCollection","features":[{"id":1}]}')
    bumped = target.stat().st_mtime + 10
    os.utime(target, (bumped, bumped))

    second = client.get(url)
    assert second.status_code == 200
    assert second.headers["etag"] != old_tag, "a rewritten layer must get a new ETag"

    superseded = client.get(url, headers={"If-None-Match": old_tag})
    assert superseded.status_code == 200, "the old tag must no longer match"
    assert superseded.json() == second.json() != first.json()


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
