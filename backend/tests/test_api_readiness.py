"""Which fisheries exist, and which are actually processed."""

from tidescout.api import readiness


def test_slug_discovery_excludes_the_sidecar_yamls():
    """`fisheries/` holds four YAMLs but only ONE fishery. `species_weights`,
    `winyah-bay.known-spots` and `winyah-bay.tiles` are sidecars, and a naive
    `*.yaml` glob would offer all three as pickable fisheries in the §9 top bar.
    """
    slugs = readiness.fishery_slugs()
    assert "winyah-bay" in slugs
    # The property under test, stated directly so adding a second fishery does
    # not break this test for the wrong reason:
    assert "species_weights" not in slugs
    assert not any(s.endswith((".known-spots", ".tiles")) for s in slugs), slugs


def test_readiness_reports_what_is_missing_not_just_a_boolean():
    """Spec §2: the `409` must NAME what is missing, and it is backed by this
    same predicate. A bare bool could not produce that message."""
    r = readiness.readiness("winyah-bay")
    assert r.ready is True, r.missing
    assert r.missing == ()


def test_an_unknown_slug_is_not_ready_and_creates_no_directory():
    """`paths.fishery_data_dir` calls `mkdir(parents=True, exist_ok=True)`, so
    asking about an unknown slug must not reach it -- otherwise a request for a
    nonexistent fishery silently litters `data/` with empty directories, and a
    traversing slug does worse.
    """
    from tidescout.paths import DATA_DIR

    r = readiness.readiness("no-such-bay")
    assert r.ready is False
    assert "unknown fishery" in r.missing
    assert not (DATA_DIR / "no-such-bay").exists()


def test_summaries_carry_what_the_picker_needs():
    """§9's top bar needs a label and a `ready` flag per fishery."""
    rows = readiness.fishery_summaries()
    assert len(rows) == 1
    row = rows[0]
    assert row["slug"] == "winyah-bay"
    assert row["name"] == "Winyah Bay"
    assert row["timezone"] == "America/New_York"
    assert row["ready"] is True
    assert len(row["center"]) == 2


def test_get_api_fisheries_returns_the_summaries_over_http():
    """This diff's only route lives in `app.py`, and every other test in this
    file calls `readiness` directly -- none of them would notice a wiring bug
    in the route itself (wrong path, wrong verb, a decorator applied to the
    wrong function). Drive it through `TestClient` and tie the body back to
    `fishery_summaries()`'s own output, not just a 200 -- a route that
    returned `[]` would also pass a status-code-only check.
    """
    from fastapi.testclient import TestClient

    from tidescout.api.app import create_app

    client = TestClient(create_app())
    response = client.get("/api/fisheries")
    assert response.status_code == 200

    body = response.json()
    assert body == readiness.fishery_summaries()

    row = next(r for r in body if r["slug"] == "winyah-bay")
    assert row["name"] == "Winyah Bay"
    assert row["timezone"] == "America/New_York"
    assert row["ready"] is True
    assert len(row["center"]) == 2


def test_readiness_names_each_missing_artifact_by_name(tmp_path, monkeypatch):
    """The three `missing.append` branches, which nothing else reaches.

    The repo's one fishery is fully processed on disk, so every other test in
    this file exercises only the ready path -- and the 409's own test
    monkeypatches `readiness.readiness` away entirely, asserting the route
    against a hand-written fake rather than against this function. Pointing
    `DATA_DIR` at an empty tmp dir is what actually runs the branches.

    Satisfied one at a time rather than all at once: a function that returned a
    fixed three-item list, or that checked the wrong file for a given label,
    would pass an all-missing assertion and fail here.
    """
    monkeypatch.setattr(readiness, "DATA_DIR", tmp_path)
    data = tmp_path / "winyah-bay"
    data.mkdir()

    r = readiness.readiness("winyah-bay")
    assert r.ready is False
    assert set(r.missing) == {
        "flow library", "along-estuary distance field", "feature inventory",
    }

    # An EMPTY flow directory is still a missing flow library -- `flow/`
    # exists as soon as anything writes into `data/<slug>/`, so `is_dir()`
    # alone would report a library that holds no states.
    (data / "flow").mkdir()
    assert "flow library" in readiness.readiness("winyah-bay").missing

    (data / "flow" / "regime0-phase00.npz").write_bytes(b"not really an npz")
    assert "flow library" not in readiness.readiness("winyah-bay").missing
    assert set(readiness.readiness("winyah-bay").missing) == {
        "along-estuary distance field", "feature inventory",
    }

    (data / "estuary_km.npy").write_bytes(b"")
    assert readiness.readiness("winyah-bay").missing == ("feature inventory",)

    (data / "features.geojson").write_text("{}")
    done = readiness.readiness("winyah-bay")
    assert done.ready is True and done.missing == ()


def test_the_409_names_what_is_missing_using_the_real_predicate(tmp_path, monkeypatch):
    """Spec §2's promise, tested end to end instead of against a fake.

    `test_api_day.py`'s 409 test substitutes `readiness.readiness` wholesale,
    so it proves the route formats *something*, not that the real predicate's
    output survives to the client. Driving the real function over an empty
    DATA_DIR closes that gap -- and asserts the other half of "one predicate,
    two callers": the `reason` in `/api/fisheries` and the 409 detail come from
    the same list, so the picker and the error cannot disagree.
    """
    from fastapi.testclient import TestClient

    from tidescout.api.app import create_app

    monkeypatch.setattr(readiness, "DATA_DIR", tmp_path)
    client = TestClient(create_app())

    r = client.get("/api/fisheries/winyah-bay/day/2026-09-03")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    for name in ("flow library", "along-estuary distance field", "feature inventory"):
        assert name in detail, detail

    row = next(x for x in client.get("/api/fisheries").json() if x["slug"] == "winyah-bay")
    assert row["ready"] is False
    assert row["reason"] == ", ".join(readiness.readiness("winyah-bay").missing)
    assert row["reason"] in detail
