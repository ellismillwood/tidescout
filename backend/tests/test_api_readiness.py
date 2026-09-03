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
