import json

import anuga
import numpy as np
import pytest

from tidescout.pipeline import regimes

# Derived, never hardcoded: these tests are about "one regime failed and the
# others survived", and a literal 8/9 silently stops testing that the moment
# the matrix changes size -- which it just did, from 9 to 12.
N_REGIMES = len(regimes.REGIME_MATRIX)


class _FakeElevationQuantity:
    """Stand-in for `domain.get_quantity("elevation")` -- only the one method
    `_attach_river_inflows` calls."""

    def __init__(self, values: np.ndarray):
        self._values = values

    def get_values(self, location="centroids"):
        return self._values


class _FakeDomain:
    """Minimal stand-in for `anuga.Domain`, covering only what
    `_attach_river_inflows` touches before it would call `Inlet_operator`.
    Deliberately not a real mesh -- these tests are about the guard, not
    ANUGA itself."""

    def __init__(self, centroids: np.ndarray, elevation: np.ndarray):
        self._centroids = centroids
        self._elev = _FakeElevationQuantity(elevation)

    def get_centroid_coordinates(self, absolute=True):
        return self._centroids

    def get_quantity(self, name):
        assert name == "elevation"
        return self._elev


def test_regime_matrix_is_three_by_four():
    """Four discharge buckets since Plan 4 Task 7, not three.

    The fourth is `freshet`, the observed maximum of the composite record.
    Twelve regimes is the size of a full rebuild, so this number is also the
    compute budget -- 12 at 9 workers runs in two waves.
    """
    assert len(regimes.REGIME_MATRIX) == 12
    assert ("spring", "high") in regimes.REGIME_MATRIX
    assert ("spring", "freshet") in regimes.REGIME_MATRIX
    assert len(set(regimes.REGIME_MATRIX)) == 12


def test_the_build_axis_matches_the_lookup_axis():
    """What the library is BUILT at and how it is READ must name the same
    buckets, in the same order.

    These are deliberately two separate lists (pipeline vs pure engine), so
    nothing but this test stops one from gaining a bucket the other has never
    heard of -- which would either build regimes no lookup can reach, or index
    lookups to regimes that were never run.
    """
    from tidescout.engine.flow import DISCHARGE_ORDER

    assert regimes.DISCHARGE_BUCKETS == DISCHARGE_ORDER


def test_regime_name_is_filesystem_safe_and_unique():
    names = {regimes.regime_name(r, d) for r, d in regimes.REGIME_MATRIX}
    assert len(names) == 12
    assert all(n.replace("_", "").isalnum() for n in names)


def test_mass_residual_tolerance_is_not_machine_precision():
    """Measured residual on a real wetting/drying mesh is ~4e-4. A 1e-6 gate
    fails every healthy run -- this was hit during the Plan 3 spike."""
    from tidescout.config import load_fishery
    assert load_fishery("winyah-bay").anuga.mass_tolerance >= 1e-4


def test_initial_stage_leaves_dry_land_exactly_dry():
    """Regression for the CFL collapse: `elev + 1e-3` used to film every land
    cell with a nominal 1 mm, which collapsed ANUGA's timestep to ~1e-6 s on
    the real mesh (~129,000 of 315,564 cells filmed). Any elev-relative
    epsilon reintroduced here would show up as positive depth above `level`.
    """
    level = 0.0
    elev = np.array([-5.0, -0.5, 0.0, 0.5, 5.0])
    stage = regimes.initial_stage(elev, level)
    depth = stage - elev

    above = elev > level
    assert not (depth[above] > 0).any()

    below_or_at = ~above
    assert np.array_equal(stage[below_or_at], np.full(below_or_at.sum(), level))


def test_reversal_check_detects_a_one_way_domain(tmp_path):
    meta = {"snapshots": [{"index": 0, "t_s": 0, "phase": 0.0, "stage_bc_m": 0.0},
                          {"index": 1, "t_s": 1800, "phase": 0.1, "stage_bc_m": 0.1}]}
    (tmp_path / "regime.json").write_text(json.dumps(meta))
    for i in (0, 1):
        np.savez_compressed(
            tmp_path / f"snap_{i:03d}.npz",
            depth=np.array([5.0, 5.0], "float32"),
            u=np.array([0.4, 0.4], "float32"),   # always positive: never reverses
            v=np.zeros(2, "float32"),
        )
    assert regimes.reversal_check(tmp_path)["reversed"] is False


def test_reversal_check_detects_a_genuine_reversal(tmp_path):
    """Flood -> slack -> ebb, well past the noise floor, over enough samples."""
    meta = {"snapshots": [
        {"index": 0, "t_s": 0, "phase": 0.0, "stage_bc_m": 0.0},
        {"index": 1, "t_s": 1800, "phase": 0.1, "stage_bc_m": 0.1},
        {"index": 2, "t_s": 3600, "phase": 0.2, "stage_bc_m": 0.2},
    ]}
    (tmp_path / "regime.json").write_text(json.dumps(meta))
    for i, val in enumerate([-0.05, 0.0, 0.05]):  # >> REVERSAL_EPS_MPS (0.01)
        np.savez_compressed(
            tmp_path / f"snap_{i:03d}.npz",
            depth=np.array([5.0, 5.0], "float32"),
            u=np.array([val, val], "float32"),
            v=np.zeros(2, "float32"),
        )
    result = regimes.reversal_check(tmp_path)
    assert result["reversed"] is True
    assert result["n_samples"] == 3


def test_reversal_check_rejects_near_zero_noise(tmp_path):
    """Two samples straddling zero by ~1e-4 m/s must not read as a reversal --
    this is the exact case the review caught: reversed=True from noise near
    high water."""
    meta = {"snapshots": [
        {"index": 0, "t_s": 0, "phase": 0.0, "stage_bc_m": 1.5},
        {"index": 1, "t_s": 900, "phase": 0.02, "stage_bc_m": 1.5},
    ]}
    (tmp_path / "regime.json").write_text(json.dumps(meta))
    for i, val in enumerate([1e-4, -1e-4]):  # << REVERSAL_EPS_MPS (0.01)
        np.savez_compressed(
            tmp_path / f"snap_{i:03d}.npz",
            depth=np.array([5.0, 5.0], "float32"),
            u=np.array([val, val], "float32"),
            v=np.zeros(2, "float32"),
        )
    result = regimes.reversal_check(tmp_path)
    assert result["reversed"] is False
    assert result["n_samples"] == 2


def test_attach_river_inflows_raises_on_empty_region(fishery):
    """No mesh centroid at all falls near the inflow coordinate."""
    domain = _FakeDomain(
        centroids=np.array([[0.0, 0.0]]),
        elevation=np.array([-5.0]),
    )
    with pytest.raises(RuntimeError, match="no WET mesh centroids"):
        regimes._attach_river_inflows(domain, fishery, {})


def test_build_library_records_a_failed_regime_without_losing_others(monkeypatch, tmp_path):
    """One blown-up regime must not cost the other eight."""
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        if (r, d) == ("spring", "high"):
            raise RuntimeError("solver blew up")
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regime.json").write_text(
            json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
        )
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})

    # max_workers=1 takes the in-process serial path, so monkeypatching applies.
    results = regimes.build_library("winyah-bay", max_workers=1)

    assert results["spring_high"]["status"] == "failed"
    assert "solver blew up" in results["spring_high"]["error"]
    assert sum(v["status"] == "ok" for v in results.values()) == N_REGIMES - 1
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == N_REGIMES


def test_build_library_records_missing_regime_json_without_losing_others(
    monkeypatch, tmp_path
):
    """`run_regime` succeeds but its `regime.json` is missing on disk -- the
    exact state a killed-mid-flight build leaves behind. Post-processing must
    fail for that regime only, not abort the loop before the manifest write."""
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        if (r, d) != ("spring", "high"):
            (out / "regime.json").write_text(
                json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
            )
        # spring_high: out_dir exists, but regime.json was never written.
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})

    results = regimes.build_library("winyah-bay", max_workers=1)

    assert results["spring_high"]["status"] == "failed"
    assert sum(v["status"] == "ok" for v in results.values()) == N_REGIMES - 1
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == N_REGIMES


def test_build_library_records_corrupt_regime_json_without_losing_others(
    monkeypatch, tmp_path
):
    """`regime.json` exists but is unparseable -- e.g. truncated by a killed
    process mid-write. Must be isolated to that regime, not abort the loop."""
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        if (r, d) == ("spring", "high"):
            (out / "regime.json").write_text('{"regime": "spring_high", "snap')  # truncated
        else:
            (out / "regime.json").write_text(
                json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
            )
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})

    results = regimes.build_library("winyah-bay", max_workers=1)

    assert results["spring_high"]["status"] == "failed"
    assert sum(v["status"] == "ok" for v in results.values()) == N_REGIMES - 1
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == N_REGIMES


def test_build_library_records_reversal_check_failure_without_losing_others(
    monkeypatch, tmp_path
):
    """`regime.json` is fine, but `reversal_check` itself raises. Must be
    isolated to that regime, not abort the loop before the manifest write."""
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regime.json").write_text(
            json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
        )
        return out

    def fake_reversal_check(out_dir):
        if out_dir.name == "spring_high":
            raise ValueError("corrupt snapshot array")
        return {"reversed": True}

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", fake_reversal_check)

    results = regimes.build_library("winyah-bay", max_workers=1)

    assert results["spring_high"]["status"] == "failed"
    assert "corrupt snapshot array" in results["spring_high"]["error"]
    assert sum(v["status"] == "ok" for v in results.values()) == N_REGIMES - 1
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == N_REGIMES


def test_attach_river_inflows_raises_on_land_only_region(fishery):
    """Centroids exist in the injection box, but every one of them is dry --
    this is the gap the review found: the old check only tested `.any()`
    with no elevation filter, so a marsh-only box would pass silently."""
    from rasterio.warp import transform as warp_transform

    river = fishery.rivers[0]
    lon, lat = river.inflow_lonlat
    xs, ys = warp_transform(
        "EPSG:4326", f"EPSG:{fishery.bathymetry.epsg}", [lon], [lat]
    )
    cx, cy = xs[0], ys[0]
    domain = _FakeDomain(
        centroids=np.array([[cx, cy], [cx + 10.0, cy - 10.0]]),
        elevation=np.array([2.0, 3.0]),  # inside the box, but dry land/marsh
    )
    with pytest.raises(RuntimeError, match="no WET mesh centroids"):
        regimes._attach_river_inflows(domain, fishery, {})


def test_boundary_map_open_uses_reflective_boundary(tmp_path, monkeypatch):
    """Regression for the 2026-08-15 build #3 timestep collapse: `open` must
    map to Reflective_boundary, not Transmissive_boundary (see
    regimes._boundary_map for the measured history -- max dt 0.016-0.020 s vs
    ~0.2 s, per-regime wall time 2.7 h -> ~14 h).

    classify_boundary producing "open" from severed-channel geometry is
    already covered in test_mesh.py's three-way split tests; this test
    retags one real "wall" segment on an otherwise-normal built mesh so it
    can exercise only what _boundary_map does once the tag exists, without
    needing to author geometry that reproduces "open" from scratch and
    without running any simulation (no `evolve()` call here)."""
    from tidescout.config import load_fishery
    from tidescout.pipeline import mesh

    from .test_features_pipeline import _fake_bathy

    z = np.full((300, 300), -1.0, dtype="float32")  # shallow basin -> wall by default
    z[290:300, :] = -5.0                             # deep water along the south edge only
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    f.model_domain.ocean_boundary_utm_km = []  # see rationale in test_mesh.py's elevation test
    domain = mesh.build_mesh("winyah-bay", f)

    wall_key = next(k for k, v in domain.boundary.items() if v == "wall")
    domain.boundary[wall_key] = "open"

    boundary_map = regimes._boundary_map(domain, lambda t: 0.0)

    assert isinstance(boundary_map["open"], anuga.Reflective_boundary)
    assert isinstance(boundary_map["wall"], anuga.Reflective_boundary)
    assert set(boundary_map) == {"ocean", "wall", "open"}, (
        "wall and open must stay distinct dict keys even though both map to "
        "Reflective_boundary today"
    )


def test_build_library_reports_each_regime_as_it_finishes(monkeypatch, tmp_path):
    """A nine-regime build is a five-to-six-hour job. Reporting only at the
    end is how build #1 sat with six dead regimes for over an hour before
    anyone noticed, so `build_library` must hand each result to a callback the
    moment it is recorded -- successes and failures alike, and always the same
    dict that goes into the manifest.
    """
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        if (r, d) == ("spring", "high"):
            raise RuntimeError("solver blew up")
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regime.json").write_text(
            json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
        )
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})

    seen = []
    results = regimes.build_library(
        "winyah-bay", max_workers=1, on_result=lambda n, m: seen.append((n, m))
    )

    assert len(seen) == N_REGIMES, "every regime must be reported, not just the good ones"
    # Order is completion order (matrix order on the serial path, whichever
    # finishes first in the pool), so compare as sets rather than sequences.
    assert sorted(n for n, _ in seen) == sorted(results)
    assert dict(seen)["spring_high"]["status"] == "failed"
    # The callback must receive the recorded entry itself, so a caller cannot
    # print one thing while the manifest records another.
    for name, meta in seen:
        assert meta is results[name]


def test_build_library_without_a_callback_still_runs(monkeypatch, tmp_path):
    """`on_result` is optional -- omitting it must not change behaviour."""
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    def fake_run(slug, r, d, sim_hours=None):
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regime.json").write_text(
            json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
        )
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})

    results = regimes.build_library("winyah-bay", max_workers=1)

    assert sum(v["status"] == "ok" for v in results.values()) == N_REGIMES


def test_build_library_regimes_param_runs_only_the_given_subset(monkeypatch, tmp_path):
    """`regimes` lets a caller rebuild a subset (e.g. regimes that failed on a
    prior pass) without re-running the rest of REGIME_MATRIX. Added for the
    2026-08-23 recovery: three regimes died mid-build to a config mismatch
    and the other nine, already good, were not to be re-simulated."""
    from tidescout import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    seen_pairs = []

    def fake_run(slug, r, d, sim_hours=None):
        seen_pairs.append((r, d))
        out = regimes.regime_dir(slug) / regimes.regime_name(r, d)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regime.json").write_text(
            json.dumps({"regime": regimes.regime_name(r, d), "snapshots": []})
        )
        return out

    monkeypatch.setattr(regimes, "run_regime", fake_run)
    monkeypatch.setattr(regimes, "reversal_check", lambda d: {"reversed": True})

    subset = [("spring", "med"), ("spring", "high"), ("spring", "freshet")]
    results = regimes.build_library("winyah-bay", max_workers=1, regimes=subset)

    assert sorted(seen_pairs) == sorted(subset)
    assert sorted(results) == ["spring_freshet", "spring_high", "spring_med"]
    assert all(v["status"] == "ok" for v in results.values())
    # The nine other regimes' full-matrix coverage (regimes=None) is already
    # exercised by the tests above this one -- they all call build_library()
    # with no `regimes` argument and assert N_REGIMES entries land.


def test_store_sww_knob_defaults_on_and_can_be_disabled():
    from tidescout.models import AnugaConfig

    assert AnugaConfig().store_sww is True
    assert AnugaConfig(store_sww=False).store_sww is False
