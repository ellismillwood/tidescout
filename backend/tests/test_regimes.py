import json

import numpy as np
import pytest

from tidescout.pipeline import regimes


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


def test_regime_matrix_is_three_by_three():
    assert len(regimes.REGIME_MATRIX) == 9
    assert ("spring", "high") in regimes.REGIME_MATRIX
    assert len(set(regimes.REGIME_MATRIX)) == 9


def test_regime_name_is_filesystem_safe_and_unique():
    names = {regimes.regime_name(r, d) for r, d in regimes.REGIME_MATRIX}
    assert len(names) == 9
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
    assert sum(v["status"] == "ok" for v in results.values()) == 8
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == 9


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
    assert sum(v["status"] == "ok" for v in results.values()) == 8
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == 9


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
    assert sum(v["status"] == "ok" for v in results.values()) == 8
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == 9


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
    assert sum(v["status"] == "ok" for v in results.values()) == 8
    manifest = json.loads((regimes.regime_dir("winyah-bay") / "library.json").read_text())
    assert len(manifest["regimes"]) == 9


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
