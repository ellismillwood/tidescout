import json

import numpy as np
import pytest

from tidescout.pipeline import flowlib


def test_nearest_centroid_rasterisation_matches_known_values():
    centroids = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([[5.0, 5.0], [95.0, 5.0], [5.0, 95.0]])
    out = flowlib.nearest_sample(centroids, values, targets)
    assert list(out) == [1.0, 2.0, 3.0]


def test_direction_is_derived_not_stored():
    u = np.array([1.0, 0.0, -1.0])
    v = np.array([0.0, 1.0, 0.0])
    speed, direction = flowlib.speed_direction(u, v)
    assert np.allclose(speed, [1.0, 1.0, 1.0])
    assert np.allclose(direction, [0.0, 90.0, 180.0])


def test_shear_is_zero_in_uniform_flow():
    u = np.full((20, 20), 0.5)
    v = np.zeros((20, 20))
    shear = flowlib.shear_magnitude(u, v, cell_m=20.0)
    assert np.nanmax(np.abs(shear)) < 1e-9


def test_shear_is_zero_in_solid_body_rotation():
    """Rotating water is not a seam.

    A whole eddy turning as a rigid disc has no fast-water/slow-water boundary
    in it at all -- neighbouring parcels never slide past one another. Only the
    strain rate captures that; the raw velocity-gradient magnitude
    sqrt(du_dy^2 + dv_dx^2) returns 1.41*omega here and would light up the
    interior of every eddy in the bay as holding water.
    """
    n, cell = 21, 20.0
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # rows run south on a north-up raster, so northing decreases with row
    x = (xx - c) * cell
    y = (c - yy) * cell
    omega = 0.01
    u, v = -omega * y, omega * x

    shear = flowlib.shear_magnitude(u, v, cell_m=cell)

    assert np.nanmax(np.abs(shear[1:-1, 1:-1])) < 1e-9


def test_shear_recovers_the_rate_of_a_pure_shear_flow():
    """u = a*y, v = 0 has strain rate exactly a, whatever the grid spacing."""
    n, cell, a = 21, 20.0, 0.004
    yy, _ = np.mgrid[0:n, 0:n].astype(float)
    y = (n // 2 - yy) * cell
    u = a * y
    v = np.zeros_like(u)

    shear = flowlib.shear_magnitude(u, v, cell_m=cell)

    assert np.allclose(shear[1:-1, 1:-1], a, rtol=1e-9)


def test_shear_ignores_pure_divergence_sign_convention():
    """Regression guard for the row-vs-northing axis flip.

    `np.gradient` returns the ROW derivative first, and rows run south while
    `v` is true-north. Naming that row derivative `du_dy` without flipping its
    sign turns the stretching term `du_dx - dv_dy` into the divergence, so a
    purely diverging flow (which has real strain) and a purely rotating one
    (which has none) both come out wrong. Pin the diverging case explicitly.
    """
    n, cell, k = 21, 20.0, 0.003
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    c = n // 2
    x = (xx - c) * cell
    y = (c - yy) * cell
    u, v = k * x, k * y  # du_dx = dv_dy = k -> stretching 0, shearing 0

    shear = flowlib.shear_magnitude(u, v, cell_m=cell)

    # Isotropic expansion deforms no parcel's shape: strain magnitude is zero
    # even though the divergence is 2k.
    assert np.nanmax(np.abs(shear[1:-1, 1:-1])) < 1e-9


def test_speed_direction_is_stable_across_the_360_wrap():
    """Why u/v are stored rather than speed/direction."""
    u = np.array([1.0, 1.0])
    v = np.array([-1e-9, 1e-9])
    _, direction = flowlib.speed_direction(u, v)
    assert direction[0] > 359.0 and direction[1] < 1.0
    # ...and the mean of those two directions is 180 degrees wrong, which is
    # exactly the interpolation failure the u/v storage choice avoids.
    assert abs(direction.mean() - 180.0) < 1.0


def test_load_state_round_trips_a_written_regime(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay") / "flow" / "mean_med" / "grid"
    d.mkdir(parents=True)
    np.savez_compressed(
        d / "phase_003.npz",
        u=np.array([1.0, 2.0], dtype="float32"),
        v=np.array([3.0, 4.0], dtype="float32"),
        depth=np.array([5.0, 6.0], dtype="float32"),
        phase=np.float32(0.25),
    )

    state = flowlib.load_state("winyah-bay", "mean_med", 3)

    assert np.allclose(state["u"], [1.0, 2.0])
    assert np.allclose(state["depth"], [5.0, 6.0])
    assert state["phase"] == pytest.approx(0.25)


def test_load_state_names_the_missing_file(tmp_path, monkeypatch):
    """A missing phase must say which regime and phase, not raise a bare
    FileNotFoundError on a path the caller has to decode."""
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")

    with pytest.raises(FileNotFoundError, match="mean_med.*phase 7"):
        flowlib.load_state("winyah-bay", "mean_med", 7)


def test_grid_json_records_what_a_reader_needs(tmp_path, monkeypatch):
    """`grid.json` is the only description of the flat array layout, so it must
    carry enough to rebuild the mapping back onto the raster."""
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    spec = flowlib.GridSpec(
        shape=(4, 5),
        transform=None,
        cell_m=20.0,
        flat_index=np.array([0, 6, 12]),
        xs=np.array([0.0, 1.0, 2.0]),
        ys=np.array([0.0, 1.0, 2.0]),
    )
    payload = flowlib._grid_json(spec, [{"index": 0, "phase": 0.0}], transform6=[1, 0, 0, 0, 1, 0])
    got = json.loads(json.dumps(payload))

    assert got["shape"] == [4, 5]
    assert got["n_cells"] == 3
    assert got["flat_index_len"] == 3
    assert got["phases"] == [0.0]
