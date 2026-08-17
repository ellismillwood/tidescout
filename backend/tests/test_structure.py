"""Analytic flow fields with known answers.

Every test here builds a velocity field whose derived structure can be worked
out on paper, so a failure localises to the formula rather than to the data.
"""

import numpy as np

from tidescout.engine import structure


def _xy(n=64, cell=20.0):
    """Centred coordinate grids in metres, north-up (row 0 is the NORTH edge)."""
    c = (np.arange(n) - n / 2) * cell
    x = np.tile(c, (n, 1))
    y = np.tile(-c[:, None], (1, n))  # rows run south, so y decreases downward
    return x, y, cell


def test_grid_round_trip_restores_values_at_their_cells():
    shape = (5, 4)
    flat_index = np.array([0, 6, 19])
    values = np.array([1.5, -2.0, 7.25])
    grid = structure.to_grid(values, flat_index, shape)

    assert grid.shape == shape
    assert grid.ravel()[0] == 1.5
    assert grid.ravel()[6] == -2.0
    assert grid.ravel()[19] == 7.25
    assert np.isnan(grid.ravel()[1])  # out-of-domain stays NaN, never 0.0
    assert np.array_equal(structure.from_grid(grid, flat_index), values)


def test_zero_fill_is_not_the_default_because_zero_is_a_real_speed():
    """0.0 m/s is slack water; NaN is 'not in the domain'. Conflating them
    would make every land cell look like a stagnant ambush pocket."""
    grid = structure.to_grid(np.array([1.0]), np.array([3]), (2, 2))
    assert np.isnan(grid[0, 0])


def test_xy_gradients_return_true_east_and_north_derivatives():
    """np.gradient gives the ROW derivative first, and rows run SOUTH."""
    x, y, cell = _xy()
    d_dx, d_dy = structure.xy_gradients(3.0 * x, cell)
    assert np.allclose(d_dx[1:-1, 1:-1], 3.0)
    assert np.allclose(d_dy[1:-1, 1:-1], 0.0, atol=1e-9)

    d_dx, d_dy = structure.xy_gradients(3.0 * y, cell)
    assert np.allclose(d_dx[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(d_dy[1:-1, 1:-1], 3.0)  # NOT -3.0


def test_gradient_tensor_recovers_a_known_pure_shear():
    x, y, cell = _xy()
    a = 0.004
    t = structure.gradient_tensor(a * y, np.zeros_like(y), cell)
    assert np.allclose(t.du_dy[1:-1, 1:-1], a)
    assert np.allclose(t.du_dx[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(t.dv_dx[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(t.dv_dy[1:-1, 1:-1], 0.0, atol=1e-9)
    assert np.allclose(structure.strain_rate(t)[1:-1, 1:-1], a)


def test_strain_rate_is_zero_in_solid_body_rotation():
    """A rigidly turning eddy deforms no parcel, so it is not a seam --
    neighbouring water never slides past itself. Task 4 is what finds it."""
    x, y, cell = _xy()
    omega = 0.002
    t = structure.gradient_tensor(-omega * y, omega * x, cell)
    assert np.nanmax(np.abs(structure.strain_rate(t)[1:-1, 1:-1])) < 1e-9


def test_strain_rate_ignores_isotropic_expansion():
    x, y, cell = _xy()
    k = 0.003
    t = structure.gradient_tensor(k * x, k * y, cell)
    assert np.nanmax(np.abs(structure.strain_rate(t)[1:-1, 1:-1])) < 1e-9


def test_strain_rate_is_galilean_invariant():
    """A seam reads the same whether the whole bay is drifting past it."""
    x, y, cell = _xy()
    a = 0.004
    still = structure.strain_rate(structure.gradient_tensor(a * y, np.zeros_like(y), cell))
    drift = structure.strain_rate(
        structure.gradient_tensor(a * y + 0.7, np.full_like(y, -0.3), cell)
    )
    assert np.allclose(still[1:-1, 1:-1], drift[1:-1, 1:-1])
