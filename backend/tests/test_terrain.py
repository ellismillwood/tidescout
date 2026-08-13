import numpy as np

from tidescout.engine.terrain import curvature, slope_deg, zones


def test_slope_on_inclined_plane():
    # z drops 1 m per cell along x; cell 10 m -> slope = atan(0.1) = 5.71 deg
    z = np.tile(np.arange(50, dtype="float32") * -1.0, (50, 1))
    s = slope_deg(z, cell_m=10.0)
    interior = s[5:-5, 5:-5]
    assert np.allclose(interior, np.degrees(np.arctan(0.1)), atol=0.01)


def test_slope_nan_safe():
    z = np.zeros((20, 20), dtype="float32")
    z[10, 10] = np.nan
    s = slope_deg(z, 10.0)
    assert np.isnan(s[10, 10])


def test_curvature_sign_in_pit():
    yy, xx = np.mgrid[0:41, 0:41]
    z = (((xx - 20) ** 2 + (yy - 20) ** 2) / 100.0).astype("float32")  # bowl, min at center
    c = curvature(z, 10.0)
    assert c[20, 20] > 0  # concave up at the pit bottom


def test_zones_bands():
    z = np.array([[2.0, 0.0, -1.0, -5.0, np.nan]], dtype="float32")
    out = zones(z, land_elev_m=1.5, shallow_max_m=-0.3, deep_min_m=-3.0)
    assert out.tolist() == [[1, 2, 3, 4, 0]]
