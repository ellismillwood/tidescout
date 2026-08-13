import numpy as np
from rasterio.transform import from_origin

from tidescout.engine.render import contour_lines, depth_rgba, hillshade


def _cone(size=80, depth=-10.0):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2, yy - size / 2)
    z = (depth * (1 - r / (size / 2))).astype("float32")
    return np.minimum(z, 0.0)


def test_hillshade_range_and_nan():
    z = _cone()
    z[0, 0] = np.nan
    hs = hillshade(z, cell_m=10.0)
    assert hs.dtype == np.uint8
    assert hs[0, 0] == 0
    assert 0 < hs[40, 20] <= 255


def test_depth_rgba_shape_and_alpha():
    z = np.array([[np.nan, -5.0, 2.0]], dtype="float32")
    rgba = depth_rgba(z, deep_min_m=-3.0, land_elev_m=1.5)
    assert rgba.shape == (1, 3, 4)
    assert rgba[0, 0, 3] == 0  # nan transparent
    assert rgba[0, 1, 3] == 255  # water opaque
    assert rgba[0, 2, 3] == 255  # land opaque


def test_contours_on_cone():
    z = _cone()
    transform = from_origin(500000, 3690000, 10, 10)  # UTM-ish coords
    lines = contour_lines(z, transform, crs_epsg=26917, depths_m=[-5.0, -2.0])
    depths = {round(li["depth_m"], 1) for li in lines}
    assert depths == {-5.0, -2.0}
    for li in lines:
        assert len(li["coords"]) >= 5
        for lon, lat in li["coords"]:
            assert -180 <= lon <= 180 and -90 <= lat <= 90
