import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

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


def test_hillshade_lights_the_azimuth_facing_slope():
    # Ground truth, derived independently (not from either candidate formula):
    # a plane rising toward SE in array coords (rows increase southward, cols
    # increase eastward) has its downhill flank facing NW. With the default
    # az=315 (NW) light, that NW-facing flank must render bright; flipping the
    # plane (rises toward NW instead) points the same flank SE -- away from
    # the light -- and must render dark.
    #
    # coef=5.0 (~35 deg slope) rather than a shallower plane: verified by
    # direct sweep that a shallow ~4 deg plane (as first tried) makes
    # nw>170/se<80/margin>60 unreachable by ANY formula, correct or not --
    # max possible byte range there is [167,193], a 26-level spread, because
    # the aspect-dependent term is scaled by sin(slope)~0.07. A realistic
    # 30-40 deg slope is required before these thresholds mean anything.
    z = (np.add.outer(np.arange(60), np.arange(60)) * 5.0).astype("float32")
    nw_facing = hillshade(z, 10.0)  # az default 315 -> NW-facing flank, should be bright
    se_facing = hillshade(z[::-1, ::-1].copy(), 10.0)  # flipped -> faces away from light
    assert nw_facing[30, 30] > 170
    assert se_facing[30, 30] < 80
    assert int(nw_facing[30, 30]) > int(se_facing[30, 30]) + 60


def _snap_near_integer(v: float, eps: float = 1e-6) -> float:
    """Round v to the nearest integer if it's within eps of one, else leave it.

    Every `find_contours` vertex has row OR col exactly on an integer grid
    line (that's what a marching-squares edge crossing is), by construction.
    Reprojecting such a vertex out to EPSG:4326 and back through
    `rasterio.warp.transform` (as this test does, to verify the function's
    actual output rather than its internals) round-trips through PROJ and
    reintroduces ~1e-10 floating-point noise around that exact integer --
    real, but ~9 orders of magnitude below any genuine fractional crossing
    (which differ from an integer by at least ~1e-3 on this grid). Left
    unsnapped, `ceil()` on a value like 5017.000000000046 overshoots to 5018
    instead of 5017, silently widening the checked neighborhood by a full
    cell -- verified directly against the real winyah-bay output: 11 of
    171,248 points looked like violations through this round trip, but 0 of
    186,209 raw (pre-reprojection) vertices actually violate the fix's own
    2x2 check when inspected directly from inside contour_lines. Snapping
    here makes the test check what contour_lines actually computed, not an
    artifact of this test's own verification path.
    """
    r = round(v)
    return float(r) if abs(v - r) < eps else v


def test_contours_drop_nodata_boundary_artifacts():
    # The reviewer's suggested `z[:, :6] = np.nan` turns out too narrow to
    # matter for this cone: the -5/-2 circles' leftmost extents are col 20
    # and col 8, and empirically a 6-col stripe already returns zero points
    # near the stripe under the PRE-fix code -- that would make this test
    # pass vacuously regardless of whether the fix works.
    #
    # A 15-col stripe reaches deep into the -2 circle (leftmost col 8, well
    # inside cols 0-14) while leaving the -5 circle (leftmost col exactly
    # 20) 5 columns of real clearance -- verified pre-fix, this returns 39
    # near-stripe points out of 390. Deliberately NOT col 20 (exactly the -5
    # circle's radius): z[40,20] == -5.0 exactly there, and col 20 is also
    # exactly the stripe boundary, so that specific vertex's already-exact
    # integer column sits precisely at the real/nodata edge -- a needlessly
    # fragile combination on top of the general float-noise issue
    # `_snap_near_integer` handles. 5 columns of clearance avoids stacking
    # both fragilities in one test.
    z = _cone()
    z[:, :15] = np.nan
    transform = from_origin(500000, 3690000, 10, 10)
    inv = ~transform
    lines = contour_lines(z, transform, crs_epsg=26917, depths_m=[-5.0, -2.0])
    depths = {round(li["depth_m"], 1) for li in lines}
    assert depths == {-5.0, -2.0}  # real contours still come out for both depths
    for li in lines:
        assert len(li["coords"]) >= 5
        for lon, lat in li["coords"]:
            xs, ys = warp_transform("EPSG:4326", "EPSG:26917", [lon], [lat])
            col_f, row_f = inv * (xs[0], ys[0])
            col = _snap_near_integer(col_f - 0.5)  # undo contour_lines' pixel-center +0.5 shift
            row = _snap_near_integer(row_f - 0.5)
            c0, c1 = max(int(np.floor(col)), 0), min(int(np.ceil(col)), z.shape[1] - 1)
            r0, r1 = max(int(np.floor(row)), 0), min(int(np.ceil(row)), z.shape[0] - 1)
            assert not np.isnan(z[r0 : r1 + 1, c0 : c1 + 1]).any(), (
                f"coord ({lon},{lat}) -> pixel neighborhood "
                f"rows[{r0}:{r1}] cols[{c0}:{c1}] touches the NaN stripe"
            )
