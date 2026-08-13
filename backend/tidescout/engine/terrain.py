import numpy as np
from scipy import ndimage


def slope_deg(z: np.ndarray, cell_m: float) -> np.ndarray:
    """Slope in degrees via np.gradient; nodata mask == source NaNs, ~1-cell dilated."""
    gy, gx = np.gradient(z.astype("float64"), cell_m)
    s = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    # np.gradient's central difference at an interior NaN cell only reads
    # that cell's *neighbors* (i-1, i+1), never its own value, so it
    # silently yields a finite 0.0 there instead of NaN -- even though the
    # cells on either side of it do correctly come out NaN (their own
    # central difference does read the NaN cell, since a single first-
    # derivative stencil's support is exactly the 4 distance-1 axis
    # neighbors). Masking just the source cell itself therefore closes
    # slope's *only* gap, and does so completely: no cell beyond distance 1
    # ever depends on a given source cell through a single np.gradient
    # pass, so there is nothing left for a wider mask to catch. (Contrast
    # with curvature below, whose two-pass stencil has a real gap that a
    # self-cell-only mask does NOT close.)
    s[np.isnan(z)] = np.nan
    return s


def curvature(z: np.ndarray, cell_m: float) -> np.ndarray:
    """Laplacian curvature via np.gradient; nodata mask == source NaNs, 2-cell dilated."""
    gy, gx = np.gradient(z.astype("float64"), cell_m)
    gyy, _ = np.gradient(gy, cell_m)
    _, gxx = np.gradient(gx, cell_m)
    c = (gxx + gyy).astype("float32")
    # Differentiating twice makes this an effective *stride-2* stencil:
    # gyy[i,j] and gxx[i,j] algebraically reduce to
    # (z[i+2]-2z[i]+z[i-2])/4h^2 along each axis, which never references
    # the distance-1 axis neighbors or any diagonal neighbor at all. So a
    # single isolated source NaN leaves all 8 immediate neighbors (4 axis +
    # 4 diagonal) completely finite/uncontaminated in the raw output --
    # confirmed empirically, not just derived. A self-cell-only mask (the
    # naive analogue of slope_deg's fix, which is NOT the same situation --
    # slope's neighbor propagation is provably complete, curvature's is
    # not) would miss all 8 of them. Contiguous nodata blobs (e.g. the real
    # bathymetry raster's edge/mosaic gaps) stay safe only incidentally,
    # via overlapping halos between adjacent source cells -- not
    # guaranteed for thin strips or isolated dead pixels. binary_dilation
    # with iterations=2 builds a deterministic superset mask (a 2-cell
    # diamond around every source NaN) that covers the true 5-point
    # stride-2 support exactly, plus the 8 immediate neighbors besides,
    # regardless of nodata shape.
    bad = ndimage.binary_dilation(np.isnan(z), iterations=2)
    c[bad] = np.nan
    return c


def zones(
    z: np.ndarray, land_elev_m: float, shallow_max_m: float, deep_min_m: float
) -> np.ndarray:
    out = np.zeros(z.shape, dtype="uint8")
    valid = ~np.isnan(z)
    out[valid & (z >= land_elev_m)] = 1
    out[valid & (z >= shallow_max_m) & (z < land_elev_m)] = 2
    out[valid & (z >= deep_min_m) & (z < shallow_max_m)] = 3
    out[valid & (z < deep_min_m)] = 4
    return out
