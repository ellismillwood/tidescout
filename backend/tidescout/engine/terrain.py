import numpy as np


def slope_deg(z: np.ndarray, cell_m: float) -> np.ndarray:
    gy, gx = np.gradient(z.astype("float64"), cell_m)
    s = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    # np.gradient's central difference at an interior NaN cell only reads
    # that cell's *neighbors* (i-1, i+1), never its own value, so it
    # silently yields a finite 0.0 there instead of NaN -- even though the
    # cells on either side of it do correctly come out NaN (their own
    # central difference does read the NaN cell). Force the documented
    # "NaN in -> NaN out" contract explicitly rather than relying on
    # np.gradient's incidental (and incomplete) propagation.
    s[np.isnan(z)] = np.nan
    return s


def curvature(z: np.ndarray, cell_m: float) -> np.ndarray:
    gy, gx = np.gradient(z.astype("float64"), cell_m)
    gyy, _ = np.gradient(gy, cell_m)
    _, gxx = np.gradient(gx, cell_m)
    c = (gxx + gyy).astype("float32")
    # Same NaN-at-its-own-cell gap as slope_deg above, and it matters here
    # too: pipeline.derivatives._write masks nodata via np.isnan(c), which
    # only works if a NaN input cell reliably produces a NaN output cell.
    c[np.isnan(z)] = np.nan
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
