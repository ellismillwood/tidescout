from pathlib import Path

import numpy as np
import rasterio

from tidescout.engine import terrain
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy


def _write(path: Path, arr: np.ndarray, template_meta: dict, dtype: str, nodata) -> None:
    t = template_meta["transform"]
    with rasterio.open(
        path, "w", driver="GTiff", height=template_meta["height"], width=template_meta["width"],
        count=1, dtype=dtype, crs=template_meta["crs"],
        transform=rasterio.transform.Affine(t[0], t[1], t[2], t[3], t[4], t[5]),
        nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def build_derivatives(slug: str, fishery: Fishery) -> dict[str, Path]:
    """Write slope/curv/zones next to the source raster (same grid/CRS); nodata
    masks relative to the source: zones == source, slope ⊇ source by ~1 cell,
    curv ⊇ source by 2 cells (see engine.terrain for why each differs)."""
    z, _, meta = read_bathy(slug)
    d = fishery_data_dir(slug)
    s = terrain.slope_deg(z, fishery.bathymetry.cell_m)
    c = terrain.curvature(z, fishery.bathymetry.cell_m)
    zn = terrain.zones(
        z, fishery.bathymetry.land_elev_m,
        fishery.bathymetry.zone_shallow_max_m, fishery.bathymetry.zone_deep_min_m,
    )
    s_out = np.where(np.isnan(s), -9999.0, s)
    c_out = np.where(np.isnan(c), -9999.0, c)
    paths = {
        "slope": d / "slope.tif", "curv": d / "curv.tif", "zones": d / "zones.tif",
    }
    _write(paths["slope"], s_out, meta, "float32", -9999.0)
    _write(paths["curv"], c_out, meta, "float32", -9999.0)
    _write(paths["zones"], zn, meta, "uint8", 0)
    return paths
