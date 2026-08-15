"""Mesh snapshots -> gridded flow-state library.

Stored on a coarser grid than the 10 m analysis raster and masked to the model
domain: the naive full-grid float32 layout would be ~52 GB for nine regimes,
this is ~1.8 GB. u/v are stored rather than speed/direction because direction
wraps at 360 degrees and cannot be interpolated -- averaging 359.9 deg and
0.1 deg gives 180 deg, pointing the flow backwards.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from scipy.spatial import cKDTree

from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import mesh
from tidescout.pipeline.bathy import read_bathy


@dataclass
class GridSpec:
    shape: tuple[int, int]
    transform: Affine
    cell_m: float
    flat_index: np.ndarray  # indices into the flattened grid that are in-domain
    xs: np.ndarray  # in-domain cell-centre eastings
    ys: np.ndarray  # in-domain cell-centre northings


def grid_spec(slug: str, fishery: Fishery) -> GridSpec:
    """The library grid: the model domain, decimated to `library_cell_m`.

    The mask MUST be built with the same arguments `mesh.build_mesh` uses.
    `clean_mask` defaults `min_island_hole_km2` and `pixel_area_m2`, and those
    defaults happen to match this fishery -- but a fishery with a different
    `bathymetry.cell_m` or `model_domain.min_island_hole_km2` would silently
    get a library grid covering a different area than the mesh it samples,
    with no error anywhere. Pass them explicitly.
    """
    z, transform, _ = read_bathy(slug)
    md = fishery.model_domain
    cell = fishery.anuga.library_cell_m
    step = int(round(cell / fishery.bathymetry.cell_m))
    if step < 1:
        raise ValueError(
            f"library_cell_m ({cell} m) is finer than bathymetry.cell_m "
            f"({fishery.bathymetry.cell_m} m) -- the library cannot invent "
            "detail the source raster does not have"
        )
    pixel_area_m2 = abs(transform.a) * abs(transform.e)
    mask = mesh.clean_mask(
        mesh.domain_mask(z, transform, fishery),
        md.clean_cells,
        md.min_island_hole_km2,
        pixel_area_m2,
    )[::step, ::step]
    lib_tf = transform * Affine.scale(step, step)
    rows, cols = np.nonzero(mask)
    # +0.5 puts us at cell centres, matching the project's pixel convention
    xs, ys = lib_tf * (cols + 0.5, rows + 0.5)
    return GridSpec(
        mask.shape,
        lib_tf,
        cell,
        np.ravel_multi_index((rows, cols), mask.shape),
        xs,
        ys,
    )


def nearest_sample(centroids: np.ndarray, values: np.ndarray, targets: np.ndarray):
    """Nearest-centroid lookup.

    The mesh is finer than the library grid almost everywhere, so
    nearest-neighbour is adequate and ~10x cheaper than proper barycentric
    interpolation.
    """
    tree = cKDTree(centroids)
    _, idx = tree.query(targets, k=1)
    return values[idx]


def speed_direction(u: np.ndarray, v: np.ndarray):
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return speed, direction


def _xy_gradients(a: np.ndarray, cell_m: float):
    """d/dx (east) and d/dy (north) of a grid-shaped field.

    `np.gradient` returns the ROW derivative first, and on a north-up raster
    rows run SOUTH while u/v are true east/north. So the row derivative is
    -d/dy, not d/dy. Getting this wrong is not cosmetic: it turns the
    stretching term `du_dx - dv_dy` into the divergence `du_dx + dv_dy`, which
    is a completely different field.
    """
    d_drow, d_dcol = np.gradient(a, cell_m)
    return d_dcol, -d_drow


def shear_magnitude(u: np.ndarray, v: np.ndarray, cell_m: float) -> np.ndarray:
    """Strain-rate magnitude -- the spec's 'seams' signal.

    Fish hold on the boundary between fast and slow water, so what matters is
    neighbouring parcels sliding past one another, not the speed itself and
    not the raw size of the velocity gradient.

    This is the total deformation rate,
        sqrt((du_dx - dv_dy)^2 + (du_dy + dv_dx)^2),
    the standard stretching/shearing pair. Deliberately NOT
    sqrt(du_dy^2 + dv_dx^2 + ...): that returns 1.41*omega for solid-body
    rotation, which has no seam anywhere in it -- neighbouring parcels in a
    rigidly turning eddy never slide past one another -- and would light up
    the interior of every eddy in the bay as holding water. The strain rate is
    also Galilean invariant, so a seam reads the same whether the whole body of
    water is drifting past it or not; isotropic expansion likewise deforms no
    parcel's shape and correctly returns zero.
    """
    du_dx, du_dy = _xy_gradients(u, cell_m)
    dv_dx, dv_dy = _xy_gradients(v, cell_m)
    stretching = du_dx - dv_dy
    shearing = du_dy + dv_dx
    return np.sqrt(stretching**2 + shearing**2)


def _grid_json(spec: GridSpec, snapshots: list[dict], transform6: list) -> dict:
    """The sidecar describing the flat array layout.

    The stored arrays are 1-D and masked, so without this a reader cannot put
    a value back on the map. `flat_index_len` is recorded separately from
    `n_cells` as a self-check: they must agree, and a mismatch means the
    arrays and the index were written from different specs.
    """
    return {
        "shape": list(spec.shape),
        "cell_m": spec.cell_m,
        "transform": transform6,
        "n_cells": int(spec.flat_index.size),
        "flat_index_len": int(spec.flat_index.size),
        "phases": [s["phase"] for s in snapshots],
    }


def rasterise_regime(slug: str, fishery: Fishery, regime: str, spec: GridSpec) -> Path:
    src = fishery_data_dir(slug) / "flow" / regime
    meta = json.loads((src / "regime.json").read_text())
    domain = mesh.build_mesh(slug, fishery)  # same mesh: deterministic from config
    centroids = domain.get_centroid_coordinates(absolute=True)
    targets = np.column_stack([spec.xs, spec.ys])
    tree = cKDTree(centroids)
    _, idx = tree.query(targets, k=1)

    out = src / "grid"
    out.mkdir(exist_ok=True)
    for snap in meta["snapshots"]:
        d = np.load(src / f"snap_{snap['index']:03d}.npz")
        np.savez_compressed(
            out / f"phase_{snap['index']:03d}.npz",
            u=d["u"][idx].astype("float32"),
            v=d["v"][idx].astype("float32"),
            depth=d["depth"][idx].astype("float32"),
            phase=np.float32(snap["phase"]),
        )
    # The flat index is what maps these 1-D arrays back onto the raster. It is
    # identical for every regime, but written per-regime anyway: a regime
    # directory that cannot be read without a file from somewhere else is a
    # trap for exactly the partial-build states this pipeline keeps producing.
    np.save(out / "flat_index.npy", spec.flat_index.astype("int64"))
    (out / "grid.json").write_text(
        json.dumps(_grid_json(spec, meta["snapshots"], list(spec.transform)[:6]), indent=2)
    )
    return out


def load_state(slug: str, regime: str, phase_index: int) -> dict:
    p = (
        fishery_data_dir(slug)
        / "flow"
        / regime
        / "grid"
        / f"phase_{phase_index:03d}.npz"
    )
    if not p.exists():
        # Callers index phases by number, so a bare FileNotFoundError on a long
        # path makes them decode which regime and phase actually went missing.
        raise FileNotFoundError(
            f"no rasterised state for regime {regime!r} phase {phase_index} "
            f"(expected {p}) -- has `rasterise_regime` run for this regime?"
        )
    d = np.load(p)
    return {"u": d["u"], "v": d["v"], "depth": d["depth"], "phase": float(d["phase"])}
