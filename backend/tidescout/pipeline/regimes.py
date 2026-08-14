"""Regime simulations: the 3x3 tidal-range x discharge matrix.

Each regime is one full tidal cycle plus spin-up, snapshotted at the configured
cadence. Runs are completely independent, which is what lets Task 10 execute
them as parallel OS processes rather than reaching for MPI.
"""

import json
import time
from pathlib import Path

import anuga
import numpy as np

from tidescout.config import load_fishery
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import forcing, mesh

RANGE_BUCKETS = ["neap", "mean", "spring"]
DISCHARGE_BUCKETS = ["low", "med", "high"]
REGIME_MATRIX = [(r, d) for r in RANGE_BUCKETS for d in DISCHARGE_BUCKETS]


def regime_name(range_bucket: str, discharge_bucket: str) -> str:
    return f"{range_bucket}_{discharge_bucket}"


def regime_dir(slug: str) -> Path:
    d = fishery_data_dir(slug) / "flow"
    d.mkdir(parents=True, exist_ok=True)
    return d


def initial_stage(elev: np.ndarray, level: float) -> np.ndarray:
    """Free surface at rest: water to `level`, and dry land genuinely dry.

    Deliberately NOT `elev + eps`. A nominal film over land collapses ANUGA's
    CFL timestep to ~1e-6 s on real bathymetry (measured: ~129,000 filmed
    cells of 315,564, run aborts in the first yieldstep). Cells above `level`
    must start at depth exactly 0.
    """
    return np.maximum(elev, level)


def mass_residual(domain, v0: float) -> float:
    """Relative closure of ANUGA's volume identity.

    dV must equal boundary flux plus fractional-step (inflow) volume. Returns
    the residual normalised by the volume actually moved.
    """
    v1 = domain.get_water_volume()
    flux = domain.get_boundary_flux_integral()
    frac = domain.get_fractional_step_volume_integral()
    moved = max(abs(v1 - v0), 1.0)
    return abs((v1 - v0) - (flux + frac)) / moved


def _centroid_speed(domain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth, u, v) at centroids, zeroed where dry."""
    stage = domain.get_quantity("stage").get_values(location="centroids")
    elev = domain.get_quantity("elevation").get_values(location="centroids")
    xmom = domain.get_quantity("xmomentum").get_values(location="centroids")
    ymom = domain.get_quantity("ymomentum").get_values(location="centroids")
    depth = stage - elev
    wet = depth > 0.01
    safe = np.where(wet, depth, 1.0)
    u = np.where(wet, xmom / safe, 0.0)
    v = np.where(wet, ymom / safe, 0.0)
    return depth, u, v


def run_regime(
    slug: str, range_bucket: str, discharge_bucket: str, sim_hours: float | None = None
) -> Path:
    """Run one regime; write snapshots and a per-regime metadata JSON."""
    fishery: Fishery = load_fishery(slug)
    cfg = fishery.anuga
    name = regime_name(range_bucket, discharge_bucket)
    out_dir = regime_dir(slug) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = mesh.build_mesh(slug, fishery)
    domain.set_name(name)
    domain.set_datadir(str(out_dir))
    domain.set_quantity(
        "friction", mesh.friction_field(domain, slug, fishery), location="centroids"
    )

    elev = domain.get_quantity("elevation").get_values(location="centroids")
    tide = forcing.range_scaled_tide(
        cfg.mean_range_m, range_bucket, period_s=cfg.cycle_h * 3600.0
    )
    # Start at the initial boundary level so spin-up is not a dam break.
    domain.set_quantity(
        "stage", initial_stage(elev, tide(0.0)), location="centroids"
    )
    domain.set_boundary({
        "outer": anuga.Transmissive_momentum_set_stage_boundary(
            domain=domain, function=tide
        )
    })

    inflows = forcing.river_inflow_m3s(fishery, discharge_bucket)
    _attach_river_inflows(domain, fishery, inflows)

    total_h = sim_hours if sim_hours is not None else cfg.spin_up_h + cfg.cycle_h
    yieldstep = cfg.snapshot_minutes * 60.0
    v0 = domain.get_water_volume()

    snaps = []
    t_start = time.time()
    for t in domain.evolve(yieldstep=yieldstep, finaltime=total_h * 3600.0):
        if t < cfg.spin_up_h * 3600.0:
            continue  # discard spin-up; it is not a physical state
        depth, u, v = _centroid_speed(domain)
        phase = ((t - cfg.spin_up_h * 3600.0) / (cfg.cycle_h * 3600.0)) % 1.0
        idx = len(snaps)
        np.savez_compressed(
            out_dir / f"snap_{idx:03d}.npz",
            depth=depth.astype("float32"),
            u=u.astype("float32"),
            v=v.astype("float32"),
        )
        snaps.append({"index": idx, "t_s": float(t), "phase": float(phase),
                      "stage_bc_m": float(tide(t))})

    meta = {
        "regime": name,
        "range_bucket": range_bucket,
        "discharge_bucket": discharge_bucket,
        "triangles": int(len(domain.triangles)),
        "sim_hours": total_h,
        "wall_seconds": round(time.time() - t_start, 1),
        "mass_residual": float(mass_residual(domain, v0)),
        "inflows_m3s": inflows,
        "snapshots": snaps,
    }
    (out_dir / "regime.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def _attach_river_inflows(domain, fishery: Fishery, inflows: dict[str, float]) -> None:
    """Push each river's discharge in as an inlet at its up-estuary boundary.

    Verified against the installed ANUGA build (Task 9):
        anuga.Inlet_operator(domain, region, Q=0.0, velocity=None,
            zero_velocity=False, default=0.0, description=None, label=None,
            logging=False, verbose=False)
    matching this call's usage (positional domain/region, keyword Q).
    """
    from rasterio.warp import transform as warp_transform

    epsg = fishery.bathymetry.epsg
    centroids = domain.get_centroid_coordinates(absolute=True)
    for river in fishery.rivers:
        seed = getattr(river, "inflow_lonlat", None)
        if seed is None:
            continue
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [seed[0]], [seed[1]])
        cx, cy = xs[0], ys[0]
        r = 150.0
        region = [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]
        # A coordinate that misses the meshed water body is the most
        # dangerous silent failure in this whole module: Inlet_operator
        # itself dies deep inside ANUGA with an unrelated AttributeError
        # ('Inlet' object has no attribute 'inlet_line') when its region
        # contains zero centroids, but a region that contains a handful of
        # *land* centroids would not raise at all -- the run would finish,
        # regime.json would look plausible, and this river's discharge axis
        # would be silently meaningless. Fail loudly and specifically here
        # instead of relying on that downstream crash (or worse, no crash).
        in_region = (
            (centroids[:, 0] >= cx - r) & (centroids[:, 0] <= cx + r)
            & (centroids[:, 1] >= cy - r) & (centroids[:, 1] <= cy + r)
        )
        if not in_region.any():
            raise RuntimeError(
                f"river inflow for {river.name!r} has no mesh centroids within "
                f"{r:.0f} m of inflow_lonlat={seed} (utm=({cx:.0f}, {cy:.0f})) -- "
                "the coordinate does not land in the meshed water body, so this "
                "river's discharge axis would be silently dropped"
            )
        anuga.Inlet_operator(domain, region, Q=inflows[river.name])


def reversal_check(out_dir: Path) -> dict:
    """Flow must reverse direction across a tidal cycle.

    A domain that only ever drains (or only fills) means the boundary never
    drove it -- the most likely silent failure in the whole pipeline.
    """
    meta = json.loads((out_dir / "regime.json").read_text())
    signs = []
    for snap in meta["snapshots"]:
        d = np.load(out_dir / f"snap_{snap['index']:03d}.npz")
        u, v, depth = d["u"], d["v"], d["depth"]
        deep = depth > 2.0
        if not deep.any():
            continue
        # net along-channel transport, projected on the dominant flow axis
        signs.append(float(np.mean(u[deep]) + np.mean(v[deep])))
    return {
        "n_samples": len(signs),
        "reversed": bool(signs and min(signs) < 0 < max(signs)),
        "min": min(signs) if signs else 0.0,
        "max": max(signs) if signs else 0.0,
    }
