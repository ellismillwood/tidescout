"""Instrumented single-regime run to find what accumulates before the crash.

A crash dump shows the first cell to FAIL, never the thing that has been
GROWING. Two root causes were declared from crash dumps and both were wrong;
the instrumented time series found the real one (cause #1, the mis-tagged
ocean boundary at the Pee Dee head) in a single run.

Logs every `yieldstep` of sim time:
  - mass identity terms (volume, boundary flux, fractional-step inflow, residual)
  - timestep, wet fraction
  - speed distribution and how many cells exceed physical thresholds
  - WHERE the fastest cell is, and whether the hot cells PERSIST between
    yieldsteps (a stationary hotspot growing) or move around (noise)
  - whether the hot cells sit ON a boundary, and under which tag. Both known
    instabilities were boundary-driven, so "which tag is the hotspot touching"
    is the single most diagnostic column here.

The boundary map comes from `regimes._boundary_map`, NOT a local copy, so this
harness cannot drift from production.

Usage:
  instrumented_regime.py <range> <discharge> <sim_hours> <tag> [options]

Options:
  --no-inflow           disable river inflows entirely (control)
  --ocean-bc=<kind>     override the seaward `ocean` tag only. One of:
                          prod       production Transmissive_momentum_set_stage
                          dirichlet  Time_boundary([stage, 0, 0])
                        Used to test whether a seaward-boundary failure is
                        caused by the BC choice rather than the geometry.
  --yieldstep=<s>       sample interval, default 900
"""

import csv
import sys
import time

import anuga
import numpy as np

from tidescout.config import load_fishery
from tidescout.pipeline import forcing, mesh, regimes

HERE = "/Users/ellismillwood/Documents/tidescout/.superpowers/sdd/2026-08-13-03-anuga-flow-library"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = [a for a in sys.argv[1:] if a.startswith("--")]


def _opt(name, default):
    for o in opts:
        if o.startswith(f"--{name}="):
            return o.split("=", 1)[1]
    return default


RANGE_B = args[0] if len(args) > 0 else "mean"
DISCH_B = args[1] if len(args) > 1 else "med"
SIM_H = float(args[2]) if len(args) > 2 else 14.0  # ~50,400 s, past every crash
TAG = args[3] if len(args) > 3 else f"{RANGE_B}_{DISCH_B}"
NO_INFLOW = "--no-inflow" in opts
OCEAN_BC = _opt("ocean-bc", "prod")
YIELD = float(_opt("yieldstep", 900.0))
OUT = f"{HERE}/instr-{TAG}.csv"

f = load_fishery("winyah-bay")
cfg = f.anuga
domain = mesh.build_mesh("winyah-bay", f)
domain.set_name(f"instr_{TAG}")
domain.set_quantities_to_be_stored(None)  # no .sww; we log our own
domain.set_quantity(
    "friction", mesh.friction_field(domain, "winyah-bay", f), location="centroids"
)
elev = domain.get_quantity("elevation").get_values(location="centroids")
cx, cy = domain.get_centroid_coordinates(absolute=True).T
radii = domain.radii

# triangle index -> set of boundary tags it carries. domain.boundary maps
# (volume_id, edge_id) -> tag. Both known instabilities grew against a
# boundary, so this is what tells us which one, without waiting for a dump.
tri_tags: dict[int, set] = {}
for (vol_id, _edge), tag in domain.boundary.items():
    tri_tags.setdefault(int(vol_id), set()).add(tag)
ocean_tris = {i for i, t in tri_tags.items() if "ocean" in t}
open_tris = {i for i, t in tri_tags.items() if "open" in t}
wall_tris = {i for i, t in tri_tags.items() if "wall" in t}

tide = forcing.range_scaled_tide(cfg.mean_range_m, RANGE_B, period_s=cfg.cycle_h * 3600.0)
domain.set_quantity("stage", regimes.initial_stage(elev, tide(0.0)), location="centroids")

# Production map, then a surgical override of the seaward tag only.
bmap = regimes._boundary_map(domain, tide)
if OCEAN_BC == "dirichlet":
    bmap["ocean"] = anuga.Time_boundary(
        domain=domain, function=lambda t: [tide(t), 0.0, 0.0]
    )
elif OCEAN_BC != "prod":
    raise SystemExit(f"unknown --ocean-bc={OCEAN_BC!r}")
domain.set_boundary(bmap)

inflows = forcing.river_inflow_m3s(f, DISCH_B)
if NO_INFLOW:
    inflows = {k: 0.0 for k in inflows}
    print("CONTROL RUN: river inflows DISABLED entirely", flush=True)
else:
    regimes._attach_river_inflows(domain, f, inflows)

tag_counts = {t: sum(1 for v in domain.boundary.values() if v == t)
              for t in set(domain.boundary.values())}
print(f"{len(domain.triangles):,} triangles | regime {RANGE_B}_{DISCH_B} | "
      f"target {SIM_H} sim-h ({SIM_H*3600:.0f} s) | yieldstep {YIELD:.0f} s", flush=True)
print(f"boundary segments: {tag_counts} | ocean-bc={OCEAN_BC}", flush=True)
print(f"boundary triangles: ocean={len(ocean_tris)} open={len(open_tris)} "
      f"wall={len(wall_tris)}", flush=True)
print(f"inflows: { {k: round(v,1) for k,v in inflows.items()} }", flush=True)
print(f"logging to {OUT}", flush=True)

v0 = domain.get_water_volume()
prev_hot = set()
fh = open(OUT, "w", newline="")
w = csv.writer(fh)
w.writerow(["t_s", "phase", "tide_m", "dt", "volume_m3", "dvol", "bflux", "fracstep",
            "mass_resid", "wet_frac", "spd_p50", "spd_p99", "spd_p999", "spd_max",
            "n_gt_1ms", "n_gt_3ms", "max_x", "max_y", "max_depth", "max_elev",
            "max_radius", "hot20_persist", "max_btag", "hot20_ocean", "hot20_open",
            "hot20_wall", "steps", "dt_min", "dt_max", "wall_s"])
fh.flush()

t_start = time.time()
try:
    for t in domain.evolve(yieldstep=YIELD, finaltime=SIM_H * 3600.0):
        stage = domain.get_quantity("stage").get_values(location="centroids")
        xm = domain.get_quantity("xmomentum").get_values(location="centroids")
        ym = domain.get_quantity("ymomentum").get_values(location="centroids")
        dep = stage - elev
        wet = dep > 0.01
        safe = np.where(wet, dep, 1.0)
        spd = np.where(wet, np.hypot(xm, ym) / safe, 0.0)

        vol = domain.get_water_volume()
        bflux = domain.get_boundary_flux_integral()
        frac = domain.get_fractional_step_volume_integral()
        moved = max(abs(vol - v0), 1.0)
        resid = abs((vol - v0) - (bflux + frac)) / moved

        i = int(np.argmax(spd))
        hot = set(np.argsort(spd)[::-1][:20].tolist())
        persist = len(hot & prev_hot)
        prev_hot = hot
        btag = "+".join(sorted(tri_tags.get(i, set()))) or "-"
        sw = spd[wet]
        row = [f"{t:.1f}", f"{(t/(cfg.cycle_h*3600))%1:.4f}", f"{tide(t):+.4f}",
               f"{domain.timestep:.3e}", f"{vol:.1f}", f"{vol-v0:.1f}", f"{bflux:.1f}",
               f"{frac:.1f}", f"{resid:.3e}", f"{wet.mean():.4f}",
               f"{np.median(sw):.4f}", f"{np.percentile(sw,99):.4f}",
               f"{np.percentile(sw,99.9):.4f}", f"{spd.max():.4f}",
               int((spd > 1).sum()), int((spd > 3).sum()),
               f"{cx[i]:.0f}", f"{cy[i]:.0f}", f"{dep[i]:.3f}", f"{elev[i]:.3f}",
               f"{radii[i]:.2f}", persist, btag,
               len(hot & ocean_tris), len(hot & open_tris), len(hot & wall_tris),
               # These three are reset by ANUGA at each yield, so they describe
               # the yieldstep just completed, not the whole run. steps is the
               # honest cost signal: 900 s / steps is the mean dt actually
               # achieved, and build #3's transmissive regression showed up
               # here as 10x more steps for the same sim time.
               int(domain.number_of_steps),
               f"{domain.recorded_min_timestep:.3e}",
               f"{domain.recorded_max_timestep:.3e}",
               f"{time.time()-t_start:.0f}"]
        w.writerow(row)
        fh.flush()
        print(f"  t={t:7.0f} ph={(t/(cfg.cycle_h*3600))%1:.3f} tide={tide(t):+.3f} "
              f"dt={domain.timestep:.2e} wet={wet.mean()*100:5.1f}% "
              f"spd p99={np.percentile(sw,99):6.3f} max={spd.max():9.3f} "
              f">1m/s={int((spd>1).sum()):5d} >3={int((spd>3).sum()):4d} "
              f"hot20persist={persist:2d} resid={resid:.1e} "
              f"@({cx[i]:.0f},{cy[i]:.0f}) d={dep[i]:.2f} btag={btag} "
              f"hot20[oc={len(hot & ocean_tris)} op={len(hot & open_tris)} "
              f"wl={len(hot & wall_tris)}] steps={domain.number_of_steps:6d} "
              f"dtmax={domain.recorded_max_timestep:.2e} "
              f"wall={(time.time()-t_start)/60:.0f}m",
              flush=True)
    print(f"\nCOMPLETED {SIM_H} sim-h in {(time.time()-t_start)/3600:.2f} h wall", flush=True)
except Exception as exc:
    print(f"\nCRASHED after {(time.time()-t_start)/3600:.2f} h wall: {str(exc).strip()[:300]}",
          flush=True)
finally:
    fh.close()
    print(f"csv written: {OUT}", flush=True)
