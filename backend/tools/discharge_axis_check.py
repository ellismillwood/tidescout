"""Owed check (review Important 3): is the discharge axis real, or decorative?

The Pee Dee `Inlet_operator` sits ~2,456 m downstream of `open` segment 7 at
the river head, so injected discharge can flow UP-estuary and leave through
that boundary. Nothing crashes if it does, and `mass_residual` correctly does
not catch it -- the volume identity still closes, the water just exits. The
regime library would then carry a discharge axis that does nothing, and every
bite score built on it would be reading noise.

Note this may already be fixed as a side effect: cebfb5a made `open`
Reflective, and a wall cannot pass water out. Under builds 1-2 that head was
`ocean` (tide imposed) and under build 3 Transmissive -- both could leak. This
script is what decides that, rather than assuming it.

Scale of the signal to expect, computed from the configured buckets:
  low 78.6 m3/s, med 128.4, high 178.2 -> high-low = 99.6 m3/s
  over one 12.42 h cycle that is ~4.45e6 m3, against a domain holding
  ~6.7e8 m3 over ~238 km2 of water: only ~1.9 cm of DOMAIN-MEAN depth.
So a domain-wide comparison sits near the noise floor and proves little. The
discriminating test is LOCAL, in the upper rivers where the water is injected
and where a leak would show as an absent gradient.

Usage: discharge_axis_check.py [range_bucket]   (default: mean)
"""

import json
import sys
from pathlib import Path

import numpy as np
from rasterio.warp import transform as warp_transform

from tidescout.config import load_fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import mesh

SLUG = "winyah-bay"
RANGE_B = sys.argv[1] if len(sys.argv) > 1 else "mean"
NEAR_M = 5000.0  # radius around each inflow point counted as "upper river"

f = load_fishery(SLUG)
flow_dir = fishery_data_dir(SLUG) / "flow"

missing = [b for b in ("low", "med", "high")
           if not (flow_dir / f"{RANGE_B}_{b}" / "regime.json").exists()]
if missing:
    raise SystemExit(
        f"no completed {RANGE_B}_[{'/'.join(missing)}] regime(s) under {flow_dir} -- "
        "this check needs all three discharge buckets of one range bucket"
    )

print("building mesh for centroid coordinates...", flush=True)
domain = mesh.build_mesh(SLUG, f)
cx, cy = domain.get_centroid_coordinates(absolute=True).T
print(f"{len(cx):,} centroids", flush=True)

# Regions of interest: within NEAR_M of each river's injection point.
epsg = f.bathymetry.epsg
regions = {}
for river in f.rivers:
    seed = getattr(river, "inflow_lonlat", None)
    if seed is None:
        continue
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [seed[0]], [seed[1]])
    regions[river.name] = np.hypot(cx - xs[0], cy - ys[0]) <= NEAR_M
regions["DOMAIN"] = np.ones(len(cx), dtype=bool)


def load(bucket: str) -> tuple[dict, Path]:
    d = flow_dir / f"{RANGE_B}_{bucket}"
    return json.loads((d / "regime.json").read_text()), d


metas = {b: load(b) for b in ("low", "med", "high")}
n_snaps = {b: len(m["snapshots"]) for b, (m, _) in metas.items()}
print(f"snapshots per bucket: {n_snaps}")
n = min(n_snaps.values())
if n == 0:
    raise SystemExit("a regime completed with zero snapshots -- nothing to compare")

# Compare at MATCHED phase index. Buckets share range bucket and therefore
# tide, so snapshot i is the same point in the cycle in all three.
stats = {name: {b: {"depth": [], "speed": []} for b in ("low", "med", "high")}
         for name in regions}
for b, (_meta, d) in metas.items():
    for i in range(n):
        z = np.load(d / f"snap_{i:03d}.npz")
        depth, u, v = z["depth"], z["u"], z["v"]
        spd = np.hypot(u, v)
        wet = depth > 0.05
        for name, sel in regions.items():
            m = sel & wet
            if not m.any():
                continue
            stats[name][b]["depth"].append(float(depth[m].mean()))
            stats[name][b]["speed"].append(float(spd[m].mean()))

print(f"\ncomparing {n} matched-phase snapshots, range bucket {RANGE_B!r}")
print(f"regions: within {NEAR_M:.0f} m of each inflow, plus the whole domain\n")
print(f"{'region':12s} {'quantity':7s} {'low':>10s} {'med':>10s} {'high':>10s} "
      f"{'high-low':>10s} {'monotonic':>10s}")
verdicts = {}
for name in regions:
    for q in ("depth", "speed"):
        vals = {}
        for b in ("low", "med", "high"):
            arr = stats[name][b][q]
            vals[b] = float(np.mean(arr)) if arr else float("nan")
        diff = vals["high"] - vals["low"]
        mono = vals["low"] < vals["med"] < vals["high"]
        verdicts[(name, q)] = (vals, diff, mono)
        print(f"{name:12s} {q:7s} {vals['low']:10.5f} {vals['med']:10.5f} "
              f"{vals['high']:10.5f} {diff:+10.5f} {str(mono):>10s}")

# The verdict. A leak shows up as no gradient where the water goes IN; a
# healthy axis shows depth rising monotonically with discharge there.
print("\nVERDICT")
bad = []
for name in regions:
    if name == "DOMAIN":
        continue
    vals, diff, mono = verdicts[(name, "depth")]
    if not np.isfinite(diff) or abs(diff) < 1e-4:
        bad.append(f"{name}: depth is flat across low/med/high ({diff:+.2e} m) -- "
                   "discharge is not reaching this river's water, or is leaving "
                   "through the boundary above it")
    elif not mono:
        bad.append(f"{name}: depth is not monotonic in discharge "
                   f"({vals['low']:.5f} / {vals['med']:.5f} / {vals['high']:.5f})")
if bad:
    print("  DISCHARGE AXIS SUSPECT:")
    for b in bad:
        print(f"    - {b}")
else:
    print("  discharge axis is real: upper-river depth rises monotonically with "
          "discharge at every inflow.")
print("\nNOTE: the domain-wide row is expected to be small (~2 cm predicted for "
      "high-low).\nJudge the axis on the per-river rows, not on DOMAIN.")
