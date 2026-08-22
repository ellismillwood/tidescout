"""Reproducible source of the numbers quoted in schedule.py's docstring.

Task 7's review flagged that those percentages were prose assertions with no
committed way to reproduce them -- this project has already lost a diagnostic
harness once (see README.md in this directory) by leaving one only in the SDD
scratch directory, which is gitignored. This script is the checked-in source:
run it, and its output is what schedule.py's docstring is quoting.

Prints, per built regime: intertidal cell count and share, flood_phase p50,
the p25/p50/p75 of the wrap-safe wet-window length (drain_phase - flood_phase)
% 1.0, the share of intertidal cells that drain in phase 0.0-0.2 (just past
the low-water snapshot), the share with flood_phase == 0.0 exactly (the
snapshot-resolution population), and how much those last two overlap -- the
subset whose early "drain" is plausibly the SAME residual-puddle artifact as
their flood_phase == 0.0, not independent slow-marsh drainage.

Reads the gitignored data/<slug>/flow/ library on disk, so this is a tool,
not a test -- it is not wired into pytest.

Usage: schedule_stats.py [slug]   (default: winyah-bay)
"""

import sys

import numpy as np

from tidescout.paths import fishery_data_dir
from tidescout.pipeline import schedule
from tidescout.pipeline.regimes import REGIME_MATRIX, regime_name

SLUG = sys.argv[1] if len(sys.argv) > 1 else "winyah-bay"

flow_dir = fishery_data_dir(SLUG) / "flow"
regimes = [
    regime_name(r, d)
    for r, d in REGIME_MATRIX
    if (flow_dir / regime_name(r, d) / "grid" / "grid.json").exists()
]
if not regimes:
    raise SystemExit(f"no rasterised regimes found under {flow_dir}")

print(
    f"{'regime':12s} {'intertidal':>17s} {'flood p50':>9s} "
    f"{'window p25/p50/p75':>19s} {'drain in [0,.2)':>16s} "
    f"{'flood==0.0':>13s} {'overlap':>22s}"
)
for r in regimes:
    s = schedule.cell_schedule(SLUG, r)
    inter = np.isfinite(s.flood_phase)
    n_inter = int(inter.sum())
    window = (s.drain_phase - s.flood_phase) % 1.0
    p25, p50, p75 = np.percentile(window[inter], [25, 50, 75])
    early_drain = inter & (s.drain_phase >= 0.0) & (s.drain_phase < 0.2)
    flood_zero = inter & (s.flood_phase == 0.0)
    overlap = early_drain & flood_zero
    n_early, n_zero, n_overlap = int(early_drain.sum()), int(flood_zero.sum()), int(overlap.sum())
    print(
        f"{r:12s} {n_inter:7,d} ({inter.mean() * 100:5.2f}%)  "
        f"{np.nanmedian(s.flood_phase):9.3f}  "
        f"{p25:.3f}/{p50:.3f}/{p75:.3f}        "
        f"{n_early:6,d} ({n_early / n_inter * 100:4.1f}%)  "
        f"{n_zero:6,d} ({n_zero / n_inter * 100:4.1f}%)  "
        f"{n_overlap:6,d} ({n_overlap / max(n_zero, 1) * 100:4.1f}% of "
        f"flood==0.0, {n_overlap / max(n_early, 1) * 100:4.1f}% of early-drain)"
    )
