# Diagnostic tools

Not part of the package or the CLI — standalone scripts, run directly:

    ~/.venvs/tidescout/bin/python backend/tools/<script>.py

They live here rather than in the SDD scratch directory because that directory
is gitignored (`.superpowers/sdd/.gitignore` is `*`), and Plan 3 lost
`instrumented_regime.py` to exactly that once, then found the surviving copy had
silently drifted to a two-tag boundary map that would crash on the current
three-tag mesh. This is the tool that diagnosed the instability which cost three
library builds; it should not be disposable.

| script | what it answers |
|---|---|
| `instrumented_regime.py` | Runs ONE regime logging per-yieldstep mass identity, timestep, speed distribution, and where the fastest cells are — including whether they sit on a boundary and under which tag. A crash dump shows the first cell to fail, never the thing that has been growing; this shows the growth. Calls `regimes._boundary_map` directly so it cannot drift from production. |
| `boundary_audit.py` | Static audit of boundary classification in RING ORDER. Per-tag counts hide the pathology that cost two builds — four `ocean` segments embedded in a solid `wall`, imposing the ocean tide 40 km inland. Needs no simulation. |
| `discharge_axis_check.py` | Whether the discharge axis is real or decorative: compares matched-phase snapshots across low/med/high near each river inflow. A leak through an upstream boundary does not crash and `mass_residual` correctly will not catch it. |
| `schedule_stats.py` | Per-regime flood/drain schedule stats: intertidal share, flood_phase p50, the p25/p50/p75 of the wrap-safe wet-window length, and the size and overlap of the early-drain and flood_phase==0.0 populations. Committed source for the numbers quoted in `tidescout/pipeline/schedule.py`'s docstring — Task 7's review flagged those as unreproducible prose. |
| `compare_variant.py` | Whether a config change moves the water enough to justify a rebuild: differences a one-change variant run against the shipped regime of the same name at matched phase, reporting a per-phase p99 speed delta and mean absolute speed change at each known spot. Every "should we rebuild?" call in Plans 3 and 4 was made on this number, and reproduces to the digit here. |

`instrumented_regime.py --help` is the docstring at the top of the file.
