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

`instrumented_regime.py --help` is the docstring at the top of the file.
