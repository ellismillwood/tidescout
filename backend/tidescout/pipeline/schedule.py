"""Flats flood/drain schedule -- the one derived signal that is not
instantaneous.

A flat's fishing value is WHEN it floods and drains, not what the water is
doing at one instant, so this is computed across a whole regime's phase series
and stored as a small per-cell table. Three float32 arrays over 587,325 cells
is ~7 MB per regime, against ~529 MB to store any instantaneous field, which is
why this one is precomputed and the others are not.

`flood_phase` and `drain_phase` are both phases on a CIRCLE (period 1.0), so
verifying them means using a wrap-safe statistic, not a raw phase comparison
or an ordinary median of one column alone. `(drain_phase - flood_phase) % 1.0`
is the wet-window length -- how long the cell holds water -- and it is the
right thing to check: it is invariant to where on the circle the window sits,
so it cannot land on the wrong side of an arbitrary cut point the way a raw
median of `drain_phase` can. Measured on the shipped `winyah-bay` library,
that window length has p50 = 0.523 in all nine shipped regimes -- a flat is
wet for about half a cycle regardless of tidal range, which is the physically
expected, stable answer. These figures, and the ones below, are reproducible
via `backend/tools/schedule_stats.py <slug>`; do not take them on faith.

An earlier check used `np.nanmedian(drain_phase)` directly and read 0.403 for
`spring_high` against 0.765 for `neap_low`, which looked like an inversion.
It wasn't: roughly a third of intertidal cells (33.8% at neap, 35.7% at
spring) drain in phase 0.0-0.2, just past the low-water snapshot -- most of
these are slow-draining high-marsh cells that hold water past low water, and
that population grows with tidal range because a bigger tide wets more high
marsh. With a third of the distribution sitting just past the wrap, an
ordinary median of `drain_phase` alone lands on whichever side of 0.5 that
cluster happens to fall, which is an artifact of the cut point, not of the
physics.

Separately: ~7% of `spring_high`'s intertidal cells report `flood_phase ==
0.0` exactly. This is a snapshot-resolution effect, not a logic error: at
30-minute snapshots on a 12.42 h cycle, a cell whose previous-cycle tail is
dry and which is wet at the phase-0 snapshot genuinely does flood within that
bin. It is not evidence that flood and drain are miscomputed.

These two populations are NOT independent: 2,697 of `spring_high`'s 4,037
`flood_phase == 0.0` cells (66.8%) also fall in the 0.0-0.2 early-drain
bucket above, and those 2,697 cells are 13.1% of that bucket's full 20,617.
For that overlapping subset, `drain_phase` most likely reports the SAME
residual puddle's own quick dry-out that gives it `flood_phase == 0.0`, not
genuine slow-marsh drainage -- read it under the snapshot-resolution caveat,
not as physics. The "slow-draining high marsh" explanation still stands for
the other ~87% of the early-drain cluster, whose `flood_phase` is unremarkable.
"""

import json
from dataclasses import dataclass

import numpy as np

from tidescout.engine.flow import wet_mask
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.flowlib import load_state


@dataclass
class CellSchedule:
    wet_fraction: np.ndarray  # 0-1, share of the cycle holding water
    flood_phase: np.ndarray   # phase at which it goes wet; NaN if never/always
    drain_phase: np.ndarray   # phase at which it goes dry; NaN if never/always


def schedule_from_depths(depths: list[np.ndarray], phases: list[float]) -> CellSchedule:
    """Per-cell wet fraction and flood/drain phases from a depth series.

    The series is CYCLIC: a cell wet at the last phase and the first floods
    somewhere before phase 0, and treating the record as a line rather than a
    ring would report it as never flooding at all. `np.roll` gives the previous
    phase's state with that wrap built in.

    Phase 0 is LOW water (spin_up_h / cycle_h = 0.4831 of a cycle), so a flat
    that floods on the rising half floods in phase 0.0-0.5.
    """
    wet = np.array([wet_mask(d) for d in depths])  # (n_phases, n_cells)
    ph = np.asarray(phases, dtype="float64")
    n = wet.shape[0]

    wet_fraction = wet.mean(axis=0).astype("float32")
    was_wet = np.roll(wet, 1, axis=0)
    goes_wet = wet & ~was_wet   # dry -> wet transition at this phase
    goes_dry = ~wet & was_wet

    has_flood = goes_wet.any(axis=0)
    flood_idx = goes_wet.argmax(axis=0)

    # Drain is the first dry transition AFTER the flood, walking the cycle
    # forward from it -- NOT an independent first crossing from index 0. Some
    # cells hold a shallow residual pool at the recorded low-water snapshot
    # that finishes draining a phase or two later; without pairing, that
    # early dry-out (the first `goes_dry` in array order) could be picked
    # over the drain that actually closes the wet window the flood opens.
    # This is a correctness guard for cells with more than one wet/dry cycle
    # in the record -- see test_schedule.py's residual-puddle regression
    # test, which goes RED against the unpaired version. On the shipped
    # library it changes ZERO cells: it neither caused nor
    # fixed the 0.403 median-drain-phase reading discussed in this module's
    # docstring above, which was an artifact of an unsafe verification
    # statistic, not of this computation.
    #
    # offset = forward steps from the flood to a given index, cyclically;
    # n + 1 marks non-`goes_dry` indices so they always lose the min() below.
    offsets = (np.arange(n)[:, None] - flood_idx[None, :]) % n
    ranked = np.where(goes_dry, offsets, n + 1)
    drain_off = ranked.min(axis=0)
    has_drain = drain_off < n
    drain_idx = (flood_idx + drain_off) % n

    flood_phase = np.where(has_flood, ph[flood_idx], np.nan).astype("float32")
    drain_phase = np.where(has_flood & has_drain, ph[drain_idx], np.nan).astype("float32")

    # Always-wet and never-wet cells have no transition; say so explicitly
    # rather than relying on has_flood/has_drain alone.
    static = (wet.sum(axis=0) == n) | (wet.sum(axis=0) == 0)
    flood_phase[static] = np.nan
    drain_phase[static] = np.nan
    return CellSchedule(wet_fraction, flood_phase, drain_phase)


def cell_schedule(slug: str, regime: str) -> CellSchedule:
    """Read a regime's whole phase series off disk and build its schedule.

    Computed on demand, not cached to a file. A `write_schedule` helper used to
    sit here and write `schedule.npz` beside the grid; nothing ever called it,
    nothing in the Phase 2 or Phase 3 plans referenced it, and nothing would
    have invalidated that file when the library behind it was rebuilt -- a
    stale-cache bug waiting on the next rebuild, in exchange for one pass over
    the regime's already-loaded depth arrays. It is in git if a caller ever
    appears.
    """
    grid_meta = json.loads(
        (fishery_data_dir(slug) / "flow" / regime / "grid" / "grid.json").read_text()
    )
    phases = grid_meta["phases"]
    depths = [load_state(slug, regime, i)["depth"] for i in range(len(phases))]
    return schedule_from_depths(depths, phases)
