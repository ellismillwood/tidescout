"""Flats flood/drain schedule -- the one derived signal that is not
instantaneous.

A flat's fishing value is WHEN it floods and drains, not what the water is
doing at one instant, so this is computed across a whole regime's phase series
and stored as a small per-cell table. Three float32 arrays over 587,325 cells
is ~7 MB per regime, against ~529 MB to store any instantaneous field, which is
why this one is precomputed and the others are not.
"""

import json
from dataclasses import dataclass
from pathlib import Path

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

    def first_phase(events: np.ndarray) -> np.ndarray:
        # A cell can cross more than once in a cycle; the first crossing is the
        # one that matters for "when can I fish it", and taking argmax of a
        # boolean gives that for free.
        any_event = events.any(axis=0)
        idx = events.argmax(axis=0)
        out = np.where(any_event, ph[idx], np.nan)
        return out.astype("float32")

    flood_phase = first_phase(goes_wet)
    drain_phase = first_phase(goes_dry)
    # Always-wet and never-wet cells have no transition; first_phase already
    # returns NaN for them, but say so explicitly rather than relying on it.
    static = (wet.sum(axis=0) == n) | (wet.sum(axis=0) == 0)
    flood_phase[static] = np.nan
    drain_phase[static] = np.nan
    return CellSchedule(wet_fraction, flood_phase, drain_phase)


def cell_schedule(slug: str, regime: str) -> CellSchedule:
    grid_meta = json.loads(
        (fishery_data_dir(slug) / "flow" / regime / "grid" / "grid.json").read_text()
    )
    phases = grid_meta["phases"]
    depths = [load_state(slug, regime, i)["depth"] for i in range(len(phases))]
    return schedule_from_depths(depths, phases)


def write_schedule(slug: str, regime: str) -> Path:
    s = cell_schedule(slug, regime)
    out = fishery_data_dir(slug) / "flow" / regime / "grid" / "schedule.npz"
    np.savez_compressed(
        out,
        wet_fraction=s.wet_fraction,
        flood_phase=s.flood_phase,
        drain_phase=s.drain_phase,
    )
    return out
