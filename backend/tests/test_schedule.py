"""Flood/drain timing from a synthetic depth series.

Phase 0 is LOW water here -- spin-up is 0.4831 of a cycle -- so a flat that
floods on the rising half floods in phase 0.0-0.5. Getting that backwards
inverts every flat in the bay, so these fixtures pin it explicitly.
"""

import numpy as np
import pytest

from tidescout.pipeline import schedule


def _series(depths_by_phase):
    """depths_by_phase: (n_phases, n_cells) -> the shape load_state returns."""
    return [np.asarray(row, dtype="float32") for row in depths_by_phase]


def test_wet_fraction_counts_the_share_of_the_cycle_a_cell_holds_water():
    depths = _series([[1.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 0.0]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.wet_fraction[0] == pytest.approx(1.0)
    assert s.wet_fraction[1] == pytest.approx(0.25)


def test_flood_phase_is_when_the_cell_first_goes_wet_on_the_rising_half():
    """Cell floods at phase 0.25 (rising) and drains at 0.75 (falling)."""
    depths = _series([[0.0], [0.4], [0.6], [0.0]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.flood_phase[0] == pytest.approx(0.25)
    assert s.drain_phase[0] == pytest.approx(0.75)


def test_always_wet_and_never_wet_cells_have_no_schedule():
    """A channel has no flood time and a marsh hummock has no drain time.
    NaN says 'this question does not apply here' -- 0.0 would be a lie that
    reads as 'floods at low water'."""
    depths = _series([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.wet_fraction[0] == pytest.approx(1.0)
    assert np.isnan(s.flood_phase[0]) and np.isnan(s.drain_phase[0])
    assert s.wet_fraction[1] == pytest.approx(0.0)
    assert np.isnan(s.flood_phase[1]) and np.isnan(s.drain_phase[1])


def test_schedule_wraps_cyclically_across_the_end_of_the_series():
    """A cell wet at the end and start of the record floods before phase 0.
    Treating the series as a line rather than a cycle would report it as
    never flooding."""
    depths = _series([[0.5], [0.0], [0.0], [0.5]])
    s = schedule.schedule_from_depths(depths, phases=[0.0, 0.25, 0.5, 0.75])
    assert s.flood_phase[0] == pytest.approx(0.75)
    assert s.drain_phase[0] == pytest.approx(0.25)
    assert s.wet_fraction[0] == pytest.approx(0.5)


def test_a_residual_puddle_at_low_water_does_not_become_the_drain_phase():
    """Pins the bug found by Step 5 of Task 7: a shallow residual pool still
    sitting in a cell at the recorded low-water snapshot -- matching real
    ANUGA output, e.g. spring_high cell 373: 0.10 m at phase 0, dry within a
    phase or two -- finishes draining long before the cell's real flood/drain
    cycle happens. An independent 'first crossing from phase 0' search for
    drain_phase picked that early puddle dry-out instead of the drain that
    actually closes the wet window the flood opens, which dragged
    spring_high's median drain phase to 0.403 -- into the flooding half.
    drain_phase must be paired with its flood: the first dry crossing walking
    the cycle forward FROM the flood, not the first dry crossing from phase 0.

    Here the cell is wet at phase 0 (the puddle), dry by phase 0.1, stays dry
    through the early rising half, floods for real at phase 0.4, and drains
    for real at phase 0.6 -- well after the flood, not before it.
    """
    depths = _series(
        [[0.1], [0.0], [0.0], [0.0], [0.3], [0.6], [0.0], [0.0], [0.0], [0.1]]
    )
    phases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    s = schedule.schedule_from_depths(depths, phases=phases)
    assert s.flood_phase[0] == pytest.approx(0.4)
    assert s.drain_phase[0] == pytest.approx(0.6)


def test_wet_window_length_is_wrap_safe_when_the_window_spans_the_wrap():
    """A cell whose wet window straddles the phase-0 seam: wet at the last
    two phases and the first two, dry in the middle. It floods at phase 0.75
    and drains at phase 0.25 -- drain_phase is numerically SMALLER than
    flood_phase, so a raw `drain_phase - flood_phase` would read -0.5, as if
    the cell drained before it ever flooded. Phase is circular, so the
    wet-window length must be read off `(drain_phase - flood_phase) % 1.0`,
    which gives the true 0.5 (four of the eight phase steps) instead of a
    nonsensical negative duration.
    """
    depths = _series([[0.4], [0.4], [0.0], [0.0], [0.0], [0.0], [0.4], [0.4]])
    phases = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    s = schedule.schedule_from_depths(depths, phases=phases)
    assert s.flood_phase[0] == pytest.approx(0.75)
    assert s.drain_phase[0] == pytest.approx(0.25)
    window = (s.drain_phase[0] - s.flood_phase[0]) % 1.0
    assert window == pytest.approx(0.5)
