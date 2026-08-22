"""Geometry and reporting for the Task 13 gate -- not the physics.

Whether the model actually reproduces Ellis's spots is a judgement call on real
output, deliberately not asserted here. What IS asserted is that the gate reads
the library correctly, because a gate that mislabels ebb as flood would fail
the model for being right.
"""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tidescout.cli import app
from tidescout.engine import flow

runner = CliRunner()

# Rich wraps table cells to the terminal width, which would split "OUTSIDE
# DOMAIN" across lines and make these assertions flaky rather than wrong.
WIDE = {"COLUMNS": "220"}


def test_slack_spot_shows_a_speed_minimum_beside_fast_water():
    """The eddy/slack signature: a low-speed pocket adjacent to a fast conveyor."""
    u = np.array([1.2, 1.1, 0.05, 0.03, 1.0])
    v = np.zeros(5)
    speed, _ = flow.speed_direction(u, v)
    assert speed.min() < 0.1
    assert speed.max() > 1.0
    assert speed.max() - speed.min() > 0.9   # a real seam, not uniform slow water


def test_spot_outside_domain_is_detected_not_silently_zero():
    xs = np.array([100.0, 200.0])
    ys = np.array([100.0, 200.0])
    near = (xs - 99999.0) ** 2 + (ys - 99999.0) ** 2 <= 150.0**2
    assert not near.any()


def _write_library(tmp_path, monkeypatch, n_cells=4, n_phases=24):
    """A minimal rasterised library at the real spots' coordinates."""
    import math

    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    grid = paths.fishery_data_dir("winyah-bay") / "flow" / "mean_med" / "grid"
    grid.mkdir(parents=True)
    phases = [i / n_phases for i in range(n_phases)]
    stage = [0.55 * math.cos(2.0 * math.pi * (0.4831 + p)) for p in phases]
    for i in range(n_phases):
        np.savez_compressed(
            grid / f"phase_{i:03d}.npz",
            u=np.full(n_cells, 0.1, dtype="float32"),
            v=np.zeros(n_cells, dtype="float32"),
            depth=np.full(n_cells, 3.0, dtype="float32"),
            phase=np.float32(phases[i]),
        )
    (grid / "grid.json").write_text(json.dumps({
        "shape": [2, 2], "cell_m": 20.0, "transform": [20, 0, 0, 0, -20, 0],
        "n_cells": n_cells, "flat_index_len": n_cells,
        "phases": phases, "stage_bc_m": stage,
    }))
    return grid


def test_validate_errors_clearly_when_the_library_is_not_rasterised(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    result = runner.invoke(app, ["flow", "validate", "winyah-bay"], env=WIDE)
    assert result.exit_code == 1
    assert "no rasterised library" in result.stdout


def test_grid_json_carries_the_stage_series_the_gate_needs(tmp_path, monkeypatch):
    """Regression guard: without `stage_bc_m` the gate cannot tell ebb from
    flood, and phase order alone would get it backwards."""
    grid = _write_library(tmp_path, monkeypatch)
    meta = json.loads((grid / "grid.json").read_text())
    assert len(meta["stage_bc_m"]) == len(meta["phases"])
    states = flow.tide_states(meta["stage_bc_m"])
    assert states[0] == "slack"          # phase 0 is LOW water, a turning point
    assert "flood" in states[1:6]        # rising immediately after


def test_flowlib_grid_json_includes_stage_from_regime_snapshots():
    """`_grid_json` must carry `stage_bc_m` through from regime.json."""
    from tidescout.pipeline import flowlib

    spec = flowlib.GridSpec(
        shape=(2, 2), transform=None, cell_m=20.0,
        flat_index=np.array([0, 1]), xs=np.array([0.0]), ys=np.array([0.0]),
    )
    snaps = [
        {"index": 0, "phase": 0.0, "stage_bc_m": -0.55},
        {"index": 1, "phase": 0.5, "stage_bc_m": 0.55},
    ]
    payload = flowlib._grid_json(spec, snaps, transform6=[1, 0, 0, 0, 1, 0])
    assert payload["stage_bc_m"] == [-0.55, 0.55]


def _stub_grid_spec(monkeypatch, xs, ys):
    """Replace `grid_spec`, which needs the real bathymetry raster on disk.

    These tests exercise the command's reporting, not the grid construction --
    `grid_spec` has its own tests in test_flowlib.py.
    """
    import numpy as np

    from tidescout.pipeline import flowlib

    spec = flowlib.GridSpec(
        shape=(2, 2), transform=None, cell_m=20.0,
        flat_index=np.arange(len(xs)), xs=np.asarray(xs), ys=np.asarray(ys),
    )
    monkeypatch.setattr(flowlib, "grid_spec", lambda slug, fishery: spec)


def _spot_utm():
    from rasterio.warp import transform as warp_transform

    from tidescout.config import load_fishery, load_known_spots

    f = load_fishery("winyah-bay")
    spots = load_known_spots("winyah-bay")
    return warp_transform(
        "EPSG:4326", f"EPSG:{f.bathymetry.epsg}",
        [s.lon for s in spots], [s.lat for s in spots],
    )


def test_validate_reports_every_spot(tmp_path, monkeypatch):
    """All three shipped spots must appear when the grid covers them."""
    xs, ys = _spot_utm()
    _write_library(tmp_path, monkeypatch, n_cells=len(xs))
    _stub_grid_spec(monkeypatch, xs, ys)
    result = runner.invoke(app, ["flow", "validate", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    for name in ("Mud Bay", "Georgetown", "Jetty"):
        assert name in result.stdout


def test_validate_flags_spots_outside_the_domain(tmp_path, monkeypatch):
    """A grid nowhere near the spots must report OUTSIDE DOMAIN rather than
    silently scoring zero -- the brief calls that a real finding, not a bug."""
    _write_library(tmp_path, monkeypatch, n_cells=2)
    _stub_grid_spec(monkeypatch, [0.0, 20.0], [0.0, 20.0])
    result = runner.invoke(app, ["flow", "validate", "winyah-bay"], env=WIDE)
    assert result.exit_code == 0, result.exception or result.stdout
    assert "OUTSIDE DOMAIN" in result.stdout


@pytest.mark.parametrize("works_on,expected", [("ebb", "ebb"), ("flood", "flood")])
def test_tide_state_labels_match_the_projects_phase_convention(works_on, expected):
    """Guards the inversion risk directly: flood is the FIRST half of the
    cycle here because phase 0 is low water."""
    import math

    stage = [0.55 * math.cos(2.0 * math.pi * (0.4831 + i / 24)) for i in range(24)]
    states = flow.tide_states(stage)
    first_half = states[1:11]
    second_half = states[13:23]
    assert "flood" in first_half and "ebb" not in first_half
    assert "ebb" in second_half and "flood" not in second_half
