"""`tidescout flow structure` end to end, over a hand-built library.

Two things this covers that nothing else did. First the reprojection path:
`sample_features(already_projected=False)` is what production runs -- the
inventory is EPSG:4326 GeoJSON and the library grid is UTM -- and every unit
test short-circuits it. Second the command's own ~15 lines of selection logic:
best-phase-per-feature, the `n_cells` filter and the ranking, none of which sit
behind a tested function.

Fixtures follow `test_flow_validation.py`'s pattern (monkeypatched DATA_DIR, a
hand-written grid.json plus phase npz files, a stubbed `grid_spec`), since that
is the shape the sibling command's tests already use.
"""

import json
import math

import numpy as np
import pytest
from rasterio.warp import transform as warp_transform
from typer.testing import CliRunner

from tidescout.cli import app

runner = CliRunner()

# Rich wraps table cells to the terminal width, which would split feature ids
# across lines and make these assertions flaky rather than wrong.
WIDE = {"COLUMNS": "240"}

N = 40                      # 40x40 cells of 20 m: 800 m across
CELL = 20.0
ORIGIN = (662000.0, 3690000.0)   # a real EPSG:26917 easting/northing
CRS = "EPSG:26917"

# Columns holding the fast strip at each phase, and the column each feature
# sits in. The two neighbourhoods are ~500 m apart, well beyond the 150 m
# ambush/sampling radius, so neither feature can see the other's strip.
FAST_AT = {1: (5, 7, 1.2), 2: (30, 32, 0.8)}
COL_A, COL_B, ROW = 9, 34, 20
BACKGROUND = 0.1


def _cell_xy(row, col):
    return ORIGIN[0] + (col + 0.5) * CELL, ORIGIN[1] - (row + 0.5) * CELL


def _lonlat(row, col):
    x, y = _cell_xy(row, col)
    lons, lats = warp_transform(CRS, "EPSG:4326", [x], [y])
    return lons[0], lats[0]


def _write_library(tmp_path, monkeypatch, n_phases=4):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    grid = d / "flow" / "mean_med" / "grid"
    grid.mkdir(parents=True)
    # `flow structure` checks these exist before it reads anything; `grid_spec`
    # is stubbed below, so their contents are never opened.
    (d / "bathy_meta.json").write_text("{}")
    (d / "bathy_utm.tif").write_bytes(b"")

    phases = [i / n_phases for i in range(n_phases)]
    stage = [0.55 * math.cos(2.0 * math.pi * (0.4831 + p)) for p in phases]
    for i in range(n_phases):
        u = np.full((N, N), BACKGROUND, dtype="float32")
        if i in FAST_AT:
            lo, hi, speed = FAST_AT[i]
            u[:, lo:hi] = speed
        np.savez_compressed(
            grid / f"phase_{i:03d}.npz",
            u=u.ravel(), v=np.zeros(N * N, dtype="float32"),
            depth=np.full(N * N, 3.0, dtype="float32"), phase=np.float32(phases[i]),
        )
    (grid / "grid.json").write_text(json.dumps({
        "shape": [N, N], "cell_m": CELL, "transform": [CELL, 0, ORIGIN[0], 0, -CELL, ORIGIN[1]],
        "n_cells": N * N, "flat_index_len": N * N,
        "phases": phases, "stage_bc_m": stage,
    }))
    return d


def _write_features(data_dir):
    """A polygon beside the phase-1 strip, a point beside the phase-2 strip,
    and a point far outside the grid."""
    lon_a, lat_a = _lonlat(ROW, COL_A)
    half = 0.00005          # ~5 m, so the centroid stays inside its own cell
    ring = [
        [lon_a - half, lat_a - half], [lon_a + half, lat_a - half],
        [lon_a + half, lat_a + half], [lon_a - half, lat_a + half],
        [lon_a - half, lat_a - half],
    ]
    lon_b, lat_b = _lonlat(ROW, COL_B)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "hole-aaaaaaaaaaaa",
             "properties": {"type": "hole"},
             "geometry": {"type": "Polygon", "coordinates": [ring]}},
            {"type": "Feature", "id": "bar-bbbbbbbbbbbb",
             "properties": {"type": "bar"},
             "geometry": {"type": "Point", "coordinates": [lon_b, lat_b]}},
            {"type": "Feature", "id": "flat-cccccccccccc",
             "properties": {"type": "flat"},
             "geometry": {"type": "Point", "coordinates": [-70.0, 40.0]}},
        ],
    }
    (data_dir / "features.geojson").write_text(json.dumps(fc))


def _stub_grid_spec(monkeypatch):
    from tidescout.pipeline import flowlib

    rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    xs, ys = _cell_xy(rows.ravel(), cols.ravel())
    spec = flowlib.GridSpec(
        shape=(N, N), transform=None, cell_m=CELL,
        flat_index=np.arange(N * N), xs=xs, ys=ys, crs=CRS,
    )
    monkeypatch.setattr(flowlib, "grid_spec", lambda slug, fishery: spec)


def _run(tmp_path, monkeypatch, *args):
    d = _write_library(tmp_path, monkeypatch)
    _write_features(d)
    _stub_grid_spec(monkeypatch)
    return runner.invoke(app, ["flow", "structure", "winyah-bay", *args], env=WIDE)


def test_flow_structure_keeps_each_features_best_phase_and_ranks_by_it(
    tmp_path, monkeypatch
):
    """The fixture is built so the right answer cannot come from the last phase
    or from any single one: the polygon feature only sees fast water at phase 1
    (ambush 1.2 - 0.1 = 1.100) and the point feature only at phase 2
    (0.8 - 0.1 = 0.700), and phases 0 and 3 are uniform, where both read 0.000.
    So printing 1.100 and 0.700 in that order proves the command kept each
    feature's own best phase and ranked on it.

    It also proves the EPSG:4326 -> UTM reprojection landed: the features are
    written in lon/lat, and a reprojection that missed by even one 20 m cell
    would move them out of, or into, the fast strip."""
    result = _run(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.exception or result.stdout
    out = result.stdout
    assert "1.100" in out and "0.700" in out
    assert out.index("hole-aaaaaaaaaaaa") < out.index("bar-bbbbbbbbbbbb")


def test_flow_structure_drops_features_with_no_cells_in_the_domain(
    tmp_path, monkeypatch
):
    """The third feature sits at (-70, 40), in the Atlantic off New England.
    `sample_features` returns it with n_cells = 0 rather than dropping it, and
    the command's `if m.n_cells` filter is what keeps it out of the table --
    otherwise it would rank alongside real features on an all-NaN row."""
    result = _run(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.exception or result.stdout
    assert "flat-cccccccccccc" not in result.stdout


def test_flow_structure_top_limits_the_table(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, "--top", "1")
    assert result.exit_code == 0, result.exception or result.stdout
    assert "hole-aaaaaaaaaaaa" in result.stdout
    assert "bar-bbbbbbbbbbbb" not in result.stdout


def test_flow_structure_single_phase_reports_only_that_phase(tmp_path, monkeypatch):
    """`--phase 2` must show the point feature's 0.700 and the polygon's 0.000,
    not the polygon's cross-phase best of 1.100."""
    result = _run(tmp_path, monkeypatch, "--phase", "2")
    assert result.exit_code == 0, result.exception or result.stdout
    assert "0.700" in result.stdout
    assert "1.100" not in result.stdout


@pytest.mark.parametrize("bad", ["-2", "99"])
def test_flow_structure_rejects_a_phase_outside_the_library(tmp_path, monkeypatch, bad):
    """`phase < 0` means "every phase", so an unvalidated `--phase -2` silently
    ran the whole sweep, and an out-of-range positive failed deep inside
    `load_state` instead of at the boundary."""
    result = _run(tmp_path, monkeypatch, "--phase", bad)
    assert result.exit_code == 1, result.stdout
    assert "out of range" in result.stdout


def test_flow_structure_errors_clearly_when_a_prerequisite_is_missing(
    tmp_path, monkeypatch
):
    """`load_features` and `grid_spec` (which reads the bathymetry raster) both
    ran before the library guard, so a fresh checkout got a raw
    FileNotFoundError traceback instead of the command that fixes it."""
    d = _write_library(tmp_path, monkeypatch)
    _stub_grid_spec(monkeypatch)          # features.geojson deliberately absent
    result = runner.invoke(app, ["flow", "structure", "winyah-bay"], env=WIDE)
    assert result.exit_code == 1, result.stdout
    assert "no features.geojson" in result.stdout
    assert "tidescout features winyah-bay --rebuild" in result.stdout
    assert d.exists()
