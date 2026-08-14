import json

import numpy as np

from tidescout.pipeline import regimes


def test_regime_matrix_is_three_by_three():
    assert len(regimes.REGIME_MATRIX) == 9
    assert ("spring", "high") in regimes.REGIME_MATRIX
    assert len(set(regimes.REGIME_MATRIX)) == 9


def test_regime_name_is_filesystem_safe_and_unique():
    names = {regimes.regime_name(r, d) for r, d in regimes.REGIME_MATRIX}
    assert len(names) == 9
    assert all(n.replace("_", "").isalnum() for n in names)


def test_mass_residual_tolerance_is_not_machine_precision():
    """Measured residual on a real wetting/drying mesh is ~4e-4. A 1e-6 gate
    fails every healthy run -- this was hit during the Plan 3 spike."""
    from tidescout.config import load_fishery
    assert load_fishery("winyah-bay").anuga.mass_tolerance >= 1e-4


def test_reversal_check_detects_a_one_way_domain(tmp_path):
    meta = {"snapshots": [{"index": 0, "t_s": 0, "phase": 0.0, "stage_bc_m": 0.0},
                          {"index": 1, "t_s": 1800, "phase": 0.1, "stage_bc_m": 0.1}]}
    (tmp_path / "regime.json").write_text(json.dumps(meta))
    for i in (0, 1):
        np.savez_compressed(
            tmp_path / f"snap_{i:03d}.npz",
            depth=np.array([5.0, 5.0], "float32"),
            u=np.array([0.4, 0.4], "float32"),   # always positive: never reverses
            v=np.zeros(2, "float32"),
        )
    assert regimes.reversal_check(tmp_path)["reversed"] is False
