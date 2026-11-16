"""Explicit native smoke tests; excluded from ordinary pytest (no downloads in fast tests)."""

from copy import deepcopy

import pytest
from current_flow import explicit_choice, study_flow
from ml_platform_symbolic import finalize_hypothesis
from ml_platform_symbolic.data import read_json


@pytest.mark.native
@pytest.mark.parametrize("engine", ["pysr", "physo"])
def test_real_native_export(engine, synthetic_config, tmp_path):
    cfg = deepcopy(synthetic_config)
    cfg["search"].update(
        engines=[engine],
        max_wall_seconds=180,
        engine_options={engine: {"epochs": 3, "batch_size": 64} if engine == "physo" else {"niterations": 2}},
    )
    _, discoveries, study = study_flow(cfg, tmp_path / engine, purpose="local_prediction", engineering_tolerance=100)
    _, decision = explicit_choice(study, tmp_path / engine)
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / engine / "final-audit")
    summary = read_json(discoveries[0]["summary"])
    assert summary["status"] == "ok"
    assert summary["selection_candidates"] > 0
    assert result.metrics["final_rmse"] >= 0
