"""Native particle multiplicity/weight semantics and inference safety."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.ensemble import evaluate_prepared_ensemble, load_ensemble
from ml_platform_symbolic.expressions import parse_expression


def save_ensemble(tmp_path, weights):
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0, 3]}]
    # Duplicate row is deliberate: unique equations are not uniformly weighted.
    payload = {
        "schema_version": 1,
        "kind": "structural_point_ensemble",
        "schema": {"variables": variables, "target": {"unit": "1", "domain": [0, 10]}},
        "members": [
            {"equation": parse_expression(expr, variables, trainable=False), "weight": weight}
            for expr, weight in zip(["x0", "x0", "2*x0"], weights, strict=True)
        ],
        "operators": ["mul"],
        "constraints": {"max_nodes": 30, "require_units": True},
        "metadata": {"uncertainty_scope": "structure, conditional on native fitted coefficients"},
    }
    path = tmp_path / "ensemble.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_native_multiplicity_weighted_predictions_and_structural_quantiles(tmp_path):
    model = load_ensemble(save_ensemble(tmp_path, [0.2, 0.6, 0.2]))
    frame = pd.DataFrame({"ignored": [99, 99], "x": [1.0, 2.0]})
    np.testing.assert_allclose(model.predict(frame), [1.2, 2.4])
    np.testing.assert_array_equal(model.predict(frame), model.predict(frame[frame.columns[::-1]]))
    np.testing.assert_allclose(model.structural_quantiles(frame, [0.5, 0.9]), [[1, 2], [2, 4]])
    assert model.predict_members(frame).shape == (2, 3)
    with pytest.raises(ValueError, match="domain"):
        model.predict(pd.DataFrame({"x": [-1.0]}))


def test_missing_weight_mass_is_not_silently_reweighted(tmp_path):
    with pytest.raises(ValueError, match="weights"):
        load_ensemble(save_ensemble(tmp_path, [0.2, 0.3, 0.2]))


def test_required_invalid_member_is_not_removed_from_posterior(tmp_path):
    path = save_ensemble(tmp_path, [0.2, 0.6, 0.2])
    payload = json.loads(path.read_text())
    payload["operators"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="constraints"):
        load_ensemble(path)


def test_cross_prepared_ensemble_is_rejected_before_partition_access(tmp_path):
    path = save_ensemble(tmp_path, [0.2, 0.6, 0.2])
    payload = json.loads(path.read_text())
    payload["metadata"]["provenance"] = {"prepared_sha256": "another run"}
    path.write_text(json.dumps(payload))
    prepared = tmp_path / "run.json"
    prepared.write_text("{}")
    with pytest.raises(ValueError, match="prepared run identity"):
        evaluate_prepared_ensemble(
            path, {"run": prepared, "final": tmp_path / "absent_final.npz"}, {}, tmp_path / "score"
        )


@pytest.mark.native
def test_actual_sampling_state_mean_survives_native_point_refit(tmp_path):
    pytest.importorskip("pysips")
    from current_flow import prepare_discover
    from ml_platform_symbolic.data import digest

    rng = np.random.default_rng(71)
    x = rng.uniform(0.2, 2, 36)
    frame = pd.DataFrame(
        {
            "sample_id": [f"n{i}" for i in range(36)],
            "x": x,
            "y": 0.37 + 1.43 * x + 0.4 * x * x + rng.normal(0, 0.08, 36),
        }
    )
    table, split = tmp_path / "table.csv", tmp_path / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 24 + ["selection"] * 12).to_csv(split, index=False)
    _, discoveries = prepare_discover(
        {
            "task": "symbolic_pipeline",
            "data": {
                "local_path": str(table),
                "feature_columns": ["x"],
                "target_column": "y",
                "units": {"x": "1", "y": "1"},
                "domain": {"x": [0.1, 2.1]},
            },
            "split": {"method": "fixed", "file": str(split)},
            "search": {
                "engines": ["pysips"],
                "operators": ["add", "mul"],
                "seeds": [42],
                "max_wall_seconds": 90,
                "engine_options": {"pysips": {"particles": 20, "mcmc_samples": 2, "max_complexity": 8, "seconds": 3}},
            },
            "constraints": {"max_nodes": 100, "require_units": False},
            "evaluation": {"mode": "exploratory", "coefficient_fit": "none"},
        },
        root=tmp_path / "run",
    )
    summary = json.loads(Path(discoveries[0]["summary"]).read_text())
    assert summary["structural_ensemble_status"] == "available"
    paths = summary["native_artifacts"]
    assert digest(paths["ensemble"]) == summary["native_artifact_sha256"]["ensemble"]
    raw = json.loads(open(paths["population"], encoding="utf-8").read())
    model = load_ensemble(paths["ensemble"])
    np.testing.assert_allclose(
        model.weights, [r["weight"] for r in raw["population"] if r["weight"] > 0], rtol=0, atol=0
    )
    probe = pd.DataFrame(raw["native_ensemble_probe"]["X"], columns=["x"])
    np.testing.assert_allclose(
        model.predict(probe), raw["native_ensemble_probe"]["weighted_prediction"], rtol=1e-10, atol=1e-10
    )
