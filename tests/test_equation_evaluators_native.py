"""Small real-library oracles. Run only in the named local Docker evaluator image."""

import importlib.util
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.equation_evaluation import create_study, evaluate_equations, load_study

pytestmark = pytest.mark.native


def offset_plan(config):
    config = deepcopy(config)
    config["search"]["operators"] = ["add", "mul", "neg", "sin", "exp"]
    return {
        "problem": config,
        "engineering_tolerance": 0.1,
        "sources": [
            {"manual": {"id": "first", "expression": "x"}},
            {"manual": {"id": "offset", "expression": "x+10"}},
        ],
    }


@pytest.mark.skipif(importlib.util.find_spec("effector") is None, reason="equation-analysis Docker only")
@pytest.mark.parametrize("kind", ["ale", "rhale"])
def test_real_effector_center_does_not_remove_prediction_offset(synthetic_config, tmp_path, kind):
    study = create_study(offset_plan(synthetic_config), run_dir=tmp_path / "study")
    result = evaluate_equations(study, "effector", run_dir=tmp_path / "effects", options={"bins": 4, "kind": kind})
    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    first, second = data["results"]
    np.testing.assert_allclose(first["curves"][0]["effect"], second["curves"][0]["effect"], atol=1e-8)
    assert second["mean_prediction"] - first["mean_prediction"] == pytest.approx(10)
    assert first["curves"][0]["points_per_bin"] == second["curves"][0]["points_per_bin"]
    assert "not coefficient CI" in first["curves"][0]["heterogeneity_meaning"]


@pytest.mark.skipif(importlib.util.find_spec("reggression") is None, reason="equation-analysis Docker only")
def test_real_reggression_preserves_frozen_sources(synthetic_config, tmp_path):
    config = offset_plan(synthetic_config)
    config["sources"] = [
        {"manual": {"id": "sine", "expression": "sin(x)"}},
        {"manual": {"id": "exponential", "expression": "exp(x)"}},
        {
            "manual": {
                "id": "decay",
                "expression": "exp(-a*x)",
                "parameters": {"a": {"value": 1.27, "trainable": True, "bounds": [0, 10], "unit": "1"}},
            }
        },
    ]
    study = create_study(config, run_dir=tmp_path / "study")
    before = read_json(study / "hypotheses.json")
    result = evaluate_equations(study, "reggression", run_dir=tmp_path / "structure")
    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    assert len({r["native_eclass"] for r in data["results"] if r["outcome"] == "ok"}) == 3
    assert data["native_pattern_queries"].startswith("unsupported")
    assert read_json(study / "hypotheses.json") == before


@pytest.mark.skipif(importlib.util.find_spec("effector") is None, reason="equation-analysis Docker only")
def test_real_effector_interaction_regions_and_centering(tmp_path):
    x = np.linspace(0.2, 2.2, 90)
    z = 0.5 * x + 0.2 * np.sin(11 * x)
    frame = pd.DataFrame({"sample_id": [f"r{i}" for i in range(len(x))], "x": x, "z": z, "y": x * z})
    frame.to_csv(tmp_path / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=np.tile(["train", "selection", "final"], 30)).to_csv(
        tmp_path / "split.csv", index=False
    )
    study = create_study(
        {
            "problem": {
                "data": {
                    "local_path": str(tmp_path / "data.csv"),
                    "feature_columns": ["x", "z"],
                    "target_column": "y",
                    "units": {"x": "1", "z": "1", "y": "1"},
                },
                "split": {"file": str(tmp_path / "split.csv")},
                "search": {"engines": ["mean"], "operators": ["add", "mul"]},
            },
            "sources": [
                {"manual": {"id": "interaction", "expression": "x*z"}},
                {"manual": {"id": "offset", "expression": "x*z+10"}},
            ],
            "regions": {"all": {}, "low": {"x": [0.2, 1.1]}},
        },
        run_dir=tmp_path / "study",
    )
    result = evaluate_equations(
        study, "effector", run_dir=tmp_path / "effects", options={"bins": 4, "centering": "zero_integral"}
    )
    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    first, second = data["results"]
    assert len(first["curves"]) == 4
    assert any(np.max(c["heterogeneity"]) > 1e-5 for c in first["curves"])
    for left, right in zip(first["curves"], second["curves"], strict=True):
        assert left["centering"] == "zero_integral"
        np.testing.assert_allclose(left["effect"], right["effect"], atol=1e-7)
        assert sum(left["points_per_bin"]) == left["n_reference_rows"]
    assert second["mean_prediction"] - first["mean_prediction"] == pytest.approx(10)


@pytest.mark.skipif(not os.getenv("SYMBOLIC_LIKELIHOOD_PROJECT"), reason="equation-evidence Docker only")
def test_real_likelihood_parameter_and_functional_oracle(tmp_path):
    frame = pd.DataFrame({"sample_id": [f"r{i}" for i in range(6)], "x": [1.0, 0.0] * 3, "y": [3.0, 1.0] * 3})
    table, split = tmp_path / "table.csv", tmp_path / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 2 + ["selection"] * 2 + ["final"] * 2).to_csv(split, index=False)
    params = {
        k: {"value": v, "trainable": True, "bounds": [-20, 20], "unit": "1"} for k, v in {"a": 3.0, "b": 4.0}.items()
    }
    config = {
        "problem": {
            "data": {
                "local_path": str(table),
                "feature_columns": ["x"],
                "target_column": "y",
                "units": {"x": "1", "y": "1"},
            },
            "split": {"file": str(split)},
            "constraints": {"max_nodes": 40},
            "search": {"engines": ["mean"], "operators": ["add", "sub", "mul"]},
        },
        "sources": [{"manual": {"id": "linear_system", "expression": "a*x+(b-a)*(1-x)", "parameters": params}}],
    }
    study = create_study(config, run_dir=tmp_path / "study")
    before = load_study(study)[1]
    result = evaluate_equations(
        study,
        "likelihood_profiler",
        run_dir=tmp_path / "profiles",
        options={
            "sigma": 1.0,
            "parameter_ranges": {"a": [-1, 7]},
            "step": 0.2,
            "maxiters": 500,
            "quantities": [{"id": "sum", "expression": "a+b", "range": [-1, 15], "unit": "1"}],
            "max_wall_seconds": 180,
        },
    )
    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    native = data["results"][0]["native"]
    profile = native["profiles"][0]
    x = np.array(profile["x"])
    np.testing.assert_allclose(np.array(profile["objective"]) - native["objective_min"], (x - 3) ** 2 / 2, atol=1e-5)
    fitted = np.array(profile["parameters"])
    np.testing.assert_allclose(fitted[:, 1], fitted[:, 0] + 1, atol=1e-4)
    quantity = native["quantities"][0]
    q = np.array(quantity["x"])
    np.testing.assert_allclose(np.array(quantity["objective"]) - native["objective_min"], (q - 7) ** 2 / 10, atol=1e-3)
    assert "endpoints only" in quantity["kind"]
    from scipy.stats import chi2

    quantile = chi2.ppf(0.95, 1)
    assert native["threshold_increment"] == pytest.approx(quantile / 2)
    np.testing.assert_allclose(
        list(profile["native_endpoints"].values()), [3 - np.sqrt(quantile), 3 + np.sqrt(quantile)], atol=0.003
    )
    np.testing.assert_allclose(
        list(quantity["native_endpoints"].values()), [7 - np.sqrt(5 * quantile), 7 + np.sqrt(5 * quantile)], atol=0.001
    )
    assert set(profile["retcodes"].values()) == {"Identifiable"}
    assert set(quantity["retcodes"].values()) == {"Identifiable"}
    assert load_study(study)[1] == before


@pytest.mark.skipif(not os.getenv("SYMBOLIC_LIKELIHOOD_PROJECT"), reason="equation-evidence Docker only")
def test_real_likelihood_tied_bound_and_failed_optimization(synthetic_config, tmp_path):
    tied = {
        "a": {"value": 0.5, "trainable": True, "bounds": [0, 10], "unit": "1"},
        "b": {"value": 0.5, "trainable": False, "bounds": [0, 1], "unit": "1", "tied_to": "a"},
    }
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [{"manual": {"id": "tied", "expression": "a*x+b", "parameters": tied}}],
        },
        run_dir=tmp_path / "tied",
    )
    result = evaluate_equations(
        study,
        "likelihood_profiler",
        run_dir=tmp_path / "bound",
        options={
            "sigma": 1,
            "quantities": [{"id": "response", "prediction_at": {"x": 1}, "range": [0, 30], "unit": "1"}],
        },
    )
    data = read_json(result / "result.json")
    assert data["outcome"] == "unavailable", data
    item = data["results"][0]
    assert item["native"]["outcome"] == "ok"  # The fit executed; the functional analysis is unavailable.
    assert item["native"]["optimum"][0] == pytest.approx(1, abs=1e-7)
    assert "strictly inside" in item["native"]["quantities"][0]["reason"]
    fit = read_json(item["fit_artifact"])
    assert fit["equation"]["parameters"]["a"]["value"] <= 1
    assert fit["equation"]["parameters"]["b"]["value"] <= 1
    assert read_json(result / "job_0.json")["upper"] == [1]

    parameters = {
        k: {"value": v, "trainable": True, "bounds": [-10, 10], "unit": "1"} for k, v in {"a": 0.01, "b": -4.0}.items()
    }
    failed = create_study(
        {
            "prepared_run": load_study(study)[0]["prepared"]["run"],
            "sources": [{"manual": {"id": "early", "expression": "a*x*x+b", "parameters": parameters}}],
        },
        run_dir=tmp_path / "unconverged",
    )
    result = evaluate_equations(
        failed,
        "likelihood_profiler",
        run_dir=tmp_path / "failure",
        options={"sigma": 1, "parameter_ranges": {"a": [-5, 5]}, "maxiters": 1},
    )
    data = read_json(result / "result.json")
    assert data["outcome"] == "failed", data
    assert data["results"][0]["native"]["outcome"] == "failed"
    assert "did not converge" in data["results"][0]["native"]["reason"]


@pytest.mark.skipif(not os.getenv("SYMBOLIC_LIKELIHOOD_PROJECT"), reason="equation-evidence Docker only")
def test_real_redundant_coefficients_do_not_gain_evidence_from_prediction_fit(synthetic_config, tmp_path):
    params = {
        name: {"value": value, "trainable": True, "bounds": [0.1, 10], "unit": "1"}
        for name, value in {"a": 1.0, "b": 2.0}.items()
    }
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [{"manual": {"id": "product", "expression": "a*b*x*x+1", "parameters": params}}],
        },
        run_dir=tmp_path / "study",
    )
    assert load_study(study)[2][0]["metrics"]["selection"]["rmse"] < 1e-10
    result = evaluate_equations(
        study,
        "likelihood_profiler",
        run_dir=tmp_path / "profile",
        options={"sigma": 1.0, "parameter_ranges": {"a": [0.5, 4.0]}, "step": 0.25},
    )
    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    profile = data["results"][0]["native"]["profiles"][0]
    # Compensation b=2/a preserves the same equation. This finite scan does not
    # claim global non-identifiability from an optimizer status alone.
    values = [v for v in profile["delta_nll"] if v is not None]
    assert len(values) > 4 and max(values) < 1e-4
    assert set(profile["retcodes"].values()) != {"Identifiable"}
