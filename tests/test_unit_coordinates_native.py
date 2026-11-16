"""Real optional-library coordinate checks; retained with --basetemp in the task output."""

import importlib.util
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from current_flow import explicit_choice, study_flow
from ml_platform_symbolic import choose_hypothesis, compare_equations, finalize_hypothesis, load_model, render_report
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import prepare, read_json, write_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.equation_evaluation import create_study, evaluate_equations, load_study
from ml_platform_symbolic.question_results import conditional_request

pytestmark = pytest.mark.native


@pytest.mark.skipif(not os.getenv("SYMBOLIC_LIKELIHOOD_PROJECT"), reason="equation-evidence image only")
def test_real_pressure_interval_conversion_and_precision_choice(tmp_path):
    x = np.linspace(-1, 1, 40)
    y = 1 + 0.2 * x + np.random.default_rng(321).normal(0, 0.01, len(x))
    frame = pd.DataFrame({"sample_id": [f"p{i}" for i in range(len(x))], "x": x, "pressure": y})
    frame.to_csv(tmp_path / "MPa.csv", index=False)
    frame.assign(pressure=y * 1e6).to_csv(tmp_path / "Pa.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 30 + ["selection"] * 10).to_csv(
        tmp_path / "split.csv", index=False
    )
    parameters = {
        k: {"value": v, "trainable": True, "bounds": bounds, "unit": "Pa", "origin": "user_declared"}
        for k, v, bounds in [("p", 1e6, [0.5e6, 1.5e6]), ("q", 0.2e6, [-1e6, 1e6])]
    }
    common = {
        "split": {"file": str(tmp_path / "split.csv")},
        "evaluation": {"mode": "exploratory"},
        "constraints": {"require_units": True},
        "search": {"engines": ["mean"], "operators": ["add", "mul"]},
    }
    plans = []
    for unit, width in [("Pa", 1), ("MPa", 1e-6), ("Pa", 20000), ("MPa", 0.02)]:
        problem = deepcopy(common)
        problem["data"] = {
            "local_path": str(tmp_path / f"{unit}.csv"),
            "feature_columns": ["x"],
            "target_column": "pressure",
            "units": {"x": "1", "pressure": unit},
            "domain": {"x": [-1, 1]},
        }
        plans.append(
            {
                "problem": problem,
                "purpose": "parameter_interpretation",
                "engineering_tolerance": 1e5,
                "required_evidence": ["likelihood_profiler"],
                "quantity_tolerances": {
                    "at_zero": {
                        "unit": unit,
                        "max_width": width,
                        "request": conditional_request(
                            {"prediction_at": {"x": 0}, "input_units": {"x": "1"}},
                            confidence_level=0.95,
                            observation_model={"family": "iid_gaussian_known_sigma", "sigma": 0.01, "unit": "MPa"},
                        ),
                    }
                },
                "sources": [{"manual": {"id": "pressure", "expression": "p+q*x", "parameters": parameters}}],
            }
        )
    outcomes = []
    intervals = []
    for index, plan in enumerate(plans):
        unit = next(iter(plan["quantity_tolerances"].values()))["unit"]
        factor = 1e6 if unit == "Pa" else 1
        study = create_study(plan, run_dir=tmp_path / f"study{index}")
        before = load_study(study)[1]
        result = evaluate_equations(
            study,
            "likelihood_profiler",
            run_dir=tmp_path / f"profile{index}",
            options={
                "sigma": 0.01,
                "sigma_unit": "MPa",
                "parameter_ranges": {"p": [0.8, 1.2]},
                "parameter_units": {"p": "MPa"},
                "max_wall_seconds": 180,
                "maxiters": 500,
                "quantities": [
                    {"id": "at_zero", "prediction_at": {"x": 0}, "range": [0.8 * factor, 1.2 * factor], "unit": unit}
                ],
            },
        )
        data = read_json(result / "result.json")
        assert data["outcome"] == "ok", data
        native = data["results"][0]["native"]
        design = np.column_stack([np.ones(30), x[:30]])
        least_squares = np.linalg.lstsq(design, y[:30] * 1e6, rcond=None)[0]
        np.testing.assert_allclose(native["optimum"], least_squares, rtol=1e-6, atol=0.1)
        quantity = native["quantities"][0]
        assert quantity["endpoint_validation"]["all_verified"], quantity
        interval = np.array(list(quantity["accepted_endpoints"].values()))
        intervals.append(interval / factor)
        assert 0.003 < abs(interval[1] - interval[0]) / factor < 0.02
        profile = native["profiles"][0]
        assert profile["unit"] == "MPa"
        assert max(profile["x"]) < 2
        comparison = compare_equations(study, evaluations=[result], run_dir=tmp_path / f"comparison{index}")
        decision = choose_hypothesis(
            comparison,
            run_dir=tmp_path / f"decision{index}",
            candidate_id="pressure@manual",
            reason="Predetermined conditional pressure precision, not mechanism discovery",
        )
        outcomes.append(read_json(decision / "decision.json")["status"])
        report = render_report(comparison, output=tmp_path / f"report{index}.html", decision_dir=decision)
        assert "CSV/public unit" in report.read_text()
        if index == 2:
            legacy = deepcopy(data)
            legacy.pop("coordinate_policy")
            legacy_path = write_json(tmp_path / "historical_unverified.json", legacy)
            legacy_comparison = compare_equations(
                study, evaluations=[legacy_path], run_dir=tmp_path / "legacy_comparison"
            )
            legacy_decision = choose_hypothesis(
                legacy_comparison,
                run_dir=tmp_path / "legacy_decision",
                candidate_id="pressure@manual",
                reason="Historical coordinate assurances are unverified; raw native evidence retained",
            )
            assert read_json(legacy_decision / "decision.json")["status"] == "defer"
            legacy_report = render_report(
                legacy_comparison, output=tmp_path / "legacy_report.html", decision_dir=legacy_decision
            )
            assert "Historical unit coordinates unverified" in legacy_report.read_text()
        assert load_study(study)[1] == before
    assert outcomes == ["defer", "defer", "adopt", "adopt"]
    np.testing.assert_allclose(intervals, np.repeat([intervals[0]], 4, axis=0), atol=2e-5)


@pytest.mark.skipif(not os.getenv("SYMBOLIC_LIKELIHOOD_PROJECT"), reason="equation-evidence image only")
def test_real_absolute_temperature_functional_and_difference_sigma(tmp_path):
    x = np.linspace(-1, 1, 30)
    frame = pd.DataFrame(
        {
            "sample_id": [f"t{i}" for i in range(30)],
            "x": x,
            "temperature": 26.85 + 2 * x + np.random.default_rng(9).normal(0, 0.1, 30),
        }
    )
    frame.to_csv(tmp_path / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 20 + ["selection"] * 10).to_csv(
        tmp_path / "split.csv", index=False
    )
    parameters = {
        name: {
            "value": value,
            "trainable": True,
            "bounds": bounds,
            "unit": "K",
            "coordinate_role": "value" if name == "p" else "difference",
        }
        for name, value, bounds in [("p", 300, [280, 330]), ("q", 2, [-10, 10])]
    }
    study = create_study(
        {
            "problem": {
                "data": {
                    "local_path": str(tmp_path / "data.csv"),
                    "feature_columns": ["x"],
                    "target_column": "temperature",
                    "units": {"x": "1", "temperature": "degC"},
                    "coordinate_roles": {"temperature": "value"},
                },
                "split": {"file": str(tmp_path / "split.csv")},
                "evaluation": {"mode": "exploratory"},
                "search": {"engines": ["mean"], "operators": ["add", "mul"]},
                "constraints": {"require_units": True},
            },
            "sources": [{"manual": {"id": "temperature", "expression": "p+q*x", "parameters": parameters}}],
        },
        run_dir=tmp_path / "study",
    )
    result = evaluate_equations(
        study,
        "likelihood_profiler",
        run_dir=tmp_path / "profiles",
        options={
            "sigma": 0.1,
            "sigma_unit": "degC",
            "max_wall_seconds": 180,
            "parameter_ranges": {"p": [20, 35], "q": [0, 5]},
            "parameter_units": {"p": "degC", "q": "degC"},
            "quantities": [
                {"id": "at_zero", "prediction_at": {"x": 0}, "range": [20, 35], "unit": "degC"},
                {"id": "difference", "expression": "p-(p-q)", "range": [0, 5], "unit": "degC"},
            ],
        },
    )
    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    native = data["results"][0]["native"]
    profile, quantity = native["profiles"][0], native["quantities"][0]
    assert profile["endpoint_validation"]["all_verified"] and quantity["endpoint_validation"]["all_verified"]
    assert read_json(result / "job_0.json")["sigma"] == 0.1
    for entry in (profile, quantity):
        low, high = entry["accepted_endpoints"].values()
        assert 25 < low < high < 28
        assert high - low < 0.3
    difference = native["quantities"][1]
    assert difference["coordinate_role"] == "difference"
    assert difference["endpoint_validation"]["all_verified"]
    assert all(1 < value < 3 for value in difference["accepted_endpoints"].values())
    assert native["profiles"][1]["coordinate_role"] == "difference"
    assert all(1 < value < 3 for value in native["profiles"][1]["accepted_endpoints"].values())


@pytest.mark.skipif(importlib.util.find_spec("TorchSisso") is None, reason="classic image only")
def test_real_torchsisso_composite_target_after_cm_boundary(tmp_path):
    x = np.linspace(0.2, 2, 45)
    frame = pd.DataFrame(
        {"sample_id": [f"a{i}" for i in range(45)], "length": x * 100, "area": (1.23456789 * x * x + 0.37) * 1e4}
    )
    frame.to_csv(tmp_path / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 30 + ["selection"] * 15).to_csv(
        tmp_path / "split.csv", index=False
    )
    _, _, study = study_flow(
        {
            "data": {
                "local_path": str(tmp_path / "data.csv"),
                "feature_columns": ["length"],
                "target_column": "area",
                "units": {"length": "cm", "area": "cm^2"},
                "domain": {"length": [10, 220], "area": [0, 10e4]},
            },
            "split": {"file": str(tmp_path / "split.csv")},
            "evaluation": {"mode": "exploratory", "selection_policy": "accuracy"},
            "constraints": {"require_units": True},
            "search": {
                "engines": ["torchsisso"],
                "operators": ["add", "mul", "pow_int"],
                "max_wall_seconds": 60,
                "engine_options": {"torchsisso": {"dimensional": True, "n_expansion": 2, "n_term": 1, "k": 4}},
            },
        },
        root=tmp_path / "run", purpose="local_prediction", engineering_tolerance=100,
    )
    _, decision = explicit_choice(study, tmp_path / "run")
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / "run/final-audit")
    model = load_model(result.artifacts["model_bundle"])
    assert model.schema["target"]["unit"] == "m^(2)"
    assert result.extra["selected_native_check"]["passed"]
    probe = np.array([25.0, 55.0, 175.0])
    expected = (1.23456789 * (probe / 100) ** 2 + 0.37) * 1e4
    np.testing.assert_allclose(model.predict(pd.DataFrame({"length": probe})), expected, rtol=1e-10)
    np.testing.assert_allclose(
        model.predict(pd.DataFrame({"length": probe / 100}), units={"length": "m"}, output_unit="m^2"),
        expected / 1e4,
        rtol=1e-10,
    )


@pytest.mark.skipif(importlib.util.find_spec("physo") is None, reason="gpu image only")
def test_real_physo_nested_alias_catalog_is_executable(tmp_path):
    rng = np.random.default_rng(223)
    a = rng.uniform(0.2, 2, 32)
    t = rng.uniform(0.3, 1.7, 32)
    frame = pd.DataFrame(
        {"sample_id": [f"r{i}" for i in range(32)], "acceleration": a, "time": t, "height": 0.37 + 0.71 * a * t * t}
    )
    frame.to_csv(tmp_path / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 24 + ["selection"] * 8).to_csv(
        tmp_path / "split.csv", index=False
    )
    cfg = resolve_config(
        {
            "data": {
                "local_path": str(tmp_path / "data.csv"),
                "feature_columns": ["acceleration", "time"],
                "target_column": "height",
                "units": {"acceleration": "m/s2", "time": "s", "height": "m"},
            },
            "split": {"file": str(tmp_path / "split.csv")},
            "evaluation": {"mode": "exploratory"},
            "constraints": {"require_units": True},
            "search": {
                "engines": ["physo"],
                "seeds": [7],
                "operators": ["add", "mul", "pow_int"],
                "max_wall_seconds": 90,
                "engine_options": {
                    "physo": {"epochs": 2, "batch_size": 32, "max_n_evaluations": 64, "constants_per_unit": 1}
                },
            },
        }
    )
    refs = prepare(cfg, tmp_path / "prepared")
    discovery = run_discovery(refs, read_json(refs["run"]), "physo", 7, tmp_path / "discover")
    summary = read_json(discovery["summary"])
    assert summary["native_execution_status"] == "completed", summary
    assert summary["status"] == "ok", summary
    assert summary["native_candidates"] > 0
    assert summary["parameter_units"]
