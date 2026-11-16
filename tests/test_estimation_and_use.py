from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import (
    assess_prediction_use,
    calibrate_prediction,
    choose_hypothesis,
    compare_equations,
    create_study,
    finalize_hypothesis,
    load_prediction_interval,
    render_report,
)
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.estimation import deming_line
from ml_platform_symbolic.execution_views import bind_view
from ml_platform_symbolic.expressions import parse_expression
from ml_platform_symbolic.hypotheses import predict_working
from ml_platform_symbolic.question_results import conditional_request, missing_precision


def test_deming_recovers_latent_slope_and_coordinate_changes_preserve_it():
    rng = np.random.default_rng(87011)
    latent = rng.normal(size=20000)
    X = (latent + rng.normal(0, 0.7, len(latent)))[:, None]
    y = 4 * latent + 2 + rng.normal(0, 0.3, len(latent))
    variables = [{"id": "x0", "column": "x", "unit": "m"}]
    parameters = {
        "b": {"value": 1.0, "unit": "N/m", "bounds": [0, 10], "trainable": True, "origin": "user_declared"},
        "a": {"value": 0.0, "unit": "N", "bounds": [-10, 10], "trainable": True, "origin": "user_declared"},
    }
    equation = parse_expression("b*x0+a", variables, parameters=parameters, trainable=False)
    original = deepcopy(equation)
    options = dict(feature="x", slope="b", intercept="a", sigma_x=0.7, sigma_y=0.3, sigma_x_unit="m", sigma_y_unit="N")
    fitted, meaning = deming_line(equation, X, y, variables, {"unit": "N"}, options)
    assert fitted["parameters"]["b"]["value"] == pytest.approx(4, abs=0.05)
    assert np.polyfit(X[:, 0], y, 1)[0] < 3.0
    scaled = deepcopy(equation)
    scaled["parameters"]["b"].update(value=0.001, unit="kN/mm", bounds=[0, 0.01])
    scaled["parameters"]["a"].update(value=0, unit="kN", bounds=[-1, 1])
    # Compatible force/length units plus explicitly specified numerical origin shifts.
    converted, _ = deming_line(
        scaled,
        X * 1000 + 7,
        y / 1000 + 0.12,
        [{"id": "x0", "column": "x", "unit": "mm"}],
        {"unit": "kN"},
        {**options, "sigma_x": 700, "sigma_x_unit": "mm", "sigma_y": 0.0003, "sigma_y_unit": "kN"},
    )
    assert converted["parameters"]["b"]["value"] == pytest.approx(fitted["parameters"]["b"]["value"] / 1e6)
    assert equation == original and "latent" in meaning["estimation_target"]


def test_common_precision_question_does_not_depend_on_backend_name():
    request = conditional_request(
        {"expression": "slope"},
        confidence_level=0.95,
        observation_model={"family": "iid_gaussian_known_sigma", "sigma": 0.1, "unit": "N"},
    )
    question = dict(
        request=request,
        candidate_id="law",
        kind="conditional_interval",
        estimand="conditional_model",
        quantity="slope",
        status="established",
        unit="N/m",
        endpoints=[3.9, 4.1],
    )
    analysis = [{"result": {"method": "replacement_evaluator", "results": [{"id": "law", "questions": [question]}]}}]
    assert missing_precision(analysis, "law", {"slope": {"unit": "N/m", "max_width": 0.3, "request": request}}) == []
    assert missing_precision(analysis, "law", {"slope": {"unit": "N/m", "max_width": 0.1, "request": request}})


def test_weighted_views_preserve_mass_and_freeze_settings():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    records = {
        "id": "smc",
        "kind": "ensemble",
        "members": [
            {"equation": parse_expression("x0", variables, trainable=False), "weight": 0.8},
            {"equation": parse_expression("x0+10", variables, trainable=False), "weight": 0.2},
        ],
    }
    mean = bind_view(records)
    high = bind_view(records, {"name": "weighted_quantile", "settings": {"q": 0.9}})
    np.testing.assert_allclose(predict_working(mean, np.array([[1.0]]), variables)[0], [3])
    np.testing.assert_allclose(predict_working(high, np.array([[1.0]]), variables)[0], [11])
    assert mean["members"] == high["members"] and mean["subject"] != high["subject"]
    high["execution_view"]["settings"]["q"] = 0.1
    with pytest.raises(ValueError, match="subject"):
        predict_working(high, np.array([[1.0]]), variables)


def test_frozen_point_calibration_then_use_not_adoption_before_width(synthetic_config, tmp_path):
    study = create_study(
        {
            "problem": synthetic_config,
            "engineering_tolerance": 0.1,
            "sources": [{"manual": {"id": "point", "expression": "2*x*x+1"}}],
        },
        run_dir=tmp_path / "study",
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    policy = dict(
        alpha=0.1,
        exchangeability="exchangeable_groups",
        group_column="group",
        observation_model="independent grouped additive observations",
        max_width=0.1,
        unit="1",
    )
    choice = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "frozen",
        candidate_id="point@manual",
        reason="Fixed before independent calibration",
        calibration_plan=policy,
    )
    assert read_json(choice / "decision.json")["status"] == "frozen_prediction"
    with pytest.raises(ValueError, match="adopted"):
        finalize_hypothesis(study, choice, run_dir=tmp_path / "premature")
    x = np.linspace(0.5, 3, 120)
    calibration = tmp_path / "cal.csv"
    pd.DataFrame(
        dict(
            sample_id=[f"cal{i}" for i in range(120)],
            group=np.repeat([f"calg{i}" for i in range(40)], 3),
            x=x,
            y=2 * x * x + 1 + 0.02,
        )
    ).to_csv(calibration, index=False)
    interval = calibrate_prediction(
        study,
        choice,
        calibration,
        run_dir=tmp_path / "interval",
        **{k: policy[k] for k in ("alpha", "exchangeability", "group_column", "observation_model")},
    )
    use = assess_prediction_use(choice, interval, run_dir=tmp_path / "use")
    assessed = read_json(use / "decision.json")
    assert assessed["status"] == "adopt"
    assert "does not preserve nominal coverage" in assessed["use_scope"]
    assert "conditional on acceptance" in read_json(interval / "interval.json")["limits"]
    assert load_prediction_interval(interval).metadata["candidate_id"] == "point@manual"
    render_report(comparison, output=tmp_path / "report.html", decision_dir=use, prediction_interval=interval)
    from ml_platform_symbolic.report_data import build_report_view

    displayed = build_report_view(comparison, decision_dir=use, prediction_interval=interval)
    assert displayed["decision"]["use_scope"] == assessed["use_scope"]
    result = finalize_hypothesis(study, use, run_dir=tmp_path / "final")
    assert result.artifacts["model_bundle"].is_file()


def test_public_estimator_reads_error_assumptions_from_problem_only(synthetic_config, tmp_path):
    from ml_platform_symbolic import fit_hypotheses

    config = deepcopy(synthetic_config)
    config["observation_model"] = {
        "target_meaning": "observed_response",
        "errors": {
            "kind": "independent_additive_known_sigma",
            "inputs": {"x": {"value": 0.1, "unit": "1"}},
            "target": {"value": 0.2, "unit": "1"},
        },
        "assumptions": "synthetic known homoscedastic error example",
    }
    params = {
        n: {"value": 1.0, "unit": "1", "trainable": True, "bounds": [-100.0, 100.0], "origin": "user_declared"}
        for n in ("slope", "offset")
    }
    study = create_study(
        {
            "problem": config,
            "sources": [{"manual": {"id": "line", "expression": "slope*x+offset", "parameters": params}}],
        },
        run_dir=tmp_path / "study",
    )
    before = (study / "hypotheses.json").read_bytes()
    fits = fit_hypotheses(
        study,
        run_dir=tmp_path / "deming",
        ids=["line@manual"],
        estimator="linear_deming",
        options={"feature": "x", "slope": "slope", "intercept": "offset"},
    )
    artifact = read_json(read_json(fits / "fits.json")["artifacts"][0])
    assert artifact["observation_model"]["variance_ratio"] == pytest.approx(4.0)
    assert artifact["fit_role"] == "train" and (study / "hypotheses.json").read_bytes() == before
    with pytest.raises(ValueError, match="known errors belong"):
        fit_hypotheses(
            study,
            run_dir=tmp_path / "duplicate",
            ids=["line@manual"],
            estimator="linear_deming",
            options={"feature": "x", "slope": "slope", "intercept": "offset", "sigma_x": 0.1},
        )


def test_zero_covariance_deming_horizontal_limit_is_not_misreported_as_unidentified():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    parameters = {
        name: {"value": 0.5, "unit": "1", "trainable": True, "bounds": [-5, 5], "origin": "user_declared"}
        for name in ("a", "b")
    }
    equation = parse_expression("a*x0+b", variables, parameters=parameters, trainable=False)
    fitted, _ = deming_line(
        equation,
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([1.0, 0.0, 1.0]),
        variables,
        {"unit": "1"},
        dict(feature="x", slope="a", intercept="b", sigma_x=1.0, sigma_y=1.0, sigma_x_unit="1", sigma_y_unit="1"),
    )
    assert fitted["parameters"]["a"]["value"] == 0
    assert fitted["parameters"]["b"]["value"] == pytest.approx(2 / 3)


def test_predeclared_prediction_criterion_does_not_turn_selection_adopt_into_final_success():
    from ml_platform_symbolic.selection import prediction_support

    frozen_tolerance = 8.0
    assert prediction_support({"rmse": 7.28}, tolerance=frozen_tolerance)["established"] is True
    result = prediction_support({"rmse": 12.35}, tolerance=frozen_tolerance)
    assert result["established"] is False and result["criteria"]["rmse_max"] == frozen_tolerance
    assert prediction_support({"rmse": 1.0}, tolerance=frozen_tolerance, coverage=0.9)["established"] is False
    assert prediction_support(None, tolerance=frozen_tolerance)["established"] is None


def test_input_error_cannot_promote_fixed_x_interval_to_latent_coefficients():
    request = conditional_request(
        {"expression": "slope"},
        confidence_level=0.95,
        observation_model={"family": "iid_gaussian_known_sigma", "sigma": 0.1, "unit": "N"},
    )
    question = {
        "request": request,
        "kind": "conditional_interval",
        "estimand": "conditional_model",
        "quantity": "slope",
        "status": "established",
        "unit": "N/m",
        "endpoints": [3.0, 3.01],
    }
    analyses = [{"result": {"method": "any_fixed_X_backend", "results": [{"id": "line", "questions": [question]}]}}]
    required = {"slope": {"unit": "N/m", "max_width": 0.1, "request": request}}
    assert not missing_precision(analyses, "line", required, estimand="conditional_model")
    assert missing_precision(analyses, "line", required, estimand="latent_relation")


def test_coefficient_choice_uses_questions_not_prediction_rmse(synthetic_config, tmp_path):
    from ml_platform_symbolic.data import digest, write_json

    request = conditional_request(
        {"expression": "slope"},
        confidence_level=0.95,
        observation_model={"family": "iid_gaussian_known_sigma", "sigma": 0.1, "unit": "1"},
    )
    study = create_study(
        {
            "problem": synthetic_config,
            "purpose": "parameter_interpretation",
            "engineering_tolerance": 1e-8,
            "quantity_tolerances": {"slope": {"unit": "1", "max_width": 0.5, "request": request}},
            "sources": [
                {
                    "manual": {
                        "id": "line",
                        "expression": "slope*x",
                        "parameters": {"slope": {"value": 1.0, "unit": "1", "trainable": True}},
                    }
                }
            ],
        },
        run_dir=tmp_path / "study",
    )
    # Criteria-only fixture, not claimed native interval evidence.
    analysis = write_json(
        tmp_path / "fixture.json",
        {
            "method": "replacement",
            "outcome": "ok",
            "study_sha256": digest(study / "study.json"),
            "results": [
                {
                    "id": "line@manual",
                    "outcome": "ok",
                    "questions": [
                        {
                            "kind": "conditional_interval",
                            "request": request,
                            "estimand": "conditional_model",
                            "quantity": "slope",
                            "unit": "1",
                            "status": "established",
                            "endpoints": [0.9, 1.1],
                        }
                    ],
                }
            ],
        },
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison", evaluations=[analysis])
    choice = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "choice",
        candidate_id="line@manual",
        reason="Conditional coefficient question only",
    )
    assert read_json(choice / "decision.json")["status"] == "adopt"
    result = finalize_hypothesis(study, choice, run_dir=tmp_path / "result")
    assert set(result.artifacts) == {"coefficient_result"} and result.extra["final_accessed"] is False
    assert not list(tmp_path.rglob("final_access.json"))
