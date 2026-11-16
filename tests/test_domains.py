"""Observed scientific failures: outside-domain probes and real-power parameter identity."""

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import load_model, run_task
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.constraints import check_constraints
from ml_platform_symbolic.domains import validation_constraints
from ml_platform_symbolic.engines.base import DiscoveryRequest, checked_candidate
from ml_platform_symbolic.expressions import equation_text, evaluate, parse_expression
from ml_platform_symbolic.fitting import fit_coefficients


def test_arrhenius_parity_uses_independent_points_in_declared_temperature_domain():
    variables = [{"id": "x0", "column": "T", "unit": "K", "domain": [300, 900]}]
    equation = parse_expression("exp(-810/x0)", variables, trainable=False)
    X = np.array([[400.0], [600.0], [800.0]])
    request = DiscoveryRequest(
        X, np.exp(-810 / X[:, 0]), np.arange(3), variables, {"unit": "1"}, ["exp", "div", "neg"], {}, 42, {}
    )
    observed = []

    def native(probes):
        observed.append(probes.copy())
        return np.exp(-810 / probes[:, 0])

    candidate = checked_candidate("native", 0, equation, request, native)
    assert candidate.native_check["passed"]
    assert observed[0].min() >= 300 and observed[0].max() <= 900
    assert len(np.unique(observed[0])) > len(X)


def test_real_power_preserves_free_exponent():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    params = {"C": {"value": 1.37, "trainable": True, "bounds": [0.2, 2.0], "unit": "1"}}
    equation = parse_expression("x0**C", variables, parameters=params)
    assert equation["parameters"] == params
    np.testing.assert_allclose(evaluate(equation, np.array([[0.2], [2.0]]), variables), np.array([0.2, 2.0]) ** 1.37)


def test_domain_roundtrip_and_outside_input_rejected(synthetic_config, tmp_path):
    synthetic_config["data"]["domain"] = {"x": [0.1, 10], "y": [0, 300]}
    from current_flow import study_flow
    from ml_platform_symbolic.equation_evaluation import load_study
    from ml_platform_symbolic.model import save_model
    refs, discoveries, study = study_flow(synthetic_config, tmp_path / 'run')
    context, records, rows, _ = load_study(study)
    schema = {key: context[key] for key in ('variables', 'target', 'unit_coordinates', 'input_geometry', 'feature_ranges') if key in context}
    bundle = save_model(tmp_path / 'model.zip', records[0]['equation'], schema, {})
    model = load_model(bundle)
    assert model.schema["variables"][0]["domain"] == [0.1, 10]
    assert model.schema["target"]["domain"] == [0, 300]
    with pytest.raises(ValueError, match="outside declared domain"):
        model.predict(pd.DataFrame({"x": [0.0]}))


def test_target_domain_is_checked_beyond_training_range(synthetic_config):
    cfg = resolve_config(synthetic_config)
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0.1, 3.0]}]
    target = {"unit": "1", "domain": [0.0, 1.0]}
    task_constraints = validation_constraints(cfg, variables, target, np.array([[0.1], [0.2]]))
    equation = parse_expression("x0", variables)
    checks = check_constraints(equation, variables, target, ["add", "mul"], task_constraints)
    assert any(c["name"] == "target_domain_validity" and c["status"] == "fail" for c in checks)


@pytest.mark.parametrize("outer", ["tanh", "atan"])
def test_internal_shifted_pole_is_not_hidden_by_random_domain_probes(outer):
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0.2, 3.0]}]
    equation = parse_expression(f"{outer}(1/(x0-.37))", variables, trainable=False)
    X = np.array([[0.5], [1.0], [2.0]])
    request = DiscoveryRequest(X, np.ones(3), np.arange(3), variables, {"unit": "1"}, [outer, "div", "sub"], {}, 42, {})
    with pytest.raises(ValueError, match="IEEE"):
        checked_candidate("native", 0, equation, request, lambda probes: getattr(np, outer)(1 / (probes[:, 0] - 0.37)))


def test_literal_fractional_exponent_stays_fixed_during_refit_and_display():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("2*x0**(1/3)", variables)
    assert len(equation["parameters"]) == 1
    X = np.linspace(0.1, 2, 20).reshape(-1, 1)
    fitted = fit_coefficients(equation, X, 4 * X[:, 0] ** (1 / 3), variables)
    np.testing.assert_allclose(evaluate(fitted, X, variables), 4 * X[:, 0] ** (1 / 3), rtol=1e-7)
    assert fitted["ast"]["args"][1] == equation["ast"]["args"][1]
    assert "**" in equation_text(equation, variables) and "1 / 3" in equation_text(equation, variables)
    root = {"schema_version": 1, "ast": {"op": "rootn", "args": [{"var": "x0"}, {"const": -3}]}, "parameters": {}}
    assert equation_text(root, variables) == "rootn(x, -3)"


def test_dimensioned_exponent_fails_even_with_unknown_base_units():
    variables = [{"id": "x0", "column": "x", "unit": None}]
    equation = parse_expression(
        "x0**E", variables, parameters={"E": {"value": 1.3, "trainable": True, "bounds": [None, None], "unit": "K"}}
    )
    checks = check_constraints(equation, variables, {"unit": None}, ["pow"], {"max_nodes": 30, "require_units": False})
    assert any(c["name"] == "units" and c["status"] == "fail" and c["required"] for c in checks)


def test_unsupported_required_fixed_constant_prevents_execution(synthetic_config, tmp_path):
    synthetic_config["data"]["fixed_constants"] = {"physical_constant": {"value": 1.23, "unit": "1"}}
    with pytest.raises(RuntimeError, match="fixed constants are unsupported"):
        from current_flow import prepare_discover
        prepare_discover(synthetic_config, tmp_path / "unsupported")
    import json

    summary = json.loads((tmp_path / "unsupported/discover/dictionary/42/discover/engine_summary.json").read_text())
    assert summary["native_execution_status"] == "not_started"
    assert not (tmp_path / "unsupported/discover/dictionary/42/discover/candidates.jsonl").exists()


def test_display_of_negative_fitted_power_base_keeps_precedence():
    equation = {
        "schema_version": 1,
        "ast": {"op": "pow_int", "args": [{"param": "c0"}, {"const": 4}]},
        "parameters": {"c0": {"value": -0.84, "trainable": True, "unit": "1", "bounds": [None, None]}},
    }
    rendered = equation_text(equation, [])
    replay = parse_expression(rendered, [], trainable=False)
    np.testing.assert_allclose(evaluate(replay, np.empty((2, 0)), []), [0.84**4, 0.84**4])
