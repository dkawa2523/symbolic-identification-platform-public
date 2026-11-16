"""Original-scale RASR export and explicit common transcendental semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.constraints import check_constraints
from ml_platform_symbolic.engines.rasr import export_candidate
from ml_platform_symbolic.expressions import evaluate


def test_rasr_normalization_is_fixed_and_native_coefficients_are_free():
    variables = [{"id": "x0", "column": "length", "unit": "m"}]
    native = SimpleNamespace(tokens=["+", "*", "<constant>", "x1", "<constant>"], constants=[1.23456789123456, 0.3])
    fitted = SimpleNamespace(X_mean_=np.array([2.0]), X_std_=np.array([3.0]), y_mean_=7.0, y_std_scale_=5.0)
    equation = export_candidate(native, fitted, variables, {"unit": "s"})
    X = np.array([[0.0], [2.0], [8.0]])
    np.testing.assert_allclose(
        evaluate(equation, X, variables), (((X[:, 0] - 2) / 3) * native.constants[0] + 0.3) * 5 + 7
    )
    assert sum(p["trainable"] for p in equation["parameters"].values()) == 2
    assert equation["parameters"]["x0_scale"]["unit"] == "m"
    assert equation["parameters"]["y_scale"]["unit"] == "s"


@pytest.mark.parametrize("op", ["sin", "cos", "tanh", "sinh", "cosh"])
def test_transcendental_numeric_semantics_and_dimension_requirement(op):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = {"schema_version": 1, "ast": {"op": op, "args": [{"var": "x0"}]}, "parameters": {}}
    X = np.array([[-1.0], [0.0], [np.pi / 2]])
    np.testing.assert_array_equal(evaluate(equation, X, variables), getattr(np, op)(X[:, 0]))
    variables[0]["unit"] = "m"
    checks = check_constraints(equation, variables, {"unit": "1"}, [op], {"max_nodes": 30, "require_units": True})
    assert any(c["name"] == "units" and c["status"] == "fail" for c in checks)


def test_rasr_ieee_roundtrip_preserves_invalid_intermediate(tmp_path):
    import json

    from ml_platform_symbolic.expressions import evaluate_native

    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    native = SimpleNamespace(tokens=["inv", "inv", "x1"], constants=[])
    fitted = SimpleNamespace(X_mean_=np.array([0.0]), X_std_=np.array([1.0]), y_mean_=0.0, y_std_scale_=1.0)
    equation = export_candidate(native, fitted, variables, {"unit": "1"})
    path = tmp_path / "equation.json"
    path.write_text(json.dumps(equation))
    restored = json.loads(path.read_text())
    points = np.array([[0.0], [2.0], [-2.0]])
    assert restored["arithmetic"] == "numpy_ieee"
    np.testing.assert_array_equal(evaluate_native(restored, points, variables), [0.0, 2.0, -2.0])
    common = evaluate(restored, points, variables)
    assert np.isnan(common[0])
    np.testing.assert_array_equal(common[1:], [2.0, -2.0])
