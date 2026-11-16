import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.constraints import check_constraints
from ml_platform_symbolic.expressions import evaluate, lower_integer_powers, parse_expression
from ml_platform_symbolic.fitting import fit_coefficients
from ml_platform_symbolic.model import load_model, save_model

VARIABLES = [{"id": "x0", "column": "temperature", "unit": "1"}]


def test_fixed_exponent_shared_coefficient_and_bundle(tmp_path):
    eq = parse_expression("1.0*x0**2 + 3.0", VARIABLES)
    X = np.linspace(1, 2, 30).reshape(-1, 1)
    fitted = fit_coefficients(eq, X, 2 * X[:, 0] ** 2 + 5, VARIABLES)
    assert eq["parameters"]["c0"]["value"] == 1
    assert fitted["ast"]["args"][0]["args"][1]["args"][1] == {"const": 2}
    schema = {"variables": VARIABLES, "feature_ranges": [[1, 2]]}
    path = save_model(tmp_path / "model.zip", fitted, schema, {})
    prediction = load_model(path).predict(pd.DataFrame({"ignored": 99, "temperature": X[:, 0]}))
    np.testing.assert_allclose(prediction, 2 * X[:, 0] ** 2 + 5)
    with pytest.raises(ValueError, match="Missing"):
        load_model(path).predict(pd.DataFrame({"wrong": [1]}))
    # Hash verification is independent of ZIP CRC verification.
    with zipfile.ZipFile(path) as source:
        payloads = {n: source.read(n) for n in source.namelist()}
    payloads["equation.json"] = json.dumps(eq).encode()
    with zipfile.ZipFile(tmp_path / "tampered.zip", "w") as bundle:
        for name, value in payloads.items():
            bundle.writestr(name, value)
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path / "tampered.zip")


@pytest.mark.parametrize("text", ["x0/x0", "log(x0)", "1/x0"])
def test_real_domain_is_not_repaired(text):
    eq = parse_expression(text, VARIABLES)
    assert not np.isfinite(evaluate(eq, np.array([[0.0]]), VARIABLES)).all()


def test_constraint_failures_are_separate_from_prediction_quality():
    variables = [{"id": "x0", "column": "length", "unit": "m"}]
    eq = parse_expression("x0+x0", variables)
    checks = check_constraints(eq, variables, {"unit": "s"}, ["add"], {"max_nodes": 30, "require_units": False})
    assert any(c["name"] == "units" and c["status"] == "fail" and c["required"] for c in checks)
    checks = check_constraints(eq, variables, {"unit": None}, ["add"], {"max_nodes": 30, "require_units": True})
    assert any(c["status"] == "unknown" and c["required"] for c in checks)


def test_engine_coefficients_do_not_inherit_input_units():
    equation = parse_expression("2*x0 + 1", VARIABLES)
    assert all(parameter["unit"] is None for parameter in equation["parameters"].values())
    checks = check_constraints(
        equation, VARIABLES, {"unit": "s"}, ["add", "mul"], {"max_nodes": 30, "require_units": True}
    )
    assert any(c["name"] == "units" and c["status"] == "unknown" and c["required"] for c in checks)


@pytest.mark.parametrize("exponent", [-3, 0, 2, 3])
def test_native_power_lowering_retains_holes_and_fixed_exponents(exponent):
    equation = parse_expression(f"(1/x0)**{exponent}", VARIABLES, trainable=False)
    lowered = lower_integer_powers(equation)
    X = np.array([[-2.0], [0.0], [2.0]])
    original, portable = evaluate(equation, X, VARIABLES), evaluate(lowered, X, VARIABLES)
    assert np.isnan(original[1]) and np.isnan(portable[1])
    np.testing.assert_array_equal(np.isfinite(original), np.isfinite(portable))
    np.testing.assert_allclose(original[np.isfinite(original)], portable[np.isfinite(portable)])
    assert lowered["parameters"] == equation["parameters"] == {}


@pytest.mark.parametrize("text", ["__import__('os')", "clip(x0)", "x9"])
def test_unsafe_or_unportable_expressions_rejected(text):
    with pytest.raises(ValueError):
        parse_expression(text, VARIABLES)


def test_fixed_power_canonicalization_preserves_holes_and_parameter_identity():
    from copy import deepcopy

    from ml_platform_symbolic.expressions import canonical_integer_powers, evaluate, evaluate_native, inspect_equation

    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    fixed = {
        "schema_version": 2,
        "arithmetic": "numpy_ieee",
        "parameters": {},
        "ast": {"op": "pow", "args": [{"var": "x0"}, {"const": -1}]},
    }
    old = deepcopy(fixed)
    converted = canonical_integer_powers(fixed)
    assert fixed == old
    assert inspect_equation(converted, variables)["operators"] == {"pow_int"}
    X = np.array([[-2.0], [-0.0], [0.0], [0.5]])
    np.testing.assert_array_equal(evaluate_native(converted, X, variables), evaluate_native(old, X, variables))
    assert np.isnan(evaluate(converted, X, variables)[1:3]).all()
    free = deepcopy(fixed)
    free["parameters"] = {"p": {"value": -1.0, "trainable": True, "unit": "1", "bounds": [None, None]}}
    free["ast"]["args"][1] = {"param": "p"}
    assert canonical_integer_powers(free) == free
