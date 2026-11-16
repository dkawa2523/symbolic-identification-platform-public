import numpy as np
import pytest
from ml_platform_symbolic.engines.pysr import export_expression, native_constraints
from ml_platform_symbolic.expressions import evaluate


def test_native_square_and_cube_keep_literal_exponents():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = export_expression("1.234567891*square(x0) + cube(x0)", variables)
    assert [p["value"] for p in equation["parameters"].values()] == [1.234567891]
    X = np.array([[-2.0], [0.0], [1.37]])
    np.testing.assert_allclose(evaluate(equation, X, variables), 1.234567891 * X[:, 0] ** 2 + X[:, 0] ** 3)
    with pytest.raises(ValueError):
        export_expression("square(__import__('os'))", variables)


def test_native_restrictions_use_requested_common_operator_names():
    arguments, nested = native_constraints(
        {"operator_constraints": {"div": [-1, 8], "sin": 8}, "nested_constraints": {"sin": {"sin": 0}}},
        ["div", "sin"],
    )
    assert arguments == {"/": (-1, 8), "sin": 8}
    assert nested == {"sin": {"sin": 0}}
    with pytest.raises(ValueError, match="inactive"):
        native_constraints({"operator_constraints": {"log": 8}}, ["sin"])


def test_native_fitted_power_is_not_a_fixed_structural_exponent():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = export_expression("2.37*x0^0.41+square(x0)", variables)
    assert [p["value"] for p in equation["parameters"].values()] == [2.37, 0.41]
    assert all(p["origin"] == "pysr_native_constant_node" for p in equation["parameters"].values())
    X = np.array([[0.2], [1.3], [2.7]])
    np.testing.assert_allclose(evaluate(equation, X, variables), 2.37 * X[:, 0] ** 0.41 + X[:, 0] ** 2)
