from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.expressions import evaluate, parse_expression
from ml_platform_symbolic.fitting import fit_coefficients


@pytest.mark.parametrize("x_scale", [1e-10, 1.0, 1e10])
@pytest.mark.parametrize("y_scale", [1e-10, 1.0, 1e10])
def test_fit_is_equivariant_to_input_and_output_coordinates(x_scale, y_scale):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    parameters = {
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": [None, None], "origin": "user_declared"},
        "fixed": {"value": 2.0, "unit": "1", "trainable": False, "bounds": [2, 2], "origin": "fixed_literal"},
    }
    equation = parse_expression("a*x0+fixed*0", variables, parameters=parameters, trainable=False)
    original = deepcopy(equation)
    X = (np.linspace(0, 1, 20) * x_scale)[:, None]
    y = np.linspace(0, 1, 20) * y_scale
    fitted = fit_coefficients(equation, X, y, variables)
    np.testing.assert_allclose(evaluate(fitted, X, variables) / y_scale, y / y_scale, rtol=1e-7, atol=1e-12)
    assert equation == original
    assert fitted["parameters"]["fixed"] == original["parameters"]["fixed"]


@pytest.mark.parametrize("target", [0.0, 1e-10, 1.0, 1e10])
def test_constant_target_fit_retains_real_bounds_and_ties(target):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    scale = abs(target) or 1.0
    parameters = {
        "a": {"value": 0.5 * scale, "unit": "1", "trainable": True, "bounds": [-2 * scale, 2 * scale]},
        "alias": {"value": 0.5 * scale, "unit": "1", "trainable": False, "tied_to": "a", "bounds": [-2 * scale, 2 * scale]},
    }
    equation = parse_expression("alias", variables, parameters=parameters, trainable=False)
    fitted = fit_coefficients(equation, np.arange(20)[:, None], np.full(20, target), variables)
    assert fitted["parameters"]["a"]["value"] / scale == pytest.approx(target / scale, abs=1e-7)
    assert fitted["parameters"]["alias"]["value"] == fitted["parameters"]["a"]["value"]



@pytest.mark.parametrize("bounds, expected", [([0, None], 1.0), ([0, 10], 1.0), ([None, 0], -1.0)])
def test_fit_leaves_a_feasible_initial_bound_when_data_require_it(bounds, expected):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("a*x0", variables, parameters={
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": bounds}
    }, trainable=False)
    X = np.linspace(0, 1, 20)[:, None]
    fitted = fit_coefficients(equation, X, expected * X[:, 0], variables)
    assert fitted["parameters"]["a"]["value"] == pytest.approx(expected, rel=1e-7)


def test_nonlinear_fit_from_initial_bound_preserves_bounded_optimum():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("exp(a*x0)", variables, parameters={
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": [0, 3]}
    }, trainable=False)
    X = np.linspace(0, 1, 20)[:, None]
    fitted = fit_coefficients(equation, X, np.exp(2 * X[:, 0]), variables)
    assert fitted["parameters"]["a"]["value"] == pytest.approx(2, rel=1e-7)
    # An unreachable target retains the declared bound rather than clipping
    # or changing the source family's constraints.
    constrained = fit_coefficients(equation, X, np.exp(4 * X[:, 0]), variables)
    assert constrained["parameters"]["a"]["value"] == pytest.approx(3, rel=1e-7)


def test_fit_does_not_turn_an_exhausted_budget_into_a_fitted_variant():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("a*x0", variables, parameters={
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": [0, 10]}
    }, trainable=False)
    X = np.linspace(0, 1, 20)[:, None]
    with pytest.raises(ValueError, match="Coefficient fit failed"):
        fit_coefficients(equation, X, X[:, 0], variables, max_nfev=1)


@pytest.mark.parametrize("kind", ["nan", "empty", "shape"])
def test_fit_rejects_invalid_observation_arrays(kind):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("a*x0", variables, parameters={
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": [None, None]}
    }, trainable=False)
    X = np.linspace(0, 1, 20)[:, None]
    y = X[:, 0].copy()
    if kind == "nan":
        y[3] = np.nan
    elif kind == "empty":
        X, y = X[:0], y[:0]
    else:
        y = y[:, None]
    with pytest.raises(ValueError):
        fit_coefficients(equation, X, y, variables)


@pytest.mark.parametrize("target, lower, upper, expected", [
    (-2.0, 0.0, np.inf, 0.0),
    (3.0, -np.inf, 1.0, 1.0),
    (0.25, 0.0, 1.0, 0.25),
    (3.0, 1.0, 1.0, 1.0),
])
def test_explicit_response_constraints_preserve_shared_and_fixed_parameters(target, lower, upper, expected):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("a+alias+fixed", variables, parameters={
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": [-10, 10]},
        "alias": {"value": 0.0, "unit": "1", "trainable": False, "tied_to": "a", "bounds": [-10, 10]},
        "fixed": {"value": 1.0, "unit": "1", "trainable": False},
    }, trainable=False)
    original = deepcopy(equation)
    X = np.ones((10, 1))
    def values(eq):
        return evaluate(eq, X[:1], variables)
    fitted = fit_coefficients(equation, X, np.full(10, target), variables,
                              data_scale=1.0, constraints=(values, [lower], [upper]))
    value = values(fitted)[0]
    assert lower <= value <= upper
    assert value == pytest.approx(expected, abs=1e-6)
    assert fitted["parameters"]["a"]["value"] == fitted["parameters"]["alias"]["value"]
    assert fitted["parameters"]["fixed"] == original["parameters"]["fixed"]
    assert equation == original


@pytest.mark.parametrize("lower, upper", [(2.0, 1.0), (100.0, np.inf)])
def test_explicit_response_infeasibility_never_returns_smallest_violation(lower, upper):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("a", variables, parameters={
        "a": {"value": 0.0, "unit": "1", "trainable": True, "bounds": [-1, 1]},
    }, trainable=False)
    X = np.ones((4, 1))
    with pytest.raises(ValueError, match="Contradictory|No feasible"):
        fit_coefficients(equation, X, np.zeros(4), variables, data_scale=1.0,
                         constraints=(lambda eq: evaluate(eq, X[:1], variables), [lower], [upper]))


def test_constrained_fit_reports_budget_separately_from_feasible_candidate():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = parse_expression("a", variables, parameters={
        "a": {"value": 1.0, "unit": "1", "trainable": True, "bounds": [-10, 10]},
    }, trainable=False)
    X, diagnostics = np.ones((4, 1)), {}
    fitted = fit_coefficients(equation, X, np.zeros(4), variables, data_scale=1.0,
                             max_nfev=1, diagnostics=diagnostics,
                             constraints=(lambda eq: evaluate(eq, X[:1], variables), [0.0], [np.inf]))
    assert evaluate(fitted, X, variables).min() >= 0
    assert diagnostics["nfev"] == 1
    assert not diagnostics["success"]
    assert diagnostics["termination"] == "objective evaluation budget"


def test_inward_solver_bounds_do_not_relax_original_conditions():
    from ml_platform_symbolic.response_search import inward_response_bounds
    lower = np.array([0.0, -np.inf, 1.0, 2.0])
    upper = np.array([np.inf, 1.0, 1.0, 2.0 + 1e-10])
    lo, hi = inward_response_bounds(lower, upper)
    assert np.all(lo >= lower) and np.all(hi <= upper)
    assert lo[2] == hi[2] == 1.0
    assert lo[3] < hi[3]
