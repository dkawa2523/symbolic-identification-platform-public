"""PhySO export and declared coefficient dimensions, without native search dependencies."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.constraints import dimensions
from ml_platform_symbolic.engines.physo import _export, coefficient_unit_catalog
from ml_platform_symbolic.expressions import parse_expression
from ml_platform_symbolic.fitting import fit_coefficients
from ml_platform_symbolic.units import unit_dimensions


@pytest.mark.parametrize("target_unit", ["s", "day"])
def test_physo_export_keeps_configured_coefficient_units(target_unit):
    program = SimpleNamespace(
        tokens=[
            SimpleNamespace(var_type=0, name="add", arity=2),
            SimpleNamespace(var_type=20, name="c0", var_id=0),
            SimpleNamespace(var_type=20, name="c1", var_id=1),
        ],
        free_consts=SimpleNamespace(class_values=np.array([[1.25, 2.5]])),
    )
    equation = _export(program, [], {"c0": "1", "c1": target_unit})
    assert equation["parameters"]["c0"]["unit"] == "1"
    assert equation["parameters"]["c1"]["unit"] == target_unit
    assert [p["value"] for p in equation["parameters"].values()] == [1.25, 2.5]


def test_physo_known_acceleration_stays_named_fixed_and_dimensioned():
    g = {"value": 9.80665, "unit": "m/s2"}
    variables = [{"id": "x0", "column": "time", "unit": "s"}]
    program = SimpleNamespace(
        tokens=[
            SimpleNamespace(var_type=0, name="mul", arity=2),
            SimpleNamespace(var_type=20, name="c0", var_id=0),
            SimpleNamespace(var_type=0, name="mul", arity=2),
            SimpleNamespace(var_type=3, name="9.8066", fixed_const=g["value"]),
            SimpleNamespace(var_type=0, name="n2", arity=1),
            SimpleNamespace(var_type=1, name="x0", var_id=0),
        ],
        free_consts=SimpleNamespace(class_values=np.array([[0.9]])),
    )
    equation = _export(program, variables, {"c0": "1"}, {"9.8066": ("g", g)})
    assert dimensions(equation, variables, {"unit": "m"})[0] == "pass"
    X = np.linspace(0.1, 2, 30).reshape(-1, 1)
    fitted = fit_coefficients(equation, X, 0.5 * g["value"] * X[:, 0] ** 2, variables)
    assert fitted["parameters"]["g"] == {
        "value": g["value"],
        "unit": "m/s2",
        "trainable": False,
        "bounds": [g["value"], g["value"]],
        "origin": "declared_fixed_constant",
    }
    assert fitted["parameters"]["c0"]["value"] == pytest.approx(0.5)


def test_generic_coefficient_catalog_can_express_time_dependent_height_without_truth():
    request = SimpleNamespace(
        variables=[{"id": "x0", "column": "time", "unit": "s"}], target={"unit": "m"}, options={"unit_basis_degree": 2}
    )
    catalog, scope = coefficient_unit_catalog(request)
    assert (1, 0, -1, 0) in {unit_dimensions(unit) for unit in catalog.values()}
    assert (1, 0, -2, 0) in {unit_dimensions(unit) for unit in catalog.values()}
    assert list(catalog.values()).count("1") == 2  # Independent same-unit coefficients remain distinct.
    request.y = np.array([999.0, -234.0])
    assert coefficient_unit_catalog(request) == (catalog, scope)
    request.options = {"parameter_units": {"intercept": "m", "speed": "m/s", "accel": "m/s^2"}}
    explicit, _ = coefficient_unit_catalog(request)
    equation = parse_expression(
        "intercept+speed*x0+accel*x0**2",
        request.variables,
        parameters={k: {"value": 1.0, "trainable": True, "unit": v} for k, v in explicit.items()},
    )
    assert dimensions(equation, request.variables, request.target)[0] == "pass"
