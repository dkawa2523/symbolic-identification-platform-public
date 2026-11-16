"""Strong geometry counterexamples; not a second configuration/contract suite."""

from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.coordinates import (
    execution_region,
    freeze_input_geometry,
    map_input_covariance,
    map_input_values,
    source_input_values,
)
from ml_platform_symbolic.domains import region_mask, region_points, register_region


def problem(columns, bounds):
    variables = [
        {"id": f"x{i}", "column": c, "unit": "1", "domain": bounds[i], "coordinate_role": None}
        for i, c in enumerate(columns)
    ]
    cfg = {
        "data": {
            "units": dict.fromkeys(columns, "1"),
            "source_units": dict.fromkeys(columns, "1"),
            "coordinate_roles": {},
        },
        "domain": None,
        "input_map": None,
    }
    return cfg, variables


def test_original_simplex_membership_precedes_two_coordinate_execution():
    cfg, variables = problem(["a", "b", "c"], [[0, 1]] * 3)
    cfg["domain"] = {"kind": "affine", "columns": ["a", "b", "c"], "equalities": {"matrix": [[1, 1, 1]], "rhs": [1]}}
    cfg["input_map"] = {
        "kind": "affine_chart",
        "target_columns": ["u", "v"],
        "target_units": {"u": "1", "v": "1"},
        "forward": {"matrix": [[1, 0, 0], [0, 1, 0]], "offset": [0, 0]},
        "inverse": {"matrix": [[1, 0], [0, 1], [-1, -1]], "offset": [0, 0, 1]},
    }
    before = deepcopy(cfg)
    geometry, execution = freeze_input_geometry(cfg, variables)
    assert cfg == before
    X = np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
    Z = map_input_values(X, geometry)
    np.testing.assert_allclose(Z, X[:, :2])
    np.testing.assert_allclose(source_input_values(Z, geometry), X)
    region = execution_region(geometry, execution)
    assert region_mask(np.array([[0.2, 0.3], [0.8, 0.8]]), region).tolist() == [True, False]
    assert region_mask(region_points(Z, execution, 12, region), region).all()
    with pytest.raises(ValueError, match="original-coordinate"):
        map_input_values(np.array([[0.6, 0.6, 0.6]]), geometry)


def test_mixed_affine_coordinates_preserve_nonbox_domain_and_full_covariance():
    cfg, variables = problem(["x", "y"], [[0, 1]] * 2)
    cfg["input_map"] = {
        "kind": "affine_chart",
        "target_columns": ["u", "v"],
        "target_units": {"u": "1", "v": "1"},
        "forward": {"matrix": [[1, 1], [0, 1]], "offset": [0, 0]},
        "inverse": {"matrix": [[1, -1], [0, 1]], "offset": [0, 0]},
    }
    geometry, execution = freeze_input_geometry(cfg, variables)
    assert region_mask(np.array([[0.5, 0.2], [1.8, 0.2]]), execution_region(geometry, execution)).tolist() == [
        True,
        False,
    ]
    np.testing.assert_allclose(map_input_covariance(np.diag([1.0, 4.0]), geometry), [[5, 4], [4, 4]])
    cfg["input_map"]["target_columns"] = ["u"]
    cfg["input_map"]["target_units"] = {"u": "1"}
    cfg["input_map"]["forward"] = {"matrix": [[1, 1]], "offset": [0]}
    cfg["input_map"]["inverse"] = {"matrix": [[0.5], [0.5]], "offset": [0, 0]}
    with pytest.raises(ValueError, match="not a chart"):
        freeze_input_geometry(cfg, variables)


def test_region_unit_conversion_uses_values_but_constraint_residual_scale_is_retained():
    cfg, variables = problem(["length", "width"], [None, None])
    for v in variables:
        v["unit"] = "m"
    cfg["data"]["units"] = {"length": "m", "width": "m"}
    cfg["data"]["source_units"] = {"length": "cm", "width": "cm"}
    cfg["domain"] = {
        "kind": "affine",
        "columns": ["length", "width"],
        "equalities": {"matrix": [[1, 1]], "rhs": [100]},
        "bounds": [[0, 100], [0, 100]],
    }
    geometry, _ = freeze_input_geometry(cfg, variables)
    assert region_mask(np.array([[0.2, 0.8], [0.2, 0.3]]), geometry["domain"]).tolist() == [True, False]
    np.testing.assert_allclose(geometry["domain"]["equalities"]["matrix"], [[100, 100]])


def test_domain_extension_is_explicit_and_never_loaded_from_artifact():
    region = {"kind": "extension", "columns": ["x"], "name": "positive-test", "version": "1", "settings": {}}
    X = np.array([[-1.0], [1.0]])
    with pytest.raises(ValueError, match="install explicitly"):
        region_mask(X, region)
    register_region("positive-test", "1", contains=lambda values, _: values[:, 0] >= 0)
    assert region_mask(X, region).tolist() == [False, True]


def test_source_chart_actual_fit_and_saved_public_inference(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/geometry.py"
    spec = spec_from_file_location("geometry_workflow_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_example(tmp_path / "workflow")
    assert result["replay"]["passed"]
    assert "original-coordinate" in result["outside_original_domain_rejected"]


def test_composition_response_uses_declared_feasible_replacement_path():
    from ml_platform_symbolic.constraints import response_check
    from ml_platform_symbolic.coordinates import execution_constraints
    from ml_platform_symbolic.expressions import parse_expression

    cfg, variables = problem(["a", "b", "c"], [[0, 1]] * 3)
    cfg["domain"] = {"kind": "affine", "columns": ["a", "b", "c"], "equalities": {"matrix": [[1, 1, 1]], "rhs": [1]}}
    cfg["input_map"] = {
        "kind": "affine_chart",
        "target_columns": ["u", "v"],
        "target_units": {"u": "1", "v": "1"},
        "forward": {"matrix": [[1, 0, 0], [0, 1, 0]], "offset": [0, 0]},
        "inverse": {"matrix": [[1, 0], [0, 1], [-1, -1]], "offset": [0, 0, 1]},
    }
    geometry, execution = freeze_input_geometry(cfg, variables)
    check = {
        "direction": {"a": 1.0, "b": -1.0, "c": 0.0},
        "path_unit": "1",
        "step": 0.01,
        "points": [{"a": 0.2, "b": 0.3, "c": 0.5}],
        "order": 1,
        "sign": -1,
        "tolerance": 1e-10,
    }
    compiled = execution_constraints({"response_checks": [check]}, geometry, execution)["response_checks"][0]
    equation = parse_expression("3*x0+5*x1", execution, trainable=False)
    evidence = response_check(equation, execution, {"unit": "1"}, compiled)
    assert evidence["status"] == "pass" and evidence["worst_response"] == pytest.approx(-2.0)
    assert evidence["direction"] == check["direction"]
    with pytest.raises(ValueError, match="original-coordinate"):
        execution_constraints(
            {"response_checks": [{**check, "direction": {"a": 1.0, "b": 0.0, "c": 0.0}}]}, geometry, execution
        )
