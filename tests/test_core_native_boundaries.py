"""Scientific boundary counterexamples, not field/registry mirror tests."""

from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.constraints import check_constraints, equation_unit
from ml_platform_symbolic.domains import expression_bounds
from ml_platform_symbolic.engines.base import _check_ieee_compositions
from ml_platform_symbolic.expression_graph import compact_equation
from ml_platform_symbolic.expressions import (
    evaluate,
    evaluate_native,
    inspect_equation,
    parse_expression,
)
from ml_platform_symbolic.parameters import free_parameter_names, with_parameters
from ml_platform_symbolic.units import coordinate_conversion, unit_spec


def test_saved_coordinate_map_owns_default_input_and_output_without_duplicate_units():
    import pandas as pd
    from ml_platform_symbolic.data import model_matrix, model_output

    schema = {
        "variables": [{"id": "x0", "column": "position", "unit": "m"}],
        "target": {"column": "response", "unit": "m"},
        "unit_coordinates": {
            "columns": {
                "position": {"source_unit": "cm", "problem_unit": "m", "scale": 0.01, "offset": 0.0},
                "response": {"source_unit": "mm", "problem_unit": "m", "scale": 0.001, "offset": 0.0},
            }
        },
    }
    np.testing.assert_array_equal(model_matrix(pd.DataFrame({"position": [100.0, 200.0]}), schema), [[1.0], [2.0]])
    np.testing.assert_array_equal(
        model_matrix(pd.DataFrame({"position": [1.0, 2.0]}), schema, {"position": "m"}), [[1.0], [2.0]]
    )
    np.testing.assert_array_equal(model_output(np.array([1.0, 2.0]), schema), [1000.0, 2000.0])
    inconsistent = deepcopy(schema)
    inconsistent["public_units"] = {"position": "m"}
    with pytest.raises(ValueError, match="conflicts"):
        model_matrix(pd.DataFrame({"position": [1.0]}), inconsistent)


def test_deep_structural_comparison_key_does_not_recursively_quote_strings():
    import json

    from ml_platform_symbolic.equation_comparison import canonical

    left = right = {"var": "x0"}
    for index in range(70):
        constant = {"const": float(index)}
        left = {"op": "add", "args": [left, constant]}
        right = {"op": "add", "args": [constant, right]}
    a = canonical({"schema_version": 1, "ast": left, "parameters": {}})
    b = canonical({"schema_version": 1, "ast": right, "parameters": {}})
    assert a == b
    assert len(json.dumps(a)) < 5000


def test_dictionary_retains_valid_fits_when_later_export_cannot_resolve_signal(tmp_path):
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.dictionary import discover

    X = np.linspace(1, 8, 50)[:, None]
    y = 0.001 + 0.005 / (1 + (X[:, 0] - 4) ** 2)
    request = DiscoveryRequest(
        X,
        y,
        np.arange(len(X)),
        [{"id": "x0", "column": "x", "unit": "1", "domain": [1, 100]}],
        {"unit": "1"},
        ["add", "mul", "pow_int", "exp"],
        {"max_nodes": 100, "probes": []},
        0,
        {"max_terms": 12},
    )
    result = discover(request, DiscoveryContext(tmp_path))
    assert result.status == "ok" and result.candidates
    assert result.summary["rejected_exports"]
    assert all(c.native_check["passed"] for c in result.candidates)
    assert any("precision cannot resolve" in row["reason"] for row in result.summary["rejected_exports"])
    assert result.summary["native_candidates"] == len(result.candidates) + len(result.summary["rejected_exports"])


V = [{"id": "x0", "column": "x", "unit": "1", "domain": [0.2, 3.0]}]


def test_shared_execution_retains_parameters_holes_and_expanded_complexity():
    p = {
        "a": {"value": 2.0, "trainable": True, "unit": "1"},
        "b": {"value": 2.0, "trainable": False, "tied_to": "a", "unit": "1"},
    }
    eq = parse_expression("(a*x0+b)/(a*x0+b)+sin(a*x0+b)", V, parameters=p, trainable=False)
    graph = compact_equation(eq)
    assert graph["definitions"]
    before = deepcopy(graph)
    X = np.array([[0.2], [1.0], [2.0]])
    stats = inspect_equation(graph, V)
    assert stats["n_nodes"] == inspect_equation(eq, V)["n_nodes"]
    assert stats["n_stored_nodes"] < stats["n_nodes"]
    np.testing.assert_array_equal(evaluate(graph, X, V), evaluate(eq, X, V))
    fitted = with_parameters(graph, ["a"], [3.0])
    assert fitted["parameters"]["b"]["value"] == 3.0
    assert graph == before and free_parameter_names(graph) == ["a"]
    constraints = {"max_nodes": stats["n_nodes"] - 1, "require_units": False, "probes": []}
    checks = check_constraints(graph, V, {"unit": "1"}, ["add", "div", "mul", "sin"], constraints)
    assert next(c for c in checks if c["name"] == "max_nodes")["status"] == "fail"
    pole = compact_equation(parse_expression("(x0-1)/(x0-1)", V, trainable=False))
    assert np.isnan(evaluate(pole, np.array([[1.0]]), V)[0])
    broken = deepcopy(graph)
    broken["definitions"]["s0"] = {"ref": "s0"}
    with pytest.raises(ValueError, match="cyclic"):
        inspect_equation(broken, V)


def test_explicit_ieee_parity_does_not_turn_invalid_real_rows_into_scientific_predictions():
    variables = [{**V[0], "domain": [-1.0, 1.0]}]
    eq = parse_expression("tanh(1/x0)", variables, trainable=False)
    eq.update(schema_version=2, arithmetic="numpy_ieee")
    X = np.array([[0.0], [-0.0], [0.5]])
    raw = evaluate_native(eq, X, variables)
    np.testing.assert_array_equal(raw, [1.0, -1.0, np.tanh(2.0)])
    assert np.isnan(evaluate(eq, X, variables)[:2]).all()
    strict = deepcopy(eq)
    strict["arithmetic"] = "strict_real"
    assert np.isnan(evaluate_native(strict, X, variables)[:2]).all()


def test_bounded_integer_power_math_is_dtype_independent_and_resolves_square_denominator():
    for literal in ["2**-1", "2.0**-1"]:
        eq = parse_expression(literal, V, trainable=False)
        assert expression_bounds(eq["ast"], eq["parameters"], V) == (0.5, 0.5)
    eq = parse_expression("atan(1/(1+(x0-.37)**2))", V, trainable=False)
    _check_ieee_compositions(eq, V)
    assert np.isfinite(evaluate(eq, np.array([[0.37], [3.0]]), V)).all()


def test_numeric_angular_sum_is_order_independent_without_inventing_quantity_kind():
    v = [{"id": "x0", "column": "a", "unit": "rad/s"}, {"id": "x1", "column": "b", "unit": "1/s"}]
    a, err_a = equation_unit(parse_expression("x0+x1", v, trainable=False), v)
    b, err_b = equation_unit(parse_expression("x1+x0", v, trainable=False), v)
    assert a == b and not err_a and not err_b and not a.kind_known
    with pytest.raises(ValueError, match="meaning"):
        coordinate_conversion(a, unit_spec("1/s"))
    hz = [v[0], {**v[1], "unit": "Hz"}]
    assert equation_unit(parse_expression("x0+x1", hz, trainable=False), hz)[1]

    points = [
        {"id": "x0", "column": "origin", "unit": "K", "coordinate_role": "value"},
        {"id": "x1", "column": "increment", "unit": "delta_K", "coordinate_role": "difference"},
    ]
    a, errors = equation_unit(parse_expression("x0+x1", points, trainable=False), points)
    b, other_errors = equation_unit(parse_expression("x1+x0", points, trainable=False), points)
    assert a == b and a.role == "value" and not errors and not other_errors


def test_atomic_completed_progress_is_bound_to_job_and_preserves_native_values(tmp_path):
    from ml_platform_symbolic.data import read_json, write_json
    from ml_platform_symbolic.engines.base import Candidate, DiscoveryResult
    from ml_platform_symbolic.engines.progress import load_progress, save_progress

    job = write_json(tmp_path / "job.json", {"seed": 4})
    eq = parse_expression("x0+1", V, trainable=False)
    native = write_json(tmp_path / "native.json", {"prefix": ["add", "x0", "1"], "constant": 1.0})
    result = DiscoveryResult(
        [Candidate("test/4/0", "test", 4, eq, {"passed": True})],
        "partial",
        {"native_work": {"optimizer_steps": 3}},
        {"native": str(native)},
    )
    save_progress(job, result, {})
    native.write_text("incomplete later write", encoding="utf-8")
    recovered, _, _ = load_progress(job, V)
    assert read_json(recovered.native_artifacts["native"])["constant"] == 1.0
    assert recovered.summary["native_work"]["optimizer_steps"] == 3
    assert recovered.candidates[0].equation == eq
    job.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_progress(job, V)


def test_saved_replay_equality_does_not_override_signal_adequacy_or_seed_denominators():
    from ml_platform_symbolic.numerical_fidelity import compare_replay
    from ml_platform_symbolic.result_statistics import summarize_seed_metrics

    unresolved = np.array([1.0e9, 1.0e9 + np.spacing(1.0e9)])
    check = compare_replay(unresolved, unresolved, scale=np.spacing(1.0e9))
    assert check["serialization_check_passed"] and check["resolution_limited"] and not check["passed"]
    lost = compare_replay(np.array([1.0e-20, 2.0e-20]), np.zeros(2), scale=0.5e-20)
    assert not lost["serialization_check_passed"]
    report = summarize_seed_metrics(
        [
            {"seed": 0, "scored": True, "n_rows": 140, "n_valid": 140, "rmse": 0.01},
            {"seed": 1, "scored": True, "n_rows": 140, "n_valid": 133, "rmse": None},
            {"seed": 2, "scored": True, "n_rows": 140, "n_valid": 140, "rmse": 0.11},
        ],
        expected_seeds=[0, 1, 2],
        unit="m",
    )
    assert report["valid_seeds"] == [0, 2] and report["invalid_seeds"] == [1]
    assert report["median"] == pytest.approx(0.06) and report["n_valid"] == 2


def test_sparse_polynomial_keeps_zero_terms_and_shared_basis_without_faking_complexity():
    from ml_platform_symbolic.expression_graph import expanded_equation, polynomial_node

    features = [{"var": "x0"}]
    terms = [
        {"coefficient": {"const": 0.0}, "powers": [[0, 1]]},
        {"coefficient": {"const": 2.0}, "powers": [[0, 2]]},
        {"coefficient": {"const": 1.0}, "powers": []},
    ]
    eq = {"schema_version": 2, "arithmetic": "numpy_ieee", "parameters": {}, "ast": polynomial_node(features, terms)}
    stats = inspect_equation(eq, V)
    assert stats["n_stored_nodes"] < stats["n_nodes"]
    tree = expanded_equation(eq)
    X = np.array([[0.2], [1.0], [3.0]])
    np.testing.assert_array_equal(evaluate_native(eq, X, V), evaluate_native(tree, X, V))
    np.testing.assert_array_equal(evaluate(eq, X, V), evaluate(tree, X, V))
    assert len(eq["ast"]["poly"]) == 3 and eq["ast"]["poly"][0]["coefficient"]["const"] == 0
    strict = deepcopy(eq)
    strict["schema_version"] = 1
    strict.pop("arithmetic")
    with pytest.raises(ValueError, match="schema2"):
        inspect_equation(strict, V)
