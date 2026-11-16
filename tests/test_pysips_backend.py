"""Real Bingo backend compatibility; never part of the ordinary fast suite."""

import numpy as np
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest


@pytest.mark.native
def test_pysips_extended_search_uses_python_backend_and_restores_caller(tmp_path):
    from bingo.symbolic_regression.agraph import agraph
    from bingocpp import evaluation_backend, simplification_backend
    from ml_platform_symbolic.engines.pysips import discover

    original = agraph.evaluation_backend, agraph.simplification_backend
    agraph.evaluation_backend, agraph.simplification_backend = evaluation_backend, simplification_backend
    x = np.linspace(0.2, 2.4, 40)[:, None]
    request = DiscoveryRequest(
        x,
        0.31 + 1.27 * np.exp(-0.83 * x[:, 0]) + np.random.default_rng(712).normal(0, 0.01, len(x)),
        np.arange(len(x)),
        [{"id": "x0", "column": "x", "unit": "1", "domain": [0.1, 3.0]}],
        {"column": "y", "unit": "1", "domain": None},
        ["add", "sub", "mul", "div", "exp", "log", "abs", "sqrt", "pow_int"],
        {"max_nodes": 40, "max_wall_seconds": 30, "probes": [], "require_units": False},
        0,
        {"particles": 24, "mcmc_samples": 2, "max_complexity": 8, "seconds": 5},
    )
    try:
        result = discover(request, DiscoveryContext(tmp_path))
        assert result.candidates and all(candidate.native_check["passed"] for candidate in result.candidates)
        assert result.summary["smc_observation"]["likelihood_calls"] > 0
        assert (agraph.evaluation_backend, agraph.simplification_backend) == (
            evaluation_backend,
            simplification_backend,
        )
        request.options["particles"] = 0
        with pytest.raises(ValueError, match="positive integers"):
            discover(request, DiscoveryContext(tmp_path))
        assert (agraph.evaluation_backend, agraph.simplification_backend) == (
            evaluation_backend,
            simplification_backend,
        )
    finally:
        agraph.evaluation_backend, agraph.simplification_backend = original


@pytest.mark.native
def test_bingo_ieee_fidelity_is_separate_from_real_domain_validity():
    from bingo.symbolic_regression.agraph import agraph
    from bingo.symbolic_regression.agraph import operator_definitions as op
    from ml_platform_symbolic.engines.base import checked_candidate
    from ml_platform_symbolic.engines.pysips import export_graph
    from ml_platform_symbolic.expressions import evaluate, evaluate_native

    original = agraph.evaluation_backend, agraph.simplification_backend
    agraph.force_use_of_python_backends()
    try:
        graph = agraph.AGraph(use_simplification=False)
        graph.command_array = np.array(
            [[op.INTEGER, -1, -1], [op.VARIABLE, 0, 0], [op.DIVISION, 0, 1], [op.EXPONENTIAL, 2, 2]], dtype=int
        )
        variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [-1.0, 1.0]}]
        equation = export_graph(graph, variables)
        points = np.array([[0.0], [-0.0], [0.5], [-0.5]])
        native = graph.evaluate_equation_at(points).reshape(-1)
        np.testing.assert_array_equal(evaluate_native(equation, points, variables), native)
        assert native[0] == 0.0 and np.isposinf(native[1])
        real = evaluate(equation, points, variables)
        assert np.isnan(real[:2]).all() and np.isfinite(real[2:]).all()
        train = np.linspace(0.2, 1.0, 20)[:, None]
        request = DiscoveryRequest(
            train,
            graph.evaluate_equation_at(train).ravel(),
            np.arange(20),
            variables,
            {"unit": "1"},
            ["div", "exp"],
            {"max_nodes": 20, "probes": [{"x": 0.0}]},
            0,
            {},
        )
        candidate = checked_candidate("pysips", 0, equation, request, lambda X: graph.evaluate_equation_at(X).ravel())
        assert candidate.native_check["passed"]
        assert candidate.native_check["common_real_domain_valid_points"] < (
            candidate.native_check["train_points"] + candidate.native_check["boundary_and_configured_points"]
        )
    finally:
        agraph.evaluation_backend, agraph.simplification_backend = original


@pytest.mark.native
def test_bingo_fixed_negative_power_is_not_a_free_power_request():
    from bingo.symbolic_regression.agraph import agraph
    from bingo.symbolic_regression.agraph import operator_definitions as op
    from ml_platform_symbolic.constraints import check_constraints
    from ml_platform_symbolic.engines.pysips import export_graph
    from ml_platform_symbolic.expressions import evaluate, evaluate_native

    original = agraph.evaluation_backend, agraph.simplification_backend
    agraph.force_use_of_python_backends()
    try:
        graph = agraph.AGraph(use_simplification=True)
        graph.command_array = np.array([[op.VARIABLE, 0, 0], [op.INTEGER, -1, -1], [op.POWER, 0, 1]])
        variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [-2.0, 2.0]}]
        equation = export_graph(graph, variables)
        X = np.array([[-2.0], [-0.0], [0.0], [0.5]])
        native = graph.evaluate_equation_at(X).reshape(-1)
        np.testing.assert_array_equal(evaluate_native(equation, X, variables), native)
        assert np.isnan(evaluate(equation, X, variables)[1:3]).all()
        checks = check_constraints(
            equation,
            variables,
            {"unit": "1"},
            ["div", "pow_int"],
            {
                "max_nodes": 20,
                "max_free_parameters": None,
                "require_units": True,
                "allowed_variables": None,
                "probes": [],
                "response_bounds": [None, None],
                "response_checks": [],
            },
        )
        assert all(c["status"] == "pass" for c in checks if c["required"])
    finally:
        agraph.evaluation_backend, agraph.simplification_backend = original
