"""Pinned native grammar/fit/export check; independent probe, no rounded formula."""

import numpy as np
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.proged import _export, discover
from ml_platform_symbolic.expressions import evaluate


@pytest.mark.native
@pytest.mark.parametrize("optimizer", ["local", "DE"])
def test_proged_native_subset_order_and_full_coefficients(tmp_path, optimizer, monkeypatch):
    pytest.importorskip("ProGED")
    import pandas as pd
    from ProGED import EqDisco
    from ProGED.generators.grammar import GeneratorGrammar

    X = np.column_stack([np.linspace(2, 4, 30), np.linspace(0.2, 2, 30)])
    y = 1.234567891 * X[:, 1] + 0.370000019
    grammar = "S -> 'C' '*' 'x1' '+' 'C' [1.0]"
    native = EqDisco(
        data=pd.DataFrame(X, columns=["x0", "x1"]).assign(target=y),
        lhs_vars=["target"],
        rhs_vars=["x0", "x1"],
        generator=GeneratorGrammar(grammar),
        sample_size=1,
        max_attempts=2,
    )
    native.generate_models()
    native.fit_models()
    model = next(iter(native.models.values()))
    variables = [{"id": f"x{i}", "column": f"col{i}", "unit": "1", "domain": [0.1, 5]} for i in range(2)]
    equation = _export(model, variables)
    probe = np.array([[4.1, 0.37], [3.7, 1.43], [2.9, 2.73]])
    columns = [["x0", "x1"].index(str(v)) for v in model.rhs_vars]
    np.testing.assert_allclose(
        evaluate(equation, probe, variables),
        np.asarray(model.evaluate(probe[:, columns], model.params)).ravel(),
        rtol=1e-12,
        atol=1e-12,
    )
    request = DiscoveryRequest(
        X,
        y,
        np.arange(len(X)),
        variables,
        {"unit": "1"},
        ["add", "mul"],
        {"max_nodes": 20},
        42,
        {
            "grammar": grammar,
            "sample_size": 1,
            "max_attempts": 2,
            "optimizer": optimizer,
            "optimizer_iterations": 10,
            **({"optimizer_population": 12} if optimizer == "DE" else {}),
        },
    )
    if optimizer == "DE":
        import ProGED.parameter_estimation as estimation
        original_de, original_termination = estimation.DE, estimation.BestTermination
        observed = {}
        def de(**options):
            observed["population"] = options["pop_size"]
            return original_de(**options)
        def termination(**options):
            observed["iterations"] = options["n_max_gen"]
            return original_termination(**options)
        monkeypatch.setattr(estimation, "DE", de)
        monkeypatch.setattr(estimation, "BestTermination", termination)
    import sympy as sp
    original_lambdify = sp.lambdify
    snapshots = []
    result = discover(request, DiscoveryContext(tmp_path, checkpoint=lambda value: snapshots.append(value)))
    assert sp.lambdify is original_lambdify
    assert len(snapshots) == 1
    assert snapshots[0].summary["native_execution_status"] == "running"
    assert snapshots[0].summary["progress"]["fit_completed_count"] == 1
    assert result.summary["native_execution_status"] == "completed"
    assert result.summary["progress"]["fit_started_count"] == result.summary["progress"]["fit_completed_count"] == 1
    assert result.status == "ok"
    assert result.candidates[0].native_check["boundary_and_configured_points"] > 0
    assert result.native_artifacts["native_models"].is_file()
    if optimizer == "DE":
        assert observed == {"population": 12, "iterations": 10}
        values = [p["value"] for p in result.candidates[0].equation["parameters"].values()]
        margin = min(min(v + 10, 10 - v) / 20 for v in values)
        assert result.summary["parameter_boundary_usage"]["minimum_normalized_margin"] == pytest.approx(margin)
        assert result.summary["parameter_boundary_usage"]["outside_bounds"] == 0
        assert all(p["bounds"] == [-10.0, 10.0] for p in result.candidates[0].equation["parameters"].values())
    else:
        assert result.summary["effective_optimizer_options"] == {"maxiter": 10}
        assert result.summary["optimizer_work"]
        assert all(row["nit"] <= 10 for row in result.summary["optimizer_work"])


@pytest.mark.native
def test_completed_rejected_native_candidate_is_counted_not_exported(tmp_path):
    from ml_platform_symbolic.data import read_json
    X = np.linspace(0.2, 2, 24)[:, None]
    request = DiscoveryRequest(X, X[:, 0]**2, np.arange(24),
        [{"id": "x0", "column": "x", "unit": "1"}], {"unit": "1"}, ["add", "mul", "pow_int"],
        {"max_nodes": 30}, 17, {"grammar": "S -> 'C' '*' 'x0' '**' '2' '+' 'C' [1.0]",
        "sample_size": 1, "max_attempts": 2, "max_constants": 1})
    import sympy as sp
    original_lambdify = sp.lambdify
    snapshots = []
    result = discover(request, DiscoveryContext(tmp_path, checkpoint=lambda value: snapshots.append(value)))
    assert not result.candidates and result.status == "failed"
    assert len(snapshots) == 1 and snapshots[0].candidates == []
    assert result.summary["progress"]["fit_completed_count"] == 1
    assert result.summary["native_execution_status"] == "completed"
    assert len(result.summary["rejected_exports"]) == 1
    raw = read_json(result.native_artifacts["native_models"])
    assert raw["models"][0]["valid"] is False
    assert "**2" in raw["models"][0]["expression"].replace(" ", "")


@pytest.mark.native
@pytest.mark.parametrize("external_change", [False, True])
def test_proged_lambdify_scope_restored_on_error(tmp_path, monkeypatch, external_change):
    from ProGED import EqDisco
    import sympy as sp
    original = sp.lambdify
    external = lambda *args, **kwargs: None
    def fail(self, **kwargs):
        assert sp.lambdify.func is original
        assert sp.lambdify.keywords == {"docstring_limit": 0}
        if external_change: sp.lambdify = external
        raise RuntimeError("injected native failure")
    monkeypatch.setattr(EqDisco, "fit_models", fail)
    request = DiscoveryRequest(np.array([[1.],[2.]]), np.array([1.,2.]), np.arange(2),
        [{"id":"x0","column":"x","unit":"1"}], {"unit":"1"}, ["add","mul"], {}, 17,
        {"grammar":"S -> 'C' '*' 'x0' [1.0]", "sample_size":1, "max_attempts":2})
    try:
        with pytest.raises(RuntimeError,match="injected"):
            discover(request, DiscoveryContext(tmp_path))
        assert sp.lambdify is (external if external_change else original)
    finally:
        sp.lambdify = original
