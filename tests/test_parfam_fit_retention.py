"""Adapter failure routing with an actual native fit before a terminal exception."""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.native


def _request():
    from ml_platform_symbolic.engines.base import DiscoveryRequest

    x = np.linspace(0.2, 1.5, 32).reshape(-1, 1)
    variable = {"var": "x"}
    def call(arg):
        return {"slot": "h", "args": [arg]}
    template = {"coordinate_space": "working", "parameters": {}, "components": {},
        "knowledge": {"kind": "hypothesis", "statement": "Shared-slot failure-retention mechanism"},
        "ast": {"op": "add", "args": [call(variable),
            call({"op": "mul", "args": [{"const": 2}, variable]})]},
        "slots": {"h": {"arguments": [{"name": "u", "unit": None}], "unit": None,
                         "max_nodes": 20, "allowed_operators": ["add", "mul"]}}}
    return DiscoveryRequest(x, 6*x[:, 0]+6, np.arange(len(x)),
        [{"id": "x", "column": "x", "unit": None}], {"column": "y", "unit": None},
        ["add", "mul", "pow_int"], {"max_wall_seconds": 60, "max_nodes": 60,
            "require_units": False, "probes": [], "response_checks": [],
            "response_bounds": [None, None], "response_mode": "search"},
        17, {"family": "polynomial", "degree": 1, "maxiter": 5, "seconds": 30,
             "candidate_scope": "winner"}, equation_template=template)


@pytest.mark.parametrize("complete_first", [False, True])
def test_terminal_native_fit_failure_keeps_only_completed_finite_families(tmp_path, monkeypatch, complete_first):
    import pandas as pd
    import parfam.train as native_train
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover
    from ml_platform_symbolic.expressions import evaluate
    from ml_platform_symbolic.model import load_model, save_model
    from parfam import ParFamWrapper

    original = ParFamWrapper.fit
    error = native_train.NoFiniteFamilyError

    def controlled_terminal_failure(self, *args, **kwargs):
        if complete_first:
            original(self, *args, **kwargs)
        raise error("Controlled terminal failure for retention acceptance")

    monkeypatch.setattr(ParFamWrapper, "fit", controlled_terminal_failure)
    request = _request()
    result = discover(request, DiscoveryContext(tmp_path / "native"))
    assert result.status == ("partial" if complete_first else "failed")
    assert not result.summary["native_fit_completed"]
    assert "Controlled terminal failure" in result.summary["reason"]
    assert bool(result.candidates) is complete_first
    if not complete_first:
        assert result.summary["native_completed_family_fits"] == 0
        return
    assert result.summary["returned_scope"] == "completed_families_before_failure"
    assert set(result.summary["variants"].values()) == {"completed_family_0"}
    for candidate in result.candidates:
        assert candidate.native_check["passed"]
        equation = json.loads(json.dumps(candidate.equation))
        expected = evaluate(equation, request.X, request.variables)
        assert np.isfinite(expected).all()
        assert np.sqrt(np.mean((expected-request.y)**2)) < 1e-3
        bundle = save_model(tmp_path / "retained.zip", equation,
            {"variables": request.variables, "target": request.target, "feature_ranges": [[.2, 1.5]]}, {})
        np.testing.assert_array_equal(load_model(bundle).predict(pd.DataFrame({"x": request.X[:, 0]})), expected)


def test_real_composite_family_boundary_fails_without_fabricating_candidate(tmp_path, monkeypatch):
    import parfam.train as native_train
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover

    request = _request()
    # A single native linear family, all-zero explicit test starts, no basin hops.
    request.equation_template["ast"] = {"op": "pow_int", "args": [
        {"op": "sqrt", "args": [{"slot": "h", "args": [{"var": "x"}]}]}, {"const": 2}]}
    request.operators.append("sqrt")
    monkeypatch.setattr(native_train.np.random, "randn", lambda *shape: np.zeros(shape))
    result = discover(request, DiscoveryContext(tmp_path / "boundary"))
    assert result.status == "failed" and not result.candidates
    assert result.summary["native_execution_status"] == "failed"
    assert result.summary["native_completed_family_fits"] == 0
    assert len(result.summary["failed_family_fits"]) == 1
    assert result.summary["nonfinite_gradient_fits"] >= 1
    assert "no finite completed family" in result.summary["reason"]
