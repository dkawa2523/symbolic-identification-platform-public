"""Full-problem proposal context, exact fragment binding and fixed budget semantics."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines import pretrained_template as adapter
from ml_platform_symbolic.engines.base import Candidate, DiscoveryContext, DiscoveryResult
from ml_platform_symbolic.engines.pretrained_proposals import collect_parts, fingerprint, full_problem_request
from ml_platform_symbolic.expressions import evaluate
from test_template_enumeration import request_for


def equation(ast, parameters=None):
    return {"schema_version": 2, "ast": ast, "parameters": parameters or {}, "arithmetic": "numpy_ieee"}


def parameter(value):
    return {"value": value, "trainable": True, "unit": "1", "bounds": [None, None]}


def test_full_problem_and_shared_calls_are_not_slot_target_regressions(tmp_path, monkeypatch):
    calls = []

    def discover(request, context):
        calls.append(deepcopy(request))
        body = equation(
            {
                "op": "add",
                "args": [
                    {"op": "mul", "args": [{"param": "a"}, {"var": "x0"}]},
                    {"op": "mul", "args": [{"param": "b"}, {"var": "x1"}]},
                ],
            },
            {"a": parameter(0.5), "b": parameter(1.7)},
        )
        return DiscoveryResult([Candidate("native/1", "e2e", 1, body, {"passed": True})], "ok", {})

    native = SimpleNamespace(validate_request=lambda *args: None, discover=discover)
    monkeypatch.setattr(adapter, "get_engine", lambda engine: native)
    X = np.random.default_rng(123).uniform(-1, 1, (30, 2))
    request = request_for(
        X, 2.3 * (X[:, 0] + X[:, 1]), shared=True, options={"proposal_mode": "subexpressions", "max_proposals": 4}
    )
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0].X, X)
    np.testing.assert_array_equal(calls[0].y, request.y)
    assert calls[0].equation_template is None
    assert result.candidates, result.summary
    assert min(np.max(np.abs(evaluate(c.equation, X, request.variables) - request.y)) for c in result.candidates) < 1e-6
    sources = result.summary["native_runs"]["h"]["proposal_transformations"]
    assert {v["call_index"] for v in sources.values()} == {0, 1}
    assert all(v["native_candidate_id"] == "native/1" and "ast_path" in v for v in sources.values())
    assert all(
        v["ast_path_basis"] == "normalized proposal AST after recorded coordinate transformation"
        for v in sources.values()
    )
    assert len(sources) <= 4


def test_derived_features_are_exact_forward_maps_with_original_X_kept():
    X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    request = request_for(X, X[:, 0], options={})
    options = {"proposal_options": {}}
    calls = [[{"op": "log", "args": [{"var": "x0"}]}], [{"var": "x1"}]]
    proposed, maps = full_problem_request(request, "h", calls, options, 0)
    np.testing.assert_array_equal(proposed.X[:, :2], X)
    np.testing.assert_allclose(proposed.X[:, 2], np.log(X[:, 0]))
    assert maps[0]["variable_mapping"] == {"__proposal_2": "u"}
    assert maps[1]["variable_mapping"] == {"x1": "u"}
    np.testing.assert_array_equal(proposed.y, request.y)
    request.X[0, 0] = -1
    with pytest.raises(ValueError, match="Nonfinite derived"):
        full_problem_request(request, "h", calls, options, 0)


def test_identical_argument_alias_is_explicit_and_not_an_inverse():
    request = request_for(np.ones((3, 2)), np.ones(3), options={})
    request.equation_template["slots"]["h"]["arguments"].append({"name": "v", "unit": "1"})
    proposed, maps = full_problem_request(request, "h", [[{"var": "x1"}, {"var": "x1"}]], {"proposal_options": {}}, 0)
    assert proposed.X.shape == (3, 2)
    assert maps[0]["variable_mapping"] == {"x1": "u"}
    assert maps[0]["aliased_unused_formals"] == {"v": "u"}


def test_full_context_feature_limit_fails_before_native_or_fallback(tmp_path):
    X = np.ones((4, 10))
    request = request_for(X, np.ones(4), options={"proposal_mode": "subexpressions"})
    request.equation_template["ast"]["args"] = [{"op": "add", "args": [{"var": "x0"}, {"var": "x1"}]}]
    with pytest.raises(NotImplementedError, match="10"):
        adapter.discover(request, DiscoveryContext(tmp_path / "not-created"), engine="e2e")
    assert not (tmp_path / "not-created").exists()


def test_structural_fingerprint_retains_parameter_sharing_and_fixed_values():
    shared = equation({"op": "add", "args": [{"param": "a"}, {"param": "a"}]}, {"a": parameter(1.0)})
    independent = equation(
        {"op": "add", "args": [{"param": "a"}, {"param": "b"}]}, {"a": parameter(1.0), "b": parameter(1.0)}
    )
    assert fingerprint(shared, initial_values=False) != fingerprint(independent, initial_values=False)
    other = deepcopy(shared)
    other["parameters"]["a"]["value"] = 2.0
    assert fingerprint(shared, initial_values=False) == fingerprint(other, initial_values=False)
    assert fingerprint(shared, initial_values=True) != fingerprint(other, initial_values=True)
    for value in (shared, other):
        value["parameters"]["a"]["trainable"] = False
    assert fingerprint(shared, initial_values=False) != fingerprint(other, initial_values=False)


def test_parent_call_round_robin_does_not_exhaust_first_parent():
    first = equation(
        {
            "op": "add",
            "args": [
                {"op": "mul", "args": [{"param": "a"}, {"var": "x0"}]},
                {"op": "mul", "args": [{"param": "b"}, {"var": "x0"}]},
            ],
        },
        {"a": parameter(1.0), "b": parameter(2.0)},
    )
    second = equation({"op": "cos", "args": [{"var": "x0"}]})
    candidates = [Candidate(str(i), "e2e", 1, body, {"passed": True}) for i, body in enumerate([first, second])]
    slot = {"arguments": [{"name": "u", "unit": "1"}], "allowed_operators": ["add", "mul", "cos"], "max_nodes": 7}
    maps = [
        {"call_index": i, "variable_mapping": {"x0": "u"}, "binding": [{"var": "x0"}], "aliased_unused_formals": {}}
        for i in range(2)
    ]
    items, audit = collect_parts(
        candidates, slot, maps, lambda candidate, transform: deepcopy(candidate.equation), adapter._check_slot_body
    )
    assert [item[2]["native_candidate_id"] for item in items[:2]] == ["0", "1"]
    assert audit["exact_duplicates"] > 0
    assert any(item[1]["ast"] == {"var": "u"} for item in items)
    assert any("param" in item[1]["ast"] for item in items)


def test_subexpression_does_not_revive_invalid_real_domain(tmp_path, monkeypatch):
    body = equation({"op": "exp", "args": [{"op": "log", "args": [{"var": "x0"}]}]})
    native = SimpleNamespace(
        validate_request=lambda *a: None,
        discover=lambda *a: DiscoveryResult([Candidate("native/0", "e2e", 1, body, {"passed": True})], "ok", {}),
    )
    monkeypatch.setattr(adapter, "get_engine", lambda engine: native)
    X = np.array([[0.0], [1.0], [2.0]])
    request = request_for(X, X[:, 0], options={"proposal_mode": "subexpressions", "max_proposals": 8})
    request.operators = ["exp", "log"]
    request.equation_template["slots"]["h"].update(allowed_operators=request.operators, max_nodes=3)
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert result.candidates
    assert all(np.isfinite(evaluate(c.equation, X, request.variables)).all() for c in result.candidates)
    assert any(a["status"] == "failed" for a in result.summary["attempts"])
    with np.load(result.native_artifacts["template_predictions"], allow_pickle=False) as arrays:
        assert arrays["proposal_h_0_raw"][0] == 0
        assert np.isnan(arrays["proposal_h_0_real"][0])


def test_extraction_alias_and_variable_derived_parameters_fail_explicitly():
    slot = {"arguments": [{"name": "u", "unit": "1"}]}
    maps = [{"call_index": 0, "variable_mapping": {"x0": "u"}}]
    for extra, expected in [({"tied_to": "b"}, "tied_to"), ({"derived_from": {"var": "x0"}}, "parameter-only")]:
        body = equation({"param": "a"}, {"a": {**parameter(1), **extra}, "b": parameter(1)})
        candidate = Candidate("native/0", "e2e", 1, body, {})
        parts, audit = collect_parts([candidate], slot, maps, lambda c, t: c.equation, lambda p, s: p)
        assert parts == []
        assert expected in audit["sources"][0]["reason"]


def test_extraction_deadline_never_silently_returns_truncated_pool():
    candidate = Candidate("native/0", "e2e", 1, equation({"var": "x0"}), {})
    with pytest.raises(TimeoutError, match="no partial pool"):
        collect_parts([candidate], {}, [], lambda c, t: c.equation, lambda p, s: p, deadline=0)


def test_native_proposal_scope_does_not_mislabel_derived_columns_as_final_allowed_variables():
    X = np.array([[1.0], [2.0]])
    request = request_for(X, X[:, 0], options={})
    request.limits["allowed_variables"] = ["x0"]
    proposed, _ = full_problem_request(
        request, "h", [[{"op": "log", "args": [{"var": "x0"}]}]], {"proposal_options": {}}, 0
    )
    assert proposed.limits["allowed_variables"] is None
    assert request.limits["allowed_variables"] == ["x0"]
    assert proposed.X.shape[1] == 2
