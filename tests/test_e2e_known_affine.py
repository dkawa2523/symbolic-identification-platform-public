"""Ordinary E2E template dispatch, without importing native torch in unit checks."""

import json
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.engines import e2e
from ml_platform_symbolic.engines.base import Candidate, DiscoveryContext, DiscoveryRequest, DiscoveryResult
from ml_platform_symbolic.engines.e2e_template import prepare_known_affine
from ml_platform_symbolic.expressions import evaluate, inspect_equation
from ml_platform_symbolic.model import load_model, save_model


def request():
    X = np.linspace(-1, 1, 35)[:, None]
    call = {"slot": "h", "args": [{"var": "x"}]}
    template = {
        "coordinate_space": "working",
        "ast": {"op": "add", "args": [{"var": "x"}, {"op": "mul", "args": [{"const": -2}, call]}]},
        "components": {},
        "parameters": {},
        "slots": {
            "h": {
                "arguments": [{"name": "u", "unit": "1"}],
                "unit": "1",
                "allowed_operators": ["mul", "add", "sub", "div"],
                "max_nodes": 20,
            }
        },
        "knowledge": {"kind": "hypothesis", "statement": "Known affine outer"},
    }
    return DiscoveryRequest(
        X,
        X[:, 0] - 2 * 2.3 * X[:, 0],
        np.arange(len(X)),
        [{"id": "x", "column": "x", "unit": "1", "domain": [-1, 1]}],
        {"column": "y", "unit": "1"},
        ["add", "sub", "mul", "div", "exp", "log"],
        {"max_nodes": 30, "response_mode": "search", "probes": [{"x": 1.0}], "response_bounds": [-3.7, -3.5]},
        7,
        {},
        equation_template=template,
    )


def native_stub(monkeypatch, *, empty=False):
    calls, parents = [], []

    def native(proposed, context):
        calls.append((proposed, context))
        raw = e2e.export_candidate("mul,0.4,x_0", proposed.variables, [0.0], [2.0])
        parent = Candidate("e2e/7/0", "e2e", 7, raw, {"passed": True, "refinement_type": "NoRef"})
        parents.append(deepcopy(parent))
        return DiscoveryResult(
            [] if empty else [parent],
            "failed" if empty else "ok",
            {"complexity_budget": {"max_nodes": proposed.limits["max_nodes"]}},
        )

    monkeypatch.setattr(e2e, "_discover", native)
    monkeypatch.setattr(e2e, "torch_thread_scope", lambda options: nullcontext())
    return calls, parents


def test_ordinary_e2e_dispatches_identified_observations_then_original_response_fit(tmp_path, monkeypatch):
    calls, parents = native_stub(monkeypatch)
    r = request()
    saved = deepcopy(r)
    result = e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert result.status == "ok"
    assert len(calls) == 1
    reduced, native_context = calls[0]
    np.testing.assert_allclose(reduced.y, 2.3 * r.X[:, 0], atol=1e-15)
    assert reduced.X is r.X and reduced.row_ids is r.row_ids
    assert reduced.domain is r.domain and reduced.variables is r.variables
    assert reduced.equation_template is None and reduced.fixed_constants == {}
    assert reduced.limits["response_mode"] == "filter" and reduced.limits["probes"] == []
    assert native_context.checkpoint is None
    candidate = result.candidates[0]
    assert candidate.engine == "e2e" and candidate.candidate_id.endswith("/known-affine-fit")
    assert candidate.native_check["independent_native_oracle"] is False
    assert candidate.native_check["native_slot_parity"]["passed"] is True
    np.testing.assert_allclose(evaluate(candidate.equation, r.X, r.variables), r.y, atol=2e-6)
    archive = json.loads((tmp_path / "known_affine_native/candidates.json").read_text())
    assert archive[0]["equation"] == parents[0].equation
    assert archive[0]["equation"]["parameters"]["c1"]["unit"] is None
    assert result.summary["decoder_conditioned_on_outer_ast"] is False
    assert result.summary["decoder_conditioned_on_response"] is False
    assert r.equation_template == saved.equation_template
    np.testing.assert_array_equal(r.y, saved.y)
    schema = {"variables": r.variables, "target": r.target, "feature_ranges": [[-1, 1]]}
    path = save_model(tmp_path / "saved.zip", candidate.equation, schema, {"native_check": candidate.native_check})
    np.testing.assert_array_equal(
        load_model(path).predict(pd.DataFrame(r.X, columns=["x"])), evaluate(candidate.equation, r.X, r.variables)
    )


def test_actual_exported_normalization_and_source_budget_are_preserved(tmp_path, monkeypatch):
    native_stub(monkeypatch)
    r = request()
    r.limits["max_nodes"] = 11
    reduced, plan = prepare_known_affine(r, DiscoveryContext(tmp_path))
    assert reduced.limits["max_nodes"] == 7
    result = e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    eq = result.candidates[0].equation
    assert inspect_equation(eq, r.variables)["n_nodes"] == 11
    normalization = [p for p in eq["parameters"].values() if p.get("origin") == "train_normalization"]
    assert sorted(p["value"] for p in normalization) == [0.0, 2.0]
    assert all(p["trainable"] is False for p in normalization)
    r.limits["max_nodes"] = 10
    result = e2e.discover(r, DiscoveryContext(tmp_path / "small", assets_dir=tmp_path))
    assert not result.candidates
    assert "complexity" in result.summary["attempts"][0]["reason"]


def test_identical_shared_calls_use_one_fitted_vector_and_mse_identity(tmp_path, monkeypatch):
    native_stub(monkeypatch)
    r = request()
    call = {"slot": "h", "args": [{"var": "x"}]}
    r.equation_template["ast"] = {"op": "add", "args": [deepcopy(call), deepcopy(call)]}
    r.y = 4.6 * r.X[:, 0]
    r.limits.update(max_nodes=15, response_bounds=[4.5, 4.7])
    proposed, plan = prepare_known_affine(r, DiscoveryContext(tmp_path))
    assert plan["calls"] == 2 and proposed.limits["max_nodes"] == 7
    for values in [np.zeros(len(r.y)), np.ones(len(r.y)), np.sin(r.X[:, 0])]:
        assert np.mean((2 * values - r.y) ** 2) == pytest.approx(4 * np.mean((values - proposed.y) ** 2))
    result = e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert len([p for p in result.candidates[0].equation["parameters"].values() if p["trainable"]]) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "variable_slope",
        "nonlinear",
        "different_binding",
        "unknown_coefficient",
        "zero_slope",
        "invalid_offset",
        "multiple_slots",
        "nonunit",
        "missing_normalization_ops",
    ],
)
def test_unsupported_rejected_before_native(tmp_path, monkeypatch, mode):
    calls, _ = native_stub(monkeypatch)
    r = request()
    ast = r.equation_template["ast"]
    if mode == "variable_slope":
        ast["args"][1]["args"][0] = {"var": "x"}
    if mode == "nonlinear":
        r.equation_template["ast"] = {"op": "exp", "args": [ast]}
    if mode == "different_binding":
        ast["args"][1]["args"][1]["args"] = [{"op": "mul", "args": [{"const": 2}, {"var": "x"}]}]
    if mode == "unknown_coefficient":
        ast["args"][1]["args"][0] = {"param": "a"}
        r.equation_template["parameters"] = {"a": {"value": 2.0, "trainable": True, "unit": "1"}}
    if mode == "zero_slope":
        ast["args"][1]["args"][0] = {"const": 0}
    if mode == "invalid_offset":
        ast["args"][0] = {"op": "log", "args": [{"var": "x"}]}
    if mode == "multiple_slots":
        r.equation_template["slots"]["g"] = deepcopy(r.equation_template["slots"]["h"])
    if mode == "nonunit":
        r.variables[0]["unit"] = "m"
    if mode == "missing_normalization_ops":
        r.equation_template["slots"]["h"]["allowed_operators"] = ["mul"]
    with pytest.raises((ValueError, NotImplementedError)):
        e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert calls == []


def test_float64_observation_reconstruction_is_separate_from_constant_problem(tmp_path):
    r = request()
    r.y = np.ones(len(r.y))
    r.equation_template["ast"]["args"][0] = {"const": 1e30}
    with pytest.raises(ValueError, match="resolution"):
        e2e.validate_request(r, DiscoveryContext(tmp_path))
    r = request()
    r.equation_template["ast"] = {"slot": "h", "args": [{"var": "x"}]}
    r.limits.update(probes=[], response_bounds=[None, None])
    r.y = np.zeros(len(r.y))
    e2e.validate_request(r, DiscoveryContext(tmp_path))
    r.y = np.ones(len(r.y))
    e2e.validate_request(r, DiscoveryContext(tmp_path))


def test_nonidentity_source_mapping_and_domain_rejected_early(tmp_path, monkeypatch):
    calls, _ = native_stub(monkeypatch)
    r = request()
    provenance = {
        "unit_coordinates": {"columns": {"x": {"problem_unit": "1", "source_unit": "1", "scale": 2.0, "offset": 0.0}}}
    }
    with pytest.raises(NotImplementedError, match="Nonidentity source"):
        e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path, provenance=provenance))
    r.variables[0]["domain"] = [-0.5, 0.5]
    with pytest.raises(ValueError):
        e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert calls == []


def test_no_native_pool_no_fallback_and_completed_fit_checkpoint(tmp_path, monkeypatch):
    native_stub(monkeypatch, empty=True)
    r = request()
    result = e2e.discover(r, DiscoveryContext(tmp_path / "empty", assets_dir=tmp_path))
    assert result.status == "failed" and not result.candidates
    assert result.summary["termination"] == "no_native_slot_candidates"
    native_stub(monkeypatch)
    checkpoints = []
    result = e2e.discover(r, DiscoveryContext(tmp_path / "done", assets_dir=tmp_path, checkpoint=checkpoints.append))
    assert checkpoints and checkpoints[0].candidates[0].candidate_id.endswith("/known-affine-fit")
    calls, _ = native_stub(monkeypatch)
    result = e2e.discover(r, DiscoveryContext(tmp_path / "expired", assets_dir=tmp_path, deadline=0.0))
    assert result.status == "partial" and result.summary["unattempted_candidates"] == 1


def test_template_capability_is_stage_qualified_and_explicit_hybrid_unchanged():
    from ml_platform_symbolic.engines import e2e_template

    assert e2e.capabilities()["equation_template"] is True
    assert "no template-conditioned decoder" in e2e.capabilities()["constraints"]["equation_template"]
    assert "not decoder conditioning" in e2e_template.capabilities()["constraints"]["equation_template"]


def test_original_outer_operation_order_roundtrip_prevents_false_equivalence(tmp_path):
    r = request()
    call = {"slot": "h", "args": [{"var": "x"}]}
    r.equation_template["ast"] = {
        "op": "sub",
        "args": [{"op": "add", "args": [call, {"const": 1e16}]}, {"const": 1e16}],
    }
    r.y = np.ones(len(r.y))
    with pytest.raises(ValueError, match="resolution"):
        e2e.validate_request(r, DiscoveryContext(tmp_path))


def test_forbidden_variables_in_known_outer_are_not_deferred_to_post_filter(tmp_path, monkeypatch):
    calls, _ = native_stub(monkeypatch)
    r = request()
    r.limits["allowed_variables"] = []
    with pytest.raises(NotImplementedError, match="Known outer AST references forbidden"):
        e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert calls == []


def test_explicit_original_hard_enforcement_is_not_leaked_into_internal_filter(tmp_path, monkeypatch):
    calls, _ = native_stub(monkeypatch)
    r = request()
    r.limits["response_enforcement"] = "hard"
    result = e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert result.candidates
    assert calls[0][0].limits["response_enforcement"] is None
    assert r.limits["response_enforcement"] == "hard"
    assert result.candidates[0].native_check["finite_point_feasible"] is True
    assert result.summary["attempts"][0]["solver"]["solver"] == "SLSQP"
