"""Opt-in iterate admission: no native/checkpoint or benchmark data used."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import fitting
from ml_platform_symbolic.engines import e2e
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.e2e_domain import prepare_admission
from ml_platform_symbolic.engines.e2e_response import fit_responses, prepare_response_fit
from ml_platform_symbolic.engines.e2e_template import fit_known_affine_candidate, prepare_known_affine
from ml_platform_symbolic.expressions import evaluate
from ml_platform_symbolic.model import load_model, save_model


def request():
    X = np.linspace(2.1, 4, 32)[:, None]
    return DiscoveryRequest(
        X,
        1 / (X[:, 0] - 0.5),
        np.arange(len(X)),
        [{"id": "x0", "column": "x", "unit": "1", "domain": [-5, 5]}],
        {"column": "y", "unit": "1"},
        ["add", "sub", "mul", "div"],
        {"max_nodes": 100, "response_mode": "search", "probes": [{"x": 4.0}], "response_bounds": [-100.0, 100.0]},
        17,
        {
            "response_max_nfev": 300,
            "rational_domain_admission": {
                "column": "x",
                "interval": [-1.0, 1.0],
                "max_calls": 1000,
                "max_total_operations": 1000000,
            },
        },
    )


def equation(r, text="div,1,sub,x_0,2.0"):
    return e2e.export_candidate(text, r.variables, [0.0], [1.0])


def gate(r, tmp_path):
    return prepare_admission(r, DiscoveryContext(tmp_path), prepare_response_fit(r))


def solve(r, eq, admission=None, diagnostics=None):
    # The finite-point constraint is intentionally loose: no point evidence is
    # mislabelled as global absence of divisor roots.
    return fitting.fit_coefficients(
        eq,
        r.X,
        r.y,
        r.variables,
        max_nfev=300,
        constraints=(lambda v: evaluate(v, np.array([[4.0]]), r.variables), np.array([-100.0]), np.array([100.0])),
        diagnostics=diagnostics,
        iterate_acceptance=admission,
    )


def test_real_slsqp_returns_safe_visited_fit_instead_of_lower_loss_pole(tmp_path):
    r = request()
    eq = equation(r)
    original = deepcopy(eq)
    plain = solve(r, eq)
    g = gate(r, tmp_path)
    diag = {}
    fitted = solve(r, eq, g, diag)
    from ml_platform_symbolic.domains import rational_domain_check

    variables = [{**r.variables[0], "domain": [-1.0, 1.0]}]
    assert rational_domain_check(plain, variables)["status"] == "unsafe"
    assert rational_domain_check(fitted, variables)["status"] == "safe"
    assert np.mean((evaluate(fitted, r.X, r.variables) - r.y) ** 2) > np.mean(
        (evaluate(plain, r.X, r.variables) - r.y) ** 2
    )
    assert g.summary()["statuses"]["unsafe"] > 0
    assert g.certificate()["proof"]["stage"] == "final"
    assert g.summary()["solver_objective_conditioned"] is False
    assert eq == original


def test_no_safe_visited_iterate_rejected_not_postfit_fallback(tmp_path):
    r = request()
    g = gate(r, tmp_path)
    # All permitted coefficients place a root inside the proof interval.
    eq = equation(r, "div,1,sub,x_0,0.5")
    next(p for p in eq["parameters"].values() if p.get("trainable"))["bounds"] = [0.4, 0.6]
    with pytest.raises(ValueError, match="No feasible coefficient iterate"):
        solve(r, eq, g)
    assert g.summary()["statuses"]["unsafe"] > 0
    assert g.summary()["statuses"]["safe"] == 0


def test_initially_unsafe_structure_is_allowed_to_reach_safe_coefficients(tmp_path):
    r = request()
    r.y = 1 / (r.X[:, 0] - 1.8)
    eq = equation(r, "div,1,sub,x_0,0.5")
    g = gate(r, tmp_path)
    fitted = solve(r, eq, g)
    assert g.summary()["statuses"]["unsafe"] > 0
    assert g.certificate()["proof"]["status"] == "safe"
    assert float(next(p["value"] for p in fitted["parameters"].values() if p.get("trainable"))) > 1


@pytest.mark.parametrize("text,status", [("exp,x_0", "out_of_scope"), ("div,1,sub,x_0,0", "unsafe")])
def test_fixed_equations_are_not_a_bypass(tmp_path, text, status):
    r = request()
    g = gate(r, tmp_path)
    eq = equation(r, text)
    with pytest.raises(ValueError, match="Fixed equation failed"):
        solve(r, eq, g)
    assert g.last_result()["status"] == status


def test_fixed_safe_checked_and_saved_metadata_roundtrip(tmp_path):
    r = request()
    g = gate(r, tmp_path)
    eq = equation(r, "div,1,sub,x_0,3")
    fitted = solve(r, eq, g)
    cert = g.certificate()
    assert cert["proof"]["stage"] == "fixed"
    path = save_model(
        tmp_path / "model.zip",
        fitted,
        {"variables": r.variables, "target": r.target, "feature_ranges": [[2.1, 4.0]]},
        {"native_check": {"rational_domain_admission": cert}, "options": r.options},
    )
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict(pd.DataFrame({"x": r.X[:, 0]})), evaluate(fitted, r.X, r.variables))
    assert loaded.metadata["native_check"]["rational_domain_admission"] == cert
    assert loaded.metadata["options"] == r.options


@pytest.mark.parametrize("key,value", [("max_calls", 1), ("max_total_operations", 2)])
def test_global_budgets_are_shared_and_do_not_admit_unknown(tmp_path, key, value):
    r = request()
    r.options["rational_domain_admission"][key] = value
    g = gate(r, tmp_path)
    eq = equation(r)
    with pytest.raises(ValueError, match="admission|feasible"):
        solve(r, eq, g)
    assert g.last_result()["status"] == "unknown"
    assert g.summary()["proof_calls"] <= r.options["rational_domain_admission"]["max_calls"]
    assert g.summary()["arithmetic_operation_charges"] <= r.options["rational_domain_admission"]["max_total_operations"]
    assert not g(eq, "iterate")
    assert g.last_result()["status"] == "unknown"


def test_deadline_denial_is_unknown_not_proved_unsafe(tmp_path):
    r = request()
    compiled = prepare_response_fit(r)
    g = prepare_admission(r, DiscoveryContext(tmp_path, deadline=0.0), compiled)
    assert not g(equation(r), "iterate")
    assert g.last_result()["reason"] == "discovery_deadline_before_proof"
    assert g.summary()["proof_calls"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.options.__setitem__("rational_domain_admission", False),
        lambda r: r.options["rational_domain_admission"].__setitem__("interval", [None, 1]),
        lambda r: r.options["rational_domain_admission"].__setitem__("interval", [1, -1]),
        lambda r: r.options["rational_domain_admission"].__setitem__("interval", [0, float("inf")]),
        lambda r: r.options["rational_domain_admission"].__setitem__("max_calls", True),
        lambda r: r.options["rational_domain_admission"].__setitem__("column", "other"),
        lambda r: r.options["rational_domain_admission"].__setitem__("extra", 1),
    ],
)
def test_invalid_policy_fails_preflight(tmp_path, mutation):
    r = request()
    mutation(r)
    with pytest.raises((ValueError, NotImplementedError)):
        e2e.validate_request(r, DiscoveryContext(tmp_path))


def test_gate_only_filter_and_multidimensional_are_explicitly_unsupported(tmp_path):
    r = request()
    r.limits["response_mode"] = "filter"
    with pytest.raises(NotImplementedError, match="domain-only and filter"):
        e2e.validate_request(r, DiscoveryContext(tmp_path))
    r = request()
    r.limits["probes"] = []
    r.limits["response_bounds"] = [None, None]
    with pytest.raises(NotImplementedError, match="requires finite-point"):
        e2e.validate_request(r, DiscoveryContext(tmp_path))
    r = request()
    r.variables.append({"id": "x1", "column": "z", "unit": "1", "domain": [-5, 5]})
    r.X = np.column_stack([r.X, r.X])
    r.limits["probes"][0]["z"] = 4.0
    with pytest.raises(NotImplementedError, match="exactly one"):
        e2e.validate_request(r, DiscoveryContext(tmp_path))


def test_explicit_public_source_interval_not_working_interval(tmp_path):
    from ml_platform_symbolic.domains import rational_domain_check

    r = request()
    r.variables[0]["unit"] = "m"
    provenance = {
        "unit_coordinates": {"columns": {"x": {"problem_unit": "m", "source_unit": "cm", "scale": 0.01, "offset": 0.0}}}
    }
    g = prepare_admission(r, DiscoveryContext(tmp_path, provenance=provenance), prepare_response_fit(r))
    eq = equation(r, "div,1,sub,x_0,0.5")
    # Source interval [-1,1] cm is NOT working [-1,1] m. A pole at .5m
    # lies outside the directly declared public interval, without inverting endpoints.
    assert rational_domain_check(eq, [{**r.variables[0], "domain": [-1.0, 1.0]}])["status"] == "unsafe"
    assert g(eq, "final")
    assert g.certificate()["public_source_unit"] == "cm"
    assert g.summary()["policy"]["interval"] == [-1.0, 1.0]


def test_acceptance_never_changes_smooth_objective_values_or_evaluation_sequence(monkeypatch, tmp_path):
    r = request()
    eq = equation(r)
    visits = []

    def fake(fun, start, **kwargs):
        values = [fun(start), fun(start * 0.8), fun(start * 0.2)]
        visits.append(values)
        return SimpleNamespace(x=start * 0.2, success=True, status=0, message="controlled")

    monkeypatch.setattr(fitting, "minimize", fake)
    solve(r, eq)
    g = gate(r, tmp_path)
    solve(r, eq, g)
    assert visits[0] == visits[1]
    assert g.summary()["statuses"]["unsafe"] > 0


def test_absent_none_gate_leaves_old_result_diagnostics_and_no_new_metadata(tmp_path):
    r = request()
    eq = equation(r)
    a = {}
    b = {}
    first = solve(r, eq, diagnostics=a)
    second = solve(r, eq, None, b)
    assert first == second and a == b
    r.options.pop("rational_domain_admission")
    assert gate(r, tmp_path) is None
    r.options["rational_domain_admission"] = None
    assert gate(r, tmp_path) is None


def test_ordinary_response_fit_adds_final_evidence_and_preserves_originals(tmp_path):
    from ml_platform_symbolic.engines.base import Candidate

    r = request()
    eq = equation(r)
    parent = Candidate("native", "e2e", 17, eq, {"passed": True, "refinement_type": "NoRef"})
    original = deepcopy(parent)
    fitted, meta = fit_responses([parent], r, DiscoveryContext(tmp_path), prepare_response_fit(r))
    assert len(fitted) == 1 and parent == original
    assert fitted[0].native_check["rational_domain_admission"]["proof"]["status"] == "safe"
    assert meta["rational_domain_admission"]["statuses"]["unsafe"] > 0
    assert meta["attempts"][0]["rational_domain_admission"]["stage"] == "final"


def test_known_outer_checks_full_ast_and_does_not_send_policy_to_slot_decoder(tmp_path):
    r = request()
    r.variables[0]["domain"] = [-1, 1]
    r.X = np.linspace(-0.8, 0.8, 20)[:, None]
    r.y = r.X[:, 0]
    r.limits["probes"] = [{"x": 0.0}]
    r.equation_template = {
        "coordinate_space": "working",
        "ast": {
            "op": "div",
            "args": [{"slot": "h", "args": [{"var": "x0"}]}, {"op": "sub", "args": [{"var": "x0"}, {"const": 0.5}]}],
        },
        "components": {},
        "parameters": {},
        "slots": {
            "h": {
                "arguments": [{"name": "x0", "unit": "1"}],
                "unit": "1",
                "allowed_operators": ["add", "sub", "mul", "div"],
                "max_nodes": 30,
            }
        },
        "knowledge": {"kind": "hypothesis", "statement": "test"},
    }
    # Variable-dependent slope is already unsupported; never bypass outer guards.
    with pytest.raises(NotImplementedError, match="Variable-dependent"):
        prepare_known_affine(r, DiscoveryContext(tmp_path))
    r.equation_template["ast"] = {"op": "add", "args": [{"var": "x0"}, {"slot": "h", "args": [{"var": "x0"}]}]}
    proposed, plan = prepare_known_affine(r, DiscoveryContext(tmp_path))
    assert "rational_domain_admission" not in proposed.options
    native = equation(r, "div,1,sub,x_0,0.5")
    next(p for p in native["parameters"].values() if p.get("trainable"))["bounds"] = [0.4, 0.6]
    with pytest.raises(ValueError, match="No feasible"):
        fit_known_affine_candidate(native, r, plan)
    assert plan["domain_admission"].summary()["statuses"]["unsafe"] > 0


def test_unknown_is_not_fixed_success_and_hook_cannot_silently_switch_solver(tmp_path):
    r = request()
    g = gate(r, tmp_path)
    eq = equation(r, "div,1,pow2,pow2,pow2,pow2,pow2,pow2,x_0")
    with pytest.raises(ValueError, match="Fixed equation failed"):
        solve(r, eq, g)
    assert g.last_result()["status"] == "unknown"
    assert g.last_result()["reason"] == "polynomial_degree_budget"
    with pytest.raises(NotImplementedError, match="explicit SLSQP"):
        fitting.fit_coefficients(equation(r), r.X, r.y, r.variables, iterate_acceptance=g)


def test_full_known_outer_ast_evidence_and_saved_native_lineage(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from ml_platform_symbolic.engines.base import Candidate, DiscoveryResult

    r = request()
    r.variables[0]["domain"] = [-1, 1]
    r.X = np.linspace(-0.8, 0.8, 20)[:, None]
    r.y = 2 * r.X[:, 0]
    r.limits["probes"] = [{"x": 0.0}]
    r.equation_template = {
        "coordinate_space": "working",
        "ast": {"op": "add", "args": [{"var": "x0"}, {"slot": "h", "args": [{"var": "x0"}]}]},
        "components": {},
        "parameters": {},
        "slots": {
            "h": {
                "arguments": [{"name": "x0", "unit": "1"}],
                "unit": "1",
                "allowed_operators": ["add", "sub", "mul", "div"],
                "max_nodes": 30,
            }
        },
        "knowledge": {"kind": "hypothesis", "statement": "test"},
    }
    original = deepcopy(r.options)
    seen = []

    def native(proposed, context):
        assert "rational_domain_admission" not in proposed.options
        seen.append(proposed)
        return DiscoveryResult(
            [Candidate("native", "e2e", 17, equation(r, "mul,0.3,x_0"), {"passed": True, "refinement_type": "NoRef"})],
            "ok",
            {"fixture": "not native execution"},
        )

    monkeypatch.setattr(e2e, "_discover", native)
    monkeypatch.setattr(e2e, "torch_thread_scope", lambda opts: nullcontext())
    result = e2e.discover(r, DiscoveryContext(tmp_path, assets_dir=tmp_path))
    assert len(result.candidates) == 1 and len(seen) == 1
    cert = result.candidates[0].native_check["rational_domain_admission"]
    assert cert["proof"]["status"] == "safe" and cert["proof"]["stage"] == "final"
    assert result.summary["rational_domain_admission"]["proof_calls"] > 1
    assert r.options == original
    assert result.candidates[0].native_check["native_slot_parity"]["passed"] is True


def test_last_candidate_budget_denial_marks_stage_incomplete(tmp_path):
    from ml_platform_symbolic.engines.base import Candidate

    r = request()
    r.options["rational_domain_admission"]["max_calls"] = 1
    parent = Candidate("native", "e2e", 17, equation(r), {"passed": True, "refinement_type": "NoRef"})
    fitted, meta = fit_responses([parent], r, DiscoveryContext(tmp_path), prepare_response_fit(r))
    assert not fitted and not meta["complete"]
    assert meta["rational_domain_admission"]["budget_denials"] > 0
    assert meta["attempts"][0]["rational_domain_admission"]["status"] == "unknown"


def test_last_fixed_candidate_shrunken_proof_budget_is_discovery_incomplete(tmp_path):
    from ml_platform_symbolic.engines.base import Candidate

    r = request()
    r.options["rational_domain_admission"]["max_total_operations"] = 2
    parent = Candidate("fixed", "e2e", 17, equation(r, "div,1,sub,x_0,3"), {"passed": True, "refinement_type": "NoRef"})
    fitted, meta = fit_responses([parent], r, DiscoveryContext(tmp_path), prepare_response_fit(r))
    assert not fitted and not meta["complete"]
    assert meta["rational_domain_admission"]["budget_denials"] == 1
    assert meta["rational_domain_admission"]["arithmetic_operation_charges"] == 2
    assert meta["attempts"][0]["rational_domain_admission"]["reason"] == "arithmetic_operation_budget"


def test_user_config_preserves_policy_and_rejects_automatic_post_refit(synthetic_config):
    from ml_platform_symbolic.config import resolve_config

    config = deepcopy(synthetic_config)
    config["search"]["engines"] = ["e2e"]
    policy = request().options["rational_domain_admission"]
    config["search"]["engine_options"] = {"e2e": {"rational_domain_admission": policy}}
    resolved = resolve_config(config)
    assert resolved["search"]["engine_options"]["e2e"]["rational_domain_admission"] == policy
    config["evaluation"]["coefficient_fit"] = "bounded_least_squares"
    with pytest.raises(NotImplementedError, match="would invalidate"):
        resolve_config(config)
    config["search"]["engine_options"]["e2e"]["rational_domain_admission"] = None
    assert resolve_config(config)["evaluation"]["coefficient_fit"] == "bounded_least_squares"
