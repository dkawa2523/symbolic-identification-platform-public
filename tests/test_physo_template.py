"""Scoped native wrapper checks, not benchmark recovery/quality claims."""

import json
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.physo_template import SharedSlot, strict_native_scope
from ml_platform_symbolic.equation_templates import validate_template
from ml_platform_symbolic.expressions import evaluate, inspect_equation


def request(tmp_path, ast=None, unit="1", target_unit="1"):
    x = np.linspace(0.2, 1.5, 24)[:, None]
    v = [{"id": "x", "column": "x", "unit": unit, "domain": [0.1, 2.0]}]
    call = {"slot": "h", "args": [{"var": "x"}]}
    raw = {
        "coordinate_space": "working",
        "ast": ast or {"op": "pow_int", "args": [call, {"const": 2}]},
        "parameters": {},
        "components": {},
        "knowledge": {"kind": "hypothesis", "statement": "Wrapper mechanism only"},
        "slots": {
            "h": {
                "arguments": [{"name": "x", "unit": unit}],
                "unit": unit,
                "max_nodes": 7,
                "allowed_operators": ["mul", "add"],
            }
        },
    }
    template = validate_template(raw, v, {"unit": target_unit}, ["mul", "add", "pow_int"], {})
    r = DiscoveryRequest(
        x,
        (2 * x[:, 0]) ** 2,
        np.arange(len(x)),
        v,
        {"column": "y", "unit": target_unit},
        ["mul", "add", "pow_int"],
        {"max_nodes": 30, "response_mode": "search"},
        17,
        {"parameter_units": {"a": "1"}, "epochs": 1, "batch_size": 48, "max_n_evaluations": 48},
        equation_template=template,
    )
    return r, DiscoveryContext(tmp_path)


def native_program(r, t, tokens):
    from ml_platform_symbolic.engines.physo import _export
    from physo.physym.library import Library
    from physo.physym.vect_programs import VectPrograms

    lib = Library(
        args_make_tokens={
            "op_names": ["mul", "add", "div", "exp", "neg", "sqrt", "n2"],
            "use_protected_ops": False,
            "input_var_ids": {"x": 0},
            "input_var_units": {"x": [0, 0, 0]},
            "class_free_constants": {"a"},
            "class_free_constants_units": {"a": [0, 0, 0]},
            "constants": {"one": 1.0},
            "constants_units": {"one": [0, 0, 0]},
        },
        superparent_units=[0, 0, 0],
    )
    p = VectPrograms(batch_size=1, max_time_step=len(tokens), library=lib, n_realizations=1, candidate_wrapper=t)
    p.set_programs(np.array([[lib.lib_name_to_idx[n] for n in tokens]]))
    return p, lambda prog: t.export(_export(prog, t.formal, {"a": "1"}))


@pytest.mark.parametrize("feature", ["multi", "free", "formal", "composed", "mapped", "cbrt"])
def test_unsupported_before_native(tmp_path, feature):
    r, c = request(tmp_path)
    t = r.equation_template
    if feature == "multi":
        t["slots"]["other"] = deepcopy(t["slots"]["h"])
    elif feature == "free":
        t["parameters"]["a"] = {"trainable": True}
    elif feature == "formal":
        t["slots"]["h"]["arguments"][0]["name"] = "u"
    elif feature == "composed":
        t["ast"]["args"][0]["args"] = [{"op": "neg", "args": [{"var": "x"}]}]
    elif feature == "mapped":
        c.provenance = {"input_geometry": {"map": {"x": {"var": "x"}}}}
    else:
        t["slots"]["h"]["allowed_operators"].append("rootn")
        r.operators.append("rootn")
    with pytest.raises(NotImplementedError):
        SharedSlot(r, c)


@pytest.mark.native
def test_shared_gradient_full_y_lbfgs_and_save_reload(tmp_path):
    import torch

    call = {"slot": "h", "args": [{"var": "x"}]}
    r, c = request(tmp_path, {"op": "add", "args": [call, deepcopy(call)]})
    r.y = 6 * r.X[:, 0]
    t = SharedSlot(r, c)
    p, export = native_program(r, t, ["mul", "a", "x"])
    prog = p.get_prog(0)
    X = torch.tensor(r.X.T)
    with strict_native_scope(t):
        prog.free_consts.class_values.requires_grad_(True)
        prog(X).sum().backward()
        torch.testing.assert_close(prog.free_consts.class_values.grad[0, 0], 2 * X.sum())
        prog.free_consts.class_values.requires_grad_(False)
        prog.optimize_constants(X, torch.tensor(r.y))
        assert abs(float(prog.free_consts.class_values[0, 0]) - 3) < 1e-4
        pred = prog(X).detach().numpy()
        eq = json.loads(json.dumps(export(prog)))
        assert len(eq["parameters"]) == 1
        np.testing.assert_allclose(evaluate(eq, r.X, r.variables), pred, rtol=0, atol=0)
        import pandas as pd
        from ml_platform_symbolic.model import load_model, save_model

        path = tmp_path / "shared.zip"
        save_model(path, eq, {"variables": r.variables, "target": r.target, "feature_ranges": [[0.1, 2.0]]}, {})
        np.testing.assert_array_equal(load_model(path).predict(pd.DataFrame({"x": r.X[:, 0]})), pred)
    assert t.audit["strict_execute_calls"] > 0


@pytest.mark.native
@pytest.mark.parametrize("tokens", [["exp", "neg", "div", "one", "x"], ["n2", "sqrt", "x"]])
def test_native_invalid_intermediate_not_finite(tmp_path, tokens):
    import torch

    r, c = request(tmp_path, {"slot": "h", "args": [{"var": "x"}]})
    t = SharedSlot(r, c)
    p, _ = native_program(r, t, tokens)
    X = torch.tensor([[0.0, -1.0, 1.0]] if tokens[0] == "exp" else [[-1.0, 0.0, 1.0]], dtype=torch.float64)
    with strict_native_scope(t):
        pred = p.get_prog(0)(X)
        assert torch.isnan(pred[0]) and torch.isfinite(pred[-1])


@pytest.mark.native
def test_nonfinite_gradient_fails_local_fit(tmp_path):
    import torch
    from physo.physym import free_const

    r, c = request(tmp_path)
    t = SharedSlot(r, c)
    a = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    with strict_native_scope(t):
        with pytest.raises(FloatingPointError, match="gradient"):
            free_const.OPTIMIZERS["LBFGS"](a, lambda p: (torch.sqrt(p) ** 2 - 1).square().sum(), n_steps=1)
    assert torch.isnan(a).all() and t.audit["nonfinite_fit_failures"] == 1


@pytest.mark.native
def test_source_budget_B_Bminus1(tmp_path):
    from ml_platform_symbolic.engines.physo_budget import source_budget_prior
    from physo.physym import prior
    from physo.physym.vect_programs import VectPrograms

    call = {"slot": "h", "args": [{"var": "x"}]}
    r, c = request(tmp_path, {"op": "add", "args": [call, deepcopy(call)]})
    t = SharedSlot(r, c)
    costs, overhead, m = t.source_budget()
    p, export = native_program(r, t, ["mul", "a", "x"])
    assert (costs, overhead, m) == ([2], 1, 2)
    B = inspect_equation(export(p.get_prog(0)), r.variables)["n_nodes"]
    assert B == 7
    for bound, allowed in [(B, 1), (B - 1, 0)]:
        empty = VectPrograms(batch_size=1, max_time_step=7, library=p.library, n_realizations=1)
        with source_budget_prior(
            bound, costs, overhead, {"mul", "add", "div", "exp", "neg", "sqrt", "n2"}, [], token_multiplier=m
        ):
            mask = prior.make_PriorCollection(p.library, empty, [])()
            assert mask[0, p.library.lib_name_to_idx["mul"]] == allowed


@pytest.mark.native
def test_native_search_slot_unit_differs_from_y_unit(tmp_path, monkeypatch):
    import torch

    torch.set_default_dtype(torch.float32)
    from ml_platform_symbolic.engines import physo as adapter
    from ml_platform_symbolic.engines.physo import discover
    from physo.physym import execute

    raw_execute, original_check = execute.ExecuteProgram, adapter.checked_candidate
    parity_calls = []

    def checked(*args, **kwargs):
        assert execute.ExecuteProgram is not raw_execute
        parity_calls.append(True)
        return original_check(*args, **kwargs)

    monkeypatch.setattr(adapter, "checked_candidate", checked)
    r, c = request(tmp_path, unit="m", target_unit="m^2")
    result = discover(r, c)
    assert result.summary["template_search"]["slot_output_unit"] == "m"
    assert result.summary["template_search"]["observed_y_unit"] == "m^2"
    assert result.summary["native_work"]["optimizer_steps"] > 0
    assert result.candidates
    assert len(parity_calls) >= 2  # completed-iteration and final exports, not only reward
    assert all(c.native_check["passed"] and "template" in c.equation for c in result.candidates)


@pytest.mark.native
def test_hard_response_full_template_writeback(tmp_path):
    import torch
    from ml_platform_symbolic.engines.physo_response import ResponseReward, prepare
    from physo.physym.reward import SquashedNRMSE

    call = {"slot": "h", "args": [{"var": "x"}]}
    r, c = request(tmp_path, {"op": "add", "args": [call, deepcopy(call)]})
    r.y = -2 * r.X[:, 0]
    r.limits.update(probes=[{"x": 1.0}], response_bounds=[2.0, None], response_enforcement="hard")
    r.options["response_max_nfev"] = 100
    t = SharedSlot(r, c)
    p, export = native_program(r, t, ["mul", "a", "x"])
    control = ResponseReward(r, prepare(r), export)
    with strict_native_scope(t):
        result = control.factory(SquashedNRMSE)(
            p, torch.tensor(r.X.T), torch.tensor(r.y), torch.ones(len(r.y)), np.array([len(r.y)]), {}
        )
    assert result[0] > 0 and float(p.free_consts.class_values[0, 0]) >= 1 - 1e-7
    assert control.counts["feasible"] == 1


@pytest.mark.native
def test_hooks_restored_on_error(tmp_path):
    from physo.physym import execute, free_const

    r, c = request(tmp_path)
    t = SharedSlot(r, c)
    exe, opt = execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"]
    with pytest.raises(RuntimeError):
        with strict_native_scope(t):
            raise RuntimeError("forced")
    assert execute.ExecuteProgram is exe and free_const.OPTIMIZERS["LBFGS"] is opt


@pytest.mark.native
def test_all_invalid_template_rewards_return_failed_not_fake_candidate(tmp_path):
    from ml_platform_symbolic.engines.physo import discover

    r, c = request(tmp_path)
    call = {"slot": "h", "args": [{"var": "x"}]}
    r.equation_template["ast"] = {"op": "div", "args": [{"const": 1.0}, {"op": "sub", "args": [call, deepcopy(call)]}]}
    r.operators += ["div", "sub"]
    result = discover(r, c)
    assert result.status == "failed" and not result.candidates
    assert result.summary["termination_reason"] == "no_positive_feasible_reward"

# PC pinned-runtime acceptance boundaries, beyond the original cloud mechanism checks.
@pytest.mark.native
@pytest.mark.parametrize("second_call", ["swap", "duplicate"])
def test_multiple_input_leaf_calls_share_fit_and_export(tmp_path, second_call):
    import torch
    from ml_platform_symbolic.engines.physo import _export
    from physo.physym.library import Library
    from physo.physym.vect_programs import VectPrograms

    r, c = request(tmp_path)
    r.X = np.column_stack([r.X[:, 0], np.linspace(1.1, 0.3, len(r.y))])
    r.variables.append({"id": "z", "column": "z", "unit": "1", "domain": [0.1, 2.0]})
    call = {"slot": "h", "args": [{"var": "x"}, {"var": "z"}]}
    other = {"slot": "h", "args": [{"var": "z" if second_call == "swap" else "x"}, {"var": "x"}]}
    raw = deepcopy(r.equation_template)
    raw["ast"] = {"op": "add", "args": [call, other]}
    raw["slots"]["h"]["arguments"].append({"name": "z", "unit": "1"})
    raw["slots"]["h"]["allowed_operators"].append("sub")
    r.operators.append("sub")
    r.equation_template = validate_template(raw, r.variables, r.target, r.operators, {})
    t = SharedSlot(r, c)
    lib = Library(args_make_tokens={
        "op_names": ["mul", "sub"], "use_protected_ops": False,
        "input_var_ids": {"x": 0, "z": 1}, "input_var_units": {"x": [0, 0, 0], "z": [0, 0, 0]},
        "class_free_constants": {"a"}, "class_free_constants_units": {"a": [0, 0, 0]},
    }, superparent_units=[0, 0, 0])
    p = VectPrograms(batch_size=1, max_time_step=5, library=lib, n_realizations=1, candidate_wrapper=t)
    p.set_programs(np.array([[lib.lib_name_to_idx[n] for n in ["sub", "mul", "a", "x", "z"]]]))
    prog, X = p.get_prog(0), torch.tensor(r.X.T)
    factor = X[0] + X[1] if second_call == "swap" else 2 * X[0]
    subtract = X[1] + X[0]
    r.y = (3 * factor - subtract).numpy()
    with strict_native_scope(t):
        prog.free_consts.class_values.requires_grad_(True)
        prog(X).sum().backward()
        torch.testing.assert_close(prog.free_consts.class_values.grad[0, 0], factor.sum())
        prog.free_consts.class_values.requires_grad_(False)
        prog.optimize_constants(X, torch.tensor(r.y))
        assert abs(float(prog.free_consts.class_values[0, 0]) - 3) < 1e-4
        eq = t.export(_export(prog, t.formal, {"a": "1"}))
        assert len(eq["parameters"]) == 1
        points = np.array([[0.25, 1.4], [1.6, 0.2], [0.75, 0.8]])
        np.testing.assert_allclose(
            evaluate(eq, points, r.variables),
            prog(torch.tensor(points.T)).detach().numpy(), rtol=0, atol=1e-12,
        )


@pytest.mark.native
def test_frozen_source_and_slot_budget_prior_intersection(tmp_path):
    from ml_platform_symbolic.coordinates import source_complexity
    from ml_platform_symbolic.engines.physo_budget import source_budget_prior
    from ml_platform_symbolic.engines.physo_template import local_budget_scope
    from physo.physym import prior
    from physo.physym.vect_programs import VectPrograms

    call = {"slot": "h", "args": [{"var": "x"}]}
    r, c = request(tmp_path, {"op": "add", "args": [call, deepcopy(call)]}, unit="m", target_unit="m")
    c.provenance = {"unit_coordinates": {"columns": {
        col: {"source_unit": "cm", "problem_unit": "m", "scale": 0.01, "offset": 0}
        for col in ["x", "y"]
    }}}
    t = SharedSlot(r, c)
    costs, overhead, multiplier = t.source_budget()
    p, export = native_program(r, t, ["mul", "a", "x"])
    B = source_complexity(export(p.get_prog(0)), t.coordinate_context)
    assert (costs, overhead, multiplier, B) == ([6], 3, 2, 13)
    operators = {"mul", "add", "div", "exp", "neg", "sqrt", "n2"}
    for bound, local, expected in [(B, 3, 1), (B - 1, 3, 0), (B, 2, 0)]:
        t.slot["max_nodes"] = local
        progs = VectPrograms(batch_size=1, max_time_step=7, library=p.library, n_realizations=1)
        if local >= 3:
            for name in ["mul", "a"]:
                progs.append(np.array([p.library.lib_name_to_idx[name]]))
            choice = "x"
        else:
            choice = "mul"
        with local_budget_scope(t, operators, []), source_budget_prior(
            bound, costs, overhead, operators, [], token_multiplier=multiplier
        ):
            mask = prior.make_PriorCollection(p.library, progs, [])()
            assert mask[0, p.library.lib_name_to_idx[choice]] == expected


@pytest.mark.native
def test_hard_derivative_uses_full_template_and_shared_writeback(tmp_path):
    import torch
    from ml_platform_symbolic.engines.physo_response import ResponseReward, prepare
    from ml_platform_symbolic.response_search import response_values
    from physo.physym.reward import SquashedNRMSE

    r, c = request(tmp_path)
    r.limits.update(response_enforcement="hard", response_checks=[{
        "feature": "x", "points": [{"x": 1.0}], "step": 0.01,
        "order": 1, "sign": 1, "tolerance": 0.0,
    }])
    r.options["response_max_nfev"] = 100
    t = SharedSlot(r, c)
    p, export = native_program(r, t, ["mul", "a", "x"])
    control = ResponseReward(r, prepare(r), export)
    with strict_native_scope(t):
        result = control.factory(SquashedNRMSE)(
            p, torch.tensor(r.X.T), torch.tensor(r.y), torch.ones(len(r.y)), np.array([len(r.y)]), {}
        )
        assert result[0] > 0
        eq = export(p.get_prog(0))
        points = np.asarray(control.compiled["points"])
        portable = evaluate(eq, points, r.variables)
        native = p.get_prog(0)(torch.tensor(points.T)).detach().numpy()
        np.testing.assert_allclose(native, portable, rtol=0, atol=1e-12)
    assert abs(abs(float(p.free_consts.class_values[0, 0])) - 2) < 1e-4
    # h(x)^2 has derivative 2*a^2*x, not the slot's derivative a.
    assert response_values(portable, control.compiled, normalized=False)[0] == pytest.approx(8, abs=1e-3)
    assert control.counts["feasible"] == 1


@pytest.mark.native
@pytest.mark.parametrize("failure_stage", ["native", "outer_prior"])
def test_discover_restores_all_scopes_on_midway_error(tmp_path, monkeypatch, failure_stage):
    from contextlib import contextmanager

    import physo
    from ml_platform_symbolic.engines import physo_budget
    from ml_platform_symbolic.engines.physo import discover
    from physo.physym import execute, free_const, prior, reward

    call = {"slot": "h", "args": [{"var": "x"}]}
    r, c = request(tmp_path, {"op": "add", "args": [call, deepcopy(call)]})
    r.limits.update(probes=[{"x": 1.0}], response_bounds=[0, None], response_enforcement="hard")
    saved = (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer)
    if failure_stage == "native":
        def fail(*args, **kwargs):
            assert all(a is not b for a, b in zip(
                (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer),
                saved, strict=True,
            ))
            raise RuntimeError("forced native failure")
        monkeypatch.setattr(physo, "SR", fail)
    else:
        original = physo_budget.source_budget_prior
        @contextmanager
        def fail(*args, **kwargs):
            if kwargs.get("token_multiplier", 1) > 1:
                assert prior.make_PriorCollection is not saved[2]
                raise RuntimeError("forced outer prior failure")
            with original(*args, **kwargs):
                yield
        monkeypatch.setattr(physo_budget, "source_budget_prior", fail)
    with pytest.raises(RuntimeError, match="forced"):
        discover(r, c)
    assert (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer) == saved


@pytest.mark.native
def test_completed_iteration_deadline_restores_all_scopes(tmp_path):
    import time

    import torch
    from ml_platform_symbolic.engines.physo import discover
    from physo.physym import execute, free_const, prior, reward

    r, c = request(tmp_path)
    r.limits.update(probes=[{"x": 1.0}], response_bounds=[0, None], response_enforcement="hard")
    c.deadline = time.monotonic() - 1
    saved = (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        result = discover(r, c)
    finally:
        torch.set_default_dtype(dtype)
    assert result.summary["termination_reason"] == "wall_budget_after_completed_iteration"
    assert result.summary["native_execution_status"] == "budget_exhausted"
    assert result.candidates and all(x.native_check["passed"] for x in result.candidates)
    assert (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer) == saved


@pytest.mark.native
def test_larger_all_invalid_batch_has_no_fabricated_candidates_and_restores(tmp_path):
    import torch
    from ml_platform_symbolic.engines.physo import discover
    from physo.physym import execute, free_const, prior, reward

    r, c = request(tmp_path)
    call = {"slot": "h", "args": [{"var": "x"}]}
    r.equation_template["ast"] = {"op": "div", "args": [{"const": 1.0}, {"op": "sub", "args": [call, deepcopy(call)]}]}
    r.operators += ["div", "sub"]
    r.options.update(batch_size=256, max_n_evaluations=256)
    saved = (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        result = discover(r, c)
    finally:
        torch.set_default_dtype(dtype)
    assert result.status == "failed" and not result.candidates
    assert result.summary["termination_reason"] == "no_positive_feasible_reward"
    assert result.summary["template_search"]["strict_real"]["nonfinite_fit_failures"] > 0
    assert (execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"], prior.make_PriorCollection, reward.make_RewardsComputer) == saved

@pytest.mark.native
def test_outer_invalid_intermediate_matches_saved_model_on_domain_boundary(tmp_path):
    import pandas as pd
    import torch
    from ml_platform_symbolic.model import load_model, save_model

    r, c = request(tmp_path)
    r.variables[0]["domain"] = [0.0, 2.0]
    call = {"slot": "h", "args": [{"var": "x"}]}
    r.equation_template["ast"] = {"op": "exp", "args": [
        {"op": "neg", "args": [{"op": "div", "args": [{"const": 1.0}, call]}]}
    ]}
    t = SharedSlot(r, c)
    p, export = native_program(r, t, ["x"])
    points = np.array([[0.0], [0.25], [2.0]])
    with strict_native_scope(t):
        native = p.get_prog(0)(torch.tensor(points.T)).detach().numpy()
        eq = export(p.get_prog(0))
    portable = evaluate(eq, points, r.variables)
    assert np.isnan(native[0]) and np.isfinite(native[1:]).all()
    np.testing.assert_allclose(portable, native, rtol=0, atol=0, equal_nan=True)
    path = tmp_path / "boundary.zip"
    save_model(path, eq, {"variables": r.variables, "target": r.target, "feature_ranges": [[0.0, 2.0]]}, {})
    model = load_model(path)
    # Deployment API rejects undefined rows rather than silently dropping them.
    with pytest.raises(ValueError, match="undefined at row 0; no rows were dropped"):
        model.predict(pd.DataFrame({"x": points[:, 0]}))
    np.testing.assert_allclose(
        model.predict(pd.DataFrame({"x": points[1:, 0]})),
        native[1:], rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        evaluate(model.equation, points, r.variables), native, rtol=0, atol=0, equal_nan=True
    )
