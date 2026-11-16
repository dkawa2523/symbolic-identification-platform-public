"""Affine reduction is same-loss native SISSO, not nonlinear outer fitting."""
import json
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.coordinates import source_complexity
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.torchsisso import _equation, validate_request
from ml_platform_symbolic.engines.torchsisso_template import LinearSlot
from ml_platform_symbolic.equation_templates import validate_template
from ml_platform_symbolic.expressions import evaluate


def request(tmp_path, shared=True):
    x = np.linspace(.3, 3., 60)[:, None]
    variables = [{"id": "x", "column": "x", "unit": "1", "domain": [.2, 3.5]}]
    logx = {"op": "log", "args": [{"var": "x"}]}
    call = {"slot": "f", "args": [logx]}
    negative = {"op": "sub", "args": [call, {"op": "mul", "args": [{"const": 3.}, deepcopy(call)]}]}
    if not shared:
        negative = {"op": "mul", "args": [{"const": -2.}, call]}
    offset = {"op": "pow_int", "args": [{"var": "x"}, {"const": 2}]}
    operators = ["add", "sub", "mul", "log", "pow_int"]
    raw = {"coordinate_space": "working", "ast": {"op": "add", "args": [offset, negative]},
           "knowledge": {"kind": "hypothesis", "statement": "General affine reduction mechanism fixture"},
           "slots": {"f": {"arguments": [{"name": "u", "unit": "1"}], "unit": "1",
                             "max_nodes": 5, "allowed_operators": ["add", "mul"]}}}
    template = validate_template(raw, variables, {"unit": "1"}, operators, {})
    r = DiscoveryRequest(x, x[:, 0]**2-2*(.7+1.3*np.log(x[:, 0])), np.arange(len(x)), variables,
                         {"unit": "1"}, operators, {"max_nodes": 80}, 17,
                         {"dimensional": True, "n_expansion": 1, "n_term": 1, "k": 3},
                         equation_template=template)
    return r, DiscoveryContext(tmp_path)


def test_exact_loss_negative_scale_input_transform_and_shared_parameters(tmp_path):
    r, c = request(tmp_path)
    t = LinearSlot(r, c)
    assert t.scale == -2 and len(t.calls) == 2
    np.testing.assert_allclose(t.X[:, 0], np.log(r.X[:, 0]))
    pred = .8+1.1*t.X[:, 0]
    np.testing.assert_allclose(np.sum((t.restore(pred, r.X)-r.y)**2), t.scale**2*np.sum((pred-t.y)**2))
    slot = _equation(["u"], [1.1], .8, t.variables, "1")
    full = t.export(slot)
    assert len(full["parameters"]) == 2  # one intercept/coefficient shared at both calls
    np.testing.assert_allclose(evaluate(full, r.X, r.variables), t.restore(pred, r.X))
    reread = json.loads(json.dumps(full))
    np.testing.assert_array_equal(evaluate(full, r.X, r.variables), evaluate(reread, r.X, r.variables))


def test_exact_source_slot_and_free_parameter_gates_before_fit(tmp_path):
    r, c = request(tmp_path)
    t = LinearSlot(r, c)
    full = t.export(_equation(["u"], [1.], 0., t.variables, "1"))
    size = source_complexity(full, t.coordinate_context)
    b = t.budget()
    costs = b.costs(["u", "(u*u)"])
    r.limits["max_nodes"] = size
    assert b.admissible([0], costs)
    r.limits["max_nodes"] -= 1
    assert not b.admissible([0], costs)
    r.limits["max_nodes"] = 100
    assert not b.admissible([1], costs)  # local 5-node slot budget, even generous total
    r.limits["max_free_parameters"] = 1
    assert not b.admissible([0], costs)


@pytest.mark.parametrize("feature", ["nonlinear", "free", "multi", "different", "soft", "nondimensional", "mapped", "singular", "variable_scale"])
def test_fail_closed_before_native(tmp_path, feature):
    r, c = request(tmp_path)
    t = r.equation_template
    if feature == "nonlinear":
        t["ast"] = {"op": "exp", "args": [t["ast"]]}
    elif feature == "free":
        t["parameters"]["a"] = {"trainable": True}
    elif feature == "multi":
        t["slots"]["g"] = deepcopy(t["slots"]["f"])
    elif feature == "different":
        t["ast"]["args"][1]["args"][0]["args"] = [{"var": "x"}]
    elif feature == "soft":
        r.limits.update(probes=[{"x": 1.}], response_bounds=[0, 1])
        if feature == "soft":
            r.limits["response_enforcement"] = "soft"
    elif feature == "nondimensional":
        r.options["dimensional"] = False
    elif feature == "mapped":
        c.provenance = {"input_geometry": {"map": {"x": {"var": "x"}}}}
    elif feature == "singular":
        r.X[0] = 0
    else:
        t["ast"] = {"op": "mul", "args": [{"var": "x"}, t["ast"]["args"][1]]}
    with pytest.raises((NotImplementedError, ValueError)):
        validate_request(r, c)


@pytest.mark.native
def test_actual_native_screening_sparse_fit_and_export(tmp_path):
    from ml_platform_symbolic.engines.torchsisso import discover
    r, c = request(tmp_path)
    result = discover(r, c)
    assert result.status == "ok", result.summary
    assert result.summary["native_candidates"] == 1
    assert result.summary["native_input_variables"] == ["u"]
    assert result.summary["equation_template"]["shared_calls"] == 2
    eq = result.candidates[0].equation
    np.testing.assert_allclose(evaluate(eq, r.X, r.variables), r.y, rtol=1e-12, atol=1e-12)
    assert result.candidates[0].native_check["passed"]
    assert len(eq["parameters"]) == 2


@pytest.mark.native
@pytest.mark.parametrize("derivative", [False, True])
def test_full_y_hard_response_enters_native_fit_with_negative_scale(tmp_path, derivative):
    from ml_platform_symbolic.engines.torchsisso import discover, prepare_response_search
    from ml_platform_symbolic.response_search import response_feasible
    r, c = request(tmp_path)
    if derivative:
        r.limits["response_checks"] = [{"feature": "x", "step": .05, "order": 1,
                                       "sign": -1, "tolerance": 0., "points": [{"x": 3.}]}]
    else:
        r.limits.update(probes=[{"x": 1.}], response_bounds=[2.9, 3.1])
    cp = prepare_response_search(r, allow_template=True)
    assert not response_feasible(np.asarray(cp["points"])[:, 0]**2-2*(.7+1.3*np.log(np.asarray(cp["points"])[:, 0])), cp)
    result = discover(r, c)
    assert result.status == "ok", result.summary
    assert result.summary["response_search"]["subset_counts"]["feasible"] > 0
    eq = result.candidates[0].equation
    assert response_feasible(evaluate(eq, np.asarray(cp["points"]), r.variables), cp)
    assert np.mean((evaluate(eq, r.X, r.variables)-r.y)**2) > .1
    assert result.candidates[0].native_check["finite_point_feasible"]


def test_multicomposed_arguments_rejected_before_singleton_pruning(tmp_path):
    r, c = request(tmp_path)
    r.equation_template["slots"]["f"]["arguments"].append({"name": "v", "unit": "1"})
    for call in [r.equation_template["ast"]["args"][1]["args"][0],
                 r.equation_template["ast"]["args"][1]["args"][1]["args"][1]]:
        call["args"].append({"var": "x"})
    with pytest.raises(NotImplementedError, match="not monotone"):
        validate_request(r, c)


@pytest.mark.native
def test_fixed_derived_gain_units_and_bundle_reload(tmp_path):
    import pandas as pd
    from ml_platform_symbolic.engines.torchsisso import discover
    from ml_platform_symbolic.model import load_model, save_model
    r, c = request(tmp_path, shared=False)
    raw = deepcopy(r.equation_template)
    raw["parameters"] = {"base": {"value": 1., "trainable": False, "unit": "m"},
                         "gain": {"derived_from": {"op": "mul", "args": [{"const": 2.}, {"param": "base"}]}, "unit": "m"},
                         "offset": {"value": 3., "trainable": False, "unit": "m"}}
    raw["ast"] = {"op": "add", "args": [{"param": "offset"}, {"op": "mul", "args": [{"param": "gain"},
                   {"slot": "f", "args": [{"var": "x"}]}]}]}
    r.target["unit"] = "m"
    r.equation_template = validate_template(raw, r.variables, r.target, r.operators, {})
    r.y = 3.+2.*(.4+1.2*r.X[:, 0])
    result = discover(r, c)
    assert result.status == "ok", result.summary
    assert result.summary["coefficient_units"] == "1; intercept=1"
    assert result.summary["public_output_unit"] == "m"
    eq = result.candidates[0].equation
    assert eq["parameters"]["base"]["trainable"] is False
    points = np.array([.41, 1.4, 2.71])
    expected = 3.+2.*(.4+1.2*points)
    path = tmp_path / "model.zip"
    save_model(path, eq, {"variables": r.variables, "target": r.target, "feature_ranges": [[.2, 3.5]]}, {})
    prediction = load_model(path).predict(pd.DataFrame({"x": points}))
    np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(prediction, evaluate(eq, points[:, None], r.variables))


@pytest.mark.native
def test_real_two_term_sis_search_multi_variable_arguments(tmp_path):
    from ml_platform_symbolic.engines.torchsisso import discover
    r, c = request(tmp_path, shared=False)
    rng = np.random.default_rng(47)
    r.X = rng.uniform(.3, 3, (70, 2))
    r.row_ids = np.arange(70)
    r.variables.append({"id": "z", "column": "z", "unit": "1", "domain": [.2, 3.5]})
    raw = deepcopy(r.equation_template)
    raw["slots"]["f"]["arguments"] = [{"name": "u", "unit": "1"}, {"name": "v", "unit": "1"}]
    raw["slots"]["f"]["max_nodes"] = 9
    raw["ast"]["args"][1]["args"][1]["args"] = [{"var": "x"}, {"var": "z"}]
    r.equation_template = validate_template(raw, r.variables, r.target, r.operators, {})
    r.options.update(n_term=2, k=2)
    r.y = r.X[:, 0]**2-2*(.7+1.3*r.X[:, 0]-.4*r.X[:, 1])
    result = discover(r, c)
    assert result.status == "ok", result.summary
    assert result.summary["native_candidates"] == 2
    assert result.summary["source_complexity_budget"]["combination_events"]
    eq = result.candidates[-1].equation
    assert len([p for p in eq["parameters"].values() if p["trainable"]]) == 3
    np.testing.assert_allclose(evaluate(eq, r.X, r.variables), r.y, rtol=1e-12, atol=1e-12)


def test_large_known_offset_loses_response_signal_and_is_rejected(tmp_path):
    r, c = request(tmp_path, shared=False)
    r.equation_template["ast"]["args"][0] = {"const": 1e30}
    r.y = np.linspace(1., 2., len(r.X))
    with pytest.raises(ValueError, match="Float64 cannot resolve"):
        validate_request(r, c)


def test_strict_descriptor_domain_is_gated_before_native_subset_fit(tmp_path):
    r, c = request(tmp_path)
    t = LinearSlot(r, c)
    # u=log(x) crosses zero at x=1; exp(log(u)) must not recover invalid log(u).
    t.validation_points = np.array([[1.]])
    b = t.budget()
    costs = b.costs(["u", "exp(ln(u))"])
    assert b.admissible([0], costs)
    assert not b.admissible([1], costs)



def test_exact_identity_reduction_not_rejected_for_preexisting_ulp_diagnostic(tmp_path):
    r, c = request(tmp_path, shared=False)
    r.equation_template["ast"] = {"slot": "f", "args": [{"var": "x"}]}
    r.y = 1e6+np.arange(len(r.X))
    t = LinearSlot(r, c)
    assert t.reduction_check["exact_observed_roundtrip"]
    assert t.reduced_resolution_check["exact_observed_roundtrip"]



@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_reduced_target_outside_native_quadratic_range_refused_before_fit(tmp_path, scale):
    r, c = request(tmp_path, shared=False)
    r.equation_template["ast"] = {"slot": "f", "args": [{"var": "x"}]}
    r.y = scale*np.linspace(1., 2., len(r.X))
    with pytest.raises(ValueError, match="native Float64 quadratic range"):
        validate_request(r, c)


@pytest.mark.native
def test_invalid_raw_ieee_descriptor_removed_before_native_sis_topk(tmp_path):
    import sympy
    import torch
    from TorchSisso.Regressor_dimension import Regressor
    r, c = request(tmp_path)
    r.equation_template["slots"]["f"]["max_nodes"] = 9
    t = LinearSlot(r, c)
    # Native raw exp(log(u)) can recover to 0 at validation u=0, but strict-real
    # descriptor domain rejects it. Both columns are finite/varying on Train.
    u = np.linspace(.2, 2., len(r.X))
    t.validation_points = np.array([[1.]])  # formal u=log(1)=0
    raw = torch.tensor(np.column_stack([u, u]), dtype=torch.float64)
    reg = Regressor(raw, torch.tensor(u), ["exp(ln(u))", "u"], [sympy.Symbol("1")]*2,
                    dimension=1, sis_features=1, output_dim=sympy.Symbol("1"), node_budget=t.budget())
    assert reg.names == ["u"]  # constructor has not called regressor_fit/SIS yet
    assert reg.x.shape[1] == 1
    reg.regressor_fit()
    assert reg.names == ["u"]
