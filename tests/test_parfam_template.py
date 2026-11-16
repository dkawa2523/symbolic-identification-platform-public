"""Shared native coefficient families: full-y arithmetic and portable identity."""

import numpy as np
import pytest
from ml_platform_symbolic.engines.parfam_template import CompositeFamily, strict_torch, validate_template_scope
from ml_platform_symbolic.expressions import evaluate, inspect_equation

torch = pytest.importorskip("torch")


class Native:
    def get_number_parameters(self):
        return 2

    def testing_mode(self):
        pass


def slot_export(native, c, reduced=False):
    return {
        "schema_version": 2,
        "ast": {"op": "add", "args": [{"op": "mul", "args": [{"param": "c0"}, {"var": "u"}]}, {"param": "c1"}]},
        "parameters": {f"c{i}": {"value": float(v), "trainable": True, "unit": None} for i, v in enumerate(c)},
    }


V = [{"id": "x", "column": "x", "unit": None}]
X = {"var": "x"}


def call(arg=X):
    return {"slot": "h", "args": [arg]}


def spec(ast):
    return {
        "coordinate_space": "working",
        "ast": ast,
        "parameters": {},
        "components": {},
        "knowledge": {"kind": "hypothesis", "statement": "Declared structure for an adapter mechanism test"},
        "slots": {
            "h": {
                "arguments": [{"name": "u", "unit": None}],
                "unit": None,
                "max_nodes": 20,
                "allowed_operators": ["add", "mul"],
            }
        },
    }


def test_shared_parameters_gradient_new_inputs_and_reload():
    template = spec({"op": "add", "args": [call(), call({"op": "mul", "args": [{"const": 2}, X]})]})
    model = CompositeFamily(Native(), template, V, slot_export)
    c = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y = model.predict(c, x)
    torch.testing.assert_close(y, 6 * x[:, 0] + 6, rtol=0, atol=0)
    y.sum().backward()
    torch.testing.assert_close(c.grad, torch.tensor([18.0, 6.0], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(model.predict(c, x + 4), 6 * (x[:, 0] + 4) + 6, rtol=0, atol=0)
    import json

    equation = json.loads(json.dumps(model.equation(c.detach().numpy())))
    assert len(equation["parameters"]) == 2
    np.testing.assert_array_equal(evaluate(equation, x.numpy(), V), y.detach().numpy())
    assert inspect_equation(equation, V)["n_nodes"] == 13
    assert model.source_complexity(60) == 13


@pytest.mark.parametrize(
    "ast",
    [
        {"op": "div", "args": [{"const": 1}, call()]},
        {"op": "mul", "args": [{"const": 0}, {"op": "div", "args": [{"const": 1}, call()]}]},
        {"op": "exp", "args": [{"op": "neg", "args": [{"op": "div", "args": [{"const": 1}, call()]}]}]},
    ],
)
def test_invalid_intermediate_is_not_repaired(ast):
    model = CompositeFamily(Native(), spec(ast), V, slot_export)
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    c = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    result = model.predict(c, x)
    assert not torch.isfinite(result[0]) and torch.isfinite(result[1])
    np.testing.assert_array_equal(
        np.isfinite(evaluate(model.equation(c.detach().numpy()), x.numpy(), V)), torch.isfinite(result).numpy()
    )
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective

    objective = ResponseObjective({"data_scale": 1})
    loss = objective.model_loss(result, torch.zeros(2), model, c)
    assert torch.isinf(loss)
    loss.backward()
    assert torch.isfinite(c.grad).all()


def test_zero_power_and_unused_nonfinite_input_match_common():
    value, valid = strict_torch({"op": "pow_int", "args": [X, {"const": 0}]}, {}, torch.tensor([[0.0]]), V)
    assert valid.all() and value.item() == 1
    value, valid = strict_torch({"const": 2}, {}, torch.tensor([[float("inf")]]), V)
    assert valid.all() and value.item() == 2


def test_expanded_budget_and_unsupported_scope_rejected():
    t = spec({"op": "add", "args": [call(), call()]})
    t["slots"]["h"]["max_nodes"] = 2
    model = CompositeFamily(Native(), t, V, slot_export)
    assert model.source_complexity(60) > 60
    t["parameters"]["a"] = {"value": 1, "unit": None, "trainable": True}
    with pytest.raises(NotImplementedError, match="outer free"):
        validate_template_scope(t)
    with pytest.raises(NotImplementedError, match="nested"):
        validate_template_scope(spec(call(call())))


@pytest.mark.native
@pytest.mark.parametrize("outer", ["shared", "square"])
def test_native_fit_pruning_winner_and_export_use_full_observed_y(tmp_path, outer):
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.parfam import discover

    x = np.linspace(-1, 1, 40).reshape(-1, 1)
    ast = (
        {"op": "add", "args": [call(), call({"op": "mul", "args": [{"const": 2}, X]})]}
        if outer == "shared"
        else {"op": "pow_int", "args": [call(), {"const": 2}]}
    )
    template = spec(ast)
    variables = [{"id": "x", "column": "x", "unit": None, "domain": [-2, 2]}]
    y = 6 * x[:, 0] + 6 if outer == "shared" else (2 * x[:, 0] + 0.3) ** 2
    r = DiscoveryRequest(
        x,
        y,
        np.arange(len(x)),
        variables,
        {"column": "y", "unit": None},
        ["add", "mul", "pow_int"],
        {
            "max_wall_seconds": 120,
            "max_nodes": 60,
            "require_units": False,
            "probes": [],
            "response_bounds": [None, None],
            "response_checks": [],
            "response_mode": "search",
        },
        42,
        {"family": "polynomial", "degree": 1, "maxiter": 5, "repetitions": 1, "seconds": 60},
        equation_template=template,
    )
    result = discover(r, DiscoveryContext(tmp_path / outer))
    assert result.candidates
    assert {"winner_full", "winner_reduced", "completed_family_0"} <= set(result.summary["variants"].values())
    assert result.summary["template_search"]["loss_target"] == "original observed y"
    import json

    for candidate in result.candidates:
        equation = json.loads(json.dumps(candidate.equation))
        assert candidate.native_check["passed"]
        assert np.sqrt(np.mean((evaluate(equation, x, variables) - y) ** 2)) < 1e-3
        assert len(equation["parameters"]) == 2
        import pandas as pd
        from ml_platform_symbolic.model import load_model, save_model

        saved = tmp_path / (candidate.candidate_id.replace("/", "_") + ".zip")
        save_model(saved, equation, {"variables": variables, "target": r.target, "feature_ranges": [[-1, 1]]}, {})
        reloaded = load_model(saved)
        np.testing.assert_array_equal(reloaded.predict(pd.DataFrame({"x": x[:, 0]})), evaluate(equation, x, variables))
    r.limits["max_nodes"] = 2
    with pytest.raises(ValueError, match="No configured ParFam family fits max_nodes"):
        discover(r, DiscoveryContext(tmp_path / "too-small"))


@pytest.mark.parametrize("feature", ["multiple_slots", "outer_free", "nested", "operator", "mapped"])
def test_unimplemented_common_template_features_rejected_before_native_import(tmp_path, monkeypatch, feature):
    import builtins
    from copy import deepcopy

    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.parfam import validate_request
    from ml_platform_symbolic.engines.parfam_template import native_template_scope

    template = spec(call())
    provenance = {}
    if feature == "multiple_slots":
        template["slots"]["other"] = deepcopy(template["slots"]["h"])
        template["ast"] = {"op": "add", "args": [call(), {"slot": "other", "args": [X]}]}
    elif feature == "outer_free":
        template["parameters"]["a"] = {"value": 1.0, "unit": None, "trainable": True}
        template["ast"] = {"op": "mul", "args": [{"param": "a"}, call()]}
    elif feature == "nested":
        template["ast"] = call(call())
    elif feature == "operator":
        template["ast"] = {"op": "tan", "args": [call()]}
    else:
        provenance = {"input_geometry": {"map": {"x": {"var": "x"}}}}
    request = DiscoveryRequest(
        np.array([[1.0], [2.0]]),
        np.array([1.0, 2.0]),
        np.arange(2),
        V,
        {"column": "y", "unit": None},
        ["add", "mul", "pow_int", "tan"],
        {
            "max_wall_seconds": 60,
            "max_nodes": 60,
            "require_units": False,
            "probes": [],
            "response_bounds": [None, None],
            "response_checks": [],
            "response_mode": "search",
        },
        42,
        {"family": "polynomial", "degree": 1},
        equation_template=template,
    )
    context = DiscoveryContext(tmp_path, provenance=provenance)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "parfam" or name.startswith("parfam."):
            pytest.fail("Unsupported request attempted a native import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(NotImplementedError):
        validate_request(request, context)
    with pytest.raises(NotImplementedError):
        with native_template_scope(request, context, None):
            pytest.fail("Unsupported scope entered native execution")
