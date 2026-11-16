"""Independent small grammar oracles and end-to-end bounded slot discovery."""

from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.template_enumeration import (
    discover,
    enumerate_balanced_shapes,
    enumerate_combinations,
    enumerate_shapes,
    preflight,
)
from ml_platform_symbolic.equation_templates import validate_template
from ml_platform_symbolic.expressions import evaluate


def request_for(X, y, *, shared=False, outer=False, options=None):
    variables = [
        {"id": f"x{i}", "column": f"x{i}", "unit": "1", "domain": [-3, 3]}
        for i in range(X.shape[1])
    ]
    call = {"slot": "h", "args": [{"var": "x0"}]}
    template = {
        "coordinate_space": "working",
        "ast": call,
        "components": {},
        "parameters": {},
        "slots": {
            "h": {
                "arguments": [{"name": "u", "unit": "1"}],
                "unit": "1",
                "allowed_operators": ["mul"],
                "max_nodes": 3,
            }
        },
        "knowledge": {"kind": "hypothesis", "statement": "One finite unknown function"},
    }
    if shared:
        template["ast"] = {
            "op": "add",
            "args": [call, {"slot": "h", "args": [{"var": "x1"}]}],
        }
    if outer:
        template["ast"] = {
            "op": "add",
            "args": [call, {"op": "mul", "args": [{"param": "a"}, {"var": "x1"}]}],
        }
        template["parameters"] = {
            "a": {"value": 0.3, "trainable": True, "unit": "1", "bounds": [0, 4]}
        }
    operators = ["add", "mul"]
    target = {"column": "y", "unit": "1"}
    template = validate_template(template, variables, target, operators, {})
    return DiscoveryRequest(
        X,
        y,
        np.arange(len(X)),
        variables,
        target,
        operators,
        {"max_nodes": 40},
        7,
        options or {"max_structures": 100},
        equation_template=template,
    )


def test_tiny_enumeration_matches_independent_multiset_oracle():
    request = request_for(np.arange(4.0)[:, None], np.arange(4.0))
    slot = request.equation_template["slots"]["h"]
    shapes = list(enumerate_shapes(slot, 3))
    leaves = [("coefficient",), ("variable", "u"), ("literal", 0.0), ("literal", 1.0)]
    expected = {
        *(leaves),
        *[
            ("mul", a, b) if repr(a) <= repr(b) else ("mul", b, a)
            for i, a in enumerate(leaves)
            for b in leaves[i:]
        ],
    }
    assert set(shapes) == expected
    assert len(shapes) == 14
    assert list(enumerate_shapes(slot, 3)) == shapes


def test_joint_original_response_fit_respects_outer_bounds(tmp_path):
    X = np.random.default_rng(55).uniform(-1, 1, (36, 2))
    request = request_for(X, 2.5 * X[:, 0] + 1.7 * X[:, 1], outer=True)
    result = discover(request, DiscoveryContext(tmp_path))
    best = min(
        result.candidates,
        key=lambda c: np.linalg.norm(
            evaluate(c.equation, X, request.variables) - request.y
        ),
    )
    np.testing.assert_allclose(
        evaluate(best.equation, X, request.variables), request.y, atol=2e-7
    )
    assert best.equation["parameters"]["a"]["value"] == pytest.approx(1.7)
    assert best.equation["parameters"]["a"]["bounds"] == [0, 4]
    assert result.summary["attempted_structures"] == 14
    assert result.summary["grammar_exhausted"]
    assert "no independent" in best.native_check["kind"]
    assert (tmp_path / "enumeration_attempts.json").is_file()


def test_shared_slot_has_one_fitted_parameter_vector_and_repeatable_order(tmp_path):
    X = np.random.default_rng(99).uniform(-1, 1, (30, 2))
    request = request_for(X, -2.0 * (X[:, 0] + X[:, 1]), shared=True)
    first = discover(request, DiscoveryContext(tmp_path / "first"))
    second = discover(deepcopy(request), DiscoveryContext(tmp_path / "second"))
    assert [c.candidate_id for c in first.candidates] == [
        c.candidate_id for c in second.candidates
    ]
    best = min(
        first.candidates,
        key=lambda c: np.linalg.norm(
            evaluate(c.equation, X, request.variables) - request.y
        ),
    )
    assert len(best.equation["parameters"]) == 1
    np.testing.assert_allclose(
        evaluate(best.equation, X, request.variables), request.y, atol=2e-7
    )
    assert [c.equation for c in first.candidates] == [
        c.equation for c in second.candidates
    ]


def test_structure_budget_is_exact_and_not_exhaustive(tmp_path):
    X = np.linspace(-1, 1, 20)[:, None]
    request = request_for(X, 2 * X[:, 0], options={"max_structures": 5})
    result = discover(request, DiscoveryContext(tmp_path))
    assert result.summary["attempted_structures"] == 5
    assert result.summary["termination"] == "structure_budget"
    assert not result.summary["grammar_exhausted"]
    assert result.summary["native_candidates"] == 5


def test_failed_domains_stay_in_denominator(tmp_path):
    X = np.linspace(-1, -0.1, 20)[:, None]
    request = request_for(X, X[:, 0])
    request.operators = ["log"]
    request.equation_template["slots"]["h"].update(
        allowed_operators=["log"], max_nodes=2
    )
    result = discover(request, DiscoveryContext(tmp_path))
    assert result.summary["attempted_structures"] == 8
    assert result.summary["rejected_structures"] >= 2
    assert (
        result.summary["attempted_structures"]
        == len(result.candidates) + result.summary["rejected_structures"]
    )


def test_unit_and_slot_scope_reject_before_fitting():
    request = request_for(np.ones((3, 1)), np.ones(3))
    request.equation_template["slots"]["h"]["unit"] = None
    with pytest.raises(NotImplementedError, match="known canonical"):
        preflight(request)
    request.equation_template = None
    with pytest.raises(NotImplementedError, match="at least one"):
        preflight(request)


def test_bounded_slot_coefficients_are_not_relaxed(tmp_path):
    X = np.linspace(-1, 1, 30)[:, None]
    request = request_for(
        X,
        3 * X[:, 0],
        options={
            "max_structures": 100,
            "starts": [0.2, 0.8],
            "coefficient_bounds": [0, 1],
        },
    )
    result = discover(request, DiscoveryContext(tmp_path))
    for candidate in result.candidates:
        assert all(
            0 <= p["value"] <= 1 for p in candidate.equation["parameters"].values()
        )
    assert all(
        np.linalg.norm(evaluate(c.equation, X, request.variables) - request.y) > 1
        for c in result.candidates
    )


@pytest.mark.parametrize("mapped", [False, True])
def test_subprocess_compare_choice_bundle_roundtrip(synthetic_config, tmp_path, mapped):
    from ml_platform_symbolic import (
        choose_hypothesis,
        compare_equations,
        create_study,
        finalize_hypothesis,
        load_model,
    )
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import prepare, read_json
    from ml_platform_symbolic.engines.runner import run_discovery

    config = deepcopy(synthetic_config)
    config.pop("constraints")
    spec = request_for(np.ones((3, 1)), np.ones(3)).equation_template
    spec["ast"] = {
        "op": "add",
        "args": [{"op": "mul", "args": [{"const": 2.0}, spec["ast"]]}, {"const": 1.0}],
    }
    config["requirements"] = {
        "allowed_operators": ["add", "mul"],
        "max_nodes": 40,
        "require_units": True,
        "equation_template": spec,
    }
    if mapped:
        config["input_map"] = {
            "kind": "affine_chart",
            "target_columns": ["u"],
            "target_units": {"u": "1"},
            "forward": {"matrix": [[2]], "offset": [1]},
            "inverse": {"matrix": [[0.5]], "offset": [-0.5]},
        }
        config["requirements"]["allowed_operators"] += ["sub", "div"]
        # Template x0 is now u, not the original CSV x. Undo the declared chart
        # explicitly in the function argument, while public inference maps once.
        spec["ast"]["args"][0]["args"][1]["args"] = [
            {"op": "div", "args": [
                {"op": "sub", "args": [{"var": "x0"}, {"const": 1.0}]}, {"const": 2.0}
            ]}
        ]
    config["search"].update(
        engines=["template_enumeration"],
        seeds=[7],
        max_wall_seconds=30,
        candidate_pool_policy="all",
        engine_options={"template_enumeration": {"max_structures": 20}},
    )
    config = resolve_config(config)
    refs = prepare(config, tmp_path / "prepare")
    discovery = run_discovery(
        refs, read_json(refs["run"]), "template_enumeration", 7, tmp_path / "discover"
    )
    summary = read_json(discovery["summary"])
    assert summary["status"] == "ok"
    assert summary["attempted_structures"] == 14
    study = create_study(
        {
            "prepared_run": str(refs["run"]),
            "sources": [{"discovery": discovery["summary"]}],
            "engineering_tolerance": 0.01,
            "final_history": "known_exposed",
        },
        run_dir=tmp_path / "study",
    )
    compared = compare_equations(study, run_dir=tmp_path / "comparison")
    from ml_platform_symbolic.report_data import build_report_view

    view = build_report_view(compared)
    best = min(view["candidates"], key=lambda c: c["selection"]["rmse"])
    choice = choose_hypothesis(
        compared,
        run_dir=tmp_path / "choice",
        candidate_id=best["id"],
        reason="Synthetic regression only; no scientific adoption",
    )
    final = finalize_hypothesis(study, choice, run_dir=tmp_path / "final")
    model = load_model(final.artifacts["model_bundle"])
    import pandas as pd

    frame = pd.read_csv(config["data"]["local_path"])
    np.testing.assert_allclose(model.predict(frame), frame.y, atol=1e-7)
    np.testing.assert_array_equal(
        model.predict(frame), model.predict(frame[frame.columns[::-1]])
    )

def test_completed_checkpoint_has_partial_status_and_retains_attempts(tmp_path):
    import json

    X = np.linspace(-1, 1, 12)[:, None]
    request = request_for(X, 2 * X[:, 0], options={"max_structures": 12})
    snapshots = []
    context = DiscoveryContext(tmp_path, checkpoint=lambda r: snapshots.append(deepcopy(r)))
    result = discover(request, context)
    assert len(snapshots) == 1
    assert snapshots[0].status == "partial"
    assert snapshots[0].summary["native_execution_status"] == "partial"
    assert snapshots[0].summary["attempted_structures"] == 10
    ledger = json.loads((tmp_path / "enumeration_attempts.json").read_text())
    assert len(ledger["structures"]) == 12
    assert result.summary["fit_attempts"] == sum(len(a["fit_attempts"]) for a in ledger["structures"])
    assert result.summary["prediction_evaluations"] > 0


def test_unit_rejection_still_checkpoints_candidates_before_wall_timeout(tmp_path):
    import json

    X = np.linspace(0.1, 1.0, 12)[:, None]
    request = request_for(X, 2 * X[:, 0], options={"max_structures": 12})
    request.variables[0]["unit"] = "s"
    request.target["unit"] = "m"
    slot = request.equation_template["slots"]["h"]
    slot["arguments"][0]["unit"] = "s"
    slot["unit"] = "m"
    snapshots = []
    context = DiscoveryContext(tmp_path)

    def checkpoint(progress):
        ledger = json.loads((tmp_path / "enumeration_attempts.json").read_text())
        snapshots.append((deepcopy(progress), ledger))
        context.deadline = 0.0  # The next iteration models an elapsed wall budget.

    context.checkpoint = checkpoint
    result = discover(request, context)
    assert len(snapshots) == 1
    saved, ledger = snapshots[0]
    assert len(ledger["structures"]) == saved.summary["attempted_structures"] == 10
    assert ledger["structures"][-1]["stage"] == "coefficient_units"
    assert ledger["structures"][-1]["status"] == "rejected"
    assert saved.candidates and saved.status == "partial"
    assert result.status == "partial" and result.summary["termination"] == "wall_budget"
    assert [c.equation for c in result.candidates] == [c.equation for c in saved.candidates]
    assert [c.candidate_id for c in result.candidates] == [c.candidate_id for c in saved.candidates]


def two_slot_request(X, y):
    request = request_for(X, y, shared=True, options={"max_structures": 1000})
    template = request.equation_template
    template["slots"]["g"] = deepcopy(template["slots"]["h"])
    template["ast"]["args"][1]["slot"] = "g"
    return request


def test_multislot_product_matches_complete_small_grammar_oracle():
    request = two_slot_request(np.ones((3, 2)), np.ones(3))
    slots, options = preflight(request)
    actual = list(enumerate_combinations(slots, options["max_slot_nodes"]))
    one = list(enumerate_shapes(slots["g"], 3))
    expected = {(g, h) for g in one for h in one}
    assert {(row["g"], row["h"]) for row in actual} == expected
    assert len(actual) == 196
    sizes = {shape: (1 if shape[0] in {"variable", "coefficient", "literal"} else 3) for shape in one}
    totals = [sum(sizes[s] for s in row.values()) for row in actual]
    assert totals == sorted(totals)
    assert list(enumerate_combinations(slots, options["max_slot_nodes"])) == actual


def test_independent_slots_fit_different_functions_and_keep_parameters_separate(tmp_path):
    X = np.random.default_rng(135).uniform(-1, 1, (36, 2))
    request = two_slot_request(X, 2.5 * X[:, 0] - 1.7 * X[:, 1])
    result = discover(request, DiscoveryContext(tmp_path))
    best = min(result.candidates, key=lambda c: np.linalg.norm(evaluate(c.equation, X, request.variables) - request.y))
    np.testing.assert_allclose(evaluate(best.equation, X, request.variables), request.y, atol=2e-7)
    assert len(best.equation["parameters"]) == 2
    assert set(best.equation["template"]["slots"]) == {"g", "h"}
    assert result.summary["attempted_structures"] == 196
    assert result.summary["grammar_exhausted"]


def test_dimensional_slot_slope_and_refusal_of_unit_gauges(tmp_path):
    import json

    from ml_platform_symbolic.constraints import dimensions
    from ml_platform_symbolic.units import unit_dimensions

    X = np.linspace(0.1, 2, 24)[:, None]
    request = request_for(X, 3.25 * X[:, 0])
    request.variables[0]["unit"] = "s"
    request.target["unit"] = "m"
    slot = request.equation_template["slots"]["h"]
    slot["arguments"][0]["unit"] = "s"
    slot["unit"] = "m"
    result = discover(request, DiscoveryContext(tmp_path))
    best = min(result.candidates, key=lambda c: np.linalg.norm(evaluate(c.equation, X, request.variables) - request.y))
    np.testing.assert_allclose(evaluate(best.equation, X, request.variables), request.y, atol=2e-7)
    assert len(best.equation["parameters"]) == 1
    parameter = next(iter(best.equation["parameters"].values()))
    assert unit_dimensions(parameter["unit"]) == unit_dimensions("m/s")
    assert dimensions(best.equation, request.variables, request.target)[0] == "pass"
    ledger = json.loads((tmp_path / "enumeration_attempts.json").read_text())
    failures = [row["reason"] for row in ledger["structures"] if row.get("stage") == "coefficient_units"]
    assert any("Inconsistent" in reason for reason in failures)
    assert any("Underdetermined" in reason for reason in failures)
    assert result.summary["attempted_structures"] == len(result.candidates) + result.summary["rejected_structures"]


def test_dimensional_inference_solves_rational_and_transcendental_constraints():
    from ml_platform_symbolic.engines._enumeration_units import assign_coefficient_units
    from ml_platform_symbolic.units import unit_dimensions

    equation = {
        "schema_version": 2,
        "ast": {"op": "mul", "args": [
            {"param": "amplitude"},
            {"op": "exp", "args": [{"op": "mul", "args": [{"param": "rate"}, {"var": "t"}]}]},
        ]},
        "parameters": {name: {"value": 1.0, "trainable": True, "unit": "1"} for name in ["amplitude", "rate"]},
    }
    slot = {"arguments": [{"name": "t", "unit": "s"}], "unit": "m"}
    assign_coefficient_units(equation, slot)
    assert unit_dimensions(equation["parameters"]["amplitude"]["unit"]) == unit_dimensions("m")
    assert unit_dimensions(equation["parameters"]["rate"]["unit"]) == unit_dimensions("1/s")
    equation["ast"] = {"op": "pow_int", "args": [{"param": "amplitude"}, {"const": 3}]}
    del equation["parameters"]["rate"]
    assign_coefficient_units(equation, slot)
    assert unit_dimensions(equation["parameters"]["amplitude"]["unit"]) == unit_dimensions("m^(1/3)")


def test_slot_specific_limits_and_joint_budget_are_not_silently_relaxed(tmp_path):
    request = two_slot_request(np.ones((3, 2)), np.ones(3))
    request.options = {"max_slot_nodes": {"g": 1, "h": 3}, "max_structures": 19}
    result = discover(request, DiscoveryContext(tmp_path))
    assert result.summary["attempted_structures"] == 19
    assert result.summary["termination"] == "structure_budget"
    for candidate in result.candidates:
        assert "op" not in candidate.equation["template"]["slots"]["g"]["ast"]
    request.options["max_slot_nodes"] = {"g": 1}
    with pytest.raises(ValueError, match="every slot"):
        preflight(request)


def test_independent_constant_slots_retain_the_identifiability_counterexample(tmp_path):
    from ml_platform_symbolic.equation_evaluation.numerical import local_parameter_evidence

    X = np.random.default_rng(199).uniform(-1, 1, (12, 2))
    request = two_slot_request(X, np.full(len(X), 3.0))
    request.options = {"max_structures": 1}
    result = discover(request, DiscoveryContext(tmp_path))
    candidate = result.candidates[0]
    np.testing.assert_allclose(evaluate(candidate.equation, X, request.variables), request.y, atol=2e-7)
    evidence = local_parameter_evidence(
        {"kind": "explicit", "equation": candidate.equation}, X, request.variables, request.target
    )
    assert evidence["status"] == "computed"
    assert evidence["rank"] == 1 and evidence["nullity"] == 1
    assert not evidence["full_column_rank"]


def test_dimensional_example_maps_public_units_and_saved_predictions(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    import pandas as pd
    from ml_platform_symbolic import choose_hypothesis, finalize_hypothesis, load_model
    from ml_platform_symbolic.report_data import build_report_view
    from ml_platform_symbolic.units import unit_dimensions

    spec = spec_from_file_location(
        "dimensional_example", Path(__file__).resolve().parents[1] / "examples/units.py"
    )
    example = module_from_spec(spec)
    spec.loader.exec_module(example)
    result = example.run(tmp_path / "example")
    view = build_report_view(result["comparison"])
    best = min(view["candidates"], key=lambda c: c["selection"]["rmse"])
    assert best["selection"]["rmse"] < 1e-8
    choice = choose_hypothesis(
        result["comparison"], run_dir=tmp_path / "choice", candidate_id=best["id"],
        reason="Synthetic integration regression only; known response, no scientific adoption",
    )
    final = finalize_hypothesis(result["study"], choice, run_dir=tmp_path / "final")
    model = load_model(final.artifacts["model_bundle"])
    frame = pd.read_csv(tmp_path / "example/calculation/table.csv")
    np.testing.assert_allclose(model.predict(frame), frame.response_mm, atol=1e-5)
    # New, independently chosen points and reversed public column order.
    external = pd.DataFrame({"length_cm": [65.0, 95.0], "time_ms": [800.0, 400.0]})
    expected = 2.5 * external.time_ms - 17 * external.length_cm
    np.testing.assert_allclose(model.predict(external), expected, atol=1e-5)
    params = model.equation["parameters"]
    assert params["baseline"]["value"] == 0.5 and not params["baseline"]["trainable"]
    assert {tuple(unit_dimensions(p["unit"])) for n, p in params.items() if n != "baseline"} == {
        tuple(unit_dimensions("m/s")), tuple(unit_dimensions("1"))
    }


def test_template_validation_binds_to_chart_units_and_chart_variable_count(synthetic_config):
    from ml_platform_symbolic.config import resolve_config

    cfg = deepcopy(synthetic_config)
    cfg.pop("constraints")
    cfg["search"]["engines"] = ["known_form"]
    cfg["input_map"] = {
        "kind": "affine_chart", "target_columns": ["elapsed"], "target_units": {"elapsed": "s"},
        "forward": {"matrix": [[1]], "offset": [0]},
        "inverse": {"matrix": [[1]], "offset": [0]},
    }
    template = {
        "coordinate_space": "working",
        "ast": {"op": "div", "args": [{"var": "x0"}, {"param": "duration"}]},
        "parameters": {"duration": {"value": 2.0, "unit": "s", "trainable": False}},
        "knowledge": {"kind": "hypothesis", "statement": "Dimensionless mapped elapsed duration"},
    }
    cfg["requirements"] = {"allowed_operators": ["div"], "require_units": True, "equation_template": template}
    resolve_config(cfg)  # Source x has unit1, but template x0 is elapsed with unit s.
    cfg["data"]["feature_columns"] = ["x", "z"]
    cfg["data"]["units"]["z"] = "1"
    cfg["domain"] = {"kind": "affine", "equalities": {"matrix": [[-2, 1]], "rhs": [0]}}
    cfg["input_map"]["forward"]["matrix"] = [[1, 0]]
    cfg["input_map"]["inverse"]["matrix"] = [[1], [2]]
    cfg["input_map"]["inverse"]["offset"] = [0, 0]
    resolve_config(cfg)
    template["ast"]["args"][0] = {"var": "x1"}
    with pytest.raises(ValueError, match="Unknown variable"):
        resolve_config(cfg)  # Source x1 exists, but there is no mapped x1.


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "2000", None])
def test_structure_budget_requires_an_explicit_positive_integer(budget):
    request = request_for(np.ones((3, 1)), np.ones(3), options={"max_structures": budget})
    with pytest.raises(ValueError, match="max_structures must be a positive integer"):
        preflight(request)


def test_explicit_budget_reaches_three_independent_dimensional_slopes(tmp_path):
    X = np.random.default_rng(8119).uniform(0.1, 1.2, (24, 3))
    request = request_for(X, X @ np.array([2.5, -1.7, 0.8]))
    request.options = {"order": "size", "max_structures": 1856, "max_nfev": 50, "starts": [1.0]}
    request.target["unit"] = "m"
    template = request.equation_template
    prototype = deepcopy(template["slots"]["h"])
    template["slots"] = {}
    calls = []
    for i, (name, unit) in enumerate(zip(["a", "b", "c"], ["s", "kg", "K"], strict=True)):
        request.variables[i]["unit"] = unit
        slot = deepcopy(prototype)
        slot["arguments"][0]["unit"] = unit
        slot["unit"] = "m"
        template["slots"][name] = slot
        calls.append({"slot": name, "args": [{"var": f"x{i}"}]})
    template["ast"] = {"op": "add", "args": [calls[0], {"op": "add", "args": calls[1:]}]}
    result = discover(request, DiscoveryContext(tmp_path))
    assert result.summary["attempted_structures"] == 1856
    assert result.summary["termination"] == "structure_budget"
    best = min(result.candidates, key=lambda c: np.linalg.norm(evaluate(c.equation, X, request.variables) - request.y))
    # All three c*u bodies first occur here in the unchanged ordered product.
    assert best.candidate_id == "template_enumeration/7/1855"
    np.testing.assert_allclose(evaluate(best.equation, X, request.variables), request.y, atol=2e-7)
    assert len(best.equation["parameters"]) == 3


def test_known_template_response_constraint_changes_joint_fit_not_grammar(tmp_path):
    X = np.linspace(-1, 1, 12)[:, None]
    base = request_for(X, np.full(12, -2.0), options={
        "max_structures": 4, "max_slot_nodes": 1, "starts": [1.0, -1.0],
        "objective": "mse",
    })
    base.equation_template["ast"] = {"op": "add", "args": [base.equation_template["ast"], {"const": 1.0}]}
    base.equation_template = validate_template(base.equation_template, base.variables, base.target, base.operators, {})
    base.limits.update(probes=[{"x0": -1.0}, {"x0": 1.0}], response_bounds=[0.0, None])
    original = deepcopy(base.equation_template)
    outcomes = {}
    for mode in ("filter", "search"):
        req = deepcopy(base)
        req.limits["response_mode"] = mode
        outcomes[mode] = discover(req, DiscoveryContext(tmp_path / mode))
    assert [c.candidate_id for c in outcomes["filter"].candidates] == [c.candidate_id for c in outcomes["search"].candidates]
    # First enumerated structure is a free constant in the known g(x) + 1 relation.
    predictions = {mode: evaluate(result.candidates[0].equation, X, base.variables) for mode, result in outcomes.items()}
    np.testing.assert_allclose(predictions["filter"], -2.0, atol=1e-7)
    np.testing.assert_allclose(predictions["search"], 0.0, atol=1e-6)
    assert outcomes["search"].summary["response_search"]["search_response_constraints"]
    assert "structure order independent of response constraints" in outcomes["search"].summary["response_search"]["application"]
    assert base.equation_template == original
    # Feasibility is checked without relaxing the declared response bound.
    assert (predictions["search"] >= 0).all()


def test_finite_only_probes_do_not_invent_response_bounds(tmp_path):
    X = np.linspace(-1, 1, 10)[:, None]
    request = request_for(X, np.full(10, -2.0), options={
        "max_structures": 1, "max_slot_nodes": 1, "starts": [1.0],
    })
    request.limits.update(response_mode="search", probes=[{"x0": 0.0}])
    result = discover(request, DiscoveryContext(tmp_path))
    assert result.candidates
    predicted = evaluate(result.candidates[0].equation, X, request.variables)
    np.testing.assert_allclose(predicted, -2.0, atol=1e-6)
    assert result.summary["response_search"]["terms"][0]["lower"] is None
    assert result.summary["response_search"]["terms"][0]["upper"] is None


@pytest.mark.parametrize("operators, maximum", [(["mul"], 5), (["add", "sub", "div", "pow_int"], 5), (["neg", "sin", "mul"], 4)])
def test_balanced_order_preserves_complete_small_grammar(operators, maximum):
    slot = {"arguments": [{"name": "u"}], "allowed_operators": operators}
    balanced = list(enumerate_balanced_shapes(slot, maximum))
    legacy = list(enumerate_shapes(slot, maximum))
    assert len(balanced) == len(set(balanced))
    assert set(balanced) == set(legacy)
    assert balanced == list(enumerate_balanced_shapes(slot, maximum))


def test_balanced_small_budget_reaches_declared_sizes_and_roots_without_lower_materialization(monkeypatch):
    from itertools import islice

    from ml_platform_symbolic.engines import template_enumeration as engine
    def prohibited(*args):
        raise AssertionError("Balanced traversal must not materialize legacy lower levels")
    monkeypatch.setattr(engine, "enumerate_shapes", prohibited)
    slot = {"arguments": [{"name": "u"}], "allowed_operators": ["add", "sub", "mul", "div", "pow_int"]}
    prefix = list(islice(engine.enumerate_balanced_shapes(slot, 9), 21))
    def size(shape):
        return 1 if shape[0] in {"coefficient", "variable", "literal"} else 1 + sum(size(c) for c in shape[1:])
    assert {size(shape) for shape in prefix} == {1, 3, 5, 7, 9}
    assert {shape[0] for shape in prefix} >= set(slot["allowed_operators"])


def test_balanced_multislot_preserves_cartesian_grammar():
    slot = {"arguments": [{"name": "u"}], "allowed_operators": ["mul"]}
    slots = {"a": slot, "b": slot}
    legacy = list(enumerate_combinations(slots, {"a": 3, "b": 3}, "size"))
    balanced = list(enumerate_combinations(slots, {"a": 3, "b": 3}, "balanced"))
    def encode(row):
        return tuple(row.items())
    assert set(map(encode, legacy)) == set(map(encode, balanced))
    assert len(balanced) == len(legacy)


def test_balanced_summary_reports_actual_coverage_and_honors_structure_budget(tmp_path):
    X = np.linspace(0.1, 1, 8)[:, None]
    req = request_for(X, X[:, 0], options={"max_structures": 3, "max_slot_nodes": 3, "max_nfev": 10})
    result = discover(req, DiscoveryContext(tmp_path))
    assert result.summary["search_space"]["order"] == "balanced"
    assert result.summary["attempted_structures"] == 3
    assert result.summary["coverage"]["h"]["sizes"] == {"1": 2, "3": 1}
    assert result.summary["termination"] == "structure_budget"


def test_balanced_expired_wall_budget_does_not_start_fitting(tmp_path, monkeypatch):
    from ml_platform_symbolic.engines import template_enumeration as engine
    def prohibited(*args, **kwargs):
        raise AssertionError("Expired discovery budget must not start coefficient optimization")
    monkeypatch.setattr(engine, "fit_coefficients", prohibited)
    X = np.ones((4, 1))
    req = request_for(X, np.ones(4))
    req.options = {"max_structures": 256}
    result = discover(req, DiscoveryContext(tmp_path, deadline=0.0))
    assert result.summary["attempted_structures"] == 0
    assert result.summary["termination"] == "wall_budget"


@pytest.mark.parametrize("arguments", [["u", "v"], ["v", "u"]])
@pytest.mark.parametrize("operators", [["log", "add", "mul", "sub"], ["sub", "mul", "add", "log"]])
def test_recursive_balanced_prefix_reaches_later_variable_and_child_operators(arguments, operators):
    from itertools import islice

    slot = {"arguments": [{"name": name} for name in arguments], "allowed_operators": operators}
    prefix = list(islice(enumerate_balanced_shapes(slot, 7), 512))
    # Not a target equation: inspect syntactic reachability of child kinds.
    assert any(t[0] == "add" and any(c[0] == "mul" for c in t[1:]) for t in prefix)
    for name in arguments:
        assert any(t[0] == "sub" and t[1] == ("variable", name) and t[2][0] == "log" for t in prefix)


def test_recursive_product_does_not_exhaust_left_or_right_stream():
    from itertools import count, islice

    from ml_platform_symbolic.engines.template_enumeration import _index_product
    prefix = list(islice(_index_product([count(), count()]), 6))
    assert set(prefix) == {(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)}


def test_commutative_high_size_prefix_does_not_exhaust_reverse_order_split():
    from itertools import islice

    slot = {"arguments": [{"name": "u"}], "allowed_operators": ["add"]}
    prefix = list(islice(enumerate_balanced_shapes(slot, 9), 5))
    def size(t):
        return 1 if t[0] in {"coefficient", "variable", "literal"} else 1 + sum(size(c) for c in t[1:])
    assert [size(t) for t in prefix] == [1, 3, 5, 7, 9]


@pytest.mark.parametrize("condition", ["nodes", "free_parameters", "variables", "operators", "units"])
def test_fit_independent_rejection_reuses_common_rules_before_any_fit(tmp_path, monkeypatch, condition):
    from ml_platform_symbolic.engines import template_enumeration as engine

    X = np.linspace(0.1, 1.0, 6)[:, None]
    req = request_for(X, X[:, 0], options={"max_structures": 1, "starts": [1.0, -1.0]})
    req.equation_template["ast"] = {"op": "add", "args": [
        req.equation_template["ast"], {"var": "x0"},
    ]}
    if condition == "nodes":
        req.limits["max_nodes"] = 1
    elif condition == "free_parameters":
        req.limits["max_free_parameters"] = 0
    elif condition == "variables":
        req.limits["allowed_variables"] = []
    elif condition == "operators":
        req.operators = ["mul"]  # Slot grammar valid; known outer add disallowed.
    else:
        req.target["unit"] = "m"
    calls = []
    def prohibited(*args, **kwargs):
        calls.append(1)
        raise AssertionError("Known structural rejection must not enter coefficient fit")
    monkeypatch.setattr(engine, "fit_coefficients", prohibited)
    result = discover(req, DiscoveryContext(tmp_path))
    assert not calls and not result.candidates
    assert result.summary["attempted_structures"] == 1
    assert result.summary["prefit_rejections"] == 1
    assert result.summary["fit_attempts"] == 0
    assert result.summary["termination"] == "structure_budget"


def test_prefit_checks_do_not_assume_variables_are_mandatory_or_test_response_values(tmp_path):
    X = np.linspace(0.1, 1.0, 8)[:, None]
    req = request_for(X, np.full(8, 2.0), options={"max_structures": 1, "starts": [-1.0]})
    req.limits.update(probes=[{"x0": 0.5}], response_bounds=[1.0, None], response_mode="search")
    result = discover(req, DiscoveryContext(tmp_path))
    assert result.summary["prefit_rejections"] == 0
    assert result.summary["fit_attempts"] == 1
    assert result.candidates
    np.testing.assert_allclose(evaluate(result.candidates[0].equation, X, req.variables), 2.0, atol=1e-6)
