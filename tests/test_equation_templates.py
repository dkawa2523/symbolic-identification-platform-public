from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.equation_templates import (
    describe_template,
    instantiate_template,
    template_checks,
    validate_template,
    validate_template_binding,
)
from ml_platform_symbolic.expressions import evaluate
from ml_platform_symbolic.parameters import free_parameter_names, with_parameters

VARS = [{"id": "x", "column": "x", "unit": "1"}, {"id": "z", "column": "z", "unit": "1"}]
OPS = ["add", "sub", "mul", "div", "log"]


def op(name, *args):
    return {"op": name, "args": list(args)}


def fixture():
    template = {
        "coordinate_space": "working",
        "ast": op(
            "add",
            {"ref": "known"},
            op("add", {"slot": "h", "args": [{"var": "x"}]}, {"slot": "h", "args": [{"var": "z"}]}),
        ),
        "components": {"known": {"ast": op("mul", {"param": "b"}, {"var": "x"}), "description": "declared baseline"}},
        "parameters": {
            "a": {"value": 2.0, "trainable": True, "unit": "1"},
            "total": {"value": 5.0, "trainable": False, "unit": "1"},
            "b": {"derived_from": op("sub", {"param": "total"}, {"param": "a"}), "unit": "1"},
        },
        "slots": {
            "h": {"arguments": [{"name": "u", "unit": "1"}], "unit": "1", "allowed_operators": ["mul"], "max_nodes": 3}
        },
        "knowledge": {"kind": "hypothesis", "statement": "Known structure plus shared unknown response"},
    }
    slot = {
        "schema_version": 2,
        "ast": op("mul", {"param": "c0"}, {"var": "u"}),
        "parameters": {"c0": {"value": 4.0, "trainable": True, "unit": "1"}},
    }
    return validate_template(template, VARS, {"unit": "1"}, OPS, {}), {"h": slot}


def test_shared_slots_dependencies_and_refit_survive_export():
    template, slots = fixture()
    equation, evidence = instantiate_template(template, slots, variables=VARS, target={"unit": "1"})
    assert set(free_parameter_names(equation)) == {"a", "__slot_h_c0"}
    assert "b" not in equation["parameters"]
    np.testing.assert_allclose(evaluate(equation, np.array([[2.0, 3.0]]), VARS), [26.0])
    refit = with_parameters(equation, ["a", "__slot_h_c0"], [1.0, 2.0])
    assert all(c["status"] == "pass" for c in template_checks(refit, VARS, {"unit": "1"}, template))
    description = describe_template(refit, VARS)
    assert description["dependent_values"]["b"] == 4.0
    np.testing.assert_allclose(evaluate(description["components"]["known"], np.array([[2.0, 3.0]]), VARS), [8.0])
    assert slots["h"]["parameters"]["c0"]["value"] == 4.0
    assert "unique" in evidence["interpretation_limit"]


@pytest.mark.parametrize("change", ["cycle", "arity", "unused", "noncanonical", "soft", "forbidden_argument_operator"])
def test_invalid_declarations_are_rejected(change):
    template, _ = fixture()
    if change == "cycle":
        template["parameters"]["b"]["derived_from"] = {"param": "b"}
    elif change == "arity":
        template["ast"]["args"][1]["args"][0]["args"] = []
    elif change == "unused":
        template["components"]["unused"] = {"ast": {"var": "x"}}
    elif change == "noncanonical":
        template["slots"]["h"]["unit"] = "cm"
    elif change == "soft":
        template["soft"] = True
    else:
        template["ast"]["args"][1]["args"][0]["args"] = [op("sin", {"var": "x"})]
    with pytest.raises(ValueError):
        validate_template(template, VARS, {"unit": "1"}, OPS, {})


def test_tampering_and_slot_requirements_fail_closed():
    template, slots = fixture()
    equation, _ = instantiate_template(template, slots, variables=VARS)
    for tamper in ("structure", "fixed", "metadata"):
        changed = deepcopy(equation)
        if tamper == "structure":
            changed["ast"] = {"const": 0.0}
        elif tamper == "fixed":
            changed["parameters"]["total"]["value"] = 6.0
        else:
            changed["parameters"]["a"]["trainable"] = False
        assert template_checks(changed, VARS, {"unit": "1"}, template)[0]["status"] == "fail"
        with pytest.raises(ValueError):
            validate_template_binding(changed, VARS)
    slots["h"]["ast"] = op("add", slots["h"]["ast"], {"const": 1.0})
    invalid, evidence = instantiate_template(template, slots, variables=VARS)
    assert sum(c["status"] == "fail" for c in evidence["checks"]) == 2
    assert any(c["status"] == "fail" for c in template_checks(invalid, VARS, {"unit": "1"}, template))


def test_unknown_units_are_not_dimensionless_and_invalid_rows_survive():
    template = {
        "coordinate_space": "working",
        "ast": op("log", {"slot": "h", "args": [{"var": "x"}]}),
        "components": {},
        "parameters": {},
        "slots": {
            "h": {"arguments": [{"name": "u", "unit": None}], "unit": None, "allowed_operators": [], "max_nodes": 1}
        },
        "knowledge": {"kind": "requirement", "statement": "Log of unknown function"},
    }
    unknown = [{"id": "x", "column": "x", "unit": None}]
    template = validate_template(template, unknown, {"unit": None}, OPS, {})
    equation, evidence = instantiate_template(
        template, {"h": {"schema_version": 2, "ast": {"var": "u"}, "parameters": {}}}, variables=unknown
    )
    assert any(c["status"] == "unknown" for c in evidence["checks"])
    values = evaluate(equation, np.array([[-1.0], [1.0]]), unknown)
    assert np.isnan(values[0]) and values[1] == 0.0


def test_global_constants_and_known_slot_units_are_explicit_requirements():
    template, slots = fixture()
    del template["parameters"]["total"]
    template = validate_template(
        template,
        VARS,
        {"unit": "1"},
        OPS,
        {"total": {"value": 5.0, "unit": "1"}, "unused": {"value": 7.0, "unit": "1"}},
    )
    assert template["parameters"]["total"]["trainable"] is False
    assert "unused" not in template["parameters"]
    slots["h"]["parameters"]["c0"]["unit"] = None
    equation, evidence = instantiate_template(template, slots, variables=VARS)
    check = next(c for c in evidence["checks"] if c["name"] == "template.slot.h.units")
    assert check["status"] == "unknown" and check["required"]
    validate_template_binding(equation, VARS)  # Preserve inadmissible candidates as evidence.
    template["knowledge"] = {"kind": "established", "statement": "Claim without source"}
    with pytest.raises(ValueError):
        validate_template(template, VARS, {"unit": "1"}, OPS, {})


def test_unused_composed_argument_cannot_drop_domain_and_local_names_do_not_share():
    template, slots = fixture()
    slots["h"]["parameters"] = {"a": slots["h"]["parameters"]["c0"]}
    slots["h"]["ast"]["args"][0] = {"param": "a"}
    equation, _ = instantiate_template(template, slots, variables=VARS)
    assert equation["parameters"]["a"]["value"] == 2.0
    assert equation["parameters"]["__slot_h_a"]["value"] == 4.0
    assert template_checks(equation, VARS, {"unit": "1"}, template)[0]["status"] == "pass"
    template["ast"]["args"][1]["args"][0]["args"] = [op("log", {"var": "x"})]
    slots["h"] = {"schema_version": 2, "ast": {"const": 2.0}, "parameters": {}}
    with pytest.raises(ValueError, match="discard"):
        instantiate_template(template, slots, variables=VARS)
    template["ast"]["args"][1]["args"][0]["args"] = [{"param": "a"}]
    equation, _ = instantiate_template(template, slots, variables=VARS)
    validate_template_binding(equation, VARS)
    assert equation["parameters"]["a"]["value"] == 2.0
