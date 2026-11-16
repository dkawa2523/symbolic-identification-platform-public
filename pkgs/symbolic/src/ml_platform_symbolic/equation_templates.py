"""Compile declared scalar structure; names do not establish causal attribution."""

from copy import deepcopy

from .constraints import dimensions
from .expression_graph import MAX_DEPTH, MAX_EXPANDED_NODES, expanded_equation
from .expressions import equation_text, evaluate, inspect_equation
from .operators import ARITY
from .units import require_canonical

LIMITATION = "Named components describe this declared representation, not unique identification or causal effects."


def _equation(ast, parameters):
    return {"schema_version": 2, "ast": ast, "parameters": deepcopy(parameters)}


def _unit(record):
    if "unit" not in record or not isinstance(record["unit"], (str, type(None))) or record["unit"] == "":
        raise ValueError("Template units must be explicit strings or null (unknown)")
    require_canonical(record["unit"])


def _names(mapping):
    if not isinstance(mapping, dict) or any(not isinstance(n, str) or not n.isidentifier() for n in mapping):
        raise ValueError("Template declaration names must be identifiers")


def _variables(ast):
    found = set()

    def visit(node):
        if "var" in node:
            found.add(node["var"])
        for child in node.get("args", []):
            visit(child)

    visit(ast)
    return [{"id": name, "column": name, "unit": None} for name in sorted(found)]


def _slot_parameter_name(slot, local, occupied):
    name = f"__slot_{slot}_{local}"
    while name in occupied:
        name += "_"
    return name


def _compile(template, slot_equations, parameter_values=None, *, placeholders=False):
    """One expansion path for validation, export and refit integrity checks."""
    parameters = deepcopy(template["parameters"])
    base = {n: p for n, p in parameters.items() if "derived_from" not in p}
    for name, value in (parameter_values or {}).items():
        if name not in base or not base[name]["trainable"]:
            raise ValueError(f"Only independent template parameters may be updated: {name}")
        base[name]["value"] = value
    used = {"components": set(), "parameters": set(), "slots": set()}
    calls, checks, expanded_slots = [], [], {}
    for name, slot in sorted(template["slots"].items()):
        if placeholders:
            key = f"__slot_output_{name}"
            while key in base:
                key += "_"
            base[key] = {"value": 1.0, "trainable": False, "unit": slot["unit"]}
            expanded_slots[name] = {"param": key}
            continue
        raw = slot_equations[name]
        formal = [{"id": arg["name"], "column": arg["name"], "unit": arg["unit"]} for arg in slot["arguments"]]
        if "template" in raw:
            raise ValueError("Nested template metadata in slot equations is unsupported")
        clean = deepcopy(raw)
        _names(clean["parameters"])
        stats = inspect_equation(clean, formal)
        if clean.get("arithmetic", "strict_real") != "strict_real":
            raise ValueError("Template slots require strict_real arithmetic")
        if set(clean["parameters"]) != stats["parameters"]:
            raise ValueError("Slot parameters must all be used")
        status, reasons = dimensions(clean, formal, {"unit": slot["unit"]})
        checks.extend(
            [
                {
                    "name": f"template.slot.{name}.operators",
                    "status": "pass" if stats["operators"] <= set(slot["allowed_operators"]) else "fail",
                    "required": True,
                },
                {
                    "name": f"template.slot.{name}.nodes",
                    "status": "pass" if stats["n_nodes"] <= slot["max_nodes"] else "fail",
                    "required": True,
                },
                {
                    "name": f"template.slot.{name}.units",
                    "status": status,
                    "required": slot["unit"] is not None or status == "fail",
                    "reason": "; ".join(reasons) or None,
                },
            ]
        )
        renames = {}
        for local, record in sorted(clean["parameters"].items()):
            if record.get("tied_to") is not None:
                raise ValueError("Use template derived_from for parameter dependencies, not slot aliases")
            _unit(record)
            key = _slot_parameter_name(name, local, base)
            renames[local] = key
            base[key] = deepcopy(record)

        def rename(node, renames=renames):
            if "param" in node:
                return {"param": renames[node["param"]]}
            return {"op": node["op"], "args": [rename(a) for a in node["args"]]} if "op" in node else deepcopy(node)

        expanded_slots[name] = rename(expanded_equation(clean)["ast"])
    count = 0

    def expand(node, active=(), parameter_only=False):
        nonlocal count
        count += 1
        if count > MAX_EXPANDED_NODES or len(active) > MAX_DEPTH or not isinstance(node, dict):
            raise ValueError("Template exceeds AST resources")
        keys = set(node)
        if keys == {"param"}:
            name = node["param"]
            if name not in parameters:
                raise ValueError(f"Unknown template parameter: {name}")
            used["parameters"].add(name)
            if "derived_from" not in parameters[name]:
                return deepcopy(node)
            key = ("parameter", name)
            if key in active:
                raise ValueError("Cyclic dependent parameter")
            return expand(parameters[name]["derived_from"], (*active, key), True)
        if keys == {"const"} or (keys == {"var"} and not parameter_only):
            return deepcopy(node)
        if keys == {"ref"} and not parameter_only:
            name = node["ref"]
            if name not in template["components"] or ("component", name) in active:
                raise ValueError("Missing or cyclic template component")
            used["components"].add(name)
            return expand(template["components"][name]["ast"], (*active, ("component", name)))
        if keys == {"slot", "args"} and not parameter_only:
            name = node["slot"]
            if name not in template["slots"] or not isinstance(node["args"], list):
                raise ValueError("Unknown template slot")
            slot = template["slots"][name]
            if len(node["args"]) != len(slot["arguments"]):
                raise ValueError("Wrong template slot arity")
            used["slots"].add(name)
            args = [expand(a, active) for a in node["args"]]
            calls.extend(zip(args, slot["arguments"], strict=True))
            bindings = dict(zip([a["name"] for a in slot["arguments"]], args, strict=True))
            used_arguments = {v["id"] for v in _variables(expanded_slots[name])}
            if not placeholders and any(
                formal not in used_arguments and not set(argument) <= {"var", "param", "const"}
                for formal, argument in bindings.items()
            ):
                raise ValueError(
                    "Unused slot argument would discard a composed expression and its domain; unsupported composition"
                )

            def bind(ast):
                if "var" in ast:
                    return deepcopy(bindings[ast["var"]])
                return {"op": ast["op"], "args": [bind(a) for a in ast["args"]]} if "op" in ast else deepcopy(ast)

            return bind(expanded_slots[name])
        if (
            keys == {"op", "args"}
            and node["op"] in ARITY
            and isinstance(node["args"], list)
            and len(node["args"]) == ARITY[node["op"]]
        ):
            return {
                "op": node["op"],
                "args": [expand(a, (*active, ("op", count)), parameter_only) for a in node["args"]],
            }
        raise ValueError("Unsupported template AST node")

    ast = expand(template["ast"])
    if any(set(template[k]) != used[k] for k in used):
        raise ValueError("All template components, parameters and slots must be used")
    components = {name: _equation(expand(c["ast"]), base) for name, c in template["components"].items()}
    dependencies = {
        name: _equation(expand(p["derived_from"], parameter_only=True), base)
        for name, p in parameters.items()
        if "derived_from" in p
    }
    return _equation(ast, base), components, dependencies, calls, checks


def validate_template(template, variables, target, operators, fixed_constants):
    """Validate declarations once in canonical working coordinates; no soft claims."""
    result = deepcopy(template)
    allowed = {"coordinate_space", "ast", "components", "parameters", "slots", "knowledge"}
    if (
        not isinstance(result, dict)
        or set(result) - allowed
        or result.get("coordinate_space") != "working"
        or "ast" not in result
    ):
        raise ValueError("Template requires ast and coordinate_space='working'")
    for key in ("components", "parameters", "slots"):
        result.setdefault(key, {})
        _names(result[key])
    referenced = set()

    def collect(node):
        if not isinstance(node, dict):
            return
        if "param" in node:
            referenced.add(node["param"])
        for child in node.get("args", []):
            collect(child)

    collect(result["ast"])
    for component in result["components"].values():
        collect(component.get("ast"))
    for parameter in result["parameters"].values():
        collect(parameter.get("derived_from"))
    for name in sorted(referenced & set(fixed_constants) - set(result["parameters"])):
        result["parameters"][name] = {
            **deepcopy(fixed_constants[name]),
            "trainable": False,
            "origin": "requirements.fixed_constants",
        }
    names = [name for key in ("components", "parameters", "slots") for name in result[key]]
    if len(names) != len(set(names)) or set(names) & {v["id"] for v in variables}:
        raise ValueError("Template declaration names must be distinct from each other and variables")
    knowledge = result.get("knowledge")
    if (
        not isinstance(knowledge, dict)
        or set(knowledge) - {"kind", "statement", "source"}
        or knowledge.get("kind") not in {"requirement", "hypothesis", "established"}
        or not isinstance(knowledge.get("statement"), str)
        or not knowledge["statement"].strip()
    ):
        raise ValueError("Template requires explicit knowledge kind and statement")
    if "source" in knowledge and not isinstance(knowledge["source"], str):
        raise ValueError("Knowledge source must be a string")
    if knowledge["kind"] == "established" and not knowledge.get("source", "").strip():
        raise ValueError("Established knowledge requires an explicit user-supplied source")
    for component in result["components"].values():
        if not isinstance(component, dict) or set(component) - {"ast", "description"} or "ast" not in component:
            raise ValueError("Component requires ast and optional description")
    for name, p in result["parameters"].items():
        _unit(p)
        if "derived_from" in p:
            if set(p) - {"derived_from", "unit", "description"}:
                raise ValueError("Dependent parameters cannot have independent value, bounds or mutability")
        elif set(p) - {
            "value",
            "trainable",
            "unit",
            "bounds",
            "origin",
            "unit_origin",
            "coordinate_role",
            "description",
        } or not {"value", "trainable", "unit"} <= set(p):
            raise ValueError(
                "Base parameters need explicit value, trainable and unit; use derived_from for dependencies"
            )
        if name in fixed_constants:
            fixed = fixed_constants[name]
            if "derived_from" in p or p["trainable"] or p["value"] != fixed["value"] or p["unit"] != fixed["unit"]:
                raise ValueError("Template cannot change a declared fixed constant")
        if "derived_from" not in p:
            p.setdefault("origin", "user_declared_template")
    for slot in result["slots"].values():
        if (
            not isinstance(slot, dict)
            or set(slot) - {"arguments", "unit", "allowed_operators", "max_nodes", "description"}
            or not {"arguments", "unit", "allowed_operators", "max_nodes"} <= set(slot)
        ):
            raise ValueError("Slot requires arguments, unit, allowed_operators and max_nodes")
        _unit(slot)
        args = slot["arguments"]
        if not isinstance(args, list) or not args:
            raise ValueError("Slot needs at least one formal argument")
        seen = set()
        for arg in args:
            if (
                not isinstance(arg, dict)
                or set(arg) != {"name", "unit"}
                or not isinstance(arg["name"], str)
                or not arg["name"].isidentifier()
                or arg["name"] in seen
            ):
                raise ValueError("Slot argument names must be unique identifiers")
            seen.add(arg["name"])
            _unit(arg)
        if (
            not isinstance(slot["allowed_operators"], list)
            or not set(slot["allowed_operators"]) <= set(operators)
            or not set(slot["allowed_operators"]) <= set(ARITY)
        ):
            raise ValueError("Slot operators must be a subset of overall operators")
        if type(slot["max_nodes"]) is not int or slot["max_nodes"] < 1:
            raise ValueError("Slot max_nodes must be a positive integer")
    equation, components, dependencies, calls, _ = _compile(result, {}, placeholders=True)
    for eq in [
        equation,
        *components.values(),
        *dependencies.values(),
        *[_equation(a, equation["parameters"]) for a, _ in calls],
    ]:
        stats = inspect_equation(eq, variables)
        if not stats["operators"] <= set(operators):
            raise ValueError("Known template structure uses a forbidden operator")
    for eq, wanted in [
        (equation, target),
        *[(e, result["parameters"][n]) for n, e in dependencies.items()],
        *[(_equation(a, equation["parameters"]), formal) for a, formal in calls],
    ]:
        status, reasons = dimensions(eq, variables, wanted)
        if status == "fail":
            raise ValueError("Template unit mismatch: " + "; ".join(reasons))
    return result


def instantiate_template(template, slot_equations, parameter_values=None, *, variables=None, target=None):
    if set(slot_equations) != set(template["slots"]):
        raise ValueError("Supply exactly one equation for every declared slot")
    equation, components, dependencies, calls, checks = _compile(template, slot_equations, parameter_values)
    variables = variables if variables is not None else _variables(equation["ast"])
    inspect_equation(equation, variables)
    for index, (ast, formal) in enumerate(calls):
        status, reasons = dimensions(_equation(ast, equation["parameters"]), variables, formal)
        checks.append(
            {
                "name": f"template.argument.{index}.units",
                "status": status,
                "required": formal["unit"] is not None or status == "fail",
                "reason": "; ".join(reasons) or None,
            }
        )
    if target is not None:
        status, reasons = dimensions(equation, variables, target)
        checks.append(
            {
                "name": "template.output.units",
                "status": status,
                "required": status != "unknown",
                "reason": "; ".join(reasons) or None,
            }
        )
    equation["template"] = {"specification": deepcopy(template), "slots": deepcopy(slot_equations)}
    return equation, {
        "knowledge": deepcopy(template["knowledge"]),
        "components": components,
        "dependencies": dependencies,
        "checks": checks,
        "raw_slot_equations": deepcopy(slot_equations),
        "interpretation_limit": LIMITATION,
    }


def _reconstruct(equation, variables, target=None):
    metadata = equation["template"]
    if not isinstance(metadata, dict) or set(metadata) != {"specification", "slots"}:
        raise ValueError("Invalid template metadata")
    spec, slots = deepcopy(metadata["specification"]), deepcopy(metadata["slots"])
    validate_template(spec, variables, target or {"unit": None}, list(ARITY), {})
    initial, _ = instantiate_template(spec, slots, variables=variables, target=target)
    current = equation["parameters"]
    if set(current) != set(initial["parameters"]):
        raise ValueError("Template parameter identities changed")
    # Keep declarations immutable; only independently trainable values may differ.
    for name, original in initial["parameters"].items():
        observed = current[name]
        if {k: v for k, v in observed.items() if k != "value"} != {
            k: v for k, v in original.items() if k != "value"
        } or (not original["trainable"] and observed["value"] != original["value"]):
            raise ValueError("Template parameter declarations or fixed values changed")
    clean = {k: deepcopy(v) for k, v in equation.items() if k != "template"}
    inspect_equation(clean, variables)
    if expanded_equation(clean)["ast"] != initial["ast"] or clean.get("arithmetic", "strict_real") != "strict_real":
        raise ValueError("Equation no longer matches declared template structure")
    occupied = {n for n, p in spec["parameters"].items() if "derived_from" not in p}
    for name, p in spec["parameters"].items():
        if name in current and p.get("trainable"):
            p["value"] = current[name]["value"]
    for slot_name, slot in sorted(slots.items()):
        for local, p in sorted(slot["parameters"].items()):
            key = _slot_parameter_name(slot_name, local, occupied)
            occupied.add(key)
            p["value"] = current.get(key, p)["value"]
    _, evidence = instantiate_template(spec, slots, variables=variables, target=target)
    for eq in [*evidence["components"].values(), *evidence["dependencies"].values()]:
        eq["parameters"].update(deepcopy(current))
    return evidence


def validate_template_binding(equation, variables):
    """Reject tampered structure or fixed values on load without trusting saved checks."""
    if "template" in equation:
        _reconstruct(equation, variables)


def template_checks(equation, variables, target, required_template=None):
    try:
        if "template" not in equation:
            if required_template is None:
                return []
            raise ValueError("Required equation template is absent")
        if required_template is not None and equation["template"]["specification"] != required_template:
            raise ValueError("Candidate template differs from configured requirement")
        evidence = _reconstruct(equation, variables, target)
        return [{"name": "template.structure", "status": "pass", "required": True}, *evidence["checks"]]
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        return [{"name": "template.structure", "status": "fail", "required": True, "reason": str(exc)}]


def describe_template(equation, variables):
    if "template" not in equation:
        return None
    evidence = _reconstruct(equation, variables)
    import numpy as np

    values = {n: float(evaluate(eq, np.empty((1, 0)), [])[0]) for n, eq in evidence["dependencies"].items()}
    evidence["dependent_values"] = {n: value if np.isfinite(value) else None for n, value in values.items()}
    evidence["component_formulas"] = {
        name: equation_text(eq, variables, parameter_names=True) for name, eq in evidence["components"].items()
    }
    evidence["dependent_units"] = {
        name: equation["template"]["specification"]["parameters"][name]["unit"] for name in evidence["dependencies"]
    }
    evidence["slot_summaries"] = {
        name: {
            "formula": equation_text(
                eq,
                [
                    {"id": a["name"], "column": a["name"]}
                    for a in equation["template"]["specification"]["slots"][name]["arguments"]
                ],
                parameter_names=True,
            ),
            "equation": eq,
            **deepcopy(equation["template"]["specification"]["slots"][name]),
        }
        for name, eq in evidence["raw_slot_equations"].items()
    }
    return evidence
