"""Pinned PySR template lowering and lossless native slot extraction.

Only generated identifiers/operator syntax reach Julia. Outer structure is fixed
in native search; slot identity and native constants survive portable composition.
"""

import math

from ..expressions import inspect_equation

OUTER_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^", "pow_int": "^"}
OUTER_UNARY = {"sqrt", "log", "exp", "abs", "sin", "cos", "tanh", "sinh", "cosh"}


def template_specification(template, variables, operators, *, parameter_initialization=None):
    """Lower validated data to a native spec; fail before search on unsupported intent."""
    slots = template["slots"]
    if not slots:
        raise NotImplementedError("PySR templates require unknown slots; use known_form for complete formulas")
    slot_names = {name: f"slot{i}" for i, name in enumerate(slots)}
    variable_names = {v["id"]: f"input{i}" for i, v in enumerate(variables)}
    parameters = template.get("parameters", {})
    free = {
        name: f"parameter{i}"
        for i, (name, p) in enumerate(parameters.items())
        if p.get("trainable", False) and "derived_from" not in p and not p.get("tied_to")
    }
    if free and parameter_initialization != "native":
        raise NotImplementedError(
            "PySR cannot honor template parameter initial values; explicitly set "
            "template_parameter_initialization='native' to authorize native initialization"
        )
    for name in free:
        if any(bound is not None for bound in parameters[name].get("bounds", [None, None])):
            raise NotImplementedError(f"PySR template does not support bounded outer free parameter: {name}")
    for name, slot in slots.items():
        if set(slot.get("allowed_operators", operators)) != set(operators):
            raise NotImplementedError(f"PySR requires every slot's allowed_operators to equal search operators: {name}")

    def render(node, active=frozenset()):
        if "const" in node:
            value = float(node["const"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite template literal")
            return repr(value)
        if "var" in node:
            return variable_names[node["var"]]
        if "ref" in node:
            key = ("ref", node["ref"])
            if key in active:
                raise ValueError("Cyclic template component")
            return render(template["components"][node["ref"]]["ast"], active | {key})
        if "param" in node:
            name = node["param"]
            key = ("param", name)
            if key in active:
                raise ValueError("Cyclic template parameter")
            parameter = parameters[name]
            if "derived_from" in parameter:
                return render(parameter["derived_from"], active | {key})
            if parameter.get("tied_to"):
                source = render({"param": parameter["tied_to"]}, active | {key})
                scale, offset = parameter.get("scale", 1.0), parameter.get("offset", 0.0)
                return f"(({float(scale)!r} * {source}) + {float(offset)!r})"
            return f"{free[name]}[1]" if name in free else repr(float(parameter["value"]))
        args = [render(arg, active) for arg in node["args"]]
        if "slot" in node:
            return f"{slot_names[node['slot']]}({', '.join(args)})"
        op = node["op"]
        if op in OUTER_BINARY:
            return f"({args[0]} {OUTER_BINARY[op]} {args[1]})"
        if op == "neg":
            return f"(-{args[0]})"
        if op in OUTER_UNARY:
            return f"{op}({args[0]})"
        raise NotImplementedError(f"PySR template outer operator is unsupported: {op}")

    return {
        "combine": render(template["ast"]),
        "expressions": list(slot_names.values()),
        "variable_names": list(variable_names.values()),
        "parameters": {name: 1 for name in free.values()} or None,
    }, {"slots": slot_names, "parameters": free}


def export_slot_tree(tree, arguments, operator_names, *, output_unit=None):
    """Read actual Node values, never rounded equations or algebraic simplifications."""
    parameters = {}

    def visit(node, depth=0):
        if depth > 128:
            raise ValueError("Native slot tree exceeds export depth")
        degree = int(node.degree)
        if degree == 0:
            if bool(node.constant):
                name = f"c{len(parameters)}"
                value = float(node.val)
                if not math.isfinite(value):
                    raise ValueError("Native slot contains nonfinite constant")
                parameters[name] = {
                    "value": value,
                    "trainable": True,
                    "bounds": [None, None],
                    "unit": None,
                    "origin": "pysr_native_template_constant_node",
                }
                if output_unit == "1" and all(arg.get("unit") == "1" for arg in arguments):
                    parameters[name]["unit"] = "1"
                    parameters[name]["unit_origin"] = "declared_dimensionless_template_slot_convention"
                return {"param": name}
            feature = int(node.feature) - 1
            if not 0 <= feature < len(arguments):
                raise ValueError("Native slot feature exceeds formal arity")
            return {"var": arguments[feature]["name"]}
        if degree not in {1, 2}:
            raise ValueError("Unsupported native slot node arity")
        index = int(node.op) - 1
        names = operator_names[degree]
        if not 0 <= index < len(names):
            raise ValueError("Native slot operator index is out of range")
        name = names[index]
        children = [visit(node.l, depth + 1)]
        if degree == 2:
            children.append(visit(node.r, depth + 1))
        if name in {"square", "cube"}:
            return {"op": "pow_int", "args": [children[0], {"const": 2 if name == "square" else 3}]}
        inverse = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "pow"}
        canonical = inverse.get(name, name)
        return {"op": canonical, "args": children}

    equation = {"schema_version": 1, "ast": visit(tree), "parameters": parameters}
    inspect_equation(equation, [{"id": arg["name"]} for arg in arguments])
    return equation


def export_template(expression, template, mapping, operator_names, *, variables=None, target=None):
    """Compose extracted slot bodies once per identity, preserving repeated-call ties."""
    from pysr.julia_import import jl

    from ..equation_templates import instantiate_template

    slots = {}
    for name, native_name in mapping["slots"].items():
        native_slot = getattr(expression.trees, native_name)
        tree = jl.SymbolicRegression.get_tree(native_slot)
        slots[name] = export_slot_tree(
            tree,
            template["slots"][name]["arguments"],
            operator_names,
            output_unit=template["slots"][name]["unit"],
        )
    values = {
        name: float(list(getattr(expression.metadata.parameters, native_name))[0])
        for name, native_name in mapping["parameters"].items()
    }
    return instantiate_template(template, slots, values, variables=variables, target=target)
