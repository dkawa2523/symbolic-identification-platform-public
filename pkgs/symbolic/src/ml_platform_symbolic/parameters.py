"""Parameter identity and explicit mutability. No inferred dimensions or literal fitting."""

from copy import deepcopy
from fractions import Fraction

import numpy as np

from .expression_graph import expanded_equation
from .units import quantity_role, same_coordinate, unit_spec


def parameter_coordinate_role(parameters, name, active=frozenset()):
    if name in active or name not in parameters:
        raise ValueError("Cyclic coordinate-role alias")
    parameter = parameters[name]
    own = quantity_role(parameter)
    if parameter.get("tied_to"):
        source = parameter_coordinate_role(parameters, parameter["tied_to"], active | {name})
        if own is not None and source is not None and own != source:
            raise ValueError("Tied point/difference parameters denote different physical quantities")
        return own if own is not None else source
    return own


def expression_coordinate_role(equation, variables):
    """Only elementary affine identities are inferred; other meanings stay unknown."""
    equation = expanded_equation(equation)
    roles = {v["id"]: quantity_role(v) for v in variables}

    def visit(node):
        if "var" in node:
            return roles[node["var"]]
        if "param" in node:
            return parameter_coordinate_role(equation["parameters"], node["param"])
        if "const" in node:
            return None
        args = [visit(a) for a in node["args"]]
        if node["op"] == "sub":
            if args == ["value", "value"]:
                return "difference"
            if args == ["value", "difference"]:
                return "value"
            if args == ["difference", "difference"]:
                return "difference"
        if node["op"] == "add":
            if args in (["value", "difference"], ["difference", "value"]):
                return "value"
            if args == ["difference", "difference"]:
                return "difference"
        if node["op"] in {"neg", "abs"} and args[0] == "difference":
            return "difference"
        if node["op"] in {"mul", "div"}:
            scalar = fixed_scalar(node["args"][1], equation["parameters"])
            literal = "const" in node["args"][1]
            parameter = equation["parameters"].get(node["args"][1].get("param"), {})
            if scalar is not None and (literal or parameter.get("unit") == "1") and args[0] == "difference":
                return "difference"
        return None

    return visit(equation["ast"])


def free_parameter_names(equation, *, require_origin=False):
    if "ast" in equation:
        equation = expanded_equation(equation)
    used = set()

    def visit(node):
        if "param" in node:
            used.add(node["param"])
        for child in node.get("args", []):
            visit(child)

    if "ast" in equation:
        visit(equation["ast"])
    else:
        used.update(equation["parameters"])
    for _ in range(len(equation["parameters"])):
        used.update(
            equation["parameters"][name]["tied_to"]
            for name in list(used)
            if name in equation["parameters"] and equation["parameters"][name].get("tied_to")
        )
    names = []
    for name, parameter in equation["parameters"].items():
        if name not in used:
            continue
        if parameter.get("tied_to") is not None:
            if parameter["trainable"]:
                raise ValueError(f"Tied parameter is not an independent free parameter: {name}")
        elif parameter["trainable"]:
            if require_origin and not parameter.get("origin"):
                raise ValueError(f"Parameter mutability provenance is unverified: {name}")
            names.append(name)
    return names


def with_parameters(equation, names, values):
    """Copy the source; change only named free coordinates and restore explicit aliases."""
    result = deepcopy(equation)
    allowed = set(free_parameter_names(result))
    if not set(names) <= allowed or len(set(names)) != len(names):
        raise ValueError("Only independent free parameters may be changed")
    for name, value in zip(names, values, strict=True):
        result["parameters"][name]["value"] = float(value)
    parameters = result["parameters"]

    def resolve(name, active):
        if name in active or name not in parameters:
            raise ValueError("Missing/cyclic tied parameter")
        parameter = parameters[name]
        parameter_coordinate_role(parameters, name)
        if parameter.get("tied_to") is not None:
            source = parameters.get(parameter["tied_to"])
            if source is None:
                raise ValueError("Missing tied parameter")
            a, b = unit_spec(parameter.get("unit")), unit_spec(source.get("unit"))
            if (a is None or b is None) and parameter.get("unit") != source.get("unit"):
                raise ValueError(f"Tied parameters need consistent unknown/known unit declarations: {name}")
            if a is not None and b is not None and not same_coordinate(a, b):
                raise ValueError(f"Tied parameters require identical physical/unit coordinates: {name}")
            value = resolve(parameter["tied_to"], active | {name})
            parameter["value"] = value
        value = parameter["value"]
        low, high = parameter.get("bounds", [None, None])
        if not np.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
            raise ValueError(f"Parameter outside declared bounds: {name}")
        return value

    for name in parameters:
        resolve(name, set())
    return result


def fixed_scalar(node, parameters, active=frozenset()):
    """Exact scalar for dimension algebra only; never freeze a fitted exponent."""
    if "const" in node:
        return Fraction(str(node["const"]))
    if "param" in node:
        name = node["param"]
        p = parameters[name]
        if name in active:
            return None
        if p.get("tied_to"):
            return fixed_scalar({"param": p["tied_to"]}, parameters, active | {name})
        return None if p["trainable"] else Fraction(str(p["value"]))
    args = [fixed_scalar(a, parameters, active) for a in node.get("args", [])]
    if any(a is None for a in args):
        return None
    try:
        op = node.get("op")
        if op == "neg":
            return -args[0]
        if op == "add":
            return args[0] + args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "mul":
            return args[0] * args[1]
        if op == "div":
            return args[0] / args[1]
    except ZeroDivisionError:
        return None
    return None


def free_parameter_bounds(equation, names):
    """Intersect each independent coordinate with all simple tied-alias bounds."""
    bounds = {name: [None, None] for name in names}
    parameters = equation["parameters"]
    for name, parameter in parameters.items():
        root, seen = name, set()
        while parameters[root].get("tied_to"):
            if root in seen:
                raise ValueError("Cyclic parameter alias")
            seen.add(root)
            root = parameters[root]["tied_to"]
        if root in bounds:
            low, high = parameter.get("bounds", [None, None])
            old_low, old_high = bounds[root]
            bounds[root] = [
                max(v for v in (old_low, low) if v is not None) if old_low is not None or low is not None else None,
                min(v for v in (old_high, high) if v is not None) if old_high is not None or high is not None else None,
            ]
    for low, high in bounds.values():
        if low is not None and high is not None and low >= high:
            raise ValueError("Independent free parameter has empty/fixed bounds after alias intersection")
    return [bounds[name] for name in names]


def parameter_table(equation):
    return [
        {
            "name": name,
            "value": parameter["value"],
            "unit": parameter.get("unit"),
            "unit_origin": parameter.get("unit_origin", "unit provenance not separately recorded"),
            "role": "tied" if parameter.get("tied_to") else "free" if parameter["trainable"] else "fixed",
            "tied_to": parameter.get("tied_to"),
            "bounds": parameter.get("bounds", [None, None]),
            "origin": parameter.get("origin", "unverified historical provenance"),
            "coordinate_role": parameter_coordinate_role(equation["parameters"], name),
        }
        for name, parameter in equation["parameters"].items()
    ]
