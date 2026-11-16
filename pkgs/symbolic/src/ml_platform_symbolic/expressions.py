"""Small, explicit real-valued AST. No eval, pickle or automatic simplification."""

from __future__ import annotations

import ast
from copy import deepcopy

import numpy as np

from .expression_graph import (
    MAX_DEPTH,
    MAX_EXPANDED_NODES,
    MAX_STORED_NODES,
    expanded_equation,
)
from .operators import ARITY, UNARY_FUNCTIONS
from .parameters import free_parameter_names, with_parameters


def _check_literal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("AST literals must be finite real numbers")


def _check_node_limit(node, count, depth):
    if count > MAX_EXPANDED_NODES or depth > MAX_DEPTH or not isinstance(node, dict):
        raise ValueError("AST exceeds execution node/depth resources or has invalid node")


def _check_parameters(parameters):
    for name, parameter in parameters.items():
        value = parameter["value"]
        bounds = parameter.get("bounds", [None, None])
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or not isinstance(parameter.get("trainable"), bool)
            or len(bounds) != 2
        ):
            raise ValueError(f"Invalid parameter: {name}")
        low, high = bounds
        if any(b is not None and not np.isfinite(b) for b in bounds):
            raise ValueError(f"Nonfinite bounds: {name}")
        if (low is not None and value < low) or (high is not None and value > high):
            raise ValueError(f"Parameter outside bounds: {name}")
    resolved = with_parameters({"parameters": parameters}, [], [])
    if any(p["value"] != resolved["parameters"][name]["value"] for name, p in parameters.items()):
        raise ValueError("Stored tied parameter values do not match their source")


def _check_operator(node):
    op, args = node["op"], node["args"]
    if op not in ARITY or not isinstance(args, list) or len(args) != ARITY[op]:
        raise ValueError(f"Unknown operator or wrong arity: {op}")
    if op == "pow_int":
        exponent = args[1].get("const")
        if (
            isinstance(exponent, bool)
            or not isinstance(exponent, (int, float))
            or int(exponent) != exponent
            or abs(exponent) > 8
        ):
            raise ValueError("pow_int requires fixed integer literal in [-8,8]")
    if op == "rootn":
        index = args[1].get("const")
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, float))
            or not np.isfinite(index)
            or int(index) != index
            or index == 0
        ):
            raise ValueError("rootn requires a fixed nonzero integer index")


def inspect_equation(equation: dict, variables: list[dict]) -> dict:
    version = equation.get("schema_version")
    mandatory = {"schema_version", "ast", "parameters"}
    if (
        version not in {1, 2}
        or not mandatory <= set(equation)
        or set(equation) - mandatory - ({"definitions", "arithmetic", "template"} if version == 2 else set())
    ):
        raise ValueError("Expected data-only equation schema1 or schema2 with local definitions/arithmetic")
    if equation.get("arithmetic", "strict_real") not in {"strict_real", "numpy_ieee"}:
        raise ValueError("Unknown numerical arithmetic semantics")

    def stored(node, depth=1):
        if not isinstance(node, dict) or depth > MAX_DEPTH:
            raise ValueError("Invalid AST node")
        count = 1 + sum(stored(child, depth + 1) for child in node.get("args", []))
        if "poly" in node:
            if version != 2:
                raise ValueError("Sparse polynomial blocks require equation schema2")
            count += sum(stored(term["coefficient"], depth + 1) for term in node["poly"])
        return count

    n_stored = stored(equation["ast"]) + sum(stored(node) for node in equation.get("definitions", {}).values())
    if n_stored > MAX_STORED_NODES:
        raise ValueError("AST exceeds stored-node execution resource")
    tree = expanded_equation(equation)
    parameters = equation["parameters"]
    _check_parameters(parameters)
    ids = {v["id"] for v in variables}
    stats = {"n_nodes": 0, "parameters": set(), "variables": set(), "operators": set()}

    def visit(node, depth):
        stats["n_nodes"] += 1
        _check_node_limit(node, stats["n_nodes"], depth)
        keys = set(node)
        if keys == {"var"}:
            if node["var"] not in ids:
                raise ValueError(f"Unknown variable: {node['var']}")
            stats["variables"].add(node["var"])
        elif keys == {"const"}:
            _check_literal(node["const"])
        elif keys == {"param"}:
            if node["param"] not in parameters:
                raise ValueError(f"Unknown parameter: {node['param']}")
            stats["parameters"].add(node["param"])
        elif keys == {"op", "args"}:
            _check_operator(node)
            op, args = node["op"], node["args"]
            stats["operators"].add(op)
            for arg in args:
                visit(arg, depth + 1)
        else:
            raise ValueError(f"Unknown AST node/reference: {node}")

    visit(tree["ast"], 1)
    stats["n_parameters"] = len(free_parameter_names(equation))
    stats["n_stored_nodes"] = n_stored
    stats["arithmetic"] = equation.get("arithmetic", "strict_real")
    return stats


def _execute(equation: dict, X: np.ndarray, variables: list[dict]):
    stats = inspect_equation(equation, variables)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(variables):
        raise ValueError("X must have shape (n_rows, n_variables)")
    batch_rows = max(1, 8_000_000 // max(1, stats["n_stored_nodes"]))
    if len(X) > batch_rows:
        chunks = [
            _execute(equation, X[start : start + batch_rows], variables) for start in range(0, len(X), batch_rows)
        ]
        return np.concatenate([x[0] for x in chunks]), np.concatenate([x[1] for x in chunks])
    columns = {v["id"]: X[:, i] for i, v in enumerate(variables)}
    cache = {}
    ieee = equation.get("arithmetic", "strict_real") == "numpy_ieee"

    def visit(node):
        if "ref" in node:
            key = node["ref"]
            if key not in cache:
                cache[key] = visit(equation["definitions"][key])
            return cache[key]
        if "var" in node:
            value = columns[node["var"]]
            return value, np.isfinite(value)
        if "const" in node:
            return float(node["const"]), True
        if "param" in node:
            return equation["parameters"][node["param"]]["value"], True
        if "poly" in node:
            features = [visit(child) for child in node["args"]]
            values, validity = [], []
            for term in node["poly"]:
                value, valid = visit(term["coefficient"])
                for index, power in term["powers"]:
                    feature, feature_valid = features[index]
                    for _ in range(power):
                        value = np.multiply(value, feature)
                    valid = valid & feature_valid & np.isfinite(feature) & np.isfinite(value)
                values.append(value)
                validity.append(valid)
            result = values[0]
            valid = validity[0]
            for value, value_valid in zip(values[1:], validity[1:], strict=True):
                result = np.add(result, value)
                valid = valid & value_valid & np.isfinite(result)
            return result if ieee else np.where(valid, result, np.nan), valid
        children = [visit(a) for a in node["args"]]
        values = [item[0] for item in children]
        op = node["op"]
        if op == "div":
            result = np.divide(*values) if ieee else np.where(np.asarray(values[1]) == 0, np.nan, np.divide(*values))
        elif op == "rootn":
            base, index = values
            result = np.sign(base) * np.power(np.abs(base), 1 / index) if int(index) % 2 else np.power(base, 1 / index)
        else:
            result = {
                "add": np.add,
                "sub": np.subtract,
                "mul": np.multiply,
                "pow_int": np.power,
                "pow": np.power,
                **UNARY_FUNCTIONS,
            }[op](*values)
        # Invalid subexpressions remain invalid even when IEEE arithmetic would
        # turn them back into a finite value (NaN**0, exp(-inf), reciprocal inf).
        valid = np.isfinite(result)
        for value, child_valid in children:
            valid = valid & child_valid & np.isfinite(value)
        return result if ieee else np.where(valid, result, np.nan), valid

    with np.errstate(all="ignore"):
        result, valid = visit(equation["ast"])
        result = np.broadcast_to(result, (len(X),)).astype(np.float64, copy=True)
        valid = np.broadcast_to(valid, (len(X),)).copy()
    return result, valid


def evaluate_native(equation, X, variables):
    """Declared raw arithmetic for native parity; not scientific row admissibility."""
    return _execute(equation, X, variables)[0]


def evaluate(equation, X, variables):
    """Common real-domain prediction: retain invalid intermediates as invalid rows."""
    result, valid = _execute(equation, X, variables)
    return np.where(valid, result, np.nan)


def equation_text(equation: dict, variables: list[dict], *, parameter_names=False) -> str:
    equation = expanded_equation(equation)
    names = {v["id"]: v["column"] for v in variables}

    def visit(node):
        if "var" in node:
            return names[node["var"]]
        if "const" in node:
            return repr(node["const"])
        if "param" in node:
            return node["param"] if parameter_names else repr(equation["parameters"][node["param"]]["value"])
        args = [visit(a) for a in node["args"]]
        op = node["op"]
        if op in {"pow", "pow_int"}:
            return f"(({args[0]}) ** {args[1]})"
        signs = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow_int": "**", "pow": "**"}
        return f"({args[0]} {signs[op]} {args[1]})" if op in signs else f"{op}({', '.join(args)})"

    return visit(equation["ast"])


def parse_expression(
    text: str, variables: list[dict], *, trainable: bool = True, parameters: dict | None = None
) -> dict:
    """Translate engine arithmetic only; retain literal exponents and domain holes."""
    parameters = deepcopy(parameters or {})
    allowed = {v["id"] for v in variables}
    binary = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow_int"}

    def visit(node, fixed=False):
        if isinstance(node, ast.Name) and node.id in allowed:
            return {"var": node.id}
        if isinstance(node, ast.Name) and node.id in parameters:
            return {"param": node.id}
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            _check_literal(node.value)
            if fixed or not trainable:
                return {"const": node.value}
            key = f"c{len(parameters)}"
            while key in parameters:
                key += "_"
            parameters[key] = {"value": float(node.value), "trainable": True, "bounds": [None, None], "unit": None}
            return {"param": key}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return visit(ast.Constant(value=-node.operand.value), fixed)
            return {"op": "neg", "args": [visit(node.operand, fixed)]}
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            op = binary[type(node.op)]
            left, right = visit(node.left, fixed), visit(node.right, fixed or op == "pow_int")
            if op == "pow_int" and not (
                "const" in right and int(right["const"]) == right["const"] and abs(right["const"]) <= 8
            ):
                op = "pow"
            return {"op": op, "args": [left, right]}
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in UNARY_FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            return {"op": node.func.id, "args": [visit(node.args[0], fixed)]}
        raise ValueError(f"Unsupported engine expression: {text}")

    equation = {
        "schema_version": 1,
        "ast": visit(ast.parse(text.replace("^", "**"), mode="eval").body),
        "parameters": parameters,
    }
    inspect_equation(equation, variables)
    return deepcopy(equation)


def canonical_integer_powers(equation):
    """Normalize only structurally fixed literal powers, never fitted exponents.

    Native libraries can canonicalize mul/div to POWER even when general powers
    were not requested. Keep the base and arithmetic intact, including holes.
    Historical artifacts are not rewritten by loaders.
    """
    result = deepcopy(equation)

    def visit(node):
        for child in node.get("args", []):
            visit(child)
        if node.get("op") == "pow":
            exponent = node["args"][1]
            if set(exponent) == {"const"}:
                value = exponent["const"]
                if type(value) in {int, float} and np.isfinite(value) and value == int(value) and abs(value) <= 8:
                    node["op"] = "pow_int"
                    node["args"][1] = {"const": int(value)}

    visit(result["ast"])
    for node in result.get("definitions", {}).values():
        visit(node)
    return result


def lower_integer_powers(equation):
    """Export native canonical integer powers using repeated real arithmetic.

    SymPy-based native engines turn multiplication into powers even when their
    requested grammar only contains mul/div. No algebraic cancellation or
    protected operation is introduced, and parameter identities stay unchanged.
    """
    result = expanded_equation(equation)

    def visit(node):
        if "op" not in node:
            return node
        if node["op"] == "pow_int":
            _check_operator(node)
            base, exponent = visit(node["args"][0]), int(node["args"][1]["const"])
            if exponent == 0:
                return {"op": "pow_int", "args": [base, {"const": 0}]}
            power = {"const": 1} if exponent == 0 else deepcopy(base)
            for _ in range(1, abs(exponent)):
                power = {"op": "mul", "args": [power, deepcopy(base)]}
            return {"op": "div", "args": [{"const": 1}, power]} if exponent < 0 else power
        return {"op": node["op"], "args": [visit(arg) for arg in node["args"]]}

    result["ast"] = visit(result["ast"])
    return result
