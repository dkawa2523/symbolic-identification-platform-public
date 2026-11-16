"""Allowlisted conversion of trusted native SymPy trees; no string evaluation."""

from functools import reduce

from ..expressions import UNARY_FUNCTIONS, canonical_integer_powers, inspect_equation, lower_integer_powers


def export_symbolic(expression, values, variables):
    """Trusted native symbolic tree, preserving fitted symbol identities and fixed powers."""
    import sympy as sp

    ids = {v["id"] for v in variables}
    parameters = {}

    def visit(node):
        if isinstance(node, sp.Symbol):
            key = str(node)
            if key in ids:
                return {"var": key}
            if key in values:
                parameters[key] = {
                    "value": values[key],
                    "trainable": True,
                    "bounds": [None, None],
                    "unit": None,
                    "origin": "native_named_fit_parameter",
                }
                return {"param": key}
            raise ValueError(f"Unknown native symbolic atom: {key}")
        if node.is_Atom and node.is_number:
            return {"const": float(node)}
        if isinstance(node, (sp.Add, sp.Mul)):
            op = "add" if isinstance(node, sp.Add) else "mul"
            return reduce(lambda left, right: {"op": op, "args": [left, right]}, map(visit, node.args))
        if isinstance(node, sp.Pow):
            base, exponent = node.args
            if exponent == sp.Rational(1, 2):
                return {"op": "sqrt", "args": [visit(base)]}
            return {"op": "pow", "args": [visit(base), visit(exponent)]}
        unary = {getattr(sp, name): name for name in UNARY_FUNCTIONS if name not in {"neg", "abs", "sqrt"}}
        unary[sp.Abs] = "abs"
        if node.func in unary:
            return {"op": unary[node.func], "args": [visit(node.args[0])]}
        raise ValueError(f"Unsupported native symbolic operator: {node.func}")

    equation = {"schema_version": 1, "ast": visit(expression), "parameters": parameters}
    equation = canonical_integer_powers(equation)
    inspect_equation(equation, variables)
    return lower_integer_powers(equation)
