"""Data-only local subexpression references; no algebraic rewrite or fitted values."""

import json
from copy import deepcopy

# Parsing/execution safeguards, NOT the user's scientific complexity constraint.
MAX_STORED_NODES = 4096
MAX_EXPANDED_NODES = 65536
MAX_DEPTH = 128


def polynomial_node(features, terms):
    """Sparse monomials over explicit feature expressions; coefficients retain AST identity."""
    used = sorted({index for term in terms for index, _ in term["powers"]})
    remap = {index: position for position, index in enumerate(used)}
    result = deepcopy(terms)
    for term in result:
        term["powers"] = [[remap[index], power] for index, power in term["powers"]]
    return {"poly": result, "args": [deepcopy(features[index]) for index in used]}


def polynomial_tree(node):
    """The scientific primitive-operator view, without factoring/cancellation."""
    if (
        set(node) != {"poly", "args"}
        or not isinstance(node["poly"], list)
        or not 1 <= len(node["poly"]) <= MAX_STORED_NODES
    ):
        raise ValueError("Invalid bounded sparse polynomial")
    features = node["args"]
    if not isinstance(features, list) or len(features) > MAX_STORED_NODES:
        raise ValueError("Invalid polynomial feature array")
    pieces = []
    all_used = set()
    for term in node["poly"]:
        if (
            not isinstance(term, dict)
            or set(term) != {"coefficient", "powers"}
            or not isinstance(term["coefficient"], dict)
            or set(term["coefficient"]) not in ({"const"}, {"param"})
        ):
            raise ValueError("Polynomial coefficients must be fixed literals or identified parameters")
        if not isinstance(term["powers"], list):
            raise ValueError("Invalid sparse monomial powers")
        seen = set()
        piece = dict(term["coefficient"])
        for pair in term["powers"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(v) is not int for v in pair)
                or not 0 <= pair[0] < len(features)
                or pair[0] in seen
                or not 1 <= pair[1] <= 8
            ):
                raise ValueError("Sparse monomial needs distinct feature indices and fixed powers1..8")
            seen.add(pair[0])
            all_used.add(pair[0])
            for _ in range(pair[1]):
                piece = {"op": "mul", "args": [piece, features[pair[0]]]}
        pieces.append(piece)
    if all_used != set(range(len(features))):
        raise ValueError("Polynomial features must be used in its explicit monomials")
    root = pieces[0]
    for piece in pieces[1:]:
        root = {"op": "add", "args": [root, piece]}
    return root


def expanded_equation(equation):
    """Bounded tree view for analyses that do not execute a shared graph."""
    definitions = equation.get("definitions", {})
    if not isinstance(definitions, dict) or len(definitions) > MAX_STORED_NODES:
        raise ValueError("Invalid local expression definitions")
    count = 0
    used = set()

    def visit(node, active, depth):
        nonlocal count
        count += 1
        if count > MAX_EXPANDED_NODES or depth > MAX_DEPTH or not isinstance(node, dict):
            raise ValueError("Expression exceeds bounded expansion/depth resources")
        if "ref" in node:
            name = node["ref"]
            if set(node) != {"ref"} or not isinstance(name, str) or name not in definitions or name in active:
                raise ValueError("Unknown or cyclic local expression reference")
            used.add(name)
            return visit(definitions[name], active | {name}, depth + 1)
        if "poly" in node:
            return visit(polynomial_tree(node), active, depth)
        return (
            {**node, "args": [visit(child, active, depth + 1) for child in node["args"]]}
            if "args" in node
            else dict(node)
        )

    for name in definitions:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Local expression names must be identifiers")
    ast = visit(equation["ast"], set(), 1)
    if used != set(definitions):
        raise ValueError("Unused local expression definitions are not accepted")
    return {key: deepcopy(value) for key, value in equation.items() if key != "definitions"} | {"ast": ast}


def compact_equation(equation):
    """Share identical operator subtrees, preserving order/holes/parameter identity."""
    from collections import Counter

    expanded_equation(equation)  # Bounded validation before sharing; preserve basis blocks.
    if equation.get("definitions"):
        return deepcopy(equation)
    source = deepcopy(equation)
    occurrences = Counter()

    def key(node):
        return json.dumps(node, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def count(node):
        if "op" in node or "poly" in node:
            occurrences[key(node)] += 1
        for child in node.get("args", []):
            count(child)

    count(source["ast"])
    definitions, names = {}, {}

    def visit(node):
        identity = key(node)
        if ("op" in node or "poly" in node) and occurrences[identity] > 1:
            if identity not in names:
                value = {**node, "args": [visit(child) for child in node["args"]]}
                name = f"s{len(definitions)}"
                definitions[name] = value
                names[identity] = name
            return {"ref": names[identity]}
        return {**node, "args": [visit(child) for child in node["args"]]} if "args" in node else dict(node)

    ast = visit(source["ast"])
    return {**source, "schema_version": 2, "ast": ast, "definitions": definitions}
