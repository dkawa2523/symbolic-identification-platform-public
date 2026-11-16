"""Explicit Train-only allocation; archived candidates are never rewritten or deduplicated."""

import json
from collections import defaultdict

from .expression_graph import expanded_equation


def structure_key(equation):
    """Syntactic scaffold, preserving fixed values and shared/tied parameter identity.

    This groups allocations only. It is not a claim of algebraic equivalence.
    """
    equation = expanded_equation(equation)
    names = {}

    def visit(node):
        if "param" in node:
            name = node["param"]
            parameter = equation["parameters"][name]
            seen = set()
            while parameter.get("tied_to"):
                if name in seen:
                    raise ValueError("Cyclic parameter alias")
                seen.add(name)
                name = parameter["tied_to"]
                parameter = equation["parameters"][name]
            if parameter["trainable"]:
                names.setdefault(name, len(names))
                return ["free", names[name], parameter.get("unit"), parameter.get("bounds")]
            return ["fixed", parameter["value"], parameter.get("unit")]
        if "op" in node:
            return [node["op"], *[visit(child) for child in node["args"]]]
        return node

    return json.dumps(visit(equation["ast"]), sort_keys=True, separators=(",", ":"))


def allocate_candidates(records, qualities, limit, policy):
    """Return references to unchanged records. Qualities contain Train evidence only."""
    if len({r["candidate_id"] for r in records}) != len(records):
        raise ValueError("Native candidate IDs must be unique within a discovery run")
    if policy == "all":
        return list(records)
    if policy != "pareto_diverse":
        raise ValueError(f"Unknown candidate pool policy: {policy}")

    def key(record):
        q = qualities[record["candidate_id"]]
        return q["train_rmse"], q["nodes"], q["free_parameters"], record["candidate_id"]

    ordered = []
    for eligible in (True, False):
        pending = [r for r in records if qualities[r["candidate_id"]]["train_checks_pass"] == eligible]
        while pending and len(ordered) < limit:
            front = []
            for record in pending:
                q = key(record)[:3]
                if not any(
                    all(a <= b for a, b in zip(key(other)[:3], q, strict=True))
                    and any(a < b for a, b in zip(key(other)[:3], q, strict=True))
                    for other in pending
                ):
                    front.append(record)
            groups = defaultdict(list)
            for record in sorted(front, key=key):
                groups[structure_key(record["equation"])].append(record)
            # Best-loss representative first, then one per scaffold before repeats.
            queues = sorted(groups.values(), key=lambda group: key(group[0]))
            while any(queues) and len(ordered) < limit:
                for group in queues:
                    if group and len(ordered) < limit:
                        ordered.append(group.pop(0))
            used = {r["candidate_id"] for r in front}
            pending = [r for r in pending if r["candidate_id"] not in used]
    return ordered[:limit]
