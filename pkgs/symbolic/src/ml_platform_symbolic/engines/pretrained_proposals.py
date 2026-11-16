"""Full-problem pretrained proposals and exact-binding AST fragment collection.

Fragments are new slot hypotheses, never equivalent replacements for their parent.
No targets for unknown slots, inverse transforms, or accuracy-based fragment ranks.
"""

import json
import time
from collections import Counter, deque
from copy import deepcopy
from dataclasses import replace

import numpy as np

from ..expressions import evaluate


def full_problem_request(request, name, calls, options, index):
    slot = request.equation_template["slots"][name]
    variables = deepcopy(request.variables)
    columns = [request.X[:, i] for i in range(len(variables))]
    feature_ids = {json.dumps({"var": v["id"]}, sort_keys=True): v["id"] for v in variables}
    used_ids = {v["id"] for v in variables}
    maps = []
    for call_index, binding in enumerate(calls):
        mapping, aliases = {}, {}
        for formal, ast in zip(slot["arguments"], binding, strict=True):
            key = json.dumps(ast, sort_keys=True)
            if key not in feature_ids:
                feature = f"__proposal_{len(variables)}"
                while feature in used_ids:
                    feature += "_"
                values = evaluate({"schema_version": 2, "ast": ast, "parameters": {}}, request.X, request.variables)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite derived proposal input; no Train rows removed")
                columns.append(values)
                variables.append({"id": feature, "column": feature, "unit": "1"})
                feature_ids[key] = feature
                used_ids.add(feature)
            feature = feature_ids[key]
            if feature in mapping:
                aliases[formal["name"]] = mapping[feature]
            else:
                mapping[feature] = formal["name"]
        maps.append(
            {
                "call_index": call_index,
                "binding": deepcopy(binding),
                "variable_mapping": mapping,
                "aliased_unused_formals": aliases,
            }
        )
    limits = {
        **request.limits,
        "response_mode": "filter",
        "probes": [],
        "response_bounds": [None, None],
        "response_checks": [],
        "equation_template": None,
        "require_units": False,
        "allowed_variables": None,
    }
    proposed = replace(
        request,
        X=np.column_stack(columns),
        variables=variables,
        options=deepcopy(options["proposal_options"]),
        limits=limits,
        seed=(request.seed + index) % 2**32,
        fixed_constants={},
        equation_template=None,
        domain=None,
    )
    return proposed, maps


def walk_nodes(root):
    queue = deque([((), root)])
    while queue:
        path, node = queue.popleft()
        yield path, node
        queue.extend(((*path, i), child) for i, child in enumerate(node.get("args", [])))


def fragment(body, node, mapping):
    def rename(value):
        if "var" in value:
            return {"var": mapping[value["var"]]}
        return {**value, "args": [rename(a) for a in value["args"]]} if "args" in value else deepcopy(value)

    used = {n["param"] for _, n in walk_nodes(node) if "param" in n}
    pending = list(used)
    while pending:
        parameter = body["parameters"][pending.pop()]
        for _, child in walk_nodes(parameter.get("derived_from", {})):
            if "param" in child and child["param"] not in used:
                used.add(child["param"])
                pending.append(child["param"])
    return {
        **deepcopy(body),
        "ast": rename(node),
        "parameters": {key: deepcopy(body["parameters"][key]) for key in body["parameters"] if key in used},
    }


def fingerprint(equation, *, initial_values):
    if any(p.get("tied_to") is not None for p in equation["parameters"].values()):
        raise ValueError("Subexpression proposals do not support tied_to parameter aliases")
    names, declarations = {}, {}

    def node(value):
        if "param" in value:
            original = value["param"]
            if original not in names:
                name = f"p{len(names)}"
                names[original] = name
                parameter = deepcopy(equation["parameters"][original])
                if "derived_from" in parameter:
                    parameter["derived_from"] = node(parameter["derived_from"])
                if not initial_values and (parameter["trainable"] or "derived_from" in parameter):
                    parameter.pop("value", None)
                declarations[name] = parameter
            return {"param": names[original]}
        return {**value, "args": [node(a) for a in value["args"]]} if "args" in value else value

    ast = node(equation["ast"])
    return json.dumps(
        {"ast": ast, "parameters": declarations, "arithmetic": equation.get("arithmetic", "strict_real")},
        sort_keys=True,
    )


def collect_parts(candidates, slot, maps, normalize, check, *, deadline=None):
    """Parent/call round-robin, then structural-group round-robin; no data scores."""

    def budget():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Subexpression extraction exceeded wall budget; no partial pool selected")

    sources, counters, rejected = [], Counter(), []
    for candidate in candidates:
        budget()
        transform = {}
        try:
            body = normalize(candidate, transform)
            if any(p.get("tied_to") is not None for p in body["parameters"].values()):
                raise ValueError("Subexpression proposals do not support tied_to parameter aliases")
            if any("var" in n for p in body["parameters"].values() for _, n in walk_nodes(p.get("derived_from", {}))):
                raise ValueError("Derived parameters must be parameter-only; variable dependencies unsupported")
        except (ValueError, TypeError) as exc:
            rejected.append({"native_candidate_id": candidate.candidate_id, "reason": str(exc)})
            continue
        for binding in maps:
            queue = deque()
            skips = Counter()
            for path, node in walk_nodes(body["ast"]):
                budget()
                counters["visited_nodes"] += 1
                variables = {n["var"] for _, n in walk_nodes(node) if "var" in n}
                if not variables <= binding["variable_mapping"].keys():
                    skips["depends_on_inputs_outside_this_call"] += 1
                    continue
                try:
                    part = check(fragment(body, node, binding["variable_mapping"]), slot)
                except (ValueError, TypeError) as exc:
                    skips[str(exc)] += 1
                    continue
                identifier = (
                    candidate.candidate_id
                    + "/part/"
                    + str(binding["call_index"])
                    + "/"
                    + (".".join(map(str, path)) or "root")
                )
                source = {
                    **deepcopy(transform),
                    **deepcopy(binding),
                    "native_candidate_id": candidate.candidate_id,
                    "ast_path": list(path),
                    "ast_path_basis": "normalized proposal AST after recorded coordinate transformation",
                    "proposal_kind": "subexpression; new slot hypothesis, not native equivalence",
                }
                queue.append((identifier, part, source, candidate))
                counters["eligible_occurrences"] += 1
            sources.append(queue)
            rejected.append(
                {
                    "native_candidate_id": candidate.candidate_id,
                    "call_index": binding["call_index"],
                    "skip_counts": dict(skips),
                }
            )
    groups, seen = {}, set()
    active = deque(source for source in sources if source)
    while active:
        budget()
        source = active.popleft()
        item = source.popleft()
        exact = fingerprint(item[1], initial_values=True)
        if exact in seen:
            counters["exact_duplicates"] += 1
        else:
            seen.add(exact)
            shape = fingerprint(item[1], initial_values=False)
            groups.setdefault(shape, deque()).append(item)
        if source:
            active.append(source)
    counters["distinct_structures"] = len(groups)
    counters["distinct_initializations"] = len(seen)
    ordered = []
    active = deque(groups.values())
    while active:
        budget()
        group = active.popleft()
        item = group.popleft()
        item[2]["selection_order"] = len(ordered)
        ordered.append(item)
        if group:
            active.append(group)
    return ordered, {
        **dict(counters),
        "sources": rejected,
        "selection_rule": "parent/call round-robin of breadth-first AST occurrences; exact value-aware dedup; first-seen structural groups round-robin; max_proposals applied after prediction-record success",
        "information_used": "native candidate order, exact AST paths/bindings, declared slot requirements, fixed values, free-parameter sharing/bounds and initializations; no fragment prediction score or Final",
    }
