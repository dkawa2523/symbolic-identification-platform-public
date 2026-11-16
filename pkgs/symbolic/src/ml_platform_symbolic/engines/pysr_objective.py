"""Compile shared finite-point constraints into the native PySR objective.

This changes native candidate scoring/selection and constant optimization, not only export.
Proposed intermediate structures may exceed slot limits; their native loss is infinite.
The declared finite stencils are not analytic derivatives or global guarantees.
"""

import math

from ..response_search import compile_response_constraints, response_enforcement, response_bounds


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("PySR objective constants must be finite")
    return repr(value)


def template_size_expression(template, variable_costs):
    """Compile the expanded known skeleton; each shared-slot call is charged."""
    slots = {name: f"slot{i}" for i, name in enumerate(template["slots"])}
    def count(node, active=()):
        if "ref" in node:
            key = ("component", node["ref"])
            if key in active:
                raise ValueError("Cyclic template size component")
            return count(template["components"][node["ref"]]["ast"], (*active, key))
        if "param" in node:
            name = node["param"]
            parameter = template.get("parameters", {})[name]
            if "derived_from" in parameter:
                key = ("parameter", name)
                if key in active:
                    raise ValueError("Cyclic template size parameter")
                return count(parameter["derived_from"], (*active, key))
            return "1"
        if "var" in node:
            return str(variable_costs[node["var"]])
        if "const" in node:
            return "1"
        children = [count(child, active) for child in node["args"]]
        if "slot" in node:
            native = slots[node["slot"]]
            return f"slot_size(SymbolicRegression.get_tree(SymbolicRegression.get_contents(tree).{native}), [{', '.join(children)}])"
        return "(1 + " + " + ".join(children) + ")"
    return count(template["ast"])


def native_response_objective(request, *, variable_costs=None, output_cost=0):
    """Return native options and evidence, without importing PySR/Julia."""
    from ml_platform_symbolic.engines.pysr import capabilities, unit_profile
    _, _, units = unit_profile(request)
    enforcement = response_enforcement(request.limits, capabilities())
    hard = enforcement["effective"] == "hard"
    if hard and "constraint_weight" in request.options:
        raise ValueError("constraint_weight applies only to soft response enforcement")
    weight = request.options.get("constraint_weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("constraint_weight must be positive and finite")
    objective = request.options.get("objective")
    if objective not in {None, "mse"}:
        raise ValueError("PySR objective accepts only 'mse'; omit it to retain native defaults")
    mode = request.limits.get("response_mode", "search")
    has_points = bool(request.limits.get("probes") or request.limits.get("response_checks"))
    template = getattr(request, "equation_template", None)
    scaled_objective = objective == "mse" or (mode == "search" and has_points)
    if not template and not scaled_objective:
        return {}, {"mode": mode, "search_response_constraints": False, "objective": "native"}
    compiled = (compile_response_constraints(request) if scaled_objective
                else {"points": [], "terms": [], "data_scale": 1.0})
    if mode == "search":
        response_bounds(compiled)  # Contradictions fail before native execution.
    points, terms = (compiled["points"], compiled["terms"]) if mode == "search" else ([], [])
    data_scale = compiled["data_scale"] if scaled_objective else 1.0
    # Compile/validate in the original coordinates first: coupled domains,
    # directional stencils and their Train-derived scales must not be projected.
    # Only the native input matrix follows the same column map as native fit.
    from .pysr import native_feature_indices

    feature_indices = native_feature_indices(request)
    domain = getattr(request, "domain", None)
    if points and domain is not None:
        from ..domains import region_mask

        if domain["columns"] != [v["column"] for v in request.variables]:
            raise ValueError("Response domain columns must match the full original input order")
        if not region_mask(points, domain).all():
            raise ValueError("Response points lie outside the original declared input domain")
    native_points = [[row[i] for i in feature_indices] for row in points]
    # Julia X uses features x samples. Retain duplicate projected points so
    # every original stencil index and coefficient keeps its exact meaning.
    matrix = ", ".join(_number(value) for row in native_points for value in row)
    source = [
        "function constrained_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}",
    ]
    if units["native_units"]:
        source.append("    unit_violation = SymbolicRegression.DimensionalAnalysisModule.violates_dimensional_constraints(tree, dataset, options)")
        if units["mode"] == "hard":
            source.append("    unit_violation && return L(Inf)")
    if template:
        # Match portable AST size, not native weighted complexity. square/cube
        # export as pow_int(base, exponent), including the exponent leaf.
        source.extend([
            "    function slot_size(node, argument_sizes)",
            "        node.degree == 0 && return node.constant ? 1 : argument_sizes[Int(node.feature)]",
            "        n = 1 + slot_size(node.l, argument_sizes)",
            "        if node.degree == 2",
            "            n += slot_size(node.r, argument_sizes)",
            "        elseif nameof(options.operators.unaops[Int(node.op)]) in (:square, :cube)",
            "            n += 1",
            "        end",
            "        return n",
            "    end",
        ])
        for i, slot in enumerate(template["slots"].values()):
            source.append(
                f"    slot_size(SymbolicRegression.get_tree(SymbolicRegression.get_contents(tree).slot{i}), ones(Int, {len(slot['arguments'])})) > {int(slot['max_nodes'])} && return L(Inf)"
            )
    if template and "max_nodes" in request.limits:
        costs = variable_costs or {v["id"]: 1 for v in request.variables}
        size = template_size_expression(template, costs)
        source.append(f"    ({size}) + {int(output_cost)} > {int(request.limits['max_nodes'])} && return L(Inf)")
    source.extend([
        "    prediction, complete = eval_tree_array(tree, dataset.X, options.operators)",
        "    (!complete || !all(isfinite, prediction)) && return L(Inf)",
        f"    data_loss = sum(abs2, (prediction .- dataset.y) ./ T({_number(data_scale)})) / length(dataset.y)",
        "    penalty = zero(L)",
    ])
    if points:
        source.extend([
            f"    probe_X = reshape(T[{matrix}], {len(feature_indices)}, {len(points)})",
            "    probe_y, probe_complete = eval_tree_array(tree, probe_X, options.operators)",
            "    (!probe_complete || !all(isfinite, probe_y)) && return L(Inf)",
        ])
    for term in terms:
        linear = " + ".join(
            f"T({_number(c)}) * probe_y[{int(i) + 1}]"
            for i, c in zip(term["indices"], term["coefficients"], strict=True)
        )
        source.extend([f"    response = {linear}", "    isfinite(response) || return L(Inf)"])
        if hard:
            if term.get("lower") is not None:
                source.append(f"    response < T({_number(term['lower'])}) && return L(Inf)")
            if term.get("upper") is not None:
                source.append(f"    response > T({_number(term['upper'])}) && return L(Inf)")
            continue
        violations = ["zero(T)"]
        if term.get("lower") is not None:
            violations.append(f"T({_number(term['lower'])}) - response")
        if term.get("upper") is not None:
            violations.append(f"response - T({_number(term['upper'])})")
        source.append(
            f"    penalty += abs2(max({', '.join(violations)}) / T({_number(term['scale'])}))"
        )
    if units["native_units"] and units["mode"] == "soft":
        source.append(f"    data_loss += unit_violation ? L({_number(units['penalty'])}) : zero(L)")
    source.extend([
        f"    return L(data_loss + T({_number(weight)}) * penalty / {max(len(terms), 1)})",
        "end",
    ])
    key = "loss_function_expression" if template is not None else "loss_function"
    return {key: "\n".join(source)}, {
        "mode": mode,
        "search_response_constraints": bool(points),
        "objective": "train_scaled_mse" if scaled_objective else "native_mse",
        "slot_max_nodes": {name: slot["max_nodes"] for name, slot in template["slots"].items()} if template else {},
        "slot_size_enforcement": "native full-expression loss gate; portable AST node count, exponent literals included; does not guarantee every proposed intermediate structure is valid" if template else None,
        "method": "native objective during structure search and constant optimization",
        "template_total_size_gate": bool(template and "max_nodes" in request.limits),
        "source_variable_costs": variable_costs, "source_output_cost": output_cost,
        "weight": None if hard else weight,
        "response_enforcement": enforcement,
        "unit_search": units,
        "points": points,
        "native_points": native_points,
        "native_variable_ids": [request.variables[i]["id"] for i in feature_indices],
        "terms": terms,
        "data_scale": data_scale,
        "scope": "declared finite points and finite-step responses; not a global guarantee",
    }
