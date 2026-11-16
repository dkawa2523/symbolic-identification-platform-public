"""Finite, explicit checks; sampled response checks never imply global validity."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from .domains import validate_input_domain, within_bounds
from .expression_graph import expanded_equation
from .expressions import evaluate, inspect_equation
from .parameters import expression_coordinate_role, fixed_scalar, free_parameter_names
from .units import ONE, quantity_role, same_coordinate, unit_spec


def _same_numeric_units(a, b):
    # Radians have no physical dimension; this is NOT a Hz/angular conversion.
    # Conversion and physical parameter aliases retain quantity-kind checks.
    return a.cycle == b.cycle and same_coordinate(a, b, quantity=False)


def equation_unit(equation, variables):
    """Infer physical dimensions AND numerical coordinates; unknown is not unit1."""
    equation = expanded_equation(equation)
    errors = []
    units = {v["id"]: unit_spec(v.get("unit")) for v in variables}

    def visit(node):
        if "var" in node:
            return units[node["var"]]
        if "const" in node:
            return ONE
        if "param" in node:
            return unit_spec(equation["parameters"][node["param"]].get("unit"))
        args = [visit(a) for a in node["args"]]
        op = node["op"]
        if op in {"pow", "pow_int", "rootn"} and args[1] is not None and not _same_numeric_units(args[1], ONE):
            errors.append("power/root index requires dimensionless canonical input")
        if any(a is None for a in args):
            return None
        if any(a.offset for a in args):
            errors.append("Affine equation coordinates unsupported; convert absolute temperature inputs to K")
            return None
        try:
            if op in {"add", "sub"}:
                if not _same_numeric_units(*args):
                    errors.append("addition/subtraction unit coordinates differ; explicit conversion required")
                # Compatible coherent numbers do not establish the same quantity
                # kind. In particular rad/1 arithmetic must not depend on which
                # summand happened to be on the left. Hz still fails above.
                if args[0].angle != args[1].angle or not all(a.kind_known for a in args):
                    return replace(args[0], angle=Fraction(0), kind_known=False)
                return args[0]
            if op in {"mul", "div"}:
                return args[0].product(args[1], 1 if op == "mul" else -1)
            if op in {"neg", "abs"}:
                return args[0]
            if op in {"pow", "pow_int", "rootn"}:
                power = fixed_scalar(node["args"][1], equation["parameters"])
                if power is None:
                    return ONE if _same_numeric_units(args[0], ONE) else None
                return args[0].power(1 / power if op == "rootn" else power)
            if op == "sqrt":
                return args[0].power(Fraction(1, 2))
            if op in {"sin", "cos", "tan"}:
                if args[0].cycle or not _same_numeric_units(args[0], ONE):
                    errors.append(f"{op} requires radians or canonical dimensionless input")
                return ONE
            if args[0].cycle or not _same_numeric_units(args[0], ONE):
                errors.append(f"{op} requires dimensionless canonical input")
            return unit_spec("rad") if op in {"asin", "acos", "atan"} else ONE
        except (ValueError, ZeroDivisionError, OverflowError):
            errors.append("Invalid dimensional power/unit coordinate")
            return None

    result = visit(equation["ast"])
    if result is not None:
        # Point/difference meaning belongs to expression-role arithmetic, not
        # whichever add/sub operand supplied the numerical unit record.
        result = replace(result, role=expression_coordinate_role(equation, variables))
    return result, errors


def dimensions(equation, variables, target):
    result, errors = equation_unit(equation, variables)
    wanted = unit_spec(target.get("unit"))
    produced_role = expression_coordinate_role(equation, variables)
    wanted_role = quantity_role(target)
    if produced_role is not None and wanted_role is not None and produced_role != wanted_role:
        errors.append("point/value and difference output meanings conflict")
    if result is not None and wanted is not None and not _same_numeric_units(result, wanted):
        errors.append("target unit coordinates differ; compatible dimensions alone do not convert numbers")
    return "fail" if errors else "unknown" if result is None or wanted is None else "pass", errors


def response_check(equation, variables, target, request, *, predict=None):
    """A declared finite-step response question, not a global derivative proof."""
    columns = [v["column"] for v in variables]
    feature = columns.index(request["feature"]) if "feature" in request else None
    direction = (
        np.asarray([request["direction"][c] for c in columns])
        if "direction" in request
        else np.eye(len(columns))[feature]
    )
    X = np.asarray([[point[column] for column in columns] for point in request["points"]], dtype=float)
    step = request["step"]
    low, high = X.copy(), X.copy()
    low -= step * direction
    high += step * direction
    for points in (low, X, high):
        validate_input_domain(points, variables)
    predictor = predict if predict is not None else lambda points: evaluate(equation, points, variables)
    a, center, b = [predictor(points) for points in (low, X, high)]
    valid = np.isfinite(a) & np.isfinite(center) & np.isfinite(b)
    for values in (a, center, b):
        valid &= within_bounds(values, target.get("domain"))
    with np.errstate(all="ignore"):
        response = (b - a) / (2 * step) if request["order"] == 1 else (b - 2 * center + a) / step**2
    valid &= np.isfinite(response)
    violation = np.maximum(0, -request["sign"] * response - request["tolerance"])
    worst = int(np.argmax(np.where(valid, violation, np.inf)))
    return {
        "name": request.get("name", f"response_{request.get('feature', 'declared_path')}_{request['order']}"),
        "status": "pass" if valid.all() and not np.any(violation > 0) else "fail",
        "required": True,
        "method": "central finite-step sampled response; no global guarantee",
        "reason": None if valid.all() else "Invalid response rows retained; none dropped",
        "feature": request.get("feature"),
        "direction": request.get("source_direction", request.get("direction")),
        "working_direction": request.get("direction"),
        "path_unit": request.get("path_unit"),
        "order": request["order"],
        "sign": request["sign"],
        "step": step,
        "tolerance": request["tolerance"],
        "response_unit": f"({target.get('unit')})/({request.get('path_unit') if feature is None else variables[feature].get('unit')})^{request['order']}",
        "n_points": len(X),
        "n_valid": int(valid.sum()),
        **response_statistics(violation, valid),
        "worst_point": request["points"][worst],
        "worst_response": float(response[worst]) if valid[worst] else None,
        "knowledge": request.get("knowledge", "user-declared response assumption"),
    }


def check_constraints(equation, variables, target, operators, constraints, *, include_response=True, context=None, size_coordinate="source"):
    """With context, enforce source size; without it, only declared AST structure.

    legacy_working is solely the compatibility check for pre-coordinate artifacts.
    New pipeline callers supply the frozen context; incomplete maps fail closed.
    """
    if size_coordinate not in {"source", "legacy_working"}:
        raise ValueError("Unknown complexity coordinate system")
    stats = inspect_equation(equation, variables)
    records = []

    def record(name, passed, required=True, method="structural", reason=None):
        records.append(
            {
                "name": name,
                "status": passed if isinstance(passed, str) else "pass" if passed else "fail",
                "required": required,
                "method": method,
                "reason": reason,
            }
        )

    record("operators", stats["operators"] <= set(operators))
    if size_coordinate == "legacy_working" or context is None:
        size = stats["n_nodes"]
        method = ("legacy working-coordinate AST; no source-size guarantee"
                  if size_coordinate == "legacy_working" else
                  "declared expression-coordinate AST; no source mapping supplied")
    else:
        from .coordinates import source_complexity
        size = source_complexity(equation, context)
        method = "expanded source-coordinate AST"
    record("max_nodes", size <= constraints["max_nodes"], method=method)
    records[-1].update(count=size, limit=constraints["max_nodes"])
    if constraints.get("max_free_parameters") is not None:
        count = len(free_parameter_names(equation))
        record("max_free_parameters", count <= constraints["max_free_parameters"])
        records[-1].update(count=count, limit=constraints["max_free_parameters"])
    allowed = constraints.get("allowed_variables")
    if allowed is not None:
        record("variables", stats["variables"] <= set(allowed))
    status, reasons = dimensions(equation, variables, target)
    record("units", status, constraints["require_units"] or status == "fail", reason="; ".join(reasons) or None)
    if constraints.get("equation_template") is not None or equation.get("template") is not None:
        from .equation_templates import template_checks

        records.extend(template_checks(equation, variables, target, constraints.get("equation_template")))
    if include_response:
        records.extend(
            prediction_constraints(lambda points: evaluate(equation, points, variables), variables, target, constraints)
        )
    return records


def prediction_constraints(predict, variables, target, constraints):
    """Checks of one prediction view, separate from component AST admissibility."""
    records = []

    def record(name, passed, method="structural"):
        records.append(
            {"name": name, "status": "pass" if passed else "fail", "required": True, "method": method, "reason": None}
        )

    if constraints.get("domain_probes"):
        # Automatic validity coverage is NOT an expansion of explicit response bounds.
        X = np.asarray([[p[v["column"]] for v in variables] for p in constraints["domain_probes"]], dtype=float)
        prediction = predict(X)
        valid = np.isfinite(prediction)
        record("domain_prediction_validity", bool(valid.all()),
               method="automatic domain points; finite prediction only, no response-bound expansion")
        records[-1].update(n_points=len(X), n_valid=int(valid.sum()))
        if target.get("domain") is not None:
            record("target_domain_validity", bool(valid.all() and within_bounds(prediction, target["domain"]).all()),
                   method="automatic domain points; declared target domain only")
    if constraints.get("probes"):
        X = np.asarray([[p[v["column"]] for v in variables] for p in constraints["probes"]], dtype=float)
        prediction = predict(X)
        passed = bool(np.isfinite(prediction).all())
        limits = constraints.get("response_bounds", [None, None])
        if limits[0] is not None:
            passed = passed and bool((prediction >= limits[0]).all())
        if limits[1] is not None:
            passed = passed and bool((prediction <= limits[1]).all())
        record(
            "probe_response",
            passed,
            method=constraints.get("probe_method", "explicit sampled probes; no global guarantee"),
        )
        valid = np.isfinite(prediction)
        violation = np.zeros(len(prediction))
        if limits[0] is not None:
            violation = np.maximum(violation, limits[0] - prediction)
        if limits[1] is not None:
            violation = np.maximum(violation, prediction - limits[1])
        worst = int(np.argmax(np.where(valid, violation, np.inf)))
        records[-1].update(
            n_points=len(X), n_valid=int(valid.sum()),
            **response_statistics(violation, valid),
            response_unit=target.get("unit"), response_bounds=limits,
            worst_point=constraints["probes"][worst],
            worst_prediction=float(prediction[worst]) if valid[worst] else None,
            reason=None if valid.all() else "Invalid probe rows retained; none dropped",
        )
    if target.get("domain") is not None:
        # Domain endpoints are physical task assumptions, distinct from train ranges.
        X = np.asarray([[p[v["column"]] for v in variables] for p in constraints.get("probes", [])], dtype=float)
        if len(X):
            prediction = predict(X)
            record(
                "target_domain_probes",
                bool(np.isfinite(prediction).all() and within_bounds(prediction, target["domain"]).all()),
                method="explicit sampled probes; no global guarantee",
            )
    records.extend(
        response_check(None, variables, target, request, predict=predict)
        for request in constraints.get("response_checks", [])
    )
    return records


def artifact_size_coordinate(payload):
    """Stored limit semantics are explicit; absent marker means historical working AST."""
    marker = payload.get("complexity_coordinate", "working")
    if not isinstance(marker, str) or marker not in {"working", "source"}:
        raise ValueError("Unknown stored complexity_coordinate")
    return "source" if marker == "source" else "legacy_working"



def response_statistics(violation, valid):
    """Invalid points count as failed checks, never as measured finite violations."""
    valid = np.asarray(valid, dtype=bool)
    violation = np.asarray(violation, dtype=float)
    n = len(valid)
    n_bad = int(np.sum(valid & (violation > 0)))
    n_invalid = int(np.sum(~valid))
    return {
        "n_violating": n_bad, "n_invalid": n_invalid,
        "violation_rate": n_bad / n if n else None,
        "failure_rate": (n_bad + n_invalid) / n if n else None,
        "rate_denominator": "all declared points; invalid points separate from finite violations",
        "max_violation": float(violation.max()) if n and valid.all() else None,
        "max_finite_violation": float(violation[valid].max()) if valid.any() else None,
    }


def response_evaluation(predict, variables, target, constraints, origin):
    """Describe existing study checks by point origin; no independent-Test claim.

    Additional study conditions participate in Selection eligibility. Their external
    history is not knowable here, and shared points are explicitly counted.
    """
    if origin is None:
        return [{"point_origin": "unrecorded", "status": "not_evaluated",
                 "reason": "Historical study lacks response-point origin metadata"}]
    n_probes, n_checks = origin["discovery_probes"], origin["discovery_checks"]
    probes, checks = constraints.get("probes", []), constraints.get("response_checks", [])
    discovery_points = probes[:n_probes] + [p for c in checks[:n_checks] for p in c["points"]]
    rows = []
    for name, points, requests in (
        ("prepared_problem", probes[:n_probes], checks[:n_checks]),
        ("additional_study", probes[n_probes:], checks[n_checks:]),
    ):
        selected = {"probes": points, "response_checks": requests,
                    "response_bounds": constraints.get("response_bounds", [None, None])}
        results = prediction_constraints(predict, variables, target, selected)
        # Target-domain checks are a separate requirement, not response violations.
        results = [r for r in results if r["name"] != "target_domain_probes"]
        if not results:
            results = [{"status": "not_evaluated", "reason": "No response evaluation points specified"}]
        for i, result in enumerate(results):
            point_rows = points if points and i == 0 else requests[i - bool(points)]["points"] if requests else []
            rows.append({**result, "point_origin": name,
                "data_role": "Selection eligibility; not independent Test",
                "independence": "not_asserted; external reuse history unknown",
                "coordinates": "working; canonical units and frozen input geometry",
                "point_units": {v["column"]: v.get("unit") for v in variables},
                "bounds_scope": "current study conditions, including additional restrictions",
                "overlap_with_prepared_points": (sum(p in discovery_points for p in point_rows)
                                                 if name == "additional_study" else None),
                "points": point_rows})
    return rows
