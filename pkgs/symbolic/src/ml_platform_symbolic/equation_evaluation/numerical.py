"""Pure shared numerical evidence. No preparation, worker, Final or selection ownership."""

from copy import deepcopy

import numpy as np

from ..constraints import check_constraints, equation_unit, prediction_constraints, response_evaluation
from ..coordinates import execution_constraints, execution_region, source_complexity
from ..data import value_digest
from ..domains import region_mask, validate_bounds, validation_constraints, within_bounds
from ..execution_views import bind_view
from ..expressions import equation_text, evaluate, inspect_equation
from ..hypotheses import predict_working, prediction_equations, residual_evidence
from ..metrics import prediction_metrics
from ..parameters import free_parameter_bounds, free_parameter_names, parameter_table, with_parameters
from ..profile_evidence import constraint_application
from ..units import coherent_unit_label


def _template_readout(equation, variables, X):
    """Compute named representation evidence once; reporting only renders it."""
    from ..equation_templates import describe_template

    evidence = describe_template(equation, variables)
    specification = equation["template"]["specification"]

    def declared_text(ast):
        # Display-only labels: use the existing operator formatter, not a second parser.
        names = list(variables)

        def labels(node):
            if "ref" in node or "slot" in node:
                label = (
                    node["ref"]
                    if "ref" in node
                    else (node["slot"] + "(" + ", ".join(declared_text(a) for a in node["args"]) + ")")
                )
                key = f"__display_{len(names)}"
                names.append({"id": key, "column": label})
                return {"var": key}
            return {**node, "args": [labels(a) for a in node["args"]]} if "args" in node else node

        display = {"schema_version": 1, "ast": labels(ast), "parameters": {}}
        return equation_text(display, names, parameter_names=True)

    components = []
    for name, part in evidence["components"].items():
        values = evaluate(part, X, variables)
        unit, errors = equation_unit(part, variables)
        components.append(
            {
                "name": name,
                "description": specification["components"][name].get("description"),
                "declared_formula": declared_text(specification["components"][name]["ast"]),
                "formula": equation_text(part, variables, parameter_names=True),
                "unit": coherent_unit_label(unit) if unit is not None and not errors else None,
                "n_rows": len(X),
                "n_valid": int(np.isfinite(values).sum()),
                "min": float(values.min()) if len(X) and np.isfinite(values).all() else None,
                "max": float(values.max()) if len(X) and np.isfinite(values).all() else None,
            }
        )
    slots = []
    for name, part in evidence["raw_slot_equations"].items():
        slot = specification["slots"][name]
        formal = [{"id": a["name"], "column": a["name"], "unit": a["unit"]} for a in slot["arguments"]]
        slots.append(
            {
                "name": name,
                "arguments": slot["arguments"],
                "unit": slot["unit"],
                "description": slot.get("description"),
                "formula": equation_text(part, formal, parameter_names=True),
                "parameters": parameter_table(part),
            }
        )
    return {
        "status": "computed",
        "knowledge": evidence["knowledge"],
        "declared_formula": declared_text(specification["ast"]),
        "sharing": "Every call with the same slot name shares one function and its coefficients. Different names denote separate functions, not proven identifiable components.",
        "components": components,
        "slots": slots,
        "dependent_parameters": [
            {
                "name": n,
                "value": value,
                "unit": specification["parameters"][n]["unit"],
                "formula": declared_text(specification["parameters"][n]["derived_from"]),
            }
            for n, value in evidence["dependent_values"].items()
        ],
        "coordinates": "canonical working; ranges on frozen Selection inputs only",
        "interpretation_limit": evidence["interpretation_limit"],
    }



def local_parameter_evidence(record, X, variables, target):
    """Bounded coefficient sensitivity on all Selection inputs, never targets/Final."""
    evidence = {
        "status": "unavailable", "data_role": "selection", "n_samples": len(X),
        "method": "bounded finite-difference scaled coefficient Jacobian SVD",
        "scope": "explicit point equation at saved coefficients; all Selection inputs, unweighted",
        "interpretation_limit": (
            "Full local rank does not prove global or structural identifiability. "
            "Rank deficiency may reflect this design, coefficient point, finite-difference resolution, "
            "or redundant parameters. This is not an uncertainty interval or a refit. "
            "Active bounds restrict feasible directions; rank is not a constrained uniqueness test."
        ),
    }
    if record["kind"] != "explicit":
        return {**evidence, "reason": "Only explicit point-equation coefficients are supported"}
    equation = record["equation"]
    try:
        names = sorted(free_parameter_names(equation, require_origin=True))
        evidence.update(parameter_names=names, n_parameters=len(names))
        if not names:
            return {**evidence, "status": "not_applicable", "reason": "No independent free parameters"}
        bounds = free_parameter_bounds(equation, names)
        values = np.asarray([equation["parameters"][name]["value"] for name in names], dtype=float)
        center, valid = predict_record(record, X, variables, target)
        evidence["n_valid_center"] = int(valid.sum())
        if not len(X) or not valid.all() or not np.isfinite(center).all():
            raise ValueError("All frozen Selection rows must have valid finite center predictions")
        output_scale = float(np.max(abs(center))) or 1.0
        evidence.update(
            output_scale=output_scale,
            output_scale_rule="max absolute center prediction; 1 only when all predictions are zero",
            parameter_scale_rule="absolute nonzero value; otherwise finite bound width; otherwise 1 working-unit coordinate",
            relative_step=float(np.cbrt(np.finfo(float).eps)), relative_rank_tolerance=1e-7, parameters=[],
        )
        columns, half_columns, roundoff_columns = [], [], []
        evidence["resolution_check"] = (
            "Compare the coefficient Jacobian at recorded steps and half steps; combine "
            "their RMS Frobenius difference with an output-roundoff estimate. Withhold "
            "rank when this exceeds rank discrimination. This is an empirical local "
            "stability check, not a rigorous error bound for arbitrary arithmetic or nonsmooth models."
        )
        evidence["roundoff_assumption"] = (
            "8*float64 epsilon times the larger normalized endpoint magnitude, divided "
            "by the normalized parameter displacement; does not bound internal AST cancellation."
        )
        for index, (name, value, (lower, upper)) in enumerate(zip(names, values, bounds, strict=True)):
            scale = abs(value) or (upper - lower if lower is not None and upper is not None else 1.0)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"No finite positive coefficient scale for {name}")
            desired = evidence["relative_step"] * scale
            plus = min(desired, upper - value) if upper is not None else desired
            minus = min(desired, value - lower) if lower is not None else desired
            if plus > 0 and minus > 0:
                plus = minus = min(plus, minus)
            high, low = float(value + plus), float(value - minus)
            if not np.isfinite([high, low]).all() or high <= low:
                raise ValueError(f"No representable finite-difference step for {name}")
            evidence["parameters"].append({
                "name": name, "value": float(value), "unit": equation["parameters"][name].get("unit"),
                "scale": float(scale), "bounds": [lower, upper],
                "step_plus": float(high - value), "step_minus": float(value - low),
                "difference": "central" if plus > 0 and minus > 0 else "forward" if plus > 0 else "backward",
                "active_bound": "lower" if lower is not None and value == lower else "upper" if upper is not None and value == upper else None,
            })

            def probe(proposed, value=value, index=index, name=name):
                if proposed == value:
                    return center
                trial_values = values.copy()
                trial_values[index] = proposed
                trial = {**record, "equation": with_parameters(equation, names, trial_values)}
                # This temporary perturbation is a distinct state, never a mutation
                # of the saved execution subject or a published fitted variant.
                if "subject" in record or "execution_view" in record:
                    trial = bind_view(trial, record.get("execution_view"))
                prediction, probe_valid = predict_record(trial, X, variables, target)
                if not probe_valid.all() or not np.isfinite(prediction).all():
                    raise ValueError(f"Finite-difference probe invalid on frozen Selection rows: {name}")
                return prediction

            def difference(hi, lo, scale=scale, name=name, probe=probe):
                delta = (hi - lo) / scale
                if not np.isfinite(delta) or delta <= 0:
                    raise ValueError(f"No representable finite-difference verification step for {name}")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    a, b = probe(hi) / output_scale, probe(lo) / output_scale
                    derivative = (a - b) / delta
                    roundoff = 8 * np.finfo(float).eps * np.maximum(abs(a), abs(b)) / delta
                if not np.isfinite(derivative).all() or not np.isfinite(roundoff).all():
                    raise ValueError(f"Nonfinite scaled sensitivity or roundoff estimate for {name}")
                return derivative, roundoff

            half_high, half_low = float(value + (high - value) / 2), float(value - (value - low) / 2)
            if half_high == high and half_low == low:
                raise ValueError(f"Half-step verification has no distinct representable probe for {name}")
            evidence["parameters"][-1].update(
                verification_step_plus=float(half_high - value), verification_step_minus=float(value - half_low)
            )
            column, roundoff = difference(high, low)
            half_column, half_roundoff = difference(half_high, half_low)
            columns.append(column)
            half_columns.append(half_column)
            roundoff_columns.append(np.maximum(roundoff, half_roundoff))
        jacobian = np.column_stack(columns) / np.sqrt(len(X))
        half_jacobian = np.column_stack(half_columns) / np.sqrt(len(X))
        roundoff_matrix = np.column_stack(roundoff_columns) / np.sqrt(len(X))
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if not np.isfinite(singular).all():
            raise ValueError("Scaled Jacobian singular values exceed finite representation")
        threshold = float(singular[0] * evidence["relative_rank_tolerance"])
        roundoff_floor = float(np.hypot.reduce(roundoff_matrix.ravel()))
        step_difference = float(np.hypot.reduce(abs(jacobian - half_jacobian).ravel()))
        resolution = roundoff_floor + step_difference
        if not np.isfinite(resolution):
            raise ValueError("Coefficient Jacobian resolution estimate exceeds finite representation")
        evidence.update(
            singular_values=singular.tolist(), rank_threshold=threshold,
            roundoff_resolution=roundoff_floor, half_step_difference=step_difference,
            numerical_resolution=resolution,
        )
        if resolution > threshold or (resolution > 0 and np.any(abs(singular - threshold) <= resolution)):
            raise ValueError(
                "Local coefficient rank is unresolved at the finite-difference numerical resolution; "
                "roundoff or step-size sensitivity prevents distinguishing rank deficiency from full rank"
            )
        rank = int(np.sum(singular > threshold))
        norms = np.hypot.reduce(abs(jacobian), axis=0)
        if not np.isfinite(norms).all():
            raise ValueError("Scaled coefficient sensitivity exceeds finite representation")
        evidence.update({
            "status": "computed", "rank": rank, "nullity": len(names) - rank,
            "singular_values": singular.tolist(), "rank_threshold": threshold,
            "sample_normalization": "Jacobian divided by sqrt(n_samples); singular values use RMS sensitivity",
            "column_rms_sensitivity": dict(zip(names, norms.tolist(), strict=True)),
            "insensitive_parameters": [name for name, norm in zip(names, norms, strict=True) if norm <= threshold],
            "full_column_rank": rank == len(names),
            "condition_number": float(singular[0] / singular[-1]) if rank == len(names) else None,
        })
    except (ValueError, np.linalg.LinAlgError) as error:
        evidence["reason"] = str(error)
    return evidence


def prerequisite_failure(record, row):
    """Explain confirmed relation failure without conflating prediction/parameter evidence.

    Missing historical prerequisite evidence or an unfinished parameter audit does
    not erase supported mathematics. Each backend still checks its own capability
    (prediction availability, audited free parameters, finite required scores).
    """
    if row.get("requirements_eligible") is False:
        return "Confirmed relation prerequisite failure: " + (
            "; ".join(row.get("requirement_failures", [])) or row.get("reason") or "required constraints failed"
        )
    if record.get("native_check", {}).get("passed") is False:
        return "Confirmed relation prerequisite failure: native fidelity failed"
    return None


def predict_record(record, X, variables, target):
    prediction, prediction_valid = predict_working(record, X, variables)
    valid = prediction_valid & within_bounds(prediction, target.get("domain"))
    valid &= within_bounds(prediction, record.get("target", target).get("domain"))
    for i, variable in enumerate(record.get("variables", variables)):
        valid &= within_bounds(X[:, i], variable.get("domain"))
    valid &= region_mask(X, record.get("execution_domain"))
    return prediction, valid


def intersect_schema(record, context):
    """Retain source restrictions in both evaluation and exported inference."""

    def intersection(left, right):
        if left is None:
            return right
        if right is None:
            return left
        low = max((v for v in (left[0], right[0]) if v is not None), default=None)
        high = min((v for v in (left[1], right[1]) if v is not None), default=None)
        result = [low, high]
        validate_bounds(result)
        return result

    variables = deepcopy(context["variables"])
    for variable, source in zip(variables, record.get("variables", variables), strict=True):
        variable["domain"] = intersection(variable.get("domain"), source.get("domain"))
    target = deepcopy(context["target"])
    target["domain"] = intersection(target.get("domain"), record.get("target", target).get("domain"))
    return variables, target


def finite_jacobian(record, X, variables, target, steps):
    """Explicit original-unit central finite differences, never called an analytic derivative."""
    jacobian = np.empty_like(X)
    center, cv = predict_record(record, X, variables, target)
    for i, step in enumerate(steps):
        high, low = X.copy(), X.copy()
        high[:, i] += step
        low[:, i] -= step
        a, av = predict_record(record, high, variables, target)
        b, bv = predict_record(record, low, variables, target)
        jacobian[:, i] = np.where(
            av & bv,
            (a - b) / (2 * step),
            np.where(av & cv, (a - center) / step, np.where(bv & cv, (center - b) / step, np.nan)),
        )
    return jacobian


def weighted_metrics(y, prediction, valid, weights):
    return prediction_metrics(y, prediction, valid=valid, weights=weights)


def group_metrics(y, prediction, valid, weights, groups):
    return {
        str(group): {
            "n_rows": int((groups == group).sum()),
            **weighted_metrics(
                y[groups == group], prediction[groups == group], valid[groups == group], weights[groups == group]
            ),
        }
        for group in np.unique(groups)
    }


def paired_response_evidence(request, ids, y, prediction, valid):
    """Compare stored paired responses, not absolute levels or causal effects."""
    lookup = {key: i for i, key in enumerate(ids)}
    a = np.asarray([lookup[pair[0]] for pair in request["pairs"]])
    b = np.asarray([lookup[pair[1]] for pair in request["pairs"]])
    with np.errstate(over="ignore", invalid="ignore"):
        observed, predicted = y[b] - y[a], prediction[b] - prediction[a]
    pair_valid = valid[a] & valid[b] & np.isfinite(observed) & np.isfinite(predicted)
    return {
        "request_sha256": value_digest(request),
        "data_role": "selection",
        "n_pairs": len(a),
        "n_valid_pairs": int(pair_valid.sum()),
        "n_observations": len(set(a) | set(b)),
        "unit": request["unit"],
        "metrics": prediction_metrics(observed, predicted, valid=pair_valid),
        "observed_difference": [float(v) if np.isfinite(v) else None for v in observed],
        "predicted_difference": [float(v) if np.isfinite(v) else None for v in predicted],
        "valid": pair_valid.tolist(),
        "scope": request["scope"],
        "meaning": "Second minus first at the frozen Selection observations. Shared rows are not independent replicates; no CI, causal or unobserved-path guarantee is inferred.",
    }


def _numerical(context, records, arrays):
    variables, target = context["variables"], context["target"]
    geometry = context.get("input_geometry")
    cfg = {
        **context["config"],
        "constraints": execution_constraints(context["config"]["constraints"], geometry, variables),
    }
    constraints = validation_constraints(
        cfg, variables, target, arrays["train_X"], domain=execution_region(geometry, variables)
    )
    arrays["domain_X"] = np.asarray(
        [[p[v["column"]] for v in variables] for p in constraints.get("domain_probes", [])], dtype=float
    ).reshape(-1, len(variables))
    rows = []
    parameter_evidence_ids = set(context.get("parameter_evidence_ids", []))
    for index, record in enumerate(records):
        row = {"id": record["id"], "kind": record["kind"], "eligible": False, "reason": None}
        row["local_parameter_identifiability"] = {
            "status": "not_computed",
            "reason": "Not requested for this record via parameter_evidence_ids; prediction metrics and eligibility are independent.",
            "data_role": "selection",
        }
        equations = prediction_equations(record)
        row["prediction_available"] = bool(equations)
        row["hypothesis"] = context.get("hypothesis_groups", {}).get(record["id"])
        if record["kind"] == "implicit":
            row["residual_evidence"] = {}
            for partition in ("train", "selection"):
                evidence, residual, valid = residual_evidence(
                    record, arrays[partition + "_X"], arrays[partition + "_y"], variables, target
                )
                row["residual_evidence"][partition] = evidence
                arrays[f"c{index}_{partition}_native_residual"] = residual
                arrays[f"c{index}_{partition}_native_residual_valid"] = valid
        if not equations:
            if record["id"] in parameter_evidence_ids:
                row["local_parameter_identifiability"] = {
                    "status": "unavailable", "reason": "No X-only prediction", "data_role": "selection",
                }
            row["requirements_eligible"] = None
            row["prediction_status"] = "unavailable"
            row["reason"] = record.get(
                "prediction_unavailable_reason", record.get("unavailable_reason", "No X-only prediction")
            )
            row["evaluation_scope"] = (
                "native relation retained; target-prediction question unavailable, not invalidated by that alone"
            )
            row["response_evaluation"] = [{
                "status": "not_evaluated", "reason": row["reason"],
                "point_origin": "current_study",
                "data_role": "response evaluation unavailable: no X-only prediction",
                "independence": "not_asserted",
            }]
            rows.append(row)
            continue
        stats = {
            key: sum(inspect_equation(eq, variables)[key] for eq in equations)
            for key in ("n_nodes", "n_parameters", "n_stored_nodes")
        }
        row["execution"] = {
            "native_arithmetic": sorted({eq.get("arithmetic", "strict_real") for eq in equations}),
            "prediction": "finite real-domain projection; invalid intermediate rows retained",
            "stored_nodes": stats["n_stored_nodes"],
            "expanded_nodes": stats["n_nodes"],
            "node_coordinate_system": "working",
        }
        if record["kind"] == "explicit":
            record["parameters"] = parameter_table(record["equation"])
        checks = []
        for number, equation in enumerate(equations):
            component = check_constraints(
                equation,
                variables,
                target,
                context["config"]["search"]["operators"],
                constraints,
                include_response=False,
                context=context,
            )
            checks.extend([{**c, "member": number} for c in component])
            if record.get("source_constraints"):
                source_checks = check_constraints(
                    equation,
                    record["variables"],
                    record["target"],
                    record["source_operators"],
                    record["source_constraints"],
                    context=record["source_schema"],
                    size_coordinate=record.get("source_size_coordinate", "legacy_working"),
                )
                checks.extend([{**c, "member": number, "name": "source/" + c["name"]} for c in source_checks])

        def view_prediction(points, record=record):
            values, valid = predict_record(record, points, variables, target)
            return np.where(valid, values, np.nan)

        checks.extend(prediction_constraints(view_prediction, variables, target, constraints))
        row["response_evaluation"] = response_evaluation(
            view_prediction, variables, target, constraints, context.get("response_point_origin")
        )
        row["constraint_application"] = constraint_application(record, checks, context)
        row["complexity_scope"] = (
            "sum of source-coordinate AST nodes over retained members; limits checked per member"
            if record["kind"] == "ensemble"
            else "expanded source-coordinate prediction AST"
        )
        row["prediction_view"] = record.get("prediction_view", "explicit equation")
        failures = [c["name"] + ":" + c["status"] for c in checks if c["required"] and c["status"] != "pass"]
        if record["native_check"].get("passed") is False:
            failures.append("native fidelity failed")
        effective_variables, effective_target = intersect_schema(record, context)
        for wanted, effective in zip(variables, effective_variables, strict=True):
            if wanted.get("domain") is not None and effective.get("domain") != wanted["domain"]:
                failures.append("Source input domain is narrower; explicitly restrict the requested study")
        # Required semantics/admissibility are distinct from point-prediction evidence.
        # A coefficient question does not inherit an engineering RMSE threshold.
        row["requirements_eligible"] = not failures
        row["requirement_failures"] = list(failures)
        if record["kind"] == "explicit" and record["equation"].get("template") is not None:
            row["template"] = (
                _template_readout(record["equation"], variables, arrays["selection_X"])
                if not failures
                else {"status": "unavailable", "reason": "; ".join(failures)}
            )
        if record["id"] in parameter_evidence_ids:
            row["local_parameter_identifiability"] = (
                local_parameter_evidence(record, arrays["selection_X"], variables, target)
                if not failures
                else {"status": "unavailable", "reason": "; ".join(failures), "data_role": "selection"}
            )
        metrics = {}
        for partition in ("train", "selection"):
            X, y = arrays[partition + "_X"], arrays[partition + "_y"]
            prediction, valid = predict_record(record, X, variables, target)
            arrays[f"c{index}_{partition}_prediction"] = prediction
            arrays[f"c{index}_{partition}_valid"] = valid
            arrays[f"c{index}_{partition}_residual"] = y - prediction
            if context.get("compute_derivatives", False):
                arrays[f"c{index}_{partition}_jacobian"] = finite_jacobian(
                    record, X, variables, target, context["derivative_steps"]
                )
            metrics[partition] = weighted_metrics(y, prediction, valid, arrays[partition + "_weights"])
            if not valid.all() or metrics[partition]["rmse"] is None:
                failures.append(f"{partition}: invalid coverage")
        if context.get("response_validation") is not None:
            row["response_validation"] = paired_response_evidence(
                context["response_validation"],
                arrays["selection_ids"],
                arrays["selection_y"],
                arrays[f"c{index}_selection_prediction"],
                arrays[f"c{index}_selection_valid"],
            )
        probe_prediction, probe_valid = predict_record(record, arrays["domain_X"], variables, target)
        arrays[f"c{index}_domain_prediction"], arrays[f"c{index}_domain_valid"] = probe_prediction, probe_valid
        for name, points in context["slices"].items():
            X = np.asarray(points["X"])
            prediction, valid = predict_record(record, X, variables, target)
            arrays[f"c{index}_slice_{name}_prediction"] = prediction
            arrays[f"c{index}_slice_{name}_valid"] = valid
        regions = {}
        for name, bounds in context["regions"].items():
            mask = np.ones(len(arrays["selection_X"]), dtype=bool)
            for i, variable in enumerate(variables):
                mask &= within_bounds(arrays["selection_X"][:, i], bounds.get(variable["column"]))
            regions[name] = {
                "n_rows": int(mask.sum()),
                **weighted_metrics(
                    arrays["selection_y"][mask],
                    arrays[f"c{index}_selection_prediction"][mask],
                    arrays[f"c{index}_selection_valid"][mask],
                    arrays["selection_weights"][mask],
                ),
            }
        row.update(
            {
                "eligible": not failures,
                "reason": "; ".join(failures) or None,
                "checks": checks,
                "metrics": metrics,
                "regions": regions,
                "n_nodes": sum(source_complexity(eq, context) for eq in equations),
                "working_n_nodes": stats["n_nodes"],
                "n_parameters": stats["n_parameters"],
                "formula": "Native-weighted structural mean (all particles retained)"
                if record["kind"] == "ensemble"
                else equation_text(equations[0], variables),
            }
        )
        row["applicable_schema"] = {"variables": effective_variables, "target": effective_target}
        if record.get("input_geometry") is not None:
            row["applicable_schema"]["input_geometry"] = record["input_geometry"]
        row["strata"] = {
            partition: group_metrics(
                arrays[partition + "_y"],
                arrays[f"c{index}_{partition}_prediction"],
                arrays[f"c{index}_{partition}_valid"],
                arrays[partition + "_weights"],
                arrays[partition + "_strata"],
            )
            for partition in ("train", "selection")
            if partition + "_strata" in arrays
        }
        row["groups"] = {}
        if "selection_groups" in arrays:
            for group in np.unique(arrays["selection_groups"]):
                mask = arrays["selection_groups"] == group
                row["groups"][str(group)] = {
                    "n_rows": int(mask.sum()),
                    **weighted_metrics(
                        arrays["selection_y"][mask],
                        arrays[f"c{index}_selection_prediction"][mask],
                        arrays[f"c{index}_selection_valid"][mask],
                        arrays["selection_weights"][mask],
                    ),
                }
        row["native_fidelity"] = (
            "not applicable: user-declared equation"
            if record["origin"].get("engine") == "manual"
            else "not applicable: common evaluator reference, no independent native oracle"
            if "shared_evaluator_reference" in record["native_check"].get("kind", "")
            else "verified within recorded native checks"
            if record["native_check"].get("passed") is True
            else "unverified"
            if record["native_check"].get("passed") is None
            else "failed"
        )
        rows.append(row)
    return rows
