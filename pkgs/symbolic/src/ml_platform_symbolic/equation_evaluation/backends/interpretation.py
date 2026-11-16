"""Explicit interpretation computations on frozen hypotheses and permitted rows.

Mathematical/representation summaries are not causal or mechanistic explanations.
The renderer consumes these values; it never executes the hypotheses itself.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ...constraints import equation_unit
from ...coordinates import source_input_values
from ...data import input_variables, model_matrix, model_output, public_unit
from ...expression_graph import expanded_equation
from ...expressions import equation_text, evaluate
from ...hypotheses import prediction_equations
from ...units import coherent_unit_label
from ..numerical import predict_record, prerequisite_failure


def _source_public(X, schema):
    source = source_input_values(X, schema.get("input_geometry"))
    return np.column_stack(
        [model_output(source[:, i], {**schema, "target": v}) for i, v in enumerate(input_variables(schema))]
    )


def input_reference_geometry(reference, points, columns):
    """Descriptive input distances only, without targets, model calls or OOD cutoffs."""
    reference, points = np.asarray(reference, dtype=float), np.asarray(points, dtype=float)
    if (
        reference.ndim != 2 or points.ndim != 2 or not len(reference)
        or reference.shape[1] != len(columns) or points.shape[1] != len(columns)
        or not np.isfinite(reference).all() or not np.isfinite(points).all()
    ):
        raise ValueError("Input geometry requires finite matrices in the same public column order")
    lower, upper = reference.min(axis=0), reference.max(axis=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        span = upper - lower
        varying = span > 0
        scaled_reference = (reference[:, varying] - lower[varying]) / span[varying]
        scaled_points = (points[:, varying] - lower[varying]) / span[varying]
    evidence = {
        "data_role": "train", "coordinates": "public source inputs", "n_reference_rows": len(reference),
        "ranges": [{"column": c, "min": float(lo), "max": float(hi), "constant": bool(lo == hi)}
                   for c, lo, hi in zip(columns, lower, upper, strict=True)],
        "outside_marginal_range": ((points < lower) | (points > upper)).any(axis=1).tolist(),
        "constant_feature_changed": (points[:, ~varying] != lower[~varying]).any(axis=1).tolist(),
        "method": "Euclidean nearest Train input distance after dividing each varying column by its Train range",
        "scope": (
            "Descriptive geometry, not a support, accuracy, safety or causal guarantee. "
            "Inside every marginal range can still leave the joint observed design. "
            "Distance depends on coordinates, feature count and sampling density; no OOD threshold is fitted. "
            "Constant columns are omitted from distances and changes are flagged separately. "
            "Identical input rows are counted once for reference neighbor spacing, not as independent observations."
        ),
    }
    if not (np.isfinite(span).all() and np.isfinite(scaled_reference).all() and np.isfinite(scaled_points).all()):
        return {**evidence, "status": "unavailable", "reason": "Train-range scaling is not numerically representable"}
    designs = np.unique(scaled_reference, axis=0)
    originals = np.unique(reference, axis=0)
    evidence["n_distinct_designs"] = len(originals)
    if len(designs) != evidence["n_distinct_designs"]:
        return {**evidence, "status": "unavailable", "reason": "Train-range scaling collapses distinct input designs"}
    spacing = None
    if varying.any():
        tree = cKDTree(designs)
        distances, indices = tree.query(scaled_points, k=1)
        # A computed zero must not conceal distinct representable input coordinates.
        if any(not np.array_equal(point[varying], originals[index, varying])
               for point, index, distance in zip(points, indices, distances, strict=True) if distance == 0):
            return {**evidence, "status": "unavailable", "reason": "Distance scaling loses a nonzero input displacement"}
        if len(designs) > 1:
            nearest_other = tree.query(designs, k=2)[0][:, 1]
            if (nearest_other <= 0).any():
                return {**evidence, "status": "unavailable", "reason": "Reference neighbor spacing loses a nonzero input displacement"}
            spacing = {"median": float(np.median(nearest_other)), "p95": float(np.quantile(nearest_other, 0.95))}
    else:
        distances = np.zeros(len(points))
    if not np.isfinite(distances).all():
        return {**evidence, "status": "unavailable", "reason": "Nearest input distance is not numerically representable"}
    return {
        **evidence, "status": "computed", "nearest_distance": distances.tolist(),
        "reference_neighbor_spacing": spacing,
        "n_outside_marginal_range": sum(evidence["outside_marginal_range"]),
        "n_constant_feature_changed": sum(evidence["constant_feature_changed"]),
        "max_nearest_distance": float(distances.max()) if len(distances) else None,
    }


def _paths(context, options, schema):
    columns = [v["column"] for v in input_variables(schema)]
    units = {v["column"]: public_unit(schema, v) for v in input_variables(schema)}
    paths, names = [], set()
    for spec in options.get("paths", []):
        name = spec.get("id")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Interpretation paths need unique nonempty string IDs within one analysis")
        names.add(name)
        if (
            set(spec) - {"id", "at", "direction", "steps", "path_unit", "scope"}
            or set(spec.get("at", {})) != set(columns)
            or set(spec.get("direction", {})) != set(columns)
        ):
            raise ValueError("Path fixes every source column, direction, steps, units and assumptions")
        steps = np.asarray(spec["steps"], dtype=float)
        if (
            steps.ndim != 1
            or not len(steps)
            or not np.isfinite(steps).all()
            or len(steps) > 301
            or not spec.get("path_unit")
        ):
            raise ValueError("Path needs1..301finite steps and a declared parameter unit")
        points = np.array([[spec["at"][c] + t * spec["direction"][c] for c in columns] for t in steps])
        if not np.isfinite(points).all():
            raise ValueError("Path inputs are not finite")
        X = model_matrix(pd.DataFrame(points, columns=columns), schema)
        paths.append(
            (
                {
                    **deepcopy(spec),
                    "input_units": units,
                    "step_widths": np.diff(steps).tolist(),
                    "source_points": points.tolist(),
                    "source_columns": columns,
                    "scope": spec.get(
                        "scope",
                        "User-declared source path; domain membership checked, joint observational support and causal meaning not certified",
                    ),
                },
                X,
            )
        )
    return paths


def _parts(node):
    if node.get("op") in {"add", "sub"}:
        left, right = node["args"]
        terms = _parts(left)
        rest = _parts(right)
        return terms + (rest if node["op"] == "add" else [{"op": "neg", "args": [v]} for v in rest])
    return [node]


def _mathematics(equation):
    counts, parameters = {}, equation["parameters"]

    def visit(node):
        if "op" in node:
            counts[node["op"]] = counts.get(node["op"], 0) + 1
            for child in node.get("args", []):
                visit(child)

    visit(equation["ast"])
    notes = ["Top-level mathematical form: " + equation["ast"].get("op", "literal/variable")]
    notes.append("Operators in subexpressions: " + ", ".join(f"{name}×{n}" for name, n in sorted(counts.items())))
    if "sin" in counts or "cos" in counts:
        notes.append(
            "Trigonometric subexpressions do not establish periodicity of the whole relation or a physical mechanism"
        )
    if "div" in counts:
        notes.append(
            "Division requires its denominators to be defined/nonzero; finite sampled checks are not a global guarantee"
        )
    if any(p.get("unit") is None for p in parameters.values()):
        notes.append(
            "Some coefficient units are unknown: physical parameter interpretation is unavailable for those coefficients"
        )
    return notes


def _term_values(equation, record, context, arrays, schema, reference):
    expanded = expanded_equation(equation)
    nodes = _parts(expanded["ast"])
    additive = len(nodes) > 1
    if not additive and expanded["ast"].get("op") == "div":
        nodes = expanded["ast"]["args"]
        names = ["numerator", "denominator"]
        scope = "Numerator/denominator values in native coordinates; no additive contribution allocation"
    else:
        names = [f"term{i + 1}" for i in range(len(nodes))]
        scope = (
            "Values of additive terms in THIS expression only"
            if additive
            else "Whole nested expression only; additive decomposition is unavailable"
        )
    terms = []
    X = np.vstack([reference, arrays["selection_X"]])
    # Subexpression values do not authorize execution outside the full relation's
    # source/context domain intersection. Reuse the same numerical authority.
    _, admissible = predict_record(record, X, context["variables"], context["target"])
    for name, node in zip(names, nodes, strict=True):
        part = {**expanded, "ast": node}
        values = evaluate(part, X, context["variables"])
        spec, errors = equation_unit(part, context["variables"])
        label = coherent_unit_label(spec) if spec is not None and not errors else "unknown / unverified dimensions"
        output_part = additive or len(nodes) == 1
        converted = model_output(values, schema, difference=additive) if output_part else values
        finite = np.isfinite(converted) & admissible
        terms.append(
            dict(
                name=name,
                formula=equation_text(
                    part, [{**v, "column": v["column"] + "_work"} for v in context["variables"]], parameter_names=True
                ),
                formula_coordinates="working",
                reference_value=float(converted[0]) if finite[0] else None,
                min=float(np.min(converted[1:])) if finite[1:].all() else None,
                max=float(np.max(converted[1:])) if finite[1:].all() else None,
                n_valid=int(finite[1:].sum()),
                n_rows=len(arrays["selection_X"]),
                unit=f"{'additive difference-scale' if additive else 'whole output coordinate'} {public_unit(schema, context['target'])}; inferred native term unit {label}"
                if output_part
                else label,
                meaning="Representation-dependent numerical term, not physical importance. Unknown dimensional compatibility does not authorize a physical contribution.",
            )
        )
    match = None
    if additive:
        total = evaluate(expanded, X, context["variables"])
        summed = sum(evaluate({**expanded, "ast": n}, X, context["variables"]) for n in nodes)
        valid = np.isfinite(total) & np.isfinite(summed) & admissible
        match = dict(
            n_valid=int(valid.sum()),
            n_rows=len(X),
            max_roundoff_difference=float(np.max(abs(total[valid] - summed[valid]))) if valid.any() else None,
            meaning="This syntactic sum reconstructs the same expression at these points; not unique/causal contributions",
        )
    return terms, scope, match


def run(context, records, rows, arrays, directory, options):
    schema = {
        key: context[key] for key in ("variables", "target", "unit_coordinates", "input_geometry") if key in context
    }
    paths = _paths(context, options, schema)
    columns = [v["column"] for v in input_variables(schema)]
    train_public = _source_public(arrays["train_X"], schema)
    for spec, _ in paths:
        spec["reference_geometry"] = input_reference_geometry(train_public, spec["source_points"], columns)
    source = _source_public(arrays["train_X"][:1], schema)[0]
    reference_public = options.get("reference", dict(zip(columns, source.tolist(), strict=True)))
    if set(reference_public) != set(columns):
        raise ValueError("Reference must fix every original column")
    reference = model_matrix(pd.DataFrame([reference_public]), schema)
    results = []
    for record, row in zip(records, rows, strict=True):
        if options.get("ids") and record["id"] not in options["ids"]:
            continue
        failure = prerequisite_failure(record, row)
        if failure:
            results.append(
                dict(id=record["id"], outcome="unavailable", reason=failure, mathematics=[], terms=[], paths=[])
            )
            continue
        equations = prediction_equations(record)
        item = dict(
            id=record["id"],
            outcome="ok",
            mathematics=[],
            terms=[],
            paths=[],
            reference=reference_public,
            reference_scope="Explicit point or first actually observed Train row, not a measured zero baseline",
            identifiability="Parameter/structure identifiability not established by prediction error. Conditional profiles, if provided separately, retain their own assumptions.",
        )
        if record["kind"] == "implicit":
            item["mathematics"] = [
                "Implicit R(X,y)=0; residual scale is arbitrary and not a cross-equation target loss"
            ]
        if record["kind"] == "explicit":
            item["mathematics"] = _mathematics(expanded_equation(equations[0]))
            item["terms"], item["decomposition_scope"], item["term_sum_check"] = _term_values(
                equations[0], record, context, arrays, schema, reference
            )
        elif record["kind"] == "ensemble":
            item["mathematics"] = [
                "Native-weighted formula view; individual members/weights/duplicates remain authoritative",
                "A structural empirical quantile is not an observation interval or mechanism probability",
            ]
            item["decomposition_scope"] = "No additive attribution across different member formulas"
        if not equations:
            item.update(
                outcome="unavailable",
                reason=record.get(
                    "prediction_unavailable_reason", record.get("unavailable_reason", "No target-free prediction")
                ),
                decomposition_scope="Native relation retained; no invented target prediction",
            )
        else:
            for spec, X in paths:
                prediction, valid = predict_record(record, X, context["variables"], context["target"])
                public = model_output(prediction, schema)
                item["paths"].append(
                    {
                        **spec,
                        "prediction": [float(v) if np.isfinite(v) else None for v in public],
                        "valid": valid.tolist(),
                        "coverage": float(valid.mean()),
                        "unit": public_unit(schema, context["target"]),
                    }
                )
        results.append(item)
    return dict(
        outcome="ok" if any(r["outcome"] == "ok" for r in results) else "unavailable",
        result_kind="interpretation.readout/v1",
        results=results,
        data_role="Train reference / Selection term distribution / explicitly requested feasible paths; no Final",
        options=options,
        meaning="Mathematical/model-property evidence only. Representation-dependent parts, no inferred mechanism or causal attribution",
    )
