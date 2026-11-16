"""Example explicit evaluator extension: response along a feasible physical path.

Installed/imported intentionally by the caller, never unpickled from a model.
No selection, fitting, Final reads or physical sign assertions live here.
"""

import numpy as np
from ml_platform_symbolic.coordinates import map_input_values
from ml_platform_symbolic.equation_evaluation.numerical import predict_record
from ml_platform_symbolic.units import convert_width


def run(context, records, rows, arrays, directory, options):
    geometry = context.get("input_geometry")
    if not geometry:
        return {
            "outcome": "unavailable",
            "results": [],
            "reason": "This source-path example needs explicit input geometry",
            "result_kind": "behavior.feasible_path/v1",
        }
    columns = [v["column"] for v in geometry["source_variables"]]
    if set(options["direction"]) != set(columns):
        raise ValueError("Path direction names all source columns, including zeros")
    center = np.asarray([[p[c] for c in columns] for p in options["points"]], float)
    direction = np.asarray([options["direction"][c] for c in columns], float)
    step = float(options["step"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Positive finite path step required")
    points = [map_input_values(center + amount * step * direction, geometry) for amount in (-1, 0, 1)]
    results = []
    for record, row in zip(records, rows, strict=True):
        if options.get("ids") and record["id"] not in options["ids"]:
            continue
        if not row["prediction_available"]:
            results.append(
                {
                    "id": record["id"],
                    "outcome": "unavailable",
                    "reason": "Residual-only view has no target response",
                    "questions": [],
                }
            )
            continue
        values = [predict_record(record, matrix, context["variables"], context["target"]) for matrix in points]
        valid = np.logical_and.reduce([pair[1] for pair in values])
        response = (values[2][0] - values[0][0]) / (2 * step)
        response = convert_width(response, context["target"].get("unit"), options["output_unit"])
        result = {
            "kind": "finite_path_response",
            "status": "established" if valid.all() else "undetermined",
            "definition": "symmetric finite-step response along declared feasible source path; model property, not causal effect or global derivative",
            "points": options["points"],
            "direction": options["direction"],
            "step": step,
            "path_unit": options["path_unit"],
            "output_unit": options["output_unit"],
            "values": [float(v) if ok else None for v, ok in zip(response, valid, strict=True)],
            "unit": f"({options['output_unit']})/({options['path_unit']})",
            "n_valid": int(valid.sum()),
            "n_points": len(valid),
        }
        results.append({"id": record["id"], "outcome": "ok" if valid.all() else "unavailable", "questions": [result]})
    return {
        "outcome": "ok" if any(r["outcome"] == "ok" for r in results) else "unavailable",
        "result_kind": "behavior.feasible_path/v1",
        "results": results,
        "assumptions": "Caller supplied original working-coordinate points and a feasible replacement path, output units and explicit finite step. No fit or sign hypothesis.",
    }
