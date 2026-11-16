"""Cross-equation differences over frozen evidence. No prediction, fitting or selection."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .data import digest, read_json, value_digest, write_json
from .equation_evaluation.api import load_study, new_directory
from .expression_graph import expanded_equation
from .hypotheses import prediction_equations
from .metrics import _mark_pareto, paired_mse_difference, prediction_metrics
from .question_results import resolved_questions, same_question


def canonical(equation, *, scaffold=False):
    """Syntactic scaffold or frozen expression, never an algebraic/domain equivalence proof."""
    equation = expanded_equation(equation)
    parameters, names = equation["parameters"], {}

    def parameter(name):
        seen = set()
        while parameters[name].get("tied_to"):
            if name in seen:
                raise ValueError("Cyclic tied parameters")
            seen.add(name)
            name = parameters[name]["tied_to"]
        return name

    keys = {}

    def sorting_key(node):
        # Nested repr(list[str]) repeatedly quotes/escapes child strings and
        # grows exponentially with depth. Keep a bounded structural key instead.
        marker = id(node)
        if marker in keys:
            return keys[marker]
        if "param" in node:
            p = parameters[parameter(node["param"])]
            result = ("free",) if scaffold and p["trainable"] else ("const", float(p["value"]))
        elif "const" in node:
            result = ("const", float(node["const"]))
        elif "var" in node:
            result = ("var", node["var"])
        else:
            children = [sorting_key(c) for c in node["args"]]
            result = (node["op"], tuple(sorted(children) if node["op"] in {"add", "mul"} else children))
        keys[marker] = result
        return result

    def visit(node):
        if "param" in node:
            name = parameter(node["param"])
            p = parameters[name]
            if scaffold and p["trainable"]:
                names.setdefault(name, len(names))
                return ["free", names[name]]
            return ["const", float(p["value"])]
        if "const" in node:
            return ["const", float(node["const"])]
        if "var" in node:
            return ["var", node["var"]]
        children = node["args"]
        if node["op"] in {"add", "mul"}:
            children = sorted(children, key=sorting_key)
        return [node["op"], *[visit(c) for c in children]]

    return visit(equation["ast"])


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _analysis_comparisons(analyses):
    """Differences of stored, explicitly matched estimands; no model calls or fitting."""
    results = []
    for analysis in analyses:
        data = analysis["result"]
        if data["outcome"] != "ok":
            continue
        for left, right in combinations([r for r in data.get("results", []) if r["outcome"] == "ok"], 2):
            if data.get("result_kind") == "behavior.effects/v1":
                for a in left["curves"]:
                    for b in right["curves"]:
                        keys = (
                            "feature",
                            "region",
                            "centering",
                            "x",
                            "native_bin_limits",
                            "points_per_bin",
                            "effect_unit",
                        )
                        if any(a[k] != b[k] for k in keys):
                            continue
                        difference = np.asarray(a["effect"]) - np.asarray(b["effect"])
                        results.append(
                            {
                                "axis": "behavior",
                                "left": left["id"],
                                "right": right["id"],
                                "feature": a["feature"],
                                "region": a["region"],
                                "centering": a["centering"],
                                "curve_grid_rms_difference": prediction_metrics(
                                    np.asarray(a["effect"]), np.asarray(b["effect"])
                                )["rmse"],
                                "mean_prediction_difference": left["mean_prediction"] - right["mean_prediction"],
                                "unit": a["effect_unit"],
                                "meaning": "equal-weight stored grid; centered effects and absolute prediction levels kept separate",
                            }
                        )
        for left, right in combinations(resolved_questions(data), 2):
            if left["candidate_id"] == right["candidate_id"] or left.get("kind") != right.get("kind"):
                continue
            kind = left.get("kind")
            if kind == "conditional_interval":
                if left.get("quantity") != right.get("quantity") or not same_question(left, right):
                    continue
                results.append(
                    {
                        "axis": "conditional_quantity",
                        "left": left["candidate_id"],
                        "right": right["candidate_id"],
                        "quantity": left["quantity"],
                        "definition": left.get("definition"),
                        "unit": left.get("unit"),
                        "left_endpoints": left.get("endpoints"),
                        "right_endpoints": right.get("endpoints"),
                        "left_status": left["status"],
                        "right_status": right["status"],
                        "meaning": "same declared conditional definition and assumptions; parameter names do not certify a physical-quantity mapping. Interval overlap is not equivalence or superiority",
                    }
                )
            elif kind == "finite_path_response":
                keys = ("points", "direction", "step", "path_unit", "output_unit", "unit")
                if (
                    left.get("status") != "established"
                    or right.get("status") != "established"
                    or any(left.get(key) != right.get(key) for key in keys)
                ):
                    continue
                difference = np.asarray(left["values"]) - np.asarray(right["values"])
                results.append(
                    {
                        "axis": "feasible_response_difference",
                        "left": left["candidate_id"],
                        "right": right["candidate_id"],
                        "rms_difference": prediction_metrics(np.asarray(left["values"]), np.asarray(right["values"]))[
                            "rmse"
                        ],
                        "max_abs_difference": _finite(np.max(abs(difference))),
                        "n_points": len(difference),
                        "unit": left["unit"],
                        "direction": left["direction"],
                        "meaning": "equal-weight declared feasible path points only; response similarity is not accuracy or causal agreement",
                    }
                )
    return results


def _path_comparisons(analyses, pair_subjects):
    """Compare stored same-point public-coordinate curves; no new predictions."""
    output = []
    for analysis in analyses:
        data = analysis["result"]
        if data.get("result_kind") != "interpretation.readout/v1":
            continue
        for left, right in combinations(data.get("results", []), 2):
            if left["id"] not in pair_subjects or right["id"] not in pair_subjects:
                continue
            for a in left.get("paths", []):
                for b in right.get("paths", []):
                    keys = ("id", "at", "direction", "steps", "path_unit", "input_units", "source_points", "unit")
                    if any(a.get(k) != b.get(k) for k in keys):
                        continue
                    av = np.asarray([np.nan if v is None else v for v in a["prediction"]])
                    bv = np.asarray([np.nan if v is None else v for v in b["prediction"]])
                    valid = np.asarray(a["valid"]) & np.asarray(b["valid"]) & np.isfinite(av) & np.isfinite(bv)
                    difference = av - bv
                    index = int(np.nanargmax(np.where(valid, abs(difference), np.nan))) if valid.any() else None
                    output.append(
                        dict(
                            axis="path_response_difference",
                            left=left["id"],
                            right=right["id"],
                            path_id=a["id"],
                            analysis_sha256=analysis["sha256"],
                            steps=a["steps"],
                            path_unit=a["path_unit"],
                            unit=a["unit"],
                            difference=[float(v) if np.isfinite(v) else None for v in difference],
                            valid=valid.tolist(),
                            common_coverage=float(valid.mean()),
                            maximum_difference=float(abs(difference[index])) if index is not None else None,
                            maximum_point=dict(zip(a["source_columns"], a["source_points"][index], strict=True))
                            if index is not None
                            else None,
                            meaning="Largest valid disagreement among requested feasible grid points only; invalid coverage shown separately. Not an optimum experiment or causal recommendation.",
                        )
                    )
    return output


def compare_equations(study_dir, *, run_dir, evaluations=(), pair_ids=None):
    context, records, rows, arrays = load_study(study_dir)
    known = {record["id"] for record in records}
    if pair_ids is not None and (len(set(pair_ids)) != len(pair_ids) or not set(pair_ids) <= known):
        raise ValueError("Pair subjects must be unique IDs from this immutable study")
    pair_subjects = known if pair_ids is None else set(pair_ids)
    frozen = [canonical(record["equation"]) if record["kind"] == "explicit" else None for record in records]
    directory = new_directory(run_dir)
    study_hash = digest(Path(study_dir) / "study.json")
    analyses = []
    for path in evaluations:
        path = Path(path)
        path = path / "result.json" if path.is_dir() else path
        result = read_json(path)
        if result["study_sha256"] != study_hash:
            raise ValueError("Evaluation context/pool changed; stale evidence cannot be compared")
        analyses.append({"path": str(path.resolve()), "sha256": digest(path), "result": result})
    candidates, families = [], {}
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        metrics = row.get("metrics", {}).get("selection", {})
        item = {
            **row,
            "candidate_id": record["id"],
            "index": index,
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "n_nodes": row.get("n_nodes"),
            "n_parameters": row.get("n_parameters"),
            "family": None,
        }
        if record["kind"] == "explicit":
            family = "syntax-" + value_digest(canonical(record["equation"], scaffold=True))[:12]
            item["family"] = family
            families.setdefault(family, []).append(record["id"])
        candidates.append(item)
    _mark_pareto(candidates)
    bounded_display = pair_ids is None and len(records) > 16
    if bounded_display:
        # Retain every individual candidate. Only pairwise display calculations
        # are bounded; users can explicitly request any other/all subjects.
        ranked = sorted((r for r in candidates if r["rmse"] is not None), key=lambda r: (r["rmse"], r["candidate_id"]))
        selected, seen = [], set()
        for item in ranked:
            family = item["family"] or item["kind"]
            if item["eligible"] and family not in seen:
                selected.append(item["candidate_id"])
                seen.add(family)
            if len(selected) == 3:
                break
        if ranked and ranked[-1]["candidate_id"] not in selected:
            selected.append(ranked[-1]["candidate_id"])
        pair_subjects = set(selected)
    weights = arrays["selection_weights"]
    pairs, pair_arrays = [], {}
    for i, j in combinations([k for k, r in enumerate(records) if r["id"] in pair_subjects], 2):
        if records[i]["id"] not in pair_subjects or records[j]["id"] not in pair_subjects:
            continue
        if not prediction_equations(records[i]) or not prediction_equations(records[j]):
            continue
        a, b = arrays[f"c{i}_selection_prediction"], arrays[f"c{j}_selection_prediction"]
        av, bv = arrays[f"c{i}_selection_valid"], arrays[f"c{j}_selection_valid"]
        valid = av & bv
        complete = bool(valid.all())
        difference = a - b
        pair_key = f"pair_{len(pairs)}"
        pair_arrays[pair_key + "_selection"] = difference
        pair_arrays[pair_key + "_selection_valid"] = valid
        with np.errstate(over="ignore", invalid="ignore"):
            rms = prediction_metrics(a, b, valid=valid, weights=weights)["rmse"] if complete else None
            loss_difference, loss_reason = (
                paired_mse_difference(
                    arrays[f"c{i}_selection_residual"], arrays[f"c{j}_selection_residual"], weights=weights
                )
                if complete
                else (None, "Invalid coverage; no intersection-only loss contrast")
            )
        tolerance = context["engineering_tolerance"]
        ratio, ratio_reason = None, None
        if rms is None:
            ratio_reason = "Prediction RMS difference unavailable"
        elif tolerance is None:
            ratio_reason = "Engineering tolerance not specified"
        else:
            ratio = _finite(rms / tolerance)
            if ratio is None:
                ratio_reason = "Ratio overflow; finite prediction RMS difference retained"
        equivalent = frozen[i] is not None and frozen[i] == frozen[j]
        same_domain = rows[i]["applicable_schema"] == rows[j]["applicable_schema"]
        row = {
            "left": records[i]["id"],
            "right": records[j]["id"],
            "n_rows": len(valid),
            "array_key": pair_key,
            "coverage_left": float(np.average(av, weights=weights)),
            "coverage_right": float(np.average(bv, weights=weights)),
            "common_coverage": float(np.average(valid, weights=weights)),
            "prediction_rms_difference": rms,
            "maximum_prediction_difference": _finite(np.max(abs(difference))) if complete else None,
            "engineering_scale_ratio": ratio,
            "engineering_scale_ratio_reason": ratio_reason,
            "paired_mse_difference": loss_difference,
            "paired_mse_difference_reason": loss_reason,
            "paired_mse_difference_unit": f"({context['target']['unit']})^2",
            "target_unit": context["target"]["unit"],
            "equivalence": "identical frozen AST" if equivalent and same_domain else "inconclusive",
            "equivalence_scope": "identical domains and operator semantics; no automatic cancellation"
            if equivalent and same_domain
            else "No domain-qualified algebraic proof performed",
            "same_scaffold": candidates[i]["family"] == candidates[j]["family"]
            if candidates[i]["family"] is not None and candidates[j]["family"] is not None
            else None,
            "scientific_hypotheses": [context.get("hypothesis_groups", {}).get(records[k]["id"]) for k in (i, j)],
            "reason": "Metric overflow"
            if complete and rms is None
            else None
            if complete
            else "Invalid coverage; no intersection-only score substituted",
        }
        pa, pb = arrays[f"c{i}_domain_prediction"], arrays[f"c{j}_domain_prediction"]
        holes = np.flatnonzero(np.isfinite(pa) != np.isfinite(pb))
        if len(holes):
            row["equivalence"] = "domain counterexample"
            row["equivalence_scope"] = "One frozen expression is undefined at a permitted shared probe"
            row["counterexample_X"] = arrays["domain_X"][int(holes[0])].tolist()
        has_derivatives = all(f"c{k}_selection_jacobian" in arrays for k in (i, j))
        ja = arrays.get(f"c{i}_selection_jacobian")
        jb = arrays.get(f"c{j}_selection_jacobian")
        row["sensitivity_status"] = (
            "computed from stored derivatives" if has_derivatives else "not requested; no derivatives inferred"
        )
        row["sensitivity_difference"] = (
            [
                {
                    "feature": v["column"],
                    "unit": f"{context['target']['unit']}/{v['unit']}",
                    "rms": prediction_metrics(ja[:, k], jb[:, k], weights=weights)["rmse"]
                    if np.isfinite(ja[:, k]).all() and np.isfinite(jb[:, k]).all()
                    else None,
                    "method": context["derivative_method"],
                }
                for k, v in enumerate(context["variables"])
            ]
            if has_derivatives
            else []
        )
        row["slice_differences"] = {}
        for name in context["slices"]:
            key = pair_key + "_slice_" + name
            pair_arrays[key] = arrays[f"c{i}_slice_{name}_prediction"] - arrays[f"c{j}_slice_{name}_prediction"]
            pair_arrays[key + "_valid"] = arrays[f"c{i}_slice_{name}_valid"] & arrays[f"c{j}_slice_{name}_valid"]
            row["slice_differences"][name] = key
        pairs.append(row)
    recurrence = []
    by_id = {r["id"]: r for r in records}
    for family, members in families.items():
        runs = {
            (by_id[m]["origin"].get("engine"), by_id[m]["origin"].get("seed"))
            for m in members
            if by_id[m]["origin"].get("engine") is not None and by_id[m]["origin"].get("seed") is not None
        }
        unknown = sum(
            by_id[m]["origin"].get("engine") is None or by_id[m]["origin"].get("seed") is None for m in members
        )
        recurrence.append(
            {
                "family": family,
                "members": members,
                "distinct_source_runs": len(runs),
                "unknown_source_members": unknown,
                "meaning": "syntactic family occurrence; no posterior probability or mechanism claim",
            }
        )
    result = {
        "schema_version": 1,
        "study": str(Path(study_dir).resolve()),
        "study_sha256": study_hash,
        "purpose": context["purpose"],
        "engineering_tolerance": context["engineering_tolerance"],
        "candidates": candidates,
        "pairs": pairs,
        "pair_scope": {
            "ids": [record["id"] for record in records if record["id"] in pair_subjects],
            "mode": "bounded default display subjects; other pairs uncomputed, not zero"
            if bounded_display
            else "all subjects"
            if pair_ids is None
            else "explicit subjects; other pairs uncomputed, not zero",
            "default_rule": "For pools above16: lowest Selection-error eligible members of up to3 syntax groups plus one highest-error finite candidate. Display computation only, no adoption; explicit pair_ids overrides."
            if bounded_display
            else None,
            "all_candidate_evaluations_retained": True,
        },
        "families": recurrence,
        "analyses": analyses,
        "analysis_comparisons": _analysis_comparisons(analyses) + _path_comparisons(analyses, pair_subjects),
        "interpretation": "Similarity is not accuracy or physical superiority; no hypothesis selected",
    }
    np.savez_compressed(directory / "pair_arrays.npz", **pair_arrays)
    result["pair_arrays_sha256"] = digest(directory / "pair_arrays.npz")
    result["unavailable_prediction_comparisons"] = [
        {
            "left": a["id"],
            "right": b["id"],
            "reason": "At least one native output has no X-only prediction; residuals are not target losses",
        }
        for a, b in combinations([r for r in records if r["id"] in pair_subjects], 2)
        if not prediction_equations(a) or not prediction_equations(b)
    ]
    result["scientific_hypotheses"] = context.get("hypothesis_groups", {})
    representatives = []
    for members in families.values():
        members = [row for row in candidates if row["id"] in members and row["eligible"]]
        if members:
            representatives.append(min(members, key=lambda r: (r["n_nodes"], r["rmse"], r["id"]))["id"])
    annotated = {}
    for row in candidates:
        annotation = context.get("hypothesis_groups", {}).get(row["id"])
        if annotation and row["eligible"]:
            annotated.setdefault(annotation["label"], []).append(row)
    if annotated:
        representatives = [min(items, key=lambda r: (r["rmse"], r["id"]))["id"] for items in annotated.values()]
    representatives += [
        r["id"] for r in candidates if r["eligible"] and r["kind"] != "explicit" and r["id"] not in representatives
    ]
    result["representative_ids"] = representatives
    result["representative_rule"] = (
        "user-labeled hypothesis: lowest Selection error display member; otherwise smallest eligible syntax member; non-explicit prediction views retained. Display only, not adoption or proof of mechanism"
    )
    write_json(directory / "comparison.json", result)
    return directory
