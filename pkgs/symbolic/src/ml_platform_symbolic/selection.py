"""Purpose-aware recorded choice. No model evaluation or coefficient fitting."""

import math
from pathlib import Path

from .data import digest, frozen_labels, read_json, value_digest, write_json
from .equation_evaluation.api import final_exposure, load_study, new_directory
from .equation_evaluation.numerical import intersect_schema
from .problem import problem_document
from .question_results import missing_precision, resolved_questions


def select_representatives(records, rows):
    """Minimum eligible finite Selection RMSE per method/seed; exact ties use ID.

    This is a comparison representative, not deployment adoption or Final evidence.
    Input records, invalid predictions and failed runs remain unchanged.
    """
    selected = {}
    for record, row in zip(records, rows, strict=True):
        rmse = row.get("metrics", {}).get("selection", {}).get("rmse")
        if not row.get("eligible") or rmse is None or not math.isfinite(rmse):
            continue
        origin = record.get("origin", {})
        key = (origin.get("engine"), origin.get("seed"))
        previous = selected.get(key)
        if previous is None or (rmse, record["id"]) < (previous[1]["metrics"]["selection"]["rmse"], previous[0]["id"]):
            selected[key] = (record, row)
    return selected


def prediction_support(metrics, *, tolerance, coverage=1.0):
    """Match a predeclared prediction criterion, never select/refit on audit data."""
    if tolerance is None or metrics is None:
        established, reason = (
            None,
            "No predeclared prediction tolerance or no audit; independent support not established",
        )
    elif metrics.get("rmse") is None or coverage < 1:
        established, reason = False, "Incomplete/invalid predictions do not satisfy the frozen prediction criterion"
    else:
        established = metrics["rmse"] <= tolerance
        reason = (
            "Frozen prediction error criterion met on these rows"
            if established
            else "Audit error exceeds the already predeclared prediction tolerance"
        )
    return {
        "established": established,
        "reason": reason,
        "criteria": {"rmse_max": tolerance, "minimum_valid_coverage": 1.0},
        "observed_rmse": None if metrics is None else metrics.get("rmse"),
        "valid_coverage": coverage,
        "scope": "This frozen target predictor and audit rows only; not coefficient/structure identification, future shift or interval coverage. No reselection.",
    }


def response_support(evidence, request):
    """Match paired-response evidence to its frozen purpose; no prediction calls."""
    if request is None:
        established, reason = None, "No predeclared observed-pair response validation"
    elif not evidence or evidence.get("request_sha256") != value_digest(request):
        established, reason = None, "Matching observed-pair response evidence is unavailable"
    elif evidence.get("n_valid_pairs") != evidence.get("n_pairs") or evidence["metrics"]["rmse"] is None:
        established, reason = False, "Incomplete/invalid paired responses"
    else:
        established = evidence["metrics"]["rmse"] <= request["max_rmse"]
        reason = (
            "Frozen paired-response error criterion met"
            if established
            else "Paired-response RMSE exceeds the frozen tolerance"
        )
    return {
        "established": established,
        "reason": reason,
        "scope": "Specified Selection observation pairs only. Not absolute prediction accuracy, causality, a confidence interval or an unobserved response path.",
    }


def assess_audit_support(audit_dir, decision_dir, *, run_dir):
    """Apply already frozen criteria to STORED audit results; never reopen Final.

    This records audit support separately from the unchanged Selection/use choice.
    It cannot choose another candidate or create new predictions.
    """
    path = Path(audit_dir) / "audit.json"
    audit = read_json(path)
    decision_path = Path(decision_dir) / "decision.json"
    decision = read_json(decision_path)
    pool_path = Path(audit_dir) / "frozen_pool.json"
    pool = read_json(pool_path)
    if (
        digest(pool_path) != audit["frozen_pool_sha256"]
        or pool["study_sha256"] != decision["study_sha256"]
        or audit["study_sha256"] != decision["study_sha256"]
    ):
        raise ValueError("Audit pool/study differs from the displayed choice")
    if pool["decision_sha256"] not in {digest(decision_path), decision.get("frozen_choice_sha256")}:
        raise ValueError("Audit was authorized by a different frozen decision")
    if digest(Path(audit_dir) / "predictions.npz") != audit["predictions_sha256"]:
        raise ValueError("Audit numerical evidence changed")
    tolerance = decision.get("criteria", {}).get("engineering_tolerance")
    results = []
    for row in audit["results"]:
        results.append(
            {
                **row,
                "prediction_support": prediction_support(
                    row.get("target_metrics"), tolerance=tolerance, coverage=row.get("coverage", 0)
                )
                if decision["purpose"] != "parameter_interpretation"
                else None,
            }
        )
    directory = new_directory(run_dir)
    write_json(
        directory / "audit_support.json",
        {
            "schema_version": 1,
            "study_sha256": decision["study_sha256"],
            "decision_sha256": digest(decision_path),
            "subject": decision.get("subject"),
            "audit_path": str(path.resolve()),
            "audit_sha256": digest(path),
            "results": results,
            "selection_changed": False,
            "scope": "Stored Final audit matched to predeclared criteria; exposed-data review, not reselection or new independent confirmation",
        },
    )
    return directory


def choose_hypothesis(comparison_dir, *, run_dir, candidate_id=None, reason, retain=(), calibration_plan=None):
    directory = new_directory(run_dir)
    comparison_path = Path(comparison_dir) / "comparison.json"
    comparison = read_json(comparison_path)
    context, records, _, _ = load_study(comparison["study"])
    if digest(Path(comparison["study"]) / "study.json") != comparison["study_sha256"]:
        raise ValueError("Comparison evidence changed")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A scientific/user rationale is required")
    candidates = {r["id"]: r for r in comparison["candidates"]}
    if not set(retain) <= candidates.keys() or (candidate_id is not None and candidate_id not in candidates):
        raise ValueError("Choice/retained IDs must belong to the compared pool")
    selected, missing = None, []
    if candidate_id is not None:
        selected = candidates[candidate_id]
        purpose = context["purpose"]
        if purpose not in {"local_prediction", "response_comparison", "parameter_interpretation"}:
            raise ValueError("Purpose must be an explicit supported scientific use")
        if purpose == "parameter_interpretation":
            # Historical rows lack this separate prerequisite evidence. Do not
            # reinterpret their prediction eligibility as a physical certificate.
            admissible = selected.get("requirements_eligible")
            if admissible is False:
                raise ValueError("Required semantics/constraints exclude this coefficient interpretation")
            if admissible is None:
                missing.append("Required relation admissibility has not been evaluated for this purpose")
        elif purpose == "local_prediction":
            if not selected["eligible"]:
                raise ValueError("Cannot adopt a computationally invalid or required-constraint-ineligible predictor")
            tolerance = context["engineering_tolerance"]
            if tolerance is None:
                missing.append("No predeclared original-unit engineering tolerance")
            elif prediction_support({"rmse": selected["rmse"]}, tolerance=tolerance)["established"] is False:
                missing.append("Validation RMSE exceeds predeclared engineering tolerance")
        elif purpose == "response_comparison":
            if selected.get("requirements_eligible") is False:
                raise ValueError("Required semantics/constraints exclude this response interpretation")
            if selected.get("requirements_eligible") is None:
                missing.append("Required relation admissibility has not been evaluated for response use")
            support = response_support(selected.get("response_validation"), context.get("response_validation"))
            if support["established"] is not True:
                missing.append(support["reason"])
        for method in context["required_evidence"]:
            available = any(
                a["result"]["method"] == method
                and a["result"]["outcome"] == "ok"
                and any(
                    r.get("id") == candidate_id and r.get("outcome") == "ok" for r in a["result"].get("results", [])
                )
                for a in comparison["analyses"]
            )
            if not available:
                missing.append(f"Required evidence not established: {method}")
        if context["purpose"] == "parameter_interpretation":
            quantities = context.get("quantity_tolerances", {})
            if not quantities:
                missing.append("No predeclared physical-quantity precision requirement")
            missing.extend(
                missing_precision(
                    comparison["analyses"],
                    candidate_id,
                    quantities,
                    estimand=context.get("parameter_target", "conditional_model"),
                )
            )
    if calibration_plan is not None:
        if context["purpose"] != "local_prediction":
            raise ValueError("Predictive calibration is a separate local_prediction-purpose choice")
        if selected is None or set(calibration_plan) != {
            "alpha",
            "exchangeability",
            "group_column",
            "observation_model",
            "max_width",
            "unit",
        }:
            raise ValueError(
                "Calibration plan fixes a predictor, alpha, independent sampling unit, assumptions and allowed full width/unit"
            )
        if (
            not 0 < calibration_plan["alpha"] < 1
            or not math.isfinite(calibration_plan["max_width"])
            or calibration_plan["max_width"] <= 0
        ):
            raise ValueError("Calibration alpha and full-width criterion must be positive and finite")
    decision = {
        "schema_version": 1,
        "status": "frozen_prediction"
        if calibration_plan is not None
        else "adopt"
        if selected is not None and not missing
        else "defer",
        "calibration_plan": calibration_plan,
        "purpose": context["purpose"],
        "parameter_target": context.get("parameter_target", "conditional_model"),
        "reason": reason,
        "missing_support": missing,
        "selected_id": candidate_id,
        "retain": list(retain),
        "comparison_sha256": digest(comparison_path),
        "comparison_path": str(comparison_path.resolve()),
        "study_sha256": comparison["study_sha256"],
        "data_role": "selection",
        "mode": context["mode"],
        "final_exposure_at_choice": context["final_exposure"],
        "declared_final_history": context.get("final_history", "unknown"),
    }
    if selected is not None:
        record = next(r for r in records if r["id"] == candidate_id)
        decision["subject"] = record.get("subject")
        decision["output_kind"] = record["kind"]
        decision["applicability_schema"] = selected.get("applicable_schema")
        decision["criteria"] = {
            key: context[key]
            for key in ("engineering_tolerance", "purpose", "required_evidence", "quantity_tolerances")
        }
        if context["purpose"] == "response_comparison":
            decision["criteria"]["response_checks"] = context["config"]["constraints"]["response_checks"]
            decision["criteria"].pop("engineering_tolerance")
            decision["criteria"]["response_validation"] = context.get("response_validation")
            decision["response_evidence"] = selected.get("response_validation")
            decision["response_support"] = response_support(
                selected.get("response_validation"), context.get("response_validation")
            )
            decision["use_scope"] = (
                "Paired-response analysis for the frozen observations/criterion only. Absolute level prediction and Final support were not adopted."
            )
        if context["purpose"] == "parameter_interpretation":
            decision["criteria"].pop("engineering_tolerance")
            decision["criteria"]["parameter_target"] = context.get("parameter_target", "conditional_model")
            decision["quantity_evidence"] = [
                {"analysis_path": a["path"], "analysis_sha256": a["sha256"], "question": q}
                for a in comparison["analyses"]
                for q in resolved_questions(a["result"])
                if q["candidate_id"] == candidate_id
                and q.get("quantity") in context.get("quantity_tolerances", {})
                and q.get("estimand") == context.get("parameter_target", "conditional_model")
            ]
        if record["kind"] == "explicit":
            decision["equation"] = record["equation"]
    write_json(directory / "decision.json", decision)
    return directory


def finalize_hypothesis(study_dir, decision_dir, *, run_dir):
    """Audit one frozen prediction view; never choose, refit, or flatten native outputs."""
    import json

    import numpy as np
    import pandas as pd
    from ml_platform_core.result import RunResult

    from .data import load_prepared
    from .equation_evaluation.numerical import predict_record, weighted_metrics
    from .hypotheses import save_prediction
    from .model import save_model

    context, records, rows, _ = load_study(study_dir)
    decision_path = Path(decision_dir) / "decision.json"
    decision = read_json(decision_path)
    if decision["status"] != "adopt" or decision["study_sha256"] != digest(Path(study_dir) / "study.json"):
        raise ValueError("Final requires an adopted choice bound to this study")
    if digest(decision["comparison_path"]) != decision["comparison_sha256"]:
        raise ValueError("Decision comparison evidence changed")
    if decision.get("interval_sha256"):
        from .prediction_intervals import read_interval_evidence

        if not decision.get("interval_path"):
            raise ValueError("Provide the calibration artifact assessed by this use decision")
        read_interval_evidence(decision["interval_path"], decision=decision, decision_sha256=digest(decision_path))
    index = next(i for i, r in enumerate(records) if r["id"] == decision["selected_id"])
    record, row = records[index], rows[index]
    if decision.get("subject") != record.get("subject"):
        raise ValueError("Decision execution view/state differs from its evaluated subject")
    if decision["purpose"] == "response_comparison":
        if (
            row.get("requirements_eligible") is not True
            or not decision.get("response_support", {}).get("established")
            or decision.get("response_evidence") != row.get("response_validation")
        ):
            raise ValueError(
                "Response decision lacks its frozen matching response evidence; historical prediction decisions are not reinterpreted"
            )
        directory = new_directory(run_dir)
        result = write_json(
            directory / "response_result.json",
            {
                "schema_version": 1,
                "kind": "paired_response_analysis",
                "record": record,
                "decision": decision,
                "decision_sha256": digest(decision_path),
                "response_evidence": row["response_validation"],
                "final_accessed": False,
                "scope": decision["use_scope"],
            },
        )
        return RunResult(
            directory,
            {},
            {"response_result": result},
            {},
            extra={"purpose": decision["purpose"], "final_accessed": False},
        )
    if decision["purpose"] == "parameter_interpretation":
        # A coefficient-analysis deliverable is not a deployment predictor. It
        # preserves the exact fit/question references and never opens Final.
        if row.get("requirements_eligible") is not True or not decision.get("quantity_evidence"):
            raise ValueError("Adopted coefficient use lacks its evaluated prerequisites/questions")
        for evidence in decision["quantity_evidence"]:
            if digest(evidence["analysis_path"]) != evidence["analysis_sha256"]:
                raise ValueError("Coefficient evidence changed")
            question = evidence["question"]
            if question.get("fit_artifact") and digest(question["fit_artifact"]) != question.get("fit_sha256"):
                raise ValueError("Named coefficient fit changed")
        directory = new_directory(run_dir)
        result = write_json(
            directory / "coefficient_result.json",
            {
                "schema_version": 1,
                "kind": "coefficient_analysis",
                "record": record,
                "decision": decision,
                "decision_sha256": digest(decision_path),
                "quantities": decision["quantity_evidence"],
                "final_accessed": False,
                "scope": "Named conditional/latent estimand only as established by its evidence; not a saved prediction or universal physical identification",
            },
        )
        return RunResult(
            directory,
            {},
            {"coefficient_result": result},
            {},
            extra={"purpose": decision["purpose"], "final_accessed": False},
        )
    if not row["eligible"] or (record["kind"] == "explicit" and decision["equation"] != record["equation"]):
        raise ValueError("Frozen choice differs from its evaluated hypothesis")
    directory = new_directory(run_dir)
    schema = dict(zip(("variables", "target"), intersect_schema(record, context), strict=True))
    if record.get("input_geometry") is not None:
        schema["input_geometry"] = record["input_geometry"]
    if decision.get("applicability_schema", schema) != schema:
        raise ValueError("Decision applicability differs from evaluated source/context intersection")
    schema.update(
        feature_ranges=context["feature_ranges"],
        unit_coordinates=context.get("unit_coordinates"),
        public_units=context["config"]["data"].get("source_units", {}),
    )
    provenance = read_json(context["prepared"]["run"])
    provenance.update(problem_document(context["config"]))
    provenance["evaluation_constraint_application"] = context["constraint_application"]
    exposure = final_exposure(context["prepared"], provenance, records, history=context.get("final_history", "unknown"))
    marker = Path(context["prepared"]["run"]).parent.parent / "final_access.json"
    independent = (
        context["final_history"] == "declared_unseen"
        and context["mode"] != "historical"
        and not exposure["same_data_exposed"]
        and not marker.exists()
    )
    role = (
        "holdout under explicit user declaration"
        if independent
        else "post-selection audit; prior exposure not excluded"
    )
    frozen = write_json(
        directory / "frozen_prediction.json",
        {"record": record, "schema": schema, "decision": decision, "decision_sha256": digest(decision_path)},
    )
    metrics, tables = None, {}
    if "final" in context["prepared"]:
        if not marker.exists():
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "data_sha256": context["data_sha256"],
                        "split_sha256": context["split_sha256"],
                        "decision_sha256": digest(decision_path),
                        "role": role,
                    },
                    stream,
                    indent=2,
                )
        X, y, ids = load_prepared(context["prepared"], provenance, "final")
        prediction, valid = predict_record(record, X, context["variables"], context["target"])
        metrics = weighted_metrics(y, prediction, valid, np.ones(len(y)))
        metrics["coverage"] = float(valid.mean())
        table = directory / "predictions.csv"
        pd.DataFrame({"sample_id": ids, "observed": y, "prediction": prediction, "valid": valid}).to_csv(
            table, index=False
        )
        tables["final_predictions"] = table
    metadata = {
        "selected_id": record["id"],
        "origin": record.get("origin", {}),
        "output_kind": record["kind"],
        "decision": decision,
        "provenance": provenance,
        "final_metrics": metrics,
        "prediction_support": prediction_support(
            metrics,
            tolerance=decision["criteria"]["engineering_tolerance"],
            coverage=metrics.get("coverage", 1.0) if metrics else 1.0,
        ),
        "decision_scope": "The embedded frozen decision used Selection/calibration. Independent Final support is recorded separately; an adopt flag is not a universal deployment approval.",
        "final_evidence_role": role,
        "selected_native_check": record["native_check"],
        "frozen_prediction_sha256": digest(frozen),
        "evaluation_constraint_application": context["constraint_application"],
    }
    if record["kind"] == "explicit":
        bundle = save_model(directory / "equation_model.zip", record["equation"], schema, metadata)
    else:
        bundle = save_prediction(directory / "prediction.json", record, schema, metadata)
    summary = write_json(directory / "result_summary.json", metadata)
    return RunResult(
        directory,
        {"final_" + k: v for k, v in (metrics or {}).items() if isinstance(v, (float, int))},
        {"model_bundle": bundle, "result_summary": summary, "frozen_model": frozen},
        tables,
        extra=metadata,
    )


def audit_hypotheses(study_dir, decision_dir, *, run_dir):
    """Audit the frozen hypothesis pool after an explicit choice/defer; no ranking or refit."""
    context, records, rows, frozen_arrays = load_study(study_dir)
    decision_path = Path(decision_dir) / "decision.json"
    decision = read_json(decision_path)
    if (
        decision["study_sha256"] != digest(Path(study_dir) / "study.json")
        or digest(decision["comparison_path"]) != decision["comparison_sha256"]
    ):
        raise ValueError("Frozen decision/evidence differs from this study")
    return _score_frozen_pool(study_dir, run_dir, context, records, rows, frozen_arrays,
                              list(range(len(records))), digest(decision_path))


def score_representatives(study_dir, *, run_dir):
    """Freeze one Selection representative per method/seed, then score only these on Final.

    No refit, deployment decision, or ranking by Final. Prior external exposure
    cannot be established by this function; see the recorded evidence role.
    Returns None without reading Final when no eligible representative exists.
    """
    context, records, rows, arrays = load_study(study_dir)
    selected = {record["id"] for record, _ in select_representatives(records, rows).values()}
    if not selected:
        return None
    indices = [i for i, record in enumerate(records) if record["id"] in selected]
    return _score_frozen_pool(study_dir, run_dir, context, records, rows, arrays, indices, None)


def _score_frozen_pool(study_dir, run_dir, context, records, rows, frozen_arrays, indices, decision_hash):
    import numpy as np

    from .data import load_prepared
    from .equation_evaluation.numerical import group_metrics, predict_record, weighted_metrics
    from .hypotheses import prediction_equations, residual_evidence

    if "final" not in context["prepared"]:
        raise ValueError("The prepared split has no Final partition; Selection is not a substitute")
    study_hash = digest(Path(study_dir) / "study.json")
    directory = new_directory(run_dir)
    frozen = write_json(
        directory / "frozen_pool.json",
        {
            "study_sha256": study_hash,
            "decision_sha256": decision_hash,
            "ids": [r["id"] for r in records],
            "scored_indices": indices,
            "scored_ids": [records[i]["id"] for i in indices],
            "purpose": "Final audit only; not selection or adoption",
        },
    )
    provenance = read_json(context["prepared"]["run"])
    exposure = final_exposure(context["prepared"], provenance, records, history=context.get("final_history", "unknown"))
    marker = Path(context["prepared"]["run"]).parent.parent / "final_access.json"
    role = (
        "holdout under explicit user declaration"
        if context.get("mode") != "historical" and context["final_history"] == "declared_unseen"
        and not exposure["same_data_exposed"] and not marker.exists()
        else "post-selection audit; prior exposure not excluded"
    )
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            import json

            json.dump(
                {
                    "data_sha256": context["data_sha256"],
                    "split_sha256": context["split_sha256"],
                    "decision_sha256": decision_hash,
                    "frozen_pool_sha256": digest(frozen),
                    "role": role,
                },
                stream,
                indent=2,
            )
    X, y, ids = load_prepared(context["prepared"], provenance, "final")
    result = []
    arrays = {"sample_id": ids, "observed": y}
    strata = None
    if context.get("strata_column"):
        if "frozen_strata" in frozen_arrays:
            label_ids, labels = frozen_arrays["strata_ids"], frozen_arrays["frozen_strata"]
        else:
            # Historical studies did not freeze Final labels: require the
            # original source hash before reading target-free metadata.
            legacy = frozen_labels(provenance, context["strata_column"])
            label_ids, labels = legacy["sample_id"], legacy["label"]
        lookup = dict(zip(label_ids, labels, strict=True))
        strata = np.asarray([lookup[key] for key in ids], dtype=str)
        arrays["strata"] = strata
    for i in indices:
        record = records[i]
        row = {"id": record["id"], "source_index": i, "kind": record["kind"], "selection_eligible": rows[i]["eligible"]}
        if prediction_equations(record):
            prediction, valid = predict_record(record, X, context["variables"], context["target"])
            row["target_metrics"] = weighted_metrics(y, prediction, valid, np.ones(len(y)))
            row["coverage"] = float(valid.mean())
            row["prediction_support"] = prediction_support(
                row["target_metrics"], tolerance=context["engineering_tolerance"], coverage=row["coverage"]
            )
            if strata is not None:
                row["strata"] = group_metrics(y, prediction, valid, np.ones(len(y)), strata)
            arrays[f"c{i}_prediction"] = prediction
            arrays[f"c{i}_residual"] = y - prediction
            arrays[f"c{i}_valid"] = valid
        if record["kind"] == "implicit":
            evidence, residual, valid = residual_evidence(record, X, y, context["variables"], context["target"])
            row["native_residual_evidence"] = evidence
            arrays[f"c{i}_native_residual"] = residual
            arrays[f"c{i}_native_residual_valid"] = valid
        result.append(row)
    np.savez_compressed(directory / "predictions.npz", **arrays)
    write_json(
        directory / "audit.json",
        {
            "study_sha256": study_hash,
            "frozen_pool_sha256": digest(frozen),
            "final_evidence_role": role,
            "results": result,
            "predictions_sha256": digest(directory / "predictions.npz"),
            "selection_changed": False,
            "refit": False,
            "final_history": context.get("final_history", "unknown"),
            "history_scope": "Local recorded history and declarations only; external prior exposure cannot be verified",
        },
    )
    return directory


def assess_prediction_use(decision_dir, interval_dir, *, run_dir):
    """Decide use of the SAME frozen predictor after independent calibration."""
    from .prediction_intervals import load_prediction_interval
    from .units import convert_width

    path = Path(decision_dir) / "decision.json"
    decision = read_json(path)
    plan = decision.get("calibration_plan")
    if decision["status"] != "frozen_prediction" or plan is None:
        raise ValueError("Use assessment requires a frozen predictor and predeclared calibration/width policy")
    interval = load_prediction_interval(interval_dir, decision=decision, decision_sha256=digest(path))
    meta = interval.metadata
    missing = list(decision["missing_support"])
    width = float(convert_width(2 * meta["quantile"], meta["unit"], plan["unit"]))
    if not math.isfinite(width):
        width = None
        missing.append("Calibrated full width is not representable in the criterion coordinate")
    elif width > plan["max_width"]:
        missing.append("Independent calibrated observation interval exceeds predeclared full-width criterion")
    directory = new_directory(run_dir)
    decision.update(
        status="defer" if missing else "adopt",
        missing_support=missing,
        frozen_choice_sha256=digest(path),
        interval_sha256=digest(Path(interval_dir) / "interval.json"),
        interval_path=str(Path(interval_dir).resolve()),
        calibrated_full_width=width,
        use_scope=(
            "Use assessment applies the frozen prediction-error and interval-width criteria only. "
            "Split-conformal coverage is marginal over calibration and future sampling under exchangeability. "
            "Conditioning on acceptance by the observed calibration width does not preserve nominal coverage; "
            "adoption is not a coverage guarantee for accepted deployments."
        ),
    )
    write_json(directory / "decision.json", decision)
    return directory

