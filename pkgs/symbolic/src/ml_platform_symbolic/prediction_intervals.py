"""Split conformal observation intervals for one frozen prediction rule.

Independent rows or exchangeable independent groups only. This never refits a model,
estimates a posterior, or certifies coverage after distribution shift.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    column_to_working,
    digest,
    input_variables,
    model_output,
    observation_index,
    public_unit,
    read_json,
    read_observations,
    write_json,
)
from .equation_evaluation.api import load_study, new_directory
from .equation_evaluation.numerical import intersect_schema
from .execution_views import verify_subject
from .hypotheses import load_prediction, require_prediction_purpose, save_prediction
from .metrics import prediction_metrics
from .units import convert_width


def _prediction_observations(model, frame, units):
    """Same saved public-coordinate default as inference; observations use that boundary too."""
    schema = model.schema
    variables, target = input_variables(schema), schema["target"]
    columns = {v["column"] for v in [*variables, target]}
    if units is not None and set(units) - columns:
        raise ValueError("Unknown observation-unit columns")
    input_units = None if not units else {k: v for k, v in units.items() if k != target["column"]}
    prediction = model.working_prediction(frame, units=input_units)
    target_unit = (units or {}).get(target["column"], public_unit(schema, target))
    y = column_to_working(frame[target["column"]].to_numpy(float), schema, target, target_unit)
    if not np.isfinite(y).all():
        raise ValueError("Observation values must be finite; no filtering")
    return prediction, y


def _interval_bounds(point, width):
    with np.errstate(over="ignore", invalid="ignore"):
        lower, upper = point - width, point + width
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Prediction interval endpoints are nonfinite in the requested coordinate")
    return lower, upper


def _scores(errors, groups):
    return np.asarray([np.max(errors[groups == group]) for group in np.unique(groups)])


def calibrate_prediction(
    study_dir,
    decision_dir,
    calibration_csv,
    *,
    run_dir,
    alpha=0.1,
    exchangeability,
    observation_model,
    group_column=None,
    units=None,
):
    """Use separate calibration observations AFTER a named point choice is frozen."""
    context, records, rows, _ = load_study(study_dir)
    decision_path = Path(decision_dir) / "decision.json"
    decision = read_json(decision_path)
    require_prediction_purpose(context["purpose"])
    require_prediction_purpose(decision.get("purpose"))
    if decision.get("interval_sha256"):
        raise ValueError("Calibration is already assessed; use the original frozen point choice for a new calibration")
    if decision["study_sha256"] != digest(Path(study_dir) / "study.json") or not decision.get("selected_id"):
        raise ValueError("Calibration requires an explicit frozen point choice for this study")
    if digest(decision["comparison_path"]) != decision["comparison_sha256"]:
        raise ValueError("Decision evidence changed")
    if not 0 < alpha < 1 or not isinstance(observation_model, str) or not observation_model.strip():
        raise ValueError("Declare alpha and the observation/exchangeability assumptions")
    if exchangeability not in {"independent_rows", "exchangeable_groups"}:
        raise ValueError("Only independent rows or independent exchangeable groups are supported")
    if exchangeability == "independent_rows" and (group_column or context.get("group_column")):
        raise ValueError("Repeated/grouped observations require group calibration, not row exchangeability")
    if exchangeability == "exchangeable_groups" and not group_column:
        raise ValueError("Declare the independent group column")
    if context.get("group_column") and group_column != context["group_column"]:
        raise ValueError("Calibration must preserve the prepared independent group column; rows cannot replace groups")
    _calibration_settings(
        decision,
        dict(
            alpha=alpha,
            exchangeability=exchangeability,
            group_column=group_column,
            observation_model=observation_model,
        ),
    )
    frame = read_observations(calibration_csv, identifiers=["sample_id", *([group_column] if group_column else [])])
    if "sample_id" not in frame or frame.sample_id.duplicated().any() or frame.sample_id.isna().any():
        raise ValueError("Calibration needs unique nonmissing sample_id")
    # IDs only: never open Final target values to validate calibration separation.
    prepared = read_json(context["prepared"]["run"])
    identities = observation_index(prepared)
    ids = identities["sample_id"]
    if set(ids) & set(frame.sample_id.astype(str)):
        raise ValueError("Calibration overlaps discovery/Selection/Final sample IDs")
    if group_column:
        known = identities["group"]
        if frame[group_column].isna().any() or set(known) & set(frame[group_column].astype(str)):
            raise ValueError("Calibration groups overlap the prepared problem or have missing labels")
    index = next(i for i, r in enumerate(records) if r["id"] == decision["selected_id"])
    if not rows[index]["eligible"]:
        raise ValueError("Frozen prediction is not eligible")
    directory = new_directory(run_dir)
    variables, target = intersect_schema(records[index], context)
    schema = {
        "variables": variables,
        "target": target,
        "public_units": prepared["config"]["data"].get("source_units", {}),
        "unit_coordinates": context.get("unit_coordinates"),
        "input_geometry": records[index].get("input_geometry"),
    }
    point = save_prediction(
        directory / "prediction.json",
        records[index],
        schema,
        {"decision": decision, "decision_sha256": digest(decision_path)},
    )
    model = load_prediction(point)
    prediction, y = _prediction_observations(model, frame, units)
    groups = frame[group_column].astype(str).to_numpy() if group_column else frame.sample_id.astype(str).to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        errors = abs(y - prediction)
    if not np.isfinite(errors).all():
        raise ValueError("Calibration errors exceed the representable working coordinate")
    scores = _scores(errors, groups)
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        raise ValueError("Too few independent calibration units for a finite interval at requested alpha")
    quantile = float(np.sort(scores)[rank - 1])
    write_json(
        directory / "interval.json",
        {
            "schema_version": 1,
            "kind": "split_conformal_observation_interval",
            "alpha": alpha,
            "quantile": quantile,
            "unit": target.get("unit"),
            "exchangeability": exchangeability,
            "observation_model": observation_model,
            "group_column": group_column,
            "calibration_csv": str(Path(calibration_csv).resolve()),
            "calibration_sha256": digest(calibration_csv),
            "calibration_ids": frame.sample_id.astype(str).tolist(),
            "prepared_problem_ids": ids.astype(str).tolist(),
            "prepared_problem_groups": known.astype(str).tolist() if group_column else [],
            "calibration_groups": groups.tolist(),
            "calibration_scores": scores.tolist(),
            "rank": rank,
            "n_calibration_units": len(scores),
            "decision_sha256": digest(decision_path),
            "candidate_id": decision["selected_id"],
            "study_sha256": decision["study_sha256"],
            "prediction_sha256": digest(point),
            "selection_status": decision["status"],
            "coverage_scope": "marginal new observation"
            if not group_column
            else "simultaneous observations in one new exchangeable independent group",
            "limits": (
                "Requires exchangeability with calibration; coverage is marginal over calibration and future sampling. "
                "No nominal coverage guarantee conditional on acceptance by observed interval width. "
                "No OOD, conditional, coefficient/posterior, mechanism or full-search coverage guarantee"
            ),
        },
    )
    pd.DataFrame({"sample_id": frame.sample_id, "group": groups, "absolute_error": errors}).to_csv(
        directory / "calibration_errors.csv", index=False
    )
    return directory


@dataclass
class PredictionInterval:
    model: object
    metadata: dict

    @property
    def working_width(self):
        return convert_width(self.metadata["quantile"], self.metadata["unit"], self.model.schema["target"].get("unit"))

    def predict(self, frame, *, units=None, output_unit=None):
        point = self.model.predict(frame, units=units, output_unit=output_unit)
        width = model_output(self.working_width, self.model.schema, output_unit, difference=True)
        lower, upper = _interval_bounds(point, width)
        return point, lower, upper


def _calibration_settings(decision, metadata):
    plan = decision.get("calibration_plan")
    keys = ("alpha", "exchangeability", "group_column", "observation_model")
    if plan is not None and any(metadata.get(key) != plan[key] for key in keys):
        raise ValueError("Calibration settings differ from the policy frozen before calibration")


def read_interval_evidence(path, *, decision=None, decision_sha256=None, displayed=None):
    """Read/bind saved evidence without initializing a predictor or scoring a policy.

    The frozen point choice identifies the predictor. An assessed use decision
    additionally names ONE calibration artifact, including its data/assumptions.
    Same predictor (or same width) cannot substitute another calibration.
    Standalone historical intervals without decision metadata remain mathematical
    artifacts; they cannot stand in for a decision-bound handoff.
    """
    directory = Path(path)
    metadata = read_json(directory / "interval.json")
    if metadata.get("kind") != "split_conformal_observation_interval" or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported prediction interval")
    point_path = directory / "prediction.json"
    if point_path.stat().st_size > 20_000_000 or digest(point_path) != metadata["prediction_sha256"]:
        raise ValueError("Calibrated point rule changed or exceeds its size limit")
    point = read_json(point_path)
    if point.get("kind") != "evaluated_prediction" or point.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported evaluated prediction format")
    verify_subject(point["record"])
    embedded = point.get("metadata", {})
    if "decision_sha256" in metadata and (
        embedded.get("decision_sha256") != metadata["decision_sha256"]
        or point["record"].get("id") != metadata.get("candidate_id")
    ):
        raise ValueError("Interval candidate/decision identity differs")
    if embedded.get("decision", {}).get("purpose") is not None:
        require_prediction_purpose(embedded["decision"]["purpose"])
    if not np.isfinite(metadata["quantile"]) or metadata["quantile"] < 0 or not 0 < metadata["alpha"] < 1:
        raise ValueError("Invalid calibrated interval scale/alpha")
    width = convert_width(metadata["quantile"], metadata["unit"], point["schema"]["target"].get("unit"))
    if not np.isfinite(width):
        raise ValueError("Calibrated width conversion is nonfinite")
    fingerprint = digest(directory / "interval.json")
    if decision is not None:
        require_prediction_purpose(decision.get("purpose"))
        choice = decision.get("frozen_choice_sha256", decision_sha256)
        if (
            not choice
            or metadata.get("study_sha256") != decision.get("study_sha256")
            or metadata.get("decision_sha256") != choice
            or metadata.get("candidate_id") != decision.get("selected_id")
            or point["record"].get("subject") != decision.get("subject")
        ):
            raise ValueError("Calibration belongs to different candidates/choices or frozen predictor/view")
        expected = decision.get("interval_sha256")
        if expected is not None and fingerprint != expected:
            raise ValueError("Prediction interval differs from the calibration artifact assessed by this use decision")
        if decision.get("calibration_plan") and decision.get("status") != "frozen_prediction" and not expected:
            raise ValueError("Calibrated use decision is missing its assessed interval identity")
        _calibration_settings(decision, metadata)
    if displayed is not None:
        if displayed.get("artifact_sha256", fingerprint) != fingerprint or any(
            displayed.get(key) != value for key, value in metadata.items()
        ):
            raise ValueError("Displayed interval evidence differs from the packaged calibration")
        heldout = displayed.get("heldout")
        if heldout is not None and heldout.get("interval_sha256") != fingerprint:
            raise ValueError("Coverage belongs to another calibration")
    return metadata, point, fingerprint


def load_prediction_interval(path, *, decision=None, decision_sha256=None, displayed=None):
    metadata, _, _ = read_interval_evidence(
        path,
        decision=decision,
        decision_sha256=decision_sha256,
        displayed=displayed,
    )
    return PredictionInterval(load_prediction(Path(path) / "prediction.json"), metadata)


def audit_prediction_interval(interval_dir, evaluation_csv, *, output, units=None):
    """Explicit new-data audit after calibration; reporting consumes this saved result."""
    model = load_prediction_interval(interval_dir)
    meta = model.metadata
    frame = read_observations(
        evaluation_csv, identifiers=["sample_id", *([meta["group_column"]] if meta["group_column"] else [])]
    )
    if "sample_id" not in frame or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Evaluation needs unique sample IDs")
    if set(frame.sample_id.astype(str)) & set(meta["calibration_ids"]):
        raise ValueError("Evaluation is not independent of calibration")
    if set(frame.sample_id.astype(str)) & set(meta.get("prepared_problem_ids", [])):
        raise ValueError("New-data coverage audit overlaps discovery/Selection/previous Final observations")
    group = meta["group_column"]
    if group and (frame[group].isna().any() or set(frame[group].astype(str)) & set(meta["calibration_groups"])):
        raise ValueError("Evaluation groups overlap calibration")
    if group and set(frame[group].astype(str)) & set(meta.get("prepared_problem_groups", [])):
        raise ValueError("New-data coverage audit overlaps prepared groups")
    target = model.model.schema["target"]
    point, y = _prediction_observations(model.model, frame, units)
    low, high = _interval_bounds(point, model.working_width)
    with np.errstate(over="ignore"):
        full_width = float(2 * model.working_width)
    covered = (low <= y) & (y <= high)
    groups = frame[group].astype(str).to_numpy() if group else frame.sample_id.astype(str).to_numpy()
    group_coverage = [bool(covered[groups == g].all()) for g in np.unique(groups)]
    return write_json(
        output,
        {
            "kind": "heldout_interval_coverage",
            "interval_sha256": digest(Path(interval_dir) / "interval.json"),
            "evaluation_sha256": digest(evaluation_csv),
            "n_rows": len(y),
            "n_independent_units": len(group_coverage),
            "row_coverage": float(covered.mean()),
            "unit_coverage": float(np.mean(group_coverage)),
            "mean_width": full_width if np.isfinite(full_width) else None,
            "width_reason": None if np.isfinite(full_width) else "Full interval width outside representable coordinate",
            "rmse": prediction_metrics(y, point)["rmse"],
            "unit": target.get("unit"),
            "coverage_scope": meta["coverage_scope"],
            "limits": meta["limits"],
            "observation_model": meta["observation_model"],
        },
    )
