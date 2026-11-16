"""Faithful prediction views of explicit, ensemble and implicit native outputs.

No search, fitting, ranking or residual-to-target score conversion lives here.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import model_matrix, model_output, read_json, write_json
from .domains import region_mask, validate_input_domain, within_bounds
from .execution_views import verify_subject
from .expressions import evaluate, inspect_equation
from .metrics import weighted_rms


def require_prediction_purpose(purpose):
    """A response/parameter analysis choice does not authorize target prediction."""
    if purpose != "local_prediction":
        raise ValueError("Predictive calibration/handoff requires a local_prediction-purpose choice")


def prediction_equations(record):
    if record.get("execution_view", {}).get("name") in {"residual", "native_only"}:
        return []
    if record["kind"] == "explicit":
        return [record["equation"]]
    if record["kind"] == "ensemble":
        return [member["equation"] for member in record["members"]]
    if record["kind"] == "implicit" and record.get("predictor") is not None:
        return [record["predictor"]]
    return []


def predict_working(record, X, variables):
    """Return prediction and validity, never drop ensemble mass or fill domain holes."""
    verify_subject(record)
    equations = prediction_equations(record)
    if not equations:
        raise ValueError("No X-only prediction; an implicit residual is not a target predictor")
    if record["kind"] == "ensemble":
        values = np.column_stack([evaluate(eq, X, variables) for eq in equations])
        weights = np.asarray([m["weight"] for m in record["members"]], dtype=float)
        valid = np.isfinite(values).all(axis=1)
        domain = record.get("target", {}).get("domain")
        valid &= within_bounds(values, domain).all(axis=1)
        if record.get("execution_view", {}).get("name") == "weighted_quantile":
            from .ensemble import weighted_formula_quantiles

            return weighted_formula_quantiles(values, weights, [record["execution_view"]["settings"]["q"]])[:, 0], valid
        return values @ weights, valid
    values = evaluate(equations[0], X, variables)
    return values, np.isfinite(values)


def residual_evidence(record, X, y, variables, target):
    """Within-relation diagnostic only: arbitrary residual scale precludes ranking."""
    schema = [*variables, {"id": "target", **target}]
    matrix = np.column_stack([X, y])
    values = evaluate(record["residual"], matrix, schema)
    valid = np.isfinite(values) & within_bounds(y, record.get("target", target).get("domain"))
    valid &= region_mask(X, record.get("execution_domain"))
    for i, variable in enumerate(record.get("variables", variables)):
        valid &= within_bounds(X[:, i], variable.get("domain"))
    rms, rms_reason = None, None
    if len(values) and valid.all():
        try:
            rms = weighted_rms(values)
        except FloatingPointError as error:
            rms_reason = str(error)
    else:
        rms_reason = "Invalid residual coverage; all rows retained"
    return (
        {
            "n_rows": len(y),
            "n_finite": int(valid.sum()),
            "residual_rms": rms,
            "residual_rms_reason": rms_reason,
            "meaning": "native residual coordinates; not target RMSE, likelihood or a cross-relation ranking",
            "prediction_available": bool(record.get("predictor")),
            "prediction_unavailable_reason": record.get("prediction_unavailable_reason"),
        },
        values,
        valid,
    )


def artifact_record(payload, *, candidate_id=None):
    """Retain native metadata and particle identities; make only declared prediction views."""
    from .constraints import artifact_size_coordinate
    schema = payload["schema"]
    common = {
        "source_size_coordinate": artifact_size_coordinate(payload),
        "variables": schema["variables"],
        "target": schema["target"],
        "source_schema": schema,
        "native_metadata": payload.get("metadata", {}),
        "source_constraints": payload.get("constraints"),
        "source_operators": payload.get("operators"),
        "native_check": {"passed": None, "kind": "imported native artifact; parity is retained per native member"},
    }
    if payload.get("kind") == "structural_point_ensemble":
        members = payload["members"]
        weights = np.asarray([m["weight"] for m in members], dtype=float)
        if (
            not members
            or len(members) > 10000
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("Native ensemble weights invalid; never normalize scores or discard mass")
        common["native_check"] = {
            "passed": False
            if any(m.get("native_check", {}).get("passed") is False for m in members)
            else True
            if all(m.get("native_check", {}).get("passed") is True for m in members)
            else None,
            "kind": "all saved member native checks"
            if all(m.get("native_check", {}).get("passed") is True for m in members)
            else "imported ensemble native parity unverified",
        }
        return {
            **common,
            "kind": "ensemble",
            "members": deepcopy(members),
            "prediction_view": "native-weighted mean conditional on fitted coefficients",
            "uncertainty_scope": payload["metadata"].get(
                "uncertainty_scope", "structural points conditional on fitted constants; not observation intervals"
            ),
        }
    if payload.get("kind") == "implicit_relations":
        members = [m for m in payload["members"] if m["candidate_id"] == candidate_id]
        if len(members) != 1:
            raise ValueError("Select exactly one native implicit candidate_id")
        member = members[0]
        return {
            **common,
            "kind": "implicit",
            "residual": deepcopy(member["residual"]),
            "predictor": deepcopy(member.get("predictor")),
            "prediction_unavailable_reason": member.get("prediction_unavailable_reason"),
            "prediction_view": "declared X-only projection" if member.get("predictor") else "residual only",
            "native_member": deepcopy(member),
        }
    raise ValueError("Expected structural_point_ensemble or implicit_relations native artifact")


@dataclass
class Prediction:
    record: dict
    schema: dict
    metadata: dict

    def working_prediction(self, frame, *, units=None):
        """Validated values in the saved working coordinate, before public conversion."""
        X = model_matrix(frame, self.schema, units)
        validate_input_domain(X, self.schema["variables"])
        values, valid = predict_working(self.record, X, self.schema["variables"])
        if not (valid & within_bounds(values, self.schema["target"].get("domain"))).all():
            raise ValueError("Prediction is undefined/outside domain; no rows or ensemble mass were removed")
        return values

    def predict(self, frame, *, units=None, output_unit=None):
        return model_output(self.working_prediction(frame, units=units), self.schema, output_unit)


def save_prediction(path, record, schema, metadata):
    verify_subject(record)
    if record.get("native_check", {}).get("passed") is False:
        raise ValueError("A confirmed native fidelity failure cannot be exported as a usable prediction")
    if not prediction_equations(record):
        raise ValueError("Residual-only native result cannot be exported as a point predictor")
    return write_json(
        path,
        {
            "schema_version": 2 if schema.get("input_geometry") or record.get("subject") else 1,
            "kind": "evaluated_prediction",
            "record": record,
            "schema": schema,
            "metadata": metadata,
        },
    )


def load_prediction(path):
    if Path(path).stat().st_size > 20_000_000:
        raise ValueError("Prediction artifact too large")
    item = read_json(path)
    if item.get("kind") != "evaluated_prediction" or item.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported evaluated prediction format")
    record, schema = item["record"], item["schema"]
    if record["kind"] == "ensemble":
        artifact_record(
            {
                "kind": "structural_point_ensemble",
                "schema": schema,
                "members": record["members"],
                "metadata": record.get("native_metadata", {}),
            }
        )
    verify_subject(record)
    equations = prediction_equations(record)
    if not equations:
        raise ValueError("Saved output has no X-only prediction")
    for equation in equations:
        inspect_equation(equation, schema["variables"])
        from .equation_templates import validate_template_binding

        validate_template_binding(equation, schema["variables"])
    return Prediction(record, schema, item["metadata"])
