"""Portable native implicit candidates; residual evaluation is not target prediction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import column_to_working, model_matrix, public_unit, read_json
from .domains import validate_input_domain, within_bounds
from .expressions import evaluate, inspect_equation
from .model import EquationModel


def relation_variables(schema):
    return [*schema["variables"], {"id": "target", **schema["target"]}]


@dataclass
class ImplicitRelation:
    equation: dict
    schema: dict
    metadata: dict
    predictor: dict | None
    prediction_unavailable_reason: str | None

    def residual(self, frame, target_values=None, *, units=None, target_unit=None):
        """Evaluate R(X,y), preserving original scales and every supplied row."""
        X = model_matrix(frame, self.schema, units)
        validate_input_domain(X, self.schema["variables"])
        if target_values is None:
            target_values = frame[self.schema["target"]["column"]]
        target = self.schema["target"]
        source = target_unit if target_unit is not None else public_unit(self.schema, target)
        y = column_to_working(np.asarray(target_values, dtype=float), self.schema, target, source)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("Residual evaluation needs one finite observed target per row")
        if not within_bounds(y, self.schema["target"].get("domain")).all():
            raise ValueError("Observed target outside declared target domain")
        residual = evaluate(self.equation, np.column_stack([X, y]), relation_variables(self.schema))
        if not np.isfinite(residual).all():
            raise ValueError("Implicit relation undefined; no rows were dropped")
        return residual

    def predict(self, frame, *, units=None, output_unit=None):
        """Use an explicitly available X-only projection, never observed-y brackets."""
        if self.predictor is None:
            raise ValueError(f"X-only prediction unavailable: {self.prediction_unavailable_reason}")
        return EquationModel(self.predictor, self.schema, self.metadata).predict(
            frame, units=units, output_unit=output_unit
        )


def load_relation(path, candidate_id=None):
    """Read a native candidate, not an assertion of point eligibility or selection."""
    if Path(path).stat().st_size > 10_000_000:
        raise ValueError("Implicit relation catalog is too large")
    payload = read_json(path)
    if payload.get("schema_version") not in {1, 2} or payload.get("kind") != "implicit_relations":
        raise ValueError("Unsupported implicit relation format")
    members = payload["members"]
    if candidate_id is not None:
        members = [member for member in members if member["candidate_id"] == candidate_id]
    if len(members) != 1:
        raise ValueError("Choose exactly one candidate_id from the implicit relation catalog")
    member, schema = members[0], payload["schema"]
    inspect_equation(member["residual"], relation_variables(schema))
    predictor = member.get("predictor")
    if predictor is not None:
        inspect_equation(predictor, schema["variables"])
    metadata = {**payload["metadata"], **member.get("metadata", {}), "candidate_id": member["candidate_id"]}
    return ImplicitRelation(
        member["residual"], schema, metadata, predictor, member.get("prediction_unavailable_reason")
    )
