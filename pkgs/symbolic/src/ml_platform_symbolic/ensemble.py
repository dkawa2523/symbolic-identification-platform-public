"""Portable weighted formula points, not coefficient/noise posterior draws."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constraints import check_constraints, artifact_size_coordinate
from .data import model_matrix, model_output, read_json
from .domains import validate_input_domain, within_bounds
from .expressions import evaluate, inspect_equation


@dataclass
class FormulaEnsemble:
    schema: dict
    members: list[dict]
    metadata: dict

    @property
    def weights(self):
        return np.asarray([member["weight"] for member in self.members], dtype=float)

    def predict_members(self, frame, *, units=None, output_unit=None):
        """Rows × original particles; duplicates and their native mass are retained."""
        variables = self.schema["variables"]
        X = model_matrix(frame, self.schema, units)
        validate_input_domain(X, variables)
        predictions = np.column_stack([evaluate(m["equation"], X, variables) for m in self.members])
        if not np.isfinite(predictions).all():
            raise ValueError("Ensemble member undefined; no rows/particles were dropped or reweighted")
        if not within_bounds(predictions, self.schema["target"].get("domain")).all():
            raise ValueError("Ensemble member outside declared target domain; no particles were dropped")
        return model_output(predictions, self.schema, output_unit)

    def predict(self, frame, *, units=None, output_unit=None):
        """Native-weighted mean conditional on the saved fitted coefficients."""
        return self.predict_members(frame, units=units, output_unit=output_unit) @ self.weights

    def structural_quantiles(self, frame, probabilities, *, units=None, output_unit=None):
        """Weighted empirical formula-point quantiles, not observation coverage intervals."""
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            probabilities.ndim != 1
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Quantile probabilities must be a finite 1D sequence in [0,1]")
        return weighted_formula_quantiles(
            self.predict_members(frame, units=units, output_unit=output_unit), self.weights, probabilities
        )


def weighted_formula_quantiles(predictions, weights, probabilities):
    """One empirical inverse-CDF definition; stored weights are never modified.

    Zero-mass points do not define quantiles. The accepted sum tolerance is the
    loader's floating-roundoff tolerance; only the terminal CDF is set to one.
    Caller-owned member/domain validity is preserved separately.
    """
    predictions, weights, probabilities = (np.asarray(v, dtype=float) for v in (predictions, weights, probabilities))
    if predictions.ndim != 2 or weights.shape != (predictions.shape[1],) or probabilities.ndim != 1:
        raise ValueError("Quantiles expect rows-by-members, member weights and1D probabilities")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Invalid native weights; no normalization or score substitution")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie in [0,1]")
    positive = weights > 0
    values, mass = predictions[:, positive], weights[positive]
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(mass[order], axis=1)
    cumulative[:, -1] = 1.0
    indices = (cumulative[:, :, None] < probabilities).sum(axis=1)
    return np.take_along_axis(ordered, indices, axis=1)


def load_ensemble(path):
    """Load safe formula data without PySIPS/Bingo/SMCPy or any search dependency."""
    if Path(path).stat().st_size > 10_000_000:
        raise ValueError("Ensemble artifact is too large")
    payload = read_json(path)
    if payload.get("schema_version") not in {1, 2} or payload.get("kind") != "structural_point_ensemble":
        raise ValueError("Unsupported ensemble format")
    members, schema = payload["members"], payload["schema"]
    ids, columns = [v["id"] for v in schema["variables"]], [v["column"] for v in schema["variables"]]
    if not ids or len(set(ids)) != len(ids) or len(set(columns)) != len(columns):
        raise ValueError("Invalid ensemble feature schema")
    if not members or len(members) > 10000:
        raise ValueError("Invalid ensemble member count")
    weights = np.asarray([member["weight"] for member in members], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Invalid native weights; scores must not be substituted or silently normalized")
    size_coordinate = artifact_size_coordinate(payload)
    legacy_size = size_coordinate == "legacy_working"
    for member in members:
        inspect_equation(member["equation"], schema["variables"])
        checks = check_constraints(
            member["equation"], schema["variables"], schema["target"], payload["operators"], payload["constraints"], context=schema,
            size_coordinate=size_coordinate
        )
        if any(check["required"] and check["status"] != "pass" for check in checks):
            raise ValueError("A native ensemble member violates/has unknown required constraints; no mass was filtered")
    metadata = dict(payload["metadata"])
    if legacy_size:
        metadata["complexity_scope"] = "legacy working-coordinate limit recorded without source marker; no source-size guarantee"
    return FormulaEnsemble(schema, members, metadata)


def evaluate_prepared_ensemble(path, refs, provenance, directory):
    """Score an already frozen native ensemble on the same prepared target/rows."""
    import pandas as pd

    from .coordinates import source_input_values
    from .data import digest, input_variables, load_prepared, write_json
    from .metrics import prediction_metrics

    model = load_ensemble(path)
    if model.metadata["provenance"]["prepared_sha256"] != digest(refs["run"]):
        raise ValueError("Ensemble prepared run identity mismatch")
    directory = Path(directory)
    directory.mkdir(parents=True)
    metrics, tables = {}, {}
    for partition in ("selection", "final"):
        if partition not in refs:
            continue
        X, y, ids = load_prepared(refs, provenance, partition)
        source = source_input_values(X, model.schema.get("input_geometry"))
        variables = input_variables(model.schema)
        frame = pd.DataFrame(source, columns=[v["column"] for v in variables])
        units = {v["column"]: v.get("unit") for v in variables}
        prediction = model.predict(frame, units=units, output_unit=model.schema["target"].get("unit"))
        quantiles = model.structural_quantiles(
            frame, [0.1, 0.9], units=units, output_unit=model.schema["target"].get("unit")
        )
        metrics.update({f"{partition}_{key}": value for key, value in prediction_metrics(y, prediction).items()})
        table = directory / f"{partition}_predictions.csv"
        pd.DataFrame(
            {
                "sample_id": ids,
                "observed": y,
                "prediction": prediction,
                "structural_q10": quantiles[:, 0],
                "structural_q90": quantiles[:, 1],
            }
        ).to_csv(table, index=False)
        tables[partition] = table
        write_json(
            directory / "metrics.json",
            {
                "metrics": metrics,
                "members": len(model.members),
                "uncertainty_scope": model.metadata["uncertainty_scope"],
                "observation_interval_coverage_claimed": False,
            },
        )
    return metrics, tables, len(model.members)
