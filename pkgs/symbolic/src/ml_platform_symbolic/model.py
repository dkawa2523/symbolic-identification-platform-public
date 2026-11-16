"""Atomic, data-only model bundles. Inference never imports a search engine."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constraints import dimensions
from .data import model_matrix, model_output
from .domains import validate_input_domain, within_bounds
from .expressions import evaluate, inspect_equation


@dataclass
class EquationModel:
    equation: dict
    schema: dict
    metadata: dict

    def input_matrix(self, frame, units=None):
        return model_matrix(frame, self.schema, units)

    def predict(self, frame, *, units=None, output_unit=None):
        """Saved public coordinates by default; finite real-domain prediction.

        Raw native IEEE arithmetic is retained in the equation and can be
        replayed with expressions.evaluate_native; it does not authorize an
        invalid real intermediate as a scientific deployment prediction.
        """
        X = self.input_matrix(frame, units)
        validate_input_domain(X, self.schema["variables"])
        ranges = np.asarray(self.schema["feature_ranges"])
        outside = ((ranges[:, 0] > X) | (ranges[:, 1] < X)).any(axis=1)
        if outside.any():
            warnings.warn(
                f"{outside.sum()} rows outside training ranges; extrapolation is unverified", UserWarning, stacklevel=2
            )
        prediction = evaluate(self.equation, X, self.schema["variables"])
        if not np.isfinite(prediction).all():
            row = int(np.flatnonzero(~np.isfinite(prediction))[0])
            raise ValueError(f"Equation undefined at row {row}; no rows were dropped")
        if not within_bounds(prediction, self.schema.get("target", {}).get("domain")).all():
            raise ValueError("Prediction outside declared target domain; no rows were dropped")
        return model_output(prediction, self.schema, output_unit)


def save_model(path, equation, schema, metadata):
    inspect_equation(equation, schema["variables"])
    from .equation_templates import validate_template_binding

    validate_template_binding(equation, schema["variables"])
    status, reasons = dimensions(equation, schema["variables"], schema.get("target", {}))
    if status == "fail":
        raise ValueError("Cannot save a known unit-coordinate mismatch: " + "; ".join(reasons))
    payloads = {
        name + ".json": json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
        for name, value in {"equation": equation, "schema": schema, "metadata": metadata}.items()
    }
    manifest = {
        "schema_version": 2 if schema.get("input_geometry") else 1,
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()},
    }
    payloads["manifest.json"] = json.dumps(manifest).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".zip")
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, value in payloads.items():
                bundle.writestr(name, value)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return path


def load_model(path):
    expected = {"equation.json", "schema.json", "metadata.json", "manifest.json"}
    with zipfile.ZipFile(path) as bundle:
        if (
            set(bundle.namelist()) != expected
            or len(bundle.namelist()) != 4
            or any(i.file_size > 5_000_000 for i in bundle.infolist())
        ):
            raise ValueError("Invalid bundle members or size")
        payloads = {name: bundle.read(name) for name in expected}
    manifest = json.loads(payloads.pop("manifest.json"))
    if manifest.get("schema_version") not in {1, 2} or set(manifest.get("files", {})) != set(payloads):
        raise ValueError("Unsupported bundle manifest")
    for name, value in payloads.items():
        if manifest["files"][name] != hashlib.sha256(value).hexdigest():
            raise ValueError(f"Bundle checksum mismatch: {name}")
    values = {name.removesuffix(".json"): json.loads(value) for name, value in payloads.items()}
    schema = values["schema"]
    ids, columns = [v["id"] for v in schema["variables"]], [v["column"] for v in schema["variables"]]
    if len(set(ids)) != len(ids) or len(set(columns)) != len(columns) or len(schema["feature_ranges"]) != len(ids):
        raise ValueError("Invalid model feature schema")
    inspect_equation(values["equation"], schema["variables"])
    from .equation_templates import validate_template_binding

    validate_template_binding(values["equation"], schema["variables"])
    status, reasons = dimensions(values["equation"], schema["variables"], schema.get("target", {}))
    if status == "fail":
        raise ValueError("Stored model has a known unit-coordinate mismatch: " + "; ".join(reasons))
    return EquationModel(**values)
