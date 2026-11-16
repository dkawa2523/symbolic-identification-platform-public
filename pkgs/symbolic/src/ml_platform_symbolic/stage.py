"""Local and integration entrypoints share the same explicit stage functions."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml_platform_core.config import deep_merge
from ml_platform_core.result import RunResult

from .config import resolve_config
from .data import digest, prepare, public_unit, read_json, write_json
from .engines.runner import run_discovery
from .model import load_model
from .problem import problem_config, unpack_requirements


def infer(cfg, directory):
    model = load_model(cfg["model"]["local_model_path"])
    from .data import read_observations

    frame = read_observations(cfg["data"]["local_path"], identifiers=(), optional_identifiers=cfg["data"]["id_columns"])
    units = cfg["data"]["source_units"] or cfg["data"]["units"]
    target_column = model.schema["target"].get("column")
    output_unit = units.get(target_column)
    input_units = {k: v for k, v in units.items() if k != target_column}
    prediction = model.predict(frame, units=input_units, output_unit=output_unit)
    output = pd.DataFrame({"row_index": np.arange(len(frame)), "prediction": prediction})
    for column in cfg["data"]["id_columns"]:
        if column in frame:
            output[column] = frame[column]
    path = directory / "predictions.csv"
    output.to_csv(path, index=False)
    X = model.input_matrix(frame, input_units)
    ranges = np.asarray(model.schema["feature_ranges"])
    outside = ((ranges[:, 0] > X) | (ranges[:, 1] < X)).any(axis=1)
    summary = {
        "status": "ok",
        "n_rows": len(frame),
        "input_units": input_units,
        "prediction_unit": output_unit
        if output_unit is not None
        else public_unit(model.schema, model.schema["target"]),
        "out_of_range_rows": int(outside.sum()),
        "model_sha256": digest(cfg["model"]["local_model_path"]),
    }
    return RunResult(
        directory,
        artifacts={"result_summary": write_json(directory / "result_summary.json", summary)},
        tables={"predictions": path},
        extra=summary,
    )


def _redeclares_problem(config):
    return any(key in config for key in (
        "data", "split", "constraints", "domain", "input_map", "observation_model", "model_assets"
    ))


def _check_discovery_settings(cfg, provenance, supplied):
    """Reject explicit discovery overrides that a frozen prepare would ignore.

    Minimal calls inherit the complete prepared problem. Full redeclarations
    use the existing normalizer and whole semantic objects, not raw leaf masks.
    Engine/seed scheduling and evaluation/output-only settings stay separate.
    """
    if "problem_schema_version" in supplied:
        supplied = problem_config(supplied)
    explicit = unpack_requirements(supplied)
    requested = explicit.get("search", {})
    prepared = problem_config(provenance)
    frozen = prepared["search"]
    engine = cfg["run"]["engine"]
    changed = []
    for key in requested:
        if key in {"engines", "seeds", "allow_partial_engines"}:
            continue  # Batch scheduling, not settings of this single discovery.
        actual, expected = cfg["search"][key], frozen[key]
        if key in {"engine_options", "runtimes"}:
            if engine not in requested[key]:
                continue
            actual, expected = actual.get(engine, {}), expected.get(engine, {})
            if key == "runtimes":
                def runtime(value):
                    return {"device": value.get("device", "cpu"),
                            "python": str(Path(value.get("python", sys.executable)).absolute())}
                actual, expected = runtime(actual), runtime(expected)
        if actual != expected:
            changed.append("search." + key)
    # A problem redeclaration is a standalone complete problem, never a partial
    # patch in mixed source/working coordinates. Compare whole semantic objects
    # after the existing normalizer, including values derived from units.
    if _redeclares_problem(explicit):
        for section in ("data", "split", "constraints", "domain", "input_map", "observation_model"):
            actual, expected = deepcopy(cfg.get(section)), deepcopy(prepared.get(section))
            if section in {"data", "split"}:
                for key in ("local_path", "file"):
                    if key in actual:
                        actual[key] = str(Path(actual[key]).resolve()) if actual[key] is not None else None
                        expected[key] = str(Path(expected[key]).resolve()) if expected[key] is not None else None
            if section == "constraints":
                for value in (actual, expected):
                    if value.get("allowed_variables") is not None:
                        value["allowed_variables"] = sorted(value["allowed_variables"])
            if actual != expected:
                changed.append(section)
        if cfg["search"]["operators"] != prepared["search"]["operators"]:
            changed.append("search.operators")
    if _redeclares_problem(explicit) or engine in explicit.get("model_assets", {}):
        actual = cfg["model_assets"].get(engine)
        expected = prepared.get("model_assets", {}).get(engine)
        def asset(value):
            return str(Path(value["manifest_file"]).resolve()) if value is not None else None
        if asset(actual) != asset(expected):
            changed.append("model_assets." + engine)
    if changed:
        raise ValueError("Discovery settings differ from the frozen prepare: " + ", ".join(changed)
                         + ". No worker started. Prepare again with the intended settings and the same fixed split; "
                         "explicit overrides are not silently ignored.")


def _stage(cfg, directory, supplied):
    stage, inputs = cfg["run"].get("stage"), cfg["stage_inputs"]
    if stage == "prepare":
        refs = prepare(cfg, directory / "prepare")
        return RunResult(directory, artifacts={k: Path(v) for k, v in refs.items()}, extra=refs)
    refs = inputs["prepare"]
    provenance = read_json(refs["run"])
    if stage == "discover":
        _check_discovery_settings(cfg, provenance, supplied)
        result = run_discovery(refs, provenance, cfg["run"]["engine"], cfg["run"]["seed"], directory / "discover")
        native = read_json(result["summary"]).get("native_artifacts", {})
        return RunResult(
            directory,
            artifacts={
                **{"engine_summary" if k == "summary" else k: Path(v) for k, v in result.items() if Path(v).is_file()},
                **{f"native_{k}": Path(v) for k, v in native.items()},
            },
            extra=result,
        )
    raise ValueError(f"Unsupported symbolic stage: {stage}")


def run_task(config: Mapping[str, Any], *, run_dir: Path) -> RunResult:
    supplied = problem_config(config) if "problem_schema_version" in config else config
    if (supplied.get("task") == "symbolic_stage" and supplied.get("run", {}).get("stage") == "discover"
            and supplied.get("stage_inputs", {}).get("prepare")):
        declared = unpack_requirements(supplied)
        if _redeclares_problem(declared):
            try:
                cfg = resolve_config(supplied)
            except ValueError as exc:
                raise ValueError(
                    "Discovery problem redeclaration must be complete. Omit problem fields to inherit prepare, "
                    "or provide the complete original configuration; changes require a new prepare. " + str(exc)
                ) from exc
        else:
            prepared = problem_config(read_json(supplied["stage_inputs"]["prepare"]["run"]))
            cfg = resolve_config(deep_merge(prepared, declared))
    else:
        cfg = resolve_config(config)
    if cfg["task"] == "symbolic_pipeline":
        raise ValueError("Automatic selection/Final execution was removed. Use explicit prepare/discover, create_study and comparison; score fixed representatives separately.")
    if cfg["task"] == "symbolic_stage" and cfg["run"].get("stage") not in {"prepare", "discover"}:
        raise ValueError("Only prepare/discover are executable stages; use explicit study/comparison/selection APIs.")
    directory = Path(run_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"run_dir must be new or empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if cfg["task"] == "symbolic_infer":
        return infer(cfg, directory)
    if cfg["task"] == "symbolic_stage":
        return _stage(cfg, directory, config)
    raise ValueError("Unsupported execution task")
