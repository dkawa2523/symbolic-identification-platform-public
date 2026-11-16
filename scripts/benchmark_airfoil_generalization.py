"""Frozen Airfoil group/range audit using existing discovery and evaluation APIs.

Public, previously evaluated observations: this is not an unseen-law benchmark.
Final files are outside solver inputs and opened for scoring only after choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import (
    column_to_working,
    digest,
    model_output,
    prepare,
    read_json,
    read_observations,
    write_json,
)
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.equation_evaluation import create_study
from ml_platform_symbolic.equation_evaluation.api import load_study
from ml_platform_symbolic.equation_evaluation.numerical import predict_record
from ml_platform_symbolic.evaluation import select_row
from ml_platform_symbolic.expressions import equation_text
from ml_platform_symbolic.metrics import prediction_metrics
from ml_platform_symbolic.units import canonical_unit

FEATURES = ["frequency", "attack_angle", "chord", "velocity", "displacement_thickness"]
GROUP_FEATURES = FEATURES[1:]
SOURCE_SHA = "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
TABLE_SHA = "c800c8732ce791ae194f81d7e4db53c3210a20d7c8089c574ebd013923ac1d91"
SPLIT_SHA = "c865f5266fc51aa6c7a128e7c363fe5bb68e3aa8b4c5ab29b60b2bc755cbba5a"
PROFILES = {"mean": {}, "linear": {}, "dictionary": {"max_degree": 3, "max_terms": 8}}


def identity():
    import ml_platform_symbolic

    package = Path(ml_platform_symbolic.__file__).parent
    files = [(str(p.relative_to(package)), digest(p)) for p in sorted(package.rglob("*.py"))]
    files.append((Path(__file__).name, digest(__file__)))
    return {
        "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "image": os.environ.get("SYMBOLIC_CONTAINER_IMAGE_ID"),
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "ml-platform-core"]},
    }


def verify(root):
    plan = read_json(root / "plan.json")
    if plan["identity"] != identity():
        raise ValueError("Source/runtime changed after protocol freeze; preserve attempt and start a new campaign")
    for case in plan["cases"]:
        for item in case["solver_inputs"]:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Frozen solver input changed")
    return plan


def validate_groups(frame):
    if frame.sample_id.duplicated().any() or frame.groupby("group_id").partition.nunique().gt(1).any():
        raise ValueError("Duplicate sample IDs or condition group crosses partitions")
    if frame.groupby(GROUP_FEATURES, dropna=False).partition.nunique().gt(1).any():
        raise ValueError("An exact physical condition crosses partitions")
    if set(frame.partition) != {"train", "selection", "final"}:
        raise ValueError("Expected three nonempty partitions")
    return {
        key: {"rows": int(len(rows)), "groups": int(rows.group_id.nunique()),
              "velocities": sorted(rows.velocity.unique().tolist())}
        for key, rows in frame.groupby("partition")
    }


def split_cases(frame, old_split):
    """Decisions depend only on frozen row IDs and input conditions, never y."""
    existing = frame.merge(old_split, on="sample_id", validate="one_to_one")
    if len(existing) != len(frame):
        raise ValueError("Existing split does not cover all observations")
    held = frame.copy()
    final = held.velocity == 71.3
    remaining = sorted(held.loc[~final, "group_id"].unique())
    random.Random(2026110808).shuffle(remaining)
    selection = set(remaining[:max(1, math.ceil(0.2 * len(remaining)))])
    held["partition"] = np.where(final, "final", np.where(held.group_id.isin(selection), "selection", "train"))
    cases = {"group_split_seed42": existing, "velocity_71p3_holdout": held}
    for rows in cases.values():
        validate_groups(rows)
    return cases


def prepare_campaign(root, source):
    for filename, expected in [("source.zip", SOURCE_SHA), ("table.csv", TABLE_SHA), ("split.csv", SPLIT_SHA)]:
        if digest(source / filename) != expected:
            raise ValueError(f"Pinned official Airfoil input changed: {filename}")
    frame = read_observations(source / "table.csv", identifiers=("sample_id", "group_id"))
    old_split = read_observations(source / "split.csv")
    if len(frame) != 1503 or frame.group_id.nunique() != 106:
        raise ValueError("Unexpected Airfoil size or condition count")
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    metadata = read_json(source / "metadata.json")
    for name, rows in split_cases(frame, old_split).items():
        folder = root / "solver" / name
        folder.mkdir(parents=True)
        training = rows.loc[rows.partition != "final"].copy()
        table, split = folder / "observations.csv", folder / "split.csv"
        training.drop(columns="partition").to_csv(table, index=False)
        training[["sample_id", "partition"]].to_csv(split, index=False)
        final = root / "scorer" / f"{name}.csv"
        final.parent.mkdir(exist_ok=True)
        rows.loc[rows.partition == "final"].to_csv(final, index=False)
        config = {
            "data": {"local_path": str(table), "feature_columns": FEATURES,
                     "target_column": "sound_level", "source_units": metadata["units"],
                     "units": {name: canonical_unit(unit) for name, unit in metadata["units"].items()}},
            "split": {"file": str(split), "group_column": "group_id"},
            "search": {"engines": list(PROFILES), "seeds": [0], "max_wall_seconds": 60,
                       "operators": ["add", "sub", "mul", "div", "pow_int", "neg"],
                       "max_export_candidates": 100, "candidate_pool_policy": "all",
                       "allow_partial_engines": True, "engine_options": PROFILES},
            "requirements": {"require_units": False, "max_nodes": 100, "max_free_parameters": 30},
            "evaluation": {"mode": "exploratory", "coefficient_fit": "none"},
        }
        refs = prepare(resolve_config(config), folder / "prepared")
        cases.append({"name": name, "prepared": refs, "partitions": validate_groups(rows),
                      "solver_inputs": [{"path": str(p), "sha256": digest(p)} for p in [table, split]],
                      "final": {"path": str(final), "sha256": digest(final)},
                      "scope": "Held-out exact condition tuples" if name.startswith("group")
                      else "All 71.3 m/s rows withheld; train/selection use 31.7,39.6,55.5 m/s"})
    plan = {"identity": identity(), "cases": cases, "profiles": PROFILES, "attempts_planned": 6,
            "source_sha256": SOURCE_SHA, "source_doi": "10.24432/C5VW2C", "license": "CC-BY-4.0",
            "selection_rule": "Eligible minimum Selection RMSE, nodes, parameters, ID; no coefficient refit",
            "scope": "Three deterministic baseline searches, no native-engine leaderboard or exact-law truth",
            "units_scope": "Original Hz,deg,m,m/s,m,dB retained; require_units=false; no dimensional-validity claim",
            "history": "Public observations previously evaluated; not a fresh external test or pretraining holdout",
            "group_scope": "Exact input tuple excluding frequency, not original experiment IDs; related designs may remain",
            "criterion": "All Final rows valid and RMSE lower than training-mean baseline; descriptive utility only",
            "leakage_audit": "No condition tuple or sample crosses partitions; solver CSV/refs contain no Final rows"}
    write_json(root / "plan.json", plan)
    return plan


def discover_campaign(root):
    for case in verify(root)["cases"]:
        refs = case["prepared"]
        for engine in PROFILES:
            verify(root)
            folder = root / "discover" / case["name"] / engine
            if folder.exists():
                raise ValueError("Preserve existing attempt; no silent retries")
            result = run_discovery(refs, read_json(refs["run"]), engine, 0, folder)
            print(case["name"], engine, read_json(result["summary"])["status"], flush=True)


def select_campaign(root):
    if any((root / p).exists() for p in ["choices.json", "scores.json", "test_access.json"]):
        raise ValueError("Choices are immutable after freeze/scoring")
    choices = []
    for case in verify(root)["cases"]:
        for engine in PROFILES:
            folder = root / "discover" / case["name"] / engine
            summary_file = folder / "engine_summary.json"
            summary = read_json(summary_file)
            study = create_study({"prepared_run": case["prepared"]["run"],
                                  "sources": [{"discovery": str(summary_file)}], "purpose": "local_prediction",
                                  "question": case["scope"], "engineering_tolerance": None,
                                  "final_history": "known_exposed"}, run_dir=folder / "study")
            context, records, evidence, _ = load_study(study)
            rows = [{"candidate_id": record["id"], "eligible": row["eligible"],
                     "rmse": row.get("metrics", {}).get("selection", {}).get("rmse"),
                     "n_nodes": row.get("n_nodes"), "n_parameters": row.get("n_parameters")}
                    for record, row in zip(records, evidence, strict=True)]
            result = {"case": case["name"], "engine": engine, "native_status": summary["status"],
                      "wall_seconds": summary.get("wall_seconds"), "reason": summary.get("reason"),
                      "candidate_count": sum(r["kind"] != "native_result" for r in records),
                      "eligible_count": sum(r["eligible"] for r in rows), "selected_id": None,
                      "study": str(study), "study_sha256": digest(study / "study.json")}
            if result["eligible_count"]:
                chosen, _ = select_row(rows, {"selection_policy": "accuracy"})
                record = next(r for r in records if r["id"] == chosen["candidate_id"])
                result.update(selected_id=record["id"], selection_rmse=chosen["rmse"],
                              n_nodes=chosen["n_nodes"], n_parameters=chosen["n_parameters"],
                              equation=equation_text(record["equation"], context["variables"]))
            choices.append(result)
    write_json(root / "choices.json", {"plan_sha256": digest(root / "plan.json"), "choices": choices})


def predict_public(record, frame, context, provenance):
    """Use the frozen CSV-to-working map, including degrees-to-radians."""
    data = provenance["config"]["data"]
    units = {**data["units"], **data.get("source_units", {})}
    X = np.column_stack([
        column_to_working(frame[v["column"]].to_numpy(), provenance, v, units[v["column"]])
        for v in context["variables"]
    ])
    prediction, valid = predict_record(record, X, context["variables"], context["target"])
    prediction = model_output(prediction, provenance)
    return prediction, valid & np.isfinite(prediction)


def score_campaign(root):
    plan = verify(root)
    frozen = read_json(root / "choices.json")
    if frozen["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Plan changed after choices")
    with (root / "test_access.json").open("x", encoding="utf-8") as stream:
        json.dump({"choices_sha256": digest(root / "choices.json"), "tuning_forbidden": True}, stream)
    final = {}
    for case in plan["cases"]:
        source = case["final"]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Final source changed")
        final[case["name"]] = read_observations(source["path"])
    results = []
    for choice in frozen["choices"]:
        result = dict(choice)
        if choice["selected_id"] is None:
            result.update(status="no_eligible_candidate", coverage=0.0, test_metrics={"rmse": None})
        else:
            if digest(Path(choice["study"]) / "study.json") != choice["study_sha256"]:
                raise ValueError("Selected study changed")
            context, records, _, _ = load_study(choice["study"])
            record = next(r for r in records if r["id"] == choice["selected_id"])
            frame = final[choice["case"]]
            case = next(c for c in plan["cases"] if c["name"] == choice["case"])
            prediction, valid = predict_public(record, frame, context, read_json(case["prepared"]["run"]))
            metrics = prediction_metrics(frame.sound_level.to_numpy(), prediction, valid=valid)
            observed = frame.sound_level.to_numpy()
            result["prediction_rows"] = {
                "partition": "Frozen-choice Final", "observed": observed.tolist(),
                "predicted": [float(v) if np.isfinite(v) else None for v in prediction],
                "residual": [float(v) if np.isfinite(v) else None for v in observed - prediction],
                "valid": valid.tolist(),
            }
            result.update(status="scored" if metrics["rmse"] is not None else "invalid_prediction",
                          coverage=float(valid.mean()), test_metrics=metrics)
        results.append(result)
    for result in results:
        baseline = next(r for r in results if r["case"] == result["case"] and r["engine"] == "mean")
        value, reference = result["test_metrics"]["rmse"], baseline["test_metrics"]["rmse"]
        result["beats_training_mean"] = bool(value is not None and reference is not None and value < reference)
    write_json(root / "scores.json", {"choices_sha256": digest(root / "choices.json"),
                                      "attempts": len(results), "results": results,
                                      "case_context": {case["name"]: {
                                          "question": case["scope"],
                                          "truth_status": "Observed UCI Airfoil data; no true formula supplied",
                                          "target_unit": "dB",
                                          "variables": "frequency Hz, angle deg (formula uses working radians), chord m, speed m/s, displacement thickness m",
                                          "scope": "Reported equation retains working coordinates; prediction and errors use public dB. Public historical data, not a fresh external test.",
                                          "source": "https://doi.org/10.24432/C5VW2C",
                                      } for case in plan["cases"]},
                                      "interpretation": "Failure-inclusive predictive audit, not causal discovery or adoption"})
    if __package__:
        from .benchmark_readout import write_report
    else:
        from benchmark_readout import write_report

    print("Open report:", write_report(root / "scores.json", root / "readout"), flush=True)
    print(json.dumps({"scores": str(root / "scores.json"), "results": results}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "discover", "select", "score", "all"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/airfoil"))
    args = parser.parse_args()
    root = args.root.resolve()
    actions = {"prepare": lambda p: prepare_campaign(p, args.source.resolve()),
               "discover": discover_campaign, "select": select_campaign, "score": score_campaign}
    for phase in actions if args.phase == "all" else [args.phase]:
        actions[phase](root)


if __name__ == "__main__":
    main()
