"""Train-fitted prediction versus observed-design ambiguity, with a saved path readout.

This disclosed polynomial example is not blind law discovery. Six bounded dictionary
searches use paired input/noise draws in independent and confounded arms. Supplied
counterexample equations are controls, never search candidates or selection winners.
All choices freeze before fresh scoring inputs/targets are generated. No tuning follows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import choose_hypothesis, compare_equations, create_study, evaluate_equations, export_outputs
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, prepare, read_json, write_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.equation_evaluation.api import load_study, new_directory
from ml_platform_symbolic.equation_evaluation.backends.interpretation import input_reference_geometry
from ml_platform_symbolic.equation_evaluation.numerical import predict_record
from ml_platform_symbolic.metrics import prediction_metrics

SEEDS = [1729, 2718, 3141]
CONTROLS = {"supplied_true": "2*x+z*z", "confounded_alias": "2*z+x*x", "wrong_fixed_slope": "3*x+z*z"}


def run_example(root):
    output = new_directory(root)
    root = new_directory(output / "calculation")
    write_json(root / "plan.json", {
        "data_seeds": SEEDS, "solver_seed": 0, "score_seed_offset": 104729,
        "train_rows": 120, "selection_rows": 40, "score_rows_per_stratum": 256, "noise_sd": 0.01,
        "truth": CONTROLS["supplied_true"], "controls": CONTROLS,
        "driver_sha256": digest(__file__),
        "selection": "Eligible dictionary minimum Selection RMSE, nodes, parameters, ID; no refit",
        "scope": "Disclosed-family demonstration with paired data draws, one deterministic search per arm/draw. No unknown-class, causal or deployment claim.",
        "distance": "Train inputs only; no accuracy threshold, learned OOD decision or target access",
    })
    choices = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-1, 1, (160, 2))
        noise = rng.normal(0, 0.01, 160)
        for arm in ["independent", "confounded"]:
            case_output = output / "comparisons" / f"{arm}_{seed}"
            directory = case_output / "calculation"
            directory.mkdir(parents=True)
            X = raw.copy()
            if arm == "confounded":
                X[:, 1] = X[:, 0]
            frame = pd.DataFrame({"sample_id": [f"r{i}" for i in range(160)], "x": X[:, 0], "z": X[:, 1],
                                  "y": 2 * X[:, 0] + X[:, 1] ** 2 + noise})
            table, split = directory / "table.csv", directory / "split.csv"
            frame.to_csv(table, index=False)
            frame[["sample_id"]].assign(partition=["train"] * 120 + ["selection"] * 40).to_csv(split, index=False)
            cfg = resolve_config({
                "data": {"local_path": str(table), "feature_columns": ["x", "z"], "target_column": "y",
                         "units": {"x": "1", "z": "1", "y": "1"}},
                "split": {"file": str(split)},
                "search": {"engines": ["dictionary"], "seeds": [0],
                           "max_wall_seconds": 30, "candidate_pool_policy": "all",
                           "engine_options": {"dictionary": {"max_degree": 2, "max_terms": 6, "ridge": 1e-8}}},
                "requirements": {"allowed_operators": ["add", "mul"], "require_units": True, "max_nodes": 60},
                "evaluation": {"mode": "exploratory", "coefficient_fit": "none"},
            })
            refs = prepare(cfg, directory / "prepared")
            discovery = run_discovery(refs, read_json(refs["run"]), "dictionary", 0, directory / "search")
            summary = read_json(discovery["summary"])
            study = create_study({
                "prepared_run": refs["run"],
                "sources": [{"discovery": discovery["summary"]}, *[
                    {"manual": {"id": name, "expression": expression,
                                "knowledge": "Supplied counterexample control, not a discovered equation"}}
                    for name, expression in CONTROLS.items()]],
                "question": "How far do requested paths leave observed inputs? Does accurate same-design prediction identify a relation?",
                "data_kind": "Disclosed synthetic polynomial control", "engineering_tolerance": 0.05,
                "final_history": "declared_unseen",
            }, run_dir=directory / "study")
            context, records, rows, _ = load_study(study)
            interpretation = evaluate_equations(study, "interpretation", run_dir=directory / "interpretation", options={
                "paths": [
                    {"id": "vary_x_at_fixed_z", "at": {"x": 0, "z": 0.2}, "direction": {"x": 1, "z": 0},
                     "steps": np.linspace(-0.8, 0.8, 21).tolist(), "path_unit": "1",
                     "scope": "In marginal ranges, but leaves the confounded x=z design. A model response, not an observed effect."},
                    {"id": "diagonal_extrapolation", "at": {"x": 0, "z": 0}, "direction": {"x": 1, "z": 1},
                     "steps": np.linspace(-0.8, 2, 29).tolist(), "path_unit": "1",
                     "scope": "Partly outside Train ranges. Aliased controls agree along x=z even outside the range."},
                ]})
            comparison = compare_equations(study, run_dir=directory / "comparison", evaluations=[interpretation])
            eligible = [r for r, record in zip(rows, records, strict=True)
                        if r["eligible"] and record["origin"].get("engine") == "dictionary"]
            selected = min(eligible, key=lambda r: (r["metrics"]["selection"]["rmse"], r["n_nodes"], r["n_parameters"], r["id"])) if eligible else None
            decision = choose_hypothesis(comparison, run_dir=directory / "choice", candidate_id=selected["id"],
                reason="Predeclared Selection rule among actual dictionary candidates. Supplied controls are not eligible for this demonstration's choice.") if selected else None
            export_outputs(comparison, output_dir=case_output, decision_dir=decision)
            choices.append({"arm": arm, "data_seed": seed, "study": str(study), "study_sha256": digest(study / "study.json"),
                            "selected_id": selected["id"] if selected else None, "native_status": summary["status"],
                            "reason": summary.get("reason"), "selection_rmse": selected["metrics"]["selection"]["rmse"] if selected else None})
    write_json(root / "choices.json", {"plan_sha256": digest(root / "plan.json"), "choices": choices})
    choice_hash = digest(root / "choices.json")
    write_json(root / "score_access.json", {"choices_sha256": choice_hash, "no_reselection": True})
    scores, scoring_data = [], {}
    (root / "score_only").mkdir()
    for choice in choices:
        context, records, _, arrays = load_study(choice["study"])
        if digest(Path(choice["study"]) / "study.json") != choice["study_sha256"]:
            raise ValueError("Frozen study changed")
        rng = np.random.default_rng(choice["data_seed"] + 104729)
        for stratum, low, high in [("same_design", -1, 1), ("independent_inputs", -1, 1), ("range_extrapolation", 1.2, 2)]:
            X = rng.uniform(low, high, (256, 2))
            if choice["arm"] == "confounded" and stratum in {"same_design", "range_extrapolation"}:
                X[:, 1] = X[:, 0]
            clean = 2 * X[:, 0] + X[:, 1] ** 2
            y = clean + rng.normal(0, 0.01, len(X))
            score_file = root / "score_only" / f"{choice['arm']}_{choice['data_seed']}_{stratum}.csv"
            pd.DataFrame({"x": X[:, 0], "z": X[:, 1], "clean_y": clean, "observed_y": y}).to_csv(score_file, index=False)
            scoring_data[score_file.name] = digest(score_file)
            geometry = input_reference_geometry(arrays["train_X"], X, ["x", "z"])
            for record in records:
                if record["id"] != choice["selected_id"] and record.get("candidate_id") not in CONTROLS:
                    continue
                prediction, valid = predict_record(record, X, context["variables"], context["target"])
                scores.append({"arm": choice["arm"], "data_seed": choice["data_seed"], "stratum": stratum,
                               "id": record["id"], "selected": record["id"] == choice["selected_id"],
                               "n_rows": len(X), "coverage": float(valid.mean()),
                               "clean_rmse": prediction_metrics(clean, prediction, valid=valid)["rmse"],
                               "observed_rmse": prediction_metrics(y, prediction, valid=valid)["rmse"],
                               "n_outside_marginal_range": geometry["n_outside_marginal_range"],
                               "max_nearest_distance": geometry["max_nearest_distance"]})
        if choice["selected_id"] is None:
            scores.append({**choice, "score_status": "no_eligible_dictionary_candidate"})
    if digest(root / "choices.json") != choice_hash:
        raise ValueError("Choices changed after scoring access")
    result = {"choices_sha256": choice_hash, "scoring_data_sha256": scoring_data, "results": scores,
              "scope": "Separate frozen external audit, not a Final adoption record. All strata use identical rows for all subjects. Supplied controls are not discovered results. Distances and prediction errors cannot prove a law."}
    write_json(root / "scores.json", result)
    (output / "tables").mkdir()
    pd.DataFrame(scores).to_csv(output / "tables/scores.csv", index=False)
    (output / "README.txt").write_text("Observed-input support comparison\n\nOpen comparisons/<input condition and seed>/ for equations, figures and tables.\nAll score rows: tables/scores.csv. Calculation records: calculation/.\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    result = run_example(parser.parse_args().output)
    print(pd.DataFrame(result["results"]).to_string(index=False))
