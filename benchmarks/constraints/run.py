"""Paired finite-response benchmark; all choices freeze before independent scoring."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import compare_equations, create_study, export_outputs
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.constraints import prediction_constraints
from ml_platform_symbolic.data import digest, prepare, read_json, write_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.equation_evaluation.api import load_study
from ml_platform_symbolic.equation_evaluation.numerical import predict_record
from ml_platform_symbolic.evaluation import select_row
from ml_platform_symbolic.expressions import equation_text
from ml_platform_symbolic.input_review import format_input_review, review_inputs
from ml_platform_symbolic.metrics import prediction_metrics


def response_conditions(points):
    return {"probes": [{"x": float(x)} for x in points], "response_bounds": [1.0, 3.0],
            "response_checks": [{"name": "increasing", "feature": "x", "order": 1,
                                 "sign": 1, "step": 1e-4, "tolerance": 1e-5,
                                 "points": [{"x": float(x)} for x in points],
                                 "knowledge": "User-supplied increasing bounded response assumption"}]}


def prepare_run(root, engines, seeds, seconds, *, test_seed=None, known_relation=False, settings=None):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261113)
    x = np.r_[rng.uniform(0.05, 0.65, 16), rng.uniform(0.05, 0.65, 24)]
    frame = pd.DataFrame({"sample_id": [f"row{i:03}" for i in range(len(x))], "x": x,
                          "y": 1 + 2*x/(1+x) + rng.normal(0, 0.05, len(x))})
    frame.to_csv(root / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"]*16+["selection"]*24).to_csv(root / "split.csv", index=False)
    # Scoring-only coordinates and clean response. Never supplied to discovery.
    final_rng = np.random.default_rng(test_seed) if test_seed is not None else None
    test_x = (np.r_[final_rng.uniform(0.06, 0.64, 61), final_rng.uniform(0.75, 3.95, 81)]
              if final_rng is not None else np.r_[np.linspace(0.06, 0.64, 61), np.linspace(0.75, 3.95, 81)])
    pd.DataFrame({"x": test_x, "y": 1 + 2*test_x/(1+test_x),
                  "region": ["interpolation"]*61+["extrapolation"]*81}).to_csv(root / "test.csv", index=False)
    search_conditions = response_conditions(np.linspace(0.01, 3.99, 17))
    independent_conditions = response_conditions(final_rng.uniform(0.015, 3.985, 103)
        if final_rng is not None else np.linspace(0.015, 3.985, 103))
    if final_rng is not None:
        write_json(root / "test_conditions.json", independent_conditions)
        write_json(root / "test_lock.json", {"state": "generated_not_reviewed", "seed": test_seed,
            "test_sha256": digest(root / "test.csv"),
            "conditions_sha256": digest(root / "test_conditions.json"),
            "protocol": "Do not open Test or conditions during development. Freeze choices before one score call. "
                        "This is a procedural lock, not filesystem access control."})
    options = {
        "template_enumeration": {"objective": "mse", "max_structures": 256, "max_slot_nodes": 7, "max_nfev": 100, "starts": [1.0, -1.0]},
        "pysr": {"objective": "mse", "niterations": 20, "populations": 3, "population_size": 30,
                 "ncycles_per_iteration": 100, "seconds": seconds*0.7},
        "parfam": {"objective": "mse", "family": "rational", "functions": [], "degree": 2, "input_degree": 2,
                   "denominator_degree": 1, "maxiter": 40, "repetitions": 1,
                   "seconds": seconds*0.7, "candidate_scope": "completed_families"},
    }
    if settings is not None:
        options = settings["methods"]
    base = {
        "data": {"local_path": str(root / "data.csv"), "feature_columns": ["x"], "target_column": "y",
                 "units": {"x": "1", "y": "1"}, "domain": {"x": [0, 4]}},
        "split": {"file": str(root / "split.csv")},
        "search": {"engines": engines, "operators": ["add", "sub", "mul", "div", "pow_int"],
                   "seeds": seeds, "max_wall_seconds": seconds, "allow_partial_engines": True,
                   "engine_options": {engine: options[engine] for engine in engines}},
        "constraints": {"max_nodes": 100}, "evaluation": {"mode": "exploratory"},
    }
    if settings is not None:
        base['search']['operators'] = settings['operators']
        base['constraints']['max_nodes'] = settings['datasets']['synthetic']['max_nodes']
        if 'e2e' in engines:
            base['model_assets'] = {'e2e': {'manifest_file': str(Path('pretrained/e2e/manifest.json').resolve())}}
    if known_relation:
        base["constraints"]["equation_template"] = {
            "coordinate_space": "working",
            "ast": {"op": "add", "args": [{"const": 1.0}, {"slot": "h", "args": [{"var": "x0"}]}]},
            "parameters": {}, "components": {},
            "slots": {"h": {"arguments": [{"name": "x0", "unit": "1"}], "unit": "1",
                "allowed_operators": base["search"]["operators"], "max_nodes": 7}},
            "knowledge": {"kind": "hypothesis", "statement": "Declared offset 1 plus unknown h(x); response constraints apply to the full output"}}
    prepared = {}
    for arm in (settings["arms"] if settings is not None else ("none", "filter", "search")):
        cfg = deepcopy(base)
        cfg["search"]["response_mode"] = "search" if arm == "search" else "filter"
        if arm != "none":
            cfg["constraints"].update(search_conditions)
            if settings is not None and arm == "search":
                cfg["constraints"]["response_enforcement"] = settings["response_enforcement"]
        if known_relation:
            # Use the same capability, asset and Train-only adapter checks as the
            # public input review. Never drop the template for an unsupported engine.
            review = review_inputs(cfg)
            if review["status"] != "inputs_valid":
                raise ValueError(format_input_review(review))
        prepared[arm] = {k: str(v) for k, v in prepare(resolve_config(cfg), root / arm / "prepare").items()}
    write_json(root / "plan.json", {"engines": engines, "seeds": seeds, "seconds": seconds,
        "prepared": prepared, "data_sha256": digest(root / "data.csv"), "split_sha256": digest(root / "split.csv"),
        "test_sha256": digest(root / "test.csv"),
        **({"test_conditions_file": str(root / "test_conditions.json"),
            "test_conditions_sha256": digest(root / "test_conditions.json"), "test_seed": test_seed,
            "test_state": "generated_not_reviewed"} if final_rng is not None
           else {"independent_conditions": independent_conditions}),
        "known_relation": known_relation,
        "scope": "One small synthetic finite-response challenge, not general method ranking or physical discovery",
        "truth": "Scorer only: 1+2*x/(1+x). Search receives only the explicitly declared template, if enabled, not the unknown slot or its coefficients.",
        "pretraining_overlap": "unknown" if "e2e" in engines else "no pretrained models in this campaign",
        "settings": settings})


def attempts(root, plan):
    for arm, refs in plan["prepared"].items():
        for engine in plan["engines"]:
            for seed in plan["seeds"]:
                yield arm, engine, seed, refs, root / arm / engine / str(seed)


def discover(root, engine_filter=None):
    plan = read_json(root / "plan.json")
    for arm, engine, seed, refs, directory in attempts(root, plan):
        if engine_filter is not None and engine != engine_filter:
            continue
        if directory.exists():
            raise ValueError(f"Preserve prior attempt: {directory}")
        paths = run_discovery(refs, read_json(refs["run"]), engine, seed, directory)
        print(arm, engine, seed, read_json(paths["summary"])["status"], flush=True)


def select(root):
    if (root / "choices.json").exists() or (root / "scores.json").exists():
        raise ValueError("Choices are frozen; create a new campaign instead of reselecting")
    plan, choices = read_json(root / "plan.json"), []
    for arm, engine, seed, refs, directory in attempts(root, plan):
        summary = read_json(directory / "engine_summary.json")
        row = dict(arm=arm, engine=engine, seed=seed, execution_status=summary["status"],
                   reason=summary.get("reason"), wall_seconds=summary.get("wall_seconds"), selected_id=None)
        study = create_study({"prepared_run": refs["run"],
            "sources": [{"discovery": str(directory / "engine_summary.json")}],
            "purpose": "local_prediction", "engineering_tolerance": None, "final_history": "unknown",
            "question": plan.get("scope", "Compare fixed method outcomes under the recorded common conditions.")},
            run_dir=directory / "study")
        context, records, evidence, _ = load_study(study)
        candidates = [{"candidate_id": r["id"], "eligible": e["eligible"],
                       "rmse": e.get("metrics", {}).get("selection", {}).get("rmse"),
                       "n_nodes": e.get("n_nodes"), "n_parameters": e.get("n_parameters")}
                      for r, e in zip(records, evidence, strict=True)]
        row.update(candidate_count=sum(r["kind"] != "native_result" for r in records), eligible_count=sum(c["eligible"] for c in candidates),
                   study=str(study), study_sha256=digest(study / "study.json"))
        if row["eligible_count"]:
            selected, _ = select_row(candidates, {"selection_policy": "accuracy"})
            record = next(r for r in records if r["id"] == selected["candidate_id"])
            row.update(selected_id=record["id"], selection_rmse=selected["rmse"],
                       n_nodes=selected["n_nodes"], n_parameters=selected["n_parameters"],
                       equation=equation_text(record["equation"], context["variables"]))
        comparison = compare_equations(study, run_dir=directory / "comparison")
        export_outputs(comparison, output_dir=directory / "output")
        choices.append(row)
    write_json(root / "choices.json", {"plan_sha256": digest(root / "plan.json"), "choices": choices})


def score(root):
    plan, frozen = read_json(root / "plan.json"), read_json(root / "choices.json")
    if frozen["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Plan changed after selection")
    if digest(root / "test.csv") != plan["test_sha256"]:
        raise ValueError("Scoring data changed")
    with (root / "test_access.json").open("x", encoding="utf-8") as stream:
        stream.write('{"tuning_forbidden":true}')
    if "test_conditions_file" in plan:
        condition_file = Path(plan["test_conditions_file"])
        if digest(condition_file) != plan["test_conditions_sha256"]:
            raise ValueError("Scoring conditions changed")
        independent_conditions = read_json(condition_file)
        write_json(root / "test_lock.json", {**read_json(root / "test_lock.json"), "state": "opened_once_for_scoring",
            "choices_sha256": digest(root / "choices.json")})
    else:
        independent_conditions = plan["independent_conditions"]
    frame, rows = pd.read_csv(root / "test.csv"), []
    for choice in frozen["choices"]:
        row = dict(choice)
        if choice["selected_id"] is not None:
            study = Path(choice["study"])
            if digest(study / "study.json") != choice["study_sha256"]:
                raise ValueError("Selected study changed")
            context, records, _, _ = load_study(study)
            record = next(r for r in records if r["id"] == choice["selected_id"])
            def predict(X, record=record, context=context):
                values, valid = predict_record(record, X, context["variables"], context["target"])
                return np.where(valid, values, np.nan)
            prediction = predict(frame[["x"]].to_numpy())
            for region in frame.region.unique():
                mask = frame.region == region
                row[region + "_rmse"] = prediction_metrics(frame.y.to_numpy()[mask], prediction[mask])["rmse"]
            checks = prediction_constraints(predict, context["variables"], context["target"], independent_conditions)
            row["independent_constraints_pass"] = all(c["status"] == "pass" for c in checks)
            row["independent_constraints"] = checks
            prediction_path = root / "test_predictions" / choice["arm"] / f"{choice['engine']}-{choice['seed']}.csv"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            frame.assign(predicted=prediction).to_csv(prediction_path, index=False)
            row["predictions_file"] = str(prediction_path)
        rows.append(row)
    write_json(root / "scores.json", {"choices_sha256": digest(root / "choices.json"), "attempts": rows,
                                      "scope": plan["scope"], "scoring_grid": independent_conditions})
    pd.DataFrame([{k: v for k, v in row.items() if k != "independent_constraints"} for row in rows]).to_csv(
        root / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "discover", "select", "score"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engines", nargs="+", choices=["pysr", "parfam", "template_enumeration", "torchsisso", "e2e", "physo"], default=["pysr", "parfam"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--settings", type=Path, help="Explicit paired evaluation profile; no default changes to historical protocol")
    parser.add_argument("--engine", help="Discover only this method in its own native environment")
    parser.add_argument("--test-seed", type=int, help="Fresh procedural holdout; values stay outside plan until score")
    parser.add_argument("--known-relation", action="store_true", help="Same known offset/unknown slot for both template-capable methods")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == "prepare":
        prepare_run(root, args.engines, args.seeds, args.seconds, test_seed=args.test_seed, known_relation=args.known_relation, settings=read_json(args.settings) if args.settings else None)
    elif args.stage == "discover":
        if args.engine is not None and args.engine not in read_json(root / "plan.json")["engines"]:
            raise ValueError("Requested engine is not in frozen plan")
        discover(root, args.engine)
    else:
        {"select": select, "score": score}[args.stage](root)


if __name__ == "__main__":
    main()
