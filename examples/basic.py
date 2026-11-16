"""Small comparison: fixed holdout, one seed, equations and graphs. No Final or adoption."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import (
    compare_equations,
    create_study,
    export_outputs,
    run_task,
    select_representatives,
)
from ml_platform_symbolic.equation_evaluation import load_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Path("results/basic"), type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve() / "calculation"
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    x = rng.uniform(-1, 1, 100)
    frame = pd.DataFrame({"sample_id": [f"row{i}" for i in range(100)], "x": x,
                          "y": 0.5 + 2*x + rng.normal(0, 0.05, 100)})
    frame.to_csv(root / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"]*80 + ["selection"]*20).to_csv(root / "split.csv", index=False)
    cfg = {
        "data": {"local_path": str(root / "data.csv"), "feature_columns": ["x"],
                 "target_column": "y", "units": {"x": "1", "y": "1"}},
        "split": {"file": str(root / "split.csv")},
        "search": {"engines": ["mean", "linear", "dictionary"], "seeds": [17], "max_wall_seconds": 30},
        "evaluation": {"mode": "exploratory"},
    }
    (root / "problem.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(root / "problem.json")
        return
    prepared = run_task({**cfg, "task": "symbolic_stage", "run": {"stage": "prepare"}}, run_dir=root / "prepare")
    refs = {key: str(value) for key, value in prepared.artifacts.items()}
    sources = []
    for engine in cfg["search"]["engines"]:
        discovery = run_task(
            {**cfg, "task": "symbolic_stage", "run": {"stage": "discover", "engine": engine, "seed": 17},
             "stage_inputs": {"prepare": refs}}, run_dir=root / "native" / engine)
        sources.append({"discovery": str(discovery.artifacts["engine_summary"])})
    study = create_study(
        {"prepared_run": refs["run"], "sources": sources, "purpose": "local_prediction",
         "question": "Compare methods on the same fixed Selection rows; not an independent Final result."},
        run_dir=root / "study")
    _, records, rows, _ = load_study(study)
    representatives = select_representatives(records, rows)
    comparison = compare_equations(study, run_dir=root / "comparison", pair_ids=[])
    output = export_outputs(comparison, output_dir=args.output,
                            representative_ids=[record["id"] for record, _ in representatives.values()])
    summary = [{"method": method, "seed": seed, "candidate_id": record["id"],
                "selection_rmse": row["metrics"]["selection"]["rmse"]}
               for (method, seed), (record, row) in representatives.items()]
    (root / "summary.json").write_text(json.dumps({"partition": "Selection", "final_accessed": False,
        "rule": "Minimum eligible finite Selection RMSE per method/seed; exact ties by ID", "representatives": summary},
        indent=2), encoding="utf-8")
    print(f"Results: {output}")
    for row in summary:
        print(f"{row['method']}: Selection RMSE={row['selection_rmse']:.6g}")


if __name__ == "__main__":
    main()
