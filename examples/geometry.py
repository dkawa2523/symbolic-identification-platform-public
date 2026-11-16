"""A small synthetic source/chart/save workflow, not an unknown-law benchmark.

The linear baseline really fits Train; only the comparison and fixed choice
can authorize the subsequent Final audit. Inputs remain three original
composition columns even though the equation runs on two coordinates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import export_outputs
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import prepare, read_json, write_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.equation_comparison import compare_equations
from ml_platform_symbolic.equation_evaluation import create_study
from ml_platform_symbolic.equation_evaluation.api import load_study
from ml_platform_symbolic.model import load_model
from ml_platform_symbolic.numerical_fidelity import compare_predictions, signal_scale
from ml_platform_symbolic.selection import choose_hypothesis, finalize_hypothesis


def run_example(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Example output must be new/empty; do not overwrite Final history")
    output = directory.resolve()
    directory = output / "calculation"
    directory.mkdir(parents=True, exist_ok=False)
    X = np.random.default_rng(9011).dirichlet([2.0, 2.0, 2.0], size=90)
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    frame["response"] = 2 + 3 * X[:, 0] + 5 * X[:, 1]  # Synthetic MPa, not equipment observations.
    frame["sample_id"] = [f"recipe-{i}" for i in range(len(frame))]
    frame["recipe_id"] = frame["sample_id"]
    table, split = directory / "table.csv", directory / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 54 + ["selection"] * 18 + ["final"] * 18).to_csv(
        split, index=False
    )
    config = {
        "data": {
            "local_path": str(table),
            "feature_columns": ["a", "b", "c"],
            "target_column": "response",
            "units": {"a": "1", "b": "1", "c": "1", "response": "MPa"},
        },
        "split": {"file": str(split), "group_column": "recipe_id"},
        "search": {
            "engines": ["linear"],
            "seeds": [0],
            "max_wall_seconds": 30,
            "engine_options": {"linear": {"ridge": 0.0}},
        },
        "requirements": {"allowed_operators": ["add", "mul"], "require_units": True, "max_nodes": 30},
        "domain": {
            "kind": "affine",
            "columns": ["a", "b", "c"],
            "space_id": "composition",
            "bounds": [[0, 1]] * 3,
            "equalities": {"matrix": [[1, 1, 1]], "rhs": [1]},
        },
        "input_map": {
            "kind": "affine_chart",
            "space_id": "independent_composition",
            "target_columns": ["u", "v"],
            "target_units": {"u": "1", "v": "1"},
            "forward": {"matrix": [[1, 0, 0], [0, 1, 0]], "offset": [0, 0]},
            "inverse": {"matrix": [[1, 0], [0, 1], [-1, -1]], "offset": [0, 0, 1]},
        },
    }
    refs = prepare(resolve_config(config), directory / "prepared")
    provenance = read_json(refs["run"])
    search = run_discovery(refs, provenance, "linear", 0, directory / "discover")
    study = create_study(
        {
            "prepared_run": refs["run"],
            "sources": [{"discovery": search["summary"]}],
            "engineering_tolerance": 0.01,
            "final_history": "declared_unseen",
            "data_kind": "synthetic wiring demonstration; not independent scientific validation",
        },
        run_dir=directory / "study",
    )
    _, _, rows, _ = load_study(study)
    eligible = [r for r in rows if r["eligible"]]
    selected = min(eligible, key=lambda row: row["metrics"]["selection"]["rmse"])["id"]
    comparison = compare_equations(study, run_dir=directory / "comparison")
    decision = choose_hypothesis(
        comparison,
        run_dir=directory / "decision",
        candidate_id=selected,
        reason="Synthetic local prediction demonstration; fixed Selection rule and tolerance, no interval requested",
    )
    result = finalize_hypothesis(study, decision, run_dir=directory / "final")
    model = load_model(result.artifacts["model_bundle"])
    actual = model.predict(frame[["c", "a", "b"]])
    expected = frame["response"].to_numpy()
    parity = compare_predictions(expected, actual, scale=signal_scale(expected), relative_tolerance=1e-10)
    if not parity["passed"]:
        raise ValueError("Saved source-coordinate replay failed")
    try:
        model.predict(pd.DataFrame({"a": [0.6], "b": [0.6], "c": [0.6]}))
    except ValueError as error:
        invalid = str(error)
    else:
        raise ValueError("Original composition domain was lost in saved inference")
    summary = {
        "kind": "synthetic_geometry_workflow",
        "saved_model": str(result.artifacts["model_bundle"]),
        "source_columns": ["a", "b", "c"],
        "execution_columns": ["u", "v"],
        "public_target_unit": "MPa",
        "working_target_unit": model.schema["target"]["unit"],
        "replay": parity,
        "outside_original_domain_rejected": invalid,
        "meaning": "Actual baseline fit and saved geometry; not physical-law discovery or all-method acceptance",
    }
    write_json(directory / "geometry_result.json", summary)
    export_outputs(comparison, output_dir=output, decision_dir=decision, audit_dir=directory / "final")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    print(run_example(parser.parse_args().output))
