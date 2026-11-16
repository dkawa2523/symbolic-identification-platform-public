"""One requirements document: known structure plus a fitted/discovered correction.

Default: complete supplied family fitted on Train. --native: PySR learns h(z).
--enumerate: deterministic bounded single-slot search in the core runtime.
Stops at comparison, without scientific adoption or opening Final.
"""

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
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native", action="store_true")
    mode.add_argument("--enumerate", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = output / "calculation"
    root.mkdir()
    rng = np.random.default_rng(2026110807)
    x, z = rng.uniform(-1, 1, (2, 120))
    frame = pd.DataFrame(
        {
            "sample_id": [f"r{i}" for i in range(len(x))],
            "x": x,
            "z": z,
            "y": 2 * x + 0.5 * z * z + rng.normal(0, 0.01, len(x)),
        }
    )
    frame.to_csv(root / "table.csv", index=False)
    frame[["sample_id"]].assign(
        partition=["train"] * 72 + ["selection"] * 24 + ["final"] * 24
    ).to_csv(root / "split.csv", index=False)
    operators = ["add", "sub", "mul"]
    correction = (
        {"slot": "h", "args": [{"var": "x1"}]}
        if (args.native or args.enumerate)
        else {
            "op": "mul",
            "args": [
                {"param": "a"},
                {"op": "mul", "args": [{"var": "x1"}, {"var": "x1"}]},
            ],
        }
    )
    template = {
        "coordinate_space": "working",
        "ast": {"op": "add", "args": [{"ref": "base"}, correction]},
        "components": {
            "base": {
                "ast": {"op": "mul", "args": [{"param": "k"}, {"var": "x0"}]},
                "description": "Declared fixed contribution; the correction only receives z",
            }
        },
        "parameters": {}
        if (args.native or args.enumerate)
        else {
            "a": {
                "value": 1.0,
                "trainable": True,
                "unit": "1",
                "bounds": [-2.0, 2.0],
                "origin": "user_supplied_complete_family",
            }
        },
        "slots": {
            "h": {
                "arguments": [{"name": "u", "unit": "1"}],
                "unit": "1",
                "allowed_operators": ["mul"] if args.enumerate else operators,
                "max_nodes": 5 if args.enumerate else 10,
                "description": "Unknown z-only function, not the observation residual",
            }
        }
        if (args.native or args.enumerate)
        else {},
        "knowledge": {
            "kind": "hypothesis",
            "statement": "A fixed 2*x term plus a z-only correction",
        },
    }
    engine = (
        "template_enumeration"
        if args.enumerate
        else "pysr"
        if args.native
        else "known_form"
    )
    cfg = {
        "data": {
            "local_path": str(root / "table.csv"),
            "feature_columns": ["x", "z"],
            "target_column": "y",
            "units": {"x": "1", "z": "1", "y": "1"},
            "domain": {"x": [-1.0, 1.0], "z": [-1.0, 1.0]},
        },
        "split": {"file": str(root / "split.csv")},
        "requirements": {
            "allowed_operators": operators,
            "fixed_constants": {"k": {"value": 2.0, "unit": "1"}},
            "require_units": True,
            "max_nodes": 40,
            "equation_template": template,
        },
        "search": {
            "engines": [engine],
            "seeds": [7],
            "max_wall_seconds": 240,
            "engine_options": {
                "template_enumeration": {"max_structures": 128, "max_slot_nodes": 5}
            }
            if args.enumerate
            else {
                "pysr": {
                    "niterations": 12,
                    "populations": 2,
                    "population_size": 40,
                    "ncycles_per_iteration": 80,
                    "seconds": 30,
                }
            }
            if args.native
            else {},
        },
    }
    (root / "problem.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    prepared = run_task(
        {**cfg, "task": "symbolic_stage", "run": {"stage": "prepare"}},
        run_dir=root / "prepare",
    )
    refs = {k: str(v) for k, v in prepared.artifacts.items()}
    discovery = run_task(
        {
            **cfg,
            "task": "symbolic_stage",
            "run": {"stage": "discover", "engine": engine, "seed": 7},
            "stage_inputs": {"prepare": refs},
        },
        run_dir=root / "discovery",
    )
    study = create_study(
        {
            "prepared_run": refs["run"],
            "sources": [{"discovery": str(discovery.artifacts["engine_summary"])}],
            "purpose": "local_prediction",
            "engineering_tolerance": 0.1,
            "question": "Does this declared family predict the frozen Selection observations?",
            "final_history": "known_exposed",
        },
        run_dir=root / "study",
    )
    comparison = compare_equations(study, run_dir=root / "comparison")
    report = export_outputs(comparison, output_dir=output)
    print(
        json.dumps(
            {
                "results": str(report),
                "scope": "Selection comparison only; Final not opened; no adoption",
            }
        )
    )


if __name__ == "__main__":
    main()
