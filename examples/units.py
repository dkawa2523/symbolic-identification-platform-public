"""Two independent dimensional functions in an explicitly mapped input chart.

Public CSV units are ms, cm and mm. The template uses execution coordinates
elapsed_s and length_m, and jointly fits both unknown functions on Train.
Synthetic demonstration only; compare on Selection without opening Final.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import compare_equations, create_study, export_outputs, run_task


def run(root):
    output = Path(root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = output / "calculation"
    root.mkdir()
    rng = np.random.default_rng(2026110819)
    time_s, length_m = rng.uniform(0.3, 1.3, (2, 90))
    response_m = 2.5 * time_s - 1.7 * length_m
    frame = pd.DataFrame({
        "sample_id": [f"r{i}" for i in range(len(time_s))],
        "time_ms": 1000 * time_s,
        "length_cm": 100 * length_m,
        "response_mm": 1000 * response_m,
    })
    frame.to_csv(root / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 54 + ["selection"] * 18 + ["final"] * 18).to_csv(
        root / "split.csv", index=False
    )
    template = {
        "coordinate_space": "working",
        "ast": {"op": "add", "args": [
            {"param": "baseline"},
            {"op": "add", "args": [
                {"slot": "time_term", "args": [{"var": "x0"}]},
                {"slot": "length_term", "args": [{"var": "x1"}]},
            ]},
        ]},
        "parameters": {},
        "slots": {
            name: {
                "arguments": [{"name": "u", "unit": unit}],
                "unit": "m", "allowed_operators": ["mul"], "max_nodes": 3,
            }
            for name, unit in [("time_term", "s"), ("length_term", "m")]
        },
        "knowledge": {"kind": "hypothesis", "statement": "Known offset plus independent time-only and length-only terms"},
    }
    config = {
        "data": {
            "local_path": str(root / "table.csv"),
            "feature_columns": ["time_ms", "length_cm"], "target_column": "response_mm",
            "units": {"time_ms": "ms", "length_cm": "cm", "response_mm": "mm"},
            "domain": {"time_ms": [300, 1300], "length_cm": [30, 130]},
        },
        "split": {"file": str(root / "split.csv")},
        "input_map": {
            "kind": "affine_chart", "target_columns": ["elapsed_s", "length_m"],
            "target_units": {"elapsed_s": "s", "length_m": "m"},
            "forward": {"matrix": [[0.001, 0], [0, 0.01]], "offset": [-0.2, 0]},
            "inverse": {"matrix": [[1000, 0], [0, 100]], "offset": [200, 0]},
        },
        "requirements": {
            # Global source AST: time chart5 + coefficient/mul2, length chart3 + coefficient/mul2,
            # baseline/add/add3, target division2 = 17. Slot-local limits remain3.
            "allowed_operators": ["add", "mul"], "require_units": True, "max_nodes": 17,
            "fixed_constants": {"baseline": {"value": 0.5, "unit": "m"}},
            "equation_template": template,
        },
        "search": {
            "engines": ["template_enumeration"], "seeds": [7], "max_wall_seconds": 90,
            "candidate_pool_policy": "all",
            "engine_options": {"template_enumeration": {
                "max_structures": 196, "max_slot_nodes": {"time_term": 3, "length_term": 3},
                "starts": [1.0, -1.0, 0.1], "max_nfev": 200,
            }},
        },
    }
    (root / "problem.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    prepared = run_task(
        {**config, "task": "symbolic_stage", "run": {"stage": "prepare"}}, run_dir=root / "prepare"
    )
    refs = {k: str(v) for k, v in prepared.artifacts.items()}
    discovery = run_task({
        **config, "task": "symbolic_stage", "run": {"stage": "discover", "engine": "template_enumeration", "seed": 7},
        "stage_inputs": {"prepare": refs},
    }, run_dir=root / "discovery")
    study = create_study({
        "prepared_run": refs["run"],
        "sources": [{"discovery": str(discovery.artifacts["engine_summary"])}],
        "purpose": "local_prediction", "engineering_tolerance": 1e-6,
        "question": "Does the joint finite grammar recover this synthetic dimensional response?",
        "final_history": "known_exposed",
    }, run_dir=root / "study")
    compared = compare_equations(study, run_dir=root / "comparison")
    report = export_outputs(compared, output_dir=output)
    return {"study": str(study), "comparison": str(compared), "results": str(report),
            "scope": "Synthetic Selection demonstration; Final not opened; no unique component or scientific claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().output)))
