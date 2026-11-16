"""Synthetic observed-response use, deliberately distinct from level prediction.

The supplied formulas are hypotheses, not blind discovery. Two have the same
slope but wrong absolute levels; a constant has the wrong response. No Final is
opened, and the detached result contains no deployment predictor.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import (
    choose_hypothesis,
    compare_equations,
    create_study,
    export_outputs,
    export_result,
    finalize_hypothesis,
)


def run(directory):
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    directory = output / "calculation"
    directory.mkdir()
    x = np.arange(12, dtype=float)
    frame = pd.DataFrame({"sample_id": [f"{i:03d}" for i in range(12)], "input": x, "response": 2 * x + 5})
    frame.to_csv(directory / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 6 + ["selection"] * 4 + ["final"] * 2).to_csv(
        directory / "split.csv", index=False
    )
    study = create_study(
        {
            "problem": {
                "data": {
                    "local_path": str(directory / "table.csv"),
                    "feature_columns": ["input"],
                    "target_column": "response",
                    "units": {"input": "1", "response": "1"},
                },
                "split": {"file": str(directory / "split.csv")},
                "search": {"engines": ["linear"], "operators": ["add", "mul"]},
            },
            "sources": [
                {
                    "manual": {
                        "id": name,
                        "expression": expression,
                        "knowledge": "Fixed synthetic hypothesis, not a physical law or search result",
                    }
                }
                for name, expression in [("level_low", "2*input"), ("level_high", "2*input+1005"), ("flat", "5")]
            ],
            "purpose": "response_comparison",
            "final_history": "known_exposed",
            "data_kind": "synthetic deterministic technical example",
            "question": "Which supplied formula matches the two observed changes? Absolute levels are a separate question.",
            "response_validation": {
                "pairs": [["006", "007"], ["008", "009"]],
                "tolerance": {"value": 0.001, "unit": "1"},
                "scope": "The specified second-minus-first changes only; no causal or unobserved-path claim",
            },
        },
        run_dir=directory / "study",
    )
    comparison = compare_equations(study, run_dir=directory / "comparison")
    choice = choose_hypothesis(
        comparison,
        run_dir=directory / "choice",
        candidate_id="level_high@manual",
        retain=["level_low@manual", "flat@manual"],
        reason="Illustrate identical specified responses despite wrong absolute level; not deployable level prediction",
    )
    result = finalize_hypothesis(study, choice, run_dir=directory / "response-result")
    package = export_result(
        comparison,
        run_dir=directory / "handoff",
        decision_dir=choice,
        representative_ids=["level_high@manual", "level_low@manual", "flat@manual"],
    )
    print(result.artifacts["response_result"])
    print(export_outputs(package, output_dir=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
