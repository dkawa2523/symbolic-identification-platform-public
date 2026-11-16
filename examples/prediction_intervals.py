"""Advanced prediction intervals: real baseline discovery, explicit choice and independent calibration.

Lengths are supplied in cm/mm and evaluated in m. This is a workflow example,
not blind mechanism recovery or a ten-method benchmark. No optional native library is needed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import (
    assess_prediction_use,
    audit_prediction_interval,
    calibrate_prediction,
    choose_hypothesis,
    compare_equations,
    create_study,
    evaluate_equations,
    export_outputs,
    finalize_hypothesis,
    fit_hypotheses,
    load_model,
    load_prediction_interval,
    run_task,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Path("results/prediction-intervals"), type=Path, help="New result directory")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve() / "calculation"
    if not args.prepare_only and args.output.exists():
        raise ValueError("Result directory already exists; choose a new --output")
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026110101)

    def observations(prefix, groups):
        x = rng.uniform(0.03, 0.30, groups * 3)
        return pd.DataFrame(
            {
                "sample_id": [f"{prefix}{i}" for i in range(len(x))],
                "group": np.repeat([f"{prefix}g{i}" for i in range(groups)], 3),
                "length_cm": 100 * x,
                "response_mm": 1000 * (0.1 + 2 * x + rng.normal(0, 0.005, len(x))),
            }
        )

    frame = observations("problem", 24)
    frame.to_csv(root / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=np.repeat(["train"] * 16 + ["selection"] * 4 + ["final"] * 4, 3)).to_csv(
        root / "split.csv", index=False
    )
    cfg = {
        "data": {
            "local_path": str(root / "table.csv"),
            "feature_columns": ["length_cm"],
            "target_column": "response_mm",
            "units": {"length_cm": "cm", "response_mm": "mm"},
            "domain": {"length_cm": [3, 30], "response_mm": [-1000, 2000]},
        },
        "split": {"file": str(root / "split.csv"), "group_column": "group"},
        "search": {"engines": ["mean", "linear", "dictionary"], "seeds": [7], "max_wall_seconds": 30},
        "constraints": {"max_nodes": 60},
    }
    write(root / "problem.json", cfg)
    if args.prepare_only:
        print(root / "problem.json")
        return
    prepared = run_task({**cfg, "task": "symbolic_stage", "run": {"stage": "prepare"}}, run_dir=root / "prepare")
    refs = {k: str(v) for k, v in prepared.artifacts.items()}
    sources = []
    for engine in cfg["search"]["engines"]:
        discovery = run_task(
            {
                **cfg,
                "task": "symbolic_stage",
                "run": {"stage": "discover", "engine": engine, "seed": 7},
                "stage_inputs": {"prepare": refs},
            },
            run_dir=root / "native" / engine,
        )
        sources.append({"discovery": str(discovery.artifacts["engine_summary"])})
    manual = {
        "id": "user_linear",
        "expression": "a*length_cm+b",
        "parameters": {
            "a": {"value": 1, "trainable": True, "bounds": [0, 4], "unit": "1"},
            "b": {"value": 0, "trainable": True, "bounds": [-1, 1], "unit": "m"},
        },
        "knowledge": "User-proposed linear family; not blind family discovery. Expression uses working m coordinates.",
    }
    plan = {
        "prepared_run": refs["run"],
        "sources": [*sources, {"manual": manual}],
        "purpose": "local_prediction",
        "question": "Can a simple local response be used within this observed length range?",
        "engineering_tolerance": 0.02,
        # Public deterministic onboarding rows have already been used in checks.
        "final_history": "known_exposed",
    }
    initial = create_study(plan, run_dir=root / "initial-study")
    fitted = fit_hypotheses(initial, run_dir=root / "fits", ids=["user_linear@manual"])
    fit_records = json.loads((fitted / "fits.json").read_text())
    plan["sources"] = [*sources, *[{"derived": str(p)} for p in fit_records["artifacts"]]]
    study = create_study(plan, run_dir=root / "study")
    interpretation = evaluate_equations(
        study,
        "interpretation",
        run_dir=root / "interpretation",
        options={
            "paths": [
                {
                    "id": "length_response",
                    "at": {"length_cm": 15.0},
                    "direction": {"length_cm": 1.0},
                    "steps": np.linspace(-12, 15, 55).tolist(),
                    "path_unit": "cm",
                    "scope": "Declared1D operating range3..30cm, fixed source-domain path; model response, not a causal/physical mechanism claim",
                }
            ]
        },
    )
    comparison = compare_equations(study, run_dir=root / "comparison", evaluations=[interpretation])
    calibration_plan = dict(
        alpha=0.1,
        exchangeability="exchangeable_groups",
        group_column="group",
        observation_model="Independent exchangeable synthetic three-observation groups; iid Gaussian noise within this generator",
        max_width=40.0,
        unit="mm",
    )
    decision = choose_hypothesis(
        comparison,
        run_dir=root / "decision",
        candidate_id="user_linear@manual/train_point_fit",
        reason="Predeclared 0.02m local tolerance and40mm full observation interval; explicit informed linear family, no mechanism or extrapolation claim",
        calibration_plan=calibration_plan,
    )
    # The choice is fixed BEFORE calibration or Final inspection.
    calibration = observations("calibration", 40)
    calibration.to_csv(root / "calibration.csv", index=False)
    interval_dir = calibrate_prediction(
        study,
        decision,
        root / "calibration.csv",
        run_dir=root / "interval",
        alpha=0.1,
        exchangeability="exchangeable_groups",
        group_column="group",
        observation_model="Independent exchangeable synthetic three-observation groups; iid Gaussian noise within this generator",
    )
    use = assess_prediction_use(decision, interval_dir, run_dir=root / "use")
    final = finalize_hypothesis(study, use, run_dir=root / "final")
    audit_frame = observations("audit", 40)
    audit_frame.to_csv(root / "audit.csv", index=False)
    audit_path = audit_prediction_interval(interval_dir, root / "audit.csv", output=root / "interval_audit.json")
    model = load_model(final.artifacts["model_bundle"])
    prediction = model.predict(audit_frame)
    interval = load_prediction_interval(interval_dir)
    point, low, high = interval.predict(audit_frame)
    explicit = interval.predict(audit_frame, units={"length_cm": "cm"}, output_unit="mm")
    np.testing.assert_allclose(point, prediction)
    np.testing.assert_allclose([point, low, high], explicit)
    reordered = audit_frame[["response_mm", "group", "length_cm", "sample_id"]]
    np.testing.assert_array_equal(model.predict(reordered), prediction)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_platform_symbolic",
            "predict",
            str(interval_dir),
            "--data",
            str(root / "audit.csv"),
            "--run-dir",
            str(root / "cli-interval"),
        ],
        check=True,
    )
    cli = pd.read_csv(root / "cli-interval/predictions.csv")
    np.testing.assert_allclose(cli[["prediction", "lower", "upper"]].to_numpy(), np.column_stack([point, low, high]))
    result_directory = export_outputs(
        comparison,
        output_dir=args.output,
        decision_dir=use,
        prediction_interval=interval_dir,
        coverage=audit_path,
        audit_dir=root / "final",
    )
    write(
        root / "summary.json",
        {
            "scope": "Synthetic technical onboarding; manual family is informed, not scientific mechanism acceptance",
            "final_metrics_working_m": final.metrics,
            "independent_prediction_support": model.metadata["prediction_support"],
            "interval_audit": json.loads(audit_path.read_text()),
            "public_prediction_unit": "mm",
            "api_cli_and_reordered_columns_match": True,
            "model": str(final.artifacts["model_bundle"]),
            "interval": str(interval_dir),
            "results": str(result_directory),
        },
    )
    print(f"Results: {result_directory}")
    print(f"Figures: {result_directory / 'figures'}")
    print(f"Calculation record: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
