"""Installed-package extension check, not a blind-discovery scientific benchmark."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import (
    choose_hypothesis,
    compare_equations,
    create_study,
    evaluate_equations,
    finalize_hypothesis,
    load_model,
)
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, prepare, read_json, write_json
from ml_platform_symbolic.domains import register_runtime_extension
from ml_platform_symbolic.engines.runner import run_discovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = register_runtime_extension("workbench_demo_extension", "1.0.0")
    x = np.concatenate([np.linspace(0.2, 0.9, 40), np.linspace(1.3, 2.7, 40)])
    rng = np.random.default_rng(541901)
    rng.shuffle(x)
    frame = pd.DataFrame(dict(sample_id=[f"e{i}" for i in range(len(x))], x=x, y=1 + 2 * x + 0.4 * x * x))
    frame.to_csv(root / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 48 + ["selection"] * 16 + ["final"] * 16).to_csv(
        root / "split.csv", index=False
    )
    config = dict(
        data=dict(
            local_path=str(root / "data.csv"), feature_columns=["x"], target_column="y", units={"x": "1", "y": "1"}
        ),
        split=dict(file=str(root / "split.csv")),
        search=dict(
            engines=["numpy_polynomial_demo"],
            seeds=[0],
            max_wall_seconds=30,
            engine_options={"numpy_polynomial_demo": {"degree": 2}},
        ),
        requirements=dict(allowed_operators=["add", "mul", "pow_int"], max_nodes=40, require_units=True),
        domain=dict(
            kind="extension",
            name="demo_disconnected",
            version="1.0.0",
            space_id="disconnected_input",
            columns=["x"],
            settings={"intervals": [[0.2, 0.9], [1.3, 2.7]]},
        ),
    )
    refs = prepare(resolve_config(config), root / "prepared")
    job = run_discovery(refs, read_json(refs["run"]), "numpy_polynomial_demo", 0, root / "discovery")
    summary = read_json(job["summary"])
    assert summary["status"] == "ok", summary
    study = create_study(
        dict(
            prepared_run=refs["run"],
            sources=[{"discovery": job["summary"]}],
            engineering_tolerance=1e-6,
            final_history="known_exposed",
        ),
        run_dir=root / "study",
    )
    evaluation = evaluate_equations(study, "region_membership_demo", run_dir=root / "analysis")
    result = read_json(evaluation / "result.json")
    assert result["outcome"] == "ok" and all(r["all_feasible"] for r in result["results"]), result
    comparison = compare_equations(study, run_dir=root / "comparison", evaluations=[evaluation])
    candidate = read_json(comparison / "comparison.json")["candidates"][0]["id"]
    choice = choose_hypothesis(
        comparison,
        run_dir=root / "choice",
        candidate_id=candidate,
        reason="Technical polynomial/extension acceptance; not an unknown-family claim",
    )
    final = finalize_hypothesis(study, choice, run_dir=root / "final")
    prediction = load_model(final.artifacts["model_bundle"]).predict(frame)
    np.testing.assert_allclose(prediction, frame.y, rtol=2e-14, atol=0)
    write_json(root / "extensions.json", [{"module": runtime["module"], "version": runtime["version"]}])
    write_json(
        root / "verification.json",
        dict(
            outcome="ok",
            runtime=runtime,
            native_status=summary["native_execution_status"],
            native_check="See candidate archive",
            evaluation=result,
            model=str(final.artifacts["model_bundle"]),
            model_sha256=digest(final.artifacts["model_bundle"]),
            max_absolute_error=float(max(abs(prediction - frame.y))),
            scope="Installed third-party polynomial adapter and explicit region initialized in separate native and evaluator workers. Declared degree family; no general SR or optimizer uncertainty claim. Synthetic known-exposed Final is a technical replay check, not unseen validation.",
        ),
    )
    print(final.artifacts["model_bundle"])


if __name__ == "__main__":
    main()
