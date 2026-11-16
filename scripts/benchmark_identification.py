"""Small staged scientific challenge using the existing discovery/evaluation boundary.

prepare -> discover -> select -> score. Never tune from score: create a new study.
Public data are predictive audits, not secret formulas or structure-recovery truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, prepare, read_json, read_observations, write_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.equation_evaluation import create_study
from ml_platform_symbolic.equation_evaluation.api import load_study
from ml_platform_symbolic.equation_evaluation.numerical import predict_record
from ml_platform_symbolic.evaluation import select_row
from ml_platform_symbolic.expressions import equation_text
from ml_platform_symbolic.metrics import prediction_metrics

DATA_SEEDS = [2026110801, 2026110802, 2026110803]
SOLVER_SEEDS = [17, 29, 43]
PROTON_SHA = {
    "train.csv": "6a02432026e54d2ee1d78dcc2b51a1beef3ae201a819b44b94202b3d7057c5bf",
    "test.csv": "247d8ed8f867670e99c712021f6a78c8ceac71ea8787257d1515b28c9e08ca33",
    "metadata.yaml": "191c2da1d7f9f199b2095dbebe0ce0459e1fbe7d2a9c9f8f5c67a2f12690ac2a",
}
PROFILES = {
    "mean": {},
    "linear": {},
    "dictionary": {"max_degree": 3, "max_terms": 8},
    "parfam": {
        "family": "rational",
        "functions": [],
        "max_active_functions": 0,
        "degree": 2,
        "input_degree": 2,
        "denominator_degree": 1,
        "maxiter": 40,
        "repetitions": 1,
        "seconds": 15,
        "candidate_scope": "completed_families",
    },
}


def source_digest():
    import ml_platform_symbolic

    root = Path(ml_platform_symbolic.__file__).parent
    paths = sorted(root.rglob("*.py"))
    records = [(str(p.relative_to(root)), digest(p)) for p in paths]
    records.append(("benchmark_identification.py", digest(__file__)))
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def runtime_identity():
    return {
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "ml-platform-core", "parfam"]},
        "image": os.environ.get("SYMBOLIC_CONTAINER_IMAGE_ID"),
    }


def verify_protocol(root):
    plan = read_json(root / "plan.json")
    if plan["source_sha256_at_prepare"] != source_digest() or plan["runtime"] != runtime_identity():
        raise ValueError("Source/runtime changed after protocol freeze; create a new campaign before discovery")
    return plan


def synthetic(seed, confounded):
    """Known-family control; only observations and IDs enter discovery."""
    rng = np.random.default_rng(seed)
    frames = []
    for partition, stratum, n, limits in [
        ("train", "same_design", 60, (0.1, 2.0)),
        ("selection", "same_design", 30, (0.1, 2.0)),
        ("final", "same_design", 60, (0.1, 2.0)),
        ("final", "decoupled", 60, (0.1, 2.0)),
        ("final", "extrapolation", 60, (2.2, 4.0)),
    ]:
        x = rng.uniform(*limits, n)
        independent_z = rng.uniform(0.1, 2.0, n)
        z = x if confounded and stratum == "same_design" else independent_z
        signal = 1.2 + 2.5 * x / (0.4 + x) + 0.15 * z
        # Generate the same error draws in both support arms.
        noise = np.repeat(rng.normal(0, 0.01, n), 2) + rng.normal(0, 0.01, n * 2)
        prefix = f"{partition}-{stratum}"
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": [f"{prefix}-{i}" for i in range(n * 2)],
                    "group": np.repeat([f"{prefix}-g{i}" for i in range(n)], 2),
                    "x": np.repeat(x, 2),
                    "z": np.repeat(z, 2),
                    "response": np.repeat(signal, 2) + noise,
                    "partition": partition,
                    "stratum": stratum,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame.partition != "final"].copy(), frame[frame.partition == "final"].copy()


def prepare_campaign(root, proton):
    root.mkdir(parents=True, exist_ok=False)
    cases = []

    def add_case(name, frame, features, *, group=None, final=None, data_seed=None, scope):
        directory = root / "solver" / name
        directory.mkdir(parents=True)
        frame.drop(columns=["partition"]).to_csv(directory / "observations.csv", index=False)
        frame[["sample_id", "partition"]].to_csv(directory / "split.csv", index=False)
        cfg = {
            "data": {
                "local_path": str(directory / "observations.csv"),
                "feature_columns": features,
                "target_column": "response",
                # Proton input is Q2 in GeV^2; no supported unit parser label is
                # invented. Unknown unit semantics cannot become unit-valid.
                "units": dict.fromkeys([*features, "response"], "1" if group else None),
            },
            "split": {"file": str(directory / "split.csv"), "group_column": group},
            "search": {
                "engines": list(PROFILES),
                "seeds": [0, *SOLVER_SEEDS],
                "operators": ["add", "sub", "mul", "div", "pow_int", "neg"],
                "max_wall_seconds": 60,
                "candidate_pool_policy": "all",
                "max_export_candidates": 100,
                "allow_partial_engines": True,
                "engine_options": PROFILES,
            },
            "requirements": {"max_nodes": 100, "max_free_parameters": 30, "require_units": False},
            # External scorer files are deliberately absent from prepared refs.
            "evaluation": {"mode": "exploratory", "coefficient_fit": "none"},
        }
        refs = prepare(resolve_config(cfg), directory / "prepared")
        cases.append(
            {
                "name": name,
                "prepared": refs,
                "features": features,
                "group": group,
                "data_seed": data_seed,
                "scope": scope,
                "final": final,
                "n_train": int((frame.partition == "train").sum()),
                "n_selection": int((frame.partition == "selection").sum()),
            }
        )

    scorer = root / "scorer"
    scorer.mkdir()
    for seed in DATA_SEEDS:
        for arm in ("independent", "confounded"):
            name = f"saturation_{arm}_{seed}"
            train, final = synthetic(seed, arm == "confounded")
            final_path = scorer / (name + ".csv")
            final.to_csv(final_path, index=False)
            add_case(
                name,
                train,
                ["x", "z"],
                group="group",
                data_seed=seed,
                final={"path": str(final_path), "sha256": digest(final_path)},
                scope=f"Known-family synthetic {arm}-input control; fresh draw, not held-out structure",
            )
    if proton is not None:
        for name in ("train.csv", "metadata.yaml"):
            if digest(proton / name) != PROTON_SHA[name]:
                raise ValueError(f"Pinned proton {name} changed")
        # Test targets are not parsed or hashed by the solver preparation stage.
        train = read_observations(proton / "train.csv", identifiers=())
        if set(train.columns) != {"Q2_GeV2", "GE_over_GD"}:
            raise ValueError("Unexpected proton columns")
        train = train.rename(columns={"Q2_GeV2": "x", "GE_over_GD": "response"})
        train = train.sort_values("x", kind="stable").reset_index(drop=True)
        train["sample_id"] = [f"proton-{i}" for i in range(len(train))]
        train["partition"] = np.where(np.arange(len(train)) % 4 == 2, "selection", "train")
        add_case(
            "proton_form_factor",
            train,
            ["x"],
            final={"path": str(proton / "test.csv"), "sha256": PROTON_SHA["test.csv"]},
            scope="Public SciLaws-Real range transport: Q2 GeV^2 -> GE/GD; no structural truth; n_test=10",
        )
    plan = {
        "cases": cases,
        "profiles": PROFILES,
        "native_seeds": SOLVER_SEEDS,
        "selection_rule": "Eligible minimum Selection RMSE, nodes, parameters, ID; no refit",
        "planned_runs": len(cases) * (3 + len(SOLVER_SEEDS)),
        "source_sha256_at_prepare": source_digest(),
        "runtime": runtime_identity(),
        "evidence_scope": "No deployment criterion or exact recovery claim. All choices precede Test scoring.",
    }
    write_json(root / "plan.json", plan)
    return plan


def discover_campaign(root):
    plan = verify_protocol(root)
    for case in plan["cases"]:
        refs = case["prepared"]
        provenance = read_json(refs["run"])
        for engine in plan["profiles"]:
            for seed in SOLVER_SEEDS if engine == "parfam" else [0]:
                verify_protocol(root)
                directory = root / "discover" / case["name"] / engine / str(seed)
                if directory.exists():
                    raise ValueError(f"Preserve existing attempt; do not silently retry {directory}")
                result = run_discovery(refs, provenance, engine, seed, directory)
                summary = read_json(result["summary"])
                print(case["name"], engine, seed, summary["status"], flush=True)


def select_campaign(root):
    if (root / "choices.json").exists() or (root / "scores.json").exists():
        raise ValueError("Choices are immutable; start a new campaign, never reselect after scoring")
    plan = verify_protocol(root)
    choices = []
    for case in plan["cases"]:
        for engine in plan["profiles"]:
            for seed in SOLVER_SEEDS if engine == "parfam" else [0]:
                directory = root / "discover" / case["name"] / engine / str(seed)
                summary_path = directory / "engine_summary.json"
                summary = read_json(summary_path)
                row = {
                    "case": case["name"],
                    "engine": engine,
                    "solver_seed": seed,
                    "data_seed": case["data_seed"],
                    "native_status": summary["status"],
                    "native_execution_status": summary.get("native_execution_status"),
                    "wall_seconds": summary.get("wall_seconds"),
                    "candidate_count": 0,
                    "eligible_count": 0,
                    "selected_id": None,
                    "reason": summary.get("reason"),
                    "summary_sha256": digest(summary_path),
                }
                study = create_study(
                    {
                        "prepared_run": case["prepared"]["run"],
                        "sources": [{"discovery": str(summary_path)}],
                        "purpose": "local_prediction",
                        "question": case["scope"],
                        "engineering_tolerance": None,
                        "final_history": "unknown",
                    },
                    run_dir=directory / "study",
                )
                context, records, rows, _ = load_study(study)
                usable = []
                for record, evidence in zip(records, rows, strict=True):
                    metrics = evidence.get("metrics", {}).get("selection", {})
                    usable.append(
                        {
                            "candidate_id": record["id"],
                            "eligible": evidence["eligible"],
                            "rmse": metrics.get("rmse"),
                            "n_nodes": evidence.get("n_nodes"),
                            "n_parameters": evidence.get("n_parameters"),
                        }
                    )
                row.update(
                    candidate_count=sum(r["kind"] != "native_result" for r in records),
                    eligible_count=sum(r["eligible"] for r in usable),
                    study=str(study),
                    study_sha256=digest(study / "study.json"),
                )
                if row["eligible_count"]:
                    selected, _ = select_row(usable, {"selection_policy": "accuracy"})
                    record = next(r for r in records if r["id"] == selected["candidate_id"])
                    row.update(
                        selected_id=record["id"],
                        selection_rmse=selected["rmse"],
                        n_nodes=selected["n_nodes"],
                        n_parameters=selected["n_parameters"],
                        equation=equation_text(record["equation"], context["variables"]),
                    )
                else:
                    row["reason"] = row["reason"] or "No eligible candidate; retained in denominator"
                choices.append(row)
    frozen = {"plan_sha256": digest(root / "plan.json"), "source_sha256": source_digest(), "choices": choices}
    write_json(root / "choices.json", frozen)
    print(f"Frozen {len(choices)} choices before Test scoring", flush=True)


def score_campaign(root):
    if (root / "scores.json").exists():
        raise ValueError("Test already scored; preserve it and do not tune or reselect")
    frozen = read_json(root / "choices.json")
    if frozen["plan_sha256"] != digest(root / "plan.json") or frozen["source_sha256"] != source_digest():
        raise ValueError("Plan or implementation changed after choices; no Test access")
    plan = verify_protocol(root)
    # Written before opening any scoring targets, including failed scoring attempts.
    marker = root / "test_access.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"choices_sha256": digest(root / "choices.json"), "tuning_forbidden": True}, stream)
    frames = {}
    for case in plan["cases"]:
        source = case["final"]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Scoring input changed")
        frame = read_observations(source["path"], identifiers=())
        if case["name"] == "proton_form_factor":
            frame = frame.rename(columns={"Q2_GeV2": "x", "GE_over_GD": "response"})
            frame["stratum"] = "published_range_transport"
        frames[case["name"]] = frame
    scores = []
    for choice in frozen["choices"]:
        result = dict(choice)
        if choice["selected_id"] is None:
            result["score_status"] = "no_eligible_candidate"
            scores.append(result)
            continue
        if digest(Path(choice["study"]) / "study.json") != choice["study_sha256"]:
            raise ValueError("Frozen study changed")
        context, records, _, _ = load_study(choice["study"])
        record = next(r for r in records if r["id"] == choice["selected_id"])
        frame = frames[choice["case"]]
        columns = [v["column"] for v in context["variables"]]
        X, y = frame[columns].to_numpy(), frame.response.to_numpy()
        prediction, valid = predict_record(record, X, context["variables"], context["target"])
        result.update(
            prediction_rows={
                "partition": "Frozen-choice held-out Test / Final",
                "observed": y.tolist(),
                "predicted": [float(v) if np.isfinite(v) else None for v in prediction],
                "residual": [float(v) if np.isfinite(v) else None for v in y - prediction],
                "valid": valid.tolist(), "stratum": frame.stratum.tolist(),
            },
            score_status="scored",
            test_metrics=prediction_metrics(y, prediction, valid=valid),
            coverage=float(valid.mean()),
            strata={},
        )
        for label in frame.stratum.unique():
            mask = frame.stratum.to_numpy() == label
            result["strata"][label] = {
                "n": int(mask.sum()),
                "coverage": float(valid[mask].mean()),
                **prediction_metrics(y[mask], prediction[mask], valid=valid[mask]),
            }
        if choice["data_seed"] is not None:
            # Scorer-only known derivative at independently varied coordinates.
            grid = np.linspace(0.12, 1.98, 101)
            points = np.column_stack([grid, np.full(len(grid), 0.7)])
            high, low = points.copy(), points.copy()
            high[:, 0] += 1e-4
            low[:, 0] -= 1e-4
            a, av = predict_record(record, high, context["variables"], context["target"])
            b, bv = predict_record(record, low, context["variables"], context["target"])
            derivative = (a - b) / 2e-4
            result["x_response_rmse"] = prediction_metrics(1.0 / (0.4 + grid) ** 2, derivative, valid=av & bv)["rmse"]
            result["response_scope"] = "101 declared interior finite-difference points at z=0.7; not global equivalence"
        scores.append(result)
    write_json(
        root / "scores.json",
        {
            "choices_sha256": digest(root / "choices.json"),
            "case_context": {case["name"]: {
                "question": case["scope"],
                "truth_status": "Known synthetic generator, not a hidden structural holdout" if case["data_seed"] is not None else "Observed public data; no known true formula",
                "reference_formula": "response_clean = 1.2 + 2.5*x/(0.4+x) + 0.15*z" if case["data_seed"] is not None else None,
                "target_unit": "1" if case["data_seed"] is not None else "GE/GD (dimensionless ratio)",
                "variables": "x,z: dimensionless" if case["data_seed"] is not None else "x = Q2 [GeV^2]; solver unit semantics unknown",
                "scope": "All choices frozen before Test. Reference is scorer/readout-only; not supplied to discovery. No deployment tolerance.",
            } for case in plan["cases"]},
            "attempts": len(scores),
            "results": scores,
            "interpretation": "Failure-inclusive paired campaign. Separate data draws, solver seeds, prediction and response errors. No general ranking, causal law, or deployment approval.",
        },
    )
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in scores]).to_csv(
        root / "attempts.csv", index=False
    )
    if __package__:
        from .benchmark_readout import write_report
    else:
        from benchmark_readout import write_report

    print("Open report:", write_report(root / "scores.json", root / "readout"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "discover", "select", "score"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--proton", type=Path, help="Pinned solver metadata and train/test CSV files, no eval/simulator"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "prepare":
        prepare_campaign(root, args.proton.resolve() if args.proton else None)
    else:
        {"discover": discover_campaign, "select": select_campaign, "score": score_campaign}[args.phase](root)


if __name__ == "__main__":
    main()
