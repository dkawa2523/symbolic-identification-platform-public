"""Freeze the explicit three-arm campaign using existing data/config APIs; do not run search or open Final."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_core.config import deep_merge
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, read_json, write_json

if __package__:
    from .benchmark_symbolic import MAINTAINED_CASES, physics_options
else:
    from benchmark_symbolic import MAINTAINED_CASES, physics_options

METHODS = ("pysr", "physo", "torchsisso", "flash_ansr", "proged", "parfam", "rasr", "pysips", "pse", "e2e")
OPERATORS = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "sqrt",
    "pow_int",
    "pow",
    "rootn",
    "sin",
    "cos",
    "exp",
    "log",
    "tanh",
]
SIGNS = {
    "polynomial": [("x", 1, 1), ("z", 1, -1)],
    "noisy_extrapolation": [("x", 1, 1), ("z", 1, -1)],
    "rational_constraints": [("x", 1, -1)],
    "rational_unknown": [("x", 1, 1)],
    "nonlinear_noisy_ood": [("x", 1, -1)],
    "ambiguous_quadratic": [("x", 2, 1)],
    "correlated_interaction": [("x", 1, 1), ("z", 1, -1)],
    "implicit_positive": [("x", 1, 1)],
    "dimensional_ballistic": [("time", 1, 1), ("time", 2, 1)],
    "sparse_nuisance": [("drive", 1, 1), ("gain", 1, 1), ("offset_input", 1, -1)],
    "dimensional_sparse": [("speed", 1, 1), ("duration", 1, 1), ("offset", 1, -1)],
    "damped_resonance": [("gamma", 1, -1)],
    "activated_kinetics": [("temperature", 1, 1)],
}


def response_arm(case, config, train):
    """Explicit finite operating conditions; no generator coefficients or Final targets."""
    columns = config["data"]["feature_columns"]
    target = config["data"]["target_column"]
    points = train[columns].iloc[np.linspace(0, len(train) - 1, min(24, len(train)), dtype=int)].copy()
    result = {"response_checks": []}
    if case in {"yacht", "airfoil", "concrete"}:
        q01, q25, q75, q99 = train[target].quantile([0.01, 0.25, 0.75, 0.99]).to_numpy()
        if q75 <= q25:
            raise ValueError(f"Train IQR is degenerate: {case}; no bound invented")
        result.update(probes=points.to_dict("records"), response_bounds=[q01 - 2 * (q75 - q25), q99 + 2 * (q75 - q25)])
        return result
    for feature, order, sign in SIGNS.get(case, []):
        selected = points.copy()
        if case in {"polynomial", "noisy_extrapolation"}:
            selected = selected.loc[selected.x > 0.01]
        if case == "nonlinear_noisy_ood":
            selected = selected.loc[selected.z <= 0]
        if case == "correlated_interaction":
            selected = selected.loc[(selected.abs() <= 1).all(axis=1)]
        bounds = config["data"]["domain"].get(feature, [float(train[feature].min()), float(train[feature].max())])
        span = bounds[1] - bounds[0]
        step = span * 0.001
        selected[feature] = selected[feature].clip(bounds[0] + step, bounds[1] - step)
        if not len(selected) or step <= 0:
            raise ValueError(f"No valid response conditions for {case}/{feature}")
        # Tolerance has target/input^order units, scaled only by Train observations.
        tolerance = max(float(train[target].std(ddof=0)), 1e-12) / span**order * 1e-6
        result["response_checks"].append(
            {
                "feature": feature,
                "order": order,
                "sign": sign,
                "step": step,
                "tolerance": tolerance,
                "points": selected.to_dict("records"),
                "knowledge": "informed finite-region response assumption, predeclared before this campaign",
            }
        )
    if case in {"sine_unknown", "rational_constraints", "damped_resonance", "activated_kinetics"}:
        result.update(
            probes=points.to_dict("records"), response_bounds=[-2, 2] if case == "sine_unknown" else [0, None]
        )
    return result


def method_profiles():
    return {
        "pysr": {
            "niterations": 100,
            "populations": 8,
            "population_size": 40,
            "ncycles_per_iteration": 200,
            "seconds": 120,
            "maxdepth": 15,
        },
        "physo": {"epochs": 20, "batch_size": 256, "max_n_evaluations": 5120},
        "torchsisso": {"n_expansion": 2, "n_term": 3, "k": 10},
        "flash_ansr": {"draws": 128},
        "proged": {
            "sample_size": 80,
            "depth_limit": 8,
            "max_attempts": 600,
            "max_constants": 6,
            "optimizer": "DE",
            "optimizer_iterations": 20,
            "optimizer_population": 16,
        },
        "parfam": {
            "family": "rational",
            "functions": ["sin", "cos"],
            "degree": 2,
            "input_degree": 2,
            "denominator_degree": 1,
            "maxiter": 50,
            "repetitions": 1,
            "max_active_parameters": 12,
            "seconds": 90,
        },
        "rasr": {"candidates": 32, "decode_batch_size": 8, "restarts": 2},
        "pysips": {
            "particles": 100,
            "mcmc_samples": 10,
            "max_complexity": 12,
            "seconds": 90,
            "opt_restarts": 1,
            "param_init_bounds": [-5, 5],
        },
        "pse": {"layers": 2, "slots": 3, "use_const": True, "top_k": 10, "seconds": 90},
        "e2e": {"beam_size": 8, "n_trees_to_refine": 8, "refinement_seconds": 2},
    }


def freeze(inputs, root):
    root.mkdir(parents=True, exist_ok=False)
    manifest = read_json(inputs / "inputs.json")
    profiles = method_profiles()
    cases, cells = {}, []
    for case in MAINTAINED_CASES:
        source = inputs / case
        for name, sha in manifest["cases"][case]["files"].items():
            if digest(source / name) != sha:
                raise ValueError(f"Frozen input changed: {case}/{name}")
        problem = read_json(source / "problem.json")
        columns = problem["data"]["feature_columns"]
        split = pd.read_csv(problem["split"]["file"])
        train_ids = set(split.loc[split.partition == "train", "sample_id"])
        # Only Train rows contribute to arm design; no Final model values are computed.
        chunks = pd.read_csv(problem["data"]["local_path"], chunksize=512)
        train = pd.concat([chunk.loc[chunk.sample_id.isin(train_ids)] for chunk in chunks], ignore_index=True)
        cap = 2 if len(train) < 40 else 6 if len(columns) <= 2 else 8
        arms = {
            "A": {"require_units": True},
            "B": response_arm(case, problem, train),
            "C": {"max_free_parameters": cap},
        }
        case_dir = root / case
        for arm, constraints in arms.items():
            write_json(case_dir / f"{arm}.json", constraints)
        base = deepcopy(problem)
        base["data"]["fixed_constants"] = {}
        config = resolve_config(
            deep_merge(
                base,
                {
                    "search": {
                        "engines": list(METHODS),
                        "seeds": [0, 1, 2],
                        "operators": OPERATORS,
                        "max_wall_seconds": 240,
                        "max_export_candidates": 100,
                        "allow_partial_engines": True,
                        "engine_options": profiles,
                        "runtimes": {"pse": {"device": "cuda"}},
                    },
                    "model_assets": {
                        "flash_ansr": {"manifest_file": "pretrained/flash/manifest.json"},
                        "rasr": {"manifest_file": "pretrained/rasr/manifest.json"},
                        "e2e": {"manifest_file": "pretrained/e2e/manifest.json"},
                    },
                    "constraints": {"max_nodes": 60, "require_units": False},
                    "evaluation": {"coefficient_fit": "none", "relative_tolerance": 0.01, "absolute_tolerance": 1e-6},
                },
            )
        )
        if case in {"damped_resonance", "activated_kinetics"}:
            config = deep_merge(config, physics_options(case, ["physo"]))
        write_json(case_dir / "generation.json", config)
        # Units-aware sparse search is a real distinct native job, not postcheck reuse.
        dimensional = deepcopy(config)
        dimensional["search"]["engine_options"]["torchsisso"]["dimensional"] = True
        if case == "dimensional_ballistic":
            dimensional["data"]["fixed_constants"] = problem["data"]["fixed_constants"]
        write_json(case_dir / "dimensional_generation.json", dimensional)
        effective = {}
        for arm, constraints in arms.items():
            resolved = deep_merge(dimensional if arm == "A" else config, {"constraints": constraints})
            resolved = resolve_config(resolved)
            path = write_json(case_dir / f"{arm}_effective.json", resolved)
            effective[arm] = {"path": str(path), "sha256": digest(path)}
        cases[case] = {
            "train_rows": len(train),
            "input_hashes": manifest["cases"][case]["files"],
            "arms": {
                arm: {"path": str(case_dir / f"{arm}.json"), "sha256": digest(case_dir / f"{arm}.json")} for arm in arms
            },
            "effective_configs": effective,
            "arm_questions": {
                "A": "established dimensional admissibility",
                "B": "both-sign implicit relation coverage"
                if case == "implicit_ambiguous"
                else "declared sampled response behavior",
                "C": "independent declared free-parameter cap",
            },
            "information_note": "g supplied only in ballistic A; other inputs and partitions are shared",
            "constraint_mode": "postcheck on frozen pool except separately recorded native dimensional/constant jobs",
        }
        for arm in arms:
            for method in METHODS:
                status, reason = "not_run", None
                if case == "airfoil" and (method == "physo" or arm == "A"):
                    status, reason = (
                        "unsupported",
                        "dB is not a multiplicative target dimension supported by this unit path",
                    )
                elif case == "dimensional_ballistic" and arm == "A" and method != "physo":
                    status, reason = (
                        "unsupported",
                        "Required immutable named g injection is unsupported by this adapter",
                    )
                elif case == "implicit_ambiguous" and arm == "B":
                    status, reason = (
                        "not_evaluable",
                        "Scalar prediction cannot answer both-sign relation/branch coverage",
                    )
                cells.extend(
                    {"case": case, "arm": arm, "method": method, "seed": seed, "status": status, "reason": reason}
                    for seed in (0, 1, 2)
                )
    write_json(root / "method_profiles.json", profiles)
    write_json(
        root / "matrix.json",
        {
            "cases": cases,
            "seeds": [0, 1, 2],
            "cells": cells,
            "final_policy": "freeze all choices per stage before Final audit; historical real-data exposure retained",
            "generator_sha256": digest(Path(__file__)),
            "status": "frozen_plan_not_executed",
        },
    )
    pd.DataFrame(cells).to_csv(root / "cells.csv", index=False)
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cells = freeze(args.inputs.resolve(), args.root.resolve())
    print(pd.Series([cell["status"] for cell in cells]).value_counts().to_dict())
