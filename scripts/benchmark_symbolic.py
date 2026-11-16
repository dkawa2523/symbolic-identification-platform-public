"""Fixed benchmark input preparation and collection of saved historical predictions.

No discovery, automatic selection or Final prediction writer. Use explicit common stages.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_core.config import deep_merge
from ml_platform_symbolic.config import load_config, resolve_config
from ml_platform_symbolic.data import digest, read_json, write_json
from ml_platform_symbolic.expressions import parse_expression
from ml_platform_symbolic.metrics import prediction_metrics

MAINTAINED_CASES = (
    "polynomial",
    "noisy_extrapolation",
    "rational_constraints",
    "rational_unknown",
    "sine_unknown",
    "nonlinear_noisy_ood",
    "ambiguous_quadratic",
    "correlated_interaction",
    "implicit_positive",
    "implicit_ambiguous",
    "dimensional_ballistic",
    "sparse_nuisance",
    "dimensional_sparse",
    "yacht",
    "airfoil",
    "concrete",
    "damped_resonance",
    "activated_kinetics",
)


def synthetic(root, case, seed, data_seed=42):
    rng = np.random.default_rng(data_seed)
    if case in {"damped_resonance", "activated_kinetics"}:
        return damped_resonance(root, data_seed) if case == "damped_resonance" else activated_kinetics(root, data_seed)
    if case in {"rational_unknown", "sine_unknown"}:
        # Method seeds change search only; the task data/split remain identical.
        return coefficient_family(root, case, np.random.default_rng(data_seed), data_seed)
    if case in {"ambiguous_quadratic", "correlated_interaction"}:
        return ambiguous_family(root, case, np.random.default_rng(data_seed), data_seed)
    if case in {"implicit_positive", "implicit_ambiguous"}:
        return implicit_family(root, case, np.random.default_rng(data_seed), data_seed)
    if case == "dimensional_ballistic":
        return dimensional_ballistic(root, np.random.default_rng(data_seed), data_seed)
    if case == "sparse_nuisance":
        return sparse_nuisance(root, np.random.default_rng(data_seed), data_seed)
    if case == "dimensional_sparse":
        return sparse_nuisance(root, np.random.default_rng(data_seed), data_seed, dimensional=True)
    if case == "nonlinear_noisy_ood":
        return nonlinear_noisy_ood(root, np.random.default_rng(data_seed), data_seed)
    if case not in {"polynomial", "noisy_extrapolation", "rational_constraints"}:
        raise ValueError(f"Unknown synthetic benchmark: {case}")
    X = rng.uniform(0.5, 2.0, (300, 2))
    labels = np.array(["train"] * 192 + ["selection"] * 48 + ["final"] * 60)
    if case == "noisy_extrapolation":
        X[labels == "final", 0] = rng.uniform(2.2, 3.2, 60)
    if case == "rational_constraints":
        y = 3 / (1 + X[:, 0])
        columns = ["x"]
    else:
        y = 2 * X[:, 0] ** 2 - 0.5 * X[:, 1] + 1
        columns = ["x", "z"]
    if case == "noisy_extrapolation":
        y += rng.normal(0, 0.15, len(y))
    frame = pd.DataFrame(X[:, : len(columns)], columns=columns)
    frame["y"] = y
    frame["sample_id"] = [f"sample_{i:04}" for i in range(len(frame))]
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=labels).to_csv(split, index=False)
    return {
        "data": {
            "local_path": str(table),
            "target_column": "y",
            "feature_columns": columns,
            "units": {c: "1" for c in [*columns, "y"]},
            "domain": {"x": [0.0, 4.0]} if case == "rational_constraints" else {},
        },
        "split": {"file": str(split)},
    }


def _physics_table(root, frame, partitions, regions, clean, metadata, *, group, units, domains):
    """Two source-grounded cases use the same existing CSV/split/scorer boundary."""
    frame = frame.copy()
    frame["sample_id"] = [f"{metadata['case']}_{i:05d}" for i in range(len(frame))]
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=partitions).to_csv(split, index=False)
    pd.DataFrame({"sample_id": frame.sample_id, "region": regions, "clean": clean}).to_csv(
        root / "scorer_only.csv", index=False
    )
    features = [c for c in frame if c not in {"sample_id", group, metadata["target"]}]
    write_json(
        root / "metadata.json",
        {
            **metadata,
            "source": "independently generated synthetic measurements; no experimental equipment data",
            "generator": "NumPy PCG64 with separate design/split/noise SeedSequence children",
            "generator_sha256": digest(__file__),
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "counts": {p: int(np.sum(np.asarray(partitions) == p)) for p in ("train", "selection", "final")},
            "truth_use": "law/clean labels/regions/physical constants are scorer/reference only, not native search hints",
            "pretraining_overlap": "not established; new points do not establish structural novelty",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": metadata["target"],
            "feature_columns": features,
            "units": units,
            "domain": domains,
        },
        "split": {"file": str(split), "group_column": group},
    }


def damped_resonance(root, seed):
    design, split_rng, noise = [
        np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed, spawn_key=(23, 11)).spawn(3)
    ]
    gammas = np.exp(design.uniform(np.log(0.6), np.log(2), 24))
    order = split_rng.permutation(24)
    assignment = {int(g): "train" if i < 16 else "selection" if i < 20 else "final" for i, g in enumerate(order)}
    rows, labels, regions = [], [], []
    sweeps = [(f"normal_{i}", g, assignment[i], "id", False) for i, g in enumerate(gammas)]
    sweeps += [
        (f"low_{i}", g, "final", "low_damping", False)
        for i, g in enumerate(np.exp(design.uniform(np.log(0.15), np.log(0.35), 4)))
    ]
    sweeps += [
        (f"tail_{i}", g, "final", "high_frequency", True)
        for i, g in enumerate(np.exp(design.uniform(np.log(0.6), np.log(2), 4)))
    ]
    for name, gamma, partition, region, tail in sweeps:
        omega = design.uniform(24, 40, 32) if tail else np.r_[design.uniform(3, 18, 24), design.uniform(9, 11, 8)]
        rows.extend((name, float(w), float(gamma)) for w in omega)
        labels.extend([partition] * len(omega))
        regions.extend([region] * len(omega))
    frame = pd.DataFrame(rows, columns=["sweep_id", "omega", "gamma"])
    clean = 0.1 / np.sqrt(
        (100 - frame.omega.to_numpy() ** 2) ** 2 + frame.gamma.to_numpy() ** 2 * frame.omega.to_numpy() ** 2
    )
    frame["amplitude"] = clean + noise.normal(0, 2e-5, len(frame))
    return _physics_table(
        root,
        frame,
        labels,
        regions,
        clean,
        {
            "case": "damped_resonance",
            "target": "amplitude",
            "seed": seed,
            "spawn_key": [23, 11],
            "law": "F0/(m*sqrt((omega0^2-omega^2)^2+gamma^2*omega^2))",
            "constants_scorer_only": {"m_kg": 1, "F0_N": 0.1, "omega0_rad_per_s": 10},
            "noise": "additive Gaussian sigma=2e-5 m; raw readings are not clipped",
            "split": "whole sweep; 16 train,4 selection,4 final-ID,4 final-low-damping,4 final-high-frequency",
            "interpretation": "angular frequencies in rad/s, not cyclic Hz; half-power width uses amplitude squared",
            "sources": ["https://www.feynmanlectures.caltech.edu/I_23.html#Eq:I:23:11"],
        },
        group="sweep_id",
        units={"omega": "rad/s", "gamma": "1/s", "amplitude": "m"},
        domains={"omega": [3, 40], "gamma": [0.15, 2.0]},
    )


def activated_kinetics(root, seed):
    design, split_rng, noise = [
        np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed, spawn_key=(4, 46)).spawn(3)
    ]
    R, beta_c, tref = 8.31446261815324, 2000.0, 330.0
    conditions = []
    temperatures = np.linspace(300, 360, 24)
    # Randomize condition order with its own stream, never split its replicates.
    for T in temperatures[split_rng.permutation(len(temperatures))]:
        conditions.append((float(T), float(np.exp(beta_c * (1 / tref - 1 / T))), "train", "observational"))
    for Ts, partition in (([308, 326, 344, 354], "selection"), ([304, 317, 337, 357], "final")):
        conditions.extend((T, float(np.exp(beta_c * (1 / tref - 1 / T))), partition, "observational") for T in Ts)
        levels = [0.6, 1.0, 1.6] if partition == "selection" else [0.65, 0.9, 1.2, 1.5]
        conditions.extend((T, c, partition, "intervention") for T in Ts for c in levels)
    conditions.extend((T, c, "final", "temperature_ood") for T in [280, 290, 375, 390] for c in [0.65, 0.9, 1.2, 1.5])
    rows, labels, regions = [], [], []
    for i in design.permutation(len(conditions)):
        T, concentration, partition, region = conditions[i]
        rows.extend((f"condition_{i}", T, concentration) for _ in range(6))
        labels.extend([partition] * 6)
        regions.extend([region] * 6)
    frame = pd.DataFrame(rows, columns=["condition_id", "temperature", "concentration_ratio"])
    clean = (
        0.02
        * frame.concentration_ratio.to_numpy()
        * np.exp((48000 / R) * (1 / tref - 1 / frame.temperature.to_numpy()))
    )
    frame["rate"] = clean * np.exp(0.03 * noise.normal(size=len(frame)) - 0.03**2 / 2)
    return _physics_table(
        root,
        frame,
        labels,
        regions,
        clean,
        {
            "case": "activated_kinetics",
            "target": "rate",
            "seed": seed,
            "spawn_key": [4, 46],
            "law": "rate=k_ref*(c/c_ref)*exp((Ea/R)*(1/Tref-1/T))",
            "constants_scorer_only": {
                "k_ref_per_s": 0.02,
                "Ea_J_per_mol": 48000.0,
                "R_J_per_mol_K": R,
                "Tref_K": tref,
                "beta_c_K": beta_c,
            },
            "input_meaning": "concentration_ratio=c/c_ref, c_ref=1 mol/m^3; rate=v/c_ref in 1/s; no implicit unit conversion",
            "noise": "mean-preserving multiplicative lognormal sigma_log=.03; Gaussian likelihood is not matched",
            "split": "whole repeated condition,6replicates; Selection and Final interventions are disjoint",
            "interpretation": "n=0,1,2 reference laws are observationally identical; only independent interventions discriminate them",
            "sources": [
                "https://goldbook.iupac.org/terms/view/A00446/plain",
                "https://goldbook.iupac.org/terms/view/O04322",
            ],
        },
        group="condition_id",
        units={"temperature": "K", "concentration_ratio": "1", "rate": "1/s"},
        domains={"temperature": [280, 390], "concentration_ratio": [0.5, 1.8], "rate": [0, None]},
    )


def physics_options(case, engines):
    if case == "damped_resonance":
        variables = [{"id": "x0", "column": "omega", "unit": "rad/s"}, {"id": "x1", "column": "gamma", "unit": "1/s"}]
        parameters = {
            "scale": {"value": 0.05, "trainable": True, "bounds": [0.001, 1.0], "unit": "m/s2"},
            "natural": {"value": 9.0, "trainable": True, "bounds": [6.0, 14.0], "unit": "rad/s"},
        }
        expression = "scale/sqrt((natural*natural-x0*x0)**2+x1*x1*x0*x0)"
        palette = {"c0": "1", "c_len": "m", "c_freq": "rad/s", "c_acc": "m/s2"}
    else:
        variables = [
            {"id": "x0", "column": "temperature", "unit": "K"},
            {"id": "x1", "column": "concentration_ratio", "unit": "1"},
        ]
        parameters = {
            "rate_ref": {"value": 0.01, "trainable": True, "bounds": [1e-5, 1.0], "unit": "1/s"},
            "barrier": {"value": 4000.0, "trainable": True, "bounds": [100.0, 20000.0], "unit": "K"},
            "t_ref": {
                "value": 330.0,
                "trainable": False,
                "bounds": [330.0, 330.0],
                "unit": "K",
                "origin": "chosen_reference_temperature",
            },
        }
        expression = "rate_ref*x1*exp(barrier*(1/t_ref-1/x0))"
        palette = {"c0": "1", "c_rate": "1/s", "c_temp": "K"}
    options = {}
    if "known_form" in engines:
        options["known_form"] = {
            "equation": parse_expression(expression, variables, trainable=False, parameters=parameters)
        }
    if "physo" in engines:
        options["physo"] = {"parameter_units": palette}
    return {"search": {"engine_options": options}}


def nonlinear_noisy_ood(root, rng, seed):
    """Fresh points/coefficients do not prove absence from pretraining corpora."""
    count = 360
    x, z = rng.uniform(0.1, 2.0, count), rng.uniform(-1, 1, count)
    labels = np.array(["train"] * 192 + ["selection"] * 48 + ["final"] * 120)
    regions = np.array(["id"] * 300 + ["ood"] * 60)
    x[300:] = rng.uniform(2, 3, 60)
    clean = (0.37 + 1.43 * np.exp(-0.81 * x)) / (1 + 0.62 * z * z) + 0.19 * x * z
    ids = [f"sample_{i:04}" for i in range(count)]
    table, split = root / "table.csv", root / "split.csv"
    pd.DataFrame({"sample_id": ids, "x": x, "z": z, "y": clean + rng.normal(0, 0.05, count)}).to_csv(table, index=False)
    pd.DataFrame({"sample_id": ids, "partition": labels}).to_csv(split, index=False)
    # Scorer-only labels are outside the prepared numeric files supplied to discovery.
    pd.DataFrame({"sample_id": ids, "region": regions, "clean": clean}).to_csv(root / "scorer_only.csv", index=False)
    write_json(
        root / "dataset_metadata.json",
        {
            "source": "generated locally, no external dataset",
            "generator": "NumPy PCG64",
            "seed": seed,
            "generator_sha256": digest(__file__),
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "law": "(0.37+1.43*exp(-0.81*x))/(1+0.62*z*z)+0.19*x*z",
            "noise": "independent Gaussian sigma=0.05 in all partitions",
            "counts": {"train": 192, "selection": 48, "final_id": 60, "final_ood": 60},
            "scope": "law/clean labels/region are scoring metadata; never passed to adapters",
            "pretraining_overlap": "unknown; new coefficients/points do not imply structural novelty",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": "y",
            "feature_columns": ["x", "z"],
            "units": {key: "1" for key in ("x", "z", "y")},
            "domain": {"x": [0.1, 3.0], "z": [-1.0, 1.0], "y": [-1.0, 3.0]},
        },
        "split": {"file": str(split)},
    }


def coefficient_family(root, case, rng, seed):
    """Unknown real coefficients, ID/OOD points, and no law in native requests."""
    x = rng.uniform(0.2, 2.0, 300)
    x[240:] = rng.uniform(2.0, 3.0, 60)
    clean = (0.37 + 1.91 * x) / (1.7 + x) if case == "rational_unknown" else 0.27 + 1.61 * np.sin(1.73 * x - 0.41)
    frame = pd.DataFrame({"sample_id": [f"r{i}" for i in range(300)], "x": x, "y": clean + rng.normal(0, 0.01, len(x))})
    labels = ["train"] * 120 + ["selection"] * 40 + ["final"] * 140
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=labels).to_csv(split, index=False)
    pd.DataFrame({"sample_id": frame.sample_id, "region": ["id"] * 240 + ["ood"] * 60, "clean": clean}).to_csv(
        root / "scorer_only.csv", index=False
    )
    write_json(
        root / "metadata.json",
        {
            "source": "generated locally",
            "rng": "PCG64",
            "seed": seed,
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "law": "(.37+1.91*x)/(1.7+x)" if case == "rational_unknown" else ".27+1.61*sin(1.73*x-.41)",
            "noise_sigma": 0.01,
            "truth_use": "generator/scorer only; never an engine hint",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": "y",
            "feature_columns": ["x"],
            "units": {"x": "1", "y": "1"},
            "domain": {"x": [0.2, 3.0], "y": [-5.0, 5.0]},
        },
        "split": {"file": str(split)},
    }


def ambiguous_family(root, case, rng, seed):
    """Scorer-only laws, small noisy training sets, and a fixed distribution shift."""
    n_train, n_selection = (18, 12) if case == "ambiguous_quadratic" else (24, 16)
    count = n_train + n_selection + 120
    if case == "ambiguous_quadratic":
        x = rng.uniform(-0.4, 0.4, count)
        x[-60:] = rng.uniform(0.4, 1.2, 60)
        frame = pd.DataFrame({"x": x})
        clean = 0.37 + 1.43 * x + 0.62 * x * x
        law, shift = "0.37+1.43*x+0.62*x*x", "x[-.4,.4] to x[.4,1.2]"
    else:
        x = rng.uniform(-0.8, 0.8, count)
        z = x + rng.normal(0, 0.05, count)
        x[-60:], z[-60:] = rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 60)
        frame = pd.DataFrame({"x": x, "z": z})
        clean = 0.37 + 1.21 * x - 0.73 * z + 0.68 * x * z
        law, shift = "0.37+1.21*x-0.73*z+0.68*x*z", "correlated z=x+N(0,.05) to independent x,z[-1,1]"
    columns = list(frame.columns)
    frame["sample_id"] = [f"b{i}" for i in range(count)]
    frame["y"] = clean + rng.normal(0, 0.12, count)
    labels = ["train"] * n_train + ["selection"] * n_selection + ["final"] * 120
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=labels).to_csv(split, index=False)
    pd.DataFrame({"sample_id": frame.sample_id, "clean": clean, "region": ["id"] * (count - 60) + ["ood"] * 60}).to_csv(
        root / "scorer_only.csv", index=False
    )
    write_json(
        root / "dataset_metadata.json",
        {
            "source": "generated locally",
            "generator": "NumPy PCG64",
            "seed": seed,
            "generator_sha256": digest(__file__),
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "law": law,
            "noise": "Gaussian sigma=.12, all partitions",
            "shift": shift,
            "counts": {"train": n_train, "selection": n_selection, "final_id": 60, "final_ood": 60},
            "scope": "law/clean/region are scorer-only; never native hints. Feature domain [-1.5,1.5], target unbounded",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": "y",
            "feature_columns": columns,
            "units": {name: "1" for name in [*columns, "y"]},
            "domain": {name: [-1.5, 1.5] for name in columns},
        },
        "split": {"file": str(split)},
    }


def implicit_family(root, case, rng, seed):
    """The relation law is scorer-only; branch information is declared before search."""
    count = 300
    x = rng.uniform(0.2, 2.0, count)
    x[-60:] = rng.uniform(2.0, 3.0, 60)
    clean = np.sqrt(0.37 + 1.91 * x)
    if case == "implicit_ambiguous":
        clean *= rng.choice([-1, 1], count)
    frame = pd.DataFrame(
        {"sample_id": [f"i{i}" for i in range(count)], "x": x, "y": clean + rng.normal(0, 0.005, count)}
    )
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 120 + ["selection"] * 40 + ["final"] * 140).to_csv(
        split, index=False
    )
    pd.DataFrame({"sample_id": frame.sample_id, "clean": clean, "region": ["id"] * 240 + ["ood"] * 60}).to_csv(
        root / "scorer_only.csv", index=False
    )
    write_json(
        root / "dataset_metadata.json",
        {
            "source": "generated locally",
            "generator": "NumPy PCG64",
            "seed": seed,
            "generator_sha256": digest(__file__),
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "law": "y*y=0.37+1.91*x",
            "noise_sigma": 0.005,
            "target_branch": "nonnegative" if case == "implicit_positive" else "both signs, chosen independently of X",
            "counts": {"train": 120, "selection": 40, "final_id": 80, "final_ood": 60},
            "scope": "law/clean/branch labels are scorer-only; native receives X/y and declared domains, no true residual",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": "y",
            "feature_columns": ["x"],
            "units": {"x": "1", "y": "1"},
            "domain": {"x": [0.2, 3.0], "y": [0, 3.0] if case == "implicit_positive" else [-3.0, 3.0]},
        },
        "split": {"file": str(split)},
    }


def dimensional_ballistic(root, rng, seed):
    """A declared immutable acceleration, dimensional inputs and unknown coefficients."""
    t = rng.uniform(0.2, 2.0, 300)
    t[-60:] = rng.uniform(2.0, 3.0, 60)
    clean = 0.37 + 0.5 * 9.80665 * t * t
    frame = pd.DataFrame(
        {"sample_id": [f"p{i}" for i in range(300)], "time": t, "height": clean + rng.normal(0, 0.02, len(t))}
    )
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 120 + ["selection"] * 40 + ["final"] * 140).to_csv(
        split, index=False
    )
    pd.DataFrame({"sample_id": frame.sample_id, "clean": clean, "region": ["id"] * 240 + ["ood"] * 60}).to_csv(
        root / "scorer_only.csv", index=False
    )
    write_json(
        root / "dataset_metadata.json",
        {
            "source": "generated locally",
            "generator": "NumPy PCG64",
            "seed": seed,
            "generator_sha256": digest(__file__),
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "law": "height=.37+.5*g*time*time",
            "noise_sigma_m": 0.02,
            "counts": {"train": 120, "selection": 40, "final_id": 80, "final_ood": 60},
            "scope": "law/clean labels are scorer-only; all engines receive the same g value/unit, not the true expression",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": "height",
            "feature_columns": ["time"],
            "units": {"time": "s", "height": "m"},
            "domain": {"time": [0.2, 3.0], "height": [0, 50]},
            "fixed_constants": {"g": {"value": 9.80665, "unit": "m/s2"}},
        },
        "split": {"file": str(split)},
    }


def sparse_nuisance(root, rng, seed, *, dimensional=False):
    """Sparse interaction plus five nuisance inputs, same data across method seeds."""
    columns = (
        ["speed", "duration", "offset", "mass", "temperature", "nuisance_length"]
        if dimensional
        else ["drive", "gain", "offset_input", *[f"nuisance_{i}" for i in range(5)]]
    )
    noise = 0.02 if dimensional else 0.05
    units = (
        dict(zip(columns, ["m/s", "s", "m", "kg", "K", "m"], strict=True))
        if dimensional
        else {name: "1" for name in columns}
    )
    X = rng.uniform(0.2, 2.0, (300, len(columns)))
    X[-60:, 0] = rng.uniform(2.0, 3.0, 60)
    clean = 0.37 + 1.43 * X[:, 0] * X[:, 1] - 0.81 * X[:, 2]
    frame = pd.DataFrame(X, columns=columns)
    frame["sample_id"] = [f"s{i}" for i in range(len(X))]
    frame["response"] = clean + rng.normal(0, noise, len(X))
    table, split = root / "table.csv", root / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 120 + ["selection"] * 40 + ["final"] * 140).to_csv(
        split, index=False
    )
    pd.DataFrame({"sample_id": frame.sample_id, "clean": clean, "region": ["id"] * 240 + ["ood"] * 60}).to_csv(
        root / "scorer_only.csv", index=False
    )
    write_json(
        root / "dataset_metadata.json",
        {
            "source": "generated locally",
            "generator": "NumPy PCG64",
            "seed": seed,
            "generator_sha256": digest(__file__),
            "table_sha256": digest(table),
            "split_sha256": digest(split),
            "law": ".37[m]+1.43*speed[m/s]*duration[s]-.81*offset[m]"
            if dimensional
            else ".37+1.43*drive*gain-.81*offset_input",
            "noise_sigma": noise,
            "counts": {"train": 120, "selection": 40, "final_id": 80, "final_ood": 60},
            "scope": f"all{len(columns)}columns provided; law/relevant-feature identity/clean labels are scorer-only",
        },
    )
    return {
        "data": {
            "local_path": str(table),
            "target_column": "response",
            "feature_columns": columns,
            "units": {**units, "response": "m" if dimensional else "1"},
            "domain": {**{name: [0.2, 2.0] for name in columns}, columns[0]: [0.2, 3.0], "response": [-5, 12]},
        },
        "split": {"file": str(split)},
    }


def score_regions(root, predictions):
    scoring = root / "scorer_only.csv"
    if not scoring.exists():
        return {}
    final = pd.read_csv(predictions).merge(pd.read_csv(scoring), on="sample_id", validate="one_to_one")
    metrics = {}
    for region, rows in final.groupby("region"):
        metrics[f"final_{region}_n"] = len(rows)
        for label in ("observed", "clean"):
            metrics[f"final_{region}_{label}_rmse"] = prediction_metrics(rows[label], rows["prediction"])["rmse"]
    return metrics


def external(case):
    root = Path("data") / case
    meta = read_json(root / "metadata.json")
    return {
        "data": {
            "local_path": str(root / "table.csv"),
            "target_column": meta["target"],
            "feature_columns": meta["features"],
            "units": meta["units"],
        },
        "split": {"file": str(root / "split.csv"), "group_column": "group_id"},
    }


def run_case(root, case, engines, seed, seconds, overrides=None, data_seed=42, *, prepare_only=False):
    if not prepare_only:
        raise ValueError("Automatic benchmark selection/Final execution was removed; use --prepare-only or --collect-tasks for saved results.")
    if overrides:
        if set(overrides) - {"search", "model_assets", "constraints", "evaluation"} or set(
            overrides.get("search", {})
        ) - {"engine_options", "runtimes", "operators"}:
            raise ValueError(
                "Benchmark overrides accept runtimes/assets/native options, common operators, constraints and evaluation; engines/seeds/budgets stay in CLI arguments"
            )
    root.mkdir(parents=True)
    cfg = synthetic(root, case, seed, data_seed) if case not in {"yacht", "airfoil", "concrete"} else external(case)
    cfg.update(
        search={
            "engines": engines,
            "seeds": [seed],
            "operators": ["add", "sub", "mul", "div"],
            "max_wall_seconds": seconds,
            "max_export_candidates": 20,
            "allow_partial_engines": True,
            "engine_options": {
                k: v
                for k, v in {
                    "pysr": {"niterations": 12, "populations": 3, "ncycles_per_iteration": 80},
                    "physo": {"epochs": 5, "batch_size": 128, "max_n_evaluations": 640},
                    "dictionary": {"max_terms": 6, "max_degree": 3},
                    "linear": {"max_terms": 10, "ridge": 1e-6},
                }.items()
                if k in engines
            },
        },
        constraints={"max_nodes": 30},
        evaluation={"selection_policy": "balanced", "absolute_tolerance": 1e-6},
    )
    if case == "rational_constraints":
        cfg["constraints"].update(
            probes=[{"x": 0.0}, {"x": 0.5}, {"x": 4.0}], response_bounds=[0.0, 4.0], require_units=True
        )
    if case in {"damped_resonance", "activated_kinetics"}:
        cfg = deep_merge(cfg, physics_options(case, engines))
        points = (
            [{"omega": w, "gamma": g} for w in (3.0, 9.0, 11.0, 18.0, 40.0) for g in (0.15, 0.6, 2.0)]
            if case == "damped_resonance"
            else [
                {"temperature": T, "concentration_ratio": c}
                for T in (280.0, 300.0, 330.0, 360.0, 390.0)
                for c in (0.5, 1.0, 1.8)
            ]
        )
        cfg["constraints"].update(probes=points, response_bounds=[0.0, None])
    cfg = resolve_config(deep_merge(cfg, overrides or {}))
    if prepare_only:
        # The existing Docker launcher owns runtime-family dispatch. This script
        # only generates the same fixed data/config, without a second executor.
        write_json(root / "config.json", cfg)
        return []
def collect_container_tasks(root):
    """Evaluate declared regions of frozen predictions; never fit or select again."""
    rows = []
    tables = sorted(root.glob("*/*/task/engine_results.csv"))
    if not tables:
        raise ValueError("No completed common-container benchmark tables")
    for table in tables:
        task = table.parent
        execution = read_json(task / "container_execution.json")
        if execution["status"] not in {"succeeded", "failed"}:
            raise ValueError(f"Task has not reached a terminal state: {task}")
        provenance = read_json(task / "prepare/run.json")
        for row in pd.read_csv(table).to_dict("records"):
            row.update(
                case=task.parent.parent.name,
                seed=int(task.parent.name),
                data_sha256=provenance["data_sha256"],
                split_sha256=provenance["split_sha256"],
                prepared_sha256=digest(task / "prepare/run.json"),
            )
            if row["status"] == "ok":
                row.update(score_regions(task.parent, task / "finalize" / row["engine"] / "predictions.csv"))
            board = task / "compare" / row["engine"] / "leaderboard.csv"
            row["eligible_candidates"] = (
                int(pd.read_csv(board)["eligible"].sum()) if board.exists() and board.stat().st_size > 1 else 0
            )
            if row["engine"] != "combined":
                summaries = [read_json(p) for p in (task / "discover" / row["engine"]).glob("*/engine_summary.json")]
                for key in ("native_candidates", "native_draws", "native_ledger_rows", "selection_candidates"):
                    values = [s[key] for s in summaries if key in s]
                    row[key] = sum(values) if values and len(values) == len(summaries) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    print(
        pd.DataFrame(rows)
        .reindex(columns=["case", "seed", "engine", "status", "final_rmse", "final_ood_observed_rmse"])
        .to_string(index=False),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--engines", nargs="+", default=["mean", "linear", "dictionary", "physo", "pysr"])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=MAINTAINED_CASES,
        default=["polynomial", "noisy_extrapolation", "rational_constraints", "yacht"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Fixed PCG64 data seed for scientific task families; independent of method seeds",
    )
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--overrides", help="Local YAML/JSON settings for isolated runtimes, assets and native options")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate fixed benchmark data/configs for run_linux.py task dispatch; no search/evaluation",
    )
    parser.add_argument(
        "--collect-tasks",
        action="store_true",
        help="Collect completed run_linux task results/region scores from --output; no fitting or selection",
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.collect_tasks:
        if args.prepare_only:
            raise ValueError("Choose preparation or completed-task collection")
        collect_container_tasks(root)
        return
    if root.exists():
        raise ValueError("Benchmark output must be new; no previous evidence is overwritten")
    if not args.prepare_only:
        parser.error("Choose --prepare-only or --collect-tasks; automatic Final execution was removed")
    overrides = load_config(args.overrides) if args.overrides else None
    for case in args.cases:
        for seed in args.seeds:
            print(f"Starting {case}/{seed}", flush=True)
            run_case(
                root / case / str(seed),
                case,
                copy.copy(args.engines),
                seed,
                args.seconds,
                overrides,
                args.data_seed,
                prepare_only=args.prepare_only,
            )
            print(f"Prepared {root / case / str(seed) / 'config.json'}", flush=True)


if __name__ == "__main__":
    main()
