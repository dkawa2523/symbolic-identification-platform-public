"""Prepare frozen partial-equation benchmark data; no search or target scoring.

stdlib-only; offline by default. Outputs are immutable unless byte-identical.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "partial_equations"
SOURCE_SHA256 = "755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = data.encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"Refusing to replace a frozen artifact: {path}")
    path.write_bytes(encoded)


def table(path, columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    write(path, stream.getvalue())


def manifest(path, data):
    write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def prepare_abalone():
    folder = DATA / "abalone"
    raw = (folder / "source.zip").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Abalone source ZIP does not match pinned SHA256")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows = list(csv.reader(archive.read("abalone.data").decode().splitlines()))
        write(folder / "source.names", archive.read("abalone.names").decode())
    if len(rows) != 4177 or any(len(row) != 9 for row in rows):
        raise ValueError("Unexpected source shape")
    values, splits, audit = [], [], []
    for index, row in enumerate(rows):
        sex, length, diameter, height, whole, shucked, viscera, shell, rings = row
        length_value, d, h, y = map(float, (length, diameter, height, shell))
        if not all(math.isfinite(v) for v in (length_value, d, h, y)) or length_value <= 0:
            raise ValueError("Unexpected nonfinite or nonpositive length")
        sid = f"abalone_{index:04d}"
        # Group identical measured geometry, including differing sex, together.
        group = hashlib.sha256(f"{length},{diameter},{height}".encode()).hexdigest()[:16]
        bucket = int(group, 16) % 10
        partition = (
            "final_extrapolation"
            if length_value > 0.65
            else ("selection" if bucket == 0 else "final_interpolation" if bucket == 1 else "train")
        )
        values.append([sid, group, sex, length_value, d, h, d / length_value, h / length_value, y])
        splits.append([sid, partition])
        if h <= 0 or h > length_value:
            audit.append([sid, "nonpositive_height" if h <= 0 else "height_exceeds_length", partition])
    table(
        folder / "table.csv",
        [
            "sample_id",
            "group_id",
            "sex",
            "length",
            "diameter",
            "height",
            "diameter_ratio",
            "height_ratio",
            "shell_weight",
        ],
        values,
    )
    table(folder / "split.csv", ["sample_id", "partition"], splits)
    table(folder / "quality_flags.csv", ["sample_id", "flag", "partition"], audit)
    manifest(
        folder / "metadata.json",
        {
            "dataset": "UCI Abalone repurposed shell-weight prediction",
            "rows": len(rows),
            "source_url": "https://archive.ics.uci.edu/static/public/1/abalone.zip",
            "landing_url": "https://archive.ics.uci.edu/dataset/1/abalone",
            "retrieved_date": "2026-11-08",
            "doi": "10.24432/C55C7W",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994). Abalone. UCI Machine Learning Repository.",
            "source_sha256": SOURCE_SHA256,
            "pretraining_overlap": "unknown",
            "features": ["length", "diameter", "height", "diameter_ratio", "height_ratio"],
            "target": "shell_weight",
            "not_model_inputs": [
                "sex",
                "sample_id",
                "group_id",
                "rings",
                "whole_weight",
                "shucked_weight",
                "viscera_weight",
            ],
            "coordinate_units": "Stored UCI coordinates only; source describes scaling by 200. Do not interpret fitted constants as physical density or unscaled SI coefficients.",
            "changes": "Repurpose shell_weight target; omit other weights and rings; add shape ratios, geometry groups, fixed split and quality flags. No rows removed.",
            "hypothesis": "shell_weight = length * diameter * height * h(diameter_ratio,height_ratio)",
            "hypothesis_status": "Declared geometric allometry hypothesis, not known true biology; shell mass is not solid-body volume.",
            "split_policy": "length>0.65 is final_extrapolation. Otherwise SHA256 geometry-group modulo 10: 0 selection, 1 final_interpolation, 2..9 train.",
            "counts": {p: sum(r[1] == p for r in splits) for p in sorted({r[1] for r in splits})},
            "flagged_rows": len(audit),
            "performance_evaluated": False,
        },
    )


CASES = {
    "additive": ("2*x+h(z)", "0.7+z*z", lambda x, z: 2 * x + 0.7 + z * z),
    "multiplicative": ("x*h(z)", "0.7+z*z", lambda x, z: x * (0.7 + z * z)),
    "shared_component": ("h(x)+h(z)", "0.7+t*t", lambda x, z: 1.4 + x * x + z * z),
    "wrong_prior": ("2*x+h(z)", "No valid h(z): true x slope is 3", lambda x, z: 3 * x + 0.7 + z * z),
    "confounded": ("2*x+h(z)", "0.7+z*z; train z=x", lambda x, z: 2 * x + 0.7 + z * z),
    "zero_factor": ("x*h(z)", "0.7+z*z; x=0 train conveys no h information", lambda x, z: x * (0.7 + z * z)),
    "gauge_nonidentifiable": ("h(x)+g(x)", "h and g not separately identifiable", lambda x, z: 0.7 + x * x),
    "positive_domain": ("log(h(z))+x", "1+z*z", lambda x, z: math.log(1 + z * z) + x),
}


def prepare_synthetic():
    for case, (template, oracle, function) in CASES.items():
        for seed in (1729, 2718, 3141):
            rng = random.Random(seed)
            records, splits, truth = [], [], []
            for partition, count in (
                ("train", 120),
                ("selection", 40),
                ("final_interpolation", 80),
                ("final_extrapolation", 80),
                ("final_deconfounding", 80),
            ):
                for i in range(count):
                    x, z = rng.uniform(-1, 1), rng.uniform(-1, 1)
                    if partition == "final_extrapolation":
                        x, z = rng.uniform(1.2, 2), rng.uniform(1.2, 2)
                    if case == "confounded" and partition != "final_deconfounding":
                        z = x
                    if case == "zero_factor" and partition in ("train", "selection"):
                        x = 0.0
                    elif case == "multiplicative" and i % 10 == 0:
                        x = 0.0
                    sid = f"{case}_{seed}_{partition}_{i:03d}"
                    clean = function(x, z)
                    # Known noise; separately preserved clean oracle for scoring only.
                    y = clean + rng.gauss(0, 0.01)
                    records.append([sid, x, z, y])
                    truth.append([sid, clean])
                    splits.append([sid, partition])
            folder = DATA / "synthetic" / case / str(seed)
            table(folder / "table.csv", ["sample_id", "x", "z", "y"], records)
            table(folder / "split.csv", ["sample_id", "partition"], splits)
            table(folder / "oracle_scoring_only.csv", ["sample_id", "clean_y"], truth)
        manifest(
            DATA / "synthetic" / case / "spec.json",
            {
                "case": case,
                "template_not_runtime_syntax": template,
                "oracle_scoring_only": oracle,
                "data_seeds": [1729, 2718, 3141],
                "noise_sd": 0.01,
                "features": ["x", "z"],
                "target": "y",
                "performance_evaluated": False,
                "provenance": "Fresh stdlib random draws from disclosed formulas; not unseen formula discovery.",
            },
        )


if __name__ == "__main__":
    manifest(
        DATA / "generator_environment.json",
        {
            "python": platform.python_version(),
            "system": platform.system(),
            "reason": "Exact floating-point byte reproduction is pinned to the generation runtime; inference uses the frozen CSVs.",
        },
    )
    prepare_abalone()
    prepare_synthetic()
    manifest(
        DATA / "files_sha256.json",
        {
            str(p.relative_to(DATA)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((DATA).rglob("*.csv"))
        },
    )
    print("Prepared pinned Abalone and 8 synthetic cases x 3 draws; no model evaluation.")
