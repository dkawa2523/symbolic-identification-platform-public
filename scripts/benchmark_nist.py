"""Offline, supplied-form NIST coefficient-estimation benchmark (not discovery)."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from ml_platform_symbolic.expressions import evaluate, parse_expression
from ml_platform_symbolic.fitting import fit_coefficients

DATA = Path(__file__).resolve().parents[1] / "data/nist"
CASES = {
    "Misra1a": ("b1*(1-exp(-b2*x0))", "9d3194d537d9a6bbf92eb819504070923a1885d0191f40b6b3aaeedd130b98e2"),
    "Chwirut2": ("exp(-b1*x0)/(b2+b3*x0)", "c37e5522a34f9a57e18d7d10f8736202eabc0c33b8e7ad0010e618db04b57411"),
}
VARIABLES = [{"id": "x0", "column": "x", "unit": "1"}]
TOLERANCES = {"prediction_relative_l2": 1e-6, "rss_relative_error": 1e-6}


def parse_nist(raw):
    """Read all published observations and both starts; reject malformed rows."""
    lines = raw.decode("ascii").splitlines()
    parameters = {}
    locations = {}
    for number, line in enumerate(lines, 1):
        match = re.match(r"\s*(b\d+)\s*=\s*(.*)", line)
        if match:
            values = [float(v) for v in match[2].split()]
            if len(values) != 4:
                raise ValueError("Expected two starts, certified value and standard deviation")
            parameters[match[1]] = values[:3]
            locations[match[1]] = number
    start = next(i for i, line in enumerate(lines) if re.match(r"Data:\s+y\s+x\s*$", line)) + 1
    rows = [[float(v) for v in line.split()] for line in lines[start:] if line.strip()]
    n = int(next(line.split(":")[1] for line in lines if line.startswith("Number of Observations:")))
    rss_line = next(i for i, line in enumerate(lines) if line.startswith("Residual Sum of Squares:"))
    rss = float(lines[rss_line].split(":")[1])
    data = np.asarray(rows)
    if not parameters or data.shape != (n, 2) or not np.isfinite(data).all():
        raise ValueError("Missing parameters or malformed/incomplete observations")
    return (
        parameters,
        data[:, 1:],
        data[:, 0],
        rss,
        {"parameters": locations, "rss": rss_line + 1, "data": [start + 1, len(lines)]},
    )


def equation(formula, parameters, column):
    return parse_expression(
        formula,
        VARIABLES,
        parameters={
            name: {"value": values[column], "trainable": True, "bounds": [None, None], "unit": "1"}
            for name, values in parameters.items()
        },
        trainable=False,
    )


def agreement(prediction, reference, y, certified_rss):
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction: no rows may be omitted")
    scale = float(np.max(np.abs(reference))) or 1.0
    error = float(np.linalg.norm((prediction - reference) / scale) / np.linalg.norm(reference / scale))
    rss = float(np.sum((prediction - y) ** 2))
    return {"prediction_relative_l2": error, "rss_relative_error": abs(rss / certified_rss - 1), "rss": rss}


def run(max_nfev=1000):
    attempts = []
    for name, (formula, expected_hash) in CASES.items():
        raw = (DATA / f"{name}.dat").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"{name}: source hash mismatch")
        parameters, X, y, rss, lines = parse_nist(raw)
        reference = evaluate(equation(formula, parameters, 2), X, VARIABLES)
        for start in (0, 1):
            record = {
                "dataset": name,
                "start": start + 1,
                "formula": formula,
                "initial": {k: v[start] for k, v in parameters.items()},
                "certified": {k: v[2] for k, v in parameters.items()},
                "certified_rss": rss,
                "rows": len(y),
                "sha256": digest,
                "source_lines": lines,
                "source": f"https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/data/nist/{name}.dat",
            }
            try:
                fitted = fit_coefficients(equation(formula, parameters, start), X, y, VARIABLES, max_nfev=max_nfev)
                prediction = evaluate(fitted, X, VARIABLES)
                metrics = agreement(prediction, reference, y, rss)
                record["prediction_rows"] = {
                    "partition": "NIST all observations used for coefficient fit; not held-out",
                    "observed": y.tolist(), "predicted": prediction.tolist(),
                    "residual": (y - prediction).tolist(), "valid": np.isfinite(prediction).tolist(),
                }
                record.update(metrics)
                record["parameters"] = {k: v["value"] for k, v in fitted["parameters"].items()}
                record["status"] = (
                    "agreement" if all(metrics[k] <= v for k, v in TOLERANCES.items()) else "disagreement"
                )
            except (ValueError, FloatingPointError, OverflowError) as exc:
                record.update(status="failed", reason=str(exc))
            attempts.append(record)
    import scipy

    return {
        "purpose": "supplied_formula_numerical_estimation",
        "max_nfev": max_nfev,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "tolerances": TOLERANCES,
        "attempts": attempts,
        "case_context": {name: {
            "question": "Published NIST least-squares coefficient estimation from two starts",
            "truth_status": "Published certified fitted reference; not a true physical law or blind structure recovery",
            "reference_formula": formula, "target_unit": "source numeric coordinates; physical unit unspecified",
            "variables": "x0 = published x; y = published observations",
            "scope": "All rows used for fitting. No independent prediction test. Supplied form and certified coefficients are explicitly known.",
            "source": f"https://www.itl.nist.gov/div898/strd/nls/data/{name}.shtml",
        } for name, (formula, _) in CASES.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=1000)
    args = parser.parse_args()
    report_directory = args.output.with_suffix(".report")
    if args.output.exists() or report_directory.exists():
        raise ValueError("Use new output/report paths; preserve previous evidence")
    result = run(args.max_nfev)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from benchmark_readout import write_report

    print(json.dumps([(r["dataset"], r["start"], r["status"]) for r in result["attempts"]]))
    print("Open report:", write_report(args.output, report_directory))

