"""Verify installed-wheel inference and durable native provenance without native engines."""

import argparse
import importlib.util
import json
from pathlib import Path

import ml_platform_symbolic
import numpy as np
import pandas as pd
from ml_platform_symbolic import load_ensemble, load_model, load_relation, run_task
from ml_platform_symbolic.data import digest
from ml_platform_symbolic.implicit import relation_variables

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="+", type=Path)
parser.add_argument("--report", type=Path, default=Path("build/portable-wheel.json"))
parser.add_argument(
    "--inference-subdir",
    default="wheel-infer",
    help="New per-run inference folder; preserve older verification outputs",
)
args = parser.parse_args()
if Path(args.inference_subdir).name != args.inference_subdir or args.inference_subdir in {"", ".", ".."}:
    raise ValueError("inference-subdir must be a plain new directory name")
module_path = Path(ml_platform_symbolic.__file__)
if "site-packages" not in module_path.parts:
    raise RuntimeError(f"Expected an installed wheel, got source import: {module_path}")
for package in (
    "torch",
    "pysr",
    "physo",
    "TorchSisso",
    "ProGED",
    "parfam",
    "psrn",
    "pysips",
    "bingo",
    "smcpy",
    "sympy",
    "flash_ansr",
    "symbolicregression",
    "sympytorch",
    "numexpr",
    "sklearn",
    "clearml",
):
    if importlib.util.find_spec(package) is not None:
        raise RuntimeError(f"Lean wheel inference unexpectedly contains {package}")
report = []
for directory in args.runs:
    for bundle in sorted((directory / "finalize").glob("*/equation_model.zip")):
        model = load_model(bundle)
        provenance = model.metadata["provenance"]
        frame = pd.read_csv(provenance["config"]["data"]["local_path"])
        split = pd.read_csv(provenance["config"]["split"]["file"])
        final = split[split.partition == "final"]
        frame = frame.set_index("sample_id").loc[final.sample_id].reset_index()
        saved = pd.read_csv(bundle.with_name("predictions.csv")).set_index("sample_id").loc[frame.sample_id]
        predicted = model.predict(frame)
        reverse = model.predict(frame[frame.columns[::-1]])
        np.testing.assert_array_equal(reverse, predicted)
        np.testing.assert_allclose(predicted, saved.prediction, rtol=1e-12, atol=1e-12)
        discovery = model.metadata["selected_discovery"]
        assert discovery["runtime_image_id"].startswith("sha256:")
        if discovery["assets"] is not None:
            assert discovery["assets"]["manifest_sha256"]
        output = directory / args.inference_subdir / bundle.parent.name
        result = run_task(
            {
                "task": "symbolic_infer",
                "model": {"local_model_path": str(bundle)},
                "data": {"local_path": provenance["config"]["data"]["local_path"]},
            },
            run_dir=output,
        )
        report.append(
            {
                "bundle": str(bundle),
                "engine": discovery["engine"],
                "max_saved_error": float(np.max(np.abs(predicted - saved.prediction))),
                "inference_result": str(result.tables["predictions"]),
                "image": discovery["runtime_image_id"],
            }
        )
    for summary_path in (directory / "discover").rglob("engine_summary.json"):
        summary = json.loads(summary_path.read_text())
        path = summary.get("native_artifacts", {}).get("ensemble")
        if path is None:
            continue
        ensemble = load_ensemble(path)
        provenance = json.loads((directory / "prepare/run.json").read_text())
        frame = pd.read_csv(provenance["config"]["data"]["local_path"])
        split = pd.read_csv(provenance["config"]["split"]["file"])
        frame = frame.set_index("sample_id").loc[split.loc[split.partition == "final", "sample_id"]].reset_index()
        saved = (
            pd.read_csv(directory / "ensemble" / summary["engine"] / "final_predictions.csv")
            .set_index("sample_id")
            .loc[frame.sample_id]
        )
        predicted = ensemble.predict(frame)
        np.testing.assert_array_equal(predicted, ensemble.predict(frame[frame.columns[::-1]]))
        np.testing.assert_allclose(predicted, saved.prediction, rtol=1e-12, atol=1e-12)
        quantiles = ensemble.structural_quantiles(frame, [0.1, 0.9])
        np.testing.assert_allclose(quantiles, saved[["structural_q10", "structural_q90"]], rtol=1e-12, atol=1e-12)
        # This oracle was saved from the ACTUAL native sampling-state graphs
        # before their later point refits, not computed by the portable evaluator.
        population = json.loads(Path(summary["native_artifacts"]["population"]).read_text())
        probe = population["native_ensemble_probe"]
        probe_frame = pd.DataFrame(probe["X"], columns=[v["column"] for v in ensemble.schema["variables"]])
        probe_prediction = ensemble.predict(probe_frame)
        np.testing.assert_allclose(probe_prediction, probe["weighted_prediction"], rtol=1e-10, atol=1e-10)
        report.append(
            {
                "bundle": path,
                "engine": summary["engine"] + ":structural_mean",
                "members": len(ensemble.members),
                "max_saved_error": float(np.max(np.abs(predicted - saved.prediction))),
                "max_native_probe_error": float(np.max(np.abs(probe_prediction - probe["weighted_prediction"]))),
                "image": summary["runtime_image_id"],
                "scope": ensemble.metadata["uncertainty_scope"],
            }
        )
for directory in args.runs:
    check_path = directory / "implicit-native-check.json"
    if not check_path.exists():
        continue
    replay = json.loads(check_path.read_text())
    if replay.get("catalog_path"):
        catalog_path = Path(replay["catalog_path"])
    else:
        # Older data-only replays did not record an explicit path. Resolve the
        # exact immutable catalog by hash, without importing its former engine.
        matches = [
            path
            for path in (directory / "discover").rglob("implicit_relations.json")
            if digest(path) == replay["catalog_sha256"]
        ]
        if len(matches) != 1:
            raise ValueError("Legacy replay needs one matching catalog or an explicit catalog_path")
        catalog_path = matches[0]
    assert digest(catalog_path) == replay["catalog_sha256"]
    catalog = json.loads(catalog_path.read_text())
    frame = pd.DataFrame(replay["points"], columns=[v["column"] for v in relation_variables(catalog["schema"])])
    for check in replay["members"]:
        relation = load_relation(catalog_path, check["candidate_id"])
        expected = np.asarray([np.nan if value is None else value for value in check["native_residual"]])
        finite = np.isfinite(expected)
        error = 0.0
        if finite.any():
            actual = relation.residual(frame.loc[finite])
            np.testing.assert_array_equal(actual, relation.residual(frame.loc[finite, frame.columns[::-1]]))
            scale = max(float(np.max(np.abs(expected[finite]))), 1e-12)
            np.testing.assert_allclose(actual, expected[finite], rtol=1e-5, atol=1e-7 * scale)
            error = float(np.max(np.abs(actual - expected[finite])))
        if not finite.all():
            try:
                relation.residual(frame.loc[~finite])
            except ValueError:
                pass
            else:
                raise AssertionError("Portable residual must retain native domain failures")
        if relation.predictor is None:
            try:
                relation.predict(frame.drop(columns=catalog["schema"]["target"]["column"]))
            except ValueError:
                pass
            else:
                raise AssertionError("Ambiguous relation became a predictor")
        report.append(
            {
                "kind": "implicit_relation",
                "bundle": str(catalog_path),
                "candidate_id": check["candidate_id"],
                "engine": check["candidate_id"].split("/", 1)[0],
                "max_native_probe_error": error,
                "reordered_columns_exact": True,
                "prediction_available": relation.predictor is not None,
            }
        )
args.report.write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {
            "installed_module": str(module_path),
            "portable_bundles": len(report),
            "maximum_saved_error": max((r["max_saved_error"] for r in report if "max_saved_error" in r), default=None),
        }
    )
)
