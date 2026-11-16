"""Transport/error regressions using synthetic native records, not native execution evidence."""

import json
import subprocess

import pytest
from ml_platform_symbolic.equation_evaluation import create_study, load_study
from ml_platform_symbolic.equation_evaluation.backends import evidence


def test_native_identifiable_without_nll_crossing_is_not_a_confidence_endpoint():
    profile = {
        "x": [0.0083307, 0.0083332, 0.00833346],
        "objective": [-99 + 1.49, -99, -99 + 1.23],
        "native_endpoints": {"left": 0.0083307, "right": 0.00833346},
        "retcodes": {"left": "Identifiable", "right": "Identifiable"},
    }
    original = profile["native_endpoints"].copy()
    evidence.annotate_crossings(profile, -99, 1.920729410347062)
    assert profile["native_endpoints"] == original
    assert profile["accepted_endpoints"] == {"left": None, "right": None}
    assert not profile["endpoint_validation"]["all_verified"]
    profile["objective"] = [-99 + 1.920729410347062, -99, -99 + 1.920729410347062]
    evidence.annotate_crossings(profile, -99, 1.920729410347062)
    assert profile["accepted_endpoints"] == original
    assert profile["endpoint_validation"]["all_verified"]


def inputs(synthetic_config, tmp_path, monkeypatch):
    synthetic_config["data"]["domain"] = {"x": [0.5, 3]}
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [
                {
                    "manual": {
                        "id": "model",
                        "expression": "a*x*x+1",
                        "parameters": {"a": {"value": 2.0, "trainable": True, "bounds": [0, 10], "unit": "1"}},
                    }
                }
            ],
        },
        run_dir=tmp_path / "study",
    )
    context, records, rows, arrays = load_study(study)
    context["study_sha256"] = "test"
    monkeypatch.setenv("SYMBOLIC_LIKELIHOOD_PROJECT", "/not-used")
    monkeypatch.setattr(evidence.shutil, "which", lambda _: "julia")
    return context, records, rows, arrays


def test_profile_point_outside_domain_does_not_launch_worker(synthetic_config, tmp_path, monkeypatch):
    args = inputs(synthetic_config, tmp_path, monkeypatch)
    monkeypatch.setattr(evidence.subprocess, "run", lambda *a, **k: pytest.fail("Must reject before native execution"))
    result = evidence.run(
        *args,
        tmp_path,
        {"sigma": 1, "quantities": [{"id": "outside", "prediction_at": {"x": 4}, "unit": "1", "range": [0, 100]}]},
    )
    assert result["outcome"] == "unavailable"
    assert "input domain" in result["results"][0]["reason"]


@pytest.mark.parametrize("execution_failed", [False, True])
def test_null_native_values_and_execution_failure_keep_their_meaning(
    synthetic_config, tmp_path, monkeypatch, execution_failed
):
    args = inputs(synthetic_config, tmp_path, monkeypatch)

    def native(command, **kwargs):
        if execution_failed:
            raise subprocess.CalledProcessError(1, command)
        record = {
            "version": "1.6.1",
            "source_predictions": args[-1]["c0_train_prediction"].tolist(),
            "outcome": "ok",
            "optimum": [2.0],
            "objective_min": 1.0,
            "quantities": [],
            "profiles": [
                {
                    "name": "a",
                    "objective": [None, 1.0, None],
                    "retcodes": {"left": "Failure", "right": "Failure"},
                    "native_endpoints": {"left": None, "right": None},
                }
            ],
        }
        from pathlib import Path

        Path(command[-1]).write_text(json.dumps(record))

    monkeypatch.setattr(evidence.subprocess, "run", native)
    result = evidence.run(*args, tmp_path, {"sigma": 1, "parameter_ranges": {"a": [1, 3]}})
    assert result["outcome"] == ("failed" if execution_failed else "ok")
    if not execution_failed:
        profile = result["results"][0]["native"]["profiles"][0]
        assert profile["delta_nll"] == [None, 0.0, None]
        assert profile["native_endpoints"] == {"left": None, "right": None}
