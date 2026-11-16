"""A few public-workflow counterexamples across the repaired authorities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import calibrate_prediction, choose_hypothesis, compare_equations, create_study
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import model_output, read_json
from ml_platform_symbolic.domains import region_mask, region_points
from ml_platform_symbolic.equation_evaluation.api import load_study
from ml_platform_symbolic.metrics import prediction_metrics
from ml_platform_symbolic.model import save_model
from ml_platform_symbolic.problem import load_problem, problem_config
from ml_platform_symbolic.selection import audit_hypotheses


def problem(tmp_path, *, grouped=False, tiny=False):
    x = np.linspace(0.1, 0.9, 60)
    frame = pd.DataFrame(
        {
            "sample_id": [f"{i:03d}" for i in range(60)],
            "group": [f"{i // 3:03d}" for i in range(60)],
            "stratum": np.where(np.arange(60) % 2, "first", "second"),
            "x": x,
            "y": x * (1e-200 if tiny else 1.0),
        }
    )
    source, split = tmp_path / "data.csv", tmp_path / "split.csv"
    frame.to_csv(source, index=False)
    frame[["sample_id"]].assign(partition=["train"] * 36 + ["selection"] * 12 + ["final"] * 12).to_csv(
        split, index=False
    )
    cfg = {
        "data": {
            "local_path": str(source),
            "feature_columns": ["x"],
            "target_column": "y",
            "units": {"x": "1", "y": "1"},
        },
        "domain": {"kind": "box", "columns": ["x"], "bounds": [[0, 1]]},
        "split": {"file": str(split), "group_column": "group" if grouped else None},
        "search": {"engines": ["dictionary"], "operators": ["add", "sub", "mul", "div", "sqrt", "log"]},
    }
    return cfg, frame


def choice(tmp_path, *, grouped=False, tiny=False, strata=False, sources=None):
    cfg, frame = problem(tmp_path, grouped=grouped, tiny=tiny)
    plan = {
        "problem": cfg,
        "sources": sources or [{"manual": {"id": "point", "expression": "0" if tiny else "x"}}],
        "engineering_tolerance": 1e-201 if tiny else 1e-8,
        "final_history": "known_exposed",
    }
    if strata:
        plan["strata_column"] = "stratum"
    study = create_study(plan, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison", pair_ids=[])
    decision = (
        choose_hypothesis(
            comparison, run_dir=tmp_path / "choice", candidate_id="point@manual", reason="Controlled boundary fixture"
        )
        if sources is None
        else None
    )
    return cfg, frame, study, comparison, decision


def test_box_only_problem_and_direction_request_use_the_declared_coordinates(tmp_path):
    cfg, frame = problem(tmp_path)
    cfg["constraints"] = {
        "response_checks": [
            {
                "direction": {"x": 1},
                "path_unit": "1",
                "step": 0.01,
                "points": [{"x": 0.4}],
                "order": 1,
                "sign": 1,
                "tolerance": 0,
            }
        ]
    }
    resolved = resolve_config(cfg)
    study = create_study(
        {"problem": resolved, "sources": [{"manual": {"id": "sqrt", "expression": "sqrt(x)"}}]},
        run_dir=tmp_path / "study",
    )
    context, records, rows, arrays = load_study(study)
    assert rows[0]["eligible"]
    assert region_mask(arrays["domain_X"], records[0]["execution_domain"]).all()
    assert arrays["domain_X"].min() >= 0
    document = load_problem(context["prepared"]["run"])
    assert problem_config(document)["search"]["operators"] == resolved["search"]["operators"]
    with pytest.raises(ValueError, match="not standalone"):
        resolve_config(document["config"])
    variables = [{"id": "x0", "column": "x", "domain": None}]
    region = {"kind": "box", "columns": ["x"], "bounds": [[0, 1]]}
    assert region_mask(region_points(frame[["x"]].to_numpy(), variables, 0, region), region).all()


@pytest.mark.parametrize("kind", ["row", "group", "changed_source"])
def test_calibration_cannot_reuse_frozen_observations_via_csv_inference(tmp_path, kind):
    grouped = kind == "group"
    cfg, frame, study, _, decision = choice(tmp_path, grouped=grouped)
    reused = frame.iloc[: 36 if grouped else 12].copy()
    if grouped:
        reused["sample_id"] = [f"new_{i}" for i in range(len(reused))]
    if kind == "changed_source":
        changed = frame.copy()
        changed["sample_id"] = [f"replacement_{v}" for v in changed.sample_id]
        changed.to_csv(cfg["data"]["local_path"], index=False)
    calibration = tmp_path / "calibration.csv"
    reused.to_csv(calibration, index=False)
    with pytest.raises(ValueError, match="overlap"):
        calibrate_prediction(
            study,
            decision,
            calibration,
            run_dir=tmp_path / "interval",
            exchangeability="exchangeable_groups" if grouped else "independent_rows",
            group_column="group" if grouped else None,
            observation_model="Controlled duplicate fixture",
        )


def test_metric_scale_changes_do_not_turn_threshold_failure_into_adoption(tmp_path):
    for scale in (1e-200, 1.0, 1e200):
        metrics = prediction_metrics(np.array([scale, 2 * scale]), np.zeros(2))
        assert metrics["rmse"] / scale == pytest.approx(np.sqrt(2.5))
        assert metrics["r2"] == pytest.approx(-9)
    _, _, study, _, decision = choice(tmp_path, tiny=True)
    assert read_json(decision / "decision.json")["status"] == "defer"
    assert not list(study.parent.rglob("final_access.json"))


def test_final_strata_and_known_exposure_are_frozen_before_raw_changes(tmp_path):
    cfg, frame, study, _, decision = choice(tmp_path, strata=True)
    frame["stratum"] = "changed-after-choice"
    frame.to_csv(cfg["data"]["local_path"], index=False)
    audit = audit_hypotheses(study, decision, run_dir=tmp_path / "audit")
    result = read_json(audit / "audit.json")
    with np.load(audit / "predictions.npz") as saved:
        np.testing.assert_allclose(saved["c0_residual"], saved["observed"] - saved["c0_prediction"])
        assert len(saved["observed"]) == len(saved["sample_id"])
    assert set(result["results"][0]["strata"]) == {"first", "second"}
    context, _, _, _ = load_study(study)
    assert context["final_exposure"]["same_data_exposed"]
    assert read_json(decision / "decision.json")["declared_final_history"] == "known_exposed"


def test_bundle_sources_keep_method_seed_and_do_not_become_one_imported_run(tmp_path):
    from ml_platform_symbolic.expressions import parse_expression

    cfg, _ = problem(tmp_path)
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": None}]
    equation = parse_expression("x0", variables, trainable=False)
    sources = []
    for seed in range(3):
        bundle = tmp_path / f"point{seed}.zip"
        save_model(
            bundle,
            equation,
            {"variables": variables, "target": {"column": "y", "unit": "1"}, "feature_ranges": [[0.1, 0.9]]},
            {"engine": "declared-source", "seed": seed},
        )
        sources.append({"bundle": str(bundle)})
    study = create_study({"problem": cfg, "sources": sources}, run_dir=tmp_path / "study")
    _, records, _, _ = load_study(study)
    assert {r["origin"]["seed"] for r in records} == {0, 1, 2}
    compared = compare_equations(study, run_dir=tmp_path / "comparison", pair_ids=[])
    assert read_json(compared / "comparison.json")["families"][0]["distinct_source_runs"] == 3


def test_output_coordinate_overflow_is_rejected_without_an_absolute_floor():
    schema = {"target": {"column": "y", "unit": "m"}}
    with pytest.raises(ValueError, match="nonfinite"):
        model_output(np.array([1e308]), schema, "mm")
    np.testing.assert_allclose(model_output(np.array([1e-200]), schema, "mm"), [1e-197], rtol=1e-14, atol=0)


def test_requested_derivatives_and_pair_scope_preserve_every_candidate(tmp_path):
    cfg, _ = problem(tmp_path)
    sources = [{"manual": {"id": f"point{i}", "expression": f"x+{i}"}} for i in range(24)]
    study = create_study({"problem": cfg, "sources": sources}, run_dir=tmp_path / "study")
    _, _, _, arrays = load_study(study)
    assert not any("jacobian" in name for name in arrays)
    comparison = compare_equations(study, run_dir=tmp_path / "bounded")
    saved = read_json(comparison / "comparison.json")
    assert len(saved["candidates"]) == 24
    assert len(saved["pair_scope"]["ids"]) <= 4
    assert "uncomputed" in saved["pair_scope"]["mode"]
    explicit = compare_equations(study, run_dir=tmp_path / "explicit", pair_ids=["point0@manual", "point23@manual"])
    explicit_result = read_json(explicit / "comparison.json")
    assert len(explicit_result["candidates"]) == 24 and len(explicit_result["pairs"]) == 1
    assert "uncomputed" in explicit_result["pair_scope"]["mode"]
    pair = explicit_result["pairs"][0]
    assert pair["prediction_rms_difference"] == pytest.approx(23)
    assert pair["sensitivity_difference"] == []
    with pytest.raises(ValueError, match="unique IDs"):
        compare_equations(study, run_dir=tmp_path / "bad_pair", pair_ids=["missing"])
    requested = create_study(
        {"problem": cfg, "sources": sources[:2], "derivatives": True}, run_dir=tmp_path / "derivatives"
    )
    assert "c0_selection_jacobian" in load_study(requested)[3]


def test_frozen_evidence_readout_does_not_require_live_training_sources(tmp_path):
    from ml_platform_symbolic import render_report

    _, _, study, comparison, decision = choice(tmp_path)
    context, _, _, _ = load_study(study)
    # Only the fixture's prepared metadata is moved; saved evidence is unchanged.
    original = Path(context["prepared"]["run"])
    original.rename(original.with_suffix(".kept"))
    with pytest.raises(FileNotFoundError):
        load_study(study)  # New analysis/selection still checks live provenance.
    output = render_report(comparison, decision_dir=decision, output=tmp_path / "readout.html")
    assert "point@manual" in output.read_text(encoding="utf-8")


def test_extension_identity_includes_package_helper_files(tmp_path):
    import subprocess
    import sys
    import uuid

    name = "boundary_extension_" + uuid.uuid4().hex
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "1"\ndef initialize():\n    pass\n', encoding="utf-8")
    helper = package / "helper.py"
    helper.write_text("meaning = 1\n", encoding="utf-8")
    # Runtime initialization is process-lifetime state. Exercise it in a real
    # child so a deliberately changed temporary module cannot poison later jobs.
    script = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from ml_platform_symbolic.domains import register_runtime_extension\n"
        f"frozen = register_runtime_extension({name!r}, '1')\n"
        f"Path({str(helper)!r}).write_text('meaning = 2\\n', encoding='utf-8')\n"
        "try:\n"
        f"    register_runtime_extension({name!r}, '1', expected_sha256=frozen['sha256'], expected_implementation_sha256=frozen['implementation_sha256'])\n"
        "except ValueError as error:\n"
        "    assert 'implementation files changed' in str(error)\n"
        "else:\n    raise AssertionError('Changed helper accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_engineering_tolerance_has_an_explicit_quantity_coordinate(tmp_path):
    cfg, _ = problem(tmp_path)
    cfg["data"]["units"]["y"] = "m"
    parameters = {
        name: {
            "value": value,
            "unit": "m",
            "trainable": False,
            "bounds": [value, value],
            "origin": "controlled_reference",
        }
        for name, value in {"gain": 1.0, "offset": 0.0005}.items()
    }
    study = create_study(
        {
            "problem": cfg,
            "sources": [{"manual": {"id": "point", "expression": "gain*x+offset", "parameters": parameters}}],
            "engineering_tolerance": {"value": 1.0, "unit": "mm"},
        },
        run_dir=tmp_path / "study",
    )
    context, _, _, _ = load_study(study)
    assert context["engineering_tolerance"] == pytest.approx(0.001)
    compared = compare_equations(study, run_dir=tmp_path / "comparison")
    chosen = choose_hypothesis(
        compared,
        run_dir=tmp_path / "choice",
        candidate_id="point@manual",
        reason="Error0.5mm is below the frozen1mm criterion",
    )
    assert read_json(chosen / "decision.json")["status"] == "adopt"


def test_interval_endpoints_do_not_turn_finite_inputs_into_infinite_predictions():
    from types import SimpleNamespace

    from ml_platform_symbolic.prediction_intervals import PredictionInterval

    point = SimpleNamespace(
        schema={"target": {"column": "y", "unit": "1"}}, predict=lambda *args, **kwargs: np.array([1e308])
    )
    interval = PredictionInterval(point, {"quantile": 1e308, "unit": "1"})
    with pytest.raises(ValueError, match="endpoints are nonfinite"):
        interval.predict(pd.DataFrame({"x": [1.0]}))


def test_cleanup_failure_keeps_primary_error_and_is_not_retried(monkeypatch):
    from ml_platform_symbolic import worker_process as lifecycle

    class OwnedFixture:
        returncode = None
        pid = 1001

    fixture = OwnedFixture()
    calls = []

    def denied(process):
        calls.append(process.pid)
        raise PermissionError("controlled stop denial; no operating-system command")

    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *args, **kwargs: fixture)
    monkeypatch.setattr(lifecycle, "_stop_and_reap", denied)
    with pytest.raises(RuntimeError, match="original cancellation") as caught:
        with lifecycle.worker_process(["controlled-fixture"]):
            raise RuntimeError("original cancellation")
    assert "cleanup unconfirmed" in caught.value.__notes__[0]
    with pytest.raises(PermissionError):
        lifecycle.stop_worker(fixture)
    assert calls == [1001]
