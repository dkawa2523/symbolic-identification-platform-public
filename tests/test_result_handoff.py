"""Purpose and detached handoff counterexamples, not native performance tests."""

import shutil

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import (
    assess_prediction_use,
    calibrate_prediction,
    choose_hypothesis,
    compare_equations,
    create_study,
    export_result,
    finalize_hypothesis,
    load_result,
    render_report,
)
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, prepare, read_json, write_json


def problem(root):
    root.mkdir()
    x = np.arange(12, dtype=float)
    frame = pd.DataFrame({"sample_id": [f"{i:03d}" for i in range(12)], "x": x, "y": 2 * x + 5})
    frame.to_csv(root / "data.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 6 + ["selection"] * 4 + ["final"] * 2).to_csv(
        root / "split.csv", index=False
    )
    return {
        "data": {
            "local_path": str(root / "data.csv"),
            "feature_columns": ["x"],
            "target_column": "y",
            "units": {"x": "1", "y": "1"},
        },
        "split": {"file": str(root / "split.csv")},
        "search": {"engines": ["linear"], "operators": ["add", "mul"]},
    }, frame


def test_response_use_does_not_adopt_absolute_levels_or_open_final(tmp_path):
    cfg, _ = problem(tmp_path / "input")
    plan = {
        "problem": cfg,
        "sources": [
            {"manual": {"id": "biased", "expression": "2*x+1005"}},
            {"manual": {"id": "flat", "expression": "5"}},
        ],
        "purpose": "response_comparison",
        "engineering_tolerance": 0.001,
        "response_validation": {
            "pairs": [["006", "007"], ["008", "009"]],
            "tolerance": {"value": 0.001, "unit": "1"},
            "scope": "Two specified paired changes; level offset is outside this response purpose",
        },
    }
    study = create_study(plan, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    choice = choose_hypothesis(
        comparison, run_dir=tmp_path / "choice", candidate_id="biased@manual", reason="Observed-pair response only"
    )
    decision = read_json(choice / "decision.json")
    assert decision["status"] == "adopt"
    assert decision["response_evidence"]["metrics"]["rmse"] == 0
    with pytest.raises(ValueError, match="local_prediction-purpose"):
        calibrate_prediction(
            study,
            choice,
            tmp_path / "never-read.csv",
            run_dir=tmp_path / "wrong-purpose-interval",
            exchangeability="independent_rows",
            observation_model="Not a prediction choice",
        )
    response_handoff = export_result(comparison, decision_dir=choice, run_dir=tmp_path / "response-handoff")
    assert load_result(response_handoff)["prediction"] is None
    assert "engineering_tolerance" not in decision["criteria"]
    result = finalize_hypothesis(study, choice, run_dir=tmp_path / "response-result")
    assert read_json(result.artifacts["response_result"])["final_accessed"] is False
    assert not list(tmp_path.rglob("final_access.json"))
    flat = choose_hypothesis(
        comparison, run_dir=tmp_path / "flat-choice", candidate_id="flat@manual", reason="Wrong response retained"
    )
    assert read_json(flat / "decision.json")["status"] == "defer"
    point_study = create_study({**plan, "purpose": "local_prediction"}, run_dir=tmp_path / "point-study")
    point_comparison = compare_equations(point_study, run_dir=tmp_path / "point-comparison")
    point_choice = choose_hypothesis(
        point_comparison,
        run_dir=tmp_path / "point-choice",
        candidate_id="biased@manual",
        reason="Absolute prediction requires its own error",
    )
    assert read_json(point_choice / "decision.json")["status"] == "defer"
    html = render_report(comparison, output=tmp_path / "response.html", decision_dir=choice)
    assert "Paired-response analysis" in html.read_text(encoding="utf-8")


def test_response_pairs_cannot_name_final_or_silently_infer_numeric_ids(tmp_path):
    cfg, _ = problem(tmp_path / "input")
    for pair in [["010", "011"], [6, 7]]:
        with pytest.raises(ValueError, match="Selection IDs"):
            create_study(
                {
                    "problem": cfg,
                    "sources": [{"manual": {"id": "m", "expression": "2*x+5"}}],
                    "response_validation": {
                        "pairs": [pair],
                        "tolerance": {"value": 0.1, "unit": "1"},
                        "scope": "Not a valid Selection pair",
                    },
                },
                run_dir=tmp_path / f"bad-{pair[0]}",
            )


def test_detached_result_moves_with_same_prediction_interval_and_no_source(tmp_path, monkeypatch):
    original = tmp_path / "original"
    cfg, frame = problem(original)
    study = create_study(
        {
            "problem": cfg,
            "sources": [{"manual": {"id": "m", "expression": "2*x+5"}}],
            "engineering_tolerance": 0.001,
            "final_history": "known_exposed",
        },
        run_dir=original / "study",
    )
    comparison = compare_equations(study, run_dir=original / "comparison")
    policy = dict(
        alpha=0.1,
        exchangeability="independent_rows",
        group_column=None,
        observation_model="Technical independent-row fixture, not a scientific coverage study",
        max_width=0.5,
        unit="1",
    )
    choice = choose_hypothesis(
        comparison,
        run_dir=original / "choice",
        candidate_id="m@manual",
        reason="Frozen point/interval fixture",
        calibration_plan=policy,
    )
    x = np.linspace(0, 10, 30)
    pd.DataFrame({"sample_id": [f"cal-{i}" for i in range(30)], "x": x, "y": 2 * x + 5 + 0.01}).to_csv(
        original / "calibration.csv", index=False
    )
    interval = calibrate_prediction(
        study,
        choice,
        original / "calibration.csv",
        run_dir=original / "interval",
        alpha=0.1,
        exchangeability=policy["exchangeability"],
        observation_model=policy["observation_model"],
    )
    use = assess_prediction_use(choice, interval, run_dir=original / "use")
    final = finalize_hypothesis(study, use, run_dir=original / "final")
    from ml_platform_symbolic.expressions import parse_expression
    from ml_platform_symbolic.model import load_model, save_model

    point = load_model(final.artifacts["model_bundle"])
    wrong = save_model(
        tmp_path / "wrong-equation.zip",
        parse_expression("9999", point.schema["variables"], trainable=False),
        point.schema,
        point.metadata,
    )
    with pytest.raises(ValueError, match="equation differs"):
        export_result(comparison, run_dir=tmp_path / "wrong-handoff", decision_dir=use, prediction=wrong)
    package = export_result(
        comparison,
        run_dir=tmp_path / "handoff",
        decision_dir=use,
        audit_dir=original / "final",
        prediction=final.artifacts["model_bundle"],
        prediction_interval=interval,
    )
    assert original.resolve().is_relative_to(tmp_path.resolve()) and (
        tmp_path / "hidden-original"
    ).resolve().is_relative_to(tmp_path.resolve())
    original.rename(tmp_path / "hidden-original")  # Only this newly created fixture tree.
    moved = tmp_path / "moved"
    assert package.resolve().is_relative_to(tmp_path.resolve()) and moved.resolve().is_relative_to(tmp_path.resolve())
    shutil.move(str(package), moved)
    result = load_result(moved)
    np.testing.assert_allclose(result["prediction"].predict(frame), frame.y)
    np.testing.assert_array_equal(result["interval"].predict(frame)[0], result["prediction"].predict(frame))

    def missing_runtime(_path):
        raise ValueError("controlled unavailable predictor runtime")

    monkeypatch.setattr("ml_platform_symbolic.model.load_model", missing_runtime)
    render_report(moved, output=tmp_path / "detached.html")
    assert result["view"]["decision"]["selected_id"] == "m@manual"
    assert result["view"]["detached_result"]
    (moved / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        load_result(moved)


def test_native_only_payload_survives_detached_readout_without_becoming_predictor(tmp_path):
    cfg, _ = problem(tmp_path / "input")
    refs = prepare(resolve_config(cfg), tmp_path / "prepared")
    summary = write_json(
        tmp_path / "native-summary.json",
        {
            "engine": "unavailable_native_fixture",
            "seed": 0,
            "status": "failed",
            "native_execution_status": "completed",
            "portable_status": "unavailable",
            "reason": "Unknown native operator; conversion unimplemented, not proof the function is wrong",
            "native_formula": "native_special(x)",
            "prepared_sha256": digest(refs["run"]),
        },
    )
    study = create_study(
        {"prepared_run": refs["run"], "sources": [{"discovery": str(summary)}]}, run_dir=tmp_path / "study"
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    cid = "unavailable_native_fixture/0@native_result"
    package = export_result(comparison, run_dir=tmp_path / "handoff", representative_ids=[cid])
    result = load_result(package)
    assert result["prediction"] is None and result["interval"] is None
    card = result["view"]["candidates"][0]
    assert card["native_result"]["retained_native_record"]["native_formula"] == "native_special(x)"
    assert card["selection"] is None and card["native_result"]["prediction_status"] == "unavailable"


def test_zero_weight_extreme_row_does_not_set_metric_scale_or_hide_nonfinite_input():
    from ml_platform_symbolic.metrics import prediction_metrics

    for scale in (1e-200, 1.0, 1e200):
        result = prediction_metrics([scale, 1e308], [0, -1e308], weights=[1, 0])
        assert result["rmse"] == pytest.approx(scale, rel=1e-12, abs=0)
        assert result["mae"] == pytest.approx(scale, rel=1e-12, abs=0)
    invalid = prediction_metrics([1e-200, np.inf], [0, 0], weights=[1, 0])
    assert invalid["rmse"] is None and "nonfinite" in invalid["reason"]


def test_public_pair_comparison_preserves_unrepresentable_squared_contrast(tmp_path):
    cfg, frame = problem(tmp_path / "input")
    frame["y"] = frame.x * 1e-200
    frame.to_csv(cfg["data"]["local_path"], index=False)
    study = create_study(
        {
            "problem": cfg,
            "sources": [
                {"manual": {"id": "zero", "expression": "0"}},
                {"manual": {"id": "tiny", "expression": "1e-200*x"}},
            ],
        },
        run_dir=tmp_path / "study",
    )
    compared = compare_equations(study, run_dir=tmp_path / "comparison")
    pair = read_json(compared / "comparison.json")["pairs"][0]
    assert pair["prediction_rms_difference"] > 0
    assert pair["paired_mse_difference"] is None
    assert pair["paired_mse_difference_reason"]
    assert pair["paired_mse_difference_unit"] == "(1)^2"


def test_implicit_residual_diagnostic_reuses_stable_common_norm():
    from ml_platform_symbolic.expressions import parse_expression
    from ml_platform_symbolic.hypotheses import residual_evidence

    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    target = {"column": "y", "unit": "1"}
    equation = parse_expression("1e-200*target", [*variables, {"id": "target", **target}], trainable=False)
    evidence, _, _ = residual_evidence(
        {"kind": "implicit", "residual": equation}, np.zeros((2, 1)), np.array([1.0, 2.0]), variables, target
    )
    assert evidence["residual_rms"] == pytest.approx(np.sqrt(2.5) * 1e-200, rel=1e-12, abs=0)
    assert evidence["residual_rms_reason"] is None


def test_assessed_interval_identity_is_shared_by_display_export_and_both_readers(tmp_path):
    """Same predictor/width cannot transfer an assessed calibration decision."""
    original = tmp_path / "original"
    cfg, frame = problem(original)
    study = create_study(
        {
            "problem": cfg,
            "sources": [{"manual": {"id": "m", "expression": "2*x+5"}}],
            "engineering_tolerance": 0.001,
            "final_history": "known_exposed",
        },
        run_dir=original / "study",
    )
    comparison = compare_equations(study, run_dir=original / "comparison")
    policy = dict(
        alpha=0.1,
        exchangeability="independent_rows",
        group_column=None,
        observation_model="Artifact identity fixture, not a coverage study",
        max_width=0.5,
        unit="1",
    )
    choice = choose_hypothesis(
        comparison,
        run_dir=original / "choice",
        candidate_id="m@manual",
        reason="Freeze predictor and width policy",
        calibration_plan=policy,
    )
    intervals = {}
    for name, delta in [("A", 0.01), ("B", 10.0), ("C", 0.01)]:
        x = np.linspace(0, 10, 20)
        csv = original / (name + ".csv")
        pd.DataFrame({"sample_id": [f"{name}-{i:03d}" for i in range(20)], "x": x, "y": 2 * x + 5 + delta}).to_csv(
            csv, index=False
        )
        intervals[name] = calibrate_prediction(
            study,
            choice,
            csv,
            run_dir=original / name,
            alpha=0.1,
            exchangeability=policy["exchangeability"],
            observation_model=policy["observation_model"],
        )
    use = assess_prediction_use(choice, intervals["A"], run_dir=original / "use-A")
    assert read_json(use / "decision.json")["status"] == "adopt"
    for name in ("B", "C"):
        with pytest.raises(ValueError, match="calibration artifact assessed"):
            render_report(
                comparison, decision_dir=use, prediction_interval=intervals[name], output=tmp_path / (name + ".html")
            )
        with pytest.raises(ValueError, match="calibration artifact assessed"):
            export_result(
                comparison, decision_dir=use, prediction_interval=intervals[name], run_dir=tmp_path / ("wrong-" + name)
            )
    with pytest.raises(ValueError, match="already assessed"):
        calibrate_prediction(
            study,
            use,
            original / "A.csv",
            run_dir=tmp_path / "recalibrated",
            exchangeability=policy["exchangeability"],
            observation_model=policy["observation_model"],
        )
    # Omitting a display argument resolves the one adopted reference, never another calibration.
    html = render_report(comparison, decision_dir=use, output=tmp_path / "automatic-A.html")
    assert digest(intervals["A"] / "interval.json") in html.read_text(encoding="utf-8")
    good = export_result(comparison, decision_dir=use, prediction_interval=intervals["A"], run_dir=tmp_path / "good")
    # An over-wide, explicitly deferred calibration is retained as evidence, not deleted.
    deferred = assess_prediction_use(choice, intervals["B"], run_dir=original / "use-B")
    b_package = export_result(
        comparison, decision_dir=deferred, prediction_interval=intervals["B"], run_dir=tmp_path / "deferred-B"
    )
    assert load_result(b_package)["view"]["decision"]["status"] == "defer"
    # Equivalent to the unmodified rc2 public producer's bad package: internally
    # consistent B readout/files, but still the A use decision. Not a new producer success.
    legacy_bad = tmp_path / "legacy-A-B"
    shutil.copytree(good, legacy_bad)
    view = read_json(b_package / "report.json")
    view["decision"] = read_json(good / "report.json")["decision"]
    view["decision_sha256"] = read_json(good / "report.json")["decision_sha256"]
    write_json(legacy_bad / "report.json", view)
    for file in ("interval.json", "prediction.json"):
        shutil.copy2(intervals["B"] / file, legacy_bad / "interval" / file)
    manifest = read_json(legacy_bad / "result_manifest.json")
    manifest["files"] = {name: digest(legacy_bad / name) for name in manifest["files"]}
    write_json(legacy_bad / "result_manifest.json", manifest)
    for load_predictors in (False, True):
        with pytest.raises(ValueError, match="calibration artifact assessed"):
            load_result(legacy_bad, load_predictors=load_predictors)
    moved = tmp_path / "moved-A"
    assert original.resolve().is_relative_to(tmp_path.resolve())
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    original.rename(tmp_path / "hidden-original")
    shutil.move(str(good), moved)
    for load_predictors in (False, True):
        result = load_result(moved, load_predictors=load_predictors)
        assert result["view"]["decision"]["interval_sha256"] == digest(moved / "interval/interval.json")
        if load_predictors:
            point, lower, upper = result["interval"].predict(frame)
            np.testing.assert_allclose(point, frame.y)
            np.testing.assert_allclose(upper - lower, 0.02)
    assert not list(tmp_path.rglob("final_access.json"))
