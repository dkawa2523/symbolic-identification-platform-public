from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import (
    audit_prediction_interval,
    calibrate_prediction,
    choose_hypothesis,
    compare_equations,
    create_study,
    finalize_hypothesis,
    load_prediction,
    load_prediction_interval,
    render_report,
)
from ml_platform_symbolic.data import read_json, write_json
from ml_platform_symbolic.equation_evaluation import load_study
from ml_platform_symbolic.expressions import parse_expression


def outputs(tmp_path):
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    target = {"column": "y", "unit": "1"}
    eq = parse_expression("2*x0*x0+1", variables, trainable=False)
    other = parse_expression("2*x0*x0+3", variables, trainable=False)
    schema = {"variables": variables, "target": target}
    members = [{"equation": eq, "weight": 0.75}, {"equation": other, "weight": 0.25}]
    ensemble = write_json(
        tmp_path / "ensemble.json",
        {
            "schema_version": 1,
            "kind": "structural_point_ensemble",
            "schema": schema,
            "members": members,
            "metadata": {},
        },
    )
    residual = parse_expression("target*target-x0", [*variables, {"id": "target", **target}], trainable=False)
    relation = write_json(
        tmp_path / "relation.json",
        {
            "schema_version": 1,
            "kind": "implicit_relations",
            "schema": schema,
            "members": [
                {
                    "candidate_id": "ambiguous",
                    "residual": residual,
                    "predictor": None,
                    "prediction_unavailable_reason": "Two branches; no target-domain branch was declared",
                }
            ],
            "metadata": {},
        },
    )
    return ensemble, relation, eq


def test_native_outputs_prediction_comparison_and_residual_are_separate(synthetic_config, tmp_path):
    ensemble, relation, eq = outputs(tmp_path)
    plan = {
        "problem": synthetic_config,
        "engineering_tolerance": 1.0,
        "sources": [
            {"manual": {"id": "point", "equation": eq}},
            {"ensemble": str(ensemble)},
            {"relation": str(relation), "candidate_id": "ambiguous"},
        ],
    }
    original = deepcopy(plan)
    study = create_study(plan, run_dir=tmp_path / "study")
    assert plan == original
    context, records, rows, arrays = load_study(study)
    ensemble_id = records[1]["id"]
    np.testing.assert_allclose(arrays["c1_selection_prediction"], arrays["selection_y"] + 0.5)
    assert rows[1]["eligible"] and not rows[2]["prediction_available"]
    assert rows[2]["residual_evidence"]["selection"]["n_finite"] > 0
    response = rows[2]["response_evaluation"][0]
    assert response["status"] == "not_evaluated" and response["point_origin"] == "current_study"
    assert response["reason"] == "Two branches; no target-domain branch was declared"

    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    from ml_platform_symbolic.reporting import export_outputs
    import csv
    export_outputs(comparison, output_dir=tmp_path / "output", figures=False)
    with (tmp_path / "output/tables/response_checks.csv").open() as handle:
        response_rows = list(csv.DictReader(handle))
    unavailable = [r for r in response_rows if r["candidate_id"] == records[2]["id"]]
    assert len(unavailable) == 1 and unavailable[0]["reason"] == response["reason"]
    assert unavailable[0]["point_origin"] == "current_study"
    result = read_json(comparison / "comparison.json")
    assert len(result["pairs"]) == 1 and len(result["unavailable_prediction_comparisons"]) == 2
    assert result["pairs"][0]["equivalence"] == "inconclusive"
    decision = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "decision",
        candidate_id=ensemble_id,
        reason="Frozen weighted mean; conditional structural spread is not an observation interval",
    )
    final = finalize_hypothesis(study, decision, run_dir=tmp_path / "final")
    model = load_prediction(final.artifacts["model_bundle"])
    frame = pd.read_csv(synthetic_config["data"]["local_path"])
    np.testing.assert_allclose(model.predict(frame), frame.y + 0.5)
    assert [m["weight"] for m in model.record["members"]] == [0.75, 0.25]
    render_report(comparison, output=tmp_path / "report.html")


def test_group_calibration_is_frozen_separate_and_reloadable(synthetic_config, tmp_path):
    _, _, eq = outputs(tmp_path)
    study = create_study(
        {
            "problem": synthetic_config,
            "engineering_tolerance": 0.1,
            "sources": [{"manual": {"id": "point", "equation": eq}}],
        },
        run_dir=tmp_path / "study",
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    decision = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "decision",
        candidate_id="point@manual",
        reason="Fixed reference before calibration",
    )
    rng = np.random.default_rng(29)

    def data(prefix):
        x = rng.uniform(0.5, 3, 120)
        frame = pd.DataFrame(
            {
                "sample_id": [f"{prefix}{i}" for i in range(120)],
                "group": np.repeat([f"{prefix}g{i}" for i in range(40)], 3),
                "x": x,
                "y": 2 * x * x + 1 + rng.normal(0, 0.02, len(x)),
            }
        )
        path = tmp_path / (prefix + ".csv")
        frame.to_csv(path, index=False)
        return path, frame

    calibration, _ = data("cal")
    before = (decision / "decision.json").read_bytes()
    interval = calibrate_prediction(
        study,
        decision,
        calibration,
        run_dir=tmp_path / "interval",
        alpha=0.1,
        exchangeability="exchangeable_groups",
        observation_model="Independent Gaussian observations in independent groups",
        group_column="group",
    )
    audit, frame = data("audit")
    scored = audit_prediction_interval(interval, audit, output=tmp_path / "coverage.json")
    assert read_json(scored)["n_independent_units"] == 40
    model = load_prediction_interval(interval)
    point, low, high = model.predict(frame)
    assert np.all(high >= low) and np.all((low <= point) & (point <= high))
    assert (decision / "decision.json").read_bytes() == before
    with pytest.raises(ValueError, match="independent"):
        audit_prediction_interval(interval, calibration, output=tmp_path / "bad.json")
    with pytest.raises(ValueError, match="overlaps"):
        calibrate_prediction(
            study,
            decision,
            synthetic_config["data"]["local_path"],
            run_dir=tmp_path / "bad-cal",
            exchangeability="exchangeable_groups",
            observation_model="test",
            group_column="group",
        )
    with pytest.raises(ValueError, match="group"):
        calibrate_prediction(
            study,
            decision,
            calibration,
            run_dir=tmp_path / "bad-row",
            exchangeability="independent_rows",
            observation_model="test",
        )


def test_scientific_annotation_is_not_syntax_family(synthetic_config, tmp_path):
    _, _, eq = outputs(tmp_path)
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [{"manual": {"id": "point", "equation": eq}}],
            "hypothesis_groups": {
                "point@manual": {
                    "label": "local empirical quadratic",
                    "rationale": "user hypothesis, not a mechanism claim",
                }
            },
        },
        run_dir=tmp_path / "study",
    )
    compared = compare_equations(study, run_dir=tmp_path / "comparison")
    result = read_json(compared / "comparison.json")
    assert result["scientific_hypotheses"]["point@manual"]["label"] == "local empirical quadratic"
    assert result["candidates"][0]["family"].startswith("syntax-")


def test_confirmed_member_fidelity_failure_is_not_missing_evidence(synthetic_config, tmp_path):
    ensemble, _, _ = outputs(tmp_path)
    payload = read_json(ensemble)
    payload["members"][0]["native_check"] = {"passed": False}
    write_json(ensemble, payload)
    study = create_study(
        {"problem": synthetic_config, "sources": [{"ensemble": str(ensemble)}]}, run_dir=tmp_path / "study"
    )
    _, records, rows, _ = load_study(study)
    assert records[0]["native_check"]["passed"] is False
    assert not rows[0]["eligible"] and "fidelity failed" in rows[0]["reason"]


def test_calibration_public_coordinates_group_identity_and_display_binding(synthetic_config, tmp_path):
    cfg = deepcopy(synthetic_config)
    cfg["data"]["units"] = {"x": "cm", "y": "mm"}
    definition = {
        "id": "reference",
        "expression": "a*x*x+b",
        "parameters": {
            "a": {"value": 20, "unit": "1/m", "trainable": False},
            "b": {"value": 0.001, "unit": "m", "trainable": False},
        },
    }
    study = create_study(
        {"problem": cfg, "engineering_tolerance": 0.01, "sources": [{"manual": definition}]}, run_dir=tmp_path / "study"
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    decision = choose_hypothesis(
        comparison, run_dir=tmp_path / "choice", candidate_id="reference@manual", reason="public coordinate regression"
    )
    rng = np.random.default_rng(30)

    def frame(prefix):
        x = rng.uniform(0.5, 3, 120)
        data = pd.DataFrame(
            {
                "sample_id": [f"{prefix}-{i}" for i in range(120)],
                "group": np.repeat([f"{prefix}-g{i}" for i in range(40)], 3),
                "x": x,
                "y": 2 * x * x + 1 + rng.normal(0, 0.02, len(x)),
            }
        )
        path = tmp_path / (prefix + ".csv")
        data.to_csv(path, index=False)
        return path, data

    calibration, _ = frame("cal")
    kwargs = {
        "exchangeability": "exchangeable_groups",
        "observation_model": "synthetic independent repeated experiments",
        "group_column": "group",
    }
    default = calibrate_prediction(study, decision, calibration, run_dir=tmp_path / "default", **kwargs)
    explicit = calibrate_prediction(
        study, decision, calibration, run_dir=tmp_path / "explicit", units={"x": "cm", "y": "mm"}, **kwargs
    )
    assert read_json(default / "interval.json")["quantile"] == read_json(explicit / "interval.json")["quantile"]
    audit, data = frame("audit")
    left = read_json(audit_prediction_interval(default, audit, output=tmp_path / "left.json"))
    right = read_json(
        audit_prediction_interval(explicit, audit, output=tmp_path / "right.json", units={"x": "cm", "y": "mm"})
    )
    assert left["mean_width"] == right["mean_width"] and left["unit_coverage"] == right["unit_coverage"]
    model = load_prediction_interval(default)
    _, low, high = model.predict(data)
    np.testing.assert_allclose(high - low, 2000 * model.metadata["quantile"])
    with pytest.raises(ValueError, match="preserve.*group"):
        calibrate_prediction(
            study,
            decision,
            calibration,
            run_dir=tmp_path / "row-groups",
            exchangeability="exchangeable_groups",
            group_column="sample_id",
            observation_model="invalid weakening",
        )
    wrong = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "other-choice",
        candidate_id="reference@manual",
        reason="a genuinely different choice artifact",
    )
    with pytest.raises(ValueError, match="different candidates/choices"):
        render_report(comparison, output=tmp_path / "bad.html", decision_dir=wrong, prediction_interval=default)
