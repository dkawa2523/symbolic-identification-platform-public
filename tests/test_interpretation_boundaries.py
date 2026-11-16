from copy import deepcopy

import numpy as np
import pandas as pd
from ml_platform_symbolic import (
    assess_audit_support,
    audit_hypotheses,
    choose_hypothesis,
    compare_equations,
    create_study,
    evaluate_equations,
    load_prediction,
    render_report,
)
from ml_platform_symbolic.ensemble import FormulaEnsemble, weighted_formula_quantiles
from ml_platform_symbolic.equation_evaluation.api import load_study
from ml_platform_symbolic.execution_views import bind_view
from ml_platform_symbolic.expressions import parse_expression
from ml_platform_symbolic.hypotheses import predict_working, save_prediction
from ml_platform_symbolic.question_results import conditional_request, missing_precision, same_question


def test_quantity_identity_does_not_substitute_definition_confidence_or_noise():
    request = conditional_request(
        {"expression": "plateau"},
        confidence_level=0.95,
        observation_model={"family": "iid_gaussian_known_sigma", "sigma": 0.01, "unit": "MPa"},
    )
    same_units = conditional_request(
        {"expression": "plateau"},
        confidence_level=0.95,
        observation_model={"family": "iid_gaussian_known_sigma", "sigma": 10000.0, "unit": "Pa"},
    )
    assert request == same_units
    question = dict(
        candidate_id="law",
        kind="conditional_interval",
        estimand="conditional_model",
        quantity="plateau",
        unit="MPa",
        coordinate_role=None,
        status="established",
        endpoints=[1.0, 1.01],
        request=request,
    )
    required = {"plateau": dict(unit="MPa", max_width=0.1, request=request)}
    analyses = [{"result": {"method": "replacement_evaluator", "results": [{"id": "law", "questions": [question]}]}}]
    assert not missing_precision(analyses, "law", required)
    for definition, confidence, sigma in [
        ({"prediction_at": {"x": 1}}, 0.95, 0.01),
        ({"expression": "plateau"}, 0.5, 0.01),
        ({"expression": "plateau"}, 0.95, 0.02),
    ]:
        other = deepcopy(question)
        other["request"] = conditional_request(
            definition,
            confidence_level=confidence,
            observation_model={"family": "iid_gaussian_known_sigma", "sigma": sigma, "unit": "MPa"},
        )
        assert not same_question(question, other)
        assert missing_precision([{"result": {"results": [{"id": "law", "questions": [other]}]}}], "law", required)


def test_quantile_roundoff_and_zero_mass_share_saved_execution(tmp_path):
    variables = [dict(id="x0", column="x", unit="1", domain=None)]
    members = [
        dict(candidate_id=str(i), equation=parse_expression(str(y), variables, trainable=False), weight=w)
        for i, (y, w) in enumerate([(0.0, 0.5), (10.0, 0.49999999999)])
    ]
    record = bind_view(
        dict(
            id="ensemble",
            kind="ensemble",
            members=members,
            native_check={"passed": True, "kind": "constant-expression oracle"},
            target={"column": "y", "unit": "1"},
        ),
        dict(name="weighted_quantile", settings={"q": 0.999999999999}),
    )
    X = np.array([[1.0]])
    before = deepcopy(record)
    expected = FormulaEnsemble({"variables": variables, "target": record["target"]}, members, {}).structural_quantiles(
        pd.DataFrame({"x": [1.0]}), [0.999999999999]
    )[:, 0]
    np.testing.assert_array_equal(expected, [10.0])
    np.testing.assert_array_equal(predict_working(record, X, variables)[0], expected)
    path = save_prediction(
        tmp_path / "prediction.json",
        record,
        {"variables": variables, "target": record["target"], "feature_ranges": [[0, 2]]},
        {},
    )
    np.testing.assert_array_equal(load_prediction(path).predict(pd.DataFrame({"x": [1.0]})), expected)
    assert record == before
    np.testing.assert_array_equal(
        weighted_formula_quantiles(np.array([[-100.0, 0.0, 10.0]]), [0, 0.5, 0.5], [0, 1]), [[0.0, 10.0]]
    )


def test_report_binds_actual_selected_id_and_stored_final_without_predicting(synthetic_config, tmp_path, monkeypatch):
    synthetic_config = deepcopy(synthetic_config)
    synthetic_config["data"]["units"]["y"] = "mm"
    study = create_study(
        dict(
            problem=synthetic_config,
            engineering_tolerance=0.01,
            sources=[
                {
                    "manual": dict(
                        id="good",
                        expression="a*x*x+b",
                        parameters={
                            "a": dict(value=0.002, unit="m", trainable=False),
                            "b": dict(value=0.0012, unit="m", trainable=False),
                        },
                    )
                },
                {"manual": dict(id="bad", expression="0")},
            ],
        ),
        run_dir=tmp_path / "study",
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    choice = choose_hypothesis(
        comparison, run_dir=tmp_path / "choice", candidate_id="good@manual", reason="Technical fixture only"
    )
    audit = audit_hypotheses(study, choice, run_dir=tmp_path / "audit")
    support = assess_audit_support(audit, choice, run_dir=tmp_path / "support")
    from ml_platform_symbolic.equation_evaluation import numerical

    monkeypatch.setattr(
        numerical, "predict_record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("renderer must not predict"))
    )
    report = render_report(
        comparison,
        output=tmp_path / "report.html",
        decision_dir=choice,
        audit_dir=support,
        representative_ids=["bad@manual"],
    )
    from ml_platform_symbolic.report_data import build_report_view

    view = build_report_view(comparison, decision_dir=choice, audit_dir=support, representative_ids=["bad@manual"])
    assert "good@manual" in view["display_ids"]
    assert view["audit"]["rows"]["good@manual"]["prediction_support"]["established"] is True
    chosen = next(c for c in view["candidates"] if c["id"] == "good@manual")
    np.testing.assert_allclose(view["audit"]["rows"][chosen["id"]]["target_metrics"]["rmse"], 0.2)
    np.testing.assert_allclose(chosen["final"]["target_metrics"]["rmse"], 0.2)
    np.testing.assert_allclose(chosen["selection"]["rmse"], 0.2)
    np.testing.assert_allclose(chosen["final"]["prediction_support"]["observed_rmse"], 0.2)
    np.testing.assert_allclose(chosen["final"]["prediction_support"]["criteria"]["rmse_max"], 10)
    assert view["audit"]["unit"] == "mm"
    assert len(view["candidates"]) == 2 and report.exists()


def test_interpretation_keeps_representation_dependent_parts_separate_from_total(synthetic_config, tmp_path):
    parameters = {name: dict(value=v, unit="1", trainable=True) for name, v in [("a", 2.0), ("b", 3.0)]}
    study = create_study(
        dict(
            problem=synthetic_config,
            sources=[
                {"manual": dict(id="expanded", expression="a*x+b*x", parameters=parameters)},
                {"manual": dict(id="factored", expression="(a+b)*x", parameters=parameters)},
                {"manual": dict(id="poor", expression="0")},
            ],
        ),
        run_dir=tmp_path / "study",
    )
    result = evaluate_equations(
        study,
        "interpretation",
        run_dir=tmp_path / "analysis",
        options={
            "paths": [dict(id="x", at={"x": 1.0}, direction={"x": 1.0}, steps=[-0.1, 0, 0.1], path_unit="1")],
            "max_wall_seconds": 30,
        },
    )
    from ml_platform_symbolic.data import read_json

    data = read_json(result / "result.json")
    assert data["outcome"] == "ok", data
    left, right, poor = data["results"]
    assert len(left["terms"]) == 2 and len(right["terms"]) == 1
    np.testing.assert_allclose(left["paths"][0]["prediction"], right["paths"][0]["prediction"])
    assert left["term_sum_check"]["max_roundoff_difference"] < 1e-12
    assert "Whole nested expression" in right["decomposition_scope"]
    assert len(load_study(study)[1]) == 3
    assert poor["paths"][0]["prediction"] == [0.0, 0.0, 0.0]
    geometry = left["paths"][0]["reference_geometry"]
    assert geometry == poor["paths"][0]["reference_geometry"]
    assert geometry["data_role"] == "train" and geometry["n_reference_rows"] == 36
    assert geometry["status"] == "computed"
    comparison = compare_equations(
        study, run_dir=tmp_path / "comparison", evaluations=[result], pair_ids=[left["id"], right["id"]]
    )
    paired = read_json(comparison / "comparison.json")["analysis_comparisons"]
    assert len(paired) == 1 and paired[0]["maximum_difference"] == 0
    assert "poor@manual" not in (paired[0]["left"], paired[0]["right"])

    from ml_platform_symbolic.report_data import build_report_view

    supplement = compare_equations(study, run_dir=tmp_path / "supplement")
    view = build_report_view(comparison, supplemental_comparison=supplement)
    assert all(c["interpretation"] is not None for c in view["candidates"])
    assert all(c["interpretation"]["paths"][0]["reference_geometry"] == geometry for c in view["candidates"])
    assert len(view["analysis_comparisons"]) == 1
    assert len(view["pairs"]) == 3  # Supplemental stored pairs remain visible too.


def test_input_distance_separates_joint_design_marginal_range_and_constant_columns():
    from ml_platform_symbolic.equation_evaluation.backends.interpretation import input_reference_geometry

    x = np.linspace(0, 1, 5)
    reference = np.column_stack([x, x, np.full(len(x), 4)])
    points = np.array([[0.25, 0.25, 4], [0.25, 0.75, 4], [1.5, 1.5, 5]])
    evidence = input_reference_geometry(np.repeat(reference, 3, axis=0), points, ["x", "z", "fixed"])
    assert evidence["outside_marginal_range"] == [False, False, True]
    assert evidence["constant_feature_changed"] == [False, False, True]
    np.testing.assert_allclose(evidence["nearest_distance"], [0, np.sqrt(0.125), np.sqrt(0.5)])
    assert evidence["n_reference_rows"] == 15 and evidence["n_distinct_designs"] == 5
    assert evidence["reference_neighbor_spacing"] == input_reference_geometry(
        reference, points, ["x", "z", "fixed"]
    )["reference_neighbor_spacing"]
    # Independent affine unit conversions do not alter normalized varying-column distances.
    scale, offset = np.array([1000, 0.001, -2]), np.array([273.15, 0, 20])
    converted = input_reference_geometry(reference * scale + offset, points * scale + offset, ["x", "z", "fixed"])
    np.testing.assert_allclose(converted["nearest_distance"], evidence["nearest_distance"])
    constant = input_reference_geometry(np.array([[4], [4]]), np.array([[4], [5]]), ["fixed"])
    assert constant["nearest_distance"] == [0, 0] and constant["constant_feature_changed"] == [False, True]
    assert constant["reference_neighbor_spacing"] is None
    unresolved = input_reference_geometry(np.array([[0.0], [1e-300], [1e300]]), np.array([[0.0]]), ["x"])
    assert unresolved["status"] == "unavailable"


def test_whole_output_and_additive_parts_use_generic_coordinate_roles(synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import load_prepared, prepare, read_json
    from ml_platform_symbolic.equation_evaluation.backends.interpretation import _term_values
    from ml_platform_symbolic.units import ATOMS, ZERO, Unit

    monkeypatch.setitem(ATOMS, "sensor_readout", Unit(ZERO, scale=3.0, offset=-7.0))
    cfg = deepcopy(synthetic_config)
    cfg["data"]["units"] = {"x": "1", "y": "sensor_readout"}
    cfg["data"]["coordinate_roles"] = {"y": "value"}
    refs = prepare(resolve_config(cfg), tmp_path / "prepare")
    context = read_json(refs["run"])
    X, _, _ = load_prepared(refs, context, "selection")
    schema = {k: context[k] for k in ("variables", "target", "unit_coordinates")}
    arrays = {"selection_X": X}
    terms = []
    for expression in ("15*x0-7", "(15*x0-7)*1"):
        equation = parse_expression(expression, context["variables"], trainable=False)
        record = bind_view(dict(id="parts", kind="explicit", equation=equation, target=context["target"]))
        values, _, _ = _term_values(equation, record, context, arrays, schema, np.array([[1.0]]))
        terms.append(values)
    assert len(terms[0]) == 2 and len(terms[1]) == 1
    assert terms[1][0]["reference_value"] == 5.0
    assert sum(v["reference_value"] for v in terms[0]) != 5.0
    assert "whole output coordinate" in terms[1][0]["unit"]
    assert all("additive difference-scale" in v["unit"] for v in terms[0])

    restricted = bind_view(
        dict(id="restricted", kind="explicit", equation=equation, target={**context["target"], "domain": [100, 200]})
    )
    values, _, _ = _term_values(equation, restricted, context, arrays, schema, np.array([[1.0]]))
    assert values[0]["reference_value"] is None
    assert values[0]["n_valid"] == 0  # Finite algebra alone cannot override the saved relation domain.


def test_interpretation_withholds_confirmed_failures_not_pending_parameter_audits(synthetic_config, tmp_path):
    from ml_platform_symbolic.equation_evaluation.backends.interpretation import run

    study = create_study(
        dict(problem=synthetic_config, sources=[{"manual": {"id": "law", "expression": "2*x*x+1"}}]),
        run_dir=tmp_path / "study",
    )
    context, records, rows, arrays = load_study(study)
    original = deepcopy(records)
    options = {"paths": [dict(id="observed", at={"x": 1}, direction={"x": 1}, steps=[0, 0.1], path_unit="1")]}
    baseline = run(context, records, rows, arrays, tmp_path, options)["results"][0]
    assert baseline["outcome"] == "ok" and baseline["terms"] and baseline["paths"]
    for required_failure, native_failure in [(True, False), (False, True)]:
        candidate, evidence = deepcopy(records), deepcopy(rows)
        if required_failure:
            evidence[0].update(requirements_eligible=False, requirement_failures=["units:fail"])
        if native_failure:
            candidate[0]["native_check"]["passed"] = False
        result = run(context, candidate, evidence, arrays, tmp_path, options)["results"][0]
        assert result["outcome"] == "unavailable"
        assert not result["terms"] and not result["paths"] and not result["mathematics"]
        assert "prerequisite failure" in result["reason"]
    # Neither low predictive accuracy nor an unfinished coefficient audit alone
    # invalidates already-supported mathematical/response evidence.
    pending = deepcopy(rows)
    pending[0].update(eligible=False, reason="parameter provenance audit pending")
    result = run(context, records, pending, arrays, tmp_path, options)["results"][0]
    assert result["outcome"] == "ok" and result["paths"] == baseline["paths"]
    assert "identifiability not established" in result["identifiability"]
    assert records == original
