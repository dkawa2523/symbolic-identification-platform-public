"""Finite response evidence and coefficient regularization must retain scientific meaning."""

from copy import deepcopy

import pytest
from current_flow import prepare_discover
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.constraints import check_constraints
from ml_platform_symbolic.data import prepare, read_json
from ml_platform_symbolic.equation_comparison import compare_equations
from ml_platform_symbolic.equation_evaluation.api import create_study, load_study
from ml_platform_symbolic.expressions import parse_expression
from ml_platform_symbolic.model import load_model
from ml_platform_symbolic.selection import choose_hypothesis, finalize_hypothesis

VARIABLES = [{"id": "x0", "column": "temperature", "unit": "1", "domain": [1, 5]}]


def checks(text, request, parameters=None):
    equation = parse_expression(text, VARIABLES, trainable=False, parameters=parameters)
    return check_constraints(
        equation,
        VARIABLES,
        {"unit": "1"},
        ["mul", "add", "sub", "pow_int", "div"],
        {"max_nodes": 50, "require_units": False, **request},
    )


def question(order=1):
    return {
        "feature": "temperature",
        "order": order,
        "sign": 1,
        "step": 0.1,
        "tolerance": 1e-10,
        "points": [{"temperature": 2.0}, {"temperature": 3.0}],
        "knowledge": "declared synthetic test property",
    }


def test_direction_and_curvature_are_distinct_finite_step_questions():
    first = checks("-x0**2", {"response_checks": [question()]})[-1]
    second = checks("x0**2", {"response_checks": [question(2)]})[-1]
    assert first["status"] == "fail" and first["max_violation"] > 5.9
    assert first["worst_point"] == {"temperature": 3.0}
    assert second["status"] == "pass" and second["n_valid"] == 2
    assert second["order"] == 2 and "no global guarantee" in second["method"]


def test_undefined_response_row_cannot_be_dropped():
    request = question()
    request["points"] = [{"temperature": 2.0}, {"temperature": 3.0}]
    result = checks("1/(x0-2)", {"response_checks": [request]})[-1]
    assert result["status"] == "fail" and result["n_valid"] == 1
    assert result["max_violation"] is None and result["worst_response"] is None


def test_parameter_cap_counts_independent_used_coordinates_not_literals_or_aliases():
    params = {
        "a": {"value": 2.0, "unit": "1", "trainable": True, "bounds": [None, None]},
        "b": {"value": 2.0, "unit": "1", "trainable": False, "tied_to": "a", "bounds": [None, None]},
        "fixed": {"value": 4.0, "unit": "1", "trainable": False, "bounds": [4.0, 4.0]},
        "unused": {"value": 9.0, "unit": "1", "trainable": True, "bounds": [None, None]},
    }
    result = checks("a*x0+b+fixed+x0**2", {"max_free_parameters": 1}, params)
    cap = next(c for c in result if c["name"] == "max_free_parameters")
    assert cap["status"] == "pass" and cap["count"] == 1


@pytest.mark.parametrize("mutation", ["outside", "unknown_column", "zero_step"])
def test_invalid_response_requests_fail_before_discovery(synthetic_config, mutation):
    cfg = deepcopy(synthetic_config)
    cfg["data"]["domain"] = {"x": [0.1, 10]}
    request = {**question(), "feature": "x", "points": [{"x": 1.0}]}
    if mutation == "outside":
        request["points"] = [{"x": 0.1}]
    elif mutation == "unknown_column":
        request["feature"] = "not_a_column"
    else:
        request["step"] = 0
    cfg["constraints"]["response_checks"] = [request]
    with pytest.raises(ValueError):
        resolve_config(cfg)


def test_independent_postcheck_studies_preserve_one_source_and_never_open_final(synthetic_config, tmp_path):
    cfg = resolve_config(synthetic_config)
    refs = prepare(cfg, tmp_path / "source")
    source_bytes = (tmp_path / "source/run.json").read_bytes()
    plan = {
        "prepared_run": refs["run"],
        "sources": [{"manual": {"id": "decreasing", "expression": "0-x"}}],
    }
    base = create_study(plan, run_dir=tmp_path / "base")
    ctx, _, base_rows, _ = load_study(base)
    assert base_rows[0]["eligible"]
    request = {**question(), "feature": "x", "points": [{"x": 1.0}]}
    constrained = create_study(
        {**plan, "additional_constraints": {"response_checks": [request]}}, run_dir=tmp_path / "shape"
    )
    added, _, rows, _ = load_study(constrained)
    assert not rows[0]["eligible"]
    origin = rows[0]["response_evaluation"]
    assert origin[0]["status"] == "not_evaluated"
    assert origin[1]["point_origin"] == "additional_study"
    assert origin[1]["violation_rate"] == 1.0
    assert origin[1]["n_points"] == 1 and origin[1]["n_invalid"] == 0
    assert origin[1]["max_violation"] > .99
    assert "Selection" in origin[1]["data_role"]
    from ml_platform_symbolic.reporting import export_outputs
    import csv
    comparison = compare_equations(constrained, run_dir=tmp_path / "comparison")
    export_outputs(comparison, output_dir=tmp_path / "output", figures=False)
    with (tmp_path / "output/tables/response_checks.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    assert saved[1]["violation_rate"] == "1.0"
    assert saved[1]["point_origin"] == "additional_study"
    assert saved[1]["response_unit"] == origin[1]["response_unit"]
    assert saved[1]["feature"] == "x"
    assert "direction" in saved[1] and "path_unit" in saved[1]
    assert added["prepared_sha256"] == ctx["prepared_sha256"]
    assert "postcheck only" in added["constraint_application"]["kind"]
    assert (tmp_path / "source/run.json").read_bytes() == source_bytes
    assert not (tmp_path / "source/final_access.json").exists()
    source_problem = read_json(refs["run"])
    assert source_problem["requirements"]["required_responses"]["checks"] == []
    assert "constraints" not in source_problem["config"]


def test_added_study_constraints_cannot_weaken_source(synthetic_config, tmp_path):
    cfg = resolve_config(synthetic_config)
    refs = prepare(cfg, tmp_path / "source")
    with pytest.raises(ValueError, match="cannot increase"):
        create_study(
            {
                "prepared_run": refs["run"],
                "sources": [{"manual": {"id": "x", "expression": "x"}}],
                "additional_constraints": {"max_nodes": 60},
            },
            run_dir=tmp_path / "bad",
        )


def test_export_retains_effective_postcheck_requirements(synthetic_config, tmp_path):
    request = {**question(), "feature": "x", "points": [{"x": 1.0}]}
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [{"manual": {"id": "x", "expression": "x"}}],
            "engineering_tolerance": 1e6,
            "additional_constraints": {"max_free_parameters": 0, "response_checks": [request]},
        },
        run_dir=tmp_path / "study",
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    choice = choose_hypothesis(
        comparison, run_dir=tmp_path / "choice", candidate_id="x@manual", reason="synthetic export regression"
    )
    result = finalize_hypothesis(study, choice, run_dir=tmp_path / "finalize")
    model = load_model(result.artifacts["model_bundle"])
    from ml_platform_symbolic.problem import problem_config

    config = problem_config(model.metadata["provenance"])
    assert config["constraints"]["response_checks"] == [request]
    assert config["constraints"]["max_free_parameters"] == 0
    assert "postcheck" in model.metadata["provenance"]["evaluation_constraint_application"]["kind"]


def test_explicit_native_context_allows_method_settings_not_information_changes(synthetic_config, tmp_path):
    refs, discoveries = prepare_discover(synthetic_config, tmp_path / "native")
    source = {
        "discovery": str(discoveries[0]["summary"]),
        "prepared_run": str(refs["run"]),
    }
    other = deepcopy(synthetic_config)
    other["search"]["engines"] = ["linear"]
    study = create_study({"problem": other, "sources": [source]}, run_dir=tmp_path / "study")
    context, records, _, _ = load_study(study)
    assert records and not context["final_exposure"]["same_data_exposed"]
    # Same numeric CSV is insufficient when the physical meaning changes.
    other["data"]["units"]["x"] = "s"
    with pytest.raises(ValueError, match="information differs"):
        create_study({"problem": other, "sources": [source]}, run_dir=tmp_path / "different_units")

def test_named_direction_is_preserved_in_normal_csv(synthetic_config, tmp_path):
    import csv, json
    from ml_platform_symbolic.reporting import export_outputs
    request = {'name':'arbitrary-condition','direction':{'x':-1.0},'path_unit':'1',
               'order':1,'sign':1,'step':.1,'tolerance':0,'points':[{'x':1.0}]}
    study = create_study({'problem':synthetic_config,
        'sources':[{'manual':{'id':'direction','expression':'x'}}],
        'additional_constraints':{'response_checks':[request]}},run_dir=tmp_path/'study')
    comparison=compare_equations(study,run_dir=tmp_path/'comparison')
    export_outputs(comparison,output_dir=tmp_path/'output',figures=False)
    with (tmp_path/'output/tables/response_checks.csv').open() as handle:
        row=next(r for r in csv.DictReader(handle) if r['name']=='arbitrary-condition')
    assert json.loads(row['direction'])=={'x':-1.0}
    assert json.loads(row['working_direction'])=={'x':-1.0}
    assert row['path_unit']=='1' and row['feature']=='' and row['status']=='fail'
