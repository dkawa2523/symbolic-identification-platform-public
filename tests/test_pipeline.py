import copy
from pathlib import Path

import pandas as pd
import pytest
from current_flow import explicit_choice, study_flow
from ml_platform_spdml.symbolic import report_symbolic_result
from ml_platform_symbolic import finalize_hypothesis, load_model, run_task, select_representatives
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.equation_evaluation import load_study


def test_run_reload_infer_and_transport(synthetic_config, tmp_path):
    from unittest.mock import Mock

    import numpy as np
    from ml_platform_core.result import RunResult
    from ml_platform_symbolic.data import write_json
    refs, discoveries, study = study_flow(synthetic_config, tmp_path / 'run', purpose='local_prediction', engineering_tolerance=1e-4)
    assert Path(discoveries[0]['summary']).is_file()
    _, decision = explicit_choice(study, tmp_path / 'run')
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / 'run/audit')
    assert result.metrics['final_rmse'] < 1e-5
    frame = pd.read_csv(synthetic_config['data']['local_path'])
    model = load_model(result.artifacts['model_bundle'])
    assert model.metadata['origin']['summary'] == read_json(discoveries[0]['summary'])
    _, records, rows, _ = load_study(study)
    assert all(row['n_nodes'] > 0 for row in rows if row['eligible'])
    prediction = model.predict(frame)
    inferred = run_task({'task': 'symbolic_infer', 'data': {'local_path': synthetic_config['data']['local_path']},
                         'model': {'local_model_path': str(result.artifacts['model_bundle'])}}, run_dir=tmp_path / 'infer')
    np.testing.assert_allclose(pd.read_csv(inferred.tables['predictions']).prediction, prediction)
    with pytest.raises(ValueError, match='empty'):
        run_task({'task': 'symbolic_stage', 'run': {'stage': 'prepare'}, **synthetic_config}, run_dir=tmp_path / 'run')
    # A read-only legacy transport contract fixture, not a legacy numerical writer.
    info = write_json(tmp_path / 'historical-model-info.json', {'provenance': read_json(refs['run']),
        'selected_candidate_id': result.extra['selected_id'], 'equation': 'test fixture display'})
    transport = RunResult(result.run_dir, result.metrics, {**result.artifacts, 'model_info': info}, result.tables)
    task, factory = Mock(), Mock()
    report_symbolic_result(task, transport, register_model=True, model_factory=factory)
    assert task.upload_artifact.call_count == len(transport.artifacts) + len(transport.tables)
    assert factory.call_args.kwargs['framework'] == 'Custom'
    assert factory.return_value.update_weights.call_args.kwargs['weights_filename'] == str(result.artifacts['model_bundle'])
    assert task.get_logger.return_value.report_scalar.call_count == len(result.metrics)


def test_all_invalid_is_failure_with_inspectable_leaderboard(synthetic_config, tmp_path):
    cfg = copy.deepcopy(synthetic_config)
    cfg['constraints'] = {'max_nodes': 1, 'require_units': True}
    cfg['data']['units']['y'] = 's'
    _, _, study = study_flow(cfg, tmp_path / 'invalid')
    _, records, rows, _ = load_study(study)
    assert records and rows and not any(row['eligible'] for row in rows)
    assert select_representatives(records, rows) == {}
    assert (study / 'numerical.json').is_file()
    assert not (tmp_path / 'invalid/final_access.json').exists()


def test_explicit_partial_engine_failure(synthetic_config, tmp_path):
    cfg = copy.deepcopy(synthetic_config)
    cfg['search'].update(engines=['dictionary', 'physo'], engine_options={'physo': {'not_an_option': True}}, allow_partial_engines=True)
    _, discoveries, study = study_flow(cfg, tmp_path / 'partial')
    assert read_json(discoveries[1]['summary'])['status'] == 'failed'
    _, records, rows, _ = load_study(study)
    failed = [r for r in records if r['origin']['engine'] == 'physo']
    assert failed and failed[0]['kind'] == 'native_result'
    assert not next(row for row in rows if row['id'] == failed[0]['id'])['eligible']


def test_installed_package_preparation_does_not_assume_checkout_depth(synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic import data
    from ml_platform_symbolic.config import resolve_config

    monkeypatch.setattr(data, "__file__", "/installed/data.py")
    monkeypatch.setattr(data, "package_identity", lambda: {"version": "0.90.2", "mode": "installed"})
    def forbidden_git(*args, **kwargs):
        raise AssertionError("Installed package must not infer revision from the consumer checkout")
    monkeypatch.setattr(data.subprocess, "run", forbidden_git)
    refs = data.prepare(resolve_config(synthetic_config), tmp_path / "prepared")
    assert Path(refs["train"]).is_file()


@pytest.mark.parametrize("setting,value", [("max_wall_seconds", 1), ("engine_options", {"dictionary": {"degree": 1}})])
def test_discovery_rejects_ignored_explicit_search_overrides(synthetic_config, tmp_path, monkeypatch, setting, value):
    from ml_platform_symbolic import stage

    cfg = copy.deepcopy(synthetic_config)
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42}, stage_inputs={"prepare": prepared.extra})
    cfg["search"][setting] = value
    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched settings must be rejected before a worker starts")
    monkeypatch.setattr(stage, "run_discovery", forbidden)
    with pytest.raises(ValueError, match="No worker started"):
        run_task(cfg, run_dir=tmp_path / "discover")


def test_discovery_setting_check_preserves_inheritance_and_runtime_defaults(synthetic_config, tmp_path):
    from ml_platform_symbolic import stage
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import read_json

    cfg = copy.deepcopy(synthetic_config)
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    provenance = read_json(prepared.artifacts["run"])
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42}, stage_inputs={"prepare": prepared.extra})
    cfg["search"]["runtimes"] = {"dictionary": {"device": "cpu"}}
    cfg["evaluation"]["absolute_tolerance"] = 0.01
    stage._check_discovery_settings(resolve_config(cfg), provenance, cfg)
    # A stage call that omits search intentionally inherits the frozen settings.
    stage._check_discovery_settings(resolve_config(cfg), provenance, {"run": cfg["run"]})


def test_discovery_rejects_requirements_operator_override(synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic import stage
    cfg = copy.deepcopy(synthetic_config)
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42}, stage_inputs={"prepare": prepared.extra})
    cfg["search"].pop("operators", None)
    cfg["requirements"] = {"allowed_operators": ["add"]}
    def forbidden(*args, **kwargs):
        raise AssertionError("Requirements override must fail before worker starts")
    monkeypatch.setattr(stage, "run_discovery", forbidden)
    with pytest.raises(ValueError, match="search.operators"):
        run_task(cfg, run_dir=tmp_path / "discover")


@pytest.mark.parametrize("override", [
    {"constraints": {"max_free_parameters": 1}},
    {"data": {"local_path": "different.csv"}},
    {"data": {"domain": {"x": [0., 5.]}}},
    {"data": {"units": {"x": "m", "y": "m"}}},
    {"data": {"target_column": "other", "units": {"x": "1", "other": "1"}}},
    {"data": {"feature_columns": ["other"], "units": {"other": "1", "y": "1"}}},
    {"model_assets": {"dictionary": {"manifest_file": "different.json"}}},
    {"requirements": {"equation_template": {
        "coordinate_space": "working", "ast": {"slot": "f", "args": [{"var": "x0"}]},
        "slots": {"f": {"arguments": [{"name": "u", "unit": "1"}], "unit": "1",
                          "allowed_operators": ["add"], "max_nodes": 3}}, "parameters": {},
        "knowledge": {"kind": "requirement", "statement": "Shared unknown function"}}}},
    {"requirements": {"allowed_variables": ["x0"]}},
    {"requirements": {"required_responses": {"points": [{"x": 1.}], "bounds": [0., None]}}},
    {"requirements": {"fixed_constants": {"g": {"value": 9.8, "unit": "1"}}}},
])
def test_discovery_rejects_changed_problem_before_worker(synthetic_config, tmp_path, monkeypatch, override):
    from ml_platform_symbolic import stage
    cfg = copy.deepcopy(synthetic_config)
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42},
               stage_inputs={"prepare": prepared.extra})
    for section, values in override.items():
        cfg.setdefault(section, {}).update(values)
    def forbidden(*args, **kwargs):
        raise AssertionError("Changed problem reached the old prepared worker")
    monkeypatch.setattr(stage, "run_discovery", forbidden)
    with pytest.raises(ValueError):
        run_task(cfg, run_dir=tmp_path / "discover")


def test_discovery_problem_aliases_and_omissions_preserve_prepared_meaning(synthetic_config, tmp_path):
    from ml_platform_symbolic import stage
    from ml_platform_symbolic.config import resolve_config
    cfg = copy.deepcopy(synthetic_config)
    cfg["constraints"]["allowed_variables"] = ["x"]
    cfg["data"]["fixed_constants"] = {"g": {"value": 1, "unit": "1"}}
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    provenance = read_json(prepared.extra["run"])
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42})
    cfg["constraints"].pop("allowed_variables")
    cfg["data"].pop("fixed_constants")
    cfg["requirements"] = {"allowed_variables": ["x0"], "fixed_constants": {"g": {"value": 1., "unit": "1"}}}
    cfg["evaluation"]["absolute_tolerance"] = .02
    stage._check_discovery_settings(resolve_config(cfg), provenance, cfg)
    # A stage can inherit every frozen requirement without restating them.
    stage._check_discovery_settings(resolve_config(cfg), provenance, {"run": cfg["run"]})


def test_discovery_same_requirement_alias_executes_normal_path(synthetic_config, tmp_path):
    cfg = copy.deepcopy(synthetic_config)
    cfg["constraints"]["allowed_variables"] = ["x"]
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    cfg["constraints"].pop("allowed_variables")
    cfg["requirements"] = {"allowed_variables": ["x0"]}
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42},
               stage_inputs={"prepare": prepared.extra})
    result = run_task(cfg, run_dir=tmp_path / "discover")
    summary = read_json(result.artifacts["engine_summary"])
    assert summary["status"] == "ok"
    assert Path(result.artifacts["candidates"]).is_file()



@pytest.mark.parametrize("partial", [None, {"search": {"max_wall_seconds": 20}}])
def test_minimal_public_discover_inherits_frozen_problem(synthetic_config, tmp_path, partial):
    cfg = copy.deepcopy(synthetic_config)
    cfg["data"]["units"] = {"x": "cm", "y": "m"}
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    result = run_task({"task": "symbolic_stage", "run": {"stage": "discover", "engine": "dictionary", "seed": 42},
                      "stage_inputs": {"prepare": prepared.extra}, **(partial or {})}, run_dir=tmp_path / "discover")
    assert read_json(result.artifacts["engine_summary"])["status"] == "ok"


@pytest.mark.parametrize("partial", [
    {"data": {"units": {"x": "1"}}}, {"data": {"domain": {"x": [0., 4.]}}},
    {"requirements": {"fixed_constants": {"g": {"unit": "cm"}}}},
    {"observation_model": {"target_meaning": "observed_response"}},
])
def test_public_discover_rejects_ambiguous_partial_problem(synthetic_config, tmp_path, monkeypatch, partial):
    from ml_platform_symbolic import stage
    cfg = copy.deepcopy(synthetic_config)
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    monkeypatch.setattr(stage, "run_discovery", lambda *a, **k: pytest.fail("partial problem reached worker"))
    with pytest.raises(ValueError, match="redeclaration must be complete"):
        run_task({"task": "symbolic_stage", "run": {"stage": "discover", "engine": "dictionary", "seed": 42},
                  "stage_inputs": {"prepare": prepared.extra}, **partial}, run_dir=tmp_path / "discover")


@pytest.mark.parametrize("change", ["source_units", "constant_unit", "response_bounds"])
def test_complete_problem_checks_derived_numeric_meaning(synthetic_config, tmp_path, monkeypatch, change):
    from ml_platform_symbolic import stage
    cfg = copy.deepcopy(synthetic_config)
    cfg["data"]["units"] = {"x": "cm", "y": "m"}
    cfg["data"]["fixed_constants"] = {"g": {"value": 1., "unit": "m"}}
    cfg["constraints"].update(probes=[{"x": 1.}], response_bounds=[0., 30.])
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42}, stage_inputs={"prepare": prepared.extra})
    if change == "source_units":
        cfg["data"]["units"]["x"] = "m"
    elif change == "constant_unit":
        cfg["data"]["fixed_constants"]["g"]["unit"] = "cm"
    else:
        cfg["constraints"]["response_bounds"] = [1., 30.]
    monkeypatch.setattr(stage, "run_discovery", lambda *a, **k: pytest.fail("changed meaning reached worker"))
    with pytest.raises(ValueError, match="No worker started"):
        run_task(cfg, run_dir=tmp_path / "discover")


def test_complete_sampling_redeclaration_matches_frozen_points(synthetic_config, tmp_path):
    cfg = copy.deepcopy(synthetic_config)
    cfg["constraints"]["probe_sampling"] = {"domain": {"x": [0.5, 3.]}, "points": 4, "corners": 2, "seed": 17}
    cfg["search"]["response_mode"] = "filter"
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    cfg.update(run={"stage": "discover", "engine": "dictionary", "seed": 42}, stage_inputs={"prepare": prepared.extra})
    result = run_task(cfg, run_dir=tmp_path / "discover")
    assert read_json(result.artifacts["engine_summary"])["status"] == "ok"


@pytest.mark.parametrize("form", ["original", "prepared"])
def test_full_source_and_prepared_documents_keep_same_semantics(synthetic_config, tmp_path, form):
    cfg = copy.deepcopy(synthetic_config)
    cfg["data"]["units"] = {"x": "cm", "y": "m"}
    cfg["data"]["domain"] = {"x": [0., 4.], "y": [0., 30.]}
    cfg.update(task="symbolic_stage", run={"stage": "prepare"})
    prepared = run_task(cfg, run_dir=tmp_path / "prepare")
    if form == "prepared":
        call = read_json(prepared.extra["run"])
        call["config"].update(run={"stage": "discover", "engine": "dictionary", "seed": 42},
                              stage_inputs={"prepare": prepared.extra})
    else:
        call = cfg
        call.update(run={"stage": "discover", "engine": "dictionary", "seed": 42},
                    stage_inputs={"prepare": prepared.extra})
    result = run_task(call, run_dir=tmp_path / "discover")
    assert read_json(result.artifacts["engine_summary"])["status"] == "ok"


@pytest.mark.parametrize('task,stage', [('symbolic_pipeline', None), ('symbolic_stage', 'compare'), ('symbolic_stage', 'finalize')])
def test_removed_execution_rejected_before_output(synthetic_config, tmp_path, task, stage):
    from ml_platform_symbolic.config import resolve_config

    cfg = dict(synthetic_config, task=task, run={'stage': stage})
    resolve_config(cfg)  # Old configuration documents remain readable.
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='removed|Only prepare/discover'):
        run_task(cfg, run_dir=output)
    assert not output.exists()

