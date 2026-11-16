"""Candidate allocation keeps full archives and never hides constraint-valid choices."""

from copy import deepcopy

import pytest
from current_flow import prepare_discover, make_study
from ml_platform_symbolic.equation_evaluation import load_study
from pathlib import Path
from ml_platform_symbolic.candidate_pool import allocate_candidates, structure_key
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.expressions import parse_expression


def candidate(name, text):
    return {"candidate_id": name, "equation": parse_expression(text, [{"id": "x0", "unit": "1"}], trainable=False)}


def test_emitted_archive_allocation_is_explicit_and_diverse():
    records = [
        candidate("best", "x0*x0+1"),
        candidate("simple", "x0"),
        candidate("other", "sin(x0)"),
        candidate("bad", "x0/0"),
    ]
    qualities = {
        name: {"train_checks_pass": name != "bad", "train_rmse": error, "nodes": nodes, "free_parameters": 0}
        for name, error, nodes in [("best", 0.01, 5), ("simple", 0.1, 1), ("other", 0.1, 2), ("bad", 0, 3)]
    }
    original = deepcopy(records)
    assert allocate_candidates(records, {}, 1, "all") == records
    selected = allocate_candidates(records, qualities, 2, "pareto_diverse")
    assert [r["candidate_id"] for r in selected] == ["best", "simple"]
    assert records == original


def test_pipeline_default_does_not_hide_candidates_behind_old_train_cap(synthetic_config, tmp_path):
    cfg = deepcopy(synthetic_config)
    cfg["search"].update(engines=["dictionary"], max_export_candidates=1)
    refs, discoveries = prepare_discover(cfg, tmp_path / "all_candidates")
    discovery = discoveries[0]
    summary = read_json(discovery["summary"])
    assert summary["candidate_pool"]["policy"] == "all"
    assert summary["selection_candidates"] == summary["exported_candidates"] > 1
    assert Path(discovery["candidates"]) == Path(discovery["all_candidates"])
    assert len(Path(discovery["candidates"]).read_text().splitlines()) == summary["exported_candidates"]


def test_scaffold_does_not_erase_fixed_literals_or_parameter_ties():
    variables = [{"id": "x0", "unit": "1"}]
    parameters = {
        "a": {"value": 2.0, "trainable": True, "unit": "1"},
        "b": {"value": 2.0, "trainable": False, "tied_to": "a", "unit": "1"},
    }
    tied = parse_expression("a*x0+b", variables, parameters=parameters)
    same = deepcopy(tied)
    same["parameters"]["a"]["value"] = same["parameters"]["b"]["value"] = 4.0
    assert structure_key(tied) == structure_key(same)
    independent = deepcopy(tied)
    independent["parameters"]["b"].pop("tied_to")
    independent["parameters"]["b"]["trainable"] = True
    assert structure_key(tied) != structure_key(independent)
    assert structure_key(candidate("a", "x0+1")["equation"]) != structure_key(candidate("b", "x0+2")["equation"])


def test_candidate_cap_does_not_discard_every_constraint_valid_formula(synthetic_config, tmp_path):
    cfg = deepcopy(synthetic_config)
    cfg["search"].update(
        max_export_candidates=1, candidate_pool_policy="pareto_diverse", engine_options={"dictionary": {"max_terms": 3}}
    )
    cfg["constraints"]["max_nodes"] = 5
    # Better training fits use more terms and violate the already declared node
    # limit. A one-term candidate remains usable; allocation must not hide it.
    refs, discoveries = prepare_discover(cfg, tmp_path / "capped")
    discovery = discoveries[0]
    summary = read_json(discovery["summary"])
    assert summary["prelimit_constraint_eligible"] > 0
    assert summary["selection_candidates"] == 1
    study = make_study(refs, discoveries, tmp_path / 'study')
    context, records, rows, _ = load_study(study)
    assert any(row['eligible'] and row['metrics']['selection']['rmse'] is not None for row in rows)
    assert records and not context['final_exposure']['same_data_exposed']
    from ml_platform_symbolic import score_representatives
    audit_dir = score_representatives(study, run_dir=tmp_path / 'explicit-final-score')
    audit = read_json(audit_dir / 'audit.json')
    assert audit['results'] and audit['results'][0]['target_metrics']['rmse'] is not None
    assert all(check['status'] == 'pass' for row in rows if row['eligible'] for check in row['checks'] if check['required'])


def test_retired_train_error_pool_policy_is_rejected(synthetic_config):
    synthetic_config["search"]["candidate_pool_policy"] = "train_error"
    with pytest.raises(ValueError, match="candidate_pool_policy"):
        resolve_config(synthetic_config)
