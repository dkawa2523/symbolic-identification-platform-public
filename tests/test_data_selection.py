import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from current_flow import explicit_choice, make_study, prepare_discover, study_flow
from ml_platform_symbolic import finalize_hypothesis, load_model, select_representatives
from ml_platform_symbolic.assets import verify_assets
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, load_prepared, prepare, read_json, write_json
from ml_platform_symbolic.equation_evaluation import load_study
from ml_platform_symbolic.metrics import prediction_metrics


def test_final_targets_do_not_change_selection(synthetic_config, tmp_path):
    _, _, first_study = study_flow(synthetic_config, tmp_path / 'first', purpose='local_prediction', engineering_tolerance=1e-4)
    _, first_decision = explicit_choice(first_study, tmp_path / 'first')
    first = finalize_hypothesis(first_study, first_decision, run_dir=tmp_path / 'first/audit')
    frame = pd.read_csv(synthetic_config['data']['local_path'])
    split = pd.read_csv(synthetic_config['split']['file'])
    frame.loc[split.partition == 'final', 'y'] += 1000
    frame.to_csv(synthetic_config['data']['local_path'], index=False)
    _, _, second_study = study_flow(synthetic_config, tmp_path / 'second', purpose='local_prediction', engineering_tolerance=1e-4)
    _, second_decision = explicit_choice(second_study, tmp_path / 'second')
    second = finalize_hypothesis(second_study, second_decision, run_dir=tmp_path / 'second/audit')
    assert load_model(first.artifacts['model_bundle']).equation == load_model(second.artifacts['model_bundle']).equation
    assert first.extra['selected_id'] == second.extra['selected_id']
    assert second.metrics['final_rmse'] > 900


@pytest.mark.parametrize("kind", ["group_leak", "missing_id", "duplicate_id"])
def test_bad_splits_rejected(synthetic_config, tmp_path, kind):
    cfg = copy.deepcopy(synthetic_config)
    path = cfg["split"]["file"]
    split = pd.read_csv(path)
    if kind == "group_leak":
        split.loc[0, "partition"] = "final"
    elif kind == "missing_id":
        split = split.iloc[1:]
    else:
        split.loc[0, "sample_id"] = split.loc[1, "sample_id"]
    split.to_csv(path, index=False)
    with pytest.raises(ValueError):
        prepare(resolve_config(cfg), tmp_path / "prepare")


def test_invalid_rows_never_removed_and_constant_r2():
    assert prediction_metrics(np.ones(3), np.array([1.0, 1.0, np.nan]))["rmse"] is None
    assert prediction_metrics(np.ones(3), np.ones(3))["r2"] is None


def test_missing_engine_and_changed_partition_are_not_silently_accepted(synthetic_config, tmp_path):
    from scripts.container_stage import compare_selection, summarize_discovery
    root = tmp_path / 'run/calculation'
    refs, _ = prepare_discover(synthetic_config, root)
    provenance = read_json(refs['run'])
    provenance['config']['search'].update(engines=['dictionary', 'linear'], allow_partial_engines=True)
    summarize_discovery(root, {'prepare': refs}, provenance)
    summary = read_json(root / 'discovery_results.json')
    assert any(r['engine'] == 'linear' and r['status'] == 'not_run' for r in summary['results'])
    with pytest.raises(ValueError, match='missing discoveries'):
        compare_selection(root, {'prepare': refs}, provenance)
    assert not (root / 'comparison').exists()
    with open(refs['train'], 'ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_prepared(refs, provenance, 'train')


@pytest.mark.parametrize('stage', ['study', 'finalize'])
def test_cross_prepared_stage_artifacts_are_rejected(synthetic_config, tmp_path, stage):
    refs, discoveries, first = study_flow(synthetic_config, tmp_path / 'first', purpose='local_prediction', engineering_tolerance=1e-4)
    _, decision = explicit_choice(first, tmp_path / 'first')
    frame = pd.read_csv(synthetic_config['data']['local_path'])
    frame['x'] *= 10
    frame.to_csv(synthetic_config['data']['local_path'], index=False)
    other_refs, other_discoveries = prepare_discover(synthetic_config, tmp_path / 'other')
    if stage == 'study':
        with pytest.raises(ValueError, match='prepared run identity mismatch'):
            make_study(other_refs, discoveries, tmp_path / 'cross-study')
    else:
        other = make_study(other_refs, other_discoveries, tmp_path / 'other/study')
        with pytest.raises(ValueError, match='bound to this study'):
            finalize_hypothesis(other, decision, run_dir=tmp_path / 'cross-final')
        assert not (tmp_path / 'cross-final').exists()


def test_discovery_summary_is_bound_to_candidate_file(synthetic_config, tmp_path):
    refs, discoveries = prepare_discover(synthetic_config, tmp_path / 'run')
    candidates = Path(discoveries[0]['candidates'])
    candidates.write_bytes(candidates.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='candidates checksum mismatch'):
        make_study(refs, discoveries, tmp_path / 'study')


def test_asset_manifest_corruption_is_rejected(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    weight = root / "weights.bin"
    weight.write_bytes(b"test-only bytes, not a native checkpoint")
    manifest = write_json(
        tmp_path / "manifest.json",
        {
            "engine": "test_engine",
            "source": "unit-test fixture",
            "source_revision": "fixture-v1",
            "directory": "model",
            "files": [{"path": "weights.bin", "size_bytes": weight.stat().st_size, "sha256": digest(weight)}],
        },
    )
    assert verify_assets(manifest, "test_engine")[0] == root
    weight.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum/size"):
        verify_assets(manifest, "test_engine")


def test_explicit_runtime_keeps_old_prepared_runs_usable(synthetic_config, tmp_path):
    import sys

    from ml_platform_symbolic.engines.runner import run_discovery

    cfg = resolve_config(synthetic_config)
    cfg["search"]["runtimes"] = {"dictionary": {"python": sys.executable, "device": "cpu"}}
    refs = prepare(cfg, tmp_path / "prepare")
    provenance = read_json(refs["run"])
    explicit = run_discovery(refs, provenance, "dictionary", 42, tmp_path / "explicit")
    assert read_json(explicit["summary"])["python_executable"] == sys.executable
    provenance["config"]["search"].pop("runtimes")
    provenance["config"].pop("model_assets")
    legacy = run_discovery(refs, provenance, "dictionary", 42, tmp_path / "legacy")
    assert read_json(legacy["summary"])["status"] == "ok"


def test_native_relation_does_not_count_as_scalar_comparison(synthetic_config, tmp_path):
    refs, discoveries = prepare_discover(synthetic_config, tmp_path / 'run')
    summary = read_json(discoveries[0]['summary'])
    summary['portable_status'] = 'implicit_relations'
    write_json(discoveries[0]['summary'], summary)
    study = make_study(refs, discoveries, tmp_path / 'study')
    _, records, rows, _ = load_study(study)
    assert len(records) == 1 and records[0]['kind'] == 'native_result'
    assert not rows[0]['eligible']
    assert select_representatives(records, rows) == {}
