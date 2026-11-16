"""Regressions for observed export, status and failed-execution provenance defects."""

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, prepare, read_json, write_json
from ml_platform_symbolic.engines.base import DiscoveryRequest, checked_candidate

from scripts.container_stage import summarize_discovery


def test_nested_reciprocal_boundary_is_not_accepted_from_train_only_check():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    equation = {
        "schema_version": 1,
        "ast": {"op": "tanh", "args": [{"op": "div", "args": [{"const": 1}, {"var": "x0"}]}]},
        "parameters": {},
    }
    request = DiscoveryRequest(
        np.array([[1.0], [2.0], [3.0]]), np.ones(3), np.arange(3), variables, {"unit": "1"}, ["div", "tanh"], {}, 42, {}
    )
    with pytest.raises(ValueError, match="IEEE"):
        checked_candidate("flash_ansr", 0, equation, request, lambda X: np.tanh(1 / X[:, 0]))


@pytest.mark.parametrize("status", ["timeout", "ok"])
def test_all_failed_and_successful_ineligible_have_distinct_saved_status(synthetic_config, tmp_path, status):
    cfg = resolve_config(synthetic_config)
    cfg["search"]["allow_partial_engines"] = True
    cfg["constraints"]["allowed_variables"] = []
    root = tmp_path / "case"
    refs = prepare(cfg, root / "prepare")
    directory = root / "discover/dictionary/42"
    directory.mkdir(parents=True)
    candidates = directory / "candidates.jsonl"
    # Transport fixture only; this test does not claim a native search/checkpoint run.
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "dictionary/42/0",
                "engine": "dictionary",
                "seed": 42,
                "equation": {"schema_version": 1, "ast": {"var": "x0"}, "parameters": {}},
                "native_check": {"passed": True, "max_abs_error": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        directory / "engine_summary.json",
        {
            "engine": "dictionary",
            "seed": 42,
            "status": status,
            "reason": "Hard wall limit" if status == "timeout" else None,
            "wall_seconds": 7,
            "exported_candidates": 1 if status == "ok" else 0,
            "prepared_sha256": digest(refs["run"]),
            "candidates_sha256": digest(candidates),
        },
    )
    summarize_discovery(root, {'prepare': refs}, read_json(refs['run']))
    table = pd.read_csv(root / 'discovery_results.csv').set_index('engine')
    assert table.loc['dictionary', 'status'] == status
    assert table.loc['dictionary', 'wall_seconds'] == 7
    assert not (root / 'finalize').exists()
    # The standard readout also retains failures rather than silently losing the method.
    from ml_platform_symbolic import compare_equations, create_study, export_outputs

    study = create_study(
        {"prepared_run": refs["run"], "sources": [{"discovery": str(directory / "engine_summary.json")}]},
        run_dir=root / "study",
    )
    comparison = compare_equations(study, run_dir=root / "comparison")
    output = export_outputs(comparison, output_dir=root / "output", figures=False)
    reported = pd.read_csv(output / "tables/candidates.csv")
    assert len(reported) == 1
    assert reported.loc[0, "run_status"] == status
    assert reported.loc[0, "run_wall_seconds"] == 7
    assert not reported.loc[0, "eligible"]
    if status == "timeout":
        assert pd.isna(reported.loc[0, "rmse"])
        assert reported.loc[0, "kind"] == "native_result"



def test_launcher_saves_image_and_terminal_status_on_prepare_failure(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "containers").mkdir()
    original = Path(__file__).resolve().parents[1] / "scripts/run_linux.py"
    launcher = project / "scripts/run_linux.py"
    shutil.copyfile(original, launcher)
    (project / "containers/profiles.json").write_text(
        json.dumps({"core": {"image": "test-only", "gpu": False, "engines": ["dictionary"]}})
    )
    (project / "input.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [str(launcher), "--config", "input.json", "--run-dir", "results/failed/calculation"])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "sha256:test-fixture\n")

    def fail(arguments, **kwargs):
        raise subprocess.CalledProcessError(42, arguments)

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runpy.run_path(str(launcher), run_name="__main__")
    record = read_json(project / "results/failed/calculation/container_execution.json")
    assert record["status"] == "failed" and record["phase"] == "prepare"
    assert record["images"] == {"core": "sha256:test-fixture"}
    assert record["finished_at"] and "42" in record["error"]


def test_launcher_wait_is_interruptible_and_reaps_only_owned_client(monkeypatch):
    from scripts import run_linux

    class Client:
        def __init__(self):
            self.timeouts = []
            self.terminated = False

        def wait(self, timeout):
            self.timeouts.append(timeout)
            if self.terminated:
                return 143
            if len(self.timeouts) == 1:
                raise subprocess.TimeoutExpired("docker", timeout)
            raise KeyboardInterrupt

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    client = Client()
    monkeypatch.setattr(subprocess, "Popen", lambda args: client)
    with pytest.raises(KeyboardInterrupt):
        run_linux.checked(["docker", "run", "fixture"])
    assert client.timeouts == [0.2, 0.2, 10] and client.terminated


@pytest.mark.parametrize("present_code,present_output,warning", [(0, "", False), (0, "a" * 64, True), (1, "", True)])
def test_launcher_distinguishes_absent_container_from_unknown_cleanup(tmp_path, monkeypatch, present_code, present_output, warning):
    from scripts import run_linux

    project = tmp_path / "project"
    (project / "containers").mkdir(parents=True)
    (project / "containers/profiles.json").write_text(json.dumps({"core": {"image": "fixture", "gpu": False}}))
    monkeypatch.setattr(run_linux, "__file__", str(project / "scripts/run_linux.py"))
    monkeypatch.setattr(sys, "argv", ["run_linux.py", "--", "false"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "sha256:fixture\n")

    def failed_command(args):
        Path(args[args.index("--cidfile") + 1]).write_text("a" * 64)
        raise subprocess.CalledProcessError(7, args)

    def docker_status(args, **kwargs):
        if args[1] == "stop":
            return subprocess.CompletedProcess(args, 1, "", "No such container")
        assert args[1:3] == ["container", "ls"]
        return subprocess.CompletedProcess(args, present_code, present_output, "daemon unavailable" if present_code else "")

    monkeypatch.setattr(run_linux, "checked", failed_command)
    monkeypatch.setattr(subprocess, "run", docker_status)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_linux.main()
    assert caught.value.returncode == 7
    notes = getattr(caught.value, "__notes__", [])
    assert bool(notes) is warning
    assert all("execution record" not in note for note in notes)


def test_container_selection_comparison_exports_without_final(synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic.engines.runner import run_discovery

    from scripts import container_stage
    cfg = resolve_config(synthetic_config)
    root = tmp_path / 'result' / 'calculation'
    refs = prepare(cfg, root / 'prepare')
    provenance = read_json(refs['run'])
    for engine in cfg['search']['engines']:
        for seed in cfg['search']['seeds']:
            run_discovery(refs, provenance, engine, seed, root / 'discover' / engine / str(seed))
    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison must never finalize')
    from ml_platform_symbolic import data, selection
    monkeypatch.setattr(selection, 'finalize_hypothesis', forbidden)
    load_prepared = data.load_prepared
    def no_final(refs, provenance, partition):
        if partition == 'final':
            raise AssertionError('Selection comparison must never load Final')
        return load_prepared(refs, provenance, partition)
    monkeypatch.setattr(data, 'load_prepared', no_final)
    output = container_stage.compare_selection(root, {'prepare': refs}, provenance)
    summary = read_json(root / 'comparison_summary.json')
    assert summary['final_accessed'] is False
    assert summary['adoption_performed'] is False
    assert summary['representative_ids']
    from ml_platform_symbolic.equation_evaluation import load_study
    from ml_platform_symbolic.figure_output import build_plot_data
    _, records, _, _ = load_study(root / 'study')
    table = pd.read_csv(output / 'tables' / 'candidates.csv')
    assert len(table) == len(records) > len(summary['representative_ids'])
    plot = build_plot_data(read_json(output / 'metadata' / 'report.json'))
    accuracy = next(f for f in plot['figures'] if f['kind'] == 'selection_tradeoff')
    assert len(accuracy['rows']) == len(records)
    predictions = next(f for f in plot['figures'] if f['kind'] == 'observed_prediction')
    assert {r['candidate_id'] for r in predictions['rows']} == set(summary['representative_ids'])
    assert list(output.rglob('*.png'))
    assert not list(root.rglob('final_metrics.json'))
    with pytest.raises(ValueError):
        container_stage.compare_selection(root, {'prepare': refs}, provenance)


def test_container_comparison_refuses_unrun_methods(synthetic_config, tmp_path):
    from scripts.container_stage import compare_selection
    cfg = resolve_config(synthetic_config)
    root = tmp_path / 'result' / 'calculation'
    refs = prepare(cfg, root / 'prepare')
    with pytest.raises(ValueError, match='missing discoveries'):
        compare_selection(root, {'prepare': refs}, read_json(refs['run']))


@pytest.mark.parametrize("late_collision", [False, True])
def test_container_comparison_keeps_failed_method(synthetic_config, tmp_path, monkeypatch, late_collision):
    from scripts.container_stage import compare_selection
    cfg = resolve_config(synthetic_config)
    root = tmp_path / 'result' / 'calculation'
    refs = prepare(cfg, root / 'prepare')
    for engine in cfg['search']['engines']:
        for seed in cfg['search']['seeds']:
            path = root / 'discover' / engine / str(seed)
            path.mkdir(parents=True)
            candidates = path / 'candidates.jsonl'
            candidates.write_text('')
            write_json(path / 'engine_summary.json', {
                'engine': engine, 'seed': seed, 'status': 'failed', 'reason': 'fixture failure',
                'wall_seconds': 1, 'exported_candidates': 0,
                'prepared_sha256': digest(refs['run']), 'candidates_sha256': digest(candidates),
            })
    if late_collision:
        import ml_platform_symbolic
        original = ml_platform_symbolic.export_outputs
        sentinel = root.parent / 'report.html'
        def collide(*args, **kwargs):
            sentinel.write_text('concurrent result')
            return original(*args, **kwargs)
        monkeypatch.setattr(ml_platform_symbolic, 'export_outputs', collide)
        with pytest.raises(ValueError, match='already exists'):
            compare_selection(root, {'prepare': refs}, read_json(refs['run']))
        assert sentinel.read_text() == 'concurrent result'
        assert (root / 'study').is_dir()
        assert (root / 'comparison').is_dir()
        assert not (root / 'comparison_summary.json').exists()
        assert not (root.parent / 'tables').exists()
        return
    output = compare_selection(root, {'prepare': refs}, read_json(refs['run']))
    rows = pd.read_csv(output / 'tables/candidates.csv')
    assert set(rows['run_status']) == {'failed'}
    summary = read_json(root / 'comparison_summary.json')
    assert summary['discovery_complete'] is False
    assert summary['representative_ids'] == []
