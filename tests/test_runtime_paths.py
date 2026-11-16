"""Runtime storage boundaries and writable result mounts, without Docker/network."""

import json
import sys
from pathlib import Path

import pytest

from scripts import run_linux


@pytest.fixture
def launcher_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "containers").mkdir(parents=True)
    (project / "containers/profiles.json").write_text(
        json.dumps({"core": {"image": "fixture", "gpu": False}}), encoding="utf-8"
    )
    monkeypatch.setattr(run_linux, "__file__", str(project / "scripts/run_linux.py"))
    monkeypatch.setattr(run_linux.subprocess, "check_output", lambda *a, **k: "sha256:fixture\n")
    return project


@pytest.mark.parametrize("memory", ["8g", "10g"])
def test_launcher_separates_readonly_inputs_from_writable_results(launcher_project, monkeypatch, memory):
    project = launcher_project
    commands = []

    def checked(args):
        commands.append(args)
        cid = Path(args[args.index("--cidfile") + 1])
        assert cid.parent.parent == project / "runtime"
        cid.write_text("a" * 64)

    monkeypatch.setattr(run_linux, "checked", checked)
    monkeypatch.setattr(sys, "argv", ["run_linux.py", "--memory", memory, "--", "python", "-V"])
    run_linux.main()
    args = commands[0]
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
    assert mounts == [
        f"type=bind,source={project},target=/work,readonly",
        f"type=bind,source={project / 'results'},target=/work/results",
    ]
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--memory") + 1] == memory
    assert not list((project / "runtime").iterdir())
    assert not list((project / "results").iterdir())


@pytest.mark.parametrize("destination", ["pretrained", "runtime", "outputs", "../outside", "results/../pretrained"])
def test_launcher_rejects_non_result_write_roots(launcher_project, monkeypatch, destination):
    monkeypatch.setattr(sys, "argv", ["run_linux.py", "--output-root", destination, "--", "true"])
    with pytest.raises(ValueError, match="output-root|Writable roots"):
        run_linux.main()


@pytest.mark.parametrize("directory", ["pretrained", "data"])
def test_asset_setup_writes_only_requested_root(launcher_project, monkeypatch, directory):
    commands = []
    monkeypatch.setattr(run_linux, "checked", commands.append)
    monkeypatch.setattr(
        sys, "argv", ["run_linux.py", "--setup-network", "--output-root", directory, "--", "true"]
    )
    run_linux.main()
    args = commands[0]
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
    assert mounts == [
        f"type=bind,source={launcher_project},target=/work,readonly",
        f"type=bind,source={launcher_project / directory},target=/work/{directory}",
    ]
    assert args[args.index("--network") + 1] == "bridge"


@pytest.mark.parametrize("directory", ["results"])
def test_launcher_rejects_symlink_escape(launcher_project, tmp_path, monkeypatch, directory):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (launcher_project / directory).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")
    monkeypatch.setattr(sys, "argv", ["run_linux.py", "--", "true"])
    with pytest.raises(ValueError, match="output-root|Writable roots"):
        run_linux.main()


def test_data_preparation_writes_only_explicit_data_root_without_network(launcher_project, monkeypatch):
    commands = []
    monkeypatch.setattr(run_linux, "checked", commands.append)
    monkeypatch.setattr(sys, "argv", ["run_linux.py", "--output-root", "data", "--", "true"])
    run_linux.main()
    args = commands[0]
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
    assert mounts == [f"type=bind,source={launcher_project},target=/work,readonly",
                      f"type=bind,source={launcher_project / 'data'},target=/work/data"]
    assert args[args.index("--network") + 1] == "none"


def test_memory_cap_cannot_be_disabled_by_zero(launcher_project, monkeypatch):
    calls = []
    monkeypatch.setattr(run_linux, "checked", calls.append)
    monkeypatch.setattr(sys, "argv", ["run_linux.py", "--memory", "0", "--", "true"])
    with pytest.raises(SystemExit):
        run_linux.main()
    assert not calls


@pytest.mark.parametrize("output_root,run_dir", [
    ("results", "results/check/calculation"),
    ("results/check", "results/check/output/calculation"),
])
def test_compare_launcher_stops_before_final(launcher_project, monkeypatch, output_root, run_dir):
    project = launcher_project
    (project / 'containers/profiles.json').write_text(json.dumps({'core': {'image': 'fixture', 'gpu': False, 'engines': ['linear']}}))
    (project / 'input.json').write_text('{}')
    stages = []
    root = project / run_dir
    def checked(command):
        index = command.index('scripts/container_stage.py')
        stage = command[index + 1]
        stages.append(stage)
        if stage == 'prepare':
            root.mkdir(parents=True, exist_ok=True)
            (root / 'state.json').write_text(json.dumps({'search': {'engines': ['linear'], 'seeds': [42]}}))
    monkeypatch.setattr(run_linux, 'checked', checked)
    monkeypatch.setattr(sys, 'argv', ['run_linux.py', '--config', 'input.json', '--output-root', output_root, '--run-dir', run_dir, '--compare'])
    run_linux.main()
    assert stages == ['prepare', 'discover', 'summarize', 'compare']
    execution = json.loads((root / 'container_execution.json').read_text())
    assert execution['status'] == 'comparison_finished'
    assert execution['final_accessed'] is False


def test_compare_rejects_readonly_staging_parent_before_any_output(launcher_project, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_linux, 'checked', calls.append)
    monkeypatch.setattr(sys, 'argv', ['run_linux.py', '--config', 'input.json', '--output-root', 'results/check',
                                    '--run-dir', 'results/check/calculation', '--compare'])
    with pytest.raises(SystemExit) as error:
        run_linux.main()
    assert error.value.code == 2
    assert 'output parent inside output-root' in capsys.readouterr().err
    assert not calls
    assert not (launcher_project / 'results').exists()


@pytest.mark.parametrize("entry", ["figures", "tables", "report.html"])
def test_compare_rejects_output_collision_before_native_work(launcher_project, monkeypatch, entry):
    project = launcher_project
    (project / 'input.json').write_text('{}')
    destination = project / 'results/check'
    destination.mkdir(parents=True)
    (destination / entry).write_text('existing result')
    commands = []
    monkeypatch.setattr(run_linux, 'checked', commands.append)
    monkeypatch.setattr(sys, 'argv', ['run_linux.py', '--config', 'input.json', '--run-dir', 'results/check/calculation', '--compare'])
    with pytest.raises(ValueError, match='output already exists'):
        run_linux.main()
    assert commands == []
    assert (destination / entry).read_text() == 'existing result'
