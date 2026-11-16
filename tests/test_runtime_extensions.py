"""External-adapter workflow and actionable worker initialization failures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ml_platform_symbolic.data import read_json, write_json


def test_external_adapter_runs_discovery_analysis_and_portable_reload(tmp_path):
    # Source regression, not installed-wheel acceptance. The example uses a real
    # NumPy fit and actual discovery/evaluator subprocesses without optional SR.
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "examples/extensions/installed_extension"), str(root / "pkgs/symbolic/src"), *sys.path]
    )
    output = tmp_path / "extension"
    result = subprocess.run(
        [sys.executable, str(root / "examples/extensions/runtime.py"), "--output", str(output)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    verified = read_json(output / "verification.json")
    assert verified["native_status"] == "completed"
    assert verified["evaluation"]["outcome"] == "ok"
    assert verified["max_absolute_error"] < 1e-10
    assert read_json(output / "study/resolved.json")["final_history"] == "known_exposed"
    assert read_json(output / "discovery/job.json")["runtime_extensions"] == [verified["runtime"]]


@pytest.mark.parametrize("failure", ["missing_module", "changed_helper"])
def test_worker_preserves_extension_initialization_error(tmp_path, failure):
    module = "third_party_audit_extension"
    extension = tmp_path / f"{module}.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
    declaration = {"module": module, "version": "1.0"}
    if failure == "changed_helper":
        extension.write_text(
            "__version__ = '1.0'\n__implementation_files__ = ['helper.py']\ndef initialize():\n    pass\n",
            encoding="utf-8",
        )
        helper = tmp_path / "helper.py"
        helper.write_text("VALUE = 1\n", encoding="utf-8")
        initialized = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from ml_platform_symbolic.domains import register_runtime_extension; "
                f"print(json.dumps(register_runtime_extension('{module}', '1.0')))",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        declaration = json.loads(initialized.stdout)
        # Keep entrypoint and version unchanged: helper drift alone must fail.
        helper.write_text("VALUE = 2\n", encoding="utf-8")
    job = write_json(
        tmp_path / "job.json",
        {"engine": "dictionary", "seed": 0, "runtime_extensions": [declaration]},
    )
    result = subprocess.run(
        [sys.executable, "-m", "ml_platform_symbolic.engines.worker", str(job)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = read_json(tmp_path / "engine_summary.json")
    assert summary["status"] == "failed"
    expected = module if failure == "missing_module" else "implementation files changed"
    assert expected in summary["reason"]
    assert not (tmp_path / "candidates.jsonl").exists()
