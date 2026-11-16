"""A hard timeout must stop the task's native descendants as well as its worker."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import prepare, read_json
from ml_platform_symbolic.engines import runner


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups; Windows keeps its taskkill process-tree path")
def test_timeout_stops_descendant_process(synthetic_config, tmp_path, monkeypatch):
    cfg = resolve_config(synthetic_config)
    cfg["search"].update(max_wall_seconds=1, allow_partial_engines=True)
    refs = prepare(cfg, tmp_path / "prepare")
    pid_path = tmp_path / "child.pid"
    original_popen = subprocess.Popen
    script = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )

    def start_worker(_command, **options):
        assert options["start_new_session"]
        return original_popen([sys.executable, "-c", script, str(pid_path)], **options)  # noqa: S603

    monkeypatch.setattr(runner.subprocess, "Popen", start_worker)
    child_pid = None
    try:
        result = runner.run_discovery(refs, read_json(refs["run"]), "dictionary", 42, tmp_path / "discover")
        assert read_json(result["summary"])["status"] == "timeout"
        child_pid = int(pid_path.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            status = Path(f"/proc/{child_pid}/stat")
            if status.exists() and status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return  # A killed zombie is waiting for the OS reaper, not executing.
            time.sleep(0.05)
        pytest.fail("Task-owned descendant was still executing after its worker timed out")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
