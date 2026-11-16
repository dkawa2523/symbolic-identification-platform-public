"""Actual ProGED candidate completion survives injected worker interruption."""
import subprocess
import time

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.native


@pytest.mark.parametrize("exit_kind", ["wall", "crash"])
def test_completed_proged_snapshot_recovery(tmp_path, monkeypatch, exit_kind):
    pytest.importorskip("ProGED")
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import prepare, read_json
    from ml_platform_symbolic.engines import runner

    x = np.linspace(0.2, 2, 48)
    pd.DataFrame({"sample_id": [f"r{i}" for i in range(len(x))], "x": x,
                  "y": 0.3 + 1.7*x + 0.02*np.sin(x)}).to_csv(tmp_path / "data.csv", index=False)
    pd.DataFrame({"sample_id": [f"r{i}" for i in range(len(x))],
                  "partition": ["train"]*36 + ["selection"]*12}).to_csv(tmp_path / "split.csv", index=False)
    grammar = "S -> 'C' '*' 'x0' [0.25] | 'C' '+' 'x0' [0.25] | 'C' '*' 'x0' '+' 'C' [0.25] | 'C' '*' 'x0' '**' '2' [0.25]"
    cfg = resolve_config({
        "data": {"local_path": str(tmp_path / "data.csv"), "target_column": "y", "feature_columns": ["x"],
                 "units": {"x": "1", "y": "1"}, "domain": {"x": [0.2, 2.0]}},
        "split": {"file": str(tmp_path / "split.csv")},
        "search": {"engines": ["proged"], "seeds": [17], "max_wall_seconds": 60, "allow_partial_engines": True,
                   "operators": ["add", "mul", "pow_int"], "engine_options": {"proged": {
                       "grammar": grammar, "sample_size": 4, "max_attempts": 100,
                       "optimizer": "DE", "optimizer_iterations": 100, "optimizer_population": 20}}},
        "constraints": {"max_nodes": 30}, "evaluation": {"mode": "exploratory"}})
    refs = prepare(cfg, tmp_path / "prepared")
    provenance = read_json(refs["run"])
    directory = tmp_path / "discovery"
    original_popen = runner.subprocess.Popen

    class Interrupted:
        def __init__(self, *args, **kwargs):
            self.process = original_popen(*args, **kwargs)
            self.injected = False
        def __getattr__(self, name):
            return getattr(self.process, name)
        def wait(self, timeout=None):
            if timeout is None or self.injected:
                return self.process.wait(timeout=timeout)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if (directory / "completed_progress.json").exists():
                    self.injected = True
                    if exit_kind == "wall":
                        raise subprocess.TimeoutExpired(self.process.args, timeout)
                    self.process.kill()
                    return self.process.wait()
                if self.process.poll() is not None:
                    return self.process.returncode
                time.sleep(0.01)
            raise subprocess.TimeoutExpired(self.process.args, timeout)
    monkeypatch.setattr(runner.subprocess, "Popen", Interrupted)
    artifacts = runner.run_discovery(refs, provenance, "proged", 17, directory)
    summary = read_json(artifacts["summary"])
    assert summary["status"] == "partial", summary
    assert summary["native_execution_status"] == ("interrupted" if exit_kind == "wall" else "failed")
    progress = summary["progress"]
    assert 0 < progress["fit_completed_count"] < progress["generated_count"]
    assert progress["fit_completed_count"] == progress["fit_started_count"]
    assert summary["exported_candidates"] > 0
    assert read_json(summary["original_execution"]["artifact"])["status"] in {"failed", "timeout"}
