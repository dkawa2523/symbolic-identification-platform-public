"""Real PhySO checkpoint + injected process interruption, no fake hypotheses."""

import subprocess
import time

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import prepare, read_json, write_json
from ml_platform_symbolic.engines import runner
from ml_platform_symbolic.equation_evaluation import create_study, load_study


@pytest.mark.native
@pytest.mark.parametrize("exit_kind", ["wall", "crash"])
def test_real_physo_completed_iteration_survives_process_interruption(tmp_path, monkeypatch, exit_kind):
    pytest.importorskip("physo")
    x = np.linspace(0.2, 2.0, 72)
    y = 0.37 + 1.43 * np.exp(-0.81 * x) + np.random.default_rng(551).normal(0, 0.03, len(x))
    frame = pd.DataFrame({"sample_id": [f"r{i}" for i in range(len(x))], "x": x, "y": y})
    frame.to_csv(tmp_path / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 48 + ["selection"] * 12 + ["final"] * 12).to_csv(
        tmp_path / "split.csv", index=False
    )
    cfg = resolve_config(
        {
            "data": {
                "local_path": str(tmp_path / "table.csv"),
                "target_column": "y",
                "feature_columns": ["x"],
                "units": {"x": "1", "y": "1"},
                "domain": {"x": [0.2, 2.0]},
            },
            "split": {"file": str(tmp_path / "split.csv")},
            "search": {
                "engines": ["physo"],
                "seeds": [0],
                "max_wall_seconds": 60,
                "allow_partial_engines": True,
                "operators": ["add", "sub", "mul", "div", "exp", "sin", "pow_int"],
                "engine_options": {
                    "physo": {"epochs": 1000, "batch_size": 64, "max_n_evaluations": 100000, "constants_per_unit": 2}
                },
            },
            "constraints": {"max_nodes": 16},
        }
    )
    refs = prepare(cfg, tmp_path / "prepared")
    provenance = read_json(refs["run"])
    directory = tmp_path / "native"
    real_popen = runner.subprocess.Popen

    class Interrupted:
        def __init__(self, *args, **kwargs):
            self.process = real_popen(*args, **kwargs)
            self.interruption_injected = False

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            if timeout is None or self.interruption_injected:
                return self.process.wait(timeout=timeout)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if (directory / "completed_progress.json").exists():
                    # Inject the search interruption once; cleanup must reap the real process.
                    self.interruption_injected = True
                    if exit_kind == "wall":
                        raise subprocess.TimeoutExpired(self.process.args, timeout)
                    self.process.kill()
                    return self.process.wait()
                if self.process.poll() is not None:
                    return self.process.returncode
                time.sleep(0.05)
            raise subprocess.TimeoutExpired(self.process.args, timeout)

    monkeypatch.setattr(runner.subprocess, "Popen", Interrupted)
    artifacts = runner.run_discovery(refs, provenance, "physo", 0, directory)
    summary = read_json(artifacts["summary"])
    assert summary["status"] == "partial", summary
    assert summary["native_execution_status"] == ("interrupted" if exit_kind == "wall" else "failed")
    assert summary["termination_reason"] == ("hard_wall_limit" if exit_kind == "wall" else "native_failure")
    assert read_json(summary["original_execution"]["artifact"])["status"] in {"failed", "timeout"}
    assert summary["native_work"]["optimizer_steps"] >= 1
    assert summary["exported_candidates"] > 0
    sources = [{"discovery": artifacts["summary"], "prepared_run": refs["run"]}]
    study = create_study({"prepared_run": refs["run"], "sources": sources}, run_dir=tmp_path / "study")
    _, records, rows, _ = load_study(study)
    assert any(v["kind"] == "explicit" for v in records)
    assert all(v["origin"]["summary"]["status"] == "partial" for v in records)
    assert not (tmp_path / "final_access.json").exists()
    write_json(
        tmp_path / "native_progress_control.json",
        {
            "scope": "Actual native first completed update, followed by injected interruption. Not a natural wall-budget study or convergence claim.",
            "summary": summary,
            "records": len(records),
            "eligible": sum(v["eligible"] for v in rows),
            "Final_used": False,
        },
    )
