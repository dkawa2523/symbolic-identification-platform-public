"""Completed native candidate recovery, never unfinished decoder output."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest


def test_phase_only_checkpoint_has_no_invented_candidates(tmp_path):
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.rasr_progress import NativeProgress

    snapshots = []
    progress = NativeProgress(None, DiscoveryContext(tmp_path, checkpoint=snapshots.append), None)
    progress("decode_started")
    assert not snapshots[-1].candidates
    assert snapshots[-1].summary["phase"] == "decode_started"
    assert snapshots[-1].summary["native_execution_status"] == "partial"
    json.dumps(snapshots[-1].summary, allow_nan=False)


@pytest.mark.native
def test_native_completed_refinement_checkpoint_and_interruption(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/work/pretrained/rasr/cache/xdg")
    monkeypatch.setenv("HF_HOME", "/work/pretrained/rasr/cache/hf")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import simplipy
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.progress import load_progress, save_progress
    from ml_platform_symbolic.engines.rasr import export_candidate
    from ml_platform_symbolic.engines.rasr_progress import NativeProgress
    from ml_platform_symbolic.expressions import evaluate_native
    from src.rasr.inference.rasr import RASR

    fitted = RASR.__new__(RASR)
    fitted.simplipy_engine_name = "dev_7-3"
    fitted.simplipy_engine = simplipy.SimpliPyEngine.load("dev_7-3", install=True)
    fitted.refiner_workers, fitted.verbose, fitted.parsimony = 0, False, 0.0
    fitted.n_variables = 1
    fitted.variable_mapping = {}
    fitted._job_config = SimpleNamespace(tokenizer=SimpleNamespace(variables=["x1"]))
    fitted.logger = logging.getLogger("checkpoint-test")
    fitted.X_mean_, fitted.X_std_ = np.array([10.0]), np.array([2.0])
    fitted.y_mean_, fitted.y_std_scale_ = 50.0, 5.0
    X = np.linspace(10, 12, 20)[:, None]
    z = ((X - 10) / 2).astype(np.float32)
    target = 2.0 * z[:, 0]
    request = DiscoveryRequest(
        X,
        50 + 5 * target,
        np.arange(len(X)),
        [{"id": "x", "column": "x", "unit": "1", "domain": [10, 12]}],
        {"unit": "1"},
        ["mul", "div", "add", "sub"],
        {},
        17,
        {},
    )
    payloads = [
        {"tokens": ["*", "<constant>", "x1"], "cand_id": i, "log_prob": 0.0, "n_restarts": 2, "init_noise_scale": 0.5}
        for i in range(2)
    ]
    np.random.seed(17)
    baseline = fitted._run_refinement(payloads, z, target)
    job = tmp_path / "job.json"
    job.write_text("{}")
    context = DiscoveryContext(tmp_path, checkpoint=lambda result: save_progress(job, result, {}))
    progress = NativeProgress(request, context, export_candidate)
    progress.fitted = fitted
    fitted.symbolic_progress = progress
    np.random.seed(17)
    observed = fitted._run_refinement(payloads, z, target)
    for a, b in zip(baseline, observed, strict=True):
        np.testing.assert_array_equal(a["fits"][0][0], b["fits"][0][0])
        assert a["tokens"] == b["tokens"]
    recovered, _, _ = load_progress(job, request.variables)
    assert len(recovered.candidates) == 2
    assert recovered.status == "partial"
    assert all(c.native_check["passed"] for c in recovered.candidates)
    np.testing.assert_allclose(
        evaluate_native(recovered.candidates[0].equation, X, request.variables), request.y, rtol=1e-6
    )
    assert not hasattr(fitted, "results_")  # Oracle clone did not overwrite native search state.

    def interrupt(phase, **fields):
        if phase == "candidate_started" and fields["candidate_id"] == 1:
            raise RuntimeError("simulated interruption")
        progress(phase, **fields)

    progress = NativeProgress(request, context, export_candidate)
    progress.fitted = fitted
    fitted.symbolic_progress = interrupt
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fitted._run_refinement(payloads, z, target)
    recovered, _, _ = load_progress(job, request.variables)
    assert len(recovered.candidates) == 1
    assert recovered.summary["completed_refinements"] == 1

    # Native IEEE finite revival at an inner pole is compared raw, while common
    # real-domain evaluation keeps the row invalid after save/recovery.
    from ml_platform_symbolic.expressions import evaluate

    progress = NativeProgress(request, context, export_candidate)
    progress.fitted = fitted
    fitted.symbolic_progress = progress
    pole = [{"tokens": ["inv", "inv", "x1"], "cand_id": 3, "log_prob": 0.0, "n_restarts": 1, "init_noise_scale": 0.0}]
    fitted._run_refinement(pole, z, target)
    recovered, _, _ = load_progress(job, request.variables)
    assert len(recovered.candidates) == 1
    equation = recovered.candidates[0].equation
    assert evaluate_native(equation, X, request.variables)[0] == 50.0
    assert np.isnan(evaluate(equation, X, request.variables)[0])

    with pytest.raises(ValueError, match="Train-only native ranking"):
        fitted.fit(X, request.y, X_val=X, y_val=request.y)
