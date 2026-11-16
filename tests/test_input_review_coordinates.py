"""Preflight/worker source coordinates; real boundaries, dummy asset bytes only."""

from copy import deepcopy

import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import digest, prepare, read_json, write_json
from ml_platform_symbolic.engines import e2e
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryResult
from ml_platform_symbolic.engines.runner import _worker, discovery_job, discovery_request
from ml_platform_symbolic.input_review import review_inputs


def test_mapped_public_preflight_matches_worker_coordinates(synthetic_config, tmp_path, monkeypatch):
    cfg = deepcopy(synthetic_config)
    cfg["data"]["domain"] = {"x": [0.1, 4.0]}
    cfg["input_map"] = {
        "kind": "affine_chart",
        "target_columns": ["u"],
        "target_units": {"u": "1"},
        "forward": {"matrix": [[2.0]], "offset": [3.0]},
        "inverse": {"matrix": [[0.5]], "offset": [-1.5]},
    }
    cfg["search"].update(
        engines=["e2e"],
        operators=["add", "sub", "mul", "div"],
        response_mode="search",
        engine_options={
            "e2e": {
                "rational_domain_admission": {
                    "column": "x",
                    "interval": [0.1, 4.0],
                    "max_calls": 20,
                    "max_total_operations": 10000,
                }
            }
        },
    )
    cfg["constraints"].update(probes=[{"x": 1.0}], response_bounds=[-100.0, 100.0], response_enforcement="hard")
    assets = tmp_path / "assets"
    assets.mkdir()
    weights = assets / "dummy.pt"
    weights.write_bytes(b"fixture only; never deserialize")
    manifest = write_json(
        assets / "manifest.json",
        {
            "engine": "e2e",
            "source": "coordinate test fixture",
            "source_revision": "fixed-test-fixture",
            "directory": ".",
            "files": [{"path": weights.name, "size_bytes": weights.stat().st_size, "sha256": digest(weights)}],
        },
    )
    cfg["model_assets"] = {"e2e": {"manifest_file": str(manifest)}}
    before = deepcopy(cfg)
    seen = []
    original = e2e.validate_request

    def validate(request, context):
        seen.append(deepcopy(context.provenance))
        return original(request, context)

    monkeypatch.setattr(e2e, "validate_request", validate)
    review = review_inputs(cfg)
    assert review["status"] == "inputs_valid" and review["engines"]["e2e"]["adapter_validation"] == "passed"
    assert cfg == before
    preflight = seen[-1]
    refs = prepare(resolve_config(cfg), tmp_path / "prepare")
    job = discovery_job(refs, read_json(refs["run"]), "e2e", 731)
    request = discovery_request(job)
    assert request.variables[0]["column"] == "u"
    # Demonstrate the old empty-provenance failure, not merely a spy assertion.
    with pytest.raises(NotImplementedError, match="matching public source"):
        original(request, DiscoveryContext(tmp_path, assets_dir=assets))
    worker_context = []

    def discover(request, context):
        worker_context.append(deepcopy(context.provenance))
        original(request, context)
        return DiscoveryResult([], "failed", {"scope": "dummy boundary test; no native execution"})

    monkeypatch.setattr(e2e, "discover", discover)
    job_dir = tmp_path / "worker"
    job_dir.mkdir()
    job_path = write_json(job_dir / "job.json", job)
    _worker(job_path)
    assert len(worker_context) == 1
    for key in ["public_units", "unit_coordinates", "input_geometry"]:
        assert preflight[key] == worker_context[0][key] == job[key]
    assert preflight["input_geometry"]["source_variables"][0]["column"] == "x"
    assert read_json(job_dir / "engine_summary.json")["native_execution_status"] == "completed"
