"""Known-relation benchmark input checks, not native search or model quality."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
from ml_platform_symbolic.data import digest, read_json, write_json
from ml_platform_symbolic.engines.registry import get_engine
from ml_platform_symbolic.problem import problem_config


@pytest.fixture
def entry():
    path = Path(__file__).parents[1] / "benchmarks/constraints/run.py"
    spec = importlib.util.spec_from_file_location("constraint_known_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings(tmp_path, monkeypatch):
    # The input review verifies these fixture bytes, never loads a native model.
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "pretrained/e2e"
    assets.mkdir(parents=True)
    weights = assets / "fixture.bin"
    weights.write_bytes(b"input-review-only; not a native model")
    write_json(assets / "manifest.json", {
        "engine": "e2e", "source": "input-review fixture", "source_revision": "fixed-test-fixture",
        "directory": ".", "files": [{"path": weights.name, "size_bytes": weights.stat().st_size,
                                      "sha256": digest(weights)}],
    })
    return {
        "methods": {
            "pysr": {"niterations": 2, "seconds": 20},
            "parfam": {"family": "rational", "functions": [], "degree": 1, "maxiter": 2},
            "torchsisso": {"dimensional": True},
            "physo": {"epochs": 1, "batch_size": 8, "max_n_evaluations": 8},
            "e2e": {"operator_mask": True, "max_train_rows": None},
        },
        "operators": ["add", "sub", "mul", "div"],
        "datasets": {"synthetic": {"max_nodes": 64}},
        "arms": ["none", "search"], "response_enforcement": "hard",
    }


@pytest.mark.parametrize("engine", ["pysr", "parfam", "torchsisso", "e2e", "physo"])
def test_each_primary_adapter_receives_identity_known_relation_without_native_search(
    entry, settings, tmp_path, monkeypatch, engine,
):
    adapter = get_engine(engine)
    original = adapter.validate_request
    seen = []

    def validate(request, context):
        if request.equation_template is not None:
            seen.append(deepcopy(request.equation_template))
            assert request.variables[0]["id"] == "x0"
            assert request.equation_template["slots"]["h"]["arguments"] == [{"name": "x0", "unit": "1"}]
        return original(request, context)

    def forbidden(*args, **kwargs):
        raise AssertionError("Entry preparation must never run native discovery")

    monkeypatch.setattr(adapter, "validate_request", validate)
    monkeypatch.setattr(adapter, "discover", forbidden)
    before = deepcopy(settings)
    root = tmp_path / "campaign"
    entry.prepare_run(root, [engine], [17], 60, test_seed=17001, known_relation=True, settings=settings)
    plan = read_json(root / "plan.json")
    assert len(seen) == 2 and seen[0] == seen[1]
    assert settings == before
    for refs in plan["prepared"].values():
        cfg = problem_config(read_json(refs["run"]))
        assert cfg["constraints"]["equation_template"] == seen[0]
        assert cfg["search"]["engines"] == [engine]
        assert not refs.get("final")
    assert not (root / "test_access.json").exists()


def test_unsupported_template_adapter_is_rejected_without_unconstrained_fallback(entry, settings, tmp_path):
    settings["methods"]["proged"] = {}
    with pytest.raises(ValueError, match="template is unsupported"):
        entry.prepare_run(tmp_path / "blocked", ["proged"], [17], 60, known_relation=True, settings=settings)
    assert not (tmp_path / "blocked/plan.json").exists()
    assert not list((tmp_path / "blocked").rglob("worker.log"))


def test_adapter_specific_incompatible_template_profile_is_not_hidden(entry, settings, tmp_path):
    settings["methods"]["torchsisso"]["dimensional"] = False
    with pytest.raises(ValueError, match="dimensional"):
        entry.prepare_run(tmp_path / "blocked", ["torchsisso"], [17], 60, known_relation=True, settings=settings)
    assert not (tmp_path / "blocked/plan.json").exists()
