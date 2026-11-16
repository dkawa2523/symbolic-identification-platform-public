"""Composition semantics, independent of heavy pretrained runtimes."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines import pretrained_template as adapter
from ml_platform_symbolic.engines.base import Candidate, DiscoveryContext, DiscoveryResult
from ml_platform_symbolic.expressions import evaluate
from test_template_enumeration import request_for


def install_native(monkeypatch):
    calls = []

    def discover(request, context):
        calls.append(deepcopy(request))
        equation = {
            "schema_version": 2,
            "ast": {"op": "mul", "args": [{"param": "a"}, {"var": "u"}]},
            "parameters": {"a": {"value": 0.5, "trainable": True, "unit": None, "bounds": [None, None]}},
            "arithmetic": "numpy_ieee",
        }
        return DiscoveryResult(
            [Candidate(f"native/{request.seed}", "e2e", request.seed, equation, {"passed": True})], "ok", {}
        )

    native = SimpleNamespace(validate_request=lambda *a: None, discover=discover)
    monkeypatch.setattr(adapter, "get_engine", lambda name: native)
    return calls


def test_shared_slot_joint_fit_does_not_treat_response_as_slot_truth(tmp_path, monkeypatch):
    calls = install_native(monkeypatch)
    X = np.random.default_rng(77).uniform(-1, 1, (40, 2))
    request = request_for(X, 2.3 * (X[:, 0] + X[:, 1]), shared=True, options={"max_proposals": 2})
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert result.status == "ok", result.summary
    assert len(calls) == 1
    np.testing.assert_equal(calls[0].y, request.y)
    assert calls[0].equation_template is None
    assert calls[0].limits["response_mode"] == "filter"
    np.testing.assert_allclose(evaluate(result.candidates[0].equation, X, request.variables), request.y, atol=1e-6)
    assert result.candidates[0].native_check["independent_native_oracle"] is False
    assert '"unit": null' in (tmp_path / "proposals/h/candidates.json").read_text()


def test_outer_parameters_and_response_conditions_jointly_fit(tmp_path, monkeypatch):
    install_native(monkeypatch)
    X = np.random.default_rng(91).uniform(-1, 1, (40, 2))
    request = request_for(X, 2.5 * X[:, 0] + 1.7 * X[:, 1], outer=True, options={"max_nfev": 400})
    request.limits.update(response_mode="search", probes=[{"x0": 1.0, "x1": 0.0}], response_bounds=[2.0, 3.0])
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert result.status == "ok", result.summary
    np.testing.assert_allclose(evaluate(result.candidates[0].equation, X, request.variables), request.y, atol=1e-5)


def test_unsupported_units_rejected_before_native(monkeypatch, tmp_path):
    install_native(monkeypatch)
    request = request_for(np.ones((3, 1)), np.ones(3), options={"max_proposals": 1})
    request.variables[0]["unit"] = "m"
    with pytest.raises((ValueError, NotImplementedError)):
        adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")


def test_declared_slot_size_not_silently_relaxed(tmp_path, monkeypatch):
    install_native(monkeypatch)
    request = request_for(np.arange(4.0)[:, None], np.arange(4.0), options={"max_proposals": 1})
    request.equation_template["slots"]["h"]["max_nodes"] = 1
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert not result.candidates
    assert result.summary["termination"] == "no_supported_proposals"
    assert result.summary["rejected_proposals"]


def test_diagonal_combinations_cover_independent_slot_choices():
    combinations = list(adapter._combinations({"a": [1, 2], "b": [3, 4]}))
    assert len(combinations) == 4
    assert combinations[0] == {"a": 1, "b": 3}
    assert combinations[-1] == {"a": 2, "b": 4}


def test_distinct_slots_have_distinct_runs_and_coefficients(tmp_path, monkeypatch):
    calls = install_native(monkeypatch)
    X = np.random.default_rng(91).uniform(-1, 1, (40, 2))
    request = request_for(X, 2.5 * X[:, 0] + 1.7 * X[:, 1], shared=True, options={"max_nfev": 400})
    request.equation_template["slots"]["g"] = deepcopy(request.equation_template["slots"]["h"])
    request.equation_template["ast"]["args"][1]["slot"] = "g"
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert result.status == "ok", result.summary
    assert len(calls) == 2 and calls[0].seed != calls[1].seed
    assert not np.array_equal(calls[0].X, calls[1].X)
    np.testing.assert_allclose(evaluate(result.candidates[0].equation, X, request.variables), request.y, atol=1e-6)
    assert len(result.candidates[0].native_check["proposal_lineage"]) == 2
    assert "native_proposals" in result.native_artifacts


def test_unknown_outer_parameter_cannot_define_proposal_inputs(tmp_path, monkeypatch):
    install_native(monkeypatch)
    X = np.ones((3, 2))
    request = request_for(X, np.ones(3), outer=True, options={"max_proposals": 1})
    request.equation_template["ast"]["args"][0]["args"] = [{"param": "a"}]
    with pytest.raises(NotImplementedError, match="unknown template parameters"):
        adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")


def test_slot_coordinate_proposal_is_new_not_equivalent_and_preserves_original():
    from ml_platform_symbolic.engines.e2e import export_candidate

    slot = {"arguments": [{"name": "u", "unit": "1"}], "allowed_operators": ["mul"], "max_nodes": 3}
    original = export_candidate("mul,2.3,x_0", [{"id": "u", "unit": "1"}], [3.0], [2.0])
    candidate = Candidate("e2e/1/0", "e2e", 1, original, {"passed": True})
    saved = deepcopy(original)
    transformations = {}
    body = adapter._slot_body(candidate, slot, "slot", transformations)
    assert original == saved
    assert body["ast"] == {"op": "mul", "args": [{"param": "c1"}, {"var": "u"}]}
    assert transformations["equivalence_claim"] is False
    assert len(transformations["removed_input_normalizations"]) == 1
    assert all(p.get("origin") != "train_normalization" for p in body["parameters"].values())
    with pytest.raises(ValueError, match="outside slot grammar"):
        adapter._slot_body(candidate, slot, "native")
    points = np.array([[0.0], [4.0]])
    assert not np.allclose(
        evaluate(original, points, [{"id": "u", "unit": "1"}]), evaluate(body, points, [{"id": "u", "unit": "1"}])
    )


@pytest.mark.native
@pytest.mark.parametrize("coordinates", ["native", "slot"])
def test_native_e2e_known_template_and_joint_response_conditions(tmp_path, coordinates):
    import json
    from pathlib import Path

    pytest.importorskip("symbolicregression")
    assets = Path("pretrained/e2e").resolve()
    if not (assets / "manifest.json").is_file():
        pytest.skip("Pinned E2E assets required")
    X = np.linspace(-1.5, 1.5, 40)[:, None]
    request = request_for(
        X,
        1 + 2.3 * X[:, 0] ** 2,
        options={
            "proposal_options": {"beam_size": 8, "n_trees_to_refine": 8, "refinement_seconds": 3},
            "max_proposals": 8,
            "max_combinations": 8,
            "proposal_coordinates": coordinates,
        },
    )
    request.operators = ["add", "sub", "mul", "div", "pow_int"]
    request.equation_template["slots"]["h"].update(allowed_operators=request.operators, max_nodes=32)
    request.equation_template["ast"] = {"op": "add", "args": [{"const": 1}, request.equation_template["ast"]]}
    request.limits.update(response_mode="search", probes=[{"x0": 1.0}], response_bounds=[3.2, 3.4])
    request.seed = 42
    result = adapter.discover(request, DiscoveryContext(tmp_path, assets_dir=assets), engine="e2e")
    assert result.candidates, result.summary
    for candidate in result.candidates:
        assert 3.2 <= evaluate(candidate.equation, np.array([[1.0]]), request.variables)[0] <= 3.4
        assert candidate.native_check["independent_native_oracle"] is False
    original = json.loads((tmp_path / "proposal_archive.json").read_text())["h"]["candidates"]
    assert original and all(c["native_check"]["passed"] for c in original)
    assert all(c["equation"]["arithmetic"] == "numpy_ieee" for c in original)
    assert any(
        any(p.get("origin") == "train_normalization" for p in c["equation"]["parameters"].values()) for c in original
    )
    assert result.summary["joint_response_conditions"]["enforced_in_coefficient_fit"]
    if coordinates == "slot":
        transformations = result.summary["native_runs"]["h"]["proposal_transformations"]
        assert any(t["removed_input_normalizations"] for t in transformations.values())
        assert all(t["equivalence_claim"] is False for t in transformations.values())


def test_shared_calls_input_order_and_saved_prediction_round_trip(tmp_path, monkeypatch):
    from dataclasses import replace

    import pandas as pd
    from ml_platform_symbolic.model import load_model, save_model

    calls = install_native(monkeypatch)
    X = np.random.default_rng(123).uniform(-1, 1, (45, 2))
    y = 3 * X[:, 0] + 2.3 * (X[:, 1] + 2 * X[:, 0])
    request = request_for(X, y, shared=True, options={"max_proposals": 1})
    template = request.equation_template
    template["ast"] = {
        "op": "add",
        "args": [
            {"op": "mul", "args": [{"const": 3}, {"var": "x0"}]},
            {
                "op": "add",
                "args": [
                    {"slot": "h", "args": [{"var": "x1"}]},
                    {"slot": "h", "args": [{"op": "mul", "args": [{"const": 2}, {"var": "x0"}]}]},
                ],
            },
        ],
    }
    baseline = adapter.discover(request, DiscoveryContext(tmp_path / "normal"), engine="e2e")
    reordered = replace(request, X=X[:, ::-1], variables=request.variables[::-1])
    result = adapter.discover(reordered, DiscoveryContext(tmp_path / "reordered"), engine="e2e")
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0].X[:, 0], X[:, 1])
    np.testing.assert_array_equal(calls[1].X[:, 0], X[:, 1])
    first, second = baseline.candidates[0], result.candidates[0]
    np.testing.assert_allclose(evaluate(first.equation, X, request.variables), y, atol=1e-6)
    np.testing.assert_allclose(evaluate(second.equation, X[:, ::-1], reordered.variables), y, atol=1e-6)
    schema = {"variables": reordered.variables, "target": request.target, "feature_ranges": [[-3, 3], [-3, 3]]}
    saved = save_model(tmp_path / "shared.zip", second.equation, schema, {"native_check": second.native_check})
    Z = np.random.default_rng(456).uniform(-1, 1, (25, 2))
    frame = pd.DataFrame(Z, columns=["x0", "x1"])
    np.testing.assert_allclose(
        load_model(saved).predict(frame[["x1", "x0"]]), 3 * Z[:, 0] + 2.3 * (Z[:, 1] + 2 * Z[:, 0]), atol=1e-6
    )
    assert load_model(saved).metadata["native_check"]["independent_native_oracle"] is False


def test_rasr_slot_coordinate_extraction_preserves_original_and_rejects_unknown_scaler():
    from ml_platform_symbolic.engines.rasr import export_candidate

    native = SimpleNamespace(tokens=["*", "<constant>", "x1"], constants=[2.3])
    fitted = SimpleNamespace(X_mean_=np.array([3.0]), X_std_=np.array([2.0]), y_mean_=7.0, y_std_scale_=5.0)
    original = export_candidate(native, fitted, [{"id": "u", "unit": "1"}], {"unit": "1"})
    candidate = Candidate("rasr/1/0", "rasr", 1, original, {"passed": True})
    saved = deepcopy(original)
    slot = {"arguments": [{"name": "u", "unit": "1"}], "allowed_operators": ["mul"], "max_nodes": 3}
    transformation = {}
    body = adapter._slot_body(candidate, slot, "slot", transformation)
    assert original == saved
    assert body["ast"] == {"op": "mul", "args": [{"param": "c0"}, {"var": "u"}]}
    assert transformation["removed_output_normalizations"] == ["y_scale", "y_mean"]
    assert transformation["equivalence_claim"] is False
    original["parameters"]["y_scale"]["trainable"] = True
    with pytest.raises(ValueError, match="fixed unit1"):
        adapter._slot_body(candidate, slot, "slot")


@pytest.mark.native
def test_rasr_slot_hybrid_archives_original_and_refitted_predictions(tmp_path, monkeypatch):
    from pathlib import Path

    assets = Path("pretrained/rasr").resolve()
    monkeypatch.setenv("XDG_CACHE_HOME", str(assets / "cache/xdg"))
    monkeypatch.setenv("HF_HOME", str(assets / "cache/hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("src.rasr.inference.rasr")
    from ml_platform_symbolic.engines.rasr import capabilities

    X = np.linspace(-1.5, 1.5, 40)[:, None]
    request = request_for(
        X,
        1 + 2.3 * X[:, 0] ** 2,
        options={
            "proposal_options": {"candidates": 4, "decode_batch_size": 4, "restarts": 1},
            "proposal_coordinates": "slot",
            "max_proposals": 4,
        },
    )
    request.operators = capabilities()["operators"]
    request.equation_template["slots"]["h"].update(allowed_operators=request.operators, max_nodes=32)
    request.equation_template["ast"] = {"op": "add", "args": [{"const": 1}, request.equation_template["ast"]]}
    request.seed = 42
    result = adapter.discover(
        request, DiscoveryContext(tmp_path, assets_dir=Path("pretrained/rasr").resolve()), engine="rasr"
    )
    assert result.candidates, result.summary
    with np.load(result.native_artifacts["template_predictions"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["row_ids"], request.row_ids)
        assert any(k.startswith("proposal_") for k in arrays.files)
        assert any(k.startswith("assembled_") for k in arrays.files)
    transformations = result.summary["native_runs"]["h"]["proposal_transformations"]
    assert all(t["removed_output_normalizations"] == ["y_scale", "y_mean"] for t in transformations.values())


def test_prediction_record_failure_does_not_commit_proposal_pool(tmp_path, monkeypatch):
    install_native(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("Injected prediction archive failure")

    monkeypatch.setattr(adapter, "evaluate_native", fail)
    request = request_for(np.arange(4.0)[:, None], np.arange(4.0), options={"max_proposals": 1})
    result = adapter.discover(request, DiscoveryContext(tmp_path), engine="e2e")
    assert not result.candidates
    assert result.summary["native_runs"]["h"]["retained_proposals"] == 0
    assert result.summary["attempted_combinations"] == 0
    assert result.summary["rejected_proposals"][0]["reason"] == "Injected prediction archive failure"
    assert "template_predictions" not in result.native_artifacts
