import json
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.data import digest, read_json, write_json
from ml_platform_symbolic.equation_comparison import compare_equations
from ml_platform_symbolic.equation_evaluation import create_study, load_study
from ml_platform_symbolic.hypotheses import predict_working
from ml_platform_symbolic.hypothesis_storage import read_hypotheses, write_hypotheses


def test_lossless_context_factoring_and_mutation_isolation(tmp_path):
    origin = {"engine": "arbitrary_engine", "summary": {"ledger": ["record" * 100] * 20}}
    records = [
        {
            "id": "point",
            "kind": "explicit",
            "origin": origin,
            "equation": {"parameters": {"c": {"value": 1.0000000000000002, "trainable": True}}},
        },
        {
            "id": "ensemble",
            "kind": "ensemble",
            "origin": deepcopy(origin),
            "members": [{"weight": 0.125, "equation": {}}, {"weight": 0.875, "equation": {}}],
        },
        {
            "id": "implicit",
            "kind": "implicit",
            "origin": deepcopy(origin),
            "residual": {"const": -0.0},
            "predictor": None,
            "prediction_unavailable_reason": "two branches",
        },
        {
            "id": "failed",
            "kind": "native_result",
            "origin": {"engine": "other", "seed": 7},
            "unavailable_reason": "native failure",
            "native_payload": {"retained": [1, 2]},
        },
    ]
    before = deepcopy(records)
    path = write_hypotheses(tmp_path / "pool.json", records)
    assert records == before
    assert len(read_json(path)["contexts"]) == 2
    assert path.stat().st_size < len(json.dumps(records).encode()) / 2
    restored = read_hypotheses(path)
    assert restored == before
    assert np.signbit(restored[2]["residual"]["const"])
    restored[0]["origin"]["summary"]["ledger"][0] = "changed by caller"
    assert restored[1]["origin"] == origin
    assert read_hypotheses(path) == before


def test_context_numeric_types_and_empty_pool(tmp_path):
    records = [{"id": str(i), "origin": {"value": value}} for i, value in enumerate([True, 1, 1.0, -0.0, 0.0])]
    path = write_hypotheses(tmp_path / "pool.json", records)
    assert len(read_json(path)["contexts"]) == 5
    restored = read_hypotheses(path)
    assert [type(r["origin"]["value"]) for r in restored] == [bool, int, float, float, float]
    assert np.signbit(restored[3]["origin"]["value"])
    assert read_hypotheses(write_hypotheses(tmp_path / "empty.json", [])) == []


@pytest.mark.parametrize("corruption", ["missing", "negative", "boolean", "override", "extra", "unused", "duplicate"])
def test_bad_references_fail_closed(tmp_path, corruption):
    path = write_hypotheses(tmp_path / "pool.json", [{"id": "one", "origin": {"engine": "test"}}])
    payload = read_json(path)
    if corruption in {"missing", "negative", "boolean"}:
        payload["records"][0]["context"] = {"missing": 9, "negative": -1, "boolean": True}[corruption]
    elif corruption == "override":
        payload["records"][0]["record"]["origin"] = {"engine": "wrong"}
    elif corruption == "extra":
        payload["contexts"][0]["equation"] = {}
    elif corruption == "unused":
        payload["contexts"].append({})
    else:
        payload["records"].append(deepcopy(payload["records"][0]))
    write_json(path, payload)
    with pytest.raises(ValueError):
        read_hypotheses(path)


@pytest.mark.parametrize("failure", ["encoding", "flush", "replace"])
def test_interrupted_write_preserves_previous_file_and_allows_retry(tmp_path, monkeypatch, failure):
    from ml_platform_symbolic import data

    old = [{"id": "old", "origin": {"engine": "test"}}]
    new = [{"id": "new", "origin": {"engine": "test"}, "candidate_value": float("nan") if failure == "encoding" else 1}]
    path = write_hypotheses(tmp_path / "pool.json", old)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated interrupted persistence")

    with monkeypatch.context() as patch:
        if failure != "encoding":
            patch.setattr(data.os, "fsync" if failure == "flush" else "replace", fail)
        with pytest.raises((OSError, ValueError)):
            write_hypotheses(path, new)
    assert path.read_bytes() == before
    assert read_hypotheses(path) == old
    assert not list(tmp_path.glob("*.pending"))
    new[0]["candidate_value"] = 1
    write_hypotheses(path, new)
    assert read_hypotheses(path) == new


def test_legacy_and_normalized_studies_predict_and_compare_identically(synthetic_config, tmp_path):
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [
                {"manual": {"id": "quadratic", "expression": "2*x*x+1"}},
                {"manual": {"id": "other", "expression": "2*x+1"}},
            ],
            "engineering_tolerance": 0.01,
        },
        run_dir=tmp_path / "study",
    )
    assert read_json(study / "study.json")["schema_version"] == 3
    context, records, rows, arrays = load_study(study)
    new_comparison = read_json(compare_equations(study, run_dir=tmp_path / "new") / "comparison.json")
    predictions = [predict_working(r, arrays["selection_X"], context["variables"]) for r in records]
    # Only this disposable fixture simulates an old writer. Frozen real studies
    # are never migrated in place or silently assigned a new identity.
    write_json(study / "hypotheses.json", records)
    manifest = read_json(study / "study.json")
    manifest["schema_version"] = 2
    manifest["files"]["hypotheses.json"] = digest(study / "hypotheses.json")
    write_json(study / "study.json", manifest)
    old_context, old_records, old_rows, old_arrays = load_study(study)
    assert (old_context, old_records, old_rows) == (context, records, rows)
    for key in arrays:
        np.testing.assert_equal(old_arrays[key], arrays[key])
    for record, (values, valid) in zip(old_records, predictions, strict=True):
        actual, actual_valid = predict_working(record, old_arrays["selection_X"], old_context["variables"])
        np.testing.assert_array_equal(actual, values)
        np.testing.assert_array_equal(actual_valid, valid)
    old_comparison = read_json(compare_equations(study, run_dir=tmp_path / "old") / "comparison.json")
    for field in ("candidates", "pairs", "families"):
        assert old_comparison[field] == new_comparison[field]


def test_legacy_list_unchanged_and_checksum_still_binds_new_file(tmp_path, synthetic_config):
    path = write_json(tmp_path / "old.json", [{"id": "one", "origin": {"arbitrary": [1, "x"]}}])
    before = path.read_bytes()
    assert read_hypotheses(path) == read_json(path)
    assert path.read_bytes() == before
    study = create_study(
        {"problem": synthetic_config, "sources": [{"manual": {"id": "one", "expression": "x"}}]},
        run_dir=tmp_path / "study",
    )
    payload = read_json(study / "hypotheses.json")
    payload["contexts"][0]["origin"]["knowledge"] = "tampered"
    write_json(study / "hypotheses.json", payload)
    with pytest.raises(ValueError, match="Study evidence changed"):
        load_study(study)
