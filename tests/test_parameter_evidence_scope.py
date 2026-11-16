"""Expensive coefficient diagnostics are explicit and cannot change prediction decisions."""
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic import compare_equations, create_study
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.equation_evaluation import api, numerical
from ml_platform_symbolic.report_data import build_report_view
from ml_platform_symbolic.selection import choose_hypothesis, select_representatives


def study_plan(config):
    return {
        "problem": config,
        "engineering_tolerance": 0.01,
        "sources": [
            {"manual": {"id": "fit", "expression": "a*x*x+1", "parameters": {
                "a": {"value": 2., "trainable": True, "unit": "1"},
            }}},
            {"manual": {"id": "fixed", "expression": "2*x+1"}},
            {"manual": {"id": "invalid", "expression": "sqrt(x-2)"}},
        ],
    }


def prediction_rows(rows):
    return [{k: v for k, v in row.items() if k != "local_parameter_identifiability"} for row in rows]


def test_optional_diagnostic_preserves_metrics_masks_selection_and_old_report(synthetic_config, tmp_path, monkeypatch):
    plan = study_plan(synthetic_config)
    original = deepcopy(plan)
    calls, partitions = [], []
    original_evidence, original_load = numerical.local_parameter_evidence, api.load_prepared

    def diagnose(record, *args):
        calls.append(record["id"])
        return original_evidence(record, *args)

    def load(refs, provenance, partition):
        assert partition in {"train", "selection"}
        partitions.append(partition)
        return original_load(refs, provenance, partition)

    monkeypatch.setattr(numerical, "local_parameter_evidence", diagnose)
    monkeypatch.setattr(api, "load_prepared", load)
    skipped = create_study(plan, run_dir=tmp_path / "skipped")
    assert plan == original and calls == []
    context, records, rows, arrays = api.load_study(skipped)
    assert all(row["local_parameter_identifiability"]["status"] == "not_computed" for row in rows)
    assert not rows[2]["eligible"]  # no invalid row exclusion or diagnostic-based eligibility
    assert not arrays["c2_train_valid"].all()
    original_records = deepcopy(records)
    replay = {k: v for k, v in plan.items() if k != "problem"}
    replay["prepared_run"] = context["prepared"]["run"]
    replay["parameter_evidence_ids"] = [r["id"] for r in records]
    requested = create_study(replay, run_dir=tmp_path / "requested")
    _, replay_records, replay_rows, replay_arrays = api.load_study(requested)
    assert calls == ["fit@manual", "fixed@manual"]
    assert records == original_records == replay_records
    assert prediction_rows(rows) == prediction_rows(replay_rows)
    assert arrays.keys() == replay_arrays.keys()
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], replay_arrays[key])
    assert [r["local_parameter_identifiability"]["status"] for r in replay_rows] == [
        "computed", "not_applicable", "unavailable",
    ]
    assert partitions == ["train", "selection"] * 2
    assert context["parameter_evidence_ids"] == []
    decisions = []
    comparisons = []
    for index, (study, source, evidence) in enumerate(((skipped, records, rows), (requested, replay_records, replay_rows))):
        representative = select_representatives(source, evidence)[("manual", None)][0]["id"]
        assert representative == "fit@manual"
        comparison = compare_equations(study, run_dir=tmp_path / f"compare{index}")
        comparisons.append(comparison)
        choice = choose_hypothesis(comparison, run_dir=tmp_path / f"choice{index}", candidate_id=representative,
                                   reason="Same frozen Selection RMSE rule")
        decision = read_json(choice / "decision.json")
        decisions.append((decision["selected_id"], decision["status"]))
    assert decisions == [("fit@manual", "adopt")] * 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Reports must read saved evidence, never recompute")

    monkeypatch.setattr(numerical, "local_parameter_evidence", forbidden)
    for comparison, saved in zip(comparisons, (rows, replay_rows), strict=True):
        view = build_report_view(comparison)
        assert [r["local_parameter_identifiability"] for r in view["candidates"]] == [
            r["local_parameter_identifiability"] for r in saved
        ]


def test_only_explicit_ids_receive_diagnostic(synthetic_config, tmp_path, monkeypatch):
    plan = study_plan(synthetic_config)
    plan["parameter_evidence_ids"] = ["fit@manual"]
    called = []
    original = numerical.local_parameter_evidence

    def counted(record, *args):
        called.append(record["id"])
        return original(record, *args)

    monkeypatch.setattr(numerical, "local_parameter_evidence", counted)
    study = create_study(plan, run_dir=tmp_path / "study")
    context, _, rows, _ = api.load_study(study)
    assert called == context["parameter_evidence_ids"] == ["fit@manual"]
    assert [row["local_parameter_identifiability"]["status"] for row in rows] == [
        "computed", "not_computed", "not_computed",
    ]


@pytest.mark.parametrize("ids", [None, True, "fit@manual", {}, [None], [[]], [""], [" "], ["fit@manual"] * 2,
                                  ["missing@manual"]])
def test_rejects_ambiguous_or_unknown_diagnostic_request(synthetic_config, tmp_path, ids):
    plan = study_plan(synthetic_config)
    plan["parameter_evidence_ids"] = ids
    with pytest.raises(ValueError, match="parameter_evidence_ids"):
        create_study(plan, run_dir=tmp_path / "invalid")


def test_requested_native_failure_does_not_become_an_equation(synthetic_config, tmp_path):
    study = create_study(study_plan(synthetic_config), run_dir=tmp_path / "study")
    context, _, _, arrays = api.load_study(study)
    failed = {"id": "engine/42@native_result", "kind": "native_result", "unavailable_reason": "native failed"}
    context["parameter_evidence_ids"] = [failed["id"]]
    row = numerical._numerical(context, [failed], arrays)[0]
    assert not row["prediction_available"] and not row["eligible"]
    assert row["local_parameter_identifiability"]["status"] == "unavailable"
    assert "metrics" not in row
