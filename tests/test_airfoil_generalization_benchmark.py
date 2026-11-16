"""Small synthetic boundary checks; public performance data stay outside fast tests."""

import numpy as np
import pandas as pd
import pytest

from scripts import benchmark_airfoil_generalization as audit


def fixtures():
    rows = []
    for g in range(20):
        for f in [800, 1000]:
            rows.append({"sample_id": f"g{g}-{f}", "group_id": f"g{g}", "frequency": f,
                         "attack_angle": g, "chord": 0.3, "velocity": 71.3 if g >= 15 else 55.5,
                         "displacement_thickness": 0.002, "sound_level": 100 + g})
    frame = pd.DataFrame(rows)
    split = frame[["sample_id"]].copy()
    split["partition"] = np.repeat(["train"] * 12 + ["selection"] * 4 + ["final"] * 4, 2)
    return frame, split


def test_groups_and_high_velocity_are_held_out_without_target_decisions():
    frame, split = fixtures()
    cases = audit.split_cases(frame, split)
    changed = audit.split_cases(frame.assign(sound_level=np.arange(len(frame)) * -999), split)
    for name, rows in cases.items():
        assert rows.groupby(audit.GROUP_FEATURES).partition.nunique().eq(1).all()
        pd.testing.assert_series_equal(rows.partition, changed[name].partition)
    held = cases["velocity_71p3_holdout"]
    assert (held.partition.eq("final") == held.velocity.eq(71.3)).all()
    assert held.loc[held.partition != "final", "velocity"].max() < 71.3


def test_physical_tuple_leakage_rejected_even_when_group_ids_differ():
    frame, split = fixtures()
    rows = audit.split_cases(frame, split)["group_split_seed42"]
    rows.loc[rows.partition == "selection", audit.GROUP_FEATURES] = rows.loc[0, audit.GROUP_FEATURES].to_numpy()
    with pytest.raises(ValueError, match="physical condition"):
        audit.validate_groups(rows)


def test_scoring_respects_frozen_angle_coordinates(monkeypatch):
    context = {"variables": [{"id": "x0", "column": "attack_angle", "unit": "rad"}],
               "target": {"column": "sound_level", "unit": "dB"}}
    provenance = {**context, "config": {"data": {"units": {"attack_angle": "deg", "sound_level": "dB"}}},
                  "unit_coordinates": {"columns": {
                      "attack_angle": {"source_unit": "deg", "problem_unit": "rad",
                                       "scale": np.pi / 180, "offset": 0},
                      "sound_level": {"source_unit": "dB", "problem_unit": "dB", "scale": 1, "offset": 0}}}}
    monkeypatch.setattr(audit, "predict_record", lambda record, X, variables, target: (X[:, 0], np.ones(len(X), bool)))
    prediction, valid = audit.predict_public({}, pd.DataFrame({"attack_angle": [90.0]}), context, provenance)
    np.testing.assert_allclose(prediction, [np.pi / 2])
    assert valid.all()


@pytest.mark.parametrize("filename", ["choices.json", "scores.json", "test_access.json"])
def test_no_reselection_after_choices_or_any_test_access(tmp_path, filename):
    (tmp_path / filename).write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        audit.select_campaign(tmp_path)
