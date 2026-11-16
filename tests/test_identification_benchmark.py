"""Proportional campaign boundary checks; native performance is not a fast test."""

import json

import pandas as pd
import pytest

from scripts import benchmark_identification as campaign
from scripts.benchmark_identification import select_campaign, synthetic


def test_control_keeps_replicates_and_data_draws_distinct():
    train, final = synthetic(1729, True)
    independent, _ = synthetic(1729, False)
    assert train.groupby("group").partition.nunique().eq(1).all()
    assert set(train.sample_id).isdisjoint(final.sample_id)
    assert set(train.group).isdisjoint(final.group)
    assert train.x.equals(train.z)
    assert not independent.x.equals(independent.z)
    # Both arms deliberately pair the same x and noise draws, not solver seeds.
    pd.testing.assert_series_equal(train.x, independent.x)
    assert "truth" not in train and "latent_x" not in train
    assert set(final.stratum) == {"same_design", "decoupled", "extrapolation"}
    another, _ = synthetic(1730, True)
    assert not train.x.equals(another.x)


@pytest.mark.parametrize("filename", ["choices.json", "scores.json"])
def test_choice_cannot_be_replaced_after_freeze_or_score(tmp_path, filename):
    (tmp_path / filename).write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        select_campaign(tmp_path)


def test_protocol_rejects_source_or_runtime_drift_before_discovery(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"source_sha256_at_prepare": "frozen", "runtime": {"image": "one"}}))
    monkeypatch.setattr(campaign, "source_digest", lambda: "changed")
    monkeypatch.setattr(campaign, "runtime_identity", lambda: {"image": "one"})
    with pytest.raises(ValueError, match="Source/runtime changed"):
        campaign.verify_protocol(tmp_path)
    monkeypatch.setattr(campaign, "source_digest", lambda: "frozen")
    monkeypatch.setattr(campaign, "runtime_identity", lambda: {"image": "two"})
    with pytest.raises(ValueError, match="Source/runtime changed"):
        campaign.verify_protocol(tmp_path)
