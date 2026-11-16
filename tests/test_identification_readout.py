"""Saved evidence stays failure-inclusive and solver repeats stay nested."""

import csv
import json
from statistics import median

import pytest

from scripts.summarize_identification import write_report


def test_saved_readout_preserves_failures_and_nested_design(tmp_path):
    rows = []
    for draw, values in enumerate(([0, 0, 0], [10, None, "invalid"], [20, None, None])):
        for seed, value in enumerate(values):
            rows.append({
                "case": f"saturation_independent_{draw}", "data_seed": draw, "solver_seed": seed,
                "engine": "parfam", "selected_id": "choice" if value is not None else None,
                "eligible_count": int(value is not None),
                "native_status": "partial" if value == "invalid" else "ok" if value is not None else "failed",
                "score_status": "scored" if value is not None else "no_eligible_candidate",
                "test_metrics": {"rmse": value}, "reason": "preserved reason",
                "strata": {"decoupled": {"rmse": value, "coverage": 0.5}},
            })
    source, output = tmp_path / "scores.json", tmp_path / "readout"
    source.write_text(json.dumps({"attempts": 9, "results": rows}))
    original = source.read_bytes()
    write_report(source, output)
    summary = json.loads((output / "summary.json").read_text())["groups"][0]
    metric = summary["metrics"]["test_metrics.rmse"]
    assert metric["median_of_data_draw_solver_medians"] == 10 != median([0, 0, 0, 10, 20])
    assert metric["worst_individual_run"] == 20
    assert (metric["finite_count"], metric["missing_count"], metric["invalid_count"]) == (5, 3, 1)
    assert summary["eligible_attempt_count"] == 6 and summary["attempt_count"] == 9
    assert summary["statuses"]["native_status"] == {"ok": 5, "failed": 3, "partial": 1}
    assert summary["min_observed_coverage"] == 0.5
    with (output / "attempts.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 9 and exported[5]["test_metrics.rmse"] == "invalid"
    assert all(row["reason"] == "preserved reason" for row in exported)
    assert exported[0]["strata.decoupled.coverage"] == "0.5"
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        write_report(source, output)
