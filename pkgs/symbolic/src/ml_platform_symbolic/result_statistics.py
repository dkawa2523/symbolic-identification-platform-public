"""Aggregate once, store denominators, then render without recomputing decisions."""

import math
import statistics


def summarize_seed_metrics(records, *, expected_seeds, unit):
    """Records are already selected/audited outputs, never new model evaluations."""
    expected = set(expected_seeds)
    rows = {record["seed"]: record for record in records}
    if len(rows) != len(records) or set(rows) - expected:
        raise ValueError("One comparable result per declared seed is required")
    valid, invalid, unscored = [], [], []
    for seed in sorted(expected):
        row = rows.get(seed)
        if row is None or not row.get("scored", False):
            unscored.append(seed)
        elif (
            row["n_rows"] > 0
            and row["n_valid"] == row["n_rows"]
            and row.get("rmse") is not None
            and math.isfinite(row["rmse"])
        ):
            valid.append(seed)
        else:
            invalid.append(seed)
    values = [rows[seed]["rmse"] for seed in valid]
    eligible = (
        sorted(seed for seed, row in rows.items() if row["selection_eligible"])
        if all("selection_eligible" in row for row in rows.values())
        else None
    )
    return {
        "unit": unit,
        "expected_seeds": sorted(expected),
        "valid_seeds": valid,
        "invalid_seeds": invalid,
        "unscored_seeds": unscored,
        "n_valid": len(valid),
        "selection_eligible_seeds": eligible,
        "n_selection_eligible": len(eligible) if eligible is not None else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "scope": "Conditional on full-row valid finite metrics. Missing/invalid seeds are not successes or zero scores.",
    }
