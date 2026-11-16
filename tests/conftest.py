import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_config(tmp_path):
    x = np.linspace(0.5, 3, 60)
    frame = pd.DataFrame(
        {"sample_id": [f"r{i}" for i in range(60)], "group": np.repeat(np.arange(20), 3), "x": x, "y": 2 * x * x + 1}
    )
    labels = np.repeat(["train"] * 12 + ["selection"] * 4 + ["final"] * 4, 3)
    table, split = tmp_path / "table.csv", tmp_path / "split.csv"
    frame.to_csv(table, index=False)
    frame[["sample_id"]].assign(partition=labels).to_csv(split, index=False)
    return {
        "data": {
            "local_path": str(table),
            "target_column": "y",
            "feature_columns": ["x"],
            "units": {"x": "1", "y": "1"},
        },
        "split": {"file": str(split), "group_column": "group"},
        "search": {"engines": ["dictionary"], "max_wall_seconds": 20},
        "evaluation": {"selection_policy": "accuracy"},
        "constraints": {"max_nodes": 30},
    }
