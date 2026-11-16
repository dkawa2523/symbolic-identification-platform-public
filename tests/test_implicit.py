"""A retained implicit result is useful without pretending it is a unique predictor."""

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import load_relation
from ml_platform_symbolic.data import write_json


def test_ambiguous_relation_roundtrip_uses_observed_target_only_for_residual(tmp_path):
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0, 4]}]
    residual = {
        "schema_version": 1,
        "parameters": {},
        "ast": {
            "op": "sub",
            "args": [
                {"op": "pow_int", "args": [{"var": "target"}, {"const": 2}]},
                {"var": "x0"},
            ],
        },
    }
    member = {
        "candidate_id": "manual/42/1",
        "residual": residual,
        "predictor": None,
        "prediction_unavailable_reason": "Both target branches permitted",
    }
    path = write_json(
        tmp_path / "relations.json",
        {
            "schema_version": 1,
            "kind": "implicit_relations",
            "schema": {
                "variables": variables,
                "target": {"column": "y", "unit": "1"},
                "feature_ranges": [[0, 4]],
            },
            "metadata": {"profile": "no_retrieval"},
            "members": [member],
        },
    )
    relation = load_relation(path, "manual/42/1")
    frame = pd.DataFrame({"y": [-1.0, 2.0], "unrelated": [8, 9], "x": [1.0, 4.0]})
    np.testing.assert_array_equal(relation.residual(frame), [0, 0])
    np.testing.assert_array_equal(relation.residual(frame, [0.0, 0.0]), [-1, -4])
    with pytest.raises(ValueError, match="Both target branches"):
        relation.predict(frame.drop(columns="y"))
    frame.loc[0, "x"] = -1
    with pytest.raises(ValueError, match="declared domain"):
        relation.residual(frame)

