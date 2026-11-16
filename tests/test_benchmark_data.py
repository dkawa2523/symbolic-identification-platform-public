"""Search seeds must not silently change a fixed benchmark's data or split."""

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.constraints import dimensions
from ml_platform_symbolic.expressions import evaluate, parse_expression

from scripts.benchmark_symbolic import physics_options, synthetic
from scripts.prepare_constraint_matrix import response_arm


def test_two_input_noise_data_are_independent_of_method_seed(tmp_path):
    configs = []
    for seed in (42, 44):
        root = tmp_path / str(seed)
        root.mkdir()
        configs.append(synthetic(root, "nonlinear_noisy_ood", seed, data_seed=940117))
    for name in ("table.csv", "split.csv", "scorer_only.csv"):
        assert (tmp_path / "42" / name).read_bytes() == (tmp_path / "44" / name).read_bytes()
    assert configs[0]["data"]["domain"] == configs[1]["data"]["domain"]


@pytest.mark.parametrize(
    "case", ["polynomial", "noisy_extrapolation", "rational_constraints", "damped_resonance", "activated_kinetics"]
)
def test_fixed_data_and_grouped_replications(case, tmp_path):
    for seed in (0, 2):
        root = tmp_path / str(seed)
        root.mkdir()
        cfg = synthetic(root, case, seed, data_seed=2026102601)
        frame = pd.read_csv(root / "table.csv")
        split = pd.read_csv(root / "split.csv")
        if cfg["split"].get("group_column"):
            merged = frame.merge(split, on="sample_id", validate="one_to_one")
            assert merged.groupby(cfg["split"]["group_column"]).partition.nunique().max() == 1
    for name in ("table.csv", "split.csv"):
        assert (tmp_path / "0" / name).read_bytes() == (tmp_path / "2" / name).read_bytes()


def test_kinetic_oracles_are_observationally_equal_but_interventions_differ(tmp_path):
    synthetic(tmp_path, "activated_kinetics", 0, data_seed=2026102601)
    frame = pd.read_csv(tmp_path / "table.csv").merge(pd.read_csv(tmp_path / "scorer_only.csv"), on="sample_id")
    delta = 1 / 330 - 1 / frame.temperature.to_numpy()
    beta = 48000 / 8.31446261815324
    values = [
        0.02 * frame.concentration_ratio.to_numpy() ** n * np.exp((beta + (1 - n) * 2000) * delta) for n in (0, 1, 2)
    ]
    obs = frame.region == "observational"
    np.testing.assert_allclose(values[0][obs], values[1][obs], rtol=1e-12)
    np.testing.assert_allclose(values[2][obs], values[1][obs], rtol=1e-12)
    assert np.max(abs(values[0][~obs] - values[1][~obs])) > 0.01
    assert (frame.rate > 0).all()


@pytest.mark.parametrize(
    "case,variables,target",
    [
        ("damped_resonance", [{"id": "x0", "unit": "rad/s"}, {"id": "x1", "unit": "1/s"}], {"unit": "m"}),
        ("activated_kinetics", [{"id": "x0", "unit": "K"}, {"id": "x1", "unit": "1"}], {"unit": "1/s"}),
    ],
)
def test_known_family_references_keep_physical_dimensions(case, variables, target):
    eq = physics_options(case, ["known_form"])["search"]["engine_options"]["known_form"]["equation"]
    assert dimensions(eq, variables, target)[0] == "pass"


def test_kinetic_order_is_not_imposed_by_dimensional_constraints():
    variables = [
        {"id": "x0", "column": "temperature", "unit": "K"},
        {"id": "x1", "column": "concentration_ratio", "unit": "1"},
    ]
    T = np.linspace(300, 360, 10)
    q = np.exp(2000 * (1 / 330 - 1 / T))
    values = []
    for n in (0, 1, 2):
        parameters = {
            "rate_ref": {"value": 0.02, "unit": "1/s", "trainable": True, "bounds": [None, None]},
            "barrier": {
                "value": 48000 / 8.31446261815324 + (1 - n) * 2000,
                "unit": "K",
                "trainable": True,
                "bounds": [None, None],
            },
            "t_ref": {"value": 330.0, "unit": "K", "trainable": False, "bounds": [330.0, 330.0]},
        }
        equation = parse_expression(
            f"rate_ref*(x1**{n})*exp(barrier*(1/t_ref-1/x0))", variables, trainable=False, parameters=parameters
        )
        assert dimensions(equation, variables, {"unit": "1/s"})[0] == "pass"
        values.append(evaluate(equation, np.column_stack([T, q]), variables))
    np.testing.assert_allclose(values[0], values[1], rtol=1e-12)
    np.testing.assert_allclose(values[2], values[1], rtol=1e-12)


def test_response_arm_does_not_encode_kinetic_reaction_order_or_hidden_law(tmp_path):
    config = synthetic(tmp_path, "activated_kinetics", 0, data_seed=2026102601)
    frame = pd.read_csv(tmp_path / "table.csv").merge(pd.read_csv(tmp_path / "split.csv"), on="sample_id")
    arm = response_arm("activated_kinetics", config, frame.loc[frame.partition == "train"])
    assert [question["feature"] for question in arm["response_checks"]] == ["temperature"]
    assert arm["response_bounds"] == [0, None]
    assert "fixed_constants" not in arm
    assert all(question["order"] == 1 for question in arm["response_checks"])
