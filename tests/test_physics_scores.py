"""Scorer oracles test scientific definitions, not a solver's recovery ability."""

import json

import numpy as np
import pytest

from scripts.score_physics_benchmarks import R, _half_power_width, kinetics_audit, resonance_audit


def test_resonance_uses_squared_amplitude_half_power_width():
    frequency = np.linspace(0, 20, 20001)
    amplitude = 1 / np.sqrt((100 - frequency**2) ** 2 + frequency**2)
    width = _half_power_width(frequency, amplitude)
    peak = np.sqrt(100 - 0.5)
    denominator_at_peak = (100 - peak**2) ** 2 + peak**2
    # Analytic roots of (100-w²)²+w²=2*D_peak, not amplitude=peak/2.
    roots = np.roots([1, -199, 10000 - 2 * denominator_at_peak])
    expected = abs(np.sqrt(roots[0]) - np.sqrt(roots[1]))
    assert width == pytest.approx(expected, rel=1e-5)


def test_correct_resonance_and_missing_two_crossings():
    result, _ = resonance_audit(
        lambda frame: (
            0.1
            / np.sqrt(
                (100 - frame.omega.to_numpy() ** 2) ** 2 + frame.gamma.to_numpy() ** 2 * frame.omega.to_numpy() ** 2
            )
        )
    )
    assert all(row["peak_height_relative_error"] < 0.001 for row in result["sweeps"])
    assert all(row["half_power_width_relative_error"] < 1e-12 for row in result["sweeps"])
    flat, _ = resonance_audit(lambda frame: np.ones(len(frame)))
    assert all(row["half_power_width_rad_per_s"] is None for row in flat["sweeps"])


def test_concentration_order_is_fixed_temperature_not_observed_path_effect():
    for order in (0, 1, 2):
        barrier = 48000 / R + (1 - order) * 2000
        result, tables = kinetics_audit(
            lambda frame, order=order, barrier=barrier: (
                0.02
                * frame.concentration_ratio.to_numpy() ** order
                * np.exp(barrier * (1 / 330 - 1 / frame.temperature.to_numpy()))
            )
        )
        np.testing.assert_allclose(tables["response"].order, order, atol=1e-9)
        assert result["fixed_temperature_order_rmse"] == pytest.approx(abs(order - 1), abs=1e-9)
        if order == 1:
            assert result["fixed_concentration_activation_rmse_kJ_per_mol"] < 0.1


def test_scientific_aggregates_do_not_drop_invalid_predictions():
    def hole(frame):
        result = np.ones(len(frame))
        result[0] = np.nan
        return result

    result, tables = kinetics_audit(hole)
    assert result["n_valid"] == result["n_points"] - 1
    assert result["rate_rmse_1_per_s"] is None
    assert not tables["response"].valid.iloc[0]


def test_finite_but_huge_prediction_does_not_create_infinite_science_metric():
    with np.errstate(all="ignore"):
        result, _ = kinetics_audit(lambda frame: np.full(len(frame), 1e308))
        resonance, _ = resonance_audit(lambda frame: np.full(len(frame), 1e308))
    # The RMS itself is representable even though direct squaring overflows.
    assert result["rate_rmse_1_per_s"] == 1e308
    json.dumps(result, allow_nan=False)
    json.dumps(resonance, allow_nan=False)
