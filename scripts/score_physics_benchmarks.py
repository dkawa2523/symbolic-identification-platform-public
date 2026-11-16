"""Post-choice synthetic physics audits. Truth stays here, never in a discovery request."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic.data import digest, write_json
from ml_platform_symbolic.domains import validate_input_domain, within_bounds
from ml_platform_symbolic.expressions import evaluate
from ml_platform_symbolic.metrics import prediction_metrics
from ml_platform_symbolic.model import load_model

R = 8.31446261815324


def _rmse(delta):
    return prediction_metrics(np.zeros(len(delta)), delta)["rmse"]


def _half_power_width(frequency, amplitude):
    """Nearest two crossings of squared amplitude at half its sampled peak power."""
    peak = int(np.argmax(amplitude))
    delta = (amplitude / amplitude[peak]) ** 2 - 0.5
    left = np.flatnonzero((delta[:-1] <= 0) & (delta[1:] > 0))
    right = np.flatnonzero((delta[:-1] > 0) & (delta[1:] <= 0))
    left, right = left[left < peak], right[right >= peak]
    if not len(left) or not len(right) or peak in {0, len(frequency) - 1}:
        return None

    def crossing(i):
        return float(frequency[i] - delta[i] * (frequency[i + 1] - frequency[i]) / (delta[i + 1] - delta[i]))

    return crossing(int(right[0])) - crossing(int(left[-1]))


def resonance_audit(predict):
    frequency = np.linspace(3, 18, 1501)
    sweeps, tables = [], []
    for damping in (0.2, 0.3, 0.6, 1.2, 2.0):
        frame = pd.DataFrame({"omega": frequency, "gamma": damping})
        values = np.asarray(predict(frame), dtype=float)
        truth = 0.1 / np.sqrt((100 - frequency**2) ** 2 + damping**2 * frequency**2)
        valid = np.isfinite(values) & (values > 0)
        row = {
            "gamma_1_per_s": damping,
            "n_points": len(frame),
            "n_valid_positive": int(valid.sum()),
            "peak_frequency_rad_per_s": None,
            "peak_location_relative_error": None,
            "peak_height_m": None,
            "peak_height_relative_error": None,
            "half_power_width_rad_per_s": None,
            "half_power_width_relative_error": None,
        }
        if valid.all():
            peak = int(np.argmax(values))
            truth_peak = np.sqrt(100 - damping**2 / 2)
            truth_height = 0.1 / np.sqrt((100 - truth_peak**2) ** 2 + damping**2 * truth_peak**2)
            width, truth_width = _half_power_width(frequency, values), _half_power_width(frequency, truth)
            # A boundary maximum is not evidence of the interior resonance peak.
            row.update(
                peak_frequency_rad_per_s=float(frequency[peak]),
                peak_height_m=float(values[peak]),
                interior_peak=peak not in {0, len(frame) - 1},
                peak_location_relative_error=float(abs(frequency[peak] - truth_peak) / truth_peak),
                peak_height_relative_error=float(abs(values[peak] - truth_height) / truth_height)
                if np.isfinite(abs(values[peak] - truth_height) / truth_height)
                else None,
                half_power_width_rad_per_s=width,
                half_power_width_relative_error=abs(width - truth_width) / truth_width if width is not None else None,
            )
        sweeps.append(row)
        tables.append(frame.assign(prediction=values, oracle=truth, valid_positive=valid))
    tail = pd.DataFrame({"omega": np.linspace(24, 40, 65), "gamma": 1.2})
    tail_values = np.asarray(predict(tail), dtype=float)
    tail_truth = 0.1 / np.sqrt(
        (100 - tail.omega.to_numpy() ** 2) ** 2 + tail.gamma.to_numpy() ** 2 * tail.omega.to_numpy() ** 2
    )
    valid = np.isfinite(tail_values) & (tail_values > 0)
    return {
        "case": "damped_resonance",
        "kind": "source-grounded synthetic audit, not equipment measurements",
        "sweeps": sweeps,
        "tail": {
            "n_points": len(tail),
            "n_valid_positive": int(valid.sum()),
            "scaled_tail_rmse_m_per_s2": _rmse(tail.omega.to_numpy() ** 2 * (tail_values - tail_truth))
            if valid.all()
            else None,
            "interpretation": "finite omega24..40 comparison; not proof that the asymptotic limit has been reached",
        },
        "scope": "fixed post-choice oracle grids; no discovery, selection, refit or global-physics guarantee",
    }, {
        "sweeps": pd.concat(tables, ignore_index=True),
        "tail": tail.assign(prediction=tail_values, oracle=tail_truth, valid_positive=valid),
    }


def kinetics_audit(predict):
    T, q = np.meshgrid(np.linspace(280, 390, 23), np.linspace(0.55, 1.75, 17), indexing="ij")
    frame = pd.DataFrame({"temperature": T.ravel(), "concentration_ratio": q.ravel()})
    center = np.asarray(predict(frame), dtype=float)
    temperature_step, concentration_step = 0.1, 0.001
    temperatures, concentrations = [], []
    for sign in (-1, 1):
        high = frame.copy()
        high.temperature += sign * temperature_step
        # Endpoint derivatives are one-sided to stay inside the declared domain.
        high.temperature = high.temperature.clip(280, 390)
        temperatures.append((high.temperature.to_numpy(), np.asarray(predict(high), dtype=float)))
        high = frame.copy()
        high.concentration_ratio += sign * concentration_step
        concentrations.append(np.asarray(predict(high), dtype=float))
    with np.errstate(all="ignore"):
        dT = (temperatures[1][1] - temperatures[0][1]) / (temperatures[1][0] - temperatures[0][0])
        dq = (concentrations[1] - concentrations[0]) / (2 * concentration_step)
        order = frame.concentration_ratio.to_numpy() * dq / center
        activation = R * frame.temperature.to_numpy() ** 2 * dT / center / 1000
    valid = np.isfinite(center) & (center > 0) & np.isfinite(order) & np.isfinite(activation)
    valid &= np.isfinite(temperatures[0][1]) & np.isfinite(temperatures[1][1])
    valid &= np.isfinite(concentrations[0]) & np.isfinite(concentrations[1])
    truth = (
        0.02 * frame.concentration_ratio.to_numpy() * np.exp((48000 / R) * (1 / 330 - 1 / frame.temperature.to_numpy()))
    )
    result = {
        "case": "activated_kinetics",
        "kind": "source-grounded synthetic audit, not equipment measurements",
        "n_points": len(frame),
        "n_valid": int(valid.sum()),
        "rate_rmse_1_per_s": _rmse(center - truth) if valid.all() else None,
        "fixed_temperature_order_rmse": _rmse(order - 1) if valid.all() else None,
        "fixed_concentration_activation_rmse_kJ_per_mol": _rmse(activation - 48) if valid.all() else None,
        "derivative_method": "original-unit finite steps dT=0.1K,dq=0.001;one-sided at temperature bounds",
        "interpretation": "fixed-other-input model response, not derivative along the confounded observed path or proof of mechanism",
        "scope": "all-or-unavailable aggregates; invalid rows are retained; no new fit or selection",
    }
    return result, {
        "response": frame.assign(
            prediction=center, oracle=truth, order=order, activation_kJ_per_mol=activation, valid=valid
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["damped_resonance", "activated_kinetics"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.bundle)
    if not model.metadata.get("frozen_model_sha256"):
        raise ValueError("Audit requires an explicitly frozen selected bundle; do not use truth grids for selection")
    expected = {
        "damped_resonance": (["omega", "gamma"], ["rad/s", "1/s"], "m"),
        "activated_kinetics": (["temperature", "concentration_ratio"], ["K", "1"], "1/s"),
    }[args.case]
    if (
        [v["column"] for v in model.schema["variables"]],
        [v["unit"] for v in model.schema["variables"]],
        model.schema["target"]["unit"],
    ) != expected:
        raise ValueError("Model feature/unit meanings differ; no implicit conversion")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    raw = []

    def predict(frame):
        variables = model.schema["variables"]
        X = frame[[v["column"] for v in variables]].to_numpy(dtype=float)
        validate_input_domain(X, variables)
        values = evaluate(model.equation, X, variables)
        valid = np.isfinite(values) & within_bounds(values, model.schema["target"].get("domain"))
        raw.append(frame.assign(evaluation_call=len(raw), raw_prediction=values, valid_domain=valid))
        return np.where(valid, values, np.nan)

    try:
        result, tables = (resonance_audit if args.case == "damped_resonance" else kinetics_audit)(predict)
    except ValueError as exc:
        result, tables = {"case": args.case, "status": "not_evaluable", "reason": str(exc)}, {}
    result.update(bundle_sha256=digest(args.bundle), scorer_sha256=digest(Path(__file__)))
    write_json(args.run_dir / "scientific_metrics.json", result)
    if raw:
        pd.concat(raw, ignore_index=True).to_csv(args.run_dir / "runtime_predictions.csv", index=False)
    for name, table in tables.items():
        table.to_csv(args.run_dir / (name + ".csv"), index=False)
    print(args.run_dir)


if __name__ == "__main__":
    main()
