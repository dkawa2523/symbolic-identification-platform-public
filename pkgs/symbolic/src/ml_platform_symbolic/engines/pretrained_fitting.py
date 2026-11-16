"""Explicit hybrid coefficient starts, sharing a declared solver allocation."""

import time
from copy import deepcopy

import numpy as np

from ..expressions import evaluate
from ..metrics import prediction_metrics
from ..parameters import free_parameter_bounds, free_parameter_names, with_parameters


def _finite_diagnostics(value):
    """Encode nonfinite diagnostics as null without treating a failed fit as valid."""
    if isinstance(value, dict):
        pairs = {key: _finite_diagnostics(item) for key, item in value.items()}
        return {key: pair[0] for key, pair in pairs.items()}, any(pair[1] for pair in pairs.values())
    if isinstance(value, (list, tuple)):
        pairs = [_finite_diagnostics(item) for item in value]
        return [pair[0] for pair in pairs], any(pair[1] for pair in pairs)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None, True
    return value, False


def fit_starts(equation, request, options, combination, deadline, diagnostics, fitter, constraints, data_scale):
    count = options["fit_starts"]
    if count == 1:
        return fitter(
            equation,
            request.X,
            request.y,
            request.variables,
            max_nfev=options["max_nfev"],
            diagnostics=diagnostics,
            constraints=constraints,
            data_scale=data_scale,
        )
    names = sorted(free_parameter_names(equation))
    if any(p.get("tied_to") is not None for p in equation["parameters"].values()):
        raise ValueError("Hybrid multiple starts do not support tied_to aliases")
    count = count if names else 1
    initial = np.asarray([equation["parameters"][k]["value"] for k in names])
    bounds = free_parameter_bounds(equation, names)
    rng = np.random.default_rng(np.random.SeedSequence([request.seed, combination, 31871]))
    records, best, score = [], None, np.inf
    diagnostics.update(
        starts=records,
        requested_starts=options["fit_starts"],
        seed_stream=[request.seed, combination, 31871],
        allocation=options["max_nfev"],
        budget_semantics="total solver allocation per combination split across starts; SciPy nfev/iterations are not a hard prediction-call cap; wall checked between starts",
        selection="minimum finite feasible Train RMSE; Selection/Final unused",
    )
    for index in range(count):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Hybrid coefficient starts exceeded wall budget")
        values = initial.copy()
        if index:
            with np.errstate(over="ignore", invalid="ignore"):
                values += rng.normal(size=len(names)) * np.maximum(abs(initial), 1.0)
            # Never turn overflow into a finite boundary by clipping.
            for i, (low, high) in enumerate(bounds):
                if not np.isfinite(values[i]):
                    continue
                if low is not None:
                    values[i] = max(values[i], low)
                if high is not None:
                    values[i] = min(values[i], high)
        allocation = options["max_nfev"] // count + (index < options["max_nfev"] % count)
        record = {
            "start": index,
            "initial_parameters": {
                name: float(value) if np.isfinite(value) else None for name, value in zip(names, values, strict=True)
            },
            "max_nfev": allocation,
            "fit": {},
        }
        records.append(record)
        try:
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite proposed coefficient start; no fit or clipping performed")
            started = with_parameters(equation, names, values) if names else deepcopy(equation)
            fitted = fitter(
                started,
                request.X,
                request.y,
                request.variables,
                max_nfev=allocation,
                diagnostics=record["fit"],
                constraints=constraints,
                data_scale=data_scale,
            )
            record["fit"], invalid_diagnostics = _finite_diagnostics(record["fit"])
            if invalid_diagnostics or any(not np.isfinite(p["value"]) for p in fitted["parameters"].values()):
                raise ValueError("Nonfinite fitted parameter or solver diagnostic")
            prediction = evaluate(fitted, request.X, request.variables)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite complete Train prediction")
            if constraints is not None:
                callback, low, high = constraints
                value = np.asarray(callback(fitted))
                if not np.isfinite(value).all() or not np.all(value >= low) or not np.all(value <= high):
                    raise ValueError("Original response conditions not satisfied")
            metrics = prediction_metrics(request.y, prediction)
            rmse = metrics["rmse"]
            if rmse is None or not np.isfinite(rmse):
                raise ValueError("Invalid Train RMSE: " + str(metrics["reason"]))
            record.update(status="candidate", train_rmse=rmse)
            if np.isfinite(rmse) and rmse < score:
                best, score = fitted, rmse
                diagnostics["selected_start"] = index
        except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
            record["fit"], _ = _finite_diagnostics(record["fit"])
            record.update(status="failed", reason=str(exc), train_rmse=None)
    if best is None:
        raise ValueError("No finite feasible coefficient start")
    return best
