from copy import deepcopy

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, least_squares, minimize

from .expressions import evaluate, inspect_equation
from .numerical_fidelity import signal_scale
from .parameters import free_parameter_bounds, free_parameter_names, with_parameters
from .response_search import inward_response_bounds


def fit_coefficients(
    equation, X, y, variables, *, max_nfev=1000, diagnostics=None,
    extra_residuals=None, data_scale=None, constraints=None, iterate_acceptance=None,
):
    """Fit free parameters, optionally with explicit additional objective residuals.

    An explicit data_scale uses mean-square normalization; omitted arguments
    preserve the historical least-squares objective and stopping coordinates.
    Callbacks receive immutable fitted variants, never Selection/Final.
    constraints=(signed_values, lower, upper) enables explicit SLSQP constraints.
    iterate_acceptance(equation, stage) must return exactly True to retain an
    improving finite-point-feasible iterate and the final return (or fixed model).
    It does NOT alter the solver residual/Jacobian, and may reject all visited
    iterates. None preserves historical behavior. Only explicit SLSQP is supported.
    """
    # This hook admits retained iterates, not the smooth objective or proposal generator.
    # Keeping it SLSQP-only avoids silently changing historical least_squares.
    if iterate_acceptance is not None and constraints is None:
        raise NotImplementedError("Iterate acceptance requires explicit SLSQP constraints")
    if constraints is not None and extra_residuals is not None:
        raise ValueError("Explicit constraints and soft residuals cannot be combined")
    if data_scale is not None and (
        isinstance(data_scale, bool) or not np.isfinite(data_scale) or data_scale <= 0
    ):
        raise ValueError("data_scale must be positive and finite")
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            prediction_evaluations=0, max_nfev=max_nfev, free_parameters=[]
        )
    result = deepcopy(equation)
    inspect_equation(result, variables)
    keys = sorted(free_parameter_names(result))
    if diagnostics is not None:
        diagnostics["free_parameters"] = keys
    if not keys:
        if constraints is not None:
            callback, lower, upper = constraints
            value = np.asarray(callback(result))
            if (not np.isfinite(value).all() or not np.all(value >= lower)
                    or not np.all(value <= upper)):
                raise ValueError("No feasible fixed-parameter equation; bounds unchanged")
        if iterate_acceptance is not None and iterate_acceptance(result, "fixed") is not True:
            raise ValueError("Fixed equation failed iterate admission; no unverified fallback")
        if diagnostics is not None:
            diagnostics["termination"] = "no_free_parameters"
        return result
    initial = np.asarray([result["parameters"][k]["value"] for k in keys], dtype=float)
    limits = free_parameter_bounds(result, keys)
    lower = np.asarray([lo if lo is not None else -np.inf for lo, _ in limits])
    upper = np.asarray([hi if hi is not None else np.inf for _, hi in limits])
    y = np.asarray(y, dtype=float)

    def prediction(values):
        if diagnostics is not None:
            diagnostics["prediction_evaluations"] += 1
        current = with_parameters(result, keys, values)
        predicted = evaluate(current, X, variables)
        if predicted.shape != y.shape or not np.isfinite(predicted).all():
            raise ValueError(
                "Coefficient fit entered an invalid real domain or observation shape"
            )
        return predicted

    baseline = prediction(initial)
    scale = signal_scale(y)
    if data_scale is not None:
        scale = float(data_scale)
    elif scale == 0:
        scale = float(np.max(abs(y))) or float(np.max(abs(baseline))) or 1.0
    mean_factor = np.sqrt(len(y)) if data_scale is not None else 1.0

    # The stopping conditions must not depend on the target's physical unit.
    # Constant/zero targets still use a nonzero scale without an SI-unit floor.
    def residual(values):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            errors = (prediction(values) - y) / scale
        if not np.isfinite(errors).all():
            raise ValueError(
                "Coefficient fit residuals exceed the representable scaled coordinate"
            )
        errors = errors / mean_factor
        if extra_residuals is not None:
            extra = np.asarray(extra_residuals(with_parameters(result, keys, values)), dtype=float)
            if extra.ndim != 1 or not np.isfinite(extra).all():
                raise ValueError("Additional coefficient residuals must be a finite vector; no rows dropped")
            errors = np.concatenate([errors, extra])
        return errors

    initial_residual = residual(initial) if extra_residuals is not None else None

    # Normalize each free coordinate by its local effect on the normalized
    # residual. x_scale alone does not normalize SciPy's gradient stopping test.
    scales = np.maximum(abs(initial), 1.0)
    for index, value in enumerate(initial):
        step = np.sqrt(np.finfo(float).eps) * max(abs(value), 1.0)
        trial = initial.copy()
        trial[index] = min(value + step, upper[index])
        if trial[index] == value:
            trial[index] = max(value - step, lower[index])
        delta = trial[index] - value
        if delta == 0:
            continue
        try:
            with np.errstate(
                over="ignore", invalid="ignore", divide="ignore", under="ignore"
            ):
                difference = (
                    (residual(trial) - initial_residual) / delta
                    if extra_residuals is not None
                    else (prediction(trial) - baseline) / scale / mean_factor / delta
                )
                sensitivity = float(np.max(abs(difference)))
                proposed = 1.0 / sensitivity if sensitivity > 0 else np.inf
            if np.isfinite(proposed) and proposed > 0:
                scales[index] = max(abs(value), proposed)
        except ValueError:
            # A finite-difference probe is not permission to leave the model's
            # real domain. Retain a finite coordinate scale; the solver still
            # rejects invalid actual iterates through residual().
            pass
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = (lower / scales, upper / scales)
    if any(
        np.any(np.isfinite(original) & ~np.isfinite(scaled))
        for original, scaled in zip((lower, upper), bounds, strict=True)
    ):
        raise ValueError(
            "Coefficient bounds exceed the representable scaled coordinate"
        )
    if constraints is not None:
        return _fit_constrained(
            result, keys, initial, scales, bounds, residual, constraints,
            max_nfev, diagnostics, scale, iterate_acceptance,
        )
    fit = least_squares(
        lambda values: residual(values * scales),
        initial / scales,
        bounds=bounds,
        method="trf",
        loss="linear",
        ftol=None,  # A tiny feasible step from a bound is not convergence.
        max_nfev=max_nfev,
    )
    if diagnostics is not None:
        diagnostics.update(
            nfev=fit.nfev,
            njev=fit.njev,
            success=bool(fit.success),
            status=int(fit.status),
            termination=str(fit.message),
            parameter_scales=scales.tolist(),
            response_scale=scale,
            additional_objective_residuals=extra_residuals is not None,
            objective_sum_squares=float(2 * fit.cost),
            nfev_scope="SciPy residual evaluations; prediction_evaluations also counts finite-difference and scaling probes",
        )
    if not fit.success or not np.isfinite([fit.cost, fit.optimality]).all():
        raise ValueError(f"Coefficient fit failed: {fit.message}")
    return with_parameters(result, keys, fit.x * scales)


def _fit_constrained(equation, keys, initial, scales, bounds, residual,
                     constraints, max_evaluations, diagnostics, response_scale, iterate_acceptance=None):
    """Explicit finite-point constraints inside joint fit, never output repair."""
    callback, lower, upper = constraints
    lower, upper = np.asarray(lower), np.asarray(upper)
    if np.any(lower > upper):
        raise ValueError("Contradictory coefficient-response constraints")
    evaluations, best, fit = 0, None, None
    retried = False

    class BudgetReached(Exception):
        pass

    def values(z):
        value = np.asarray(callback(with_parameters(equation, keys, z * scales)))
        if value.shape != lower.shape or not np.isfinite(value).all():
            raise ValueError("Invalid constrained coefficient evaluation; no rows dropped")
        return value

    def objective(z):
        nonlocal evaluations, best
        if evaluations >= max_evaluations:
            raise BudgetReached
        evaluations += 1
        r = residual(z * scales)
        loss = float(r @ r)
        if not np.isfinite(loss):
            raise ValueError("Nonfinite constrained fitting objective")
        value = values(z)
        if (np.all(value >= lower) and np.all(value <= upper)
                and (best is None or loss < best[0])):
            if (iterate_acceptance is None or iterate_acceptance(
                    with_parameters(equation, keys, z * scales), "iterate") is True):
                best = loss, np.array(z, copy=True)
        return loss

    def constraint_jacobian(z):
        # NonlinearConstraint's preparation does not know coefficient bounds.
        # Respect them explicitly rather than clipping actual solver iterates.
        base = values(z)
        jac = np.empty((len(base), len(z)))
        for i in range(len(z)):
            h = np.sqrt(np.finfo(float).eps) * max(1.0, abs(z[i]))
            plus, minus = z.copy(), z.copy()
            plus[i] = min(z[i] + h, bounds[1][i])
            minus[i] = max(z[i] - h, bounds[0][i])
            delta = plus[i] - minus[i]
            jac[:, i] = (values(plus) - values(minus)) / delta if delta else 0.0
        return jac

    def solve(start, low, high):
        active = np.isfinite(low) | np.isfinite(high)
        constraint = ([NonlinearConstraint(lambda z: values(z)[active], low[active], high[active],
                                            jac=lambda z: constraint_jacobian(z)[active])]
                      if np.any(active) else [])
        return minimize(
            objective, start, method="SLSQP", bounds=Bounds(*bounds),
            constraints=constraint, options={"maxiter": max_evaluations, "ftol": 1e-12},
        )

    budget_reached = False
    try:
        fit = solve(initial / scales, lower, upper)
        value = values(fit.x)
        if not (np.all(value >= lower) and np.all(value <= upper)):
            # A boundary optimum can terminate a few ulps outside feasibility.
            # Retry the optimization with inward inequality margins, not a
            # relaxed acceptance test or post-fit coefficient clipping.
            low, high = inward_response_bounds(lower, upper)
            retried = True
            fit = solve(fit.x, low, high)
    except BudgetReached:
        budget_reached = True
    if diagnostics is not None:
        diagnostics.update(
            solver="SLSQP", nfev=evaluations,
            success=bool(fit is not None and fit.success and not budget_reached),
            status=int(fit.status) if fit is not None else None,
            termination="objective evaluation budget" if budget_reached else str(fit.message),
            parameter_scales=scales.tolist(), response_scale=response_scale,
            inward_constraint_retry=retried,
            nfev_scope="all constrained fitting objective evaluations including numerical gradient probes; scaling probes separately counted",
        )
    if best is None:
        raise ValueError("No feasible coefficient iterate found; no response tolerance relaxed")
    best_objective, best_values = best
    if diagnostics is not None:
        diagnostics.update(objective_sum_squares=best_objective,
                           result="feasible_candidate; solver convergence is reported separately")
    result = with_parameters(equation, keys, best_values * scales)
    if iterate_acceptance is not None and iterate_acceptance(result, "final") is not True:
        raise ValueError("Returned coefficient iterate failed final admission; no unverified fallback")
    return result
