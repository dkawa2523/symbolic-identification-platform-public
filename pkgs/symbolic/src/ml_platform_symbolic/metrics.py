"""Common unit-scaled numerical summaries and error/complexity dominance."""

import numpy as np


def _weight_roots(weights, shape):
    weights = np.ones(shape) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != shape or not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Metric weights must be finite nonnegative with positive total mass")
    # Root before normalization preserves tiny weight ratios and avoids an
    # overflowing sum of otherwise finite weights.
    roots = np.sqrt(weights) / np.sqrt(weights.max())
    return roots, float(np.sum(roots * roots))


def weighted_rms(values, *, weights=None):
    """Weighted norm without squaring dimensional tiny/huge values."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("RMS needs nonempty finite values")
    roots, mass = _weight_roots(weights, values.shape)
    return _rms_roots(values, roots, mass)


def _mean(values, roots, mass):
    return float(np.sum((roots * (roots * values)) / mass))


def prediction_metrics(y, prediction, *, valid=None, weights=None):
    """Incomplete coverage never becomes an intersection-only or zero score."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (
        not len(y)
        or prediction.shape != y.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(y).all()
        or (valid is not None and not np.asarray(valid).all())
    ):
        return dict(rmse=None, mae=None, r2=None, reason="empty/nonfinite/invalid prediction; all rows retained")
    roots, mass = _weight_roots(weights, y.shape)
    # Validate ALL rows above, including zero-weight nonfinite/invalid values.
    # Only positive-weight rows may set arithmetic scale/centering afterwards.
    active = roots > 0
    y, prediction, roots = y[active], prediction[active], roots[active]
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        residual = y - prediction
        multiplier = 1.0
        if not np.isfinite(residual).all():
            multiplier = float(max(np.max(abs(y)), np.max(abs(prediction))))
            residual = y / multiplier - prediction / multiplier
        try:
            rmse = _rms_roots(residual, roots, mass) * multiplier
            mae = _mean(abs(residual), roots, mass) * multiplier
            if not np.isfinite([rmse, mae]).all() or ((rmse == 0 or mae == 0) and np.any(residual != 0)):
                raise FloatingPointError("metric outside representable range")
        except FloatingPointError as exc:
            return dict(rmse=None, mae=None, r2=None, reason=str(exc))
        centered = y - y.flat[0]
        center_multiplier = 1.0
        if not np.isfinite(centered).all():
            center_multiplier = float(np.max(abs(y)))
            centered = y / center_multiplier
        centered = centered - _mean(centered, roots, mass)
        try:
            deviation = _rms_roots(centered, roots, mass) * center_multiplier
            ratio = rmse / deviation if deviation > 0 else np.nan
            r2 = float(1 - np.square(ratio)) if len(y) > 1 and np.isfinite(ratio) else None
            r2 = r2 if r2 is not None and np.isfinite(r2) else None
        except FloatingPointError:
            r2 = None
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": r2,
        "reason": None if r2 is not None else "R2 undefined or outside representable range; RMSE/MAE retained",
    }


def paired_mse_difference(left, right, *, weights=None):
    """A squared-unit contrast or an explicit float64 limitation, never fake zero."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not left.size or not np.isfinite([left, right]).all():
        return None, "Nonfinite/incompatible residuals; no rows removed"
    roots, mass = _weight_roots(weights, left.shape)
    active = roots > 0
    left, right, roots = left[active], right[active], roots[active]
    if np.array_equal(abs(left), abs(right)):
        return 0.0, None  # Equal point losses in the saved float values.
    scale = float(max(np.max(abs(left)), np.max(abs(right))))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        a, b = left / scale, right / scale
        normalized = _mean((a - b) * (a + b), roots, mass)
        value = (normalized * scale) * scale
    if normalized == 0:
        return None, "Squared-loss contrast unresolved by float64 cancellation/underflow"
    if not np.isfinite(value) or value == 0:
        return None, "Squared target-unit contrast outside the representable float64 range"
    return float(value), None


def _rms_roots(values, roots, mass):
    # Keep the same normalized weights: squaring roots and normalizing again
    # can lose representable contributions from extremely unequal weights.
    components = values * roots
    magnitude = float(np.max(abs(components)))
    if magnitude == 0:
        if np.any((values != 0) & (roots > 0)):
            raise FloatingPointError("RMS below representable range")
        return 0.0
    result = magnitude * float(np.sqrt(np.sum((components / magnitude) ** 2) / mass))
    if not np.isfinite(result) or result == 0:
        raise FloatingPointError("RMS outside representable range")
    return result


def _mark_pareto(rows):
    from itertools import groupby

    for row in rows:
        row["pareto"] = False
    ranked = sorted((r for r in rows if r["eligible"]), key=lambda r: (r["rmse"], r["n_nodes"]))
    previous_nodes = float("inf")
    for _, group in groupby(ranked, key=lambda r: r["rmse"]):
        group = list(group)
        nodes = min(r["n_nodes"] for r in group)
        for row in group:
            row["pareto"] = row["n_nodes"] == nodes and nodes < previous_nodes
        previous_nodes = min(previous_nodes, nodes)
