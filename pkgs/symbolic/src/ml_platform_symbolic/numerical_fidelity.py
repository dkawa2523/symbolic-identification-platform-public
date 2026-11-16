"""Signal-scaled numerical agreement, independent of a large common output offset."""

import numpy as np


def signal_scale(values):
    """Stable population std without squaring a dimensional tiny/huge magnitude."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Signal values must be nonempty and finite")
    with np.errstate(over="ignore"):
        centered = values - values.flat[0]
    if np.isfinite(centered).all():
        values = centered
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0:
        return 0.0
    deviation = float(np.std(values / magnitude))
    scale = magnitude * deviation
    # A dtype bound, not a physical-unit floor. Resolution checks will reject
    # an approximate tolerance below the representable range.
    return float(np.nextafter(0.0, 1.0)) if deviation > 0 and scale == 0 else scale


def compare_predictions(native, portable, *, scale, relative_tolerance=1e-5):
    native, portable = np.asarray(native, dtype=float), np.asarray(portable, dtype=float)
    if native.shape != portable.shape or not np.isfinite(scale) or scale < 0:
        raise ValueError("Prediction shapes must match and fidelity scale must be nonnegative/finite")
    finite = np.isfinite(native)
    masks_match = all(np.array_equal(test(native), test(portable)) for test in (np.isnan, np.isposinf, np.isneginf))
    # ULP allowance is float64 representation noise, never a raw-value relative tolerance.
    spacing = np.maximum(np.abs(np.spacing(native[finite])), np.abs(np.spacing(portable[finite])))
    signal_tolerance = relative_tolerance * scale
    resolution_limited = bool(scale > 0 and np.any(8 * spacing > signal_tolerance))
    tolerance = signal_tolerance + 8 * spacing if scale > 0 else 0.0
    error = np.abs(native[finite] - portable[finite])
    return {
        "passed": bool(masks_match and finite.any() and not resolution_limited and np.all(error <= tolerance)),
        "n_points": int(finite.sum()),
        "max_abs_error": float(error.max()) if len(error) else None,
        "absolute_scale": float(scale),
        "signal_rtol": relative_tolerance,
        "atol": relative_tolerance * float(scale),
        "float64_ulp_allowance": 8,
        "resolution_limited": resolution_limited,
        "zero_signal_policy": "exact equality; no dimensional absolute floor",
        "finite_and_signed_nonfinite_masks_match": masks_match,
        "policy": "signal scale plus float64 ULP; no relative tolerance on raw output offset",
    }


def compare_replay(saved, loaded, *, scale, relative_tolerance=1e-10):
    """Serialization equality is distinct from native signal-resolution adequacy."""
    check = compare_predictions(saved, loaded, scale=scale, relative_tolerance=relative_tolerance)
    a, b = np.asarray(saved, dtype=float), np.asarray(loaded, dtype=float)
    exact = bool(np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b)))
    return {
        **check,
        "serialization_exact_values_and_signs": exact,
        "serialization_check_passed": exact or check["passed"],
        "serialization_scope": "Saved function replay only; original native passed/resolution_limited evidence is unchanged",
    }
