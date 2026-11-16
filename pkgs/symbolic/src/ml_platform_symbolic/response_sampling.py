"""Expand explicit box sampling into finite response points before unit conversion.

This input shorthand reads no observations or targets. Resolved point arrays are
its sole numerical authority; consumed sampling specifications never regenerate.
"""
import math
from copy import deepcopy
from itertools import product


def _points(spec, columns, check=None):
    if not isinstance(spec, dict) or set(spec) != {"domain", "points", "corners", "seed"}:
        raise ValueError("Response sampling requires domain, points, corners and seed")
    box, count, corners, seed = (spec[k] for k in ("domain", "points", "corners", "seed"))
    if not isinstance(box, dict) or set(box) != set(columns):
        raise ValueError("Sampling domain must name every input column exactly")
    if type(count) is not int or not 1 <= count <= 4096 or count & (count - 1):
        raise ValueError("Sampling points must be a power of two between 1 and 4096")
    if type(corners) is not int or not 0 <= corners <= 256:
        raise ValueError("Sampling corners is a budget from 0 to 256; 0 disables corners")
    if corners and 2 ** len(columns) > corners:
        raise ValueError("Sampling corner budget must cover all box corners; partial corners are not selected")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Sampling seed must be an integer in [0, 2**32)")
    margins = dict.fromkeys(columns, 0.0)
    if check is not None:
        step = check.get("step")
        if type(step) not in {int, float} or not math.isfinite(step) or step <= 0:
            raise ValueError("Response sampling requires a positive finite step")
        if ("feature" in check) == ("direction" in check):
            raise ValueError("Response sampling requires feature OR direction")
        if "feature" in check:
            if check["feature"] not in columns:
                raise ValueError("Response sampling feature must be an input column")
            margins[check["feature"]] = step
        else:
            direction = check["direction"]
            if not isinstance(direction, dict) or set(direction) != set(columns):
                raise ValueError("Response sampling direction must name every input column")
            if any(type(v) not in {int, float} or not math.isfinite(v) for v in direction.values()):
                raise ValueError("Response sampling direction must be finite")
            margins = {c: abs(step * direction[c]) for c in columns}
    lower, upper = [], []
    for c in columns:
        bounds = box[c]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[1]):
            raise ValueError("Sampling domain requires finite increasing bounds")
        lo, hi = bounds[0] + margins[c], bounds[1] - margins[c]
        # Round only generated center-interval endpoints inward when cancellation
        # makes their unchanged +/- step stencil cross the declared box by an ULP.
        # Never clip a stencil endpoint, alter a step, or touch explicit points.
        for _ in range(8):
            if lo - margins[c] >= bounds[0]:
                break
            lo = math.nextafter(lo, math.inf)
        for _ in range(8):
            if hi + margins[c] <= bounds[1]:
                break
            hi = math.nextafter(hi, -math.inf)
        if (not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
                or not math.isfinite(hi - lo) or lo - margins[c] < bounds[0]
                or hi + margins[c] > bounds[1]):
            raise ValueError("Sampling box has no representable full-stencil center interval")
        lower.append(lo)
        upper.append(hi)
    from scipy.stats import qmc
    unit = qmc.Sobol(d=len(columns), scramble=True, seed=seed).random_base2(count.bit_length() - 1)
    rows = qmc.scale(unit, lower, upper).tolist()
    if corners:
        rows.extend(product(*zip(lower, upper, strict=True)))
    # Check actual floating-point stencils, not only the ideal inset bounds.
    # Fail explicitly rather than clip points or silently change a difference.
    for row in rows:
        for c, value in zip(columns, row, strict=True):
            if (not math.isfinite(value) or value - margins[c] < box[c][0]
                    or value + margins[c] > box[c][1]):
                raise ValueError("Generated point/stencil exceeds sampling box in floating-point arithmetic")
    return [dict(zip(columns, row, strict=True)) for row in rows]


def materialize_response_sampling(config):
    """Consume sampling only; leave all explicit point rows and other settings intact."""
    cfg = deepcopy(config)
    constraints = cfg.get("constraints", {})
    if not isinstance(constraints, dict):
        return cfg  # Existing configuration validation owns malformed sections.
    checks = constraints.get("response_checks", [])
    active = "probe_sampling" in constraints or (isinstance(checks, list) and any(
        isinstance(c, dict) and "sampling" in c for c in checks))
    if not active:
        return cfg
    columns = cfg.get("data", {}).get("feature_columns", [])
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Response sampling requires explicit distinct feature_columns")

    def expand(owner, key, field, check=None, target="probes"):
        if key not in owner:
            return
        declaration = owner.pop(key)
        generated = _points(declaration, columns, check)
        explicit = owner.get(field, [])
        if not isinstance(explicit, list):
            raise ValueError("Explicit response points must be a list")
        # Preserve explicit rows exactly. Avoid adding duplicate generated rows.
        combined = list(explicit)
        for point in generated:
            if point not in combined:
                combined.append(point)
        owner[field] = combined
        cfg.setdefault("_sampling_provenance", []).append({
            "target": target, "declaration": deepcopy(declaration),
            "columns": list(columns), "coordinate_units": deepcopy(cfg.get("data", {}).get("units", {})),
            "coordinate_roles": deepcopy(cfg.get("data", {}).get("coordinate_roles", {})),
            "center_rounding": "inward representable endpoints; unchanged centered step",
            "generated_count": len(generated), "added_count": len(combined) - len(explicit),
            "role": "Input shorthand provenance only; resolved response points are authoritative",
        })

    expand(constraints, "probe_sampling", "probes")
    if isinstance(checks, list):
        for index, check in enumerate(checks):
            if isinstance(check, dict):
                expand(check, "sampling", "points", check, f"response_checks[{index}].points")
    return cfg
