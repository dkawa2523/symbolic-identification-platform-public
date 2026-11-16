"""Compile declared finite response conditions for native search objectives.

This is the same finite-step meaning as constraints.response_check, not a
proof of a derivative sign between points. Only Train supplies objective scales.
"""
import numpy as np

from .domains import validate_input_domain
from .numerical_fidelity import signal_scale


def requested_response_constraints(limits):
    names = []
    if limits.get("probes"):
        names.append("probes")
    if limits.get("response_bounds", [None, None]) != [None, None]:
        names.append("response_bounds")
    if limits.get("response_checks"):
        names.append("response_checks")
    return names


def response_enforcement(limits, capabilities):
    """Resolve explicit common intent without changing an adapter's legacy default."""
    requested = limits.get("response_enforcement")
    if requested is not None and (not isinstance(requested, str) or requested not in {"hard", "soft"}):
        raise ValueError("response_enforcement must be null, hard or soft")
    if limits.get("response_mode", "search") != "search":
        if requested is not None:
            raise ValueError("Explicit response_enforcement cannot be combined with filter mode")
        return {"requested": None, "effective": "post_filter", "stage": "after search"}
    if not requested_response_constraints(limits):
        return {"requested": requested, "effective": "not_requested", "stage": "none"}
    declaration = capabilities.get("response_enforcement", {})
    effective = requested if requested is not None else declaration.get("default")
    if requested is not None and requested not in declaration.get("supported", []):
        raise NotImplementedError(f"Requested {requested} response enforcement is unsupported by this adapter; no conversion or fallback")
    return {"requested": requested, "effective": effective or "adapter_default_not_declared",
            "stage": declaration.get("stage", "not declared"),
            "scope": "declared finite points only; different defaults are not equivalent enforcement"}


def compile_response_constraints(request):
    """Return points and bounded linear combinations of their predictions."""
    scale = float(signal_scale(request.y))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.max(np.abs(request.y))), 1.0)
    columns = [v["column"] for v in request.variables]
    points, terms, lookup = [], [], {}
    target_bounds = request.target.get("domain") or [None, None]

    def point(values):
        values = tuple(float(x) for x in values)
        validate_input_domain(np.asarray([values]), request.variables)
        if values not in lookup:
            lookup[values] = len(points)
            points.append(list(values))
        return lookup[values]

    def term(indices, coefficients, bounds, name, term_scale=scale):
        if not np.isfinite(term_scale) or term_scale <= 0:
            raise ValueError("Response objective scale is not representable; revise coordinates or finite step")
        terms.append(dict(indices=indices, coefficients=coefficients,
                          lower=bounds[0], upper=bounds[1], scale=float(term_scale), name=name))

    # Internal validity points accompany an explicit response-search request.
    # They do not turn domain-only jobs into constrained-search requests, nor do
    # they inherit the response bounds supplied only for explicit probes.
    if requested_response_constraints(request.limits):
        for row in request.limits.get("domain_probes", []):
            i = point([row[c] for c in columns])
            if target_bounds != [None, None]:
                term([i], [1.0], target_bounds, "target_domain_validity")
    for row in request.limits.get("probes", []):
        i = point([row[c] for c in columns])
        bounds = request.limits.get("response_bounds", [None, None])
        lower = [v for v in (bounds[0], target_bounds[0]) if v is not None]
        upper = [v for v in (bounds[1], target_bounds[1]) if v is not None]
        term([i], [1.0], [max(lower) if lower else None, min(upper) if upper else None], "probe_response")
    for check in request.limits.get("response_checks", []):
        direction = (np.asarray([check["direction"][c] for c in columns], dtype=float)
                     if "direction" in check else np.eye(len(columns))[columns.index(check["feature"])])
        h, order = check["step"], check["order"]
        projection = np.asarray(request.X) @ direction / float(direction @ direction)
        span = max(float(np.ptp(projection)), float(h))
        response_scale = scale / span**order
        bound = -float(check["tolerance"])
        bounds = [bound, None] if check["sign"] == 1 else [None, -bound]
        coefficients = [-1/(2*h), 0.0, 1/(2*h)] if order == 1 else [1/h**2, -2/h**2, 1/h**2]
        for row in check["points"]:
            x = np.asarray([row[c] for c in columns], dtype=float)
            indices = [point(v) for v in (x-h*direction, x, x+h*direction)]
            term(indices, coefficients, bounds, check.get("name", "response"), response_scale)
            if target_bounds != [None, None]:
                for i in indices:
                    term([i], [1.0], target_bounds, "target_domain")
    return dict(points=points, terms=terms, data_scale=scale)


def response_values(predictions, compiled, *, normalized=True):
    """Signed term values for explicit constraints; retain every declared point."""
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(compiled["points"]),) or not np.isfinite(values).all():
        raise ValueError("Invalid response constraint points; no rows may be discarded")
    result = np.asarray([
        float(np.dot(values[t["indices"]], t["coefficients"])) /
        (t["scale"] if normalized else 1.0) for t in compiled["terms"]
    ])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite response combination")
    return result


def response_bounds(compiled, *, normalized=True):
    """Bounds in the same coordinate as response_values; no extra tolerance."""
    lower = np.asarray([t["lower"] if t["lower"] is not None else -np.inf for t in compiled["terms"]])
    upper = np.asarray([t["upper"] if t["upper"] is not None else np.inf for t in compiled["terms"]])
    if np.any(lower > upper):
        raise ValueError("Contradictory response constraint bounds")
    if normalized:
        scale = np.asarray([t["scale"] for t in compiled["terms"]])
        lower, upper = lower / scale, upper / scale
    return lower, upper


def response_feasible(predictions, compiled):
    """Finite declared conditions only, in original units without relaxation."""
    try:
        values = response_values(predictions, compiled, normalized=False)
        lower, upper = response_bounds(compiled, normalized=False)
    except ValueError:
        return False
    return bool(np.all(values >= lower) and np.all(values <= upper))


def response_residuals(predictions, compiled):
    """Normalized residual vector; squared norm equals the shared mean penalty."""
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(compiled["points"]),) or not np.isfinite(values).all():
        raise ValueError("Invalid response objective points; no rows may be discarded")
    residuals = []
    for item in compiled["terms"]:
        value = float(np.dot(values[item["indices"]], item["coefficients"]))
        if not np.isfinite(value):
            raise ValueError("Nonfinite response combination")
        low, high = item["lower"], item["upper"]
        violation = max(0.0, low-value if low is not None else 0.0, value-high if high is not None else 0.0)
        residuals.append(violation/item["scale"])
    return np.asarray(residuals) / np.sqrt(len(residuals)) if residuals else np.empty(0)


def response_penalty(predictions, compiled):
    """Reference loss for parity checks; invalid values are never discarded."""
    try:
        residuals = response_residuals(predictions, compiled)
    except ValueError:
        return float("inf")
    return float(residuals @ residuals)


def constraint_application(job, capabilities):
    """Describe each requested condition without implying all use search."""
    limits = job["limits"]
    declared = capabilities.get("constraints", {})
    result = {"operators": declared.get("operators", "not declared"),
              "max_nodes": declared.get("max_nodes", "post-filtered")}
    if requested_response_constraints(limits):
        try:
            result["response_enforcement"] = response_enforcement(limits, capabilities)
        except (ValueError, NotImplementedError) as error:
            result["response_enforcement"] = {"requested": limits.get("response_enforcement"), "effective": "blocked", "reason": str(error)}
    options = job.get("options", {})
    if job.get("engine") in {"e2e", "flash_ansr", "rasr"}:
        enabled = options.get("operator_mask", job.get("engine") == "e2e")
        result["operators"] += (
            f"; configured operator_mask={enabled}; "
            + ("decoder masking requested; runtime evidence required to verify execution"
               if enabled else "decoder masking disabled; portable operators checked after generation")
        )
    for key in ("max_free_parameters", "allowed_variables", "equation_template"):
        if limits.get(key) is not None:
            result[key] = declared.get(key, "post-filtered only; no declared native variable restriction"
                                       if key == "allowed_variables" else "post-filtered")
    if limits.get("require_units") or limits.get("unit_search"):
        result["units"] = declared.get("units", "post-filtered")
        if job.get("engine") == "torchsisso":
            dimensional = options.get("dimensional", False)
            result["units"] += (
                f"; configured dimensional={dimensional}; "
                + ("native dimensional search requested; runtime evidence required to verify execution"
                   if dimensional else "native dimensional search disabled; portable units checked after generation")
            )
    if job.get("engine") == "pysr":
        mode = options.get("unit_mode")
        result["units"] = (
            "post-check only; native unit fitness disabled" if mode is None else
            f"configured native unit_mode={mode}; ordinary-expression fitness; "
            + ("dimensionless constants" if options.get("dimensionless_constants_only", False)
               else "native wildcard coefficient units; only uniquely implied structural dimensions supplemented on export, not physical identification")
            + "; template native-unit search unsupported; runtime evidence required"
        )
    if limits.get("unit_search"):
        support = capabilities.get("unit_search")
        from .units import unit_spec
        definitions = [*job.get("variables", []), job.get("target", {})]
        specs = [unit_spec(item.get("unit")) for item in definitions]
        quantities = [*specs, *(unit_spec(item.get("unit")) for item in job.get("fixed_constants", {}).values())]
        kinds_ignored = any(spec is not None and (spec.angle or spec.cycle) for spec in quantities)
        dimensionless = all(not any(spec.dimensions) for spec in specs) if all(spec is not None for spec in specs) else None
        result["unit_search"] = {
            "requested": True, "supported": support is not None,
            "all_input_target_basic_dimensions_dimensionless": dimensionless,
            "warning": ("Angle/cycle quantity kinds are not enforced during native search; post-checks are not a complete physical-kind proof. "
                        + ("All input/target basic dimensions are zero; native dimensions cannot distinguish these quantities." if dimensionless else "")) if kinds_ignored else None,
            "native_options": {key: options.get(key) for key in (support or {}).get("defaults", {})},
            "scope": "native basic dimensions (L/M/T/K) used during search; angle/cycle quantity kinds remain portable post-checks; not an identical hard guarantee across methods",
        }
    if job.get("fixed_constants"):
        result["fixed_constants"] = str(capabilities.get("fixed_constants", "unsupported"))
    for key in requested_response_constraints(limits):
        if limits.get("response_mode", "search") == "filter":
            result[key] = "post-filtered only (explicit control; not constrained search)"
        elif key in capabilities.get("search_constraints", []):
            result[key] = (declared.get(key, "search objective") +
                           "; declared response points reused unchanged in post-check; automatic domain points check validity only; "
                           "separate evaluation points required for independent verification; no global guarantee")
        else:
            result[key] = "unsupported during search; execution blocked"
    return result


def inward_response_bounds(lower, upper):
    """Numerical solver retry bounds; never use these for acceptance tests.

    Inputs are normalized bounds. Tighten inequalities by 1e-9 of their scale,
    capped at a quarter interval width; equalities remain unchanged. This may
    make a boundary-only feasible set empty, so retain original feasible points.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if np.any(lower > upper):
        raise ValueError("Contradictory response constraint bounds")
    margin = 1e-9 * np.maximum(1.0, np.maximum(
        np.where(np.isfinite(lower), abs(lower), 0.0),
        np.where(np.isfinite(upper), abs(upper), 0.0)))
    margin = np.minimum(margin, (upper - lower) / 4)
    equal = lower == upper
    return np.where(equal, lower, lower + margin), np.where(equal, upper, upper - margin)
