"""Declared domains, sampled validity and explicitly scoped exact rational checks."""

from copy import deepcopy

import numpy as np


def validate_bounds(bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("A domain is [lower, upper], with null for an open-ended bound")
    if any(
        v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)) for v in bounds
    ):
        raise ValueError("Domain endpoints must be finite numbers or null")
    if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
        raise ValueError("Domain lower endpoint exceeds upper endpoint")


def within_bounds(values, bounds):
    valid = np.ones(np.shape(values), dtype=bool)
    if bounds is not None:
        if bounds[0] is not None:
            valid &= values >= bounds[0]
        if bounds[1] is not None:
            valid &= values <= bounds[1]
    return valid


def validate_input_domain(X, variables):
    for i, variable in enumerate(variables):
        if not within_bounds(X[:, i], variable.get("domain")).all():
            raise ValueError(f"Input outside declared domain: {variable['column']}")


def probe_points(X, variables, seed, configured=()):
    """Reproducible interior points, permitted endpoints and explicit caller probes."""
    lower, upper = X.min(axis=0).copy(), X.max(axis=0).copy()
    for i, variable in enumerate(variables):
        bounds = variable.get("domain")
        if bounds is not None:
            lower[i] = bounds[0] if bounds[0] is not None else lower[i]
            upper[i] = bounds[1] if bounds[1] is not None else upper[i]
    center = lower / 2 + upper / 2
    points = list(np.random.default_rng(seed + 104729).uniform(lower, upper, size=(16, len(variables))))
    points.append(center)
    for i, variable in enumerate(variables):
        domain = variable.get("domain")
        endpoints = [v for v in (domain or []) if v is not None]
        for value in [*endpoints, 0.0, -0.0, 1.0, -1.0]:
            if within_bounds(np.asarray(value), domain):
                point = center.copy()
                point[i] = value
                points.append(point)
    points.extend([p[v["column"]] for v in variables] for p in configured)
    result = np.asarray(points, dtype=float)
    validate_input_domain(result, variables)
    return result


def validation_constraints(config, variables, target, X_train, *, domain=None):
    """Keep explicit response scope; add separate internal domain-validity points."""
    constraints = deepcopy(config["constraints"])
    if domain is not None or any(v.get("domain") is not None for v in variables) or target.get("domain") is not None:
        points = region_points(X_train, variables, config["split"]["seed"], domain, constraints["probes"])
        constraints["domain_probes"] = [dict(zip([v["column"] for v in variables], point, strict=True)) for point in points]
    return constraints


def expression_bounds(node, parameters, variables):
    """Conservative elementary bounds for checking known IEEE composition hazards."""
    unknown = (-np.inf, np.inf)
    if "const" in node:
        return (node["const"], node["const"])
    if "param" in node:
        value = parameters[node["param"]]["value"]
        return (value, value)
    if "var" in node:
        domain = next(v.get("domain") for v in variables if v["id"] == node["var"])
        return tuple(unknown[i] if value is None else value for i, value in enumerate(domain)) if domain else unknown
    args = [expression_bounds(child, parameters, variables) for child in node["args"]]
    op = node["op"]
    with np.errstate(all="ignore"):
        if op == "add":
            result = (args[0][0] + args[1][0], args[0][1] + args[1][1])
        elif op == "sub":
            result = (args[0][0] - args[1][1], args[0][1] - args[1][0])
        elif op == "neg":
            result = (-args[0][1], -args[0][0])
        elif op == "mul":
            products = [a * b for a in args[0] for b in args[1]]
            result = (min(products), max(products))
        elif op == "div" and not args[1][0] <= 0 <= args[1][1]:
            values = [a / b for a in args[0] for b in args[1]]
            result = (min(values), max(values))
        elif op == "abs":
            result = (0 if args[0][0] <= 0 <= args[0][1] else min(map(abs, args[0])), max(map(abs, args[0])))
        elif op in {"sin", "cos", "tanh"}:
            result = (-1.0, 1.0)
        elif op == "exp":
            result = tuple(np.exp(value) for value in args[0])
        elif op == "sinh":
            result = tuple(np.sinh(value) for value in args[0])
        elif op == "cosh":
            values = [np.cosh(value) for value in args[0]]
            result = (1.0 if args[0][0] <= 0 <= args[0][1] else min(values), max(values))
        elif op == "sqrt" and args[0][0] >= 0:
            result = tuple(np.sqrt(value) for value in args[0])
        elif op in {"pow", "pow_int"} and args[1][0] == args[1][1] and args[0][0] >= 0:
            exponent = float(args[1][0])
            if exponent < 0 and args[0][0] <= 0 <= args[0][1]:
                result = unknown
            else:
                values = [np.power(float(value), exponent) for value in args[0]]
                result = (min(values), max(values))
        elif op in {"pow", "pow_int"} and args[1][0] == args[1][1] and np.isfinite(args[1][0]):
            exponent = float(args[1][0])
            if exponent.is_integer() and exponent >= 0:
                values = [np.power(float(value), exponent) for value in args[0]]
                result = (
                    0.0 if exponent > 0 and int(exponent) % 2 == 0 and args[0][0] <= 0 <= args[0][1] else min(values),
                    max(values),
                )
            else:
                result = unknown
        elif op == "rootn" and args[1][0] == args[1][1] and not (args[1][0] < 0 and args[0][0] <= 0 <= args[0][1]):
            index = args[1][0]
            values = [
                np.sign(v) * np.power(abs(v), 1 / index) if int(index) % 2 else np.power(v, 1 / index) for v in args[0]
            ]
            result = (min(values), max(values))
        else:
            result = unknown
    return unknown if any(np.isnan(value) for value in result) else result


# Explicitly installed development extensions; artifacts never import callables.
_REGION_EXTENSIONS = {}
_RUNTIME_EXTENSIONS = {}


def register_runtime_extension(module, version, *, expected_sha256=None, expected_implementation_sha256=None):
    """Explicit trusted application initialization, never called by an artifact reader.

    Worker requests inherit only modules explicitly initialized by their caller.
    The same installed source/version must exist in the selected worker runtime.
    """
    from hashlib import sha256
    from importlib import import_module
    from pathlib import Path

    if not isinstance(module, str) or not module or not isinstance(version, str) or not version:
        raise ValueError("Runtime extension needs an installed module and explicit version")
    native = import_module(module)
    source = sha256(Path(native.__file__).read_bytes()).hexdigest()
    entry = Path(native.__file__).resolve()
    root = entry.parent
    declared = getattr(native, "__implementation_files__", None)
    files = [entry]
    if declared is not None:
        for name in declared:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Extension implementation file must remain within its installed module directory")
            files.append(path)
    elif entry.name == "__init__.py":
        files.extend(root.rglob("*.py"))
    import json

    payload = {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}
    implementation = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if expected_implementation_sha256 is not None and implementation != expected_implementation_sha256:
        raise ValueError("Runtime extension implementation files changed")
    declaration = {"module": module, "version": version, "sha256": source, "implementation_sha256": implementation}
    if getattr(native, "__version__", None) != version or (expected_sha256 is not None and source != expected_sha256):
        raise ValueError("Explicit runtime extension version/source differs from its trusted initialization")
    if module in _RUNTIME_EXTENSIONS:
        if _RUNTIME_EXTENSIONS[module] != declaration:
            raise ValueError("Runtime extension changed in an initialized process")
        return declaration.copy()
    native.initialize()
    _RUNTIME_EXTENSIONS[module] = declaration
    return declaration.copy()


def runtime_extensions():
    return [entry.copy() for entry in _RUNTIME_EXTENSIONS.values()]


def initialize_runtime_extensions(declarations):
    """Worker boundary for trusted job options, not serialized model metadata."""
    for item in declarations:
        if (
            set(item) - {"module", "version", "sha256", "implementation_sha256"}
            or not {"module", "version"} <= item.keys()
        ):
            raise ValueError("Runtime initialization accepts only module/version/optional source hash")
        register_runtime_extension(
            item["module"],
            item["version"],
            expected_sha256=item.get("sha256"),
            expected_implementation_sha256=item.get("implementation_sha256"),
        )


def register_region(name, version, *, contains, validation_points=None):
    """Install trusted functions for this process, not executable artifact data."""
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version or not callable(contains):
        raise ValueError("Region extension needs a name, version and explicit callable")
    if validation_points is not None and not callable(validation_points):
        raise ValueError("Region point provider must be explicitly installed")
    key = (name, version)
    if key in _REGION_EXTENSIONS:
        raise ValueError("Region extension already installed; choose an explicit new version")
    _REGION_EXTENSIONS[key] = (contains, validation_points)


def region_mask(X, region):
    """Membership in the stated coordinate; no inferred box or hidden inverse."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Region membership expects a row matrix")
    valid = np.isfinite(X).all(axis=1)
    if region is None:
        return valid
    if X.shape[1] != len(region["columns"]):
        raise ValueError("Region columns belong to a different input space")
    kind = region["kind"]
    if kind == "extension":
        key = (region["name"], region["version"])
        if key not in _REGION_EXTENSIONS:
            raise ValueError(f"Region extension is unavailable; install explicitly: {key}")
        result = np.asarray(_REGION_EXTENSIONS[key][0](X, region.get("settings", {})), dtype=bool)
        if result.shape != (len(X),):
            raise ValueError("Region extension must return one membership value per row")
        return valid & result
    if kind == "intersection":
        for child in region["regions"]:
            valid &= region_mask(X, child)
        return valid
    if kind == "pullback":
        inverse = region["inverse"]
        source = X @ np.asarray(inverse["matrix"], dtype=float).T + np.asarray(inverse["offset"], dtype=float)
        return valid & region_mask(source, region["source_domain"])
    if kind not in {"box", "affine", "observed"}:
        raise ValueError("Unsupported region representation")
    for i, bounds in enumerate(region.get("bounds", [None] * X.shape[1])):
        valid &= within_bounds(X[:, i], bounds)
    for field, equality in (("equalities", True), ("inequalities", False)):
        rows = region.get(field, {})
        if not rows:
            continue
        matrix, rhs = np.asarray(rows["matrix"], dtype=float), np.asarray(rows["rhs"], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            values = X @ matrix.T
            # Only floating arithmetic resolution is automatic, not a physical tolerance.
            rounding = 32 * np.finfo(float).eps * (abs(X) @ abs(matrix).T + abs(rhs))
            tolerance = np.asarray(rows.get("atol", 0), dtype=float) + rounding
            satisfied = abs(values - rhs) <= tolerance if equality else values <= rhs + tolerance
        valid &= np.isfinite(values).all(axis=1) & satisfied.all(axis=1)
    return valid


def domain_variables(variables, region):
    """Safe axis-bound hints from the same region, never a substitute for membership."""
    result = deepcopy(variables)
    if region is None:
        return result
    regions = region.get("regions", []) if region["kind"] == "intersection" else [region]
    for item in regions:
        if item["kind"] not in {"box", "affine"}:
            continue
        for variable, bounds in zip(result, item.get("bounds", [None] * len(result)), strict=True):
            if bounds is None:
                continue
            previous = variable.get("domain") or [None, None]
            low = max((v for v in (previous[0], bounds[0]) if v is not None), default=None)
            high = min((v for v in (previous[1], bounds[1]) if v is not None), default=None)
            validate_bounds([low, high])
            variable["domain"] = [low, high]
    return result


def region_points(X, variables, seed, region, configured=()):
    """Use declared geometry or actual feasible rows; never fill a manifold's box."""
    if region is None or region["kind"] == "box":
        result = probe_points(X, domain_variables(variables, region), seed, configured)
        if not region_mask(result, region).all():
            raise ValueError("Validation point is outside its declared region")
        return result
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    points = list(X[rng.permutation(len(X))[:16]])
    convex = region["kind"] == "affine" or (
        region["kind"] == "pullback" and region["source_domain"]["kind"] in {"affine", "box"}
    )
    if convex and len(X):
        left, right = rng.integers(len(X), size=(2, 16))
        mix = rng.uniform(size=(16, 1))
        points.extend(mix * X[left] + (1 - mix) * X[right])
    if region["kind"] == "extension":
        key = (region["name"], region["version"])
        if key not in _REGION_EXTENSIONS:
            raise ValueError(f"Region extension is unavailable; install explicitly: {key}")
        provider = _REGION_EXTENSIONS[key][1]
        if provider is not None:
            points.extend(provider(seed, region.get("settings", {})))
    points.extend(region.get("validation_points", []))
    points.extend([p[v["column"]] for v in variables] for p in configured)
    result = np.asarray(points, dtype=float).reshape(-1, len(variables))
    validate_input_domain(result, variables)
    if not region_mask(result, region).all():
        raise ValueError("Validation point is outside its declared region")
    return result


def check_region_definition(region, columns):
    """Validate the small data-only forms at the problem boundary."""
    if not isinstance(region, dict) or region.get("columns") != columns:
        raise ValueError("Region needs the exact declared column order")
    allowed = {
        "kind",
        "space_id",
        "columns",
        "units",
        "coordinate_roles",
        "bounds",
        "equalities",
        "inequalities",
        "validation_points",
        "name",
        "version",
        "settings",
    }
    if set(region) - allowed:
        raise ValueError(f"Unknown region fields: {sorted(set(region) - allowed)}")
    if region.get("kind") not in {"box", "affine", "observed", "extension"}:
        raise ValueError("Public region is box, affine, observed or an explicitly installed extension")
    if len(region.get("bounds", [None] * len(columns))) != len(columns):
        raise ValueError("Region bounds shape differs from columns")
    for bounds in region.get("bounds", []):
        if bounds is not None:
            validate_bounds(bounds)
    for name in ("equalities", "inequalities"):
        rows = region.get(name, {})
        if not rows:
            continue
        if set(rows) - {"matrix", "rhs", "atol"} or not {"matrix", "rhs"} <= rows.keys():
            raise ValueError("Affine constraints need matrix/rhs and optional coordinate tolerance")
        A, b = np.asarray(rows["matrix"], dtype=float), np.asarray(rows["rhs"], dtype=float)
        if A.ndim != 2 or A.shape[1] != len(columns) or b.shape != (len(A),):
            raise ValueError("Affine region matrix/rhs shape mismatch")
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            raise ValueError("Affine region coefficients must be finite")
        tolerance = np.asarray(rows.get("atol", 0), dtype=float)
        if tolerance.shape not in {(), (len(A),)} or not np.isfinite(tolerance).all() or (tolerance < 0).any():
            raise ValueError("Affine tolerance is nonnegative in each constraint's stated coordinate")


def rational_domain_check(equation, variables, **options):
    """Bounded exact 1D divisor-zero check; safe is not IEEE/monotonic safety."""
    from .rational_domain import check

    return check(equation, variables, **options)
