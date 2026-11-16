"""Freeze CSV/public units into coherent problem coordinates once, before search.

All changes are explicit affine maps. Native profiles and learned normalization
are untouched. Resolved source_units marks the already-frozen input mapping.
"""

from copy import deepcopy

import numpy as np

from .units import (
    apply_coordinate,
    canonical_unit,
    conversion,
    convert_derivative,
    convert_width,
    quantity_conversion,
    quantity_role,
)


def canonical_problem(config):
    cfg = deepcopy(config)
    data = cfg["data"]
    if data["source_units"]:
        # Already resolved, or an explicit canonical-input workflow.
        for column, label in data["source_units"].items():
            conversion(label, data["units"].get(column))
        return cfg
    public = dict(data["units"])
    working = {column: canonical_unit(label) for column, label in public.items()}
    data["source_units"] = public
    data["units"] = working
    for column, unit in public.items():
        data["coordinate_roles"][column] = quantity_role(
            {"unit": unit, "coordinate_role": data["coordinate_roles"].get(column)}
        )

    def value(column, x):
        if x is None:
            return None
        a, b = quantity_conversion(public.get(column), working.get(column), data["coordinate_roles"].get(column))
        return apply_coordinate(x, a, b)

    def bounds(column, limits):
        values = [value(column, x) for x in limits]
        scale, _ = conversion(public.get(column), working.get(column))
        return values if scale > 0 else values[::-1]

    for column, limits in data["domain"].items():
        data["domain"][column] = bounds(column, limits)
    for constant in data["fixed_constants"].values():
        unit = canonical_unit(constant["unit"])
        role = quantity_role(constant)
        a, b = quantity_conversion(constant["unit"], unit, role)
        constant["value"] = apply_coordinate(constant["value"], a, b)
        constant["coordinate_role"] = role
        constant["unit"] = unit
    target = data["target_column"]
    constraints = cfg["constraints"]
    constraints["probes"] = [{c: value(c, x) for c, x in point.items()} for point in constraints["probes"]]
    constraints["response_bounds"] = bounds(target, constraints["response_bounds"])
    for request in constraints["response_checks"]:
        if "direction" in request:
            path_public = request["path_unit"]
            path_working = canonical_unit(path_public)
            path_scale = convert_width(1.0, path_public, path_working)
            request["direction"] = {
                c: apply_coordinate(d, conversion(public.get(c), working.get(c))[0] / path_scale)
                for c, d in request["direction"].items()
            }
            request["step"] = apply_coordinate(request["step"], abs(path_scale))
            request["path_unit"] = path_working
            factor = conversion(public.get(target), working.get(target))[0] / path_scale ** request["order"]
        else:
            column = request["feature"]
            feature_scale, _ = conversion(public.get(column), working.get(column))
            request["step"] = apply_coordinate(request["step"], abs(feature_scale))
            factor = convert_derivative(
                1.0,
                public.get(column),
                working.get(column),
                public.get(target),
                working.get(target),
                order=request["order"],
            )
        request["tolerance"] = apply_coordinate(request["tolerance"], abs(factor))
        request["sign"] *= 1 if factor > 0 else -1
        request["points"] = [{c: value(c, x) for c, x in point.items()} for point in request["points"]]
    for key in ("absolute_tolerance", "acceptable_rmse"):
        if cfg["evaluation"][key] is not None:
            cfg["evaluation"][key] = convert_width(cfg["evaluation"][key], public.get(target), working.get(target))
    return cfg


def _affine_array(mapping, inputs, outputs):
    matrix, offset = np.asarray(mapping["matrix"], dtype=float), np.asarray(mapping["offset"], dtype=float)
    if matrix.shape != (outputs, inputs) or offset.shape != (outputs,):
        raise ValueError("Input-map matrix/offset differs from the declared source and target columns")
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("Input-map coefficients must be finite")
    return matrix, offset


def freeze_input_geometry(config, source_variables):
    """Resolve source units/domain once, then a declared reversible affine chart.

    Learned native normalization remains adapter-owned. A projection that is
    not invertible on the declared set is not silently accepted as a chart.
    """
    from .domains import check_region_definition

    source_variables = deepcopy(source_variables)
    columns = [v["column"] for v in source_variables]
    data = config["data"]
    raw = deepcopy(config.get("domain") or {})
    expected = raw.get("columns", columns)
    if expected != columns:
        raise ValueError("Original domain columns must match the observed input order")
    raw["columns"] = columns
    raw.setdefault("space_id", "source_inputs")
    raw.setdefault("kind", "box" if any(v.get("domain") is not None for v in source_variables) else "observed")
    raw.setdefault("units", {c: data["source_units"].get(c, data["units"].get(c)) for c in columns})
    raw.setdefault("coordinate_roles", {c: data["coordinate_roles"].get(c) for c in columns})
    if set(raw["units"]) != set(columns) or set(raw["coordinate_roles"]) != set(columns):
        raise ValueError("Original domain units/roles must describe its exact columns")
    for v in source_variables:
        if raw["coordinate_roles"][v["column"]] != v.get("coordinate_role"):
            raise ValueError("Domain and input coordinates have different value/difference roles")
    scales, offsets = [], []
    for v in source_variables:
        a, b = quantity_conversion(raw["units"][v["column"]], v.get("unit"), v.get("coordinate_role"))
        scales.append(a)
        offsets.append(b)
    scales, offsets = np.asarray(scales), np.asarray(offsets)
    if raw["kind"] == "extension" and (np.any(scales != 1) or np.any(offsets != 0)):
        raise ValueError(
            "Declare an extension domain in canonical coordinates; callback unit conversion is not inferred"
        )
    check_region_definition(raw, columns)
    if "bounds" in raw:
        bounds = []
        for limits, a, b in zip(raw["bounds"], scales, offsets, strict=True):
            values = None if limits is None else [None if x is None else apply_coordinate(x, a, b) for x in limits]
            bounds.append(values if values is None or a > 0 else values[::-1])
        raw["bounds"] = bounds
    else:
        raw["bounds"] = [v.get("domain") for v in source_variables]
    for i, variable in enumerate(source_variables):
        old, new = variable.get("domain"), raw["bounds"][i]
        if old is not None and new is not None:
            raw["bounds"][i] = [
                max((v for v in (old[0], new[0]) if v is not None), default=None),
                min((v for v in (old[1], new[1]) if v is not None), default=None),
            ]
        elif old is not None:
            raw["bounds"][i] = old
    for field in ("equalities", "inequalities"):
        if raw.get(field):
            rows = raw[field]
            matrix = np.asarray(rows["matrix"], dtype=float)
            rows["matrix"] = (matrix / scales).tolist()
            rows["rhs"] = (np.asarray(rows["rhs"]) + matrix @ (offsets / scales)).tolist()
    if raw.get("validation_points"):
        points = np.asarray(raw["validation_points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != len(columns):
            raise ValueError("Provided domain points must use its original column order")
        raw["validation_points"] = apply_coordinate(points, scales, offsets).tolist()
    raw["units"] = {v["column"]: v.get("unit") for v in source_variables}
    geometry = {"schema_version": 1, "source_variables": source_variables, "domain": raw, "map": None}
    mapping = deepcopy(config.get("input_map"))
    if mapping is None:
        return geometry, source_variables
    allowed = {
        "kind",
        "space_id",
        "source_columns",
        "source_units",
        "target_columns",
        "target_units",
        "target_roles",
        "forward",
        "inverse",
    }
    if not isinstance(mapping, dict) or set(mapping) - allowed or mapping.get("kind") != "affine_chart":
        raise ValueError("Current portable input map is an explicitly declared affine_chart")
    if mapping.get("source_columns", columns) != columns:
        raise ValueError("Chart source columns differ from original inputs")
    target_columns = mapping["target_columns"]
    if not target_columns or len(set(target_columns)) != len(target_columns):
        raise ValueError("Chart target columns must be nonempty and unique")
    source_units = mapping.get("source_units", {c: data["source_units"].get(c, data["units"].get(c)) for c in columns})
    target_units = mapping["target_units"]
    if set(source_units) != set(columns) or set(target_units) != set(target_columns):
        raise ValueError("Chart source/target units need their exact column names")
    target_roles = mapping.get("target_roles", {})
    if set(target_roles) - set(target_columns):
        raise ValueError("Chart roles refer to unknown target coordinates")
    variables = [
        {
            "id": f"x{i}",
            "column": c,
            "unit": canonical_unit(target_units[c]),
            "domain": None,
            "coordinate_role": quantity_role({"unit": target_units[c], "coordinate_role": target_roles.get(c)}),
        }
        for i, c in enumerate(target_columns)
    ]
    sx, bx, sz, bz = [], [], [], []
    for v in source_variables:
        a, b = quantity_conversion(source_units[v["column"]], v["unit"], v.get("coordinate_role"))
        sx.append(a)
        bx.append(b)
    for v in variables:
        a, b = quantity_conversion(target_units[v["column"]], v["unit"], v.get("coordinate_role"))
        sz.append(a)
        bz.append(b)
    sx, bx, sz, bz = map(np.asarray, (sx, bx, sz, bz))
    F, f = _affine_array(mapping["forward"], len(columns), len(variables))
    G, g = _affine_array(mapping["inverse"], len(variables), len(columns))
    forward, shift = sz[:, None] * F / sx, sz * (f - F @ (bx / sx)) + bz
    inverse, back = sx[:, None] * G / sz, sx * (g - G @ (bz / sz)) + bx
    # A chart is reversible on the affine source set, not on an arbitrary box.
    E = np.asarray(raw.get("equalities", {}).get("matrix", np.empty((0, len(columns)))), dtype=float)
    rhs = np.asarray(raw.get("equalities", {}).get("rhs", []), dtype=float)
    if len(E):
        _, singular, vh = np.linalg.svd(E, full_matrices=True)
        rank = int(np.sum(singular > np.finfo(float).eps * max(E.shape) * singular[0])) if len(singular) else 0
        tangent = vh[rank:].T
        anchor = np.linalg.lstsq(E, rhs, rcond=None)[0]
    else:
        tangent, anchor = np.eye(len(columns)), np.zeros(len(columns))
    tests = [
        (forward @ inverse, np.eye(len(variables))),
        (forward @ back + shift, np.zeros(len(variables))),
        ((inverse @ forward - np.eye(len(columns))) @ tangent, np.zeros_like(tangent)),
        ((inverse @ forward - np.eye(len(columns))) @ anchor + inverse @ shift + back, np.zeros(len(columns))),
    ]
    for actual, expected in tests:
        scale = max(
            np.max(abs(actual), initial=0),
            np.max(abs(expected), initial=0),
            np.linalg.norm(forward) * np.linalg.norm(inverse),
        )
        if not np.allclose(actual, expected, rtol=0, atol=128 * np.finfo(float).eps * scale):
            raise ValueError("Input map is not invertible on the declared affine source set; it is not a chart")
    geometry["map"] = {
        "kind": "affine_chart",
        "source_space": raw["space_id"],
        "source_columns": columns,
        "source_units": raw["units"],
        "target_space": mapping.get("space_id", "execution_inputs"),
        "target_columns": target_columns,
        "target_units": {v["column"]: v["unit"] for v in variables},
        "forward": {"matrix": forward.tolist(), "offset": shift.tolist()},
        "inverse": {"matrix": inverse.tolist(), "offset": back.tolist()},
    }
    return geometry, variables


def map_input_values(X, geometry):
    """Input is already in the source working unit; original membership comes first."""
    from .domains import region_mask, validate_input_domain

    validate_input_domain(X, geometry["source_variables"])
    if not region_mask(X, geometry["domain"]).all():
        raise ValueError("Input is outside the declared original-coordinate domain")
    mapping = geometry.get("map")
    if mapping is None:
        return X
    forward = mapping["forward"]
    result = np.asarray(X) @ np.asarray(forward["matrix"]).T + np.asarray(forward["offset"])
    if not np.isfinite(result).all():
        raise ValueError("Input coordinate map produced nonfinite values")
    return result


def source_input_values(X, geometry):
    """Recover source working coordinates only for a verified reversible chart."""
    if not geometry or not geometry.get("map"):
        return X
    inverse = geometry["map"]["inverse"]
    return np.asarray(X) @ np.asarray(inverse["matrix"]).T + np.asarray(inverse["offset"])


def map_input_covariance(covariance, geometry):
    """Covariance is already in source working coordinates; affine mixing is AΣAᵀ."""
    covariance = np.asarray(covariance, dtype=float)
    n = len(geometry["source_variables"])
    if covariance.shape[-2:] != (n, n) or not np.isfinite(covariance).all():
        raise ValueError("Input covariance shape/coordinates differ from the source variables")
    if not geometry.get("map"):
        return covariance.copy()
    matrix = np.asarray(geometry["map"]["forward"]["matrix"])
    return matrix @ covariance @ matrix.T


def execution_region(geometry, variables):
    if not geometry:
        return None
    if not geometry.get("map"):
        return geometry["domain"]
    mapping = geometry["map"]
    inverse = mapping["inverse"]
    F = mapping["forward"]
    original_points = np.asarray(geometry["domain"].get("validation_points", []), dtype=float)
    points = (
        (original_points @ np.asarray(F["matrix"]).T + np.asarray(F["offset"])).tolist() if len(original_points) else []
    )
    return {
        "kind": "pullback",
        "columns": [v["column"] for v in variables],
        "space_id": mapping["target_space"],
        "inverse": inverse,
        "source_domain": geometry["domain"],
        "validation_points": points,
    }


def execution_constraints(constraints, geometry, variables):
    """Materialize source requests once; finite paths must remain feasible before mapping."""
    result = deepcopy(constraints)
    if not geometry:
        return result
    source_columns = [v["column"] for v in geometry["source_variables"]]
    target_columns = [v["column"] for v in variables]
    mapping = geometry.get("map")
    if mapping and result.get("allowed_variables") is not None:
        raise ValueError(
            "Source variable-usage restrictions require an explicit mapped-expression audit; not silently weakened"
        )
    points = result.get("probes", [])
    if points:
        original = np.asarray([[p[c] for c in source_columns] for p in points], dtype=float)
        mapped = map_input_values(original, geometry)
        result["probes"] = [dict(zip(target_columns, row, strict=True)) for row in mapped]
    for request in result.get("response_checks", []):
        points = np.asarray([[p[c] for c in source_columns] for p in request["points"]], dtype=float)
        if "direction" in request:
            direction = np.asarray([request["direction"][c] for c in source_columns], dtype=float)
        else:
            direction = np.zeros(len(source_columns))
            direction[source_columns.index(request["feature"])] = 1
            if mapping:
                request["path_unit"] = next(
                    v.get("unit") for v in geometry["source_variables"] if v["column"] == request["feature"]
                )
        for offset in (-request["step"], 0, request["step"]):
            map_input_values(points + offset * direction, geometry)
        if mapping:
            request["source_direction"] = dict(zip(source_columns, direction.tolist(), strict=True))
            direction = np.asarray(mapping["forward"]["matrix"]) @ direction
            mapped = map_input_values(points, geometry)
            request["points"] = [dict(zip(target_columns, row, strict=True)) for row in mapped]
            request["direction"] = dict(zip(target_columns, direction.tolist(), strict=True))
            request.pop("feature", None)
    return result


def intersect_input_geometry(source, requested):
    """Retain source restrictions; incompatible charts need an explicit new view."""
    if source is None:
        return deepcopy(requested)
    if requested is None:
        return deepcopy(source)

    def signature(g):
        return [(v["column"], v.get("unit"), v.get("coordinate_role")) for v in g["source_variables"]]

    if source.get("map") != requested.get("map") or signature(source) != signature(requested):
        raise ValueError("Source and requested input maps differ; an explicit coordinate-composition view is required")
    if source["domain"] == requested["domain"]:
        return deepcopy(requested)
    result = deepcopy(requested)
    result["domain"] = {
        "kind": "intersection",
        "columns": requested["domain"]["columns"],
        "space_id": requested["domain"]["space_id"],
        "regions": [deepcopy(source["domain"]), deepcopy(requested["domain"])],
    }
    return result

def _source_expression_map(context, residual=False):
    """Frozen numerical coordinate composition, not a dimensional proof.

    Used by equation presentation and search node budgets. Identity operations
    are omitted here only; no fitted expression simplification is performed.
    """
    from .data import input_variables, public_unit

    def op(name, *args):
        return {"op": name, "args": list(args)}

    def affine(tree, scale, offset):
        if scale != 1:
            tree = op("mul", {"const": scale}, tree)
        if offset != 0:
            tree = op("add", tree, {"const": offset})
        return tree

    frozen = (context.get("unit_coordinates") or {}).get("columns", {})
    def coordinate(definition):
        column = definition["column"]
        public = public_unit(context, definition)
        record = frozen.get(column)
        if record is None:
            if public != definition.get("unit"):
                raise ValueError("Source coordinate conversion is not frozen: " + column)
            return 1.0, 0.0, public
        if record.get("problem_unit") != definition.get("unit"):
            raise ValueError("Frozen working unit differs: " + column)
        scale, offset = record.get("scale"), record.get("offset")
        if (isinstance(scale, bool) or isinstance(offset, bool) or
                not isinstance(scale, (int, float)) or not isinstance(offset, (int, float)) or
                not np.isfinite([scale, offset]).all() or scale == 0):
            raise ValueError("Invalid frozen source coordinate: " + column)
        return scale, offset, public

    source = input_variables(context)
    columns = [v["column"] for v in source]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate source columns")
    variables, expressions = [], {}
    for i, definition in enumerate(source):
        scale, offset, unit = coordinate(definition)
        variable = {**deepcopy(definition), "id": f"source_{i}", "unit": unit, "domain": None}
        variables.append(variable)
        expressions[definition["column"]] = affine({"var": variable["id"]}, scale, offset)
    mapping = (context.get("input_geometry") or {}).get("map")
    if mapping is not None:
        if mapping.get("kind") != "affine_chart":
            raise NotImplementedError("Source expression supports frozen affine charts only")
        if mapping.get("source_columns") != columns:
            raise ValueError("Frozen chart source order differs")
        targets = mapping["target_columns"]
        if len(targets) != len(set(targets)):
            raise ValueError("Duplicate chart targets")
        matrix, offset = _affine_array(mapping["forward"], len(source), len(targets))
        originals, expressions = expressions, {}
        for column, row, bias in zip(targets, matrix, offset, strict=True):
            terms = [affine(deepcopy(originals[v["column"]]), float(c), 0)
                     for c, v in zip(row, source, strict=True) if c != 0]
            if bias != 0 or not terms:
                terms.append({"const": float(bias)})
            tree = terms[0]
            for term in terms[1:]:
                tree = op("add", tree, term)
            expressions[column] = tree
    substitutions = {v["id"]: expressions[v["column"]] for v in context["variables"]}
    scale, offset, unit = coordinate(context["target"])
    if residual:
        variables.append({**deepcopy(context["target"]), "id": "source_target", "unit": unit, "domain": None})
        substitutions["target"] = affine({"var": "source_target"}, scale, offset)
    return substitutions, variables, scale, offset


def source_equation(equation, context, *, residual=False):
    """Compose a saved working equation with frozen public coordinates.

    Coefficient identity/value/unit and arithmetic are retained. Returned source
    variables have no inferred domain. Unit-role validation remains separate.
    Template binding metadata applies to the working equation, not this derived AST.
    """
    from .expressions import inspect_equation

    substitutions, variables, scale, offset = _source_expression_map(context, residual)
    def replace(node):
        if "var" in node:
            return deepcopy(substitutions[node["var"]])
        result = deepcopy(node)
        if "args" in result:
            result["args"] = [replace(child) for child in node["args"]]
        return result
    working_variables = list(context["variables"])
    if residual:
        working_variables.append({**context["target"], "id": "target"})
    inspect_equation(equation, working_variables)
    ast = replace(equation["ast"])
    if not residual:
        if offset != 0:
            ast = {"op": "sub", "args": [ast, {"const": offset}]}
        if scale != 1:
            ast = {"op": "div", "args": [ast, {"const": scale}]}
    result = {"schema_version": 2, "ast": ast, "parameters": deepcopy(equation["parameters"]),
              "arithmetic": equation.get("arithmetic", "strict_real")}
    if equation.get("definitions"):
        result["definitions"] = {name: replace(node) for name, node in equation["definitions"].items()}
    inspect_equation(result, variables)
    return result, variables


def source_complexity_costs(context, *, residual=False):
    """Costs derived from exactly the same source AST substitution as rendering."""
    substitutions, _, scale, offset = _source_expression_map(context, residual)
    def count(node):
        return 1 + sum(count(child) for child in node.get("args", []))
    return {key: count(value) for key, value in substitutions.items()}, (0 if residual else 2 * ((scale != 1) + (offset != 0)))


def source_complexity(equation, context, *, residual=False):
    """Expanded user-coordinate AST size; shared by checks, ranking and output."""
    from .expressions import inspect_equation
    source, variables = source_equation(equation, context, residual=residual)
    return inspect_equation(source, variables)["n_nodes"]
