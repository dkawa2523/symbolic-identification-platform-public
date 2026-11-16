"""Explicit Train estimators. Search, comparison and reporting never call these implicitly."""

from pathlib import Path

import numpy as np

from .data import digest, value_digest, write_json
from .expressions import inspect_equation
from .fitting import fit_coefficients
from .parameters import free_parameter_bounds, free_parameter_names, with_parameters
from .units import convert_width


def _linear_deming(equation, X, y, variables, options):
    """Known-error-ratio homoscedastic independent additive EIV, with an intercept."""
    if set(options) != {"feature", "slope", "intercept", "sigma_x", "sigma_y", "sigma_x_unit", "sigma_y_unit"}:
        raise ValueError("Deming needs explicit feature/parameter mapping and known x/y standard deviations with units")
    columns = [v["column"] for v in variables]
    index = columns.index(options["feature"])
    slope, intercept = options["slope"], options["intercept"]
    names = free_parameter_names(equation, require_origin=True)
    if set(names) != {slope, intercept} or slope == intercept:
        raise ValueError(
            "Linear Deming supports exactly two independent audited free parameters; fixed/tied restrictions are not discarded"
        )
    # A bounded, declared model family, not a numerical guess that an arbitrary
    # expression is affine. Commuted sums/products are accepted without rewriting.
    node = equation["ast"]
    if node.get("op") != "add":
        raise ValueError("Linear Deming requires the declared slope*x + intercept expression")
    terms = node["args"]
    product = next((n for n in terms if n.get("op") == "mul"), None)
    if (
        len(terms) != 2
        or {"param": intercept} not in terms
        or product is None
        or len(product["args"]) != 2
        or {"param": slope} not in product["args"]
        or {"var": variables[index]["id"]} not in product["args"]
    ):
        raise ValueError("Deming model mapping differs from the source expression; no silent structure substitution")
    sx = float(convert_width(options["sigma_x"], options["sigma_x_unit"], variables[index].get("unit")))
    return index, sx


def deming_line(equation, X, y, variables, target, options):
    # Keep target units outside the user option mapping: they are authoritative
    # prepared coordinates, not a second editable noise target.
    index, sx = _linear_deming(equation, X, y, variables, options)
    sy = float(convert_width(options["sigma_y"], options["sigma_y_unit"], target.get("unit")))
    if not np.isfinite([sx, sy]).all() or min(sx, sy) <= 0:
        raise ValueError("Deming requires strictly positive finite known standard deviations")
    x = np.asarray(X[:, index], float)
    y = np.asarray(y, float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Deming needs at least three finite paired observations")
    dx, dy = x - x.mean(), y - y.mean()
    xx, xy, yy = np.mean(dx * dx), np.mean(dx * dy), np.mean(dy * dy)
    ratio = (sy / sx) ** 2
    difference = yy - ratio * xx
    root = np.hypot(difference, 2 * np.sqrt(ratio) * xy)
    if xx <= 0:
        raise ValueError("No observed x variation for a finite y-on-x Deming slope")
    if xy == 0:
        if difference >= 0:
            raise ValueError("Vertical or orientation-undetermined Deming optimum; finite y-on-x form unavailable")
        b = 0.0
    else:
        b = (difference + root) / (2 * xy) if difference >= 0 else 2 * ratio * xy / (root - difference)
    a = float(y.mean() - b * x.mean())
    values = {options["slope"]: float(b), options["intercept"]: a}
    names = free_parameter_names(equation, require_origin=True)
    bounds = free_parameter_bounds(equation, names)
    if any(
        (lo is not None and values[n] < lo) or (hi is not None and values[n] > hi)
        for n, (lo, hi) in zip(names, bounds, strict=True)
    ):
        raise ValueError(
            "Unconstrained Deming optimum violates physical bounds; bounded EIV estimator is not implemented, no clipping"
        )
    result = with_parameters(equation, names, [values[n] for n in names])
    return result, {
        "family": "linear_deming_known_error_ratio",
        "sigma_x_working": sx,
        "sigma_y_working": sy,
        "variance_ratio": ratio,
        "assumptions": "independent additive homoscedastic x/y errors, known variance ratio, affine latent relation with intercept",
        "estimation_target": "latent affine relation parameters under declared observation assumptions; not general EIV or conditional E[y_observed|x_observed]",
    }


def fit_hypotheses(study_dir, *, run_dir, ids, estimator="least_squares", options=None):
    """Create named immutable fit variants, using Train only; never update native particles."""
    from .equation_evaluation.api import load_study, new_directory
    from .equation_evaluation.numerical import intersect_schema

    context, records, _, arrays = load_study(study_dir)
    known = {r["id"]: r for r in records}
    if not ids or not set(ids) <= known.keys():
        raise ValueError("Fit requires explicit IDs from this immutable study")
    if estimator not in {"least_squares", "linear_deming"}:
        raise ValueError("Unsupported estimator; no automatic fallback")
    directory = new_directory(run_dir)
    artifacts = []
    for index, name in enumerate(ids):
        record = known[name]
        if record["kind"] != "explicit" or not free_parameter_names(record["equation"], require_origin=True):
            raise ValueError("Point fitting requires an explicit equation with audited free parameters")
        variables, target = intersect_schema(record, context)
        if estimator == "least_squares":
            if options:
                raise ValueError("No extra least-squares options supported here")
            equation = fit_coefficients(record["equation"], arrays["train_X"], arrays["train_y"], variables)
            assumptions = {
                "family": "least_squares",
                "estimation_target": "best fixed-observed-X squared-error approximation; not latent physical coefficients with input error",
            }
        else:
            mapping = options or {}
            if set(mapping) != {"feature", "slope", "intercept"}:
                raise ValueError(
                    "Deming estimator options map feature/slope/intercept; known errors belong to Problem.observation_model"
                )
            observation = context["config"].get("observation_model") or {}
            errors = observation.get("errors", {})
            if errors.get("kind") != "independent_additive_known_sigma" or mapping["feature"] not in errors.get(
                "inputs", {}
            ):
                raise ValueError("Deming needs explicit known independent input and target errors in the problem")
            if (record.get("input_geometry") or {}).get("map"):
                raise ValueError(
                    "Deming on mixed chart coordinates needs the transformed full error model; scalar-error substitution is unsupported"
                )
            x_error, y_error = errors["inputs"][mapping["feature"]], errors["target"]
            resolved = {
                **mapping,
                "sigma_x": x_error["value"],
                "sigma_x_unit": x_error["unit"],
                "sigma_y": y_error["value"],
                "sigma_y_unit": y_error["unit"],
            }
            equation, assumptions = deming_line(
                record["equation"], arrays["train_X"], arrays["train_y"], variables, target, resolved
            )
            assumptions["declared_observation_model"] = observation
        inspect_equation(equation, variables)
        path = write_json(
            directory / f"fit_{index}.json",
            {
                "kind": "derived_equation",
                "variant": "train_point_fit" if estimator == "least_squares" else "train_" + estimator + "_fit",
                "equation": equation,
                "variables": variables,
                "target": target,
                "input_geometry": record.get("input_geometry"),
                "parent": {
                    "id": name,
                    "equation_sha256": value_digest(record["equation"]),
                    "study_sha256": digest(Path(study_dir) / "study.json"),
                },
                "data_sha256": context["data_sha256"],
                "split_sha256": context["split_sha256"],
                "fit_role": "train",
                "method": estimator,
                "observation_model": assumptions,
                "options": options or {},
                "meaning": "named estimator variant, not native search, posterior update, or source mutation",
            },
        )
        artifacts.append(str(path))
    write_json(directory / "fits.json", {"artifacts": artifacts, "estimator": estimator})
    return directory
