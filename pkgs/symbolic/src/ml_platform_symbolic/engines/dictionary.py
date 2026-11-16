"""Small forward descriptor search: bounded function space, no pretrained assets."""

from itertools import combinations

import numpy as np

from ..constraints import equation_unit
from ..expression_graph import polynomial_node
from ..expressions import evaluate
from ..metrics import prediction_metrics, weighted_rms
from ..units import coherent_unit_label, unit_spec
from .base import DiscoveryResult, checked_candidate, validate_options

SUPPORTED_OPTIONS = {
    "max_degree",
    "max_terms",
    "ridge",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": ["add", "sub", "mul", "div", "pow_int", "log", "exp", "sqrt", "abs", "neg"],
        "constraints": {
            "operators": "enforced during construction",
            "units": "post-filtered",
            "max_nodes": "post-filtered",
            "variables": "enforced during construction",
            "allowed_variables": "enforced during term construction before coefficient fitting",
            "probes": "post-filtered",
        },
        "requires_assets": False,
    }


def _power_terms(request, variables):
    terms = []
    if "pow_int" in request.operators:
        terms.extend(
            {"op": "pow_int", "args": [v, {"const": p}]}
            for v in variables
            for p in range(2, request.options.get("max_degree", 3) + 1)
        )
    elif "mul" in request.operators:
        for variable in variables:
            power = variable
            for _ in range(2, request.options.get("max_degree", 3) + 1):
                power = {"op": "mul", "args": [power, variable]}
                terms.append(power)
    return terms


def _terms(request, name):
    allowed = request.limits.get("allowed_variables")
    variables = [{"var": v["id"]} for v in request.variables if allowed is None or v["id"] in allowed]
    if name == "mean":
        return [{"const": 1.0}]
    terms = [{"const": 1.0}, *variables]
    if name == "linear":
        return terms
    terms.extend(_power_terms(request, variables))
    if "mul" in request.operators:
        terms.extend({"op": "mul", "args": [a, b]} for a, b in combinations(variables, 2))
    for op in ("log", "sqrt", "exp", "abs"):
        if op in request.operators:
            terms.extend({"op": op, "args": [v]} for v in variables)
    if "div" in request.operators:
        terms.extend({"op": "div", "args": [{"const": 1.0}, v]} for v in variables)
    return terms


def _equation(terms, coefficients, request):
    params, features, pieces = {}, [], []
    target_unit = unit_spec(request.target.get("unit"))
    for i, (term, value) in enumerate(zip(terms, coefficients, strict=True)):
        key = f"c{i}"
        basis_unit, unit_errors = equation_unit({"schema_version": 1, "ast": term, "parameters": {}}, request.variables)
        coefficient_unit = None
        if target_unit is not None and basis_unit is not None and not unit_errors:
            coefficient_unit = coherent_unit_label(target_unit.product(basis_unit, -1))
        params[key] = {
            "value": float(value),
            "trainable": True,
            "origin": "baseline_regression_solve",
            "bounds": [None, None],
            # The additive basis defines target/basis dimensions. This is a
            # model construction convention, not evidence of a physical mechanism.
            "unit": request.target["unit"] if term == {"const": 1.0} else coefficient_unit,
            "unit_origin": "declared target and known additive basis; unknown basis stays unknown",
        }
        powers = []
        if term != {"const": 1.0}:
            features.append(term)
            powers = [[len(features) - 1, 1]]
        pieces.append({"coefficient": {"param": key}, "powers": powers})
    return {"schema_version": 2, "ast": polynomial_node(features, pieces), "parameters": params}


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    validate_options(request.options, SUPPORTED_OPTIONS)
    degree, count = request.options.get("max_degree", 3), request.options.get("max_terms", 6)
    ridge = request.options.get("ridge", 1e-8)
    if (
        type(degree) is not int or not 1 <= degree <= 8
        or type(count) is not int or not 1 <= count <= 12
        or isinstance(ridge, bool) or not isinstance(ridge, (int, float))
        or not np.isfinite(ridge) or ridge < 0
    ):
        raise ValueError("dictionary max_degree=1..8, max_terms=1..12, ridge>=0")
    return count, ridge


def discover(request, context, *, name="dictionary"):
    count, ridge = validate_request(request, context)
    terms, columns = [], []
    for term in _terms(request, name):
        values = evaluate({"schema_version": 1, "ast": term, "parameters": {}}, request.X, request.variables)
        if np.isfinite(values).all():
            terms.append(term)
            columns.append(values)
    design = np.column_stack(columns)
    # Preserve the usual L2-normalized baseline without a unit-dependent
    # floor or dimensional squaring. Fall back to max scaling only when the
    # L2 magnitude itself is unrepresentable.
    with np.errstate(over="ignore"):
        scale = np.asarray([weighted_rms(column) * np.sqrt(len(column)) for column in columns])
    maximum = np.max(abs(design), axis=0)
    scale = np.where(np.isfinite(scale), scale, maximum)
    scale = np.where(scale == 0, 1.0, scale)
    normalized = design / scale
    selected, candidates, rejected = [], [], []
    for step in range(min(count, len(terms))):
        best = None
        for index in range(len(terms)):
            if index in selected:
                continue
            indices = [*selected, index]
            matrix = normalized[:, indices]
            coefficients = (
                np.linalg.lstsq(
                    np.vstack([matrix, np.sqrt(ridge) * np.eye(len(indices))]),
                    np.r_[request.y, np.zeros(len(indices))],
                    rcond=None,
                )[0]
                / scale[indices]
            )
            prediction = design[:, indices] @ coefficients
            # RMSE has the same ordering as MSE without squaring dimensional
            # values that may underflow/overflow. This is our baseline's own
            # Train selection, not a replacement for any native solver.
            loss = prediction_metrics(request.y, prediction)["rmse"]
            if loss is None:
                continue
            if best is None or loss < best[0]:
                best = loss, index, coefficients, indices
        if best is None:
            break
        _, index, coefficients, indices = best
        selected.append(index)
        equation = _equation([terms[i] for i in indices], coefficients, request)
        # Independently evaluate the native fitted design on held training probes.
        chosen_terms, weights = [terms[i] for i in indices], coefficients.copy()

        def native(X, terms=chosen_terms, weights=weights):
            basis = np.column_stack(
                [evaluate({"schema_version": 1, "ast": t, "parameters": {}}, X, request.variables) for t in terms]
            )
            return basis @ weights

        try:
            candidates.append(checked_candidate(name, step, equation, request, native, native_ieee=False))
        except (ValueError, TypeError, OverflowError) as exc:
            # One unresolvable export does not invalidate earlier native fits.
            # Preserve the actual fitted coordinates; never loosen fidelity.
            rejected.append({"index": step, "reason": str(exc), "equation": equation})
        if name == "mean":
            break
    return DiscoveryResult(
        candidates,
        "ok" if candidates else "failed",
        {
            "native_candidates": len(candidates) + len(rejected),
            "rejected_exports": rejected,
            "dictionary_terms": len(terms),
            "train_selection_loss": "stable RMSE; equivalent ordering to squared error",
            "normalization": "L2 with no physical-unit floor; max magnitude if L2 is unrepresentable",
            "constraint_support": capabilities()["constraints"],
        },
    )
