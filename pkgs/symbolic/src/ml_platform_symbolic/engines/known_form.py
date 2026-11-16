"""Explicit user-supplied family reference; coefficient fitting, not discovery."""

from copy import deepcopy
from importlib.metadata import version

import numpy as np

from ..expressions import ARITY, evaluate, inspect_equation
from ..fitting import fit_coefficients
from ..units import quantity_conversion, quantity_role
from .base import DiscoveryResult, checked_candidate, validate_options

SUPPORTED_OPTIONS = {
    "equation",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": list(ARITY),
        "requires_assets": False,
        "fixed_constants": True,
        "equation_template": True,
        "constraints": {
            "operators": "user template; common post-filter",
            "units": "explicit template plus structural post-filter",
            "fixed_constants": "immutable authoritative values/units",
            "coefficient_bounds": "SciPy fit bounds",
            "probes": "common post-filter",
            "max_nodes": "common post-filter",
        },
    }


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    validate_options(request.options, SUPPORTED_OPTIONS)
    template = getattr(request, "equation_template", None)
    if template is not None:
        if "equation" in request.options:
            raise ValueError("Declare the known form once in requirements.equation_template, not also engine_options")
        if template.get("slots"):
            raise NotImplementedError(
                "known_form fits a complete supplied template; unknown functions require a template search adapter"
            )
        from ..equation_templates import instantiate_template

        equation, _ = instantiate_template(template, {}, variables=request.variables, target=request.target)
    else:
        if "equation" not in request.options:
            raise ValueError("known_form requires an equation or equation_template")
        equation = deepcopy(request.options["equation"])
    for parameter in equation["parameters"].values() if template is None else []:
        parameter.setdefault("origin", "user_declared_family")
    for name, definition in request.fixed_constants.items() if template is None else []:
        existing = equation["parameters"].get(name)
        if existing is not None:
            role = quantity_role(definition)
            existing_role = quantity_role(existing)
            if role is not None and existing_role is not None and role != existing_role:
                raise ValueError("Immutable constant point/difference meanings conflict")
            a, b = quantity_conversion(existing.get("unit"), definition["unit"], role)
            converted = existing["value"] * a + b
            if existing["trainable"] or not np.isclose(
                converted, definition["value"], rtol=4 * np.finfo(float).eps, atol=0
            ):
                raise ValueError(f"User template conflicts with declared immutable constant: {name}")
        value = definition["value"]
        equation["parameters"][name] = {
            "value": value,
            "unit": definition["unit"],
            "trainable": False,
            "origin": "declared_fixed_constant",
            **(
                {"coordinate_role": definition["coordinate_role"]}
                if definition.get("coordinate_role") is not None
                else {}
            ),
            "bounds": [value, value],
        }
    inspect_equation(equation, request.variables)
    return equation, template


def discover(request, context):
    equation, template = validate_request(request, context)
    fitted = fit_coefficients(equation, request.X, request.y, request.variables)
    candidate = checked_candidate(
        "known_form", 0, fitted, request, lambda X: evaluate(fitted, X, request.variables), native_ieee=False
    )
    candidate.native_check["kind"] = "shared_evaluator_reference; no independent engine export oracle"
    return DiscoveryResult(
        [candidate],
        "ok",
        {
            "method_kind": "user_supplied_family_reference",
            "native_candidates": 1,
            "scipy": version("scipy"),
            "fit_rows": len(request.X),
            "fit_provenance": "shared SciPy least_squares on Train only, marked free coefficients only",
            "structure_provenance": (
                "authoritative requirements.equation_template; complete structure, no unknown-function discovery"
                if template is not None
                else "explicit user template; additional prior knowledge, not supplied to discovery engines"
            ),
            "constraint_support": capabilities()["constraints"],
        },
    )
