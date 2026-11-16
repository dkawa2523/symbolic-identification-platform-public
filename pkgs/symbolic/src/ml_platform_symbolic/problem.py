"""Canonical common requirements and the single legacy configuration boundary.

Stored problems have one requirement authority. Existing numerical/native
functions receive a derived configuration projection, never a second editable
source. Legacy documents are read without rewriting their artifacts.
"""

import hashlib
import json
from copy import deepcopy

_LIMITS = ("max_nodes", "max_free_parameters", "require_units", "allowed_variables", "equation_template")
_RESPONSES = {"points": "probes", "bounds": "response_bounds", "checks": "response_checks", "enforcement": "response_enforcement"}


def requirement_identity(requirements):
    return hashlib.sha256(json.dumps(requirements, sort_keys=True, allow_nan=False).encode()).hexdigest()


def unpack_requirements(config):
    """Resolve new public input once; reject two editable sources for one request."""
    config = deepcopy(dict(config))
    requested = config.pop("requirements", None)
    if requested is None:
        return config
    allowed = {"allowed_operators", "fixed_constants", "required_responses", "unit_search", *_LIMITS}
    if not isinstance(requested, dict) or set(requested) - allowed:
        raise ValueError("Unknown common requirements; use semantic operators, fixed constants and response requests")

    def assign(section, key, value):
        values = config.setdefault(section, {})
        if key in values:
            raise ValueError(f"Requirement is declared twice: requirements and {section}.{key}")
        values[key] = deepcopy(value)

    if "allowed_operators" in requested:
        assign("search", "operators", requested["allowed_operators"])
    if "fixed_constants" in requested:
        assign("data", "fixed_constants", requested["fixed_constants"])
    for key in (*_LIMITS, "unit_search"):
        if key in requested:
            assign("constraints", key, requested[key])
    responses = requested.get("required_responses", {})
    if not isinstance(responses, dict) or set(responses) - {*_RESPONSES, "sampling"}:
        raise ValueError("required_responses accepts points, bounds, checks, enforcement and sampling")
    responses = deepcopy(responses)
    # Resolve new common input once, before per-engine defaults can diverge.
    # Frozen documents and legacy constraints bypass this public-input boundary.
    has_response = bool(responses.get("points") or responses.get("checks")) or responses.get("bounds", [None, None]) != [None, None]
    if has_response and config.get("search", {}).get("response_mode", "search") == "search":
        if responses.get("enforcement") is None:
            responses["enforcement"] = "hard"
    for key, value in responses.items():
        assign("constraints", "probe_sampling" if key == "sampling" else _RESPONSES[key], value)
    return config


def requirements_from_config(config):
    constraints = config["constraints"]
    return {
        "allowed_operators": deepcopy(config["search"]["operators"]),
        "fixed_constants": deepcopy(config["data"]["fixed_constants"]),
        **{key: deepcopy(constraints.get(key)) for key in _LIMITS},
        **({"unit_search": True} if constraints.get("unit_search") else {}),
        "required_responses": {key: deepcopy(constraints.get(value)) for key, value in _RESPONSES.items()},
    }


def problem_document(config):
    """Store canonical working-coordinate requirements without legacy aliases."""
    config = deepcopy(config)
    requirements = requirements_from_config(config)
    del config["search"]["operators"]
    del config["data"]["fixed_constants"]
    del config["constraints"]
    config["_prepared_requirements_ref"] = requirement_identity(requirements)
    return {
        "problem_schema_version": 2,
        "config": config,
        "requirements": requirements,
        "requirements_sha256": requirement_identity(requirements),
    }


def problem_requirements(document):
    if "problem_schema_version" not in document:
        return requirements_from_config(document["config"])
    if document["problem_schema_version"] not in {1, 2}:
        raise ValueError("Unsupported problem schema")
    requirements = document["requirements"]
    if requirement_identity(requirements) != document["requirements_sha256"]:
        raise ValueError("Common requirements changed after problem preparation")
    return deepcopy(requirements)


def requirement_reference(document):
    return {
        "requirements_sha256": requirement_identity(problem_requirements(document)),
        "scope": "canonical common request; native settings/checks are derived evidence",
    }


def problem_config(document):
    """Derived in-memory projection for existing algorithms, never saved as a new authority."""
    config = deepcopy(document["config"])
    if "problem_schema_version" not in document:
        return config
    requirements = problem_requirements(document)
    marker = config.pop("_prepared_requirements_ref", None)
    if document["problem_schema_version"] == 2 and marker != requirement_identity(requirements):
        raise ValueError("Prepared configuration lost its authoritative requirement reference")
    config["search"]["operators"] = requirements["allowed_operators"]
    config["data"]["fixed_constants"] = requirements["fixed_constants"]
    config["constraints"] = {
        **{key: requirements.get(key) for key in _LIMITS},
        **({"unit_search": requirements["unit_search"]} if "unit_search" in requirements else {}),
        **{value: requirements["required_responses"].get(key) for key, value in _RESPONSES.items()},
    }
    return config


def load_problem(path):
    """Public prepared-document reader; validates requirements without adding defaults."""
    from pathlib import Path

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problem_config(document)
    return document
