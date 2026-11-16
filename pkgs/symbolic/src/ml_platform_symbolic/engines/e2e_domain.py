"""Opt-in E2E SLSQP iterate admission, not decoder/optimizer conditioning."""

import hashlib
import json
import time
from copy import deepcopy

import numpy as np

from ..coordinates import source_equation
from ..domains import rational_domain_check, validate_bounds

OPTION = "rational_domain_admission"


def prepare_admission(request, context, compiled):
    """Validate explicit public-source interval and finite campaign proof budgets.

    This does not expand or replace the existing prediction/Train domain. The
    interval is directly declared in public source-column coordinates: no inverse
    endpoint conversion or observed-data range is used to infer proof coverage.
    """
    value = request.options.get(OPTION)
    if value is None:
        return None
    required = {"column", "interval", "max_calls", "max_total_operations"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{OPTION} requires exactly {sorted(required)}")
    if not isinstance(value["column"], str) or not value["column"]:
        raise ValueError("Rational admission column must name one public source column")
    interval = value["interval"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("Rational admission requires a finite closed interval [lower, upper]")
    validate_bounds(interval)
    if any(isinstance(v, bool) or not isinstance(v, int | float) or not np.isfinite(v) for v in interval):
        raise ValueError("Rational admission requires finite numeric closed endpoints")
    for key in ("max_calls", "max_total_operations"):
        if type(value[key]) is not int or value[key] < (2 if key == "max_total_operations" else 1):
            raise ValueError(f"Rational admission {key} must be a positive finite integer budget")
    if compiled is None:
        raise NotImplementedError(
            "Rational admission currently requires finite-point response search/SLSQP; domain-only and filter paths are unsupported"
        )
    coordinates = {**context.provenance, "variables": request.variables, "target": request.target}
    # Reuse the real source composition boundary before loading native weights.
    _, variables = source_equation({"schema_version": 2, "ast": {"const": 0}, "parameters": {}}, coordinates)
    if len(variables) != 1 or variables[0]["column"] != value["column"]:
        raise NotImplementedError("Rational admission requires exactly one matching public source input column")
    return Admission(value, coordinates, context.deadline, variables[0].get("unit"))


class Admission:
    """One sequential discovery-local budget shared across all structure fits."""

    def __init__(self, policy, coordinates, deadline, public_unit):
        self.policy = deepcopy(policy)
        self.coordinates = deepcopy(coordinates)
        self.deadline = deadline
        self.public_unit = public_unit
        self.calls = self.operations = self.invocations = 0
        self.seconds = 0.0
        self.budget_denials = 0
        self.statuses = {key: 0 for key in ("safe", "unsafe", "unknown", "out_of_scope")}
        self.stages = {}
        self.last = None

    def begin(self):
        """Avoid attributing a previous candidate's proof to a fit failing early."""
        self.last = None

    def unavailable_reason(self):
        if self.calls >= self.policy["max_calls"]:
            return "discovery_gate_call_budget"
        if self.policy["max_total_operations"] - self.operations < 2:
            return "discovery_gate_operation_budget"
        return None

    def __call__(self, equation, stage):
        self.invocations += 1
        self.stages[stage] = self.stages.get(stage, 0) + 1
        start = time.monotonic()
        remaining = self.policy["max_total_operations"] - self.operations
        if self.calls >= self.policy["max_calls"]:
            result = {"status": "unknown", "reason": "discovery_gate_call_budget"}
        elif remaining < 2:
            result = {"status": "unknown", "reason": "discovery_gate_operation_budget"}
        elif self.deadline is not None and start >= self.deadline:
            result = {"status": "unknown", "reason": "discovery_deadline_before_proof"}
        else:
            self.calls += 1
            try:
                source, variables = source_equation(equation, self.coordinates)
                variables[0]["domain"] = deepcopy(self.policy["interval"])
                # Reserve one charge: common gate reports limit+1 when rejecting
                # the next operation, so even attempted-charge totals stay bounded.
                result = rational_domain_check(source, variables, max_operations=min(20000, remaining - 1))
                if result["status"] == "safe" and stage in {"final", "fixed"}:
                    for name, value in [("working_equation", equation), ("source_equation", source)]:
                        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                        result[name + "_sha256"] = hashlib.sha256(payload).hexdigest()
            except (ValueError, TypeError, NotImplementedError, KeyError) as error:
                result = {"status": "unknown", "reason": "source_composition_failed", "detail": str(error)}
            self.operations += result.get("arithmetic_operations", 0)
        if result.get("reason") in {
            "discovery_gate_call_budget",
            "discovery_gate_operation_budget",
            "discovery_deadline_before_proof",
        }:
            self.budget_denials += 1
        # A shared remainder can shrink this call's local proof limit. If that
        # last call exhausts it, classify discovery incompleteness even when no
        # later candidate/final callback remains to observe the empty budget.
        if (
            result.get("reason") == "arithmetic_operation_budget"
            and self.operations >= self.policy["max_total_operations"]
        ):
            self.budget_denials += 1
        self.seconds += time.monotonic() - start
        self.statuses[result["status"]] += 1
        self.last = {"stage": stage, **result}
        return result["status"] == "safe"

    def last_result(self):
        return deepcopy(self.last)

    def certificate(self):
        if self.last is None or self.last["status"] != "safe" or self.last["stage"] not in {"final", "fixed"}:
            raise ValueError("No final safe rational-domain evidence")
        return {
            "policy": deepcopy(self.policy),
            "coordinates": "public source column and unit",
            "public_source_unit": self.public_unit,
            "stage": "SLSQP retained-iterate admission and final recheck; fixed equations checked",
            "decoder_conditioned": False,
            "solver_objective_conditioned": False,
            "proof": self.last_result(),
        }

    def summary(self):
        return {
            "policy": deepcopy(self.policy),
            "coordinates": "public source column and unit",
            "public_source_unit": self.public_unit,
            "stage": "finite-point-feasible improving iterate retention and final recheck",
            "decoder_conditioned": False,
            "solver_objective_conditioned": False,
            "unproved_native_candidates": "archived only; excluded from returned candidate pool",
            "non_safe_policy": "unsafe, unknown and out_of_scope are not admitted; no finite-point fallback",
            "unavailable_reason": self.unavailable_reason(),
            "budget_denials": self.budget_denials,
            "invocations": self.invocations,
            "proof_calls": self.calls,
            "arithmetic_operation_charges": self.operations,
            "elapsed_seconds": self.seconds,
            "statuses": deepcopy(self.statuses),
            "stages": deepcopy(self.stages),
            "per_call_limits": {"max_nodes": 256, "max_degree": 32, "max_integer_bits": 4096, "max_operations": 20000},
            "not_certified": ["IEEE_finiteness", "conditioning", "monotonicity", "physical_validity"],
            "trajectory": "smooth residual/Jacobian and proposals unchanged; returned best may differ; added cost can reduce fits under a wall deadline",
        }
