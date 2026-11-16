"""Separate E2E known-affine reduction and explicit heuristic hybrid variant.

The ordinary e2e adapter can reduce one exactly observed affine slot. The explicit
legacy e2e_template variant still proposes shapes from original y and jointly fits
later; it is not the same structure-generation mechanism.
"""

import time
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np

from ..coordinates import source_complexity, source_complexity_costs
from ..data import write_json
from ..domains import domain_variables, expression_bounds, region_mask, validate_input_domain
from ..equation_templates import instantiate_template, validate_template
from ..expressions import evaluate, inspect_equation
from ..fitting import fit_coefficients
from ..response_search import response_bounds, response_feasible, response_values
from . import pretrained_template
from .base import DiscoveryResult, checked_candidate
from .e2e_response import prepare_response_fit


def capabilities():
    return pretrained_template.capabilities("e2e")


def validate_request(request, context):
    return pretrained_template.validate_request(request, context, engine="e2e")


def discover(request, context):
    return pretrained_template.discover(request, context, engine="e2e")


def prepare_known_affine(request, context):
    coordinate_context = {
        **context.provenance,
        "variables": request.variables,
        "target": {"column": "target", **request.target},
    }
    variable_costs, output_cost = source_complexity_costs(coordinate_context)
    if output_cost != 0 or any(c != 1 for c in variable_costs.values()):
        raise NotImplementedError("Nonidentity source coordinates need separate slot-local and source prefix budgets")
    validate_input_domain(request.X, request.variables)
    if request.domain is not None and request.domain["columns"] != [v["column"] for v in request.variables]:
        raise ValueError("Original domain column order differs")
    if not region_mask(request.X, request.domain).all():
        raise ValueError("Original Train is outside the declared joint domain")
    template = deepcopy(request.equation_template)
    if not isinstance(template, dict) or not isinstance(template.get("slots"), dict) or len(template["slots"]) != 1:
        raise NotImplementedError(
            "E2E does not support arbitrary equation templates; exactly one observed affine slot is required"
        )
    template = validate_template(
        template, request.variables, request.target, request.operators, request.fixed_constants
    )
    name, slot = next(iter(template["slots"].items()))
    declarations = [*request.variables, request.target, slot, *slot["arguments"], *template["parameters"].values()]
    if any(d.get("unit") != "1" for d in declarations):
        raise NotImplementedError("Only explicitly dimensionless declarations are supported")
    if any(p.get("trainable") or "derived_from" in p for p in template["parameters"].values()):
        raise NotImplementedError("Unknown outer coefficients and dependent parameters are unsupported")
    if len(slot["arguments"]) != len(request.variables):
        raise NotImplementedError("Slot must observe the complete unchanged original X")
    expected = [{"var": v["id"]} for v in request.variables]
    calls = 0

    def coefficient(node):
        nonlocal calls
        if "ref" in node:
            return coefficient(template["components"][node["ref"]]["ast"])
        if "slot" in node:
            if node["slot"] != name or node["args"] != expected:
                raise NotImplementedError("Every shared call must use complete X in the identical order")
            calls += 1
            return 1.0, None
        children = node.get("args", [])
        if not children:
            return 0.0, node
        parts = [coefficient(child) for child in children]
        if not any(a != 0 or b is None for a, b in parts):
            return 0.0, node
        op = node["op"]
        if op in {"add", "sub"}:
            sign = 1 if op == "add" else -1
            return parts[0][0] + sign * parts[1][0], None
        if op in {"mul", "div"}:
            if op == "mul" and parts[0][0] == 0 and parts[0][1] is not None:
                known, unknown = children[0], parts[1]
            elif parts[1][0] == 0 and parts[1][1] is not None:
                known, unknown = children[1], parts[0]
            else:
                raise NotImplementedError("Products/quotients of unknown slot outputs are unsupported")

            def has_variable(value):
                if "ref" in value:
                    return has_variable(template["components"][value["ref"]]["ast"])
                return "var" in value or any(has_variable(x) for x in value.get("args", []))

            if has_variable(known):
                raise NotImplementedError("Variable-dependent slope changes native least-squares weights")

            def expand_known(value):
                if "ref" in value:
                    return expand_known(template["components"][value["ref"]]["ast"])
                return (
                    {**value, "args": [expand_known(a) for a in value["args"]]} if "args" in value else deepcopy(value)
                )

            value = evaluate(
                {"schema_version": 2, "ast": expand_known(known), "parameters": template["parameters"]},
                request.X[:1],
                request.variables,
            )[0]
            if not np.isfinite(value) or (op == "div" and value == 0):
                raise ValueError("Known multiplier/divisor must be finite and divisor nonzero")
            return (unknown[0] / value if op == "div" else unknown[0] * value), None
        raise NotImplementedError("Nonlinear unknown-slot inverse transformations are unsupported")

    slope, _ = coefficient(template["ast"])
    if not np.isfinite(slope) or slope == 0 or not calls:
        raise ValueError("Known aggregate slope must be finite and nonzero; unknown is otherwise unidentified")
    zero = {"schema_version": 2, "ast": {"const": 0.0}, "parameters": {}}
    baseline, _ = instantiate_template(template, {name: zero}, variables=request.variables, target=request.target)
    bounded_variables = domain_variables(request.variables, request.domain)

    def finite_intermediates(node):
        for child in node.get("args", []):
            finite_intermediates(child)
        if not np.isfinite(expression_bounds(node, baseline["parameters"], bounded_variables)).all():
            raise NotImplementedError("Known offset cannot be certified finite throughout declared axis bounds")

    finite_intermediates(baseline["ast"])
    allowed = request.limits.get("allowed_variables")
    if allowed is not None and not inspect_equation(baseline, request.variables)["variables"] <= set(allowed):
        raise NotImplementedError("Known outer AST references forbidden input variables; no unconstrained fallback")
    offset = evaluate(baseline, request.X, request.variables)
    target = (request.y - offset) / slope
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite exact reduced observations; no rows may be removed")
    # Re-evaluate the ORIGINAL outer operation order, not only collapsed a*t+b.
    # E.g. (h+1e16)-1e16 cannot encode an observed response of 1 at h=1.
    observed_id = "__e2e_observed_slot"
    while observed_id in {v["id"] for v in request.variables}:
        observed_id += "_"

    def observed_outer(node):
        if "ref" in node:
            return observed_outer(template["components"][node["ref"]]["ast"])
        if "slot" in node:
            return {"var": observed_id}
        return {**node, "args": [observed_outer(a) for a in node["args"]]} if "args" in node else deepcopy(node)

    restored = evaluate(
        {"schema_version": 2, "ast": observed_outer(template["ast"]), "parameters": template["parameters"]},
        np.column_stack([request.X, target]),
        [*request.variables, {"id": observed_id, "unit": "1"}],
    )
    tolerance = 64 * np.finfo(float).eps * np.maximum(np.abs(request.y), np.finfo(float).tiny)
    if not np.all(np.abs(restored - request.y) <= tolerance):
        raise ValueError("Float64 subtraction loses original-response resolution; reduction rejected")
    # Known outer domain is retained in baseline, never cancelled symbolically.
    compiled = prepare_response_fit(request)
    if (
        compiled is not None
        and not np.isfinite(evaluate(baseline, np.asarray(compiled["points"]), request.variables)).all()
    ):
        raise ValueError("Known outer expression is undefined at a declared constraint point")
    from .e2e_domain import prepare_admission
    admission = prepare_admission(request, context, compiled)
    known_nodes = source_complexity(baseline, coordinate_context) - calls
    budget = slot["max_nodes"]
    if request.limits.get("max_nodes") is not None:
        budget = min(budget, (request.limits["max_nodes"] - known_nodes) // calls)
    if budget < 1:
        raise ValueError("Expanded known outer expression leaves no slot node budget")
    limits = {
        **request.limits,
        "max_nodes": budget,
        "response_mode": "filter",
        "response_enforcement": None,
        "probes": [],
        "response_checks": [],
        "response_bounds": [None, None],
        "equation_template": None,
    }
    proposed = replace(
        request,
        y=target,
        operators=list(slot["allowed_operators"]),
        limits=limits,
        fixed_constants={},
        equation_template=None,
        # The policy is on the original full response, never on reduced slot proposals.
        options={k: v for k, v in request.options.items() if k != "rational_domain_admission"},
    )
    return proposed, {
        "domain_admission": admission,
        "template": template,
        "slot": name,
        "slope": slope,
        "calls": calls,
        "offset_equation": baseline,
        "compiled_response": compiled,
        "slot_working_node_budget": budget,
        "coordinate_context": deepcopy(coordinate_context),
        "source_budget": "identity source coordinates verified; repeated expanded slot ASTs count separately",
        "loss_identity": "MSE(original response) = slope**2 * MSE(reduced observations)",
        "scope": "exact affine observations and source budget for verified unit-cost source coordinates; no outer-AST/response-conditioned decoding",
    }


def fit_known_affine_candidate(native_equation, request, plan, *, max_nfev=1000, diagnostics=None):
    """New common strict-real variant, preserving original native equation separately."""
    body = deepcopy(native_equation)
    body["arithmetic"] = "strict_real"
    formal = {
        v["id"]: a["name"]
        for v, a in zip(request.variables, plan["template"]["slots"][plan["slot"]]["arguments"], strict=True)
    }

    def rename(node):
        if "var" in node:
            return {"var": formal[node["var"]]}
        return {**node, "args": [rename(a) for a in node["args"]]} if "args" in node else deepcopy(node)

    body["ast"] = rename(body["ast"])
    for parameter in body["parameters"].values():
        if parameter.get("unit") is None and parameter.get("origin") == "native_decimal_coefficient":
            parameter["unit"] = "1"
    equation, evidence = instantiate_template(
        plan["template"], {plan["slot"]: body}, variables=request.variables, target=request.target
    )
    if any(c["required"] and c["status"] != "pass" for c in evidence["checks"]):
        raise ValueError("Declared slot structure, grammar, complexity or units failed")
    if (
        request.limits.get("max_nodes") is not None
        and source_complexity(equation, plan["coordinate_context"]) > request.limits["max_nodes"]
    ):
        raise ValueError("Expanded source-coordinate complexity exceeded")
    compiled = plan["compiled_response"]
    constraints = None
    if compiled is not None:
        points = np.asarray(compiled["points"])
        constraints = (
            lambda eq: response_values(evaluate(eq, points, request.variables), compiled),
            *response_bounds(compiled),
        )
    diagnostics = {} if diagnostics is None else diagnostics
    fitted = fit_coefficients(
        equation,
        request.X,
        request.y,
        request.variables,
        max_nfev=max_nfev,
        constraints=constraints,
        diagnostics=diagnostics,
        iterate_acceptance=plan.get("domain_admission"),
    )
    if not np.isfinite(evaluate(fitted, request.X, request.variables)).all():
        raise ValueError("Full original Train prediction is not finite")
    if compiled is not None and not response_feasible(evaluate(fitted, points, request.variables), compiled):
        raise ValueError("Original response constraints failed without relaxation")
    return fitted, diagnostics


def discover_known_affine(request, context):
    """Use native E2E on identified observations, then fit the original outer AST."""
    from . import e2e

    proposed, plan = prepare_known_affine(request, context)
    e2e.validate_request(proposed, context)
    directory = context.work_dir / "known_affine_native"
    directory.mkdir(parents=True, exist_ok=True)
    native = e2e.discover(proposed, replace(context, work_dir=directory, checkpoint=None))
    archive = write_json(directory / "candidates.json", [asdict(c) for c in native.candidates])
    write_json(directory / "summary.json", native.summary)
    # Observation evidence is Train-only and never substitutes true slot structure.
    np.savez_compressed(
        directory / "observations.npz",
        X=proposed.X,
        original_y=request.y,
        reduced_y=proposed.y,
        row_ids=request.row_ids,
    )
    candidates, attempts = [], []
    summary = {
        "method_kind": "e2e_exact_affine_observation_reduction",
        "known_formula": {
            "slot": plan["slot"],
            "slope": plan["slope"],
            "calls": plan["calls"],
            "loss_identity": plan["loss_identity"],
            "slot_source_node_budget": plan["slot_working_node_budget"],
            "original_source_node_budget": request.limits.get("max_nodes"),
            "source_budget": plan["source_budget"],
        },
        "generation": "Pinned native E2E receives exact reduced Train observations and unchanged X; no unknown-slot pseudo-label heuristic",
        "template_stage": "known outer formula determines observations before generation; exact original outer AST is assembled afterwards",
        "decoder_conditioned_on_outer_ast": False,
        "decoder_conditioned_on_response": False,
        "response_stage": "original-y finite-point coefficient constraints on every returned native slot candidate; not decoder conditioning",
        "fidelity": "native parity belongs to archived slot candidates; assembled strict-real variants use common evaluator consistency only",
        "numerical_scope": "MSE proportionality is algebraic, not invariant pretrained proposals, BFGS tolerances, convergence or performance",
        "native_summary": native.summary,
        "attempts": attempts,
    }
    artifacts = {
        "native_slots": archive,
        "native_summary": str(directory / "summary.json"),
        "slot_observations": str(directory / "observations.npz"),
        **{"slot_" + key: value for key, value in native.native_artifacts.items()},
    }

    def result(termination):
        if plan.get("domain_admission") is not None:
            summary["rational_domain_admission"] = plan["domain_admission"].summary()
        summary.update(
            termination=termination,
            native_candidates=len(native.candidates),
            exported_candidates=len(candidates),
            attempted_candidates=len(attempts),
            unattempted_candidates=len(native.candidates) - len(attempts),
        )
        partial = (termination in {"wall_budget", "in_progress", "gate_budget"} or native.status == "partial"
                   or (plan.get("domain_admission") is not None and plan["domain_admission"].budget_denials > 0))
        if not candidates:
            summary["reason"] = (
                "Native E2E produced no portable slot candidates; no substitute"
                if termination == "no_native_slot_candidates"
                else "Deadline reached before an accepted original-y fit"
                if termination == "wall_budget"
                else "No accepted original-y known-affine fit; see attempts"
            )
        else:
            summary.pop("reason", None)
        return DiscoveryResult(
            list(candidates), "partial" if partial else "ok" if candidates else "failed", deepcopy(summary), artifacts
        )

    for parent in native.candidates:
        if context.deadline is not None and time.monotonic() >= context.deadline:
            return result("wall_budget")
        if plan.get("domain_admission") is not None and plan["domain_admission"].unavailable_reason() is not None:
            return result("gate_budget")
        row = {"parent_id": parent.candidate_id, "parent_refinement_type": parent.native_check.get("refinement_type")}
        if plan.get("domain_admission") is not None:
            plan["domain_admission"].begin()
        row["solver"] = {}
        try:
            equation, fit = fit_known_affine_candidate(
                parent.equation, request, plan, max_nfev=request.options.get("response_max_nfev", 1000),
                diagnostics=row["solver"]
            )
            candidate = checked_candidate(
                "e2e",
                len(candidates),
                equation,
                request,
                lambda X, equation=equation: evaluate(equation, X, request.variables),
                native_ieee=False,
            )
            candidate.candidate_id = parent.candidate_id + "/known-affine-fit"
            candidate.native_check.update(
                kind="original_y_known_affine_fit; common evaluator consistency, not native full-response parity",
                independent_native_oracle=False,
                parent_native_slot_id=parent.candidate_id,
                parent_refinement_type=row["parent_refinement_type"],
                native_slot_parity=deepcopy(parent.native_check),
                solver_converged=fit.get("success"),
                source_n_nodes=source_complexity(equation, plan["coordinate_context"]),
                finite_point_feasible=plan["compiled_response"] is not None,
            )
            if plan.get("domain_admission") is not None:
                candidate.native_check["rational_domain_admission"] = plan["domain_admission"].certificate()
            candidates.append(candidate)
            row.update(status="accepted", candidate_id=candidate.candidate_id, solver=fit)
        except (ValueError, TypeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            row.update(status="rejected", reason=str(error))
        if plan.get("domain_admission") is not None:
            row["rational_domain_admission"] = plan["domain_admission"].last_result()
        attempts.append(row)
        if context.checkpoint is not None:
            context.checkpoint(result("in_progress"))
    return result("native_candidates_exhausted" if native.candidates else "no_native_slot_candidates")
