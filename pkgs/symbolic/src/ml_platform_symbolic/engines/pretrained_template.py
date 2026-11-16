"""Explicit pretrained proposals followed by joint real-valued template fitting.

The decoder is not conditioned on the template or response constraints. Its
original-response fits propose slot bodies only; they are not observations of
the unknown slot. Native candidates remain separate, immutable evidence.
"""

import time
from copy import deepcopy
from dataclasses import asdict, replace
from heapq import heappop, heappush

import numpy as np

from ..data import write_json
from ..equation_templates import instantiate_template, validate_template
from ..expressions import ARITY, evaluate, evaluate_native, inspect_equation
from ..fitting import fit_coefficients
from ..metrics import prediction_metrics
from ..response_search import compile_response_constraints, response_bounds, response_feasible, response_values
from .base import DiscoveryResult, checked_candidate, validate_options
from .pretrained_fitting import fit_starts
from .pretrained_proposals import collect_parts, full_problem_request
from .registry import get_engine

OPTIONS = {"proposal_options", "max_proposals", "max_combinations", "max_nfev", "proposal_coordinates", "proposal_mode", "fit_starts"}


def capabilities(engine):
    native = get_engine(engine).capabilities()
    return {
        "operators": list(ARITY), "proposal_operators": native["operators"], "requires_assets": True,
        "asset_engine": engine, "devices": native.get("devices", ["cpu"]),
        "supported_options": sorted(OPTIONS), "equation_template": True,
        "fixed_constants": "template_only",
        "search_constraints": ["probes", "response_bounds", "response_checks"],
        "constraints": {
            "equation_template": "assembled fixed outer structure and shared slot parameters; not decoder conditioning",
            "operators": "native vocabulary unchanged; slot proposals filtered against declared grammar before joint fit",
            "max_nodes": "slot proposal and full output checks; native generation can exceed the limit",
            "units": "explicit unit1 inputs, target and slots only; no unit relabeling",
            "probes": "explicit joint coefficient constraints; decoder proposals are unconstrained",
            "response_bounds": "explicit joint coefficient constraints; finite points only",
            "response_checks": "explicit joint coefficient finite-step constraints; no between-point guarantee",
        },
    }


def _bindings(template, *, all_calls=False):
    """First declared call per slot, with only fixed known argument expressions."""
    found = {}

    def argument(node):
        if "slot" in node:
            raise NotImplementedError("Pretrained slot proposals cannot condition on another unknown slot")
        if "ref" in node:
            return argument(template["components"][node["ref"]]["ast"])
        if "param" in node:
            parameter = template["parameters"][node["param"]]
            if "derived_from" in parameter:
                return argument(parameter["derived_from"])
            if parameter["trainable"]:
                raise NotImplementedError("Proposal input bindings cannot depend on unknown template parameters")
            return {"const": parameter["value"]}
        return {"op": node["op"], "args": [argument(a) for a in node["args"]]} if "op" in node else deepcopy(node)

    def visit(node):
        if "ref" in node:
            visit(template["components"][node["ref"]]["ast"])
        elif "slot" in node:
            args = [argument(a) for a in node["args"]]
            found.setdefault(node["slot"], []).append(args)
        else:
            for child in node.get("args", []):
                visit(child)

    visit(template["ast"])
    return found if all_calls else {name: calls[0] for name, calls in found.items()}


def _proposal_request(request, name, binding, options, index):
    slot = request.equation_template["slots"][name]
    X = np.column_stack([
        evaluate({"schema_version": 2, "ast": ast, "parameters": {}}, request.X, request.variables)
        for ast in binding
    ])
    if not np.isfinite(X).all():
        raise ValueError(f"Nonfinite proposal input for slot {name}; no Train rows removed")
    variables = [{"id": arg["name"], "column": arg["name"], "unit": "1"} for arg in slot["arguments"]]
    limits = {**request.limits, "response_mode": "filter", "probes": [], "response_bounds": [None, None],
              "response_checks": [], "equation_template": None, "require_units": False,
              "max_nodes": slot["max_nodes"]}
    return replace(request, X=X, variables=variables, target={"unit": "1"},
                   operators=list(slot["allowed_operators"]), limits=limits,
                   options=deepcopy(options["proposal_options"]), seed=(request.seed + index) % 2**32,
                   fixed_constants={}, equation_template=None, domain=None)


def validate_request(request, context, *, engine):
    validate_options(request.options, OPTIONS)
    options = {"proposal_options": {}, "max_proposals": 8, "max_combinations": 64,
               "max_nfev": 300, "fit_starts": 1, "proposal_coordinates": "native", "proposal_mode": "whole", **deepcopy(request.options)}
    if not isinstance(options["proposal_options"], dict):
        raise ValueError("proposal_options must be a native option mapping")
    for key in ("max_proposals", "max_combinations", "max_nfev", "fit_starts"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if options["fit_starts"] > options["max_nfev"]:
        raise ValueError("fit_starts cannot exceed the total per-combination max_nfev allocation")
    if options["proposal_coordinates"] not in {"native", "slot"}:
        raise ValueError("proposal_coordinates must be native or slot")
    if options["proposal_coordinates"] == "slot" and engine not in {"e2e", "rasr"}:
        raise NotImplementedError("Raw-slot coordinate proposal extraction currently supports E2E and RASR only")
    if options["proposal_mode"] not in {"whole", "subexpressions"}:
        raise ValueError("proposal_mode must be whole or subexpressions")
    template = request.equation_template
    if not template or not template["slots"]:
        raise ValueError("Pretrained template variants require a template with unknown slots")
    validate_template(template, request.variables, request.target, request.operators, request.fixed_constants)
    declarations = [*request.variables, request.target, *template["slots"].values(),
                    *(a for slot in template["slots"].values() for a in slot["arguments"])]
    if any(d.get("unit") != "1" for d in declarations):
        raise NotImplementedError("Pretrained template proposals currently require explicit unit1 coordinates and slots")
    bindings = _bindings(template, all_calls=options["proposal_mode"] == "subexpressions")
    for index, name in enumerate(sorted(template["slots"])):
        proposed = (full_problem_request(request, name, bindings[name], options, index)[0]
                    if options["proposal_mode"] == "subexpressions"
                    else _proposal_request(request, name, bindings[name], options, index))
        get_engine(engine).validate_request(proposed, context)
    return options, bindings


def _combinations(pools):
    """Diagonal product traversal avoids exhausting one slot before another."""
    names = sorted(pools)
    first = (0,) * len(names)
    frontier, seen = [(0, first)], {first}
    while frontier:
        _, indices = heappop(frontier)
        yield {name: pools[name][i] for name, i in zip(names, indices, strict=True)}
        for axis in range(len(names)):
            next_indices = (*indices[:axis], indices[axis] + 1, *indices[axis + 1:])
            if next_indices not in seen and next_indices[axis] < len(pools[names[axis]]):
                seen.add(next_indices)
                heappush(frontier, (sum(next_indices), next_indices))


def _input_slot_coordinates(body, arguments):
    """New hypothesis, not equivalent simplification: discard input scaler wrappers."""
    transformed = []
    parameters = body["parameters"]
    indices = {a["name"]: i for i, a in enumerate(arguments)}

    def visit(node):
        if node.get("op") == "div":
            numerator, denominator = node["args"]
            if numerator.get("op") == "sub" and "param" in denominator:
                variable, mean = numerator["args"]
                if "var" in variable and "param" in mean:
                    names = (mean["param"], denominator["param"])
                    if all(parameters[n].get("origin") == "train_normalization"
                           and parameters[n].get("trainable") is False
                           and parameters[n].get("unit") == "1" for n in names):
                        expected = (f"x{indices[variable['var']]}_mean", f"x{indices[variable['var']]}_scale")
                        if names == expected and parameters[names[1]]["value"] > 0:
                            transformed.append({"variable": variable["var"], "removed_fixed_parameters": list(names)})
                            return deepcopy(variable)
        return {**node, "args": [visit(a) for a in node["args"]]} if "args" in node else deepcopy(node)

    body["ast"] = visit(body["ast"])
    used = set()
    def collect(node):
        if "param" in node:
            used.add(node["param"])
        for child in node.get("args", []):
            collect(child)
    collect(body["ast"])
    body["parameters"] = {key: value for key, value in parameters.items() if key in used}
    if any(p.get("origin") == "train_normalization" for p in body["parameters"].values()):
        raise ValueError("Unrecognized native input normalization; cannot form a raw-slot hypothesis")
    return transformed


def _rasr_slot_output(body):
    """Drop tagged output normalization only for a new slot-coordinate hypothesis."""
    ast, parameters = body["ast"], body["parameters"]
    if ast.get("op") != "add" or ast["args"][1] != {"param": "y_mean"}:
        raise ValueError("Unrecognized RASR output normalization")
    scaled = ast["args"][0]
    if scaled.get("op") != "mul" or scaled["args"][1] != {"param": "y_scale"}:
        raise ValueError("Unrecognized RASR output scaling")
    names = ["y_scale", "y_mean"]
    if any(parameters[n].get("origin") != "train_normalization"
           or parameters[n].get("trainable") is not False
           or parameters[n].get("unit") != "1" for n in names) or parameters["y_scale"]["value"] <= 0:
        raise ValueError("RASR slot-coordinate extraction requires fixed unit1 positive output scaling")
    body["ast"] = deepcopy(scaled["args"][0])
    return names


def _normalize_proposal(candidate, arguments, coordinates="native", transformation=None):
    body = deepcopy(candidate.equation)
    if coordinates == "slot":
        output = _rasr_slot_output(body) if candidate.engine == "rasr" else []
        removed = _input_slot_coordinates(body, arguments)
        if transformation is not None:
            transformation.update(removed_input_normalizations=removed, removed_output_normalizations=output,
                                  equivalence_claim=False)
    # This is a new real-valued hypothesis, NOT a claim that IEEE arithmetic
    # equals strict-real arithmetic. Preserve the original native record.
    body["arithmetic"] = "strict_real"
    return body


def _check_slot_body(body, slot):
    stats = inspect_equation(body, [{"id": a["name"], "unit": "1"} for a in slot["arguments"]])
    if not stats["operators"] <= set(slot["allowed_operators"]) or stats["n_nodes"] > slot["max_nodes"]:
        raise ValueError(f"Proposal outside slot grammar: nodes={stats['n_nodes']} limit={slot['max_nodes']}; "
                         f"forbidden_operators={sorted(stats['operators'] - set(slot['allowed_operators']))}")
    for parameter in body["parameters"].values():
        if parameter.get("unit") is None and parameter["trainable"]:
            # Explicit dimensionless coefficient prior, permitted only for the
            # all-unit1 problem accepted above. Fixed transforms keep their unit.
            parameter["unit"] = "1"
        if parameter.get("unit") != "1":
            raise ValueError("Proposal has unknown or non-unit1 fixed parameter units")
    return body


def _slot_body(candidate, slot, coordinates="native", transformation=None):
    return _check_slot_body(_normalize_proposal(candidate, slot["arguments"], coordinates, transformation), slot)


def discover(request, context, *, engine):
    options, bindings = validate_request(request, context, engine=engine)
    name = {"flash_ansr": "flash"}.get(engine, engine) + "_template"
    context.work_dir.mkdir(parents=True, exist_ok=True)
    pools, native_runs, rejected, candidates, attempts = {}, {}, [], [], []
    archive = {}
    archive_path = context.work_dir / "proposal_archive.json"
    predictions = {"row_ids": np.asarray(request.row_ids)}
    predictions_path = context.work_dir / "template_predictions.npz"
    summary = {
        "method_kind": "pretrained_proposals_with_joint_template_fit", "proposal_engine": engine,
        "proposal_target": ("Complete original problem X/y, plus exact known call-argument features; no unknown-slot target"
                            if options["proposal_mode"] == "subexpressions" else
                            "Original observed response used only as a shape heuristic; not an observation of the slot"),
        "proposal_binding": ("Every declared call inspected by exact feature binding; Train only; distinct native run per slot"
                             if options["proposal_mode"] == "subexpressions" else
                             "First declared call per slot; Train only; distinct native run for each slot name"),
        "proposal_constraints": "Decoder is not template/response conditioned; assembled hypotheses use the complete problem",
        "coefficient_policy": "Native literals remain fixed; normalization fixed in native coordinates, explicitly removed in slot coordinates; independent slot coefficients fitted jointly; shared calls share one vector",
        "proposal_coordinates": options["proposal_coordinates"],
        "coordinate_semantics": "slot discards tagged E2E input or RASR input/output normalization as a new coordinate hypothesis, not equivalent simplification; native preserves fixed transforms",
        "arithmetic": "Native proposals immutable; assembled model is an explicitly new strict-real hypothesis",
        "fidelity": "Final model uses common evaluator only; native fidelity belongs solely to the archived proposal",
        "resolved_options": options, "native_runs": native_runs, "rejected_proposals": rejected,
        "prediction_evidence": "Train arrays for retained proposals replay the faithful native-arithmetic export, not a new native oracle; assembled arrays use the refitted strict-real equation",
    }

    def result(termination):
        if not candidates:
            summary["reason"] = (
                "No portable native proposals met the declared slot requirements; joint fitting was not reached (see proposal rejection reasons)"
                if termination == "no_supported_proposals"
                else "Budget ended before a feasible assembled template candidate was produced"
                if termination in {"wall_budget", "combination_budget", "in_progress"}
                else "Joint fitting produced no feasible assembled template candidate; see recorded attempts"
            )
        else:
            summary.pop("reason", None)
        summary.update(termination=termination, exported_candidates=len(candidates),
                       attempted_combinations=len(attempts), attempts=attempts,
                       native_execution_status="partial" if termination in {"wall_budget", "in_progress"} else "completed")
        artifacts = {"native_proposals": str(archive_path)} if archive_path.exists() else {}
        if len(predictions) > 1:
            np.savez_compressed(predictions_path, **predictions)
            artifacts["template_predictions"] = str(predictions_path)
        return DiscoveryResult(list(candidates), "ok" if candidates else "failed", deepcopy(summary), artifacts)

    for index, slot_name in enumerate(sorted(request.equation_template["slots"])):
        if context.deadline is not None and time.monotonic() >= context.deadline:
            return result("wall_budget")
        maps = None
        if options["proposal_mode"] == "subexpressions":
            proposed_request, maps = full_problem_request(request, slot_name, bindings[slot_name], options, index)
            predictions[f"proposal_input_{slot_name}"] = proposed_request.X
            predictions["observed_y"] = request.y
        else:
            proposed_request = _proposal_request(request, slot_name, bindings[slot_name], options, index)
        directory = context.work_dir / "proposals" / slot_name
        directory.mkdir(parents=True, exist_ok=True)
        native_context = replace(context, work_dir=directory, checkpoint=None)
        native = get_engine(engine).discover(proposed_request, native_context)
        write_json(directory / "candidates.json", [asdict(c) for c in native.candidates])
        write_json(directory / "summary.json", native.summary)
        archive[slot_name] = {"candidates": [asdict(c) for c in native.candidates],
                              "summary": native.summary, "native_artifacts": {k: str(v) for k, v in native.native_artifacts.items()}}
        write_json(archive_path, archive)
        native_runs[slot_name] = {"status": native.status, "candidates": len(native.candidates),
                                  "binding": bindings[slot_name], "seed": proposed_request.seed,
                                  "summary": str(directory / "summary.json")}
        if maps is not None:
            native_runs[slot_name].update(variables=proposed_request.variables, call_bindings=maps,
                                         input_prediction_key=f"proposal_input_{slot_name}",
                                         alias_policy="First formal for identical input bindings; other aliased formals unused and unidentifiable on this call")
            arguments = [{"name": v["id"]} for v in proposed_request.variables]
            try:
                items, extraction = collect_parts(native.candidates, request.equation_template["slots"][slot_name], maps,
                    lambda candidate, transform, arguments=arguments: _normalize_proposal(candidate, arguments, options["proposal_coordinates"], transform),
                    _check_slot_body, deadline=context.deadline)
            except TimeoutError as exc:
                native_runs[slot_name]["part_extraction"] = {"status": "wall_budget", "reason": str(exc)}
                return result("wall_budget")
            native_runs[slot_name]["part_extraction"] = extraction
        else:
            items = [(candidate.candidate_id, None, {}, candidate) for candidate in native.candidates]
        pool = []
        transformations = {}
        capped = 0
        for identifier, body, transformation, proposal in items:
            try:
                if body is None:
                    body = _slot_body(proposal, request.equation_template["slots"][slot_name], options["proposal_coordinates"], transformation)
                if len(pool) < options["max_proposals"]:
                    raw_prediction = evaluate_native(proposal.equation, proposed_request.X, proposed_request.variables)
                    real_prediction = evaluate(proposal.equation, proposed_request.X, proposed_request.variables)
                    key = f"proposal_{slot_name}_{len(pool)}"
                    transformation["prediction_keys"] = [key + "_raw", key + "_real"]
                    pool.append((identifier, body))
                    predictions.update({key + "_raw": raw_prediction, key + "_real": real_prediction})
                    transformations[identifier] = transformation
                else:
                    capped += 1
            except (ValueError, TypeError) as exc:
                rejected.append({"slot": slot_name, "candidate_id": identifier, "native_candidate_id": proposal.candidate_id, "reason": str(exc)})
        pools[slot_name] = pool
        native_runs[slot_name].update(retained_proposals=len(pool), supported_proposals_omitted_by_cap=capped, proposal_transformations=transformations)
        if not pool:
            return result("no_supported_proposals")

    constrained = request.limits.get("response_mode", "search") == "search" and bool(
        request.limits.get("probes") or request.limits.get("response_checks"))
    compiled = compile_response_constraints(request)
    summary["joint_response_conditions"] = {"mode": request.limits.get("response_mode", "search"),
                                            "enforced_in_coefficient_fit": constrained, **compiled}

    def constraint_values(equation):
        return response_values(evaluate(equation, np.asarray(compiled["points"]), request.variables), compiled)

    constraints = (constraint_values, *response_bounds(compiled)) if constrained else None
    termination = "proposal_combinations_exhausted"
    for index, combination in enumerate(_combinations(pools)):
        if index >= options["max_combinations"]:
            termination = "combination_budget"
            break
        if context.deadline is not None and time.monotonic() >= context.deadline:
            termination = "wall_budget"
            break
        lineage = {slot: value[0] for slot, value in combination.items()}
        fit = {}
        try:
            equation, evidence = instantiate_template(request.equation_template,
                {slot: value[1] for slot, value in combination.items()}, variables=request.variables, target=request.target)
            if any(c["required"] and c["status"] != "pass" for c in evidence["checks"]):
                raise ValueError("Proposal template fails declared structural or unit conditions")
            fitted = fit_starts(equation, request, options, index, context.deadline, fit,
                                fit_coefficients, constraints, compiled["data_scale"])
            prediction = evaluate(fitted, request.X, request.variables)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite complete Train response; no rows dropped")
            if constrained and not response_feasible(
                evaluate(fitted, np.asarray(compiled["points"]), request.variables), compiled
            ):
                raise ValueError("Joint fit did not satisfy the original response conditions")
            candidate = checked_candidate(name, index, fitted, request,
                lambda X, fitted=fitted: evaluate(fitted, X, request.variables), native_ieee=False)
            candidate.native_check.update(kind="shared_evaluator_reference; not an independent native export check",
                                          proposal_lineage=lineage, independent_native_oracle=False,
                                          comparison_semantics="shared evaluator consistency only; no independent native comparison")
            predictions[f"assembled_{index}_real"] = prediction
            candidates.append(candidate)
            attempts.append({"lineage": lineage, "status": "candidate", "fit": fit,
                             "prediction_key": f"assembled_{index}_real",
                             "train_rmse": prediction_metrics(request.y, prediction)["rmse"]})
        except TimeoutError as exc:
            attempts.append({"lineage": lineage, "status": "failed", "reason": str(exc), "fit": fit})
            return result("wall_budget")
        except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
            attempts.append({"lineage": lineage, "status": "failed", "reason": str(exc), "fit": fit})
        if context.checkpoint is not None:
            context.checkpoint(result("in_progress"))
    return result(termination)
