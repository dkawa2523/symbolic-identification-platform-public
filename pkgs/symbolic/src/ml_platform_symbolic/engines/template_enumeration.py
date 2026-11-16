"""Bounded slot AST enumeration with joint original-response coefficient fitting.

Discrete coverage is deterministic. Nonlinear fits are local, not certified global
optima. This adapter uses the common evaluator and makes no independent-native
fidelity claim. Selection and Final are never available to this module.
"""

import time
from collections import Counter, deque
from copy import deepcopy
from heapq import heappop, heappush
from importlib.metadata import version

import numpy as np

from ..constraints import check_constraints
from ..data import write_json
from ..equation_templates import instantiate_template
from ..expressions import ARITY, evaluate, inspect_equation
from ..fitting import fit_coefficients
from ..metrics import prediction_metrics
from ..response_search import (
    compile_response_constraints,
    response_bounds,
    response_feasible,
    response_values,
)
from ..units import require_canonical, unit_spec
from ._enumeration_units import assign_coefficient_units
from .base import DiscoveryResult, checked_candidate, validate_options

BINARY = ("add", "sub", "mul", "div")
UNARY = ("neg", "abs", "sqrt", "log", "exp", "sin", "cos", "tanh")
SLOT_OPERATORS = (*BINARY, *UNARY, "pow_int")


SUPPORTED_OPTIONS = {
    "coefficient_bounds",
    "objective",
    "order",
    "max_nfev",
    "max_slot_nodes",
    "max_structures",
    "starts",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": list(ARITY),
        "equation_template": True,
        "fixed_constants": "template_only",
        "requires_assets": False,
        "search_constraints": ["probes", "response_bounds", "response_checks"],
        "constraints": {
            "operators": "slot grammar enforced; full-template operator admissibility checked before fitting",
            "units": "declared dimensionless or uniquely solved dimensional coefficients; common full-template unit check before fitting",
            "fixed_constants": "immutable outer declarations",
            "coefficient_bounds": "bounded joint coefficient fit on original Train response; SLSQP with response constraints, least_squares otherwise",
            "probes": "explicit constrained joint coefficient fit and feasible restart selection; deterministic structure order unchanged",
            "response_checks": "explicit constrained joint coefficient finite-step fit; no between-point guarantee",
            "response_bounds": "explicit constrained joint coefficient fit and feasible restart selection; deterministic structure order unchanged",
            "equation_template": "outer structure and shared slot identity retained in every fitted candidate",
            "max_nodes": "slot enforced during enumeration; expanded equation common structural check before fitting",
            "max_free_parameters": "common independent-parameter count before fitting",
            "variables": "common allowed-variable check before fitting; no undeclared dependency requirement",
        },
    }


def preflight(request):
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    template = request.equation_template
    if template is None or not template.get("slots"):
        raise NotImplementedError(
            "template_enumeration requires at least one declared slot (shared calls allowed)"
        )
    slots = dict(sorted(template["slots"].items()))
    for slot in slots.values():
        for record in [slot, *slot["arguments"]]:
            if unit_spec(record["unit"]) is None:
                raise NotImplementedError("Enumeration requires known canonical slot argument/output units")
            require_canonical(record["unit"])
        unsupported = set(slot["allowed_operators"]) - set(SLOT_OPERATORS)
        if unsupported:
            raise NotImplementedError(f"Unsupported enumeration slot operators: {sorted(unsupported)}")
        if not set(slot["allowed_operators"]) <= set(request.operators):
            raise ValueError("Enumeration cannot replace the declared slot grammar")
    options = {
        "max_structures": 128,
        "order": "balanced",
        "max_slot_nodes": {name: min(slot["max_nodes"], 5) for name, slot in slots.items()},
        "max_nfev": 200,
        "starts": [1.0, -1.0, 0.1],
        "coefficient_bounds": [None, None],
        **deepcopy(request.options),
    }
    if type(options["max_structures"]) is not int or options["max_structures"] < 1:
        raise ValueError("max_structures must be a positive integer")
    for key, maximum in (
        ("max_nfev", 10000),
    ):
        if type(options[key]) is not int or not 1 <= options[key] <= maximum:
            raise ValueError(f"{key} must be an integer in 1..{maximum}")
    if options["order"] not in {"balanced", "size"}:
        raise ValueError("order must be balanced or size")
    maximums = options["max_slot_nodes"]
    if type(maximums) is int:
        maximums = dict.fromkeys(slots, maximums)
    if not isinstance(maximums, dict) or set(maximums) != set(slots):
        raise ValueError("max_slot_nodes must be an integer or a mapping naming every slot")
    for name, maximum in maximums.items():
        if type(maximum) is not int or not 1 <= maximum <= 9:
            raise ValueError("Every max_slot_nodes must be an integer in 1..9")
        if maximum > slots[name]["max_nodes"]:
            raise ValueError("max_slot_nodes cannot exceed the declared slot max_nodes")
    options["max_slot_nodes"] = maximums
    bounds, starts = options["coefficient_bounds"], options["starts"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(
            b is not None
            and (
                isinstance(b, bool)
                or not isinstance(b, (int, float))
                or not np.isfinite(b)
            )
            for b in bounds
        )
        or (all(b is not None for b in bounds) and bounds[0] >= bounds[1])
    ):
        raise ValueError(
            "coefficient_bounds must be [lower, upper], finite or null, with lower < upper"
        )
    if (
        not isinstance(starts, list)
        or not 1 <= len(starts) <= 8
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not np.isfinite(v)
            or (bounds[0] is not None and v < bounds[0])
            or (bounds[1] is not None and v > bounds[1])
            for v in starts
        )
        or len(set(starts)) != len(starts)
    ):
        raise ValueError(
            "starts must contain 1..8 distinct finite values inside coefficient_bounds"
        )
    if options.get("objective") not in {None, "mse"}:
        raise ValueError("template_enumeration objective accepts only 'mse'; omit for native defaults")
    return slots, options


def enumerate_shapes(slot, max_nodes):
    """Node-count order, declared argument order, then documented operator order.

    Every coefficient leaf is independent within one slot; repeated calls share
    the entire fitted slot. Only add/mul operand permutation is deduplicated.
    No algebraic cancellations that could erase a domain are performed.
    """
    levels = {}
    leaves = [
        ("coefficient",),
        *[("variable", a["name"]) for a in slot["arguments"]],
        ("literal", 0.0),
        ("literal", 1.0),
    ]
    for size in range(1, max_nodes + 1):
        current = []
        seen = set()

        def proposals(size=size):
            if size == 1:
                yield from leaves
            for op in UNARY:
                if op in slot["allowed_operators"]:
                    for child in levels.get(size - 1, []):
                        yield (op, child)
            for op in BINARY:
                if op not in slot["allowed_operators"]:
                    continue
                for left_size in range(1, size - 1):
                    for left in levels.get(left_size, []):
                        for right in levels.get(size - 1 - left_size, []):
                            if op in {"add", "mul"} and repr(left) > repr(right):
                                continue
                            yield (op, left, right)
            if "pow_int" in slot["allowed_operators"]:
                for child in levels.get(size - 2, []):
                    for exponent in (-2, -1, 2, 3):
                        yield ("pow_int", child, ("literal", exponent))

        for shape in proposals():
            if shape not in seen:
                seen.add(shape)
                current.append(shape)
                yield shape
        levels[size] = current


def _round_robin(streams):
    pending = deque(map(iter, streams))
    while pending:
        stream = pending.popleft()
        value = next(stream, None)
        if value is not None:
            yield value
            pending.append(stream)


def _index_product(streams):
    """Lazy Cartesian frontier, ordered by index sum then index tuple.

    Each finite combination is reached without exhausting another stream first.
    Caches contain only visited prefixes, never complete grammar levels.
    """
    streams = list(map(iter, streams))
    caches = [[] for _ in streams]

    def get(i, n):
        while len(caches[i]) <= n:
            item = next(streams[i], None)
            if item is None:
                return None
            caches[i].append(item)
        return caches[i][n]

    first = (0,) * len(streams)
    if any(get(i, 0) is None for i in range(len(streams))):
        return
    frontier, seen = [(0, first)], {first}
    while frontier:
        _, indices = heappop(frontier)
        yield tuple(get(i, n) for i, n in enumerate(indices))
        for i in range(len(streams)):
            following = (*indices[:i], indices[i] + 1, *indices[i + 1:])
            if following not in seen:
                seen.add(following)
                if get(i, following[i]) is not None:
                    heappush(frontier, (sum(following), following))


def enumerate_balanced_shapes(slot, max_nodes):
    """Round-robin exact-size/root streams without materializing lower levels.

    Same finite grammar as enumerate_shapes. Fair scheduling changes order, not
    the expression family; algebraic equivalence and domain are not simplified.
    """
    operators = [op for op in (*UNARY, *BINARY, "pow_int") if op in slot["allowed_operators"]]
    reachable = {1}
    for size in range(2, max_nodes + 1):
        if (any(op in UNARY for op in operators) and size - 1 in reachable
                or any(op in BINARY for op in operators)
                and any(i in reachable and size - 1 - i in reachable for i in range(1, size - 1))
                or "pow_int" in operators and size - 2 in reachable):
            reachable.add(size)

    def shapes(size, root=None):
        if size == 1:
            yield ("coefficient",)
            yield from (("variable", a["name"]) for a in slot["arguments"])
            yield ("literal", 0.0)
            yield ("literal", 1.0)
            return
        if root is None:
            yield from _round_robin(shapes(size, op) for op in operators)
            return
        if root in UNARY and size - 1 in reachable:
            for child in shapes(size - 1):
                yield (root, child)
        elif root in BINARY:
            commutative = root in {"add", "mul"}
            seen = set()

            def split(left_size, right_size):
                for left, right in _index_product([shapes(left_size), shapes(right_size)]):
                    # Canonicalize instead of scanning an entire split whose
                    # children all occur in reverse lexical order.
                    if commutative and repr(left) > repr(right):
                        left, right = right, left
                    value = (root, left, right)
                    if not commutative or value not in seen:
                        if commutative:
                            seen.add(value)
                        yield value
            yield from _round_robin(
                split(left_size, size - 1 - left_size)
                for left_size in range(1, size - 1)
                if left_size in reachable and size - 1 - left_size in reachable
                and (not commutative or left_size <= size - 1 - left_size)
            )
        elif root == "pow_int" and size - 2 in reachable:
            for child, exponent in _index_product([shapes(size - 2), (-2, -1, 2, 3)]):
                yield (root, child, ("literal", exponent))

    yield from _round_robin([
        shapes(1), *(shapes(size, op) for size in sorted(reachable - {1}) for op in operators)
    ])


def enumerate_combinations(slots, maximums, order="size"):
    """Lazy Cartesian grammar: legacy node order or balanced stream-index order.

    A sorted frontier avoids materializing a slot grammar or its full product.
    Balanced priority is the sum of per-slot indices, not AST complexity.
    Distinct slot names have independent bodies; repeated calls remain shared.
    """
    names = list(slots)
    generator = enumerate_balanced_shapes if order == "balanced" else enumerate_shapes
    streams = [iter(generator(slots[name], maximums[name])) for name in names]
    if order == "balanced":
        for combination in _index_product(streams):
            yield dict(zip(names, combination, strict=True))
        return
    caches = [[] for _ in names]

    def get(i, index):
        while len(caches[i]) <= index:
            value = next(streams[i], None)
            if value is None:
                return None
            caches[i].append(value)
        return caches[i][index]

    def size(shape):
        return 1 if shape[0] in {"variable", "coefficient", "literal"} else 1 + sum(map(size, shape[1:]))

    def priority(indices):
        return sum(indices) if order == "balanced" else sum(size(get(i, n)) for i, n in enumerate(indices))

    first = (0,) * len(names)
    frontier = [(priority(first), first)]
    seen = {first}
    while frontier:
        _, indices = heappop(frontier)
        yield {name: get(i, index) for i, (name, index) in enumerate(zip(names, indices, strict=True))}
        for i in range(len(names)):
            following = (*indices[:i], indices[i] + 1, *indices[i + 1 :])
            if following not in seen:
                seen.add(following)
                if get(i, following[i]) is not None:
                    heappush(frontier, (priority(following), following))


def _slot_equation(shape, start, bounds, slot):
    parameters = {}

    def ast(node):
        if node[0] == "coefficient":
            key = f"c{len(parameters)}"
            parameters[key] = {
                "value": float(start),
                "trainable": True,
                "unit": "1",
                "bounds": list(bounds),
                "origin": "enumerated_slot_joint_train_fit",
            }
            return {"param": key}
        if node[0] == "variable":
            return {"var": node[1]}
        if node[0] == "literal":
            return {"const": node[1]}
        return {"op": node[0], "args": [ast(child) for child in node[1:]]}

    result = {"schema_version": 2, "ast": ast(shape), "parameters": parameters}
    assign_coefficient_units(result, slot)
    return result


def validate_request(request, context):
    """Validate the explicit grammar without enumeration or fitting."""
    return preflight(request)


def discover(request, context):
    slots, options = validate_request(request, context)
    mode = request.limits.get("response_mode", "search")
    constrained = mode == "search" and bool(request.limits.get("probes") or request.limits.get("response_checks"))
    compiled = compile_response_constraints(request) if constrained or options.get("objective") == "mse" else None

    def constraint_values(equation):
        predicted = evaluate(equation, np.asarray(compiled["points"]), request.variables)
        return response_values(predicted, compiled)

    explicit_constraints = (constraint_values, *response_bounds(compiled)) if constrained else None

    candidates, attempts = [], []
    coverage = {name: {"sizes": Counter(), "roots": Counter()} for name in slots}
    stop = "grammar_exhausted"
    start_time = time.monotonic()
    summary = {
        "method_kind": "finite_slot_enumeration_with_local_joint_fitting",
        "scipy": version("scipy"),
        "fit_rows": len(request.X),
        "resolved_options": options,
        "search_space": {
            "slots": slots,
            "max_slot_nodes": options["max_slot_nodes"],
            "literals": [0.0, 1.0],
            "power_exponents": [-2, -1, 2, 3],
            "order": options["order"],
            "order_detail": "recursive size/root/split round-robin; child and slot Cartesian index-sum frontiers" if options["order"] == "balanced" else "total slot node count, then per-slot shape indices",
            "candidate_index": "position in the selected enumeration order; not comparable across order modes",
            "operator_order": list(UNARY) + list(BINARY) + ["pow_int"],
            "complexity": "expanded executable portable AST nodes, including constants, power exponent leaves and repeated slot calls; native engine complexity and SymPy node counts are not interchangeable",
            "deduplication": "exact shape and add/mul operand permutation only; independent coefficient leaves; no algebraic simplification",
            "sharing": "independent body/vector per slot name; each is shared by its calls",
            "coefficient_units": "all-unit1 slots retain unit1 grammar; otherwise exact dimensional equations must uniquely determine every coefficient; no quantity-kind interpretation inferred",
        },
        "fit_policy": {
            "objective": "unweighted least squares on all original Train response rows; no transformed target or row dropping",
            "solver": "common fit_coefficients: scipy least_squares, trf, linear loss, ftol=None, xtol=gtol=1e-8; signal/parameter scaling",
            "initialization": "each start initializes every slot coefficient; outer supplied initial values are unchanged across starts",
            "restart_choice": "minimum Train RMSE, then earliest declared start; no Selection or Final access",
            "limitation": "local fits and finite ordered grammar only; no global optimum, unique component, or exhaustive scientific discovery claim",
        },
        "constraint_support": capabilities()["constraints"],
        "response_search": {
            "mode": mode, "search_response_constraints": constrained,
            "application": "joint coefficient optimization and restart choice; structure order independent of response constraints",
            "objective": "train_scaled_mse" if compiled else "native_train_rmse",
            "enforcement": "explicit SLSQP finite-point inequalities/equalities" if constrained else "none",
            "data_scale": compiled["data_scale"] if compiled else None,
            "points": compiled["points"] if constrained else [],
            "terms": compiled["terms"] if constrained else [],
        },
    }

    if constrained:
        summary["fit_policy"]["solver"] = "SLSQP explicit constraints; feasible iterates retained; no clipping or relaxed response bounds; max_nfev limits objective evaluations; objective finite-difference probes included; constraint/scaling probes separate"
    if compiled is not None:
        summary["fit_policy"].update(
            objective="Train-scaled mean square subject to explicit finite response constraints" if constrained
            else "Train-scaled mean square (control; no response penalty)",
            restart_choice="minimum full fitting objective, then earliest declared start; no Selection/Final access",
        )

    def result(status, termination):
        summary.update(
            coverage=deepcopy(coverage),
            native_candidates=len(attempts),
            exported_candidates=len(candidates),
            attempted_structures=len(attempts),
            prefit_rejections=sum(a.get("stage") in {"coefficient_units", "structural"} for a in attempts),
            rejected_structures=sum(a["status"] != "candidate" for a in attempts),
            termination=termination,
            grammar_exhausted=termination == "grammar_exhausted",
            fit_attempts=sum(len(a["fit_attempts"]) for a in attempts),
            failed_fit_attempts=sum(
                f["status"] == "failed" for a in attempts for f in a["fit_attempts"]
            ),
            prediction_evaluations=sum(
                f["fit"].get("prediction_evaluations", 0)
                for a in attempts
                for f in a["fit_attempts"]
            ),
            elapsed_seconds=time.monotonic() - start_time,
            native_execution_status="partial"
            if termination in {"wall_budget", "in_progress"}
            else "completed",
        )
        return DiscoveryResult(list(candidates), status, deepcopy(summary))

    context.work_dir.mkdir(parents=True, exist_ok=True)
    path = context.work_dir / "enumeration_attempts.json"
    shapes = enumerate_combinations(slots, options["max_slot_nodes"], options["order"])
    for index, combination in enumerate(shapes):
        if index >= options["max_structures"]:
            stop = "structure_budget"
            break
        if context.deadline is not None and time.monotonic() >= context.deadline:
            stop = "wall_budget"
            break
        def node_count(shape):
            return 1 if shape[0] in {"coefficient", "variable", "literal"} else 1 + sum(node_count(c) for c in shape[1:])

        for name, shape in combination.items():
            coverage[name]["sizes"][str(node_count(shape))] += 1
            coverage[name]["roots"][shape[0]] += 1
        ledger = {
            "index": index,
            "shapes": combination,
            "status": "rejected",
            "fit_attempts": [],
        }
        attempts.append(ledger)
        best = None
        starts = []
        # Parameter-free shapes need one fit only, while outer coefficients still
        # use their supplied initial values on the joint original response.
        stage = "coefficient_units"
        try:
            slot_probes = {
                name: _slot_equation(shape, options["starts"][0], options["coefficient_bounds"], slots[name])
                for name, shape in combination.items()
            }
            stage = "structural"
            prototype, _ = instantiate_template(
                request.equation_template, slot_probes,
                variables=request.variables, target=request.target,
            )
            # Reuse final structural rules. Fitting cannot change AST, declared
            # dimensions, parameter identity/bounds, or template structure.
            checks = check_constraints(
                prototype, request.variables, request.target, request.operators,
                {"require_units": False, **request.limits}, include_response=False,
                context={**(context.provenance or {}), "variables": request.variables, "target": request.target},
            )
            failed = [c for c in checks if c["required"] and c["status"] != "pass"]
            if failed:
                ledger.update(stage=stage, reason="Fit-independent required conditions failed", checks=failed)
            else:
                starts = (
                    options["starts"] if any(p["parameters"] for p in slot_probes.values()) else options["starts"][:1]
                )
        except (ValueError, TypeError, OverflowError) as exc:
            ledger.update(stage=stage, reason=str(exc))
            # This completed rejection still belongs to the periodic recovery
            # snapshot. Skip fitting, not the common end-of-attempt checkpoint.
        for initial in starts:
            if context.deadline is not None and time.monotonic() >= context.deadline:
                stop = "wall_budget"
                break
            stage = "instantiate"
            fit_info = {}
            try:
                local = deepcopy(slot_probes)
                for body in local.values():
                    for parameter in body["parameters"].values():
                        parameter["value"] = float(initial)
                equation, _ = instantiate_template(
                    request.equation_template,
                    local,
                    variables=request.variables,
                    target=request.target,
                )
                stage = "fit"
                fitted = fit_coefficients(
                    equation,
                    request.X,
                    request.y,
                    request.variables,
                    max_nfev=options["max_nfev"],
                    diagnostics=fit_info,
                    constraints=explicit_constraints,
                    data_scale=compiled["data_scale"] if compiled else None,
                )
                predicted = evaluate(fitted, request.X, request.variables)
                if not np.isfinite(predicted).all():
                    raise ValueError("Nonfinite Train prediction; no rows removed")
                loss = prediction_metrics(request.y, predicted)["rmse"]
                if loss is None:
                    raise ValueError("Unrepresentable Train loss")
                objective = loss
                if compiled is not None:
                    objective = float(np.mean(((predicted - request.y) / compiled["data_scale"])**2))
                    if constrained and not response_feasible(
                        evaluate(fitted, np.asarray(compiled["points"]), request.variables), compiled
                    ):
                        raise ValueError("No feasible coefficient result at declared finite points; bounds unchanged")
                    if not np.isfinite(objective):
                        raise ValueError("Nonfinite constrained objective; no constraint rows removed")
                ledger["fit_attempts"].append(
                    {
                        "start": initial,
                        "status": "ok",
                        "train_rmse": loss,
                        "objective": objective,
                        "fit": fit_info,
                    }
                )
                if best is None or objective < best[0]:
                    best = objective, fitted, loss
            except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
                ledger["fit_attempts"].append(
                    {
                        "start": initial,
                        "status": "failed",
                        "stage": stage,
                        "reason": str(exc),
                        "fit": fit_info,
                    }
                )
        if best is not None:
            try:
                fitted = best[1]
                candidate = checked_candidate(
                    "template_enumeration",
                    index,
                    fitted,
                    request,
                    lambda X, fitted=fitted: evaluate(fitted, X, request.variables),
                    native_ieee=False,
                )
                candidate.native_check["kind"] = (
                    "shared_evaluator_reference; no independent engine export oracle"
                )
                candidates.append(candidate)
                ledger.update(
                    status="candidate",
                    candidate_id=candidate.candidate_id,
                    train_rmse=best[2],
                    objective=best[0],
                    expanded_nodes_coordinate="working resource statistic; not source budget",
                    expanded_nodes=inspect_equation(fitted, request.variables)[
                        "n_nodes"
                    ],
                )
            except (ValueError, TypeError, OverflowError) as exc:
                ledger.update(stage="portable_check", reason=str(exc))
        if context.checkpoint is not None and candidates and (index + 1) % 10 == 0:
            progress = result("partial", "in_progress")
            write_json(path, {"summary": progress.summary, "structures": attempts})
            progress.native_artifacts["enumeration_attempts"] = str(path)
            context.checkpoint(progress)
        if stop == "wall_budget":
            break
    final = result(
        "partial"
        if stop == "wall_budget" and candidates
        else "ok"
        if candidates
        else "failed",
        stop,
    )
    write_json(path, {"summary": final.summary, "structures": attempts})
    final.native_artifacts["enumeration_attempts"] = str(path)
    return final

