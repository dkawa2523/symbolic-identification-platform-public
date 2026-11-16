"""Official ProGED static grammar discovery; preserve symbolic parameters and RHS order."""

import hashlib
import time
from copy import deepcopy
from importlib.metadata import version
from functools import partial

import numpy as np
import pandas as pd

from ..data import write_json
from .base import DiscoveryResult, checked_candidate, validate_options
from .symbolic_tree import export_symbolic

OPERATORS = {"add", "sub", "mul", "div", "neg", "pow_int", "pow", "exp", "log", "sqrt", "abs", "sin", "cos", "tanh"}


SUPPORTED_OPTIONS = {
    "coefficient_bounds",
    "depth_limit",
    "grammar",
    "max_attempts",
    "max_constants",
    "optimizer",
    "optimizer_iterations",
    "optimizer_population",
    "sample_size",
    "template",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": sorted(OPERATORS),
        "requires_assets": False,
        "constraints": {
            "operators": "grammar support plus portable post-filter",
            "grammar": "PCFG hard productions; probabilities are soft search guidance",
            "units": "unknown fitted coefficient units; portable post-filter",
            "parameter_bounds": "native DE bounds; local initialization bounds are not hard fit bounds",
            "max_nodes": "native depth limit plus post-filter",
            "variables": "post-filtered",
            "probes": "post-filtered",
        },
    }


def _export(model, variables):
    return export_symbolic(model.expr[0], {str(key): float(value) for key, value in model.params.items()}, variables)


def allowed_grammar(variables, operators):
    """One generic arithmetic/function PCFG, never a supplied generating formula."""
    operators = set(operators)
    affine = [("V", 3.0), ("'C'", 1.0)]
    if "mul" in operators:
        affine.append(("'C' '*' V", 2.0))
        if "add" in operators:
            affine.append(("'C' '*' V '+' 'C'", 2.0))
    expressions = [("A", 6.0)]
    for op, spelling in {"add": "+", "sub": "-", "mul": "*", "div": "/"}.items():
        if op in operators:
            expressions.append((f"'(' E ')' '{spelling}' '(' E ')'", 1.0))
    if "div" in operators:
        # General affine-ratio motif is shared across all datasets, not a truth hint.
        expressions.append(("'(' A ')' '/' '(' A ')'", 2.0))
    for op in ("sin", "cos", "tanh", "exp", "log", "sqrt", "abs"):
        if op in operators:
            expressions.append((f"'{op if op != 'abs' else 'Abs'}' '(' E ')'", 0.5))
    if "neg" in operators:
        expressions.append(("'-' '(' E ')'", 0.5))
    if "pow_int" in operators:
        expressions.append(("'(' E ')' '**' '2'", 0.5))

    def production(symbol, choices):
        total = sum(weight for _, weight in choices)
        return symbol + " -> " + " | ".join(f"{text} [{weight / total:.16g}]" for text, weight in choices)

    return "\n".join(
        [
            "S -> E [1.0]",
            production("E", expressions),
            production("A", affine),
            production("V", [(f"'{variable}'", 1.0) for variable in variables]),
        ]
    )


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    if "grammar" in request.options and "template" in request.options:
        raise ValueError("Choose a native grammar or template, not both")
    optimizer = request.options.get("optimizer", "DE")
    if optimizer not in {"local", "DE"}:
        raise ValueError("ProGED supported native optimizers are local and DE")
    iterations = request.options.get("optimizer_iterations")
    if iterations is not None and (type(iterations) is not int or iterations < 1):
        raise ValueError("optimizer_iterations must be a positive integer")
    if optimizer == "local" and "optimizer_population" in request.options:
        raise ValueError("optimizer_population applies to DE, not the local optimizer")
    bounds = request.options.get("coefficient_bounds", [-10.0, 10.0])
    if (
        not isinstance(bounds, (list, tuple))
        or len(bounds) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bounds)
        or not np.isfinite(bounds).all()
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("ProGED coefficient_bounds needs two finite increasing values; local uses initialization only")
    for key in ('sample_size', 'depth_limit', 'max_attempts', 'max_constants', 'optimizer_population'):
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"proged {key} must be a positive integer")
    return optimizer, iterations, bounds


def discover(request, context):
    optimizer, iterations, bounds = validate_request(request, context)
    import ProGED.parameter_estimation as native_estimation
    from ProGED import EqDisco
    from ProGED.configs import settings as native_defaults
    from ProGED.generators.grammar import GeneratorGrammar

    np.random.seed(request.seed)
    ids = [v["id"] for v in request.variables]
    frame = pd.DataFrame(request.X, columns=ids).assign(target=request.y)
    depth = request.options.get("depth_limit", 8)
    estimation = deepcopy(native_defaults)
    estimation["experiment"].update(seed=request.seed, verbosity=0)
    estimation["parameter_estimation"].update(
        optimizer=optimizer, max_constants=request.options.get("max_constants", 8), param_bounds=[tuple(bounds)]
    )
    estimation["optimizer_DE"].update(
        max_iter=request.options.get("optimizer_iterations", 20),
        pop_size=request.options.get("optimizer_population", 16),
    )
    settings = {
        "data": frame,
        "lhs_vars": ["target"],
        "rhs_vars": ids,
        "task_type": "algebraic",
        "generator_template_name": request.options.get("template", "polynomial"),
        "sample_size": request.options.get("sample_size", 80),
        "depth_limit": depth,
        "max_attempts": request.options.get("max_attempts", 400),
        "verbosity": 0,
        "estimation_settings": estimation,
    }
    if "template" not in request.options:
        grammar = request.options.get("grammar", allowed_grammar(ids, request.operators))
        settings["generator"] = GeneratorGrammar(grammar, depth_limit=depth)
    model = EqDisco(**settings)
    candidates, rejected, completed, bound_usage, optimizer_work = [], [], [], [], []
    grammar = request.options.get("grammar", str(model.generator.grammar))
    phase = {"phase": "generation", "generated_count": None, "fit_started_count": 0,
             "fit_completed_count": 0, "generation_seconds": 0.0, "native_fit_seconds": 0.0,
             "export_seconds": 0.0, "checkpoint_seconds": 0.0}
    phase_path = context.work_dir / "proged_phase.json"

    def save_phase():
        temporary = phase_path.with_suffix(".pending")
        write_json(temporary, phase)
        temporary.replace(phase_path)

    def completed_result(finalized=False):
        raw = write_json(context.work_dir / "proged_native.json", {
            "models": completed, "optimizer_work": optimizer_work,
            "parameter_bound_distances": bound_usage, "progress": deepcopy(phase),
        })
        return DiscoveryResult(
            list(candidates), ("ok" if candidates else "failed") if finalized else ("partial" if candidates else "timeout"),
        {
            "version": version("ProGED"),
            "source_revision": "875c05535286392080afc012c6b1a6dac1b77b35",
            "license": "BSD-3-Clause",
            "native_candidates": len(completed),
            "rejected_exports": rejected,
            "grammar": grammar,
            "grammar_sha256": hashlib.sha256(grammar.encode("utf-8")).hexdigest(),
            "native_optimizer": optimizer,
            "effective_optimizer_options": {"maxiter": iterations}
            if optimizer == "local" and iterations is not None
            else {
                "max_iter": estimation["optimizer_DE"]["max_iter"],
                "pop_size": estimation["optimizer_DE"]["pop_size"],
            }
            if optimizer == "DE"
            else "native SciPy defaults",
            "optimizer_work": optimizer_work,
            "optimizer_integration": "same native local objective/SciPy optimizer; adapter forwards requested maxiter for fit lifetime, then restores module"
            if optimizer == "local"
            else "native DE settings",
            "coefficient_bounds": "native DE search bounds"
            if optimizer == "DE"
            else "initialization only, not enforced",
            "coefficient_bound_values": list(bounds),
            "parameter_boundary_usage": {
                "scope": "finite valid native DE coefficients; exact distances, no near-boundary threshold" if optimizer == "DE" else "not applicable: initialization interval is not a fitting bound",
                "parameters_checked": len(bound_usage),
                "at_boundary": sum(v["lower_distance"] == 0 or v["upper_distance"] == 0 for v in bound_usage),
                "outside_bounds": sum(v["normalized_margin"] < 0 for v in bound_usage),
                "minimum_normalized_margin": min((v["normalized_margin"] for v in bound_usage), default=None),
            },
            "generation_seconds": phase["generation_seconds"],
            "fit_seconds": phase["native_fit_seconds"],
            "progress": deepcopy(phase),
            "native_execution_status": "completed" if finalized else "running",
            "termination_reason": "native_return" if finalized else "completed_candidate_snapshot",
            "exported_candidates": len(candidates),
            "native_check_source": "each native model.evaluate with its own ordered/subselected rhs_vars on independent X",
            "input_rows": len(request.X),
            "constraint_support": capabilities()["constraints"],
        },
            {"native_models": raw},
        )

    def export_completed(index, candidate):
        completed.append({"index": index, "expression": str(candidate.expr[0]),
            "parameters": {str(k): float(v) if np.isfinite(float(v)) else str(float(v)) for k, v in candidate.params.items()},
            "rhs_order": list(map(str, candidate.rhs_vars)), "valid": bool(candidate.valid)})
        if optimizer == "DE" and candidate.valid:
            for key, value in candidate.params.items():
                value = float(value)
                if np.isfinite(value):
                    bound_usage.append({"native_index": index, "parameter": str(key), "value": value,
                        "lower_distance": value - bounds[0], "upper_distance": bounds[1] - value,
                        "normalized_margin": min(value - bounds[0], bounds[1] - value) / (bounds[1] - bounds[0])})
        try:
            if not candidate.valid:
                raise ValueError("Native ProGED model did not fit successfully")
            equation = _export(candidate, request.variables)
            if optimizer == "DE":
                for parameter in equation["parameters"].values():
                    parameter["bounds"] = list(bounds)
            columns = [ids.index(str(symbol)) for symbol in candidate.rhs_vars]
            candidates.append(
                checked_candidate(
                    "proged",
                    index,
                    equation,
                    request,
                    lambda X, native=candidate, order=columns: np.asarray(
                        native.evaluate(X[:, order], native.params)
                    ).reshape(-1),
                )
            )
        except (ValueError, TypeError, OverflowError) as exc:
            rejected.append({"index": index, "reason": str(exc)})

    def fit_native_models(function, models):
        for index, candidate in enumerate(models):
            if optimizer == "DE":
                function.__self__.settings["parameter_estimation"]["param_bounds"] = [tuple(bounds)] * len(candidate.params)
            phase.update(phase="fit", active_native_index=index)
            phase["fit_started_count"] += 1
            save_phase()
            started = time.monotonic()
            try:
                fitted = function(candidate)
            except BaseException:
                phase["native_fit_seconds"] += time.monotonic() - started
                phase["phase"] = "fit_interrupted"
                save_phase()
                raise
            phase["native_fit_seconds"] += time.monotonic() - started
            phase["fit_completed_count"] += 1
            phase["phase"] = "export"
            save_phase()
            started = time.monotonic()
            export_completed(index, fitted)
            phase["export_seconds"] += time.monotonic() - started
            phase["phase"] = "completed_candidate"
            if context.checkpoint is not None:
                started = time.monotonic()
                context.checkpoint(completed_result())
                phase["checkpoint_seconds"] += time.monotonic() - started
            save_phase()
            yield fitted

    native_minimize = native_estimation.sp_minimize
    # This adapter runs in an isolated worker. Preserve substitution/codegen;
    # omit only repeatedly constructed function documentation during fitting.
    import sympy as sp
    native_lambdify = sp.lambdify
    numerical_lambdify = partial(native_lambdify, docstring_limit=0)

    def configured_minimize(*args, **kwargs):
        if iterations is not None:
            kwargs["options"] = {**kwargs.get("options", {}), "maxiter": iterations}
        result = native_minimize(*args, **kwargs)
        optimizer_work.append({"nit": int(result.nit), "nfev": int(result.nfev),
            "success": bool(result.success), "status": int(result.status), "message": str(result.message)})
        return result

    save_phase()
    started = time.monotonic()
    model.generate_models()
    phase.update(generation_seconds=time.monotonic() - started, generated_count=len(model.models), phase="generated")
    save_phase()
    if optimizer == "local":
        native_estimation.sp_minimize = configured_minimize
    sp.lambdify = numerical_lambdify
    try:
        model.fit_models(pool_map=fit_native_models)
    finally:
        native_estimation.sp_minimize = native_minimize
        if sp.lambdify is numerical_lambdify:
            sp.lambdify = native_lambdify
    phase["phase"] = "completed"
    save_phase()
    return completed_result(finalized=True)
