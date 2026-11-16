"""NASA PySIPS native SMC population and explicitly separate portable point candidates."""

import json
import random
import time
from collections import Counter
from importlib.metadata import version

import numpy as np

from ..data import write_json
from ..domains import probe_points, validation_constraints
from ..ensemble import load_ensemble
from ..expressions import canonical_integer_powers, inspect_equation
from .base import DiscoveryResult, checked_candidate, validate_options

NATIVE_OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "absolute value",
    "pow": "pow",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
}


SUPPORTED_OPTIONS = {
    "constant_probability",
    "crossover_pool_size",
    "max_complexity",
    "mcmc_samples",
    "model_selection",
    "opt_restarts",
    "param_init_bounds",
    "particles",
    "search_operators",
    "seconds",
    "terminal_probability",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": [*NATIVE_OPERATORS, "pow_int"],
        "requires_assets": False,
        "constraints": {
            "operators": "native Bingo generator; log/sqrt use abs and require allowed abs; ordinary power/division, fixed square/cube; common post-filter",
            "units": "native dimensions unsupported; coefficient units unknown",
            "parameter_bounds": "native initialization range is not a hard bound; common post-filter",
            "max_nodes": "native command-stack budget plus common post-filter",
            "variables": "post-filtered",
            "probes": "post-filtered",
            "posterior": "native weighted structural ensemble conditional on fitted coefficients; no coefficient/noise posterior or calibrated observation intervals",
        },
    }


def export_graph(model, variables):
    from bingo.symbolic_regression.agraph import operator_definitions as op

    # get_local_optimization_params does NOT update a dirty graph in Bingo0.5.7.
    # The public complexity getter performs that update before pairing constants/stack.
    model.get_complexity()
    constants = list(model.get_local_optimization_params())
    commands = np.asarray(model._simplified_command_array)
    parameters = {}

    def visit(index):
        node, first, second = map(int, commands[index])
        if node == op.INTEGER:
            return {"const": first}
        if node == op.VARIABLE:
            if not 0 <= first < len(variables):
                raise ValueError("Native Bingo variable exceeds supplied width")
            return {"var": variables[first]["id"]}
        if node == op.CONSTANT:
            key = f"c{first}"
            parameters[key] = {
                "value": float(constants[first]),
                "trainable": True,
                "origin": "native_bingo_constant_slot",
                "bounds": [None, None],
                "unit": None,
            }
            return {"param": key}
        binary = {
            op.ADDITION: "add",
            op.SUBTRACTION: "sub",
            op.MULTIPLICATION: "mul",
            op.DIVISION: "div",
            op.POWER: "pow",
        }
        unary = {
            op.SIN: "sin",
            op.COS: "cos",
            op.EXPONENTIAL: "exp",
            op.ABS: "abs",
            op.SINH: "sinh",
            op.COSH: "cosh",
            op.TANH: "tanh",
            op.TAN: "tan",
            op.ARCSIN: "asin",
            op.ARCCOS: "acos",
            op.ARCTAN: "atan",
        }
        if not 0 <= first < index:
            raise ValueError(f"Unsupported or invalid native Bingo command {node}")
        if node in binary and 0 <= second < index:
            return {"op": binary[node], "args": [visit(first), visit(second)]}
        if node in unary:
            return {"op": unary[node], "args": [visit(first)]}
        if node in {op.LOGARITHM, op.SQRT}:
            return {"op": "log" if node == op.LOGARITHM else "sqrt", "args": [{"op": "abs", "args": [visit(first)]}]}
        if node in {op.SQUARE, op.CUBE}:
            return {"op": "pow_int", "args": [visit(first), {"const": 2 if node == op.SQUARE else 3}]}
        raise ValueError(f"Unsupported or invalid native Bingo command {node}")

    equation = {
        "schema_version": 2,
        "arithmetic": "numpy_ieee",
        "ast": visit(len(commands) - 1),
        "parameters": parameters,
    }
    equation = canonical_integer_powers(equation)
    inspect_equation(equation, variables)
    return equation


def validate_request(request, context):
    """Validate native configuration without loading PySIPS or Bingo."""
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    particles = request.options.get("particles", 100)
    moves = request.options.get("mcmc_samples", 10)
    if type(particles) is not int or particles < 1 or type(moves) is not int or moves < 1:
        raise ValueError("PySIPS particles and MCMC moves must be positive integers")
    restarts = request.options.get("opt_restarts", 1)
    initialization = request.options.get("param_init_bounds", [-5, 5])
    if (
        type(restarts) is not int
        or restarts < 1
        or not isinstance(initialization, (list, tuple))
        or len(initialization) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in initialization)
        or not np.isfinite(initialization).all()
        or initialization[0] >= initialization[1]
    ):
        raise ValueError("Invalid native optimizer restart/initialization settings; initialization is not a hard bound")

    search_operators = request.options.get("search_operators", request.operators)
    if not search_operators or not set(search_operators) <= set(request.operators):
        raise ValueError("PySIPS search_operators must be a nonempty subset of the shared allowed vocabulary")
    native_ops = [
        NATIVE_OPERATORS[op]
        for op in search_operators
        if op in NATIVE_OPERATORS and (op not in {"log", "sqrt"} or "abs" in request.operators)
    ]
    if "pow_int" in search_operators:
        native_ops.extend(["square", "cube"])
    if not native_ops:
        raise ValueError("No faithful Bingo operator remains; native log/sqrt require abs in the requested vocabulary")
    native_settings = {
        name: request.options[name]
        for name in ("terminal_probability", "constant_probability", "crossover_pool_size", "model_selection")
        if name in request.options
    }
    for name in ("terminal_probability", "constant_probability"):
        if name in native_settings and (
            isinstance(native_settings[name], bool)
            or not isinstance(native_settings[name], (int, float)) or not 0 <= native_settings[name] <= 1
        ):
            raise ValueError(f"Invalid native {name}")
    if "crossover_pool_size" in native_settings and (
        type(native_settings["crossover_pool_size"]) is not int or native_settings["crossover_pool_size"] < 2
    ):
        raise ValueError("Native crossover_pool_size must be at least2")
    if native_settings.get("model_selection", "mode") not in {"mode", "max_nml"}:
        raise ValueError("Native model_selection must be mode or max_nml")
    for key in ('max_complexity',):
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"pysips {key} must be a positive integer")
    if "seconds" in request.options:
        seconds = request.options["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("pysips seconds must be positive and finite")
    return particles, moves, restarts, initialization, native_ops, native_settings, search_operators


def discover(request, context):
    resolved = validate_request(request, context)
    # PySIPS requests Python AGraph objects, but Bingo can still default their
    # evaluation/simplification to an older installed C++ opcode set. Use the
    # official Python backend for the complete fit/export lifetime, then restore
    # caller state. This changes no SMC, optimization or operator semantics.
    from bingo.symbolic_regression.agraph import agraph

    previous = agraph.evaluation_backend, agraph.simplification_backend
    try:
        agraph.force_use_of_python_backends()
        return _discover(request, context, resolved)
    finally:
        agraph.evaluation_backend, agraph.simplification_backend = previous


def _discover(request, context, resolved):
    import pysips.regressor as implementation
    import pysips.sampler as sampler
    from pysips import PysipsRegressor
    from pysips.crossover_proposal import CrossoverProposal

    particles, moves, restarts, initialization, native_ops, native_settings, search_operators = resolved

    class SeededRegressor(PysipsRegressor):
        def _get_proposal(self, dimension, generator):
            proposal = super()._get_proposal(dimension, generator)
            # The pinned wrapper omits public seed arguments when constructing its
            # native proposal RNGs. Seed those exact objects; preserve their algorithms.
            seeds = np.random.SeedSequence(self.random_state).spawn(3)
            proposal._rng = np.random.default_rng(seeds[0])
            for child, seed in zip(proposal._proposals, seeds[1:], strict=True):
                child._rng = np.random.default_rng(seed)
            crossover = proposal._proposals[1]
            crossover._gene_pool = sorted(crossover._gene_pool, key=lambda m: m.command_array.tobytes())
            return proposal

    fitted = SeededRegressor(
        operators=native_ops,
        max_complexity=request.options.get("max_complexity", 12),
        num_particles=particles,
        num_mcmc_samples=moves,
        max_time=request.options.get("seconds", request.limits["max_wall_seconds"] * 0.6),
        random_state=request.seed,
        show_progress_bar=False,
        **native_settings,
    )
    native_crossover, native_smc = CrossoverProposal.__call__, sampler._smc_call
    native_likelihood = implementation.LaplaceNmll
    observations = {"collapsed_pool_self_transitions": 0, "likelihood_calls": 0}
    before_refit = []
    artifacts = {}

    def record_particle(model, index, weight):
        original = model
        model = model.copy()  # Preserve the actual sampled object and coefficient mode.
        constants = [float(v) if np.isfinite(v) else str(float(v)) for v in model.get_local_optimization_params()]
        record = {
            "particle": index,
            "weight": float(weight),
            "commands": model._simplified_command_array.tolist(),
            "constants": constants,
        }
        candidate = None
        try:
            equation = export_graph(model, request.variables)
            candidate = checked_candidate(
                "pysips", index, equation, request, lambda X: np.asarray(original.evaluate_equation_at(X)).reshape(-1)
            )
            record.update(equation=equation, native_check=candidate.native_check)
        except (ValueError, TypeError, IndexError) as exc:
            record["export_failure"] = str(exc)
        return record, candidate

    def crossover(proposal, model):
        if not any(parent != model for parent in proposal._gene_pool):
            observations["collapsed_pool_self_transitions"] += 1
            return model.copy()
        return native_crossover(proposal, model)

    def observe_smc(*args, **kwargs):
        start = time.monotonic()
        steps, phases = native_smc(*args, **kwargs)
        observations["sampling_seconds"] = time.monotonic() - start
        weights = np.asarray(steps[-1].weights).reshape(-1)
        observations["final_smc_weights"] = weights.tolist()
        probes = probe_points(request.X, request.variables, request.seed)
        native_mean = np.zeros(len(probes))
        # sampler.run_smc next refits these SAME graphs. Preserve their sampling
        # state now, not its later NMLL scores or newly optimized coefficients.
        for index, (model, weight) in enumerate(zip(steps[-1].params[:, 0], weights, strict=True)):
            if weight > 0:
                native_mean += weight * np.asarray(model.evaluate_equation_at(probes)).reshape(-1)
            record, _ = record_particle(model, index, weight)
            before_refit.append(record)
        raw_path = write_json(
            context.work_dir / "native_smc_population.json",
            {
                "population": before_refit,
                "tempering_sequence": np.asarray(phases).reshape(-1).tolist(),
                "scope": "actual final SMC state BEFORE native per-particle point refit; coefficient modes are not posterior draws",
                "scores_are_weights": False,
                "native_ensemble_probe": {
                    "X": probes.tolist(),
                    "weighted_prediction": native_mean.tolist() if np.isfinite(native_mean).all() else None,
                    "scope": "actual weighted native sampling-state prediction before point refit, no Final rows",
                },
                "provenance": context.provenance,
            },
        )
        artifacts["population"] = raw_path
        return steps, phases

    class ObservedLikelihood(native_likelihood):
        def __call__(self, *args, **kwargs):
            observations["likelihood_calls"] += 1
            return super().__call__(*args, **kwargs)

    def configured_likelihood(X, y):
        # Native fit() forgets to forward these exposed settings. Only plumb them
        # into the unchanged native likelihood/LM implementation.
        return ObservedLikelihood(X, y, opt_restarts=restarts, param_init_bounds=initialization)

    np.random.seed(request.seed)
    random.seed(request.seed)
    CrossoverProposal.__call__, sampler._smc_call = crossover, observe_smc
    implementation.LaplaceNmll = configured_likelihood
    try:
        fitted.fit(request.X, request.y)
    finally:
        CrossoverProposal.__call__, sampler._smc_call = native_crossover, native_smc
        implementation.LaplaceNmll = native_likelihood
    models, scores = fitted.get_models()
    scores = np.asarray(scores).reshape(-1)
    phases = np.asarray(fitted.phis_, dtype=float).reshape(-1)
    weights = observations["final_smc_weights"]
    candidates, rejected, population = [], [], []
    for index, (model, score, weight) in enumerate(zip(models, scores, weights, strict=True)):
        record, candidate = record_particle(model, index, weight)
        record.update(
            native_score=float(score) if np.isfinite(score) else None,
            score_nonfinite=str(score) if not np.isfinite(score) else None,
        )
        population.append(record)
        if candidate is not None:
            candidates.append(candidate)
        else:
            rejected.append({"particle": index, "reason": record["export_failure"]})
    artifacts["refitted_points"] = write_json(
        context.work_dir / "native_refitted_points.json",
        {
            "population": population,
            "scope": "native final point refits; NMLL scores are NOT weights",
            "weight_scope": "original SMC row weights; coefficient refits do not make posterior parameter samples",
        },
    )
    mass = Counter()
    for record in before_refit:
        mass[json.dumps(record["commands"], separators=(",", ":"))] += record["weight"]
    ensemble_status, ensemble_reason = (
        "unavailable",
        "A positive-weight native particle could not be faithfully exported; no mass was dropped",
    )
    positive = [record for record in before_refit if record["weight"] > 0]
    if positive and all("equation" in record for record in positive):
        split_seed = context.provenance.get("split_seed")
        constraints = validation_constraints(
            {"constraints": request.limits, "split": {"seed": request.seed if split_seed is None else split_seed}},
            request.variables,
            request.target,
            request.X,
        )
        path = write_json(
            context.work_dir / "structural_ensemble.json",
            {
                "schema_version": 2 if context.provenance.get("input_geometry") else 1,
                "kind": "structural_point_ensemble",
                "schema": {
                    "variables": request.variables,
                    "public_units": context.provenance.get("public_units", {}),
                    "unit_coordinates": context.provenance.get("unit_coordinates"),
                    "input_geometry": context.provenance.get("input_geometry"),
                    "target": request.target,
                    "feature_ranges": np.column_stack([request.X.min(axis=0), request.X.max(axis=0)]).tolist(),
                },
                "members": [
                    {
                        "particle": r["particle"],
                        "equation": r["equation"],
                        "weight": r["weight"],
                        "native_check": r["native_check"],
                    }
                    for r in positive
                ],
                "operators": request.operators,
                "constraints": constraints,
                "complexity_coordinate": "source",
                "metadata": {
                    "engine": "pysips",
                    "seed": request.seed,
                    "version": version("pysips"),
                    "source_revision": "e155455d1dcd569495655d49f8438ab4e2636c3f",
                    "provenance": context.provenance,
                    "uncertainty_scope": "weighted structural points conditional on native fitted coefficient modes; no observation/parameter posterior intervals",
                    "zero_weight_particles_omitted": len(before_refit) - len(positive),
                    "coefficient_refit": False,
                    "score_weight_conversion": False,
                },
            },
        )
        artifacts["ensemble"] = path
        try:
            load_ensemble(path)  # Common constraints, never condition/renormalize native mass.
            ensemble_status, ensemble_reason = "available", None
        except ValueError as exc:
            ensemble_reason = str(exc)
    return DiscoveryResult(
        candidates,
        "ok" if candidates else "failed",
        {
            "version": version("pysips"),
            "bingo_version": version("bingo-nasa"),
            "smcpy_version": version("smcpy"),
            "source_revision": "e155455d1dcd569495655d49f8438ab4e2636c3f",
            "license": "NASA-1.3",
            "native_candidates": len(models),
            "native_graph_backend": "official Bingo Python evaluation and simplification; caller backends restored after export",
            "native_search_operators": native_ops,
            "requested_generation_operators": search_operators,
            "native_operator_semantics": {
                "log": "log(abs(x)); abs must be explicitly allowed",
                "sqrt": "sqrt(abs(x)); abs must be explicitly allowed",
                "pow": "ordinary power; not safe_pow",
                "div": "ordinary IEEE division, portable domain checks mandatory",
            },
            "native_generation_settings": native_settings,
            "unavailable_generation_operators": [
                op for op in ("log", "sqrt") if op in request.operators and "abs" not in request.operators
            ],
            "particles": particles,
            "mcmc_moves_per_step": moves,
            "tempering_final": float(phases[-1]) if len(phases) else None,
            "tempering_endpoint_one": bool(len(phases) and np.isclose(phases[-1], 1)),
            "sampling_scope": "fixed-time sampler can force phi=1; endpoint/ESS is not convergence or mixing evidence",
            "native_structure_prior": "native Metropolis constant graph prior; generation probabilities are not an evaluated grammar prior",
            "structural_ensemble_status": ensemble_status,
            "structural_ensemble_reason": ensemble_reason,
            "distinct_native_command_stacks": len(mass),
            "top_structure_mass": max(mass.values()) if mass else None,
            "point_selection": "common Selection selects a point; neither common selection nor refitting mutates the saved sampling ensemble",
            "rng_scope": "NumPy/Bingo and native proposal RNGs seeded; stable crossover pool order; wall-time trajectory is not bit-reproducible",
            "optimizer": {
                "restarts": restarts,
                "initialization_range": initialization,
                "hard_coefficient_bounds": False,
            },
            "smc_observation": observations,
            "rejected_exports": rejected,
            "constraint_support": capabilities()["constraints"],
        },
        artifacts,
    )
