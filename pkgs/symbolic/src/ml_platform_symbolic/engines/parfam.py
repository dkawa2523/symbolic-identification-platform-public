"""Actual ParFam fitted families and winner variants; no common coefficient fitting."""

import hashlib
import inspect
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ..data import write_json
from ..expression_graph import compact_equation, polynomial_node
from ..expressions import inspect_equation
from .base import DiscoveryResult, checked_candidate, native_input_indices, validate_options
from .torch_runtime import thread_count, torch_thread_scope


def _model_parameters(parameters):
    return {
        key: [getattr(v, "__name__", str(v)) for v in value] if key in {"functions", "function_names"} else value
        for key, value in parameters.items()
    }


SUPPORTED_OPTIONS = {
    "threads",
    "objective",
    "candidate_scope",
    "degree",
    "denominator_degree",
    "family",
    "family_search",
    "function_power",
    "functions",
    "input_degree",
    "max_active_functions",
    "max_active_parameters",
    "maxiter",
    "repetitions",
    "seconds",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "response_enforcement": {"default": "hard", "supported": ['hard'], "stage": "native family coefficient optimization and selection"},
        "operators": ["add", "sub", "mul", "div", "pow_int", "sin", "cos", "exp", "log", "sqrt"],
        "requires_assets": False,
        "equation_template": True,
        "family_search": "fixed keeps one configured family; up_to searches bounded degree families. Default polynomial=fixed, rational=up_to; Complete includes lower-degree extension",
        "search_constraints": ["response_bounds", "response_checks", "probes"],
        "constraints": {
            "equation_template": "one shared slot, fixed outer parameters; native coefficient-family search in the full strict-real composition against original y; nested slots, outer free parameters and mapped coordinates rejected",
            "operators": "native polynomial/rational family with explicitly configured ordinary torch functions; exports post-filtered",
            "functions_and_denominators": "ordinary functions installed through public extend_function_dict during fit/reconstruction; upstream protected defaults are not relabelled ordinary",
            "units": "unknown fitted coefficient units; portable post-filter",
            "max_nodes": "source AST family-size filtering before coefficient fit; extends Complete/Max with missing lower-degree families, preserves original order/repetitions and fixed No mode, rejects an empty feasible set; active-parameter count remains a stopping preference",
            "variables": "disallowed columns excluded before native search; full-schema export/prediction checked",
            "allowed_variables": "native input-column subset with original IDs; no constant-only or mapped-input/template subset fallback",
            "probes": "declared-point finite predictions required during coefficient search and native family/reduction selection; post-filter only in explicit filter mode",
            "response_bounds": "explicit SLSQP inequalities/equalities in native coefficient fitting and pruning refits; strict feasibility in native family/reduction selection; declared points only",
            "response_checks": "finite-step response inequalities/equalities passed to SLSQP coefficient fitting and pruning refits; strict feasibility in native family/reduction selection; no global guarantee",
        },
    }


def _export(model, values, variables, function_names, *, reduced=False):
    """Native monomial/block order, including native denominator normalization."""
    coefficients = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(coefficients).all() or model.stabilize:
        raise ValueError("ParFam export requires finite coefficients and ordinary division throughout native fitting")
    parameters = {}

    def polynomial(powers, block, offset, features, normalize=False):
        block = np.asarray(block)
        if len(powers) != len(block):
            raise ValueError("Unexpected ParFam coefficient block")
        if normalize and model.normalize_denom:
            norm = np.linalg.norm(block)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Zero/invalid native denominator coefficient norm")
            block = block / norm
        terms = []
        for index, (indices, value) in enumerate(zip(powers, block, strict=True)):
            key = f"c{offset + index}"
            parameters[key] = {
                "value": float(value),
                "trainable": True,
                "bounds": [None, None],
                "unit": None,
                "origin": "native_normalized_denominator_coordinate"
                if normalize and model.normalize_denom
                else "native_coefficient_block",
            }
            if reduced and coefficients[offset + index] == 0:
                parameters[key].update(trainable=False, bounds=[0.0, 0.0], origin="native_reduction_inactive_zero")
            if normalize and model.normalize_denom:
                parameters[key].update(
                    native_value=float(coefficients[offset + index]),
                    normalization_group=f"denominator_{offset}",
                    normalization_norm=float(norm),
                )
            powers = []
            for feature_index, exponent in enumerate(indices):
                if int(exponent) != exponent or exponent < 0:
                    raise ValueError("Invalid native monomial exponent")
                if exponent:
                    powers.append([feature_index, int(exponent)])
            terms.append({"coefficient": {"param": key}, "powers": powers})
        return polynomial_node(features, terms) if terms else {"const": 0}

    features = [{"var": v["id"]} for v in variables]
    hidden = list(features)
    for i, name in enumerate(function_names * model.width):
        offset = i * model.n_coefficients_first_layer
        length = model.n_coefficients_first_layer_numerator
        numerator = polynomial(
            model.monomial_list_input_numerator, coefficients[offset : offset + length], offset, features
        )
        if model.degree_input_denominator:
            start, end = offset + length, offset + model.n_coefficients_first_layer
            denominator = polynomial(
                model.monomial_list_input_denominator, coefficients[start:end], start, features, True
            )
            argument = {"op": "div", "args": [numerator, denominator]}
        else:
            argument = numerator
        hidden.append({"op": name, "args": [argument]})
    start = len(coefficients) - model.n_coefficients_last_layer
    end = (
        len(coefficients) - model.n_coefficients_last_layer_denominator
        if model.degree_output_denominator
        else len(coefficients)
    )
    numerator = polynomial(model.monomial_list_output_numerator, coefficients[start:end], start, hidden)
    if model.degree_output_denominator:
        denominator = polynomial(model.monomial_list_output_denominator, coefficients[end:], end, hidden, True)
        root = {"op": "div", "args": [numerator, denominator]}
    else:
        root = numerator
    equation = compact_equation(
        {"schema_version": 2, "ast": root, "parameters": parameters, "arithmetic": "numpy_ieee"}
    )
    inspect_equation(equation, variables)
    return equation


def family_search(options, family):
    mode = options.get("family_search", "fixed" if family == "polynomial" else "up_to")
    if not isinstance(mode, str) or mode not in {"fixed", "up_to"}:
        raise ValueError("ParFam family_search must be fixed or up_to")
    return mode


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    from ..response_search import response_enforcement
    response_enforcement(request.limits, capabilities())
    native_input_indices(request, context)
    if request.equation_template is not None:
        from .parfam_template import validate_template_scope
        validate_template_scope(request.equation_template, context)
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    if request.options.get("objective") not in {None, "mse"}:
        raise ValueError("ParFam objective, when supplied, must be 'mse'")
    family = request.options.get("family", "rational")
    if family not in {"polynomial", "rational"}:
        raise ValueError("ParFam family must be polynomial or rational")
    family_search(request.options, family)
    degree = request.options.get("degree", 2)
    input_degree = request.options.get("input_degree", 2)
    denominator_degree = request.options.get("denominator_degree", 1) if family == "rational" else 0
    functions = (
        request.options.get("functions", [op for op in ("sin", "cos", "exp", "log", "sqrt") if op in request.operators])
        if family == "rational"
        else []
    )
    if any(f not in {"sin", "cos", "exp", "log", "sqrt"} or f not in request.operators for f in functions):
        raise ValueError("ParFam elementary functions must be requested ordinary sin/cos/exp/log/sqrt")
    if not all(type(v) is int and 1 <= v <= 3 for v in (degree, input_degree)) or (
        type(denominator_degree) is not int or not 0 <= denominator_degree <= 2
    ):
        raise ValueError("ParFam numerator/input degree1..3 and denominator degree0..2 required")
    if denominator_degree and "div" not in request.operators:
        raise ValueError("Requested rational family requires div in the common allowed operators")
    active_functions = request.options.get("max_active_functions", len(functions))
    function_power = request.options.get("function_power", 1)
    scope = request.options.get("candidate_scope", "completed_families")
    if (
        type(active_functions) is not int
        or not 0 <= active_functions <= len(functions)
        or type(function_power) is not int
        or not 1 <= function_power <= 3
    ):
        raise ValueError("Invalid native active-function count or function_power")
    if scope not in {"winner", "completed_families"}:
        raise ValueError("ParFam candidate_scope must be winner or completed_families")
    for key in ('maxiter', 'repetitions', 'max_active_parameters'):
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"parfam {key} must be a positive integer")
    if "seconds" in request.options:
        seconds = request.options["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("parfam seconds must be positive and finite")
    thread_count(request.options)
    return family, degree, input_degree, denominator_degree, functions, active_functions, function_power, scope


def discover(request, context):
    validate_request(request, context)
    from .parfam_template import native_template_scope
    with torch_thread_scope(request.options), native_template_scope(request, context, _export):
        return _discover(request, context)


def _discover(request, context):
    family, degree, input_degree, denominator_degree, functions, active_functions, function_power, scope = (
        validate_request(request, context)
    )
    feature_indices = native_input_indices(request, context)
    native_variables = [request.variables[i] for i in feature_indices]
    import parfam.train as native_train
    import sympy as sp
    import torch
    from parfam import ParFamWrapper

    if getattr(native_train, "symbolic_handoff_division_version", None) != 1:
        raise ValueError("Rebuild the classic Docker profile for pinned ParFam ordinary-division flag plumbing")
    from ..response_search import compile_response_constraints, requested_response_constraints
    from .parfam_objective import ResponseObjective

    objective = None
    fit_failure = None
    search_responses = request.limits.get("response_mode", "search") == "search"
    if request.equation_template is not None or requested_response_constraints(request.limits) or request.options.get("objective") == "mse":
        import parfam.parfam_torch as native_tensor
        if (getattr(native_train, "symbolic_handoff_objective_version", None) != 3
                or getattr(native_tensor, "symbolic_handoff_objective_version", None) != 3):
            raise ValueError("Rebuild classic profile for ParFam search-time response objective plumbing")
        compiled = compile_response_constraints(request)
        if not search_responses:
            compiled = {**compiled, "points": [], "terms": []}
        if compiled["points"]:
            from ..domains import region_mask
            points = np.asarray(compiled["points"])
            if not region_mask(points, request.domain).all():
                raise ValueError("ParFam response stencil outside full declared joint domain")
            compiled = {**compiled, "points": points[:, feature_indices].tolist()}
        objective = ResponseObjective(compiled)
        objective.enforce = search_responses and bool(compiled["terms"])
        if request.equation_template is not None:
            if (getattr(native_train, "symbolic_handoff_fit_failure_version", None) != 1
                    or getattr(native_tensor, "symbolic_handoff_fit_failure_version", None) != 1):
                raise ValueError("Rebuild classic profile for ParFam explicit template fit failures")
            objective.reject_nonfinite_gradients = True
    response_active = objective is not None and search_responses and bool(objective.compiled["points"])
    torch.manual_seed(request.seed)
    torch.set_default_dtype(torch.float64)
    np.random.seed(request.seed)

    def make_wrapper():
        fitted = ParFamWrapper(
            iterate=family_search(request.options, family) == "up_to",
            functions=[getattr(torch, f) for f in functions],
            function_names=[getattr(sp, f) for f in functions],
            degree_input_numerator=input_degree,
            degree_output_numerator=degree,
            degree_input_denominator=denominator_degree,
            degree_output_denominator=denominator_degree,
            maximal_n_active_base_functions=active_functions,
            degree_output_numerator_specific=[function_power] * len(functions),
            degree_output_denominator_specific=[function_power] * len(functions),
            maximal_potence=degree,
            separate_test_set=False,
            device=context.device,
        )
        fitted.config["TRAININGPARAMETERS"]["stabilize_denominator"] = "False"
        fitted.config["TRAININGPARAMETERS"]["parallel"] = "False"
        fitted.config["TRAININGPARAMETERS"]["target_noise"] = "0.0"
        fitted.config["TRAININGPARAMETERS"]["feature_noise"] = "0.0"
        return fitted

    family_fits, failed_family_fits = [], []
    native_training = native_train.hyperparameter_training
    signature = inspect.signature(native_training)

    def observe_training(*args, **kwargs):
        original_setup = native_train.setup_model
        models, row_usage = [], []

        def observe_setup(*setup_args, **setup_kwargs):
            model = original_setup(*setup_args, **setup_kwargs)
            models.append(model)
            caller = inspect.currentframe().f_back
            if caller.f_code is not inspect.unwrap(native_training).__code__:
                raise ValueError("Unexpected ParFam family setup caller; cannot observe native row usage")
            local = caller.f_locals
            row_usage.append({
                "supplied_train_rows": len(local["x"]),
                "coefficient_fit_rows": len(local["x_train"]),
                "internal_validation_rows": len(local["x_val"]),
                "subsampled_out_rows": len(local["x"]) - len(local["x_val"]) - len(local["x_train"]),
                "max_dataset_length": int(local["max_dataset_length"]),
                "split_seed": int(local["seed"]),
                "scope": "actual native family arrays after internal split and optional subsampling; not common Selection",
            })
            del caller
            return model

        native_train.setup_model = observe_setup
        try:
            result = native_training(*args, **kwargs)
        finally:
            native_train.setup_model = original_setup
        if len(models) != 1:
            raise ValueError("Unexpected pinned ParFam family model construction; inspect native API")
        arguments = signature.bind(*args, **kwargs).arguments
        coefficients = (None if request.equation_template is not None and result[4] is None
                        else torch.as_tensor(result[4]).detach().clone())
        if request.equation_template is not None and (coefficients is None or not (
                np.isfinite(float(result[0])) and np.isfinite(float(result[1]))
                and bool(torch.isfinite(coefficients).all()))):
            failed_family_fits.append({"reason": "No finite native family fit",
                "native_training_seconds": float(result[3]), "native_evaluations": int(result[5]),
                "row_usage": row_usage[0]})
            return result
        family_fits.append(
            {
                "parameters": deepcopy(arguments["model_parameters"]),
                "model": models[0],
                "coefficients": coefficients,
                "native_training_seconds": float(result[3]),
                "native_evaluations": int(result[5]),
                "native_internal_validation_loss": float(result[1]),
                "row_usage": row_usage[0],
            }
        )
        return result  # Same native optimization, budget and native family selection.

    finetuning_rows = []
    native_finetune = native_train.finetune_coeffs
    finetune_signature = inspect.signature(native_finetune)

    def observe_finetune(*args, **kwargs):
        arguments = finetune_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        record = {"supplied_rows": len(arguments.arguments["x_train"]),
                  "max_dataset_length": int(arguments.arguments["max_dataset_length"]),
                  "coefficient_fit_rows": 0, "optimizer_started": False}
        original_evaluator = native_train.Evaluator

        def observe_evaluator(*values, **options):
            caller = inspect.currentframe().f_back
            if caller.f_code is not native_finetune.__code__:
                raise ValueError("Unexpected native finetuning evaluator caller")
            record.update(coefficient_fit_rows=len(caller.f_locals["x_train"]), optimizer_started=True)
            del caller
            return original_evaluator(*values, **options)

        native_train.Evaluator = observe_evaluator
        try:
            return native_finetune(*args, **kwargs)
        finally:
            native_train.Evaluator = original_evaluator
            finetuning_rows.append(record)

    native_train.finetune_coeffs = observe_finetune
    native_train.hyperparameter_training = observe_training
    original_functions = native_train.function_dict
    original_names = native_train.function_name_dict
    try:
        fitted = make_wrapper()
        from ..coordinates import source_complexity_costs, source_equation
        from .parfam_budget import family_budget
        coordinate_context = {**context.provenance, "variables": request.variables,
                              "target": {"column": "target", **request.target}}
        costs, output_cost = source_complexity_costs(coordinate_context)
        family_budget_audit = []
        with family_budget(native_train, request.limits["max_nodes"],
                           [costs[v["id"]] for v in native_variables], output_cost, audit=family_budget_audit):
            fitted.fit(
                request.X[:, feature_indices],
                request.y,
                time_limit=request.options.get("seconds", request.limits["max_wall_seconds"] * 0.8),
                seed=request.seed,
                normalization=False,
                custom_loss=objective,
                maxiter1=request.options.get("maxiter", 50),
                repetitions=request.options.get("repetitions", 1),
                max_n_active_parameters=request.options.get("max_active_parameters", 12),
            )
    except Exception as exc:
        errors = tuple(cls for cls in (getattr(native_train, "NoFiniteFamilyError", None),
                       getattr(native_train, "NonfiniteGradientError", None)) if isinstance(cls, type))
        if request.equation_template is None or not isinstance(exc, errors):
            raise
        fit_failure = str(exc)
    finally:
        native_train.hyperparameter_training = native_training
        native_train.finetune_coeffs = native_finetune
        native_train.function_dict = original_functions
        native_train.function_name_dict = original_names
    if not family_budget_audit and fit_failure is None:
        raise RuntimeError("ParFam family size gate was bypassed")
    if fit_failure is not None and not family_fits:
        failure_path = write_json(context.work_dir / "parfam_native.json", {
            "failure": fit_failure, "failed_family_fits": failed_family_fits,
            "nonfinite_gradient_fits": getattr(objective, "nonfinite_gradient_fits", 0),
            "nonfinite_result_fits": getattr(objective, "nonfinite_result_fits", 0)})
        return DiscoveryResult([], "failed", {"reason": fit_failure,
            "native_fit_completed": False, "native_execution_status": "failed",
            "native_completed_family_fits": 0,
            "failed_family_fits": failed_family_fits,
            "nonfinite_gradient_fits": getattr(objective, "nonfinite_gradient_fits", 0),
            "nonfinite_result_fits": getattr(objective, "nonfinite_result_fits", 0),
            "gradient_failure_policy": "nonfinite autodiff rejects that local fit; no numerical fallback",
            "runtime_fit_failure_version": getattr(native_train, "symbolic_handoff_fit_failure_version", None)},
            {"model": failure_path})
    retained = min(family_fits, key=lambda r: r["native_internal_validation_loss"]) if fit_failure else None
    model = retained["model"] if retained is not None else fitted.model
    winner_parameters = retained["parameters"] if retained is not None else fitted.best_model_parameters
    if model.stabilize:
        raise ValueError("Native ParFam unexpectedly enabled protected denominator stabilization")
    function_names = [getattr(f, "__name__", str(f)) for f in model.function_names]
    candidates, rejected = [], []
    variants = ([] if fit_failure else
                [("winner_full", model, fitted.coefficients), ("winner_reduced", model, fitted.coefficients_reduced)])
    if scope == "completed_families" or fit_failure:
        for index, record in enumerate(family_fits):
            native_model = record["model"]
            variants.append((f"completed_family_{index}", native_model, record["coefficients"]))
    native_path = write_json(
        context.work_dir / "parfam_native.json",
        {
            "model_parameters": _model_parameters(winner_parameters),
            "coefficients": {
                name: [
                    float(v) if np.isfinite(v) else str(float(v))
                    for v in torch.as_tensor(raw).reshape(-1).cpu().numpy()
                ]
                for name, _, raw in variants
            },
            "denominator_normalization": bool(model.normalize_denom),
            "stabilize_denominator": bool(model.stabilize),
            "function_order": function_names,
            "completed_families": [
                {
                    "index": i,
                    "row_usage": record["row_usage"],
                    "model_parameters": _model_parameters(record["parameters"]),
                    "function_order": [getattr(f, "__name__", str(f)) for f in record["model"].function_names],
                    "denominator_normalization": bool(record["model"].normalize_denom),
                    "native_training_seconds": record["native_training_seconds"],
                    "native_evaluations": record["native_evaluations"],
                    "native_internal_validation_loss": record["native_internal_validation_loss"]
                    if np.isfinite(record["native_internal_validation_loss"])
                    else None,
                }
                for i, record in enumerate(family_fits)
            ],
            "candidate_scope": scope,
            "family_scope": "completed native family fits before wrapper winner selection/reduction; includes last completed fit even if native search then stops on its internal budget; no new optimizer call",
            "monomials": {
                name: getattr(model, name)
                for name in [
                    "monomial_list_input_numerator",
                    "monomial_list_input_denominator",
                    "monomial_list_output_numerator",
                    "monomial_list_output_denominator",
                ]
            },
        },
    )
    for index, (name, native_model, raw) in enumerate(variants):
        try:
            coefficients = torch.as_tensor(raw, dtype=torch.float64, device=context.device).detach().clone()
            if request.equation_template is not None:
                equation = native_model.equation(coefficients.cpu().numpy(), reduced=name == "winner_reduced")
            else:
                equation = _export(
                    native_model, coefficients.cpu().numpy(), native_variables,
                    [getattr(f, "__name__", str(f)) for f in native_model.function_names],
                    reduced=name == "winner_reduced",
                )
            public_equation, public_variables = source_equation(equation, coordinate_context)
            if inspect_equation(public_equation, public_variables)["n_nodes"] > request.limits["max_nodes"]:
                raise RuntimeError("ParFam fitted family violated source-coordinate search budget")
            candidates.append(
                checked_candidate(
                    "parfam",
                    index,
                    deepcopy(equation),
                    request,
                    lambda X, c=coefficients, m=native_model: (
                        m.predict(c, torch.as_tensor(X[:, feature_indices], dtype=torch.float64, device=context.device))
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    ),
                    native_ieee=request.equation_template is None,
                )
            )
            if request.equation_template is not None:
                candidates[-1].native_check["division_policy"] = "strict-real differentiable native family composition; invalid intermediates propagate"
        except (ValueError, TypeError) as exc:
            rejected.append({"variant": name, "reason": str(exc)})
    return DiscoveryResult(
        candidates,
        ("partial" if fit_failure else "ok") if candidates else "failed",
        {
            "version": version("parfam"),
            "source_revision": "e10379a0e19238b50178492c48da0d0444d4898f",
            "license": "MIT",
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "native_parallel": False,
            "native_dtype": "float64",
            "native_candidates": len(variants),
            "source_complexity_budget": {"max_nodes": request.limits["max_nodes"], "variable_costs": costs,
                                         "output_cost": output_cost, "families": family_budget_audit,
                                          "family_search": family_search(request.options, family),
                                          "family_enumeration": "native families plus missing lower-degree topologies in Complete/Max; fixed No unchanged; original order and repetitions retained; not exhaustive formula enumeration"},
            "native_fit_completed": fit_failure is None,
            "native_execution_status": "failed" if fit_failure else "completed",
            "reason": fit_failure,
            "returned_scope": "completed_families_before_failure" if fit_failure else scope,
            "failed_family_fits": failed_family_fits,
            "nonfinite_gradient_fits": getattr(objective, "nonfinite_gradient_fits", 0),
            "nonfinite_result_fits": getattr(objective, "nonfinite_result_fits", 0),
            "gradient_failure_policy": ("nonfinite autodiff rejects that local fit; no numerical fallback"
                                        if request.equation_template is not None else "native unchanged"),
            "runtime_fit_failure_version": getattr(native_train, "symbolic_handoff_fit_failure_version", None),
            "native_completed_family_fits": len(family_fits),
            "family_row_usage": [record["row_usage"] for record in family_fits],
            "finetuning_row_usage": finetuning_rows,
            "row_usage_scope": "wrapper sets family cap to supplied Train length; native internal 80/20 uses every fitting-partition row; winner finetuning is a separate stage with its own 5000-row cap",
            "native_family_evaluations": sum(record["native_evaluations"] for record in family_fits),
            "max_active_functions": active_functions,
            "function_power": function_power,
            "variants": {f"parfam/{request.seed}/{i}": name for i, (name, _, _) in enumerate(variants)},
            "template_search": ({"slot": next(iter(request.equation_template["slots"])),
                                 "loss_target": "original observed y",
                                 "arithmetic": "strict_real",
                                 "coefficient_sharing": "one native coefficient vector shared across all slot calls",
                                 "scope": "one slot; fixed outer parameters; no nested slots or mapped coordinates"}
                                if request.equation_template is not None else None),
            "family": family,
            "native_model_parameters": _model_parameters(winner_parameters),
            "native_train_source_sha256": hashlib.sha256(Path(native_train.__file__).read_bytes()).hexdigest(),
            "ordinary_division": True,
            "normalization": False,
            "function_semantics": {
                name: f"ordinary torch.{name}, via native wrapper public custom-function extension; no clipping/abs substitution"
                for name in functions
            },
            "native_search_seconds": request.options.get("seconds", request.limits["max_wall_seconds"] * 0.8),
            "input_rows": len(request.X),
            "native_input_variables": [v["id"] for v in native_variables],
            "native_input_columns": [v["column"] for v in native_variables],
            "rejected_exports": rejected,
            "constraint_support": capabilities()["constraints"],
            "response_search": {
                "mode": ("native family search with explicitly constrained SLSQP coefficients" if response_active else "scaled_MSE_without_response_penalty") if objective else "not_requested",
                "objective_calls": objective.calls if objective else 0,
                "response_objective_calls": objective.calls if response_active else 0,
                "objective": "Train scaled MSE without native coefficient regularization, subject to declared response bounds; native family enumeration and basin hopping retained" if response_active else ("Train scaled MSE" if objective else "native default"),
                "runtime_patch_version": getattr(native_train, "symbolic_handoff_objective_version", None),
                "coefficient_solver": "SLSQP (replaces native BFGS/LBFGS); native refit reinitialization count and early-stop threshold retained" if response_active else "native BFGS/LBFGS",
                "invalid_response_solves": getattr(objective, "invalid_response_solves", 0),
                "constrained_solves": getattr(objective, "constrained_solves", 0),
                "infeasible_solves": getattr(objective, "infeasible_solves", 0),
                "feasible_returns": getattr(objective, "constrained_solves", 0) - getattr(objective, "infeasible_solves", 0),
                "solver_outcomes": getattr(objective, "solver_outcomes", {}),
                "optimizer_result_count": sum(v["returned"] for v in getattr(objective, "solver_outcomes", {}).values()),
                "no_free_parameter_solves": getattr(objective, "no_free_parameter_solves", 0),
                "solver_outcome_scope": "solver_outcomes denominator is optimizer_result_count, not constrained_solves. constrained_solves counts coefficient-solving requests; one request can yield a primary result plus retry. basinhopping counts its best-local result only, not every internal solve. feasible_returns uses constrained_solves denominator; none establishes per-output candidate convergence",
                "scope": "declared finite probe/response points; independent final verification required",
            },
        },
        {"model": native_path},
    )
