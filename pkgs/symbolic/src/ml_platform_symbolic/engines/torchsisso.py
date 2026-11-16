"""Real TorchSISSO feature expansion and sparse regression; full-precision bridge."""

import hashlib
import sys
from copy import deepcopy
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import write_json
from ..domains import region_points
from ..expressions import inspect_equation, parse_expression
from ..units import require_canonical, unit_dimensions
from .base import DiscoveryResult, checked_candidate, validate_options, native_input_indices

NATIVE_NAMES = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "exp": "exp",
    "log": "ln",
    "sin": "sin",
    "cos": "cos",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
}
OPERATORS = {*NATIVE_NAMES, "pow_int"}


SUPPORTED_OPTIONS = {
    "dimensional",
    "k",
    "n_expansion",
    "n_term",
    "response_max_iterations",
}


def capabilities():
    return {
        "unit_search": {'defaults': {'dimensional': True}, 'choices': {'dimensional': [True]}, 'templates': True},
        "equation_template": True,
        "fixed_constants": "template_only",
        "template_scope": "one affine shared slot with identical composed arguments; fixed scalar scale and known offset; dimensional only; hard finite-point response conditions through affine linear transformation",
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": sorted(OPERATORS),
        "requires_assets": False,
        "search_constraints": ["probes", "response_bounds", "response_checks"],
        "response_enforcement": {"default": "hard", "supported": ["hard"], "stage": "constrained-SIS subset coefficient optimization and residual update"},
        "constraints": {
            "operators": "native descriptor operators; affine add/mul and all exports post-filtered",
            "units": "explicit dimensional mode: native dimensional expansion/target-unit descriptor filter, dimensionless coefficients and target-unit intercept; nondimensional mode: coefficient units unknown; common structural check",
            "max_nodes": "source AST descriptor/combination budget before native affine coefficient fits; feature expansion intermediates not bounded",
            "variables": "disallowed columns excluded before native search; full-schema export/prediction checked",
            "allowed_variables": "native input-column subset with original IDs; no constant-only or mapped-input/template subset fallback",
            "probes": "finite-point hard constrained coefficient fit for each explored SIS subset; no global guarantee",
            "response_bounds": "hard coefficient constraints within subset search",
            "response_checks": "finite-step hard coefficient constraints within subset search; not analytic derivative proof",
        },
    }


def _equation(names, coefficients, intercept, variables, target_unit=None):
    parameters = {
        "intercept": {
            "value": float(intercept),
            "trainable": True,
            "bounds": [None, None],
            "unit": target_unit,
            "origin": "native_regression_solve",
        }
    }
    root = {"param": "intercept"}
    for index, (name, value) in enumerate(zip(names, coefficients, strict=True)):
        descriptor = parse_expression(name.replace("ln(", "log("), variables, trainable=False)
        key = f"c{index}"
        parameters[key] = {
            "value": float(value),
            "trainable": True,
            "origin": "native_regression_solve",
            "bounds": [None, None],
            "unit": "1" if target_unit is not None else None,
        }
        root = {"op": "add", "args": [root, {"op": "mul", "args": [{"param": key}, descriptor["ast"]]}]}
    equation = {"schema_version": 1, "ast": root, "parameters": parameters}
    inspect_equation(equation, variables)
    return equation


def dimensional_features(frame, operators, expansion, device, variables, target):
    """Bridge known unit vectors to the native builder's existing SymPy unit field.

    Version0.1.8 calls symbols(strings), which makes 'm/s' an atomic label rather
    than m/s. Supply its intended internal products BEFORE native expansion; no
    feature/optimizer formula is substituted. Original values/labels are unchanged.
    """
    import sympy as sp
    from TorchSisso.DimensionalFeatureSpaceConstruction import feature_space_construction

    definitions = [*variables, target]
    for definition in definitions:
        require_canonical(definition.get("unit"))
    if any(unit_dimensions(v.get("unit")) is None for v in definitions):
        raise ValueError("TorchSISSO dimensional mode requires known feature and target unit labels")
    bases = sp.symbols("L M T K", positive=True)

    def unit(definition):
        vector = unit_dimensions(definition["unit"])
        return (
            sp.prod(base ** sp.Rational(power) for base, power in zip(bases, vector, strict=True))
            if any(vector)
            else sp.Symbol("1")
        )

    dimensions = [unit(v) for v in variables]
    builder = feature_space_construction(
        frame.copy(),
        operators=operators,
        no_of_operators=expansion,
        device=device,
        dimensionality=[str(d) for d in dimensions],
    )
    builder.dimensionality = dimensions
    return (*builder.feature_expansion(), unit(target))


def dimensional_models(regressor):
    """Observe actual native solves; retain full coefficients without display parsing."""
    module = import_module("TorchSisso.Regressor_dimension")
    if (
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        != "5ef9adf0069bd525cfd3b86eb8068560ed70832e81b0d0d3b7bddf87596202e8"
    ):
        raise ValueError("Dimensional coefficient observation requires the reviewed TorchSisso0.1.8 source")
    one, higher = [], []
    outer_code = type(regressor).regressor_fit.__code__
    higher_code = type(regressor).higher_dimension.__code__

    def first(values):
        index = int(values["selected_index"])
        one.append(
            {
                "indices": [index],
                "coefficients": [(values["coef"][1] / regressor.x_std[index]).item()],
                "intercept": float(values["intercept"]),
                "term_count": 1,
            }
        )

    def observe(frame, event, arg):
        if frame.f_code is higher_code:
            if event == "call" and not one:
                first(frame.f_back.f_locals)  # Before the outer intercept is overwritten.
            elif event == "return" and arg is not None:
                values = frame.f_locals
                higher.append(
                    {
                        "indices": values["indices_min"].reshape(-1).detach().cpu().tolist(),
                        "coefficients": values["non_std_coeff"].reshape(-1).detach().cpu().tolist(),
                        "intercept": float(values["non_std_intercept"]),
                        "term_count": int(values["iteration"]),
                    }
                )
        elif frame.f_code is outer_code and event == "return" and arg is not None and not one:
            first(frame.f_locals)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        regressor.regressor_fit()
    finally:
        sys.setprofile(previous)
    models = [*one, *higher]
    if not models or any(not np.isfinite([*m["coefficients"], m["intercept"]]).all() for m in models):
        raise ValueError("Native dimensional regression did not return finite full-precision models")
    return models


def usable_descriptors(expanded, names):
    """Native standardization is undefined for Train-constant generated descriptors.

    A saturated tanh(T) can be exactly1 even when T varies. Remove only columns
    whose actual native standardization is undefined; do not change rows,
    substitute features, refit coefficients, or infer global constancy.
    """
    import torch

    deviation = expanded.std(dim=0)
    keep = torch.isfinite(expanded).all(dim=0) & torch.isfinite(deviation) & (deviation > 0)
    indices = keep.nonzero(as_tuple=True)[0].detach().cpu().tolist()
    removed = [name for index, name in enumerate(names) if not bool(keep[index])]
    if not indices:
        raise ValueError("No generated descriptor has a finite nonzero native Train standard deviation")
    return expanded[:, indices], [names[index] for index in indices], indices, removed


def prepare_response_search(request, *, allow_template=False):
    from ..response_search import (response_enforcement, requested_response_constraints,
                                   compile_response_constraints, response_bounds)
    from ..domains import region_mask, validate_input_domain
    response_enforcement(request.limits, capabilities())
    if request.limits.get("response_mode", "search") != "search" or not requested_response_constraints(request.limits):
        return None
    if (request.equation_template is not None or request.fixed_constants) and not allow_template:
        raise NotImplementedError("TorchSISSO response search supports affine descriptor output only, not fixed outer/template expressions")
    if request.domain is not None and request.domain["columns"] != [v["column"] for v in request.variables]:
        raise ValueError("TorchSISSO response domain column order differs")
    validate_input_domain(request.X, request.variables)
    if not region_mask(request.X, request.domain).all():
        raise ValueError("TorchSISSO Train outside declared joint domain")
    compiled = compile_response_constraints(request)
    if not compiled["terms"]:
        raise ValueError("TorchSISSO response search needs bounded conditions")
    if not region_mask(np.asarray(compiled["points"]), request.domain).all():
        raise ValueError("TorchSISSO response stencil outside declared joint domain")
    lower, upper = response_bounds(compiled)
    if not np.any(np.isfinite(lower) | np.isfinite(upper)):
        raise NotImplementedError("Unbounded validity probes alone are not constrained coefficient search")
    return compiled


def validate_request(request, context):
    """Check options and Train variance without importing TorchSISSO."""
    native_input_indices(request, context)
    from .torchsisso_template import prepare
    template = prepare(request, context)
    validate_options(request.options, SUPPORTED_OPTIONS)
    compiled = prepare_response_search(request, allow_template=template is not None)
    if template:
        template.validate_response(compiled)
    iterations = request.options.get("response_max_iterations", 200)
    if type(iterations) is not int or iterations < 1:
        raise ValueError("TorchSISSO response_max_iterations must be a positive integer")
    dimensional = request.options.get("dimensional", False)
    if type(dimensional) is not bool:
        raise ValueError("TorchSISSO dimensional must be true or false")
    feature_module = (
        "TorchSisso.DimensionalFeatureSpaceConstruction" if dimensional else "TorchSisso.FeatureSpaceConstruction"
    )
    if any(
        type(request.options.get(name, default)) is not int or request.options.get(name, default) < 1
        for name, default in [("n_expansion", 2), ("n_term", 3), ("k", 10)]
    ):
        raise ValueError("TorchSISSO expansion/term/SIS counts must be positive integers; outer resource limits apply")
    variance_X = template.X if template else request.X[:, native_input_indices(request, context)]
    variance_y = template.y if template else request.y
    if np.any(np.ptp(variance_X, axis=0) == 0) or np.ptp(variance_y) == 0:
        raise ValueError(
            "TorchSISSO zero-variance columns/target must be excluded explicitly; use mean for constant target"
        )
    return dimensional, feature_module


def discover(request, context):
    dimensional, feature_module = validate_request(request, context)
    feature_indices = native_input_indices(request, context)
    from .torchsisso_template import prepare
    template = prepare(request, context)
    native_variables = template.variables if template else [request.variables[i] for i in feature_indices]
    native_target = template.target if template else request.target
    native_X = template.X if template else request.X[:, feature_indices]
    native_y = template.y if template else request.y
    search_operators = template.operators if template else request.operators
    import torch
    from TorchSisso.FeatureSpaceConstruction import feature_space_construction
    from TorchSisso.Regressor import Regressor

    native_module = import_module("TorchSisso.Regressor")

    if dimensional:
        from TorchSisso.Regressor_dimension import Regressor

        native_module = import_module("TorchSisso.Regressor_dimension")
    if version("TorchSisso") != "0.1.8" or (
        not dimensional and getattr(Regressor, "symbolic_handoff_precision_version", None) != 1
    ):
        raise ValueError("Requires TorchSisso 0.1.8 with scripts/patch_torchsisso_precision.py instrumentation")
    if getattr(Regressor, "symbolic_handoff_budget_version", None) != 1:
        raise ValueError("Rebuild classic runtime with the pinned TorchSISSO source AST budget extension")
    expected_native = ("5ef9adf0069bd525cfd3b86eb8068560ed70832e81b0d0d3b7bddf87596202e8" if dimensional else
                       "cbd1c8d0353231ad0ae74f9b7b8622df1fbbb59fa72ebadc01e9194c86544713")
    if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest() != expected_native:
        raise ValueError("Unreviewed TorchSISSO regression source; rebuild pinned source-budget runtime")
    from .torchsisso_budget import NodeBudget
    from ..coordinates import source_complexity_costs, source_equation
    coordinate_context = {**context.provenance, "variables": request.variables,
                          "target": {"column": "target", **request.target}}
    variable_costs, output_cost = source_complexity_costs(coordinate_context)
    def descriptor_cost(name):
        equation = parse_expression(name.replace("ln(", "log("), request.variables, trainable=False)
        def count(node):
            return variable_costs[node["var"]] if "var" in node else 1 + sum(count(c) for c in node.get("args", []))
        return count(equation["ast"])
    budget = template.budget() if template else NodeBudget(request.limits["max_nodes"] - output_cost, descriptor_cost)
    torch.manual_seed(request.seed)
    torch.set_num_threads(1)
    np.random.seed(request.seed)
    native_operators = [NATIVE_NAMES[op] for op in search_operators if op in NATIVE_NAMES]
    if "pow_int" in search_operators:
        native_operators.extend(["pow(2)", "pow(3)", "pow(4)"])
    frame = pd.DataFrame(
        np.column_stack([native_y, native_X]), columns=["target", *[v["id"] for v in native_variables]]
    )
    compiled = prepare_response_search(request, allow_template=template is not None)
    response = None
    if compiled is not None:
        from .torchsisso_response import ResponseSearch
        from .torchsisso_oracle import descriptor_values
        response = (template.response(compiled, context.device, request.options.get("response_max_iterations", 200))
            if template else ResponseSearch(compiled,
            lambda names: descriptor_values(names, np.asarray(compiled["points"]), request.variables, device=context.device).detach().cpu().numpy(),
            request.options.get("response_max_iterations", 200)))
    def fit_native(regressor):
        try:
            regressor.regressor_fit()
        finally:
            if response is not None:
                write_json(context.work_dir / "torchsisso_response.json", {
                    "compiled": compiled, "reduced_compiled": response.compiled if template else None, "subset_counts": dict(response.counts),
                    "failure_examples": response.failure_examples,
                    "models": response.models,
                    "scope": "finite-point hard constrained-SIS; all top-k singleton fits, native multivariate pools; not exhaustive structural search"})
        return regressor.portable_models_
    expansion = request.options.get("n_expansion", 2)
    if dimensional:
        expanded, y, names, unit_vectors, target_dimension = dimensional_features(
            frame, native_operators, expansion, context.device, native_variables, native_target
        )
        expanded, names, kept, removed = usable_descriptors(expanded, names)
        unit_vectors = [unit_vectors[index] for index in kept]
        if target_dimension not in unit_vectors:
            raise ValueError("Native dimensional expansion contains no descriptor with the target dimension")
        regressor = Regressor(
            expanded,
            y,
            names,
            unit_vectors,
            dimension=request.options.get("n_term", 3),
            sis_features=request.options.get("k", 10),
            device=context.device,
            output_dim=target_dimension,
            node_budget=budget, response_search=response,
        )
        models = fit_native(regressor) if response is not None else dimensional_models(regressor)
        expanded, names = regressor.x, regressor.names
    else:
        expanded, y, names = feature_space_construction(
            native_operators, frame.copy(), expansion, context.device
        ).feature_space()
        expanded, names, _, removed = usable_descriptors(expanded, names)
        regressor = Regressor(
            expanded, y, names, request.options.get("n_term", 3), request.options.get("k", 10), context.device, False, node_budget=budget, response_search=response
        )
        models = fit_native(regressor)
        expanded, names = regressor.x, regressor.names
    # Reconstruct native descriptors without fitting or held-out targets.
    # Dimensional mode retains its reviewed full constructor; nondimensional
    # mode replays selected native arithmetic, preserving nonfinite probe rows.
    extra = region_points(request.X, request.variables, request.seed, request.domain, request.limits.get("probes", []))
    oracle_X = np.vstack([request.X, extra])
    native_oracle_X = template.transform(oracle_X) if template else oracle_X[:, feature_indices]
    oracle_frame = pd.DataFrame(native_oracle_X, columns=[v["id"] for v in native_variables])
    oracle_frame.insert(0, "target", np.arange(len(oracle_X), dtype=float))
    if dimensional:
        probe_features, _, probe_names, _, _ = dimensional_features(
            oracle_frame, native_operators, expansion, context.device, native_variables, native_target
        )
        probe_columns = {name: i for i, name in reversed(list(enumerate(probe_names)))}
    candidates, rejected = [], []
    for index, model in enumerate(models):
        try:
            columns = [int(i) for i in model["indices"]]
            equation = _equation(
                [names[i] for i in columns],
                model["coefficients"],
                model["intercept"],
                native_variables if template else request.variables,
                native_target["unit"] if dimensional else None,
            )
            if template:
                equation = template.export(equation)
            if not dimensional:
                equation["schema_version"] = 2
                equation["arithmetic"] = "numpy_ieee"
            selected_names = [names[i] for i in columns]
            if dimensional:
                if any(name not in probe_columns for name in selected_names):
                    raise ValueError("Native dimensional descriptor dropped on independent domain probes")
                selected = probe_features[:, [probe_columns[name] for name in selected_names]]
            else:
                from .torchsisso_oracle import descriptor_values

                selected = descriptor_values(selected_names, oracle_X, request.variables, device=context.device)
            if not torch.allclose(selected[: len(request.X)], expanded[:, columns], rtol=1e-12, atol=1e-12):
                raise ValueError("Fresh native descriptor construction changed retained training feature values")
            native = (
                (
                    selected @ torch.as_tensor(model["coefficients"], dtype=expanded.dtype, device=expanded.device)
                    + model["intercept"]
                )
                .detach()
                .cpu()
                .numpy()
            )

            if template:
                native = template.restore(native, oracle_X)

            def native_predict(X, prediction=native, descriptors=selected_names, saved=model):
                if not dimensional:
                    fresh = descriptor_values(descriptors, X, request.variables, device=context.device)
                    coefficients = torch.as_tensor(saved["coefficients"], dtype=fresh.dtype, device=fresh.device)
                    return (fresh @ coefficients + saved["intercept"]).detach().cpu().numpy()
                rows = [
                    np.flatnonzero(np.all((oracle_X == row) & (np.signbit(oracle_X) == np.signbit(row)), axis=1))[0]
                    for row in X
                ]
                return prediction[rows]

            public_equation, public_variables = source_equation(equation, coordinate_context)
            if inspect_equation(public_equation, public_variables)["n_nodes"] > request.limits["max_nodes"]:
                raise RuntimeError("TorchSISSO affine fit violated source-coordinate search budget")
            candidate = checked_candidate("torchsisso", index, deepcopy(equation), request, native_predict)
            if response is not None:
                from ..expressions import evaluate
                from ..response_search import response_feasible
                if not response_feasible(evaluate(equation, np.asarray(compiled["points"]), request.variables), compiled):
                    raise ValueError("Portable constrained-SIS output failed original finite-point conditions")
                candidate.native_check.update(finite_point_feasible=True, solver_converged=model["optimizer_converged"],
                    response_stage="constrained-SIS subset coefficient optimization", solver=model["solver"])
            candidates.append(candidate)
        except (ValueError, TypeError) as exc:
            rejected.append({"index": index, "reason": str(exc)})
    raw = write_json(
        context.work_dir / "torchsisso_native.json",
        {
            "models": [{**model, "descriptors": [names[int(i)] for i in model["indices"]]} for model in models],
            "native_dtype": str(expanded.dtype),
            "validation_probe_count": len(extra),
            "validation_probe_scope": "declared input geometry; observed geometry reuses Train rows, not independent data",
            "oracle": "native dimensional constructor"
            if dimensional
            else "pinned native descriptor arithmetic; validation-only column filtering bypass; no search/refit",
            "rejected_exports": rejected,
            "removed_degenerate_train_descriptors": removed,
            "dimensional": dimensional,
            "input_units": {v["id"]: v["unit"] for v in native_variables},
            "target_dimension": str(target_dimension) if dimensional else None,
            "coefficient_units": "dimensionless native target-dimension descriptor weights; intercept has target unit"
            if dimensional
            else "unknown",
        },
    )
    return DiscoveryResult(
        candidates,
        "ok" if candidates else "failed",
        {
            "version": version("TorchSisso"),
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "native_source_sha256": hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest(),
            "native_feature_source_sha256": hashlib.sha256(
                Path(import_module(feature_module).__file__).read_bytes()
            ).hexdigest(),
            "native_candidates": len(models),
            "equation_template": None if template is None else {
                "scope": capabilities()["template_scope"], "slot": template.name,
                "shared_calls": len(template.calls), "fixed_scale": template.scale,
                "loss_equivalence": "mathematically original y SSE = fixed_scale**2 * native reduced-target SSE; no weights changed; finite arithmetic is checked, not exact arithmetic",
                "reduction_check": template.reduction_check,
                "reduced_resolution_check": template.reduced_resolution_check,
                "objective_scale": template.loss_scale,
                "native_stages": "dimensional descriptor expansion, SIS screening, sparse linear solve on reduced Train target",
                "constraints": "slot and fully expanded source AST budgets before subset solve; original-y finite-point feasibility in constrained-SIS",
            },
            "response_search": None if response is None else {
                "subset_counts": dict(response.counts), "failure_examples": response.failure_examples,
                "lineage": "TorchSISSO 0.1.8 + constrained-SIS extension",
                "scope": "declared finite points; SIS screening remains heuristic, all screened singleton candidates fit; feasible residual enters next SIS",
                "max_iterations_per_solver": response.max_iterations},
            "source_complexity_budget": {"max_nodes": request.limits["max_nodes"], "variable_costs": variable_costs,
                                         "output_cost": output_cost, "combination_events": regressor.node_budget_events,
                                         "scope": "descriptor and affine-combination filtering before least squares; intermediate feature expansion still unconstrained"},
            "dimensional": dimensional,
            "target_dimension": str(target_dimension) if dimensional else None,
            "public_output_unit": request.target["unit"],
            "coefficient_units": "1; intercept=" + native_target["unit"] if dimensional else "unknown",
            "expanded_features": len(names),
            "removed_degenerate_train_descriptors": removed,
            "native_search_operators": native_operators,
            "native_dtype": str(expanded.dtype),
            "input_rows": len(request.X),
            "native_input_variables": [v["id"] for v in native_variables],
            "native_input_columns": [v["column"] for v in native_variables],
            "export_precision": "observed native solve indices, raw-scale full coefficients/intercept; no rounded display parsing"
            if dimensional
            else "instrumented native indices, raw-scale full coefficients/intercept; no rounded display parsing",
            "native_check_scope": "native selected descriptors on Train and declared-domain validation points; raw IEEE masks retained, common real validity separate",
            "validation_probe_count": len(extra),
            "validation_probe_scope": "declared input geometry; observed geometry reuses Train rows, not independent data",
            "rejected_exports": rejected,
            "constraint_support": capabilities()["constraints"],
        },
        {"native_models": str(raw), **({"response_search": str(context.work_dir / "torchsisso_response.json")} if response is not None else {})},
    )
