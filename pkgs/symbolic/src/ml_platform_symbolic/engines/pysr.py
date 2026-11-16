"""Native PySR search and operator restrictions; common selection stays outside."""

import ast
import math
import time
from importlib.metadata import version

from ..data import write_json
from ..expressions import parse_expression, inspect_equation
from ..coordinates import source_complexity_costs, source_equation
from .base import DiscoveryResult, checked_candidate, validate_options

OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "neg": "neg",
    "sqrt": "sqrt",
    "log": "log",
    "exp": "exp",
    "abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "pow": "^",
    "pow_int": ("square", "cube"),
}
BINARY = {"add", "sub", "mul", "div", "pow"}


def export_expression(text, variables):
    """PySR ConstantNode values are fitted, including powers; square/cube are operators."""
    parameters = {}

    class NativeConstants(ast.NodeTransformer):
        def constant(self, value):
            key = f"c{len(parameters)}"
            parameters[key] = {
                "value": float(value),
                "trainable": True,
                "bounds": [None, None],
                "unit": None,
                "origin": "pysr_native_constant_node",
            }
            return ast.Name(id=key, ctx=ast.Load())

        def visit_Constant(self, node):
            if type(node.value) not in {int, float}:
                raise ValueError("Unsupported native constant")
            return ast.copy_location(self.constant(node.value), node)

        def visit_UnaryOp(self, node):
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                return ast.copy_location(self.constant(-node.operand.value), node)
            return self.generic_visit(node)

    class IntegerPowers(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in {"square", "cube"}:
                if len(node.args) != 1 or node.keywords:
                    raise ValueError("Malformed native integer-power operator")
                return ast.BinOp(node.args[0], ast.Pow(), ast.Constant(2 if node.func.id == "square" else 3))
            return node

    tree = NativeConstants().visit(ast.parse(text.replace("^", "**"), mode="eval"))
    tree = IntegerPowers().visit(tree)  # Structural exponents are inserted AFTER fitted leaves.
    return parse_expression(ast.unparse(ast.fix_missing_locations(tree)), variables, parameters=parameters)


def native_constraints(options, requested):
    """Canonical task operator names map to explicit native profile restrictions."""

    def names(op):
        if op not in requested:
            raise ValueError(f"PySR native restriction refers to an inactive operator: {op}")
        return OPERATORS[op] if isinstance(OPERATORS[op], tuple) else (OPERATORS[op],)

    arguments, nested = {}, {}
    for op, limit in options.get("operator_constraints", {}).items():
        arity = 2 if op in BINARY else 1
        limits = limit if isinstance(limit, (list, tuple)) else [limit]
        if len(limits) != arity or any(type(v) is not int or v == 0 or v < -1 for v in limits):
            raise ValueError(
                "Native operator argument limits are positive integers or -1; binary limits need two values"
            )
        for name in names(op):
            arguments[name] = tuple(limits) if arity == 2 else limits[0]
    for outer, inner in options.get("nested_constraints", {}).items():
        for native_outer in names(outer):
            nested[native_outer] = {}
            for op, count in inner.items():
                if type(count) is not int or count < 0:
                    raise ValueError("Native nesting counts must be nonnegative integers")
                for name in names(op):
                    nested[native_outer][name] = count
    if "pow" in requested:
        arguments.setdefault("^", (-1, 1))
    return arguments, nested


SUPPORTED_OPTIONS = {
    "unit_mode",
    "dimensional_constraint_penalty",
    "dimensionless_constants_only",
    "parallelism",
    "deterministic",
    "procs",
    "constraint_weight",
    "objective",
    "max_evals",
    "maxdepth",
    "ncycles_per_iteration",
    "nested_constraints",
    "niterations",
    "operator_constraints",
    "population_size",
    "populations",
    "seconds",
    "template_parameter_initialization",
}


def native_feature_indices(request):
    """Restrict native inputs, preserving the full common schema for validation/export."""
    allowed = request.limits.get("allowed_variables")
    indices = [i for i, variable in enumerate(request.variables)
               if allowed is None or variable["id"] in allowed]
    if not indices:
        raise NotImplementedError("PySR variable-constrained search needs at least one allowed input")
    if len(indices) != len(request.variables):
        if request.equation_template is not None:
            raise NotImplementedError(
                "PySR allowed_variables with equation templates is not yet supported; "
                "no unconstrained or post-filter-only fallback"
            )
    return indices


def execution_options(options):
    """Keep serial reproducibility by default; parallel search is explicit."""
    parallelism = options.get("parallelism", "serial")
    if not isinstance(parallelism, str) or parallelism not in {"serial", "multithreading", "multiprocessing"}:
        raise ValueError("PySR parallelism must be serial, multithreading or multiprocessing")
    deterministic = options.get("deterministic", parallelism == "serial")
    if type(deterministic) is not bool or (deterministic and parallelism != "serial"):
        raise ValueError("PySR deterministic must be boolean and true requires serial execution")
    result = {"parallelism": parallelism, "deterministic": deterministic}
    if "procs" in options:
        procs = options["procs"]
        if parallelism != "multiprocessing" or (procs is not None and (type(procs) is not int or procs <= 0)):
            raise ValueError("PySR procs requires multiprocessing and a positive integer or null")
        result["procs"] = procs
    return result


def unit_profile(request):
    """Native unit search is explicit; units themselves come only from the common schema."""
    mode = request.options.get("unit_mode")
    if mode is not None and (not isinstance(mode, str) or mode not in {"soft", "hard"}):
        raise ValueError("PySR unit_mode must be soft, hard or null")
    keys = {"dimensional_constraint_penalty", "dimensionless_constants_only"}
    if mode is None:
        if keys & request.options.keys():
            raise ValueError("PySR unit options require explicit unit_mode")
        return {}, {}, {"mode": "post_check_only", "native_units": False}
    if request.equation_template is not None:
        raise NotImplementedError("Pinned PySR native unit search does not support equation templates; omit unit_mode for the explicitly post-checked legacy template pathway")
    dimensionless = request.options.get("dimensionless_constants_only", False)
    if type(dimensionless) is not bool:
        raise ValueError("dimensionless_constants_only must be boolean")
    if mode == "hard" and "dimensional_constraint_penalty" in request.options:
        raise ValueError("dimensional_constraint_penalty applies only to soft unit_mode")
    penalty = request.options.get("dimensional_constraint_penalty", 1000.0)
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("dimensional_constraint_penalty must be positive and finite")
    from ..units import require_canonical, unit_dimensions

    def native_unit(definition):
        label = definition.get("unit")
        require_canonical(label)
        vector = unit_dimensions(label)
        if vector is None:
            raise ValueError("PySR native unit search requires known canonical input and target units; unknown is not dimensionless")
        terms = [f"{base}^({power.numerator}/{power.denominator})"
                 for base, power in zip(("m", "kg", "s", "K"), vector, strict=True) if power]
        return " * ".join(terms) or "1"

    indices = native_feature_indices(request)
    inputs = [native_unit(request.variables[i]) for i in indices]
    output = native_unit(request.target)
    model = {"dimensional_constraint_penalty": float("inf") if mode == "hard" else float(penalty),
             "dimensionless_constants_only": dimensionless}
    fit = {"X_units": inputs, "y_units": output}
    evidence = {"mode": mode, "native_units": True, "native_variable_ids": [request.variables[i]["id"] for i in indices],
                "X_units": inputs, "y_units": output, "dimensionless_constants_only": dimensionless,
                "penalty": float(penalty) if mode == "soft" else None,
                "stage": "native structure fitness and coefficient optimization, including custom response loss",
                "constant_semantics": "dimensionless" if dimensionless else "native wildcard units; coefficient dimensions exported only when uniquely implied by coherent basic-unit AST constraints; not physical parameter identification",
                "scope": "dimensional consistency under native constant assumptions; common structural checks retained; not a physical-law guarantee"}
    return model, fit, evidence


def capabilities():
    return {
        "unit_search": {'defaults': {'unit_mode': 'hard'}, 'choices': {'unit_mode': ['soft', 'hard']}, 'templates': False},
        "response_enforcement": {"default": "soft", "supported": ['soft', 'hard'], "stage": "native structure fitness and constant optimization"},
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": list(OPERATORS),
        "equation_template": True,
        "fixed_constants": "template_only",
        "constraints": {
            "operators": "enforced during search",
            "max_nodes": "weighted native source-coordinate size for ordinary expressions; template full-size fitness gate includes fixed skeleton and repeated calls",
            "native_operator_profile": "optional per-argument/nesting restrictions enforced by PySR; not additional common output requirements",
            "units": "default post-check only; explicit unit_mode soft/hard applies native dimensional fitness to ordinary expressions, including custom response loss; template native-unit search unsupported",
            "variables": "disallowed input columns excluded before native search; response points projected after full-schema validation; subset with templates unsupported; full-schema export rechecked",
            "allowed_variables": "disallowed input columns excluded before native search; response points projected after full-schema validation; subset with templates unsupported; full-schema export rechecked",
            "probes": "native finite-point objective in search mode; explicit filter mode is post-check only",
            "response_checks": "native finite-step objective in search mode; no global guarantee",
            "equation_template": "native fixed structure/shared slots; full source AST and local slot size hard-gated in search fitness before prediction; units and exported expression rechecked after search",
        },
        "search_constraints": ["response_bounds", "response_checks", "probes"],
        "requires_assets": False,
    }


def validate_request(request, context):
    """Lower options and templates without loading PySR or Julia."""
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    execution_options(request.options)
    unit_profile(request)
    native_feature_indices(request)
    unsupported = set(request.operators) - set(OPERATORS)
    if unsupported:
        raise ValueError(f"PySR adapter does not support: {unsupported}")
    binary = [OPERATORS[op] for op in request.operators if op in BINARY]
    unary = [
        name
        for op in request.operators
        if op not in BINARY
        for name in (OPERATORS[op] if isinstance(OPERATORS[op], tuple) else (OPERATORS[op],))
    ]
    arguments, nested = native_constraints(request.options, request.operators)
    depth = request.options.get("maxdepth", 20)
    seconds = request.options.get("seconds", request.limits["max_wall_seconds"] * 0.8)
    if type(depth) is not int or not 1 <= depth <= 20 or (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float))
        or not 0 < seconds < request.limits["max_wall_seconds"]
    ):
        raise ValueError("PySR maxdepth must be1..20 and native seconds positive and below the hard worker cap")
    from .pysr_objective import native_response_objective

    template = getattr(request, "equation_template", None)
    if getattr(request, "fixed_constants", {}) and template is None:
        raise NotImplementedError("PySR required fixed constants are supported only within an equation_template")
    template_mapping, expression_options = None, {}
    if "template_parameter_initialization" in request.options:
        if template is None or request.options["template_parameter_initialization"] != "native":
            raise ValueError("template_parameter_initialization only accepts 'native' with equation_template")
    if template is not None:
        from .pysr_template import template_specification

        spec, template_mapping = template_specification(
            template,
            request.variables,
            request.operators,
            parameter_initialization=request.options.get("template_parameter_initialization"),
        )
    native_response_objective(request)  # Validate after template support checks, before Julia.
    for key in ('niterations', 'populations', 'population_size', 'ncycles_per_iteration', 'max_evals'):
        if key == 'max_evals' and request.options.get(key) is None:
            continue
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"pysr {key} must be a positive integer")
    return binary, unary, arguments, nested, depth, seconds, template, template_mapping, expression_options, (
        spec if template is not None else None
    )


def source_search_budget(request, context):
    """Native weighted size is the source AST size minus a fixed output map."""
    schema = {**context.provenance, "variables": request.variables,
              "target": {"column": "target", **request.target}}
    variable_costs, output_cost = source_complexity_costs(schema)
    limit = request.limits["max_nodes"]
    if type(limit) is not int or limit - output_cost < 7:
        raise ValueError("PySR requires at least 7 source nodes after the fixed output-coordinate overhead")
    weights = [variable_costs[request.variables[i]["id"]] for i in native_feature_indices(request)]
    return schema, limit - output_cost, weights, output_cost


def discover(request, context, *, recover=False):
    started = time.monotonic()
    binary, unary, arguments, nested, depth, seconds, template, template_mapping, expression_options, spec = (
        validate_request(request, context)
    )
    if recover and template is not None:
        raise NotImplementedError("PySR template recovery needs Julia state; CSV-only recovery is unsupported")
    if template is not None and version("pysr") != "2.7.2":
        raise NotImplementedError("PySR native template extraction is verified only for pinned version 2.7.2")

    coordinate_context, native_maxsize, variable_costs, output_cost = source_search_budget(request, context)

    from pysr import PySRRegressor

    if template is not None:
        from pysr import TemplateExpressionSpec

        expression_options["expression_spec"] = TemplateExpressionSpec(**spec)
    from .pysr_objective import native_response_objective

    source_weights, _ = source_complexity_costs(coordinate_context)
    response_options, response_evidence = native_response_objective(request, variable_costs=source_weights, output_cost=output_cost)
    expression_options.update(response_options)
    imported = time.monotonic()
    search_limits = {
        "niterations": request.options.get("niterations", 100),
        "populations": request.options.get("populations", 8),
        "population_size": request.options.get("population_size", 40),
        "ncycles_per_iteration": request.options.get("ncycles_per_iteration", 380),
        "max_evals": request.options.get("max_evals"),
        "timeout_in_seconds": seconds,
    }
    execution = execution_options(request.options)
    unit_options, unit_fit, unit_evidence = unit_profile(request)
    from .pysr_recovery import begin_run, recovered_run
    recovery = None
    run_id = None if recover else begin_run(context.work_dir)
    model = PySRRegressor(
        run_id=run_id,
        **unit_options,
        **execution,
        **expression_options,
        **search_limits,
        binary_operators=binary,
        unary_operators=unary,
        maxsize=native_maxsize,
        complexity_of_operators={name: (2 if name in {"square", "cube"} else 1) for name in binary + unary},
        complexity_of_constants=1,
        maxdepth=depth,
        constraints=arguments or None,
        nested_constraints=nested or None,
        random_state=request.seed,
        precision=64,
        progress=False,
        verbosity=0,
        output_directory=str(context.work_dir / "native"),
    )
    fit_started = time.monotonic()
    feature_indices = native_feature_indices(request)
    if recover:
        snapshot, recovery = recovered_run(context.work_dir)
        model = PySRRegressor.from_file(
            run_directory=str(snapshot), binary_operators=binary, unary_operators=unary,
            n_features_in=len(feature_indices),
            feature_names_in=[request.variables[i]["id"] for i in feature_indices], precision=64,
        )
    else:
        model.fit(request.X[:, feature_indices], request.y, **unit_fit,
                  variable_names=[request.variables[i]["id"] for i in feature_indices],
                  complexity_of_variables=variable_costs if template is None else [1] * len(feature_indices))
    fit_finished = time.monotonic()
    raw = write_json(
        context.work_dir / "pysr_native.json",
        {
            "equations": [
                {
                    "index": int(index),
                    "equation": row["equation"],
                    "loss": float(row["loss"]) if math.isfinite(float(row["loss"])) else None,
                    "loss_status": "finite" if math.isfinite(float(row["loss"])) else str(float(row["loss"])),
                    "complexity": int(row["complexity"]),
                }
                for index, row in model.equations_.iterrows()
            ],
            "operator_constraints": arguments,
            "nested_constraints": nested,
        },
    )
    candidates, rejected, template_evidence = [], [], {}
    coefficient_unit_evidence = {}
    for index, row in model.equations_.iterrows():
        if not math.isfinite(float(row["loss"])):
            rejected.append({"index": int(index), "reason": "Native objective is nonfinite; infeasible or invalid candidate", "native_loss": str(float(row["loss"]))})
            continue
        try:
            if template is None:
                equation = export_expression(row["equation"], request.variables)
                if unit_evidence.get("dimensionless_constants_only"):
                    for parameter in equation["parameters"].values():
                        parameter["unit"] = "1"
                elif unit_evidence.get("native_units"):
                    from .pysr_units import infer_coefficient_units
                    evidence = infer_coefficient_units(equation, request.variables, request.target)
                    coefficient_unit_evidence[str(index)] = evidence
                    if evidence["status"] == "unique":
                        for name, label in evidence["units"].items():
                            equation["parameters"][name].update(unit=label, unit_origin=evidence["basis"])
            else:
                from .pysr_template import export_template

                equation, evidence = export_template(
                    row["julia_expression"],
                    template,
                    template_mapping,
                    {1: unary, 2: binary},
                    variables=request.variables,
                    target=request.target,
                )
                template_evidence[str(index)] = evidence
                failures = [
                    check["name"] for check in evidence["checks"] if check.get("required") and check["status"] != "pass"
                ]
                if failures:
                    raise ValueError("Template candidate does not satisfy required checks: " + ", ".join(failures))
            public_equation, public_variables = source_equation(equation, coordinate_context)
            source_nodes = inspect_equation(public_equation, public_variables)["n_nodes"]
            if source_nodes > request.limits["max_nodes"]:
                raise RuntimeError("PySR native search violated its source-coordinate node budget")
            candidates.append(
                checked_candidate("pysr", index, equation, request,
                                  lambda X, i=index: model.predict(X[:, feature_indices], index=i))
            )
        except (ValueError, TypeError) as exc:
            rejected.append({"index": int(index), "reason": str(exc)})
    return DiscoveryResult(
        candidates,
        ("partial" if recover else "ok") if candidates else "failed",
        {
            "native_recovery": recovery,
            "version": version("pysr"),
            "juliacall": version("juliacall"),
            "native_search_operators": {"binary": binary, "unary": unary},
            "native_operator_constraints": arguments,
            "native_nested_constraints": nested,
            "native_search_seconds_limit": seconds,
            "native_search_limits": search_limits,
            "source_complexity_budget": {"max_nodes": request.limits["max_nodes"], "native_maxsize": native_maxsize,
                                         "variable_costs": variable_costs, "output_cost": output_cost,
                                         "square_cube_cost": 2},
            "native_execution_options": execution,
            "unit_search": unit_evidence,
            "coefficient_unit_inference": coefficient_unit_evidence,
            "native_input_variables": [request.variables[i]["id"] for i in feature_indices],
            "native_input_columns": [request.variables[i]["column"] for i in feature_indices],
            "native_work_observation": ("Recovered CSV hypotheses after stopped worker; no search performed during recovery and original progress/stop counts unknown." if recover else "Configured ceilings and measured fit time only; actual mutation/evaluation counts and stop criterion are not reported by this adapter. A returned fit is not proof of convergence."),
            "fit_elapsed_seconds": None if recover else time.monotonic() - started,
            "recovery_elapsed_seconds": time.monotonic() - started if recover else None,
            "native_import_seconds": imported - started,
            "native_search_fit_seconds": None if recover else fit_finished - fit_started,
            "native_export_seconds": time.monotonic() - fit_finished,
            "native_candidates": len(model.equations_),
            "rejected_exports": rejected,
            "template_evidence": template_evidence,
            "template_native_support": {
                "outer_structure": "enforced during native search",
                "shared_slots": "one native slot identity across all calls",
                "slot_operators": "identical per-slot operator sets enforced natively; differing sets rejected",
                "slot_max_nodes": "native full-expression loss gate counts each portable slot AST including power exponent literals; export rechecks",
                "outer_parameter_initialization": "native initializes free values only with explicit user option",
                "outer_parameter_bounds": "finite free bounds rejected before native search",
                "export": "actual Julia slot Node values and parameter vectors; no equation string parsing",
            }
            if template is not None
            else None,
            "constraint_support": capabilities()["constraints"],
            "response_search": response_evidence,
            "native_check_source": (
                "PySRRegressor.predict(index=...) versus composed full-precision native slot Nodes and parameters"
                if template is not None
                else "PySRRegressor.predict(index=...) versus unsimplified equation AST"
            ),
        },
        {"native_models": raw},
    )
