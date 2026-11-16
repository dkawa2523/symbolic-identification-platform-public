"""PhySO adapter using raw prefix tokens, preserving shared constants and domains."""

import time
from copy import deepcopy
from importlib.metadata import version

import numpy as np

from ..data import write_json
from ..units import require_canonical, unit_dimensions, unit_spec
from .base import DiscoveryResult, checked_candidate, validate_options, native_input_indices

NATIVE_NAMES = {
    op: op
    for op in (
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "sqrt",
        "log",
        "exp",
        "abs",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sinh",
        "cosh",
    )
}
NATIVE_NAMES.update(atan="arctan", asin="arcsin", acos="arccos")
REVERSE_NAMES = {native: common for common, native in NATIVE_NAMES.items()}
POWERS = {"n2": 2, "n3": 3, "n4": 4}
OPERATORS = {*NATIVE_NAMES, "pow_int", "rootn"}


SUPPORTED_OPTIONS = {
    "profile",
    "batch_size",
    "candidate_scope",
    "constants_per_unit",
    "entropy_weight",
    "epochs",
    "max_n_evaluations",
    "max_time_step",
    "parameter_units",
    "risk_factor",
    "response_max_nfev",
    "unit_basis_degree",
}


def profile_options(options):
    """Explicit native preset choice; preserve historical lightweight defaults."""
    profile = options.get("profile", "lightweight")
    if not isinstance(profile, str) or profile not in {"lightweight", "config1", "config2"}:
        raise ValueError("PhySO profile must be lightweight, config1 or config2")
    if profile != "lightweight" and not {"epochs", "max_n_evaluations"} & options.keys():
        raise ValueError("PhySO official profiles require explicit epochs or max_n_evaluations: pinned presets set n_epochs=1e9; no implicit quality/safety budget")
    return {"profile": profile, "preset": "config0" if profile == "lightweight" else profile,
            "epochs": options.get("epochs", 5 if profile == "lightweight" else None),
            "max_n_evaluations": options.get("max_n_evaluations", 2000 if profile == "lightweight" else None)}


def configure_profile(options, preset_config):
    settings = profile_options(options)
    config = deepcopy(preset_config)
    learning = config["learning_config"]
    if settings["profile"] == "lightweight":
        learning["batch_size"] = 128
    for name in ("batch_size", "risk_factor", "entropy_weight"):
        if name in options:
            learning[name] = options[name]
    if int(learning["batch_size"] * learning["risk_factor"]) < 1:
        raise ValueError(
            f"PhySO batch_size={learning['batch_size']} and risk_factor={learning['risk_factor']} produce zero elite "
            "candidates; choose explicit settings with batch_size * risk_factor >= 1; no automatic adjustment"
        )
    return config, settings


def configure_horizon(options, source_max_nodes, config):
    """Separate opt-in native token horizon from the unchanged source AST cap."""
    explicit = "max_time_step" in options
    horizon = options["max_time_step"] if explicit else source_max_nodes
    if type(horizon) is not int or horizon < 1:
        raise ValueError("physo max_time_step must be a positive integer")
    preset_horizon = config["learning_config"]["max_time_step"]
    config["learning_config"]["max_time_step"] = horizon
    if explicit:
        # Native append requires enough token slots for all remaining dummies.
        # Source/slot weighted priors remain separate and are not relaxed.
        config["priors_config"].append(("HardLengthPrior", {"min_length": 0 if horizon == 1 else 1, "max_length": horizon}))
    return {"preset_max_time_step": preset_horizon, "source_max_nodes": source_max_nodes,
            "requested_max_time_step": options.get("max_time_step"), "effective_max_time_step": horizon,
            "origin": "explicit_physo_option" if explicit else "legacy_common_source_max_nodes",
            "token_capacity_prior": "native_HardLengthPrior" if explicit else "legacy_source_prior",
            "native_min_length": (0 if horizon == 1 else 1) if explicit else None,
            "effective_minimum_ast_length": 1,
            "scope": "Native token horizon; source AST and slot caps still enforced independently; no clipping"}


def capabilities():
    return {
        "unit_search": {'defaults': {}, 'choices': {}, 'templates': True},
        "equation_template": True,
        "template_scope": "one shared slot, fixed outer coefficients, variable-only calls, identity formal schema; no signed cbrt",
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": sorted(OPERATORS),
        "fixed_constants": True,
        "constraints": {
            "operators": "enforced during search",
            "units": "native PhysicalUnitsPrior (nonzero epsilon); required portable structural post-filter",
            "max_nodes": "weighted source AST prefix-completion prior before sampling; exponent literals and frozen coordinate transforms included",
            "variables": "disallowed columns excluded before native search; full-schema export/prediction checked",
            "allowed_variables": "native input-column subset with original IDs; no constant-only or mapped-input/template subset fallback",
            "probes": "finite-point hard constrained coefficient fit before native reward/RNN update",
            "response_bounds": "hard finite-point coefficient constraints before reward",
            "response_checks": "finite-step coefficient constraints before reward; not analytic proof",
            "fixed_constants": "enforced during search, immutable values/units; native SR rounded-name collisions rejected",
            "power_indices": "native n2/n3/n4; rootn uses native signed cbrt only",
        },
        "search_constraints": ["probes", "response_bounds", "response_checks"],
        "response_enforcement": {"default": "hard", "supported": ["hard"], "stage": "constrained coefficient fit before native reward and RNN update"},
        "requires_assets": False,
    }


def coefficient_unit_catalog(request, variables=None):
    """Explicit catalog, or a bounded per-input power basis; no y/benchmark law access."""
    if "parameter_units" in request.options:
        if {"unit_basis_degree", "constants_per_unit"} & request.options.keys():
            raise ValueError("Choose explicit parameter_units OR unit_basis_degree, not both")
        return request.options["parameter_units"], "explicit method-profile catalog"
    degree = request.options.get("unit_basis_degree", 2)
    if type(degree) is not int or not 0 <= degree <= 3:
        raise ValueError("unit_basis_degree must be an integer in0..3")
    labels = ["1", request.target["unit"]]
    for variable in request.variables if variables is None else variables:
        label = variable["unit"]
        labels.append(label)  # Input reference scales, e.g. T/tau or E/(R*T).
        labels.extend(f"({request.target['unit']})/({label})^{power}" for power in range(1, degree + 1))
    slots = request.options.get("constants_per_unit", 2)
    if type(slots) is not int or not 1 <= slots <= 16:
        raise ValueError("constants_per_unit must be an integer in1..16")
    catalog, seen = {}, set()
    for label in labels:
        vector = unit_dimensions(label)
        if vector is None:
            raise ValueError(f"Unknown coefficient-catalog dimensions: {label}")
        require_canonical(label)
        spec = unit_spec(label)
        identity = (vector, spec.angle, spec.cycle)
        if identity not in seen:
            for _ in range(slots):
                catalog[f"c{len(catalog)}"] = label
            seen.add(identity)
    return (
        catalog,
        f"input/target dimensions, per-input powers0..{degree}, {slots} independent slots per dimension plus input scales; bounded catalog, not completeness",
    )


def _export(program, variables, parameter_units, fixed_constants=None):
    tokens = iter(program.tokens)
    parameters = {}

    def visit():
        token = next(tokens)
        if token.var_type == 1:
            return {"var": variables[token.var_id]["id"]}
        if token.var_type == 3:
            if token.name in (fixed_constants or {}):
                key, definition = fixed_constants[token.name]
                value = float(token.fixed_const)
                if value != definition["value"]:
                    raise ValueError("Native PhySO fixed constant value changed")
                parameters[key] = {
                    "value": value,
                    "trainable": False,
                    "origin": "declared_fixed_constant",
                    **(
                        {"coordinate_role": definition["coordinate_role"]}
                        if definition.get("coordinate_role") is not None
                        else {}
                    ),
                    "bounds": [value, value],
                    "unit": definition["unit"],
                }
                return {"param": key}
            return {"const": float(token.fixed_const)}
        if token.var_type == 20:
            key = token.name
            parameters[key] = {
                "value": float(program.free_consts.class_values[0][token.var_id]),
                "trainable": True,
                "origin": "native_free_constant_slot",
                "bounds": [None, None],
                "unit": parameter_units.get(key),
            }
            return {"param": key}
        if token.var_type == 0:
            if token.name in POWERS:
                return {"op": "pow_int", "args": [visit(), {"const": POWERS[token.name]}]}
            if token.name == "cbrt":
                return {"op": "rootn", "args": [visit(), {"const": 3}]}
            if token.name in REVERSE_NAMES:
                return {"op": REVERSE_NAMES[token.name], "args": [visit() for _ in range(token.arity)]}
        raise ValueError(f"Unsupported PhySO token: {token.name}")

    root = visit()
    if next(tokens, None) is not None:
        raise ValueError("Unexpected trailing PhySO tokens")
    return {"schema_version": 1, "ast": root, "parameters": parameters}


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    native_input_indices(request, context)
    from .physo_template import prepare as prepare_template
    template = prepare_template(request, context)
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    profile_options(request.options)
    scope = request.options.get("candidate_scope", "complexity_ladder")
    if scope not in {"pareto_front", "complexity_ladder"}:
        raise ValueError("PhySO candidate_scope must be pareto_front or complexity_ladder")
    if set(request.operators) - OPERATORS:
        raise ValueError("Unsupported PhySO operators")
    definitions = [*[request.variables[i] for i in native_input_indices(request, context)], request.target]
    for definition in definitions:
        require_canonical(definition.get("unit"))
    missing = [v["column"] for v in definitions if unit_dimensions(v["unit"]) is None]
    if missing:
        raise ValueError(f"PhySO requires known units for {missing}; set data.units explicitly (dB is unsupported)")
    from dataclasses import replace
    slot_request = replace(request, target={**request.target, "unit": template.slot["unit"]}) if template else request
    parameter_units, catalog_scope = coefficient_unit_catalog(slot_request, [request.variables[i] for i in native_input_indices(request, context)])
    if not isinstance(parameter_units, dict) or any(
        not isinstance(name, str)
        or not name.isidentifier()
        or len(name) >= 10
        or unit_dimensions(unit) is None
        or name in {*request.fixed_constants, *(v["id"] for v in request.variables), "y"}
        for name, unit in parameter_units.items()
    ):
        raise ValueError(
            "PhySO parameter_units needs distinct identifiers shorter than10characters and known unit labels"
        )
    for label in parameter_units.values():
        require_canonical(label)
    fixed = {}
    for name, definition in request.fixed_constants.items():
        require_canonical(definition.get("unit"))
        if unit_dimensions(definition["unit"]) is None:
            raise ValueError(f"Unknown PhySO fixed constant unit: {name}")
        native_name = str(round(definition["value"], 4))
        if native_name in fixed:
            raise ValueError("PhySO SR rounds fixed token names to4decimals; requested constants collide")
        fixed[native_name] = (name, definition)
    specifications = [definition for _, definition in fixed.values()]
    if "1.0" not in fixed:
        specifications.append({"value": 1.0, "unit": "1"})
    native_operators = [NATIVE_NAMES[op] for op in request.operators if op in NATIVE_NAMES]
    if "pow_int" in request.operators:
        native_operators.extend(POWERS)
    if "rootn" in request.operators:
        native_operators.append("cbrt")
    if template:
        native_operators = [NATIVE_NAMES[op] for op in template.operators if op in NATIVE_NAMES]
        if "pow_int" in template.operators:
            native_operators.extend(POWERS)
    for setting in ("risk_factor", "entropy_weight"):
        if setting in request.options:
            value = request.options[setting]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
                or (setting == "risk_factor" and not 0 < value <= 1)
            ):
                raise ValueError(f"Invalid PhySO {setting}")
    for key in ('epochs', 'batch_size', 'max_n_evaluations', 'response_max_nfev', 'max_time_step'):
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"physo {key} must be a positive integer")
    from .physo_response import prepare
    prepare(request)
    return scope, definitions, parameter_units, catalog_scope, fixed, specifications, native_operators


def discover(request, context):
    scope, definitions, parameter_units, catalog_scope, fixed, specifications, native_operators = (
        validate_request(request, context)
    )
    feature_indices = native_input_indices(request, context)
    native_variables = [request.variables[i] for i in feature_indices]
    from .physo_template import local_budget_scope, strict_native_scope
    from .physo_template import prepare as prepare_template
    template = prepare_template(request, context)
    def export_program(program):
        slot = _export(program, template.formal if template else native_variables, parameter_units, fixed)
        return template.export(slot) if template else slot
    from .physo_response import prepare, ResponseReward, reward_scope
    response = prepare(request)
    controller = ResponseReward(request, response, export_program) if response else None
    started = time.monotonic()
    import random

    import physo
    import torch
    from importlib import import_module
    from pathlib import Path
    import hashlib
    settings = profile_options(request.options)
    if version("physo") != "1.1.11":
        raise ValueError("PhySO profiles require reviewed native version 1.1.11")
    preset_module = import_module("physo.config." + settings["preset"])
    preset = getattr(preset_module, settings["preset"])
    preset_sha = hashlib.sha256(Path(preset_module.__file__).read_bytes()).hexdigest()
    from physo.learn.monitoring import RunLogger, RunVisualiser

    imported = time.monotonic()

    np.random.seed(request.seed)
    random.seed(request.seed)
    torch.manual_seed(request.seed)
    torch.set_num_threads(1)
    config, settings = configure_profile(request.options, preset)
    from .physo_budget import source_budget_prior
    from ..coordinates import source_complexity_costs, source_equation
    from ..expressions import inspect_equation
    coordinate_context = {**context.provenance, "variables": request.variables,
                          "target": {"column": "target", **request.target}}
    source_costs, output_cost = source_complexity_costs(coordinate_context)
    budget_audit = []
    local_budget_audit = []
    token_multiplier = 1
    if template:
        costs, output_cost, token_multiplier = template.source_budget()
        source_costs = dict(zip([v["id"] for v in native_variables], costs, strict=True))
    config["priors_config"] = [
        prior
        for prior in config["priors_config"]
        if prior[0] not in {"NestedTrigonometryPrior", "HardLengthPrior"}
        and (prior[0] != "NestedFunctions" or set(prior[1]["functions"]) <= set(native_operators))
    ]
    horizon = configure_horizon(request.options, request.limits["max_nodes"], config)
    vectors = [[float(x) for x in unit_dimensions(v["unit"])] for v in definitions]
    if template:
        vectors[-1] = [float(x) for x in unit_dimensions(template.slot["unit"])]

    def config_value(value):
        if isinstance(value, dict):
            return {key: config_value(entry) for key, entry in value.items()}
        if isinstance(value, (list, tuple)):
            return [config_value(entry) for entry in value]
        if isinstance(value, np.generic):
            return value.item()
        return f"{value.__module__}.{value.__qualname__}" if callable(value) else value

    resolved_config = config_value(config)
    resolved_profile = {
        **settings, "physo_version": version("physo"), "preset_source_sha256": preset_sha,
        "preset_learning_config": config_value(preset["learning_config"]),
        "explicit_overrides": {k: request.options[k] for k in
            ("batch_size", "epochs", "max_n_evaluations", "risk_factor", "entropy_weight", "max_time_step") if k in request.options},
        "effective_epochs": settings["epochs"] if settings["epochs"] is not None else config["learning_config"]["n_epochs"],
        "effective_batch_size": config["learning_config"]["batch_size"],
        "horizon": horizon,
        "preset_hard_length_priors": config_value([p for p in preset["priors_config"] if p[0] == "HardLengthPrior"]),
        "hard_length_policy": "Existing adapter replaces preset HardLength with weighted source/slot priors; explicit horizon adds min1/maxT (T1: native min0, natural AST minimum1). Active priors are in resolved_native_config.",
        "override_order": ["pinned preset", "legacy lightweight defaults only", "explicit method options", "common source AST budget and supported-operator prior adjustments", "CPU single-thread/nonparallel resource policy"],
        "resources": {"device": "cpu", "torch_threads": int(torch.get_num_threads()), "parallel_mode": False},
        "unit_mode": "all input/target units dimensionless" if all(not any(vector) for vector in vectors) else "dimensioned input/target units",
        "scope": "preset-derived configuration plus platform extensions; not a quality guarantee; no automatic unit-based preset switch",
    }
    work = {"optimizer_steps": 0, "gradient_l2": None, "parameter_change_l2": None}
    checkpoint_seconds = [0.0]
    latest = {}

    class NativeBudgetReached(Exception):
        pass

    class NoFeasibleReward(Exception):
        pass

    epoch_trace = []
    native_optimizer = config["learning_config"]["get_optimizer"]

    def observed_optimizer(model):
        optimizer = native_optimizer(model)
        native_step = optimizer.step

        def step(*args, **kwargs):
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            before = [p.detach().clone() for p in parameters]
            gradient = sum(float(torch.sum(p.grad.detach() ** 2)) for p in parameters if p.grad is not None)
            result = native_step(*args, **kwargs)
            change = sum(float(torch.sum((p.detach() - old) ** 2)) for p, old in zip(parameters, before, strict=True))
            work.update(
                optimizer_steps=work["optimizer_steps"] + 1,
                gradient_l2=float(np.sqrt(gradient)) if np.isfinite(gradient) else None,
                parameter_change_l2=float(np.sqrt(change)) if np.isfinite(change) else None,
            )
            return result

        optimizer.step = step
        return optimizer

    def completed_result(logger, finalized=False):
        fit_finished = time.monotonic()
        _, front, _, _ = (
            logger.get_pareto_front()
            if any(
                p is not None and np.isfinite(r) and r > 0
                for p, r in zip(logger.pareto_programs, logger.pareto_rewards, strict=True)
            )
            else ([], [], [], [])
        )
        programs = (
            list(front)
            if scope == "pareto_front"
            else [
                p
                for p, reward in zip(logger.pareto_programs, logger.pareto_rewards, strict=True)
                if p is not None and np.isfinite(reward) and reward > 0
            ]
        )
        candidates, rejected = [], []
        for index, program in enumerate(programs):
            try:
                equation = export_program(program)
                public_equation, public_variables = source_equation(equation, coordinate_context)
                if inspect_equation(public_equation, public_variables)["n_nodes"] > request.limits["max_nodes"]:
                    raise RuntimeError("PhySO sampling violated its source-coordinate search budget")
                if response is not None:
                    from ..expressions import evaluate
                    from ..response_search import response_feasible
                    if not response_feasible(evaluate(equation, np.asarray(response['points']), request.variables), response):
                        raise ValueError('PhySO saved reward candidate violates requested finite-point bounds')
                candidates.append(
                    checked_candidate(
                        "physo",
                        index,
                        equation,
                        request,
                        lambda X, p=program: p(torch.tensor(X[:, feature_indices].T, dtype=torch.float64)).detach().cpu().numpy(),
                        native_ieee=template is None,
                    )
                )
                if response is not None:
                    candidates[-1].native_check.update(finite_point_feasible=True,
                        response_stage='coefficient_fit_before_native_reward_and_RNN_update',
                        solver_converged=None,
                        convergence_scope='aggregate reward-visit diagnostics only, not candidate-level convergence proof')
            except (ValueError, TypeError) as exc:
                rejected.append({"index": index, "reason": str(exc)})
        raw = write_json(
            context.work_dir / "physo_native.json",
            {
                "template_search": ({"scope": "single shared slot / fixed outer / variable-only calls",
                    "slot_output_unit": template.slot["unit"], "observed_y_unit": request.target["unit"],
                    "loss_target": "original observed y", "strict_real": template.audit} if template else None),
                "response_search": controller.summary() if controller else None,
                "fixed_constants": request.fixed_constants,
                "parameter_units": parameter_units,
                "coefficient_catalog_scope": catalog_scope,
                "candidate_scope": scope,
                "native_learning_trace": epoch_trace,
                "resolved_native_config": resolved_config,
                "physo_profile": resolved_profile,
                "native_operators": native_operators,
                "programs": [
                    {
                        "rank": i,
                        "prefix": [token.name for token in program.tokens],
                        "free_parameters": [
                            [float(v) if np.isfinite(v) else str(float(v)) for v in row]
                            for row in program.free_consts.class_values.detach().cpu().tolist()
                        ],
                    }
                    for i, program in enumerate(programs)
                ],
                "rejected_exports": rejected,
            },
        )
        return DiscoveryResult(
            candidates,
            ("ok" if finalized else "partial") if candidates else "failed",
            {
                "native_candidates": len(programs),
                "input_rows": len(request.X),
                "native_input_variables": [v["id"] for v in native_variables],
                "native_input_columns": [v["column"] for v in native_variables],
                "source_complexity_budget": {"max_nodes": request.limits["max_nodes"], "variable_costs": source_costs,
                                             "output_cost": output_cost, "prior_instances": budget_audit, "slot_prior_instances": local_budget_audit, "token_multiplier": token_multiplier},
                "native_execution_status": "completed" if finalized else "running",
                "termination_reason": "native_return" if finalized else "completed_iteration_snapshot",
                "checkpoint_seconds": checkpoint_seconds[0],
                "physo": version("physo"),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "native_import_seconds": imported - started,
                "native_search_fit_seconds": fit_finished - fit_started,
                "native_export_seconds": time.monotonic() - fit_finished,
                "timing_scope": {
                    "native_search_fit_seconds": "inclusive elapsed through the start of this export; contains earlier completed-iteration conversion/parity/snapshot work",
                    "native_export_seconds": "this export only, after the preceding elapsed interval",
                    "checkpoint_seconds": "overlapping subset: earlier completed logger hooks, including conversion/parity/atomic snapshot; never add to search elapsed",
                },
                "native_search_operators": native_operators,
                "template_search": ({"scope": "single shared slot / fixed outer / variable-only calls",
                    "slot_output_unit": template.slot["unit"], "observed_y_unit": request.target["unit"],
                    "loss_target": "original observed y", "strict_real": template.audit} if template else None),
                "response_search": controller.summary() if controller else None,
                "fixed_constants": request.fixed_constants,
                "parameter_units": parameter_units,
                "coefficient_catalog_scope": catalog_scope,
                "resolved_native_config": resolved_config,
                "physo_profile": resolved_profile,
                "candidate_scope": scope,
                "native_strict_pareto_candidates": len(front),
                "native_work": {
                    "epochs_completed": len(epoch_trace),
                    "sampled": sum(row["sampled"] for row in epoch_trace),
                    "physical": sum(row["physical"] for row in epoch_trace),
                    "positive_reward": sum(row["positive_reward"] for row in epoch_trace),
                    "optimizer_steps": work["optimizer_steps"],
                    "scope": "actual logger rows/optimizer steps; reward counts can repeat expressions and do not prove convergence",
                },
                "rejected_exports": rejected,
                "constraint_support": capabilities()["constraints"],
            },
            {"native_programs": str(raw)},
        )

    class ObservedLogger(RunLogger):
        def get_pareto_front(self):
            # Native ClassSR assumes a nonempty positive-reward front at return.
            # No fabricated HOF expression: preserve the completed failed snapshot.
            if (controller is not None or template is not None) and not any(np.isfinite(r) and r > 0 for r in self.pareto_rewards):
                raise NoFeasibleReward()
            return super().get_pareto_front()

        def log(self, *args, **kwargs):
            result = super().log(*args, **kwargs)
            loss = float(self.loss_history[-1])
            epoch_trace.append(
                {
                    "epoch": int(self.epoch),
                    "sampled": int(self.batch.batch_size),
                    "physical": int(self.n_physical[-1]),
                    "positive_reward": int(self.n_rewarded[-1]),
                    "elite": int(len(self.keep)),
                    "loss": loss if np.isfinite(loss) else None,
                    **work,
                }
            )
            checkpoint_started = time.monotonic()
            saved = completed_result(self)
            latest["result"] = saved
            if context.checkpoint is not None:
                context.checkpoint(saved)
            checkpoint_seconds[0] += time.monotonic() - checkpoint_started
            if context.deadline is not None and time.monotonic() >= context.deadline:
                raise NativeBudgetReached()
            return result

    config["learning_config"]["get_optimizer"] = observed_optimizer
    fit_started = time.monotonic()
    native_logger = ObservedLogger(do_save=False)
    try:
        with strict_native_scope(template), reward_scope(controller), local_budget_scope(template, set(native_operators), local_budget_audit), source_budget_prior(request.limits["max_nodes"], [source_costs[v["id"]] for v in native_variables],
                                 output_cost, set(native_operators), budget_audit, token_multiplier=token_multiplier):
            _, logger = physo.SR(
                request.X[:, feature_indices].T,
                request.y,
                candidate_wrapper=template,
                X_names=[v["id"] for v in native_variables],
                X_units=vectors[:-1],
                y_name="y",
                y_units=vectors[-1],
                fixed_consts=[entry["value"] for entry in specifications],
                fixed_consts_units=[[float(x) for x in unit_dimensions(entry["unit"])] for entry in specifications],
                free_consts_names=list(parameter_units),
                free_consts_units=[[float(x) for x in unit_dimensions(unit)] for unit in parameter_units.values()],
                op_names=native_operators,
                use_protected_ops=False,
                epochs=settings["epochs"],
                max_n_evaluations=settings["max_n_evaluations"],
                run_config=config,
                parallel_mode=False,
                device="cpu",
                get_run_logger=lambda: native_logger,
                get_run_visualiser=lambda: RunVisualiser(do_show=False, do_save=False, do_prints=False),
            )
        if not budget_audit:
            raise RuntimeError("PhySO sampling did not construct its required source budget prior")
    except NoFeasibleReward:
        result = completed_result(native_logger, finalized=True)
        result.summary.update(termination_reason='no_positive_feasible_reward',
                              reason='No accepted finite-point-feasible positive-reward expression was obtained; this is not a proof of infeasibility')
        return result
    except NativeBudgetReached:
        result = latest["result"]
        result.summary.update(
            native_execution_status="budget_exhausted",
            termination_reason="wall_budget_after_completed_iteration",
            checkpoint_seconds=checkpoint_seconds[0],
            native_search_fit_seconds=time.monotonic() - fit_started,
            native_export_seconds=0.0,
            timing_scope={
                "native_search_fit_seconds": "inclusive native call elapsed including all completed logger hooks, conversion/parity/atomic snapshots",
                "native_export_seconds": "zero additional export: returning the last completed-iteration result",
                "checkpoint_seconds": "overlapping measured subset of inclusive elapsed; never add to search elapsed",
            },
            checkpoint_scope="data-only hypotheses; no optimizer/RNG resume guarantee",
        )
        return result
    # Final native/export parity must use the same strict execution as reward/checkpoints.
    with strict_native_scope(template):
        return completed_result(logger, finalized=True)
