"""Official PSE/PSRN enumeration and native coefficient fitting; portable export only."""

import hashlib
import random
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ..data import write_json
from ..expressions import lower_integer_powers, parse_expression
from .base import DiscoveryResult, checked_candidate, validate_options

OPERATORS = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "sin": "Sin",
    "cos": "Cos",
    "exp": "Exp",
    "log": "Log",
    "neg": "Neg",
    "abs": "Abs",
    "sqrt": "Sqrt",
    "tanh": "Tanh",
    "cosh": "Cosh",
    "pow": "Pow",
}


def stage_configuration(request):
    """Translate explicit native stages, never infer a vocabulary from benchmark truth."""
    default = {
        "operators": request.operators,
        "slots": request.options.get("slots", 3),
        "sample_variables": min(request.options.get("slots", 3), len(request.variables)),
        "seconds": request.options.get("seconds", request.limits["max_wall_seconds"] * 0.75),
    }
    stages = request.options.get("stages", [{}])
    if not isinstance(stages, list) or not stages or len(stages) > 8:
        raise ValueError("PSE stages must be a nonempty explicit list of at most8native stages")
    resolved = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - set(default):
            raise ValueError("PSE stage accepts operators/slots/sample_variables/seconds")
        settings = {**default, **stage}
        if not settings["operators"] or not set(settings["operators"]) <= set(request.operators):
            raise ValueError("PSE stage operators must be a subset of the shared requested operators")
        if (
            type(settings["slots"]) is not int
            or settings["slots"] < 1
            or type(settings["sample_variables"]) is not int
            or not 1 <= settings["sample_variables"] <= min(settings["slots"], len(request.variables))
        ):
            raise ValueError("Invalid PSE slots/sample_variables")
        if (
            isinstance(settings["seconds"], bool)
            or not isinstance(settings["seconds"], (int, float))
            or not np.isfinite(settings["seconds"])
            or settings["seconds"] <= 0
        ):
            raise ValueError("PSE stage seconds must be positive and finite")
        ops = [OPERATORS[op] for op in settings["operators"] if op in OPERATORS]
        if "pow_int" in settings["operators"]:
            ops.extend(["Pow2", "Pow3"])
        if not ops:
            raise ValueError("PSE stage has no supported ordinary final-expression operators")
        resolved.append(
            {
                "operators": [*ops, "Identity"],
                "n_psrn_inputs": settings["slots"],
                "n_sample_variables": settings["sample_variables"],
                "time_limit": settings["seconds"],
            }
        )
    return resolved


SUPPORTED_OPTIONS = {
    "layers",
    "seconds",
    "slots",
    "stages",
    "top_k",
    "use_const",
    "use_dr_mask",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": [*OPERATORS, "pow_int"],
        "requires_assets": False,
        "devices": ["cpu", "cuda"],
        "constraints": {
            "operators": "requested enumeration operators; GP proposals can be broader, common export post-filter mandatory",
            "constants": "native Powell fit_LS templates/coefficients retained; optional constant-free profile",
            "parameter_bounds": "native unconstrained fit; common bounds post-filter only",
            "units": "native dimensions unsupported; portable post-filter",
            "max_nodes": "native layer/slot budget plus post-filter",
            "variables": "post-filtered",
            "probes": "post-filtered",
            "pretrained_weights": "not applicable: GPU enumeration is not pretrained",
        },
    }


def _export_fit(record, variables):
    parameters = {
        f"C{i}": {
            "value": float(v),
            "trainable": True,
            "bounds": [None, None],
            "unit": None,
            "origin": "native_fit_LS_slot",
        }
        for i, v in enumerate(record["coefficients"])
    }
    return lower_integer_powers(parse_expression(record["template"], variables, trainable=False, parameters=parameters))


GP_OPERATOR_SOURCE_SHA256 = "1ef344ff163145024727fe77268780c5d2dcecaa87fc437c84b654005bc0c76b"


@contextmanager
def ordinary_gp_operators():
    """Bind two missing/inconsistent GP definitions to the pinned native tensors.

    Upstream PSRN has a Cosh tensor operator but no GP expression definition;
    its Pow3 GP numeric callback squares while its expression string cubes.
    This scoped compatibility repair preserves the requested vocabulary and
    restores the caller's module even when native fitting raises.
    """
    from psrn.model.token_generator.GP.model.expr_utils import utils
    from psrn.model.token_generator.GP.model.expr_utils.exp_tree_node import Expression

    if hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest() != GP_OPERATOR_SOURCE_SHA256:
        raise ValueError("Unexpected pinned PSRN GP source; ordinary operator repair was not applied")
    original = utils.get_expression

    def get_expression(name):
        if name == "Cosh":
            return Expression(1, np.cosh, lambda x: f"cosh({x})", name)
        if name == "Pow3":
            return Expression(1, lambda x: np.power(x, 3), lambda x: f"({x})**3", name)
        return original(name)

    utils.get_expression = get_expression
    try:
        yield
    finally:
        utils.get_expression = original


def validate_request(request, context):
    """Resolve stage configuration without native imports or mask generation."""
    validate_options(request.options, SUPPORTED_OPTIONS)
    layers, slots = request.options.get("layers", 2), request.options.get("slots", 3)
    use_const = request.options.get("use_const", True)
    if type(layers) is not int or layers < 1 or type(slots) is not int or slots < 1 or not isinstance(use_const, bool):
        raise ValueError("PSE layers/slots must be positive integers and use_const must be boolean")
    stages = stage_configuration(request)
    ops = stages[0]["operators"]
    slots = stages[0]["n_psrn_inputs"]
    sample_variables = stages[0]["n_sample_variables"]
    use_dr_mask = request.options.get("use_dr_mask", False)
    if not isinstance(use_dr_mask, bool):
        raise ValueError("use_dr_mask must be boolean")
    for key in ('top_k',):
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"pse {key} must be a positive integer")
    return layers, slots, use_const, stages, ops, sample_variables, use_dr_mask


def discover(request, context):
    resolved = validate_request(request, context)
    return _discover(request, context, resolved)


@ordinary_gp_operators()
def _discover(request, context, resolved):
    layers, slots, use_const, stages, ops, sample_variables, use_dr_mask = resolved
    import torch
    from psrn import PSRN_Regressor
    from psrn.model import regressor as native_module
    from psrn.utils.data import expr_to_Y_pred

    if getattr(native_module, "symbolic_handoff_constant_version", None) != 2:
        raise ValueError("PSE requires the pinned token-safe native constant patch; rebuild the GPU profile")
    if context.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("PSE CUDA was explicitly requested but is unavailable; no CPU fallback")
    torch.manual_seed(request.seed)
    torch.set_num_threads(1)
    np.random.seed(request.seed)
    random.seed(request.seed)
    mask_dir = context.work_dir / "dr_masks"
    if use_dr_mask:
        mask_dir.mkdir(exist_ok=True)
        for stage in stages:
            path = mask_dir / f"{layers}_{stage['n_psrn_inputs']}_[{'_'.join(stage['operators'])}]_mask.npy"
            if not path.is_file():
                native_module.generate_dr_mask_core(
                    n_symbol_layers=layers,
                    n_inputs=stage["n_psrn_inputs"],
                    ops=stage["operators"],
                    save_dir=str(mask_dir),
                    device=context.device,
                )
            if not path.is_file():
                raise ValueError("Official PSE DR mask generation did not produce the requested local mask")

    class RecordedRegressor(PSRN_Regressor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fits, self.retained_fits, self.initial_candidates = {}, {}, {}
            self.fit_calls, self.forward_calls = 0, 0
            self.stage_work = []

        def fit_one(self, operators, n_psrn_inputs, n_sample_variables, time_limit):
            start = time.monotonic()
            forwards, fits = self.forward_calls, self.fit_calls
            kwargs = {
                "operators": operators,
                "n_psrn_inputs": n_psrn_inputs,
                "n_sample_variables": n_sample_variables,
                "time_limit": time_limit,
            }
            result = super().fit_one(**kwargs)
            self.stage_work.append(
                {
                    **kwargs,
                    "seconds": time.monotonic() - start,
                    "network_forward_calls": self.forward_calls - forwards,
                    "fit_calls": self.fit_calls - fits,
                    "pareto_candidates_after_stage": len(self.get_pf(sort_by="mse")),
                }
            )
            return result

        def load_dr_mask(self):
            if self.use_dr_mask:
                path = mask_dir / f"{self.n_symbol_layers}_{self.n_inputs}_[{'_'.join(self.operators)}]_mask.npy"
                if not path.is_file():
                    raise ValueError("Missing prepared DR mask; do not invoke a hidden shell fallback")
            return super().load_dr_mask()

        def fit_LS(self, *args, **kwargs):
            self.fit_calls += 1
            result = super().fit_LS(*args, **kwargs)
            coefficients, loss, template, expression = result
            if coefficients is not None and np.isfinite(loss) and np.isfinite(coefficients).all():
                self.fits[str(expression)] = {
                    "template": str(template),
                    "coefficients": np.asarray(coefficients, dtype=float).tolist(),
                    "expression": str(expression),
                    "native_loss": float(loss),
                    "fit_origin": "native fit_LS/Powell; no common refit",
                }
            return result

        def _symbolic_record_pareto(self, original, retained):
            # Observe the actual native transformation, once. Reparsing a rendered
            # float expression can round/combine coefficients and lose its identity.
            if str(original) in self.fits:
                self.retained_fits[retained] = self.fits[str(original)]
            elif str(original) in self.initial_candidates:
                self.retained_fits[retained] = self.initial_candidates[str(original)]

        def _symbolic_record_initial(self, kind, expression, coefficients):
            template = "C0"
            if kind == "linear":
                template += "".join(f" + C{i + 1}*{variable}" for i, variable in enumerate(self.variables))
            self.initial_candidates[expression] = {
                "template": template,
                "coefficients": coefficients,
                "expression": expression,
                "fit_origin": f"native initial {kind}; full fitted coefficients, separate from enumeration/Powell",
            }

    fitted = RecordedRegressor(
        variables=[v["id"] for v in request.variables],
        operators=ops,
        device=context.device,
        n_symbol_layers=layers,
        n_inputs=slots,
        n_sample_variables=sample_variables,
        use_const=use_const,
        use_dr_mask=use_dr_mask,
        dr_mask_dir=str(mask_dir),
        token_generator="GP",
        token_generator_config={
            "base": {
                "has_const": use_const,
                "tokens": list(dict.fromkeys(op for stage in stages for op in stage["operators"])),
            }
        },
        stage_config={"default": stages[0], "stages": stages},
    )

    native_forward = native_module.PSRN.forward

    def observe_forward(module, *args, **kwargs):
        # Pinned PSRN calls .forward directly, bypassing nn.Module hooks.
        if module is fitted.net:
            fitted.forward_calls += 1
        return native_forward(module, *args, **kwargs)

    native_module.PSRN.forward = observe_forward
    try:
        fitted.fit(
            request.X.copy(),
            request.y.copy().reshape(-1, 1),
            n_down_sample=len(request.X),  # Native fit_one() cannot min(None,n).
            use_threshold=False,
            probe=None,
            prun_const=False,
            together=False,
            use_replace_expo=False,
            real_time_display=False,
            top_k=request.options.get("top_k", 10),
        )
    finally:
        native_module.PSRN.forward = native_forward
    if fitted.forward_calls == 0:
        raise ValueError("No observed native PSRN enumeration; baseline-only execution is not accepted")

    native = fitted.get_pf(sort_by="mse")
    records = []
    for index, (text, reward, loss, complexity) in enumerate(native):
        key = str(text)
        record = {
            "index": index,
            "expression": str(text),
            "reward": float(reward),
            "native_loss": float(loss),
            "native_complexity": int(complexity),
        }
        if key in fitted.retained_fits:
            record.update(fitted.retained_fits[key])
            record["retained_expression"] = str(text)
        else:
            record["fit_origin"] = "constant-free enumeration" if not use_const else "unmapped native fit"
        records.append(record)
    native_path = context.work_dir / "pse_native.json"
    write_json(
        native_path,
        {
            "pareto": records,
            "fit_calls": fitted.fit_calls,
            "successful_unique_fits": len(fitted.fits),
            "successful_fits": list(fitted.fits.values()),
            "initial_fits": list(fitted.initial_candidates.values()),
            "forward_calls": fitted.forward_calls,
            "stages": fitted.stage_work,
            "effective_stage_config": stages,
            "use_dr_mask": use_dr_mask,
            "fit_X_y_dtype": "float32",
            "optimizer_coefficient_dtype": "float64",
            "fixed_exponents": True,
            "common_refit": False,
        },
    )
    candidates, rejected = [], []
    for record in records:
        index = record["index"]
        try:
            if record["fit_origin"] == "unmapped native fit":
                raise ValueError("Native fitted expression has no matching coefficient/template provenance")
            equation = (
                _export_fit(record, request.variables)
                if "template" in record
                else lower_integer_powers(parse_expression(record["expression"], request.variables, trainable=False))
            )
            expression = record.get("retained_expression", record["expression"])
            candidate = checked_candidate(
                "pse",
                index,
                equation,
                request,
                lambda X, text=expression: np.asarray(expr_to_Y_pred(text, X, fitted.variables)).reshape(-1),
            )
            candidate.native_check.update(native_record=index, parameter_origin=record["fit_origin"])
            candidates.append(candidate)
        except (ValueError, TypeError) as exc:
            rejected.append({"index": index, "reason": str(exc)})
    return DiscoveryResult(
        candidates,
        "ok" if candidates else "failed",
        {
            "version": version("psrn"),
            "source_revision": "a4f6f951c95af6a1f5c9c0f057be432e319c73f8",
            "native_regressor_sha256": hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest(),
            "native_patch": "token-safe variable/coefficient replacement; grouped signed coefficients, optimizer unchanged",
            "native_gp_operator_source_sha256": GP_OPERATOR_SOURCE_SHA256,
            "native_gp_operator_repair": "scoped Cosh definition and Pow3 cube callback; native tensor semantics unchanged",
            "original_method_source": "https://github.com/intell-sci-comput/PSE",
            "installed_implementation": "https://github.com/x66ccff/PSRN",
            "license": "MIT",
            "torch_version": torch.__version__,
            "native_fit_dtype": "float32",
            "export_evaluation_dtype": "float64",
            "native_network_forward_calls": fitted.forward_calls,
            "native_fit_calls": fitted.fit_calls,
            "native_successful_unique_fits": len(fitted.fits),
            "native_candidates": len(native),
            "native_stages": fitted.stage_work,
            "effective_stage_config": stages,
            "use_dr_mask": use_dr_mask,
            "dr_mask_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in mask_dir.glob("*.npy")}
            if use_dr_mask
            else {},
            "native_fitted_pareto_candidates": sum(
                record["fit_origin"].startswith("native fit_LS") for record in records
            ),
            "native_initial_pareto_candidates": sum(
                record["fit_origin"].startswith("native initial") for record in records
            ),
            "input_rows": len(request.X),
            "use_const": use_const,
            "native_sampling": f"all features supplied; native generator samples {sample_variables} per proposal",
            "token_generator": "official GP; proposal transforms can be protected, enumeration/fit/prediction ordinary",
            "native_affine_initialization": "native mean/linear full coefficients; prun_const=False respected; separate from enumeration/Powell",
            "gpu_name": torch.cuda.get_device_name() if context.device.startswith("cuda") else None,
            "rejected_exports": rejected,
            "constraint_support": capabilities()["constraints"],
        },
        {"pareto": native_path},
    )
