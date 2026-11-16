"""Official RASR pretrained generation with full-precision original-scale export."""

import os
import time
from importlib.metadata import version

import numpy as np

from ..data import write_json
from ..expressions import inspect_equation, lower_integer_powers
from .torch_runtime import thread_count, torch_thread_scope
from .base import DiscoveryResult, checked_candidate, validate_options

SUPPORTED_OPTIONS = {
    "threads",
    "candidates",
    "decode_batch_size",
    "restarts",
    "operator_mask",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": [
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "abs",
            "exp",
            "log",
            "sqrt",
            "rootn",
            "pow_int",
            "sin",
            "cos",
            "tanh",
        ],
        "requires_assets": True,
        "search_constraints": ["response_bounds", "response_checks", "probes"],
        "constraints": {
            "operators": "opt-in operator_mask restricts native tokens before sampling; portable full expression still checked",
            "units": "fixed train-normalization quantities retain declared units; fitted coefficient units unknown",
            "max_nodes": "post-filtered including original-scale transforms",
            "variables": "reject width>11 and native references outside supplied width",
            "probes": "serial native candidate coefficient refinement before native ranking; finite points only",
            "response_bounds": "explicit common coefficient inequalities inside native refinement; decoder unchanged",
            "response_checks": "finite-step constraints inside native coefficient refinement; no global guarantee",
            "pretraining_overlap": "unknown; inference only, no fine-tuning",
        },
    }


def export_candidate(candidate, fitted, variables, target):
    tokens, values = candidate.tokens, iter(candidate.constants)
    parameters, cursor, count = {}, 0, 0

    def fixed(key, value, unit):
        parameters[key] = {
            "value": float(value),
            "trainable": False,
            "bounds": [None, None],
            "unit": unit,
            "origin": "train_normalization",
        }
        return {"param": key}

    def variable(index):
        if not 0 <= index < len(variables):
            raise ValueError("RASR references a variable outside supplied width")
        definition = variables[index]
        mean = fixed(f"x{index}_mean", fitted.X_mean_[index], definition["unit"])
        scale = fixed(f"x{index}_scale", fitted.X_std_[index], definition["unit"])
        return {"op": "div", "args": [{"op": "sub", "args": [{"var": definition["id"]}, mean]}, scale]}

    def parse():
        nonlocal cursor, count
        if cursor >= len(tokens):
            raise ValueError("Truncated native RASR prefix")
        token = tokens[cursor]
        cursor += 1
        if token in {"+", "-", "*", "/"}:
            return {"op": {"+": "add", "-": "sub", "*": "mul", "/": "div"}[token], "args": [parse(), parse()]}
        if token in {"neg", "abs", "exp", "log", "sin", "cos", "tanh"}:
            return {"op": token, "args": [parse()]}
        if token == "inv":
            return {"op": "div", "args": [{"const": 1}, parse()]}
        if token in {"pow2", "pow3", "pow4", "pow5"}:
            return {"op": "pow_int", "args": [parse(), {"const": int(token[-1])}]}
        if token == "pow1_2":
            return {"op": "sqrt", "args": [parse()]}
        if token in {"pow1_3", "pow1_4", "pow1_5"}:
            # Pinned SimpliPy: signed odd roots, principal even root. The index
            # is an operator literal, never a fitted prefix-constant slot.
            return {"op": "rootn", "args": [parse(), {"const": int(token[-1])}]}
        if token.startswith("x") and token[1:].isdigit():
            return variable(int(token[1:]) - 1)
        try:
            if token != "<constant>":
                float(token)
            value = float(next(values))
        except (ValueError, StopIteration) as exc:
            raise ValueError(f"Unsupported native RASR operator or constant layout: {token}") from exc
        key = f"c{count}"
        count += 1
        parameters[key] = {
            "value": value,
            "trainable": True,
            "bounds": [None, None],
            "unit": None,
            "origin": "native_prefix_constant",
        }
        return {"param": key}

    normalized = parse()
    if cursor != len(tokens) or next(values, None) is not None:
        raise ValueError("Native RASR prefix/constants count mismatch")
    scale = fixed("y_scale", fitted.y_std_scale_, target["unit"])
    mean = fixed("y_mean", fitted.y_mean_, target["unit"])
    ast = {"op": "add", "args": [{"op": "mul", "args": [normalized, scale]}, mean]}
    equation = lower_integer_powers({"schema_version": 2, "ast": ast, "parameters": parameters, "arithmetic": "numpy_ieee"})
    inspect_equation(equation, variables)
    return equation


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    validate_options(request.options, SUPPORTED_OPTIONS)
    if "operator_mask" in request.options and type(request.options["operator_mask"]) is not bool:
        raise ValueError("operator_mask must be a boolean")
    if request.equation_template or request.limits.get("equation_template"):
        raise ValueError("Native RASR does not implement equation templates; use an explicit proposal + joint template fit engine")
    if not 1 <= len(request.variables) <= 11:
        raise ValueError("RASR requires 1..11 supplied features; native truncation is forbidden")
    for key in ("candidates", "decode_batch_size", "restarts"):
        if key in request.options and (type(request.options[key]) is not int or request.options[key] < 1):
            raise ValueError(f"RASR {key} must be a positive integer")
    thread_count(request.options)
    return None


def discover(request, context):
    validate_request(request, context)
    with torch_thread_scope(request.options):
        return _discover(request, context)


def _discover(request, context):
    validate_request(request, context)
    os.environ.update(
        XDG_CACHE_HOME=str(context.assets_dir / "cache/xdg"),
        HF_HOME=str(context.assets_dir / "cache/hf"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
    )
    import torch
    from src.rasr.inference.rasr import RASR

    torch.manual_seed(request.seed)
    np.random.seed(request.seed)
    from .rasr_progress import NativeProgress

    progress = NativeProgress(request, context, export_candidate)
    progress("load_started", torch_threads=torch.get_num_threads())
    start = time.monotonic()
    fitted = RASR(
        model_path=str(context.assets_dir / "model"),
        device=context.device,
        num_candidates=request.options.get("candidates", 16),
        decode_batch_size=request.options.get("decode_batch_size", 4),
        n_restarts=request.options.get("restarts", 1),
        refiner_workers=0,
    )
    load_seconds = time.monotonic() - start
    import src.rasr.inference.rasr as native_rasr
    if getattr(native_rasr, "symbolic_progress_version", None) != 1:
        raise ValueError("Rebuild RASR profile for completed-native refinement checkpoints")
    progress.fitted = fitted
    fitted.symbolic_progress = progress
    progress("load_completed")
    from ..response_search import requested_response_constraints
    response_refinement = None
    if request.limits.get("response_mode", "search") == "search" and requested_response_constraints(request.limits):
        import src.rasr.inference.rasr as native_rasr
        import src.rasr.inference.refiner as native_refiner
        if any(getattr(module, "symbolic_response_refinement_version", None) != 1
               for module in (native_rasr, native_refiner)):
            raise ValueError("Rebuild RASR profile for pinned serial constrained refinement")
        from .rasr_refinement import ResponseRefinement
        response_refinement = ResponseRefinement(request, fitted, context.work_dir / "rasr_refinement.json")
        fitted.coefficient_solver = response_refinement
    if len(request.variables) > fitted.n_variables:
        raise ValueError("Configured RASR model width is smaller than supplied features")
    operator_mask = None
    if request.options.get("operator_mask", False):
        import src.rasr.inference.rasr as native_rasr
        import src.rasr.model.model as native_model
        if any(getattr(module, "symbolic_operator_mask_version", None) != 1
               for module in (native_rasr, native_model)):
            raise ValueError("Rebuild RASR profile for pinned operator-token masking")
        from .rasr_decode import OperatorMask
        operator_mask = OperatorMask(fitted, set(request.operators))
        fitted.token_mask = operator_mask
    try:
        start = time.monotonic()
        fitted.fit(request.X, request.y, simplify_expressions=False)
        fit_seconds = time.monotonic() - start
        progress("native_fit_completed")
        candidates, rejected = [], []
        for rank, native in enumerate(fitted.results_):
            try:
                equation = export_candidate(native, fitted, request.variables, request.target)
                candidates.append(
                    checked_candidate(
                        "rasr",
                        rank,
                        equation,
                        request,
                        lambda X, index=rank: np.asarray(fitted.predict(X, nth_best_beam=index)).reshape(-1),
                    )
                )
            except (ValueError, TypeError, OverflowError) as exc:
                rejected.append({"native_rank": rank, "reason": str(exc)})
        path = write_json(
            context.work_dir / "rasr_native.json",
            {
                "candidates": [
                    {"native_rank": i, "tokens": c.tokens, "constants": c.constants}
                    for i, c in enumerate(fitted.results_)
                ],
                "X_mean": fitted.X_mean_.tolist(),
                "X_std": fitted.X_std_.tolist(),
                "y_mean": float(fitted.y_mean_),
                "y_scale": float(fitted.y_std_scale_),
            },
        )
        return DiscoveryResult(
            candidates,
            "ok" if candidates else "failed",
            {
                "version": version("rasr"),
                "export_arithmetic": "native NumPy IEEE parity; common real-domain invalid intermediates remain invalid",
                "source_revision": "f7bdf878a46f0a691d55146d7b4cf1ddac605db7",
                "license": "MIT",
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "simplipy": version("simplipy"),
                "model_load_seconds": load_seconds,
                "native_fit_seconds": fit_seconds,
                "native_progress": progress.summary,
                "native_candidates": len(fitted.results_),
                "operator_mask": {
                    "enabled": operator_mask is not None,
                    "runtime_patch_version": 1 if operator_mask else None,
                    "stage": "native logits before top-k/temperature/top-p; original model log-probability retained",
                    "forbidden_tokens": operator_mask.forbidden_tokens if operator_mask else [],
                    "decoded_steps": operator_mask.steps if operator_mask else 0,
                    "scope": "operator tokens only; no syntax, size, response or known-relation guarantee",
                    "coordinates": "generated normalized tree only; fixed normalization preserved and full portable tree checked",
                    "distribution": "restricted decoder distribution, not the original unrestricted sampling distribution",
                },
                "response_search": {
                    "method": "RASR + constrained native refinement" if response_refinement else "native RASR",
                    "stage": "native candidate refinement before native ranking; independent optional operator-token mask",
                    "runtime_patch_version": 1 if response_refinement else None,
                    "solver": "common explicit constrained coefficient fit" if response_refinement else "native curve_fit",
                    "execution": "serial; original native restart count retained",
                    "attempts": response_refinement.records if response_refinement else [],
                    "known_template_support": "not provided by this native refinement path",
                },
                "native_ledger": str(path),
                "input_rows": len(request.X),
                "native_input_dtype": "float32 before native train-only normalization",
                "transforms": "full precision stored native train statistics; fixed during common coefficient refit",
                "rejected_exports": rejected,
                "constraint_support": capabilities()["constraints"],
            },
            {"native_candidates": path, **({"response_refinement": response_refinement.record_file} if response_refinement else {})},
        )
    finally:
        fitted.close()
