"""Official Flash-ANSR candidate generation; strict native prefix/slot export."""

import json
import math
import os
import re
import time
from dataclasses import asdict
from fractions import Fraction
from importlib.metadata import version

import numpy as np

from ..expressions import UNARY_FUNCTIONS, inspect_equation
from .base import DiscoveryResult, checked_candidate, validate_options

BINARY = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
UNARY = {name: name for name in UNARY_FUNCTIONS}
VARIABLE = re.compile(r"x([1-9][0-9]*)\Z")


SUPPORTED_OPTIONS = {
    "draws",
    "emission",
    "operator_mask",
    "refinement_restarts",
    "refinement_method",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "operators": [*BINARY.values(), *UNARY, "pow_int", "pow", "rootn"],
        "requires_assets": True,
        "constraints": {
            "operators": "optional operator_mask restricts decoder operator tokens; context-dependent token lowering and portable full expression still post-filtered",
            "units": "native units unsupported; fitted coefficient units unknown, common post-filter",
            "max_nodes": "post-filtered",
            "variables": "width 1..17 enforced before native fit; padded references rejected",
            "probes": "post-filtered sampled checks, no global guarantee",
            "pretraining_overlap": "unknown; weights are inference-only, no fine-tuning",
        },
    }



def context_operator_routes(operators, masked):
    """Conservative token mask is not complete for context-dependent export lowering."""
    allowed = set(operators)
    return {
        "policy": "unchanged conservative token mapping; not complete decoder reachability or full generated grammar",
        "pow": {"token_allowed": not masked or "pow" in allowed,
                "possible_exports": ["pow", "pow_int"],
                "condition": "fixed integer exponent with abs<=8 exports pow_int; other exponents export pow; common checks still apply"},
        "rootn": {"token_allowed": not masked or "rootn" in allowed,
                  "possible_exports": ["rootn", "sqrt", "div", "identity"],
                  "condition": "fixed nonzero integer index required; 1/-1/2/-2 have special lowering; fitted or noninteger indices rejected"},
        "masked_compatible_routes": sorted({token for token, target in [("pow", "pow_int"), ("rootn", "sqrt"), ("rootn", "div")]
                                     if masked and token not in allowed and target in allowed}),
        "caveat": "masked routes may still arise through native simplification or other tokens; export capabilities do not guarantee a decoder route",
    }

def export_candidate(candidate, variables):
    tokens, positions, values = candidate.expression, candidate.slots, candidate.constants
    if not tokens or len(positions) != len(values) or len(set(positions)) != len(positions):
        raise ValueError("Native Flash prefix/slot layout mismatch")
    if any(type(p) is not int or p < 0 or p >= len(tokens) for p in positions):
        raise ValueError("Invalid Flash fitted slot position")
    parameters, fitted = {}, {}
    for index, (position, value) in enumerate(zip(positions, values, strict=True)):
        key = f"c{index}"
        token = tokens[position]
        if token not in {"<constant>", "np.pi", "np.e"}:
            try:
                float(Fraction(token))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError("Flash fitted slot references an operator/variable") from exc
        if not np.isfinite(value):
            raise ValueError("Flash fitted slot has nonfinite value")
        fitted[position] = key
        parameters[key] = {
            "value": float(value),
            "trainable": True,
            "bounds": [None, None],
            "unit": None,
            "origin": "native_fitted_slot",
        }
    cursor = 0

    def parse():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Truncated Flash prefix")
        position, token = cursor, tokens[cursor]
        cursor += 1
        if position in fitted:
            return {"param": fitted[position]}
        if token in BINARY:
            return {"op": BINARY[token], "args": [parse(), parse()]}
        if token in UNARY:
            return {"op": UNARY[token], "args": [parse()]}
        if token == "inv":
            return {"op": "div", "args": [{"const": 1}, parse()]}
        if token in {"pow", "rootn"}:
            base, exponent = parse(), parse()
            n = exponent.get("const")
            if token == "pow":
                op = "pow_int" if n is not None and int(n) == n and abs(n) <= 8 else "pow"
                return {"op": op, "args": [base, exponent]}
            if n is None or int(n) != n or n == 0:
                raise ValueError("Flash root index must be a fixed nonzero integer")
            if token == "rootn" and n in {1, -1, 2, -2}:
                value = base if abs(n) == 1 else {"op": "sqrt", "args": [base]}
                return value if n > 0 else {"op": "div", "args": [{"const": 1}, value]}
            return {"op": "rootn", "args": [base, exponent]}
        match = VARIABLE.fullmatch(token)
        if match:
            index = int(match.group(1)) - 1
            if index >= len(variables):
                raise ValueError("Flash padded variable is outside supplied feature width")
            return {"var": variables[index]["id"]}
        try:
            value = {"np.pi": math.pi, "np.e": math.e}.get(token)
            if value is None:
                value = float(Fraction(token))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Unsupported Flash operator/literal: {token}") from exc
        if not np.isfinite(value):
            raise ValueError("Nonfinite Flash literal")
        return {"const": value}

    equation = {"schema_version": 2, "ast": parse(), "parameters": parameters, "arithmetic": "numpy_ieee"}
    if cursor != len(tokens):
        raise ValueError("Trailing Flash prefix tokens")
    inspect_equation(equation, variables)
    return equation


def _diagnostic_json(value):
    if isinstance(value, dict):
        return {k: _diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_diagnostic_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    validate_options(request.options, SUPPORTED_OPTIONS)
    if not 1 <= request.X.shape[1] <= 17 or len(request.X) < 2:
        raise ValueError("Flash requires 1..17 features and at least two train rows; no truncation")
    if type(request.options.get("operator_mask", False)) is not bool:
        raise ValueError("Flash operator_mask must be boolean")
    if request.options.get("emission", "fittable") not in {"fittable", "skeleton", "constants"}:
        raise ValueError("Flash emission must be fittable, skeleton or constants")
    draws = request.options.get("draws", 64)
    if type(draws) is not int or draws < 1:
        raise ValueError("Flash draws must be a positive integer; external resource limits apply")
    restarts = request.options.get("refinement_restarts", 8)
    if type(restarts) is not int or restarts < 1:
        raise ValueError("Flash refinement_restarts must be a positive integer")
    if request.options.get("refinement_method", "curve_fit_lm") not in {"curve_fit_lm", "least_squares_trf"}:
        raise ValueError("Flash refinement_method must be curve_fit_lm or least_squares_trf")
    return draws


def discover(request, context):
    draws = validate_request(request, context)
    os.environ.update(
        XDG_CACHE_HOME=str(context.assets_dir / "cache/xdg"),
        HF_HOME=str(context.assets_dir / "cache/hf"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
    )
    from flash_ansr import FlashANSR, SoftmaxSamplingConfig

    started = time.monotonic()
    generation = SoftmaxSamplingConfig(draws=draws, emission=request.options.get("emission", "fittable"))
    model = FlashANSR.load(
        directory=str(context.assets_dir / "model"),
        generation_config=generation,
        refine={"n_restarts": request.options.get("refinement_restarts", 8),
                "method": request.options.get("refinement_method", "curve_fit_lm")},
        compute={"device": context.device, "workers": 0, "persistent_pool": False},
    )

    from .operator_mask import OperatorMask
    from .operator_mask import metadata as mask_metadata

    operator_mask = None
    if request.options.get("operator_mask", False):
        decoder = model.flash_ansr_model
        if not getattr(decoder, "symbolic_operator_mask_supported", False):
            raise ValueError("Flash runtime lacks the reviewed operator-mask hook; rebuild the pinned profile")
        mapping = {**BINARY, **{op: op for op in UNARY}, "inv": "div", "pow": "pow", "rootn": "rootn"}
        operator_mask = OperatorMask(decoder.tokenizer.token2idx,
            [token for token in decoder.tokenizer.token2idx if (model._get_operator_arity(token) or 0) > 0],
            mapping, request.operators)
        decoder.symbolic_operator_mask = operator_mask
    load_seconds = time.monotonic() - started
    result = model.fit(request.X.copy(), request.y.copy(), seed=request.seed, on_empty="raise")
    if result.n_points != len(request.X):
        raise ValueError("Flash effective train rows differ; native filtering/subsampling not accepted")
    from .flash_diagnostics import classify_failures, decode_expression

    raw_ledger = asdict(result.ledger)
    failure_diagnostics = classify_failures(
        raw_ledger,
        lambda raw: decode_expression(raw, model.tokenizer, model.simplipy_engine),
        model.simplipy_engine.is_valid,
    )
    ledger = context.work_dir / "flash_native.json"
    ledger.write_text(
        json.dumps(
            _diagnostic_json({"candidates": [asdict(c) for c in result.candidates], "ledger": raw_ledger, "failure_diagnostics": failure_diagnostics}),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    candidates, rejected = [], []
    for rank, native in enumerate(result.candidates):
        try:
            equation = export_candidate(native, request.variables)

            def predict(X, index=rank):
                return np.asarray(result.predict(X, rank=index)).reshape(-1)

            candidates.append(checked_candidate("flash_ansr", rank, equation, request, predict))
        except (ValueError, TypeError, OverflowError) as exc:
            rejected.append({"native_rank": rank, "reason": str(exc)})
    return DiscoveryResult(
        candidates,
        "ok" if candidates else "failed",
        {
            "version": version("flash-ansr"),
            "operator_mask": {**mask_metadata(operator_mask),
                              "score_semantics": "native log-probabilities retain raw pretrained scores before the operator mask; sampling uses restricted distribution"},
            "context_operator_routes": context_operator_routes(request.operators, operator_mask is not None),
            "export_arithmetic": "native NumPy IEEE parity; common real-domain invalid intermediates remain invalid",
            "source_revision": "bf2bdbc8502b58241057947ec53b91faae3ce835",
            "model_load_seconds": load_seconds,
            "native_generation_seconds": result.generation_time,
            "native_refinement_seconds": result.refinement_time,
            "native_refinement_config": model.refine.as_dict(),
            "coefficient_policy": "native fitted slots only; typed/fixed literal policy remains pinned native default; no common literal thawing",
            "native_draws": result.draws,
            "native_generation_config": generation.to_kwargs(),
            "native_initialization_policy": "fittable emits free placeholders; constants emits numerical literals. Native fittable slots with all emitted values try a verbatim start, then the pinned fallback restarts unless native acceptance criterion is reached; both fit sets retained. Mixed placeholder slots use native restart search. Typed-literal policy remains native refinement config.",
            "native_candidates": len(result.candidates),
            "native_initialization_evidence": {
                "survivor_rows_with_emitted_initial_values": sum(c.constants_emitted is not None for c in result.candidates),
                "survivor_rows_with_typed_frozen_literals": sum(bool(c.typed_frozen) for c in result.candidates),
                "survivor_rows_marked_typed_thaw": sum(c.typed_thaw is not None for c in result.candidates),
                "fallback_execution_count": None,
                "scope": "survivor records include native ladder variants, not counts of independent fits; pinned ledger does not expose per-candidate fallback executions",
            },
            "native_ledger_rows": len(result.ledger),
            "failure_diagnostics": {key: value for key, value in failure_diagnostics.items() if key != "rows"},
            "native_ledger": str(ledger),
            "diagnostic_nonfinite_encoding": "null; native fit_status preserved",
            "input_rows": len(request.X),
            "native_effective_rows": result.n_points,
            "rejected_exports": rejected,
            "constraint_support": capabilities()["constraints"],
        },
        {"native_candidates": ledger},
    )
