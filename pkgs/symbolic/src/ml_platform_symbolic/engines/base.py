from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..domains import expression_bounds, region_points
from ..expression_graph import expanded_equation
from ..expressions import evaluate, evaluate_native
from ..numerical_fidelity import compare_predictions, signal_scale


@dataclass
class DiscoveryRequest:
    X: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    variables: list[dict]
    target: dict
    operators: list[str]
    limits: dict
    seed: int
    options: dict
    fixed_constants: dict = field(default_factory=dict)
    domain: dict | None = None
    equation_template: dict | None = None


@dataclass
class DiscoveryContext:
    work_dir: Path
    device: str = "cpu"
    assets_dir: Path | None = None
    provenance: dict = field(default_factory=dict)
    checkpoint: Callable | None = None
    deadline: float | None = None


@dataclass
class Candidate:
    candidate_id: str
    engine: str
    seed: int
    equation: dict
    native_check: dict


@dataclass
class DiscoveryResult:
    candidates: list[Candidate]
    status: str
    summary: dict
    native_artifacts: dict = field(default_factory=dict)


def _check_ieee_compositions(equation, variables):
    """Only permit dangerous compositions when simple task bounds resolve the pole."""

    equation = expanded_equation(equation)

    def visit(node):
        children = node.get("args", [])
        risks = [visit(child) for child in children]
        op = node.get("op")
        if op in {"exp", "tanh", "atan"} and any(risks):
            raise ValueError(
                "Unsupported IEEE-to-finite composition: domain bounds do not exclude an inner singularity"
            )
        if op in {"pow", "pow_int", "rootn"} and risks[0]:
            low, high = expression_bounds(children[1], equation["parameters"], variables)
            if low <= 0:
                raise ValueError("Unsupported IEEE-to-finite power of an undefined inner expression")
        if op in {"pow", "pow_int", "rootn"}:
            exponent_low, _ = expression_bounds(children[1], equation["parameters"], variables)
            base_low, base_high = expression_bounds(children[0], equation["parameters"], variables)
            bounds = expression_bounds(node, equation["parameters"], variables)
            return any(risks) or (exponent_low < 0 and base_low <= 0 <= base_high) or not all(np.isfinite(bounds))
        if op == "div":
            if risks[1]:
                raise ValueError("Unsupported nested reciprocal IEEE semantics")
            low, high = expression_bounds(children[1], equation["parameters"], variables)
            bounds = expression_bounds(node, equation["parameters"], variables)
            return any(risks) or low <= 0 <= high or not all(np.isfinite(bounds))
        if op in {"add", "sub", "mul", "sinh", "cosh"}:
            return any(risks) or not all(np.isfinite(expression_bounds(node, equation["parameters"], variables)))
        if op == "log":
            low, high = expression_bounds(children[0], equation["parameters"], variables)
            return any(risks) or low <= 0 <= high
        if op == "exp":
            _, high = expression_bounds(children[0], equation["parameters"], variables)
            return any(risks) or high > np.log(np.finfo(float).max)
        if op == "atanh":
            low, high = expression_bounds(children[0], equation["parameters"], variables)
            return any(risks) or low <= -1 <= high or low <= 1 <= high
        return any(risks)

    visit(equation["ast"])


def checked_candidate(
    engine, index, equation, request, native_predict, *, native_ieee=True, parity_scale=None
):
    if native_ieee and equation.get("arithmetic", "strict_real") == "strict_real":
        from ..domains import domain_variables

        _check_ieee_compositions(equation, domain_variables(request.variables, getattr(request, "domain", None)))
    points = np.random.default_rng(request.seed).permutation(len(request.X))[:64]
    X = request.X[points]
    domain = getattr(request, "domain", None)
    extra = region_points(request.X, request.variables, request.seed, domain, request.limits.get("probes", []))
    X = np.vstack([X, extra])
    native = np.broadcast_to(np.asarray(native_predict(X), dtype=float), (len(X),))
    portable = evaluate_native(equation, X, request.variables)
    scale = signal_scale(request.y) if parity_scale is None else float(parity_scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Native/export numerical comparison scale must be nonnegative and finite")
    check = compare_predictions(native, portable, scale=scale)
    if not check["passed"]:
        if check["resolution_limited"]:
            raise ValueError("Float64 precision cannot resolve the requested native/export signal tolerance")
        raise ValueError("Native/export predictions do not agree on signal-scaled numerical/IEEE boundaries")
    check.update(
        {
            "train_points": len(points),
            "boundary_and_configured_points": len(extra),
            "scope": "train plus independent declared-domain/interior/boundary/configured probes; no global equivalence proof",
            "division_policy": (
                "explicit NumPy IEEE raw parity; common prediction separately masks invalid real intermediates"
                if equation.get("arithmetic") == "numpy_ieee"
                else "strict real; invalid intermediates propagate; checks respect declared input domain"
            )
            if native_ieee
            else "common evaluator oracle",
        }
    )
    check.update(
        domain_probe_policy=domain.get("kind") if domain else "declared box/probe policy",
        arithmetic=equation.get("arithmetic", "strict_real"),
        comparison_semantics="declared native arithmetic, including signed Inf/NaN masks",
        common_prediction_semantics="finite real-domain projection; invalid intermediate rows retained separately",
        common_real_domain_valid_points=int(np.isfinite(evaluate(equation, X, request.variables)).sum()),
    )
    return Candidate(f"{engine}/{request.seed}/{index}", engine, request.seed, equation, check)


def validate_options(options, allowed):
    unknown = set(options) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported native options: {sorted(unknown)}; allowed: {sorted(allowed)}")


def native_input_indices(request, context=None):
    """Select native columns only; common coordinates, domain and IDs remain full."""
    ids = [v["id"] for v in request.variables]
    allowed = request.limits.get("allowed_variables")
    if allowed is not None and (not isinstance(allowed, list) or any(v not in ids for v in allowed)):
        raise ValueError("allowed_variables must contain existing variable IDs")
    if allowed is not None and context is not None and (context.provenance.get("input_geometry") or {}).get("map"):
        raise NotImplementedError("Source variable restrictions with mapped inputs require an explicit mapped-expression audit")
    indices = [i for i, name in enumerate(ids) if allowed is None or name in allowed]
    if not indices:
        raise NotImplementedError("This native search needs at least one allowed input; constant-only search is unsupported")
    if len(indices) != len(ids) and request.equation_template is not None:
        raise NotImplementedError("Input-subset search with an equation template is unsupported")
    return indices
