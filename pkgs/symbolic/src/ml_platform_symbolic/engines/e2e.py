"""Pinned Meta E2E transformer inference and strict original-coordinate export."""

import inspect
import math
import random
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version

import numpy as np

from ..data import digest, write_json
from ..expressions import inspect_equation
from .base import DiscoveryResult, checked_candidate, validate_options
from .torch_runtime import thread_count, torch_thread_scope

SOURCE_REVISION = "c4144c99d078dd611795338e24fb3da49a32b9d8"
MODEL_SHA256 = "169569a0648ae1a4f1ac2bbb376fa5ebb70ebce4bfcf23d6666323596990ff91"
BINARY = {"add", "sub", "mul", "div"}
UNARY = {
    "abs": "abs",
    "sqrt": "sqrt",
    "log": "log",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
}


SUPPORTED_OPTIONS = {
    "threads",
    "response_max_nfev",
    "rational_domain_admission",
    "beam_size",
    "operator_mask",
    "max_generated_output_len",
    "max_input_points",
    "max_train_rows",
    "generation_batch_size",
    "n_trees_to_refine",
    "refinement_seconds",
}


def capabilities():
    return {
        "supported_options": sorted(SUPPORTED_OPTIONS),
        "rational_domain_admission": "opt-in source-interval divisor proof at SLSQP best-iterate retention and return; requires finite-point response search; not decoder conditioning",
        "response_enforcement": {"default": "hard", "supported": ['hard'], "stage": "exported candidate coefficient optimization; decoder not conditioned"},
        "operators": [*sorted(BINARY), *UNARY.values(), "pow_int"],
        "requires_assets": True,
        "equation_template": True,
        "fixed_constants": "template_only",
        "search_constraints": ["probes", "response_bounds", "response_checks"],
        "constraints": {
            "operators": "operator_mask defaults true and restricts decoder operator tokens; explicit false is post-check-only compatibility; portable full expression including normalization still post-filtered",
            "units": "native unit guidance unsupported; train normalization units preserved, coefficient units unknown",
            "max_nodes": "native prefix completion budget includes frozen source coordinates and fixed E2E normalization; export verification retained",
            "variables": "1..10 unchanged input columns; output variable-token masking",
            "allowed_variables": "enforced in native decoder variable tokens; original columns/IDs retained; padded tokens forbidden",
            "rows": "default resource guard 1024 rows; explicit max_train_rows=null uses all bags under outer wall/memory limits; no subsampling",
            "probes": "explicit finite-point constraints during SLSQP coefficient search over every exported structure; decoder not response-conditioned",
            "response_bounds": "finite-point hard bounds during coefficient optimization; no global guarantee",
            "response_checks": "finite-step conditions during coefficient optimization; not decoder conditioning or analytic derivative proof",
            "pretraining_overlap": "unknown; pretrained weights inference only, no fine-tuning",
            "equation_template": "limited exact affine observed slot h(X); numeric target reduction before native generation, original outer-AST assembly and coefficient fit afterwards; no template-conditioned decoder",
        },
    }


def export_candidate(prefix, variables, mean, scale):
    """Keep integer/operator literals fixed; decimal native coefficient slots may refit."""
    tokens = prefix.split(",") if isinstance(prefix, str) else list(prefix)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if mean.shape != (len(variables),) or scale.shape != mean.shape:
        raise ValueError("E2E normalization width mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("E2E normalization must be finite with positive scales")
    parameters, cursor = {}, 0

    def variable(index):
        if not 0 <= index < len(variables):
            raise ValueError("E2E native variable is outside supplied feature width")
        definition = variables[index]
        for suffix, value in (("mean", mean[index]), ("scale", scale[index])):
            parameters[f"x{index}_{suffix}"] = {
                "value": float(value),
                "trainable": False,
                "bounds": [None, None],
                "unit": definition["unit"],
                "origin": "train_normalization",
            }
        return {
            "op": "div",
            "args": [
                {"op": "sub", "args": [{"var": definition["id"]}, {"param": f"x{index}_mean"}]},
                {"param": f"x{index}_scale"},
            ],
        }

    def parse():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Truncated E2E prefix")
        position, token = cursor, str(tokens[cursor])
        cursor += 1
        if token in BINARY:
            return {"op": token, "args": [parse(), parse()]}
        if token in UNARY:
            return {"op": UNARY[token], "args": [parse()]}
        if token == "inv":
            return {"op": "div", "args": [{"const": 1}, parse()]}
        if token in {"pow2", "pow3"}:
            return {"op": "pow_int", "args": [parse(), {"const": int(token[-1])}]}
        if token.startswith("x_") and token[2:].isdigit():
            return variable(int(token[2:]))
        # General pow can contain fitted exponents and is deliberately unsupported.
        # Native decimal slots are exactly those eligible for upstream BFGS; integer
        # leaves are fixed by upstream function_to_skeleton(skeletonize_integers=False).
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Unsupported E2E operator/literal: {token}") from exc
        if not np.isfinite(value):
            raise ValueError("Nonfinite E2E literal")
        if token.lstrip("-").isdigit():
            return {"const": value}
        key = f"c{position}"
        parameters[key] = {
            "value": value,
            "trainable": True,
            "bounds": [None, None],
            "unit": None,
            "origin": "native_decimal_coefficient",
        }
        return {"param": key}

    # NumExpr uses IEEE arithmetic, including exp(-inf)==0. Preserve its raw
    # semantics for parity; the common evaluator still rejects invalid real
    # intermediates and keeps those rows in the shared constraint/score checks.
    equation = {"schema_version": 2, "ast": parse(), "parameters": parameters, "arithmetic": "numpy_ieee"}
    if cursor != len(tokens):
        raise ValueError("Trailing E2E prefix tokens")
    inspect_equation(equation, variables)
    return equation


def validate_request(request, context):
    """Check prepared Train input without importing or running the native engine."""
    # Direct adapter callers get the same explicit unsupported result as the runner.
    if request.equation_template is not None:
        from .e2e_template import prepare_known_affine
        proposed, _ = prepare_known_affine(request, context)
        return validate_request(proposed, context)
    if request.fixed_constants:
        raise NotImplementedError("E2E does not support named fixed physical constants; none are silently ignored")
    validate_options(
        request.options,
        SUPPORTED_OPTIONS,
    )
    row_limit = request.options.get("max_train_rows", 1024)
    if row_limit is not None and (type(row_limit) is not int or row_limit < 2):
        raise ValueError("E2E max_train_rows must be an integer >=2 or null for all rows under external budgets")
    if request.X.ndim == 2 and request.X.shape[1] > 10:
        raise NotImplementedError("E2E supports at most 10 columns; no silent feature selection")
    if row_limit is not None and len(request.X) > row_limit:
        raise NotImplementedError(
            "E2E exceeds configured max_train_rows resource guard; set an explicit limit or null, no silent subsampling"
        )
    if (
        request.X.ndim != 2
        or not 1 <= request.X.shape[1] <= 10
        or len(request.X) < 2
        or request.y.shape != (len(request.X),)
        or not np.isfinite(request.X).all()
        or not np.isfinite(request.y).all()
    ):
        raise ValueError("E2E requires finite X/y, 1..10 columns and at least 2 train rows; no silent subsampling")
    if len(request.variables) != request.X.shape[1]:
        raise ValueError("E2E variable definitions must match all supplied columns")
    if context.device != "cpu":
        raise NotImplementedError("This verified E2E profile supports CPU only")
    if type(request.options.get("operator_mask", True)) is not bool:
        raise ValueError("E2E operator_mask must be boolean")
    defaults = {"generation_batch_size": 1, "beam_size": 8, "max_input_points": 200, "n_trees_to_refine": 8, "max_generated_output_len": 128}
    options = {key: request.options.get(key, value) for key, value in defaults.items()}
    for key, value in options.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"E2E {key} must be a positive integer")
    if options["max_generated_output_len"] < 2:
        raise ValueError("E2E output length must include the initial and end tokens (at least 2)")
    if options["max_input_points"] > 200 or options["max_generated_output_len"] > 200:
        raise ValueError("E2E released checkpoint supports at most 200 points per bag and output tokens")
    if options["beam_size"] * options["max_generated_output_len"] > 100000:
        raise ValueError("E2E beam/output budget exceeds native batching support")
    seconds = request.options.get("refinement_seconds", 2.0)
    if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("E2E refinement_seconds must be positive and finite")
    budget = request.options.get("response_max_nfev", 1000)
    if type(budget) is not int or budget < 1:
        raise ValueError("E2E response_max_nfev must be a positive integer")
    if request.limits.get("allowed_variables") != [] and not {"sub", "div"} <= set(request.operators):
        raise NotImplementedError(
            "E2E nonconstant saved expressions require sub and div for fixed Train normalization; "
            "the common full-expression operator restriction forbids them. Include both operators "
            "or explicitly request no variable references; no normalization exemption or silent filtering."
        )
    from .e2e_response import prepare_response_fit
    compiled = prepare_response_fit(request)
    from .e2e_domain import prepare_admission
    prepare_admission(request, context, compiled)
    thread_count(request.options)
    return options, seconds


class VariableTokenMask:
    """Native positional variable tokens mapped to unchanged prepared variable IDs."""
    def __init__(self, token_ids, variables, allowed_variables=None):
        ids = [v['id'] for v in variables]
        if len(ids) != len(set(ids)):
            raise ValueError('E2E variable IDs must be unique')
        allowed = set(ids if allowed_variables is None else allowed_variables)
        if allowed - set(ids):
            raise ValueError('E2E allowed_variables must name prepared original variable IDs')
        self.mapping = {f'x_{i}': name for i, name in enumerate(ids)}
        if set(self.mapping) - set(token_ids):
            raise ValueError('Pinned E2E vocabulary lacks a supplied variable token')
        tokens = {t for t in token_ids if re.fullmatch(r'x_[0-9]+', t)}
        self.forbidden_tokens = sorted(t for t in tokens if self.mapping.get(t) not in allowed)
        self.padding_tokens = sorted(tokens - set(self.mapping))
        self.forbidden = sorted({token_ids[t] for t in self.forbidden_tokens})
        self.allowed_variables = sorted(allowed)
        self.steps = 0

    def __call__(self, prefixes, logits, remaining=None):
        import torch
        result = logits.clone()
        result[:, self.forbidden] = -float('inf')
        if not torch.isfinite(result).any(dim=1).all():
            raise ValueError('Variable mask leaves no finite next-token logits')
        self.steps += len(result)
        return result

    def metadata(self):
        return {'allowed_variables': self.allowed_variables, 'token_to_original_id': self.mapping,
                'forbidden_variable_tokens': self.forbidden_tokens, 'padding_tokens': self.padding_tokens,
                'processed_rows': self.steps,
                'scope': 'decoder variable tokens only; full original input columns remain encoder context; no unit/response/template guarantee'}


@contextmanager
def full_data_refinement(refiner_class, expected_rows, observations):
    """Pinned E2E BFGS uses every supplied row. Isolated worker only.

    Native refinement success/convergence is not implied by receiving all rows.
    The class hook is process-global and restored even after native exceptions.
    """
    original = refiner_class.go
    required = {"env", "tree", "coeffs0", "X", "y", "downsample", "stop_after"}
    if not required <= set(inspect.signature(original).parameters):
        raise ValueError("Unsupported pinned E2E BFGS refinement signature")

    def go(self, env, tree, coeffs0, X, y, downsample=-1, stop_after=10):
        if len(X) != expected_rows or len(y) != expected_rows:
            observations.append({"input_rows": len(X), "optimizer_rows": None, "optimizer_target_rows": None, "row_status": "row_violation", "native_error_type": None, "downsample": -1})
            raise ValueError("E2E coefficient optimizer must receive every Train row")
        completed = False
        error_type = None
        try:
            result = original(self, env, tree, coeffs0, X, y, downsample=-1, stop_after=stop_after)
            completed = True
            return result
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            has_X, has_y = hasattr(self, 'X'), hasattr(self, 'y')
            rows = len(self.X) if has_X else None
            targets = len(self.y) if has_y else None
            not_started = not completed and not has_X and not has_y
            valid = has_X and has_y and rows == expected_rows and targets == expected_rows
            status = 'not_started' if not_started else ('full_train' if valid else 'row_violation')
            observations.append({'input_rows': len(X), 'optimizer_rows': rows,
                                 'optimizer_target_rows': targets, 'downsample': -1,
                                 'row_status': status, 'native_error_type': error_type})
            if status == 'row_violation':
                raise ValueError('E2E effective optimizer tensors do not contain all Train rows')

    refiner_class.go = go
    try:
        yield
    finally:
        refiner_class.go = original


@contextmanager
def refinement_best_tracking(module):
    """Correct pinned native best-value bookkeeping inside an isolated worker only."""
    import hashlib
    from pathlib import Path
    expected = "070abf1b238d52dce25c913ca628168021da9e52346e926098e31e964e13f9a4"
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
        raise ValueError("Unreviewed E2E refinement source; best-value patch not applied")
    original = module.TimedFun
    calls = []

    class BestTimedFun(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            calls.append(self)

        def fun(self, x, *args):
            previous_x = None if self.best_x is None else np.array(self.best_x, copy=True)
            previous_value = self.best_fun_value
            value = super().fun(x, *args)  # Native clock, objective and history are unchanged.
            if np.isfinite(value) and (previous_x is None or value < previous_value):
                self.best_fun_value = value
                self.best_x = np.array(x, copy=True)
            else:
                self.best_fun_value, self.best_x = previous_value, previous_x
            return value

    module.TimedFun = BestTimedFun
    try:
        yield calls
    finally:
        module.TimedFun = original


def generate_all_bags(model, inputs, batch_size, observations):
    """Preserve bag order and every output; sampling RNG ordering depends on batch size."""
    outputs = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start:start + batch_size]
        result = model(chunk)
        if len(result) != len(chunk):
            raise ValueError("E2E must return an output list for every supplied bag")
        outputs.extend(result)
        observations.append({"first_bag": start, "bags": len(chunk),
                             "rows": sum(len(bag) for bag in chunk), "candidate_count": sum(map(len, result))})
    return outputs


def discover(request, context):
    if request.equation_template is not None:
        from .e2e_template import discover_known_affine
        return discover_known_affine(request, context)
    validate_request(request, context)
    if context.assets_dir is None:
        raise ValueError("E2E requires verified local assets; run explicit asset setup first")
    with torch_thread_scope(request.options):
        return _discover(request, context)


def _discover(request, context):
    options, seconds = validate_request(request, context)
    checkpoint = context.assets_dir / "model1.pt"
    if digest(checkpoint) != MODEL_SHA256:
        raise ValueError("E2E requires the exact reviewed official model1.pt checkpoint")

    import torch
    from sklearn.preprocessing import StandardScaler
    from symbolicregression.model import SymbolicTransformerRegressor
    from symbolicregression.model import utils_wrapper as refinement_module
    BFGSRefinement = refinement_module.BFGSRefinement

    torch.manual_seed(request.seed)
    np.random.seed(request.seed)
    random.seed(request.seed)
    start = time.monotonic()
    # Official release is a whole-model pickle, never load arbitrary user checkpoints.
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.to("cpu").eval()
    model.device = torch.device("cpu")
    model.env.params.cpu = True
    if model.env.params.max_input_dimension != 10 or model.env.params.max_output_dimension != 1:
        raise ValueError("Unexpected E2E checkpoint dimensions")
    model.beam_type = "sampling"
    model.beam_size = options["beam_size"]
    model.max_generated_output_len = options["max_generated_output_len"]
    from ..coordinates import source_complexity_costs, source_equation
    from .e2e_prefix import PrefixMask
    from .operator_mask import OperatorMask
    from .operator_mask import metadata as mask_metadata

    variable_mask = VariableTokenMask(model.env.equation_word2id, request.variables, request.limits.get("allowed_variables"))
    if not getattr(model.decoder, "symbolic_operator_mask_supported", False):
        raise ValueError("E2E runtime lacks the reviewed decoder-mask hook; rebuild the pinned profile")
    if getattr(model.decoder, "symbolic_prefix_mask_version", None) != 2:
        raise ValueError("E2E requires pinned decoder prefix hook v2; rebuild the e2e runtime image")
    coordinate_context = {**context.provenance, "variables": request.variables,
                          "target": {"column": "target", **request.target}}
    source_costs, output_cost = source_complexity_costs(coordinate_context)
    max_nodes = request.limits.get("max_nodes")
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < output_cost + 1):
        raise ValueError("E2E max_nodes cannot fit the source-coordinate output expression")
    native_variable_costs = {f"x_{i}": source_costs[v["id"]] + 4 for i, v in enumerate(request.variables)}
    prefix_mask = PrefixMask(model.env, max_nodes, variable_costs=native_variable_costs,
                             output_cost=output_cost, supported_operators=BINARY | set(UNARY) | {"inv", "pow2", "pow3"})
    operator_mask = None
    if request.options.get("operator_mask", True):
        if not getattr(model.decoder, "symbolic_operator_mask_supported", False):
            raise ValueError("E2E runtime lacks the reviewed operator-mask hook; rebuild the pinned profile")
        mapping = {**{op: op for op in BINARY}, **UNARY, "inv": "div", "pow2": "pow_int", "pow3": "pow_int"}
        operator_mask = OperatorMask(
            model.env.equation_word2id,
            [op for op, arity in model.env.equation_encoder.all_operators.items() if arity > 0],
            mapping,
            request.operators,
        )

    def decoder_mask(prefixes, logits, remaining=None):
        if operator_mask is not None:
            logits = operator_mask(prefixes, logits, remaining)
        logits = variable_mask(prefixes, logits, remaining)
        return prefix_mask(prefixes, logits, remaining)

    model.decoder.symbolic_operator_mask = decoder_mask

    load_seconds = time.monotonic() - start
    scaler = StandardScaler().fit(request.X)
    scaled = scaler.transform(request.X)
    generation_calls, refinement_calls, generation_batches, optimizer_calls = [], [], [], []

    class ObservedGeneration:
        # Preserve native bags; bound simultaneous generation without dropping any bag.
        def __getattr__(self, name):
            return getattr(model, name)

        def __call__(self, inputs):
            sizes = [len(bag) for bag in inputs]
            generation_calls.append(
                {
                    "bags": len(sizes),
                    "rows": sum(sizes),
                    "minimum_bag_rows": min(sizes, default=0),
                    "maximum_bag_rows": max(sizes, default=0),
                }
            )
            if sum(sizes) != len(request.X):
                raise ValueError("E2E native generation did not receive all Train rows")
            return generate_all_bags(model, inputs, options["generation_batch_size"], generation_batches)

    fitted = SymbolicTransformerRegressor(
        model=ObservedGeneration(),
        max_input_points=options["max_input_points"],
        max_number_bags=-1,
        n_trees_to_refine=options["n_trees_to_refine"],
        stop_refinement_after=seconds,
        rescale=False,
    )
    native_refine = fitted.refine

    def observed_refine(X, y, trees, *args, **kwargs):
        refinement_calls.append({"input_rows": len(X), "target_rows": len(y), "proposed_trees": len(trees)})
        if len(X) != len(request.X) or len(y) != len(request.y):
            raise ValueError("E2E native refinement did not receive all Train rows")
        return native_refine(X, y, trees, *args, **kwargs)

    fitted.refine = observed_refine
    start = time.monotonic()
    with full_data_refinement(BFGSRefinement, len(request.X), optimizer_calls), refinement_best_tracking(refinement_module) as best_calls:
        fitted.fit(scaled.copy(), request.y.copy().reshape(-1, 1))
    if any(row["row_status"] == "row_violation" for row in optimizer_calls):
        raise ValueError("E2E coefficient refinement silently changed Train rows")
    fit_seconds = time.monotonic() - start
    if fitted.top_k_features != [list(range(request.X.shape[1]))]:
        raise ValueError("E2E unexpectedly selected/reordered input features")
    candidates, rejected, records = [], [], []
    for rank, native in enumerate(fitted.tree.get(0) or []):
        tree = native["predicted_tree"]
        prefix = tree.prefix()
        records.append({"native_rank": rank, "prefix": prefix, "refinement_type": native["refinement_type"]})
        try:
            equation = export_candidate(prefix, request.variables, scaler.mean_, scaler.scale_)
            public_equation, public_variables = source_equation(equation, coordinate_context)
            public_nodes = inspect_equation(public_equation, public_variables)["n_nodes"]
            if max_nodes is not None and public_nodes > max_nodes:
                raise RuntimeError("E2E generated expression violated its source-coordinate search budget")
            oracle = model.env.simplifier.tree_to_numexpr_fn(tree)
            candidate = checked_candidate(
                "e2e",
                rank,
                equation,
                request,
                lambda X, predict=oracle: np.asarray(predict(scaler.transform(X)))[:, 0],
            )
            candidate.native_check["refinement_type"] = native["refinement_type"]
            candidate.native_check["source_n_nodes"] = public_nodes
            candidates.append(candidate)
        except (ValueError, TypeError, OverflowError) as exc:
            rejected.append({"native_rank": rank, "reason": str(exc)})
    native_portable = None
    from .e2e_response import fit_responses, prepare_response_fit
    response = None
    compiled = prepare_response_fit(request)
    if compiled is not None:
        native_portable = write_json(context.work_dir / "e2e_original_candidates.json", [asdict(c) for c in candidates])
        derived, response = fit_responses(candidates, request, context, compiled)
        # Explicit admission applies to every returned candidate. Native originals
        # remain archived, never an unproved fallback around the opt-in policy.
        candidates = (derived if request.options.get("rational_domain_admission") is not None
                      else candidates + derived)
    ledger = write_json(
        context.work_dir / "e2e_native.json",
        {
            "candidates": records,
            "X_mean": scaler.mean_.tolist(),
            "X_scale": scaler.scale_.tolist(),
            "native_predictor": "pinned env.simplifier.tree_to_numexpr_fn; avoids upstream sklearn predict keyword bug",
        },
    )
    return DiscoveryResult(
        candidates,
        "partial" if response is not None and not response["complete"] else "ok" if candidates else "failed",
        {
            "source_revision": SOURCE_REVISION,
            "response_fit": response,
            "native_best_tracking": {
                "implementation": "pinned TimedFun finite-best correction; independent worker scope",
                "optimizer_calls": len(best_calls),
                "calls_with_finite_best": sum(call.best_x is not None for call in best_calls),
                "scope": "best objective value actually observed, not solver convergence; NoRef retained",
            },
            "operator_mask": {
                **mask_metadata(operator_mask),
                "score_semantics": "beam scores retain raw pretrained log-probability; prefix grammar/variable/size masks restrict sampling; operator restriction follows enabled flag; adapter does not export sampling log-probabilities",
            },
            "runtime_patch": "PyTorch2 EOS boolean mask; corrected finite best-value tracking; bounded bag batches alter sampling RNG ordering; BFGS uses full Train instead of prefix1024; scalar grammar and source AST budget restrict generation; pretrained weights unchanged",
            "code_license": "Apache-2.0",
            "model_license": "no separate weights license confirmed; official download only",
            "torch": torch.__version__,
            "scikit_learn": version("scikit-learn"),
            "sympytorch": version("sympytorch"),
            "numexpr": version("numexpr"),
            "model_sha256": MODEL_SHA256,
            "torch_threads": torch.get_num_threads(),
            "model_load_seconds": load_seconds,
            "native_fit_seconds": fit_seconds,
            "native_candidates": len(records),
            "native_refinement_counts": {
                kind: sum(record["refinement_type"] == kind for record in records) for kind in ("NoRef", "BFGS")
            },
            "refinement_failures": (
                "Upstream suppresses individual BFGS failure reasons; zero BFGS outputs is not refinement success. "
                "Native NoRef candidates are retained independently."
            ),
            "rejected_exports": rejected,
            "input_rows": len(request.X),
            "native_effective_rows": len(request.X),
            "bags": sum(call["bags"] for call in generation_calls),
            "variable_mask": variable_mask.metadata(),
            "complexity_budget": {"max_nodes": max_nodes, "coordinate_scope": "source numerical expression AST",
                                  "variable_token_costs": native_variable_costs, "output_cost": output_cost,
                                  "processed_rows": prefix_mask.rows, "masked_token_choices": prefix_mask.masked_choices,
                                  "scope": "generation grammar and source AST size; not physical dimensions or response constraints"},
            "native_row_usage": {"generation_calls": generation_calls, "generation_batches": generation_batches,
                                 "refinement_calls": refinement_calls, "optimizer_calls": optimizer_calls},
            "max_train_rows": request.options.get("max_train_rows", 1024),
            "effective_search_options": {
                **options,
                "refinement_seconds": seconds,
                "beam_type": "sampling",
                "max_number_bags": -1,
                "supplied_features": request.X.shape[1],
            },
            "budget_semantics": "beam_size is native samples per bag plus greedy, not beam search width; all bags generated and all Train rows used for refinement; refinement_seconds is native per-tree refinement limit",
            "generation": "seeded native sampling plus greedy candidate; pretrained weights unchanged; batch-size changes can change RNG consumption and candidates",
            "transforms": "train-only StandardScaler; fixed original-coordinate mean/scale; no target scaling",
            "coefficient_policy": "native decimal slots free; integer/operator literals and normalization fixed",
            "constraint_support": capabilities()["constraints"],
        },
        {"native_candidates": ledger, **({"native_original_candidates": native_portable} if native_portable is not None else {})},
    )
