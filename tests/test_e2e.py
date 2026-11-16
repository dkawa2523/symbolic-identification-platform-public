"""E2E boundary/export checks; only the explicit native test loads official weights."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.e2e import discover, export_candidate
from ml_platform_symbolic.expressions import evaluate
from ml_platform_symbolic.fitting import fit_coefficients
from ml_platform_symbolic.model import load_model, save_model


def variables(width=1):
    return [{"id": f"x{i}", "column": f"x{i}", "unit": "1", "domain": [-3, 3]} for i in range(width)]


@pytest.mark.parametrize("limit", [None, 1100])
def test_explicit_row_budget_accepts_large_data_without_subsampling(limit, tmp_path):
    from ml_platform_symbolic.engines.e2e import validate_request

    X = np.arange(1100, dtype=float)[:, None]
    request = DiscoveryRequest(
        X, X[:, 0], np.arange(len(X)), variables(), {"unit": "1"}, ["add", "sub", "div"], {}, 1, {"max_train_rows": limit}
    )
    validate_request(request, DiscoveryContext(tmp_path))
    assert request.X is X and len(request.X) == 1100
    for invalid in (True, 0, 1, -1, 1024.0, "all"):
        request.options["max_train_rows"] = invalid
        with pytest.raises(ValueError, match="max_train_rows"):
            validate_request(request, DiscoveryContext(tmp_path))


@pytest.mark.native
def test_explicit_all_rows_native_bags_and_refinement(tmp_path):
    pytest.importorskip("symbolicregression")
    X = np.random.default_rng(731).uniform(-2, 2, (1100, 1))
    request = DiscoveryRequest(
        X,
        1.27 * X[:, 0] + 0.31,
        np.arange(len(X)),
        variables(),
        {"unit": "1"},
        ["add", "sub", "mul", "div"],
        {},
        731,
        {
            "max_train_rows": None,
            "beam_size": 1,
            "n_trees_to_refine": 2,
            "refinement_seconds": 0.5,
            "max_generated_output_len": 64,
            "operator_mask": True,
        },
    )
    result = discover(request, DiscoveryContext(tmp_path, assets_dir=Path("pretrained/e2e").resolve()))
    assert result.candidates and all(c.native_check["passed"] for c in result.candidates)
    usage = result.summary["native_row_usage"]
    assert usage["generation_calls"] == [{"bags": 6, "rows": 1100, "minimum_bag_rows": 100, "maximum_bag_rows": 200}]
    assert usage["refinement_calls"][0]["input_rows"] == usage["refinement_calls"][0]["target_rows"] == 1100


def test_full_precision_constants_fixed_integer_exponents_and_normalization_survive_refit(tmp_path):
    v = variables()
    equation = export_candidate("add,mul,1.234567891234567,pow2,x_0,2", v, [0.31], [1.91])
    X = np.linspace(-2, 2, 31).reshape(-1, 1)
    np.testing.assert_allclose(evaluate(equation, X, v), 1.234567891234567 * ((X[:, 0] - 0.31) / 1.91) ** 2 + 2)
    fitted = fit_coefficients(equation, X, 3.4 * ((X[:, 0] - 0.31) / 1.91) ** 2 + 2, v)
    assert fitted["parameters"]["c2"]["value"] == pytest.approx(3.4)
    for name in ("x0_mean", "x0_scale"):
        assert fitted["parameters"][name] == equation["parameters"][name]
    schema = {"variables": v, "target": {"unit": "1"}, "feature_ranges": [[-2, 2]]}
    path = save_model(tmp_path / "model.zip", equation, schema, {"engine": "e2e"})
    np.testing.assert_array_equal(load_model(path).predict(pd.DataFrame({"x0": X[:, 0]})), evaluate(equation, X, v))


@pytest.mark.parametrize("prefix", ["add,x_0", "x_0,1", "x_1", "pow,x_0,2.5", "nan", "CONSTANT_0", "pi"])
def test_invalid_or_unsupported_prefix_is_never_substituted(prefix):
    with pytest.raises(ValueError):
        export_candidate(prefix, variables(), [0], [1])


def test_repeated_variables_share_fixed_normalization_and_invalid_rows_remain():
    v = variables()
    equation = export_candidate("add,log,x_0,inv,x_0", v, [0], [1])
    assert len(equation["parameters"]) == 2
    result = evaluate(equation, np.array([[-1.0], [0.0], [1.0]]), v)
    assert np.isnan(result[:2]).all() and result[2] == 1


@pytest.mark.parametrize(
    ("rows", "width", "options", "device"),
    [
        (1025, 1, {}, "cpu"),
        (2, 11, {}, "cpu"),
        (1, 1, {}, "cpu"),
        (2, 1, {"max_input_points": 201}, "cpu"),
        (2, 1, {"max_generated_output_len": 1}, "cpu"),
        (2, 1, {"beam_size": True}, "cpu"),
        (2, 1, {"refinement_seconds": float("inf")}, "cpu"),
        (2, 1, {}, "cuda"),
    ],
)
def test_limits_reject_before_checkpoint_loading(rows, width, options, device, tmp_path):
    request = DiscoveryRequest(
        np.ones((rows, width)), np.ones(rows), np.arange(rows), variables(width), {"unit": "1"}, ["add"], {}, 1, options
    )
    with pytest.raises((ValueError, NotImplementedError)):
        discover(request, DiscoveryContext(tmp_path, device, tmp_path / "missing-assets"))


@pytest.mark.native
def test_real_e2e_native_export_and_shape_correct_refinement(tmp_path):
    pytest.importorskip("symbolicregression")
    assets = Path("pretrained/e2e")
    if not (assets / "manifest.json").is_file():
        pytest.skip("Run the explicit official E2E asset setup first")
    rng = np.random.default_rng(15)
    X = rng.uniform(-2, 2, (96, 2))
    y = 1.7 * X[:, 0] ** 2 + 0.8 * X[:, 1] + 0.3
    v = variables(2)
    request = DiscoveryRequest(
        X,
        y,
        np.arange(len(y)),
        v,
        {"unit": "1"},
        ["add", "sub", "mul", "div", "pow_int"],
        {"max_nodes": 100},
        42,
        {"beam_size": 8, "n_trees_to_refine": 8, "refinement_seconds": 3},
    )
    result = discover(request, DiscoveryContext(tmp_path, "cpu", assets))
    assert result.status == "ok" and result.candidates
    assert result.summary["native_effective_rows"] == len(X)
    assert result.summary["native_refinement_counts"]["BFGS"] > 0
    assert sum(result.summary["native_refinement_counts"].values()) == result.summary["native_candidates"]
    assert all(c.native_check["passed"] for c in result.candidates)
    Z = rng.uniform(-2, 2, (100, 2))
    truth = 1.7 * Z[:, 0] ** 2 + 0.8 * Z[:, 1] + 0.3
    errors = [np.sqrt(np.mean((evaluate(c.equation, Z, v) - truth) ** 2)) for c in result.candidates]
    # Behavior gate, not an exact expression recovery requirement.
    assert min(errors) < np.std(truth) * 0.15
    best = result.candidates[int(np.argmin(errors))]
    schema = {"variables": v, "target": {"unit": "1"}, "feature_ranges": [[-3, 3], [-3, 3]]}
    saved = save_model(tmp_path / "e2e.zip", best.equation, schema, {"native_check": best.native_check})
    np.testing.assert_array_equal(
        load_model(saved).predict(pd.DataFrame(Z, columns=["x0", "x1"])), evaluate(best.equation, Z, v)
    )


def test_pytorch_eos_mask_patch_is_narrow_and_rejects_unknown_source(tmp_path):
    from ml_platform_symbolic.engines.e2e_runtime import NEW, OLD, patch

    source = tmp_path / "model/transformer.py"
    source.parent.mkdir()
    source.write_text("before\n" + OLD + "\nafter\n")
    patch(tmp_path)
    patch(tmp_path)
    assert source.read_text() == "before\n" + NEW + "\nafter\n"
    source.write_text("unrecognized source")
    with pytest.raises(ValueError):
        patch(tmp_path)


@pytest.mark.native
def test_real_e2e_max_length_decode_finishes_without_candidate_fallback(tmp_path):
    pytest.importorskip("symbolicregression")
    assets = Path("pretrained/e2e")
    if not (assets / "manifest.json").is_file():
        pytest.skip("Run official E2E asset setup first")
    X = np.linspace(-2, 2, 24).reshape(-1, 1)
    request = DiscoveryRequest(
        X,
        1.7 * X[:, 0] ** 2 + 0.3,
        np.arange(len(X)),
        variables(),
        {"unit": "1"},
        ["add", "mul", "pow_int"],
        {},
        42,
        {"beam_size": 1, "max_generated_output_len": 2, "n_trees_to_refine": 1},
    )
    result = discover(request, DiscoveryContext(tmp_path, "cpu", assets))
    assert result.status == "failed" and not result.candidates
    assert result.summary["native_candidates"] == 0


def test_required_fixed_constant_is_explicitly_unsupported_without_loading_checkpoint(synthetic_config, tmp_path):
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import prepare, read_json
    from ml_platform_symbolic.engines.runner import run_discovery

    synthetic_config["search"].update(engines=["e2e"], allow_partial_engines=True)
    synthetic_config["data"]["fixed_constants"] = {"g": {"value": 9.80665, "unit": "1"}}
    cfg = resolve_config(synthetic_config)
    refs = prepare(cfg, tmp_path / "prepared")
    result = run_discovery(refs, read_json(refs["run"]), "e2e", 42, tmp_path / "native")
    summary = read_json(result["summary"])
    assert summary["status"] == "unsupported"
    assert summary["native_execution_status"] == "not_started"


def test_native_ieee_is_preserved_but_invalid_real_intermediates_remain_invalid():
    from ml_platform_symbolic.expressions import evaluate_native

    v = variables()
    equation = export_candidate("exp,mul,-1,inv,x_0", v, [0], [1])
    X = np.array([[0.0], [1.0]])
    raw = evaluate_native(equation, X, v)
    common = evaluate(equation, X, v)
    assert raw[0] == 0 and np.isnan(common[0])
    assert raw[1] == common[1] == pytest.approx(np.exp(-1))


@pytest.mark.parametrize("requirement", ["equation_template", "fixed_constants"])
def test_direct_adapter_rejects_unsupported_requirements_before_assets(requirement, tmp_path, monkeypatch):
    import ml_platform_symbolic.engines.e2e as adapter

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported requirements must fail before reading/loading assets")

    monkeypatch.setattr(adapter, "digest", forbidden)
    request = DiscoveryRequest(
        np.ones((2, 1)), np.ones(2), np.arange(2), variables(), {"unit": "1"}, ["add", "sub", "div"], {}, 1, {}
    )
    setattr(request, requirement, {"required": "explicit caller requirement"})
    with pytest.raises(NotImplementedError, match="E2E does not support"):
        discover(request, DiscoveryContext(tmp_path))


def test_direct_adapter_missing_assets_has_actionable_error(tmp_path):
    request = DiscoveryRequest(
        np.ones((2, 1)), np.ones(2), np.arange(2), variables(), {"unit": "1"}, ["add", "sub", "div"], {}, 1, {}
    )
    with pytest.raises(ValueError, match="verified local assets"):
        discover(request, DiscoveryContext(tmp_path))


def test_all_bags_keep_order_without_candidate_filtering():
    from ml_platform_symbolic.engines.e2e import generate_all_bags
    bags = [[i] * (200 if i < 28 else 134) for i in range(29)]
    calls = []
    def model(chunk):
        calls.append([bag[0] for bag in chunk])
        return [[bag[0], bag[0] + 100] for bag in chunk]
    records = []
    outputs = generate_all_bags(model, bags, 3, records)
    assert outputs == [[i, i + 100] for i in range(29)]
    assert sum(row['rows'] for row in records) == 5734
    assert sum(row['candidate_count'] for row in records) == 58
    assert [i for group in calls for i in group] == list(range(29))


def test_full_data_refinement_scope_restores_after_exception():
    from ml_platform_symbolic.engines.e2e import full_data_refinement
    class Refiner:
        def go(self, env, tree, coeffs0, X, y, downsample=1024, stop_after=10):
            assert downsample == -1
            self.X, self.y = X, y
            raise RuntimeError('native failed')
    original = Refiner.go
    records = []
    with pytest.raises(RuntimeError, match='native failed'):
        with full_data_refinement(Refiner, 1100, records):
            Refiner().go(None, None, [], [0]*1100, [0]*1100)
    assert Refiner.go is original
    assert records[0]['optimizer_rows'] == 1100


def test_full_data_refinement_rejects_actual_truncation():
    from ml_platform_symbolic.engines.e2e import full_data_refinement
    class Refiner:
        def go(self, env, tree, coeffs0, X, y, downsample=1024, stop_after=10):
            self.X, self.y = X[:1024], y[:1024]
    original = Refiner.go
    with pytest.raises(ValueError, match='effective optimizer'):
        with full_data_refinement(Refiner, 1100, []):
            Refiner().go(None, None, [], [0]*1100, [0]*1100)
    assert Refiner.go is original


def test_refinement_failure_before_optimizer_preserves_original_error():
    from ml_platform_symbolic.engines.e2e import full_data_refinement
    class Refiner:
        def go(self, env, tree, coeffs0, X, y, downsample=1024, stop_after=10):
            raise RuntimeError('unsupported native tree')
    original = Refiner.go
    observations = []
    with pytest.raises(RuntimeError, match='unsupported native tree'):
        with full_data_refinement(Refiner, 1100, observations):
            Refiner().go(None, None, [], [0]*1100, [0]*1100)
    assert Refiner.go is original
    assert observations[0]['row_status'] == 'not_started'
    assert observations[0]['native_error_type'] == 'RuntimeError'


@pytest.mark.parametrize('mode', ['missing', 'partial', 'short'])
def test_refinement_row_violation_survives_native_exception_catch(mode):
    from ml_platform_symbolic.engines.e2e import full_data_refinement
    class Refiner:
        def go(self, env, tree, coeffs0, X, y, downsample=1024, stop_after=10):
            if mode == 'partial':
                self.X = X
                raise RuntimeError('partial initialization')
            if mode == 'short':
                self.X, self.y = X[:1024], y[:1024]
            return None
    observations = []
    with full_data_refinement(Refiner, 1100, observations):
        try:
            Refiner().go(None, None, [], [0]*1100, [0]*1100)
        except Exception:
            pass  # Native per-candidate exception handling must not hide the violation.
    assert observations[0]['row_status'] == 'row_violation'


@pytest.mark.parametrize('allowed,forbidden', [(None, ['x_3']), (['x2'], ['x_0','x_1','x_3']), ([], ['x_0','x_1','x_2','x_3'])])
def test_variable_mask_uses_original_ids_and_padding(allowed, forbidden):
    torch = pytest.importorskip('torch')
    from ml_platform_symbolic.engines.e2e import VariableTokenMask
    tokens = {'x_0':0,'x_1':1,'x_2':2,'x_3':3,'add':4,'1':5}
    mask = VariableTokenMask(tokens, variables(3), allowed)
    scores = mask(None, torch.zeros((2,6)))
    assert mask.forbidden_tokens == forbidden
    assert mask.steps == 2
    assert all(torch.isneginf(scores[:,tokens[t]]).all() for t in forbidden)
    assert torch.isfinite(scores[:,4:]).all()
    if allowed == ['x2']:
        dead = torch.full((1,6), -float('inf'));dead[0,0] = 10
        with pytest.raises(ValueError, match='no finite'):
            mask(None, dead)


def test_variable_mask_rejects_unknown_id_and_missing_native_token():
    from ml_platform_symbolic.engines.e2e import VariableTokenMask
    with pytest.raises(ValueError, match='original variable IDs'):
        VariableTokenMask({'x_0':0}, variables(1), ['x2'])
    with pytest.raises(ValueError, match='vocabulary lacks'):
        VariableTokenMask({'x_0':0}, variables(3), ['x2'])


@pytest.mark.parametrize("mask", [None, True, False])
def test_e2e_normalization_operator_conflict_rejects_before_native(mask, tmp_path):
    from ml_platform_symbolic.engines.e2e import validate_request, capabilities
    from ml_platform_symbolic.response_search import constraint_application
    req = DiscoveryRequest(np.ones((3, 1)), np.ones(3), np.arange(3), variables(),
                           {"unit": "1"}, ["add"], {}, 17, {} if mask is None else {"operator_mask": mask})
    with pytest.raises(NotImplementedError, match="normalization"):
        validate_request(req, DiscoveryContext(tmp_path))
    req.operators = ["add", "sub", "div"]
    validate_request(req, DiscoveryContext(tmp_path))
    text = constraint_application({"engine": "e2e", "limits": {}, "options": req.options}, capabilities())["operators"]
    assert f"operator_mask={mask is not False}" in text
    req.operators = ["add"]
    req.limits["allowed_variables"] = []
    validate_request(req, DiscoveryContext(tmp_path))  # Explicit constant-only search needs no variable normalization.


@pytest.mark.native
def test_pinned_timedfun_keeps_finite_best_and_restores_after_exception():
    pytest.importorskip("symbolicregression")
    from symbolicregression.model import utils_wrapper as module
    from ml_platform_symbolic.engines.e2e import refinement_best_tracking
    original = module.TimedFun
    with pytest.raises(RuntimeError, match="scope probe"):
        with refinement_best_tracking(module) as calls:
            objective = module.TimedFun(lambda x: float(x[0] ** 2), stop_after=60)
            x = np.array([1.])
            objective.fun(x)
            x[0] = 99  # Native best must not alias a mutable optimizer input.
            objective.fun(np.array([2.]))
            assert objective.best_fun_value == 1
            np.testing.assert_array_equal(objective.best_x, [1.])
            objective.fun(np.array([.5]))
            assert objective.best_fun_value == .25
            invalid = module.TimedFun(lambda x: float("nan"), stop_after=60)
            invalid.fun(np.array([1.]))
            assert invalid.best_x is None and invalid.best_fun_value == np.inf
            objective.started -= 120
            with pytest.raises(ValueError, match="Time is over"):
                objective.fun(np.array([0.]))
            assert objective.best_fun_value == .25
            assert len(calls) == 2
            raise RuntimeError("scope probe")
    assert module.TimedFun is original
