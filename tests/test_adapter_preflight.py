"""Input review must not need optional search libraries, assets, or fitting."""

import builtins
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.registry import ENGINES, get_engine


@pytest.fixture
def discovery_request():
    X = np.arange(1.0, 9.0).reshape(-1, 1)
    return DiscoveryRequest(
        X, X[:, 0] ** 2, np.arange(len(X)),
        [{"id": "x0", "column": "x0", "unit": "1", "domain": [1, 8]}],
        {"column": "y", "unit": "1"}, ["add", "mul", "div", "pow_int"],
        {"max_wall_seconds": 60, "max_nodes": 20}, 42, {},
    )


@pytest.fixture
def no_native_imports(monkeypatch):
    original = builtins.__import__
    forbidden = {"torch", "pysr", "physo", "TorchSisso", "ProGED", "parfam", "psrn", "pysips", "bingo",
                 "flash_ansr", "symbolicregression", "src"}

    def guarded(name, *args, **kwargs):
        level = kwargs.get("level", args[3] if len(args) > 3 else 0)
        if level == 0 and name.split(".")[0] in forbidden:
            raise AssertionError(f"Preflight attempted native import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("name", list(ENGINES))
def test_unknown_options_fail_in_review_and_discover_before_native_import(name, discovery_request, tmp_path, no_native_imports):
    module = get_engine(name)
    discovery_request.options = {"misspelled_budget": 2}
    context = DiscoveryContext(tmp_path / "not-created")
    for method in (module.validate_request, module.discover):
        with pytest.raises(ValueError, match="Unsupported native options"):
            method(discovery_request, context)
    assert not context.work_dir.exists()


@pytest.mark.parametrize("name", list(ENGINES))
def test_valid_preflight_requires_no_native_import_assets_or_search(name, discovery_request, tmp_path, no_native_imports):
    module = get_engine(name)
    if name == "known_form":
        discovery_request.options = {"equation": {"schema_version": 1, "ast": {"var": "x0"}, "parameters": {}}}
    elif name == "template_enumeration":
        from test_template_enumeration import request_for
        discovery_request = request_for(discovery_request.X, discovery_request.y)
    elif name in {"e2e_template", "flash_template", "rasr_template"}:
        from test_template_enumeration import request_for
        discovery_request = request_for(
            discovery_request.X, discovery_request.y, options={"max_proposals": 1}
        )
    if name in {"e2e", "e2e_template"}:
        discovery_request.operators = list(dict.fromkeys([*discovery_request.operators, "sub", "div"]))
        if name == "e2e_template":
            for slot in discovery_request.equation_template["slots"].values():
                slot["allowed_operators"] = list(dict.fromkeys([*slot["allowed_operators"], "sub", "div"]))
    before = deepcopy(discovery_request)
    context = DiscoveryContext(tmp_path / "not-created", assets_dir=None)
    module.validate_request(discovery_request, context)
    assert discovery_request.options == before.options
    assert discovery_request.equation_template == before.equation_template
    np.testing.assert_array_equal(discovery_request.X, before.X)
    np.testing.assert_array_equal(discovery_request.y, before.y)
    assert not context.work_dir.exists()


@pytest.mark.parametrize(("name", "options"), [
    ("dictionary", {"max_degree": True}), ("dictionary", {"max_terms": 1.5}),
    ("dictionary", {"ridge": float("nan")}), ("dictionary", {"ridge": True}),
    ("pysr", {"niterations": True}), ("pysr", {"seconds": float("nan")}),
    ("physo", {"batch_size": True}), ("physo", {"risk_factor": True}),
    ("torchsisso", {"n_term": True}), ("proged", {"sample_size": 1.5}),
    ("parfam", {"degree": True}), ("parfam", {"seconds": float("nan")}),
    ("pse", {"top_k": True}), ("pse", {"seconds": True}),
    ("pysips", {"param_init_bounds": [False, 2]}), ("pysips", {"constant_probability": True}),
    ("flash_ansr", {"draws": True}), ("rasr", {"candidates": True}),
    ("e2e", {"beam_size": True}),
])
def test_invalid_numeric_options_fail_without_native_import(name, options, discovery_request, tmp_path, no_native_imports):
    discovery_request.options = options
    with pytest.raises(ValueError):
        get_engine(name).validate_request(discovery_request, DiscoveryContext(tmp_path))


@pytest.mark.parametrize(("rows", "width", "device"), [(1025, 1, "cpu"), (2, 11, "cpu"), (2, 1, "cuda")])
def test_e2e_shape_and_device_preflight(rows, width, device, discovery_request, tmp_path, no_native_imports):
    discovery_request.X = np.ones((rows, width))
    discovery_request.y = np.ones(rows)
    with pytest.raises(NotImplementedError):
        get_engine("e2e").validate_request(discovery_request, DiscoveryContext(tmp_path, device))


def test_torch_response_preflight_preserves_hard_meaning(discovery_request, tmp_path, no_native_imports):
    from ml_platform_symbolic.engines import torchsisso
    request = discovery_request
    request.limits.update(probes=[{"x0": 4.0}], response_bounds=[0, 20], response_enforcement="hard")
    context = DiscoveryContext(tmp_path / "not-created")
    assert torchsisso.validate_request(request, context)[0] is False
    request.limits["response_enforcement"] = "soft"
    with pytest.raises(NotImplementedError, match="no conversion"):
        torchsisso.validate_request(request, context)
    request.limits["response_enforcement"] = "hard"
    request.equation_template = {"ast": {"const": 1}}
    with pytest.raises(NotImplementedError, match="dimensional=true"):
        torchsisso.validate_request(request, context)
    request.equation_template = None
    request.limits["response_bounds"] = [10, 0]
    with pytest.raises(ValueError, match="Contradictory"):
        torchsisso.validate_request(request, context)
    assert not context.work_dir.exists()


def test_physo_response_preflight_preserves_hard_meaning(discovery_request, tmp_path, no_native_imports):
    from ml_platform_symbolic.engines import physo
    request = discovery_request
    request.limits.update(probes=[{'x0':4.0}],response_bounds=[0,20],response_enforcement='hard')
    context=DiscoveryContext(tmp_path/'not-created')
    physo.validate_request(request,context)
    request.limits['response_enforcement']='soft'
    with pytest.raises(NotImplementedError,match='no conversion'):
        physo.validate_request(request,context)
    request.limits['response_enforcement']='hard'
    request.equation_template={'ast':{'const':1},'slots':{},'parameters':{}}
    with pytest.raises(NotImplementedError,match='exactly one shared slot'):
        physo.validate_request(request,context)
    request.equation_template=None
    request.limits['response_bounds']=[10,0]
    with pytest.raises(ValueError,match='Contradictory'):
        physo.validate_request(request,context)
    assert not context.work_dir.exists()


def test_preflight_guard_still_blocks_external_physo(no_native_imports):
    with pytest.raises(AssertionError,match='native import'):
        __import__('physo')
