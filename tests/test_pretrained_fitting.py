from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines.flash_ansr import validate_request
from ml_platform_symbolic.engines.pretrained_fitting import fit_starts


def inputs():
    equation = {
        "schema_version": 2,
        "ast": {"op": "add", "args": [{"param": "a"}, {"const": 2}]},
        "parameters": {
            "a": {"value": 0.5, "trainable": True, "unit": "1", "bounds": [-1, 1]},
            "fixed": {"value": 7.0, "trainable": False, "unit": "1"},
        },
    }
    request = SimpleNamespace(X=np.zeros((4, 1)), y=np.full(4, 2.5), variables=[{"id": "x", "unit": "1"}], seed=42)
    return equation, request


def test_multistart_preserves_original_fixed_values_bounds_and_total_allocation():
    equation, request = inputs()
    original = deepcopy(equation)
    records = []

    def fitter(eq, *args, **kwargs):
        records.append((deepcopy(eq), kwargs["max_nfev"]))
        return eq

    audit = {}
    result = fit_starts(equation, request, {"fit_starts": 4, "max_nfev": 13}, 0, None, audit, fitter, None, 1)
    assert equation == original
    assert records[0][0] == original
    assert sum(budget for _, budget in records) == 13
    assert all(-1 <= eq["parameters"]["a"]["value"] <= 1 for eq, _ in records)
    assert all(
        eq["parameters"]["fixed"] == original["parameters"]["fixed"] and eq["ast"] == original["ast"]
        for eq, _ in records
    )
    assert result == original
    repeated = {}
    fit_starts(equation, request, {"fit_starts": 4, "max_nfev": 13}, 0, None, repeated, fitter, None, 1)
    assert audit == repeated
    changed = {}
    fit_starts(equation, request, {"fit_starts": 4, "max_nfev": 13}, 1, None, changed, fitter, None, 1)
    assert audit["starts"][1]["initial_parameters"] != changed["starts"][1]["initial_parameters"]


def test_single_start_keeps_original_call_and_diagnostics():
    equation, request = inputs()
    audit = {}

    def fitter(eq, *args, **kwargs):
        assert eq is equation
        assert kwargs["max_nfev"] == 13
        kwargs["diagnostics"]["original"] = True
        return eq

    fit_starts(equation, request, {"fit_starts": 1, "max_nfev": 13}, 0, None, audit, fitter, None, 1)
    assert audit == {"original": True}


def test_deadline_and_failed_start_remain_explicit():
    equation, request = inputs()
    with pytest.raises(TimeoutError):
        fit_starts(equation, request, {"fit_starts": 2, "max_nfev": 13}, 0, 0, {}, lambda *a, **k: equation, None, 1)
    audit = {}

    def fail(*args, **kwargs):
        raise ValueError("domain")

    with pytest.raises(ValueError, match="No finite feasible"):
        fit_starts(equation, request, {"fit_starts": 2, "max_nfev": 13}, 0, None, audit, fail, None, 1)
    assert len(audit["starts"]) == 2
    assert all(s["reason"] == "domain" for s in audit["starts"])


@pytest.mark.parametrize(
    "options",
    [{"refinement_restarts": 0}, {"refinement_restarts": True}, {"refinement_method": "all"}, {"scope": "all"}],
)
def test_flash_rejects_invalid_or_literal_thawing_settings(options):
    request = SimpleNamespace(X=np.zeros((3, 1)), options=options)
    with pytest.raises(ValueError):
        validate_request(request, None)


def test_flash_refinement_options_preflight_without_native_assets():
    request = SimpleNamespace(
        X=np.zeros((3, 1)), options={"refinement_restarts": 16, "refinement_method": "least_squares_trf"}
    )
    assert validate_request(request, None) == 64


def test_overflow_start_is_null_and_failed_without_blocking_later_start(monkeypatch):
    import json

    from ml_platform_symbolic.engines import pretrained_fitting as module

    equation, request = inputs()
    equation["parameters"]["a"].update(value=1e308, bounds=[None, None])
    request.y = np.full(4, 2.0)
    draws = iter([np.array([2.0]), np.array([-1.0])])
    monkeypatch.setattr(
        module.np.random, "default_rng", lambda seed: SimpleNamespace(normal=lambda **kwargs: next(draws))
    )
    audit = {}
    result = fit_starts(
        equation, request, {"fit_starts": 3, "max_nfev": 12}, 0, None, audit, lambda eq, *args, **kwargs: eq, None, 1
    )
    assert result["parameters"]["a"]["value"] == 0
    assert audit["starts"][0]["train_rmse"] == 1e308  # Squaring directly would overflow.
    assert audit["starts"][1]["status"] == "failed"
    assert audit["starts"][1]["initial_parameters"]["a"] is None
    assert audit["starts"][1]["train_rmse"] is None
    assert audit["selected_start"] == 2
    json.dumps(audit, allow_nan=False)


def test_nonrepresentable_score_and_solver_diagnostics_fail_but_next_start_runs():
    import json

    equation, request = inputs()
    request.y = np.full(4, -1e308)
    calls = []

    def fitter(eq, *args, **kwargs):
        calls.append(1)
        out = deepcopy(eq)
        out["parameters"]["a"].update(value=1e308 if len(calls) == 1 else -1e308, bounds=[None, None])
        return out

    audit = {}
    fit_starts(equation, request, {"fit_starts": 2, "max_nfev": 12}, 0, None, audit, fitter, None, 1)
    assert audit["starts"][0]["status"] == "failed"
    assert audit["starts"][0]["train_rmse"] is None
    assert audit["selected_start"] == 1
    json.dumps(audit, allow_nan=False)
    calls.clear()

    def diagnostic_fitter(eq, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            kwargs["diagnostics"]["cost"] = float("inf")
        return eq

    request.y = np.full(4, 2.5)
    audit = {}
    fit_starts(equation, request, {"fit_starts": 2, "max_nfev": 12}, 0, None, audit, diagnostic_fitter, None, 1)
    assert audit["starts"][0]["status"] == "failed"
    assert audit["starts"][0]["fit"]["cost"] is None
    assert audit["selected_start"] == 1
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("emission", ["fittable", "skeleton", "constants"])
def test_flash_official_emission_options_preflight(emission):
    request = SimpleNamespace(X=np.zeros((3, 1)), options={"emission": emission})
    assert validate_request(request, None) == 64


def test_flash_rejects_unknown_emission():
    request = SimpleNamespace(X=np.zeros((3, 1)), options={"emission": "implicit_rescale"})
    with pytest.raises(ValueError, match="emission"):
        validate_request(request, None)
