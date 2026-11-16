"""Strict original bounds, even when optimized returns round just outside."""

import numpy as np
import pytest

pytestmark = pytest.mark.native


@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_feasible_lp_seed_does_not_suppress_inward_optimizer_retry(monkeypatch, retry_succeeds):
    import torch
    from ml_platform_symbolic.engines import torchsisso_response
    from ml_platform_symbolic.engines.torchsisso_budget import NodeBudget
    from ml_platform_symbolic.response_search import response_feasible
    from TorchSisso.Regressor import Regressor

    x = np.linspace(-1., 1., 100)[:, None]
    y = 2*x[:, 0]
    compiled = {"points": [[0.]], "data_scale": 1., "terms": [
        {"indices": [0], "coefficients": [1.], "lower": 2.9, "upper": 3.1,
         "scale": 1., "name": "value"}]}
    class FullAstResponse(torchsisso_response.ResponseSearch):
        def feasible_model(self, ids, intercept, coefficients):
            from ml_platform_symbolic.engines.torchsisso import _equation
            from ml_platform_symbolic.expressions import evaluate
            variables = [{"id": "x", "column": "x", "unit": "1"}]
            full = _equation([self.reg.names[int(i)] for i in ids], coefficients, intercept, variables, "1")
            return response_feasible(evaluate(full, np.asarray(compiled["points"]), variables), compiled)

    response = FullAstResponse(compiled, lambda names: np.zeros((1, len(names))))
    original = torchsisso_response.minimize
    calls = []

    def rounded_return(*args, **kwargs):
        result = original(*args, **kwargs)
        constraint = kwargs["constraints"][0]
        calls.append({"lower": constraint.lb.copy(), "upper": constraint.ub.copy()})
        if len(calls) <= 2 or not retry_succeeds:
            # Deterministically reproduce a successful solver's just-outside raw
            # coefficient return, independent of SciPy/BLAS boundary rounding.
            coefficient = result.x[1:]/response.xstd[[0]]
            b = 2.9+response.xmean[[0]]@coefficient
            while b-response.xmean[[0]]@coefficient >= 2.9:
                b = np.nextafter(b, -np.inf)
            result.x[0] = b
            result.success = True
        return result

    monkeypatch.setattr(torchsisso_response, "minimize", rounded_return)
    torch.set_num_threads(1)
    reg = Regressor(torch.tensor(x), torch.tensor(y), ["x"], dimension=1, sis_features=1,
                    node_budget=NodeBudget(9, lambda _: 1), response_search=response)
    reg.regressor_fit()
    assert len(calls) == 3  # old guard stopped after two because LP seed was feasible
    assert calls[2]["lower"][0] > 2.9
    assert calls[2]["upper"][0] < 3.1
    model = reg.portable_models_[0]
    assert response_feasible(np.array([model["intercept"]]), compiled)
    assert compiled["terms"][0]["lower"] == 2.9
    assert compiled["terms"][0]["upper"] == 3.1
    assert model["solver"]["solver"][-1]["inward_retry"]
    assert model["solver"]["feasible_seed_count"] >= 1
    assert all(s["success"] and not s["accepted_original_conditions"] for s in model["solver"]["solver"][:2])
    assert model["solver"]["solver"][-1]["accepted_original_conditions"] is retry_succeeds
    if retry_succeeds:
        assert model["solution_origin"] == "inward_optimized"
        assert model["optimizer_converged"]
        assert model["mse"] < 9.
        assert model["coefficients"][0] == pytest.approx(2., abs=1e-5)
    else:
        assert model["solution_origin"] == "feasible_seed"
        assert not model["optimizer_converged"]
        assert not model["solver"]["optimized_candidate_accepted"]
        assert model["coefficients"][0] == 0.
