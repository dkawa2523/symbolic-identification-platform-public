"""Native sampled-objective parity and search effect, not global guarantees."""
import numpy as np
import pytest

pytestmark = pytest.mark.native


def request():
    from ml_platform_symbolic.engines.base import DiscoveryRequest

    x = np.linspace(-1, 1, 30).reshape(-1, 1)
    return DiscoveryRequest(
        x, -x[:, 0], np.arange(30),
        [{"id": "x", "column": "x", "unit": "1", "domain": [-2, 2]}],
        {"column": "y", "unit": "1"}, ["add", "mul", "pow_int"],
        {"max_wall_seconds": 60, "max_nodes": 30, "require_units": False,
         "probes": [{"x": 0.0}], "response_bounds": [None, None],
         "response_checks": [{"feature": "x", "points": [{"x": 0.0}],
                              "order": 1, "sign": 1, "step": 0.01, "tolerance": 0.0}],
         "response_mode": "search"},
        42, {"objective": "mse", "family": "polynomial", "degree": 1, "maxiter": 5,
             "repetitions": 1, "seconds": 40},
    )


def test_native_data_objective_and_masked_gradient():
    import torch
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective
    from ml_platform_symbolic.response_search import compile_response_constraints
    from parfam.parfam_torch import Evaluator, ParFamTorch

    torch.set_default_dtype(torch.float64)
    r = request()
    compiled = compile_response_constraints(r)
    model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=1,
                        functions=[], function_names=[], degree_input_denominator=0,
                        degree_output_denominator=0, width=1, device="cpu", stabilize_denominator=False)
    c = torch.tensor([0.0, -0.5], requires_grad=True)
    x, y = torch.tensor(r.X), torch.tensor(r.y)
    prediction = model.predict(c, x)
    model.prepare_input_monomials(x)
    cache = model.output_monomials_dict
    objective = ResponseObjective(compiled)
    loss = objective.model_loss(prediction, y, model, c)
    assert model.output_monomials_dict is cache
    model.testing_mode()
    expected = float(torch.mean(((prediction-y)/compiled["data_scale"])**2).detach())
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(c.grad).all() and c.grad[1] > 0
    evaluator = Evaluator(x, y, model=model, lambda_0=0, lambda_1=0, n_params=2,
                          mask=torch.tensor([False, True]), custom_loss=objective)
    assert np.isfinite(evaluator.loss_func(np.array([-1.0])))
    assert np.isfinite(evaluator.gradient(None)).all()
    assert evaluator.evaluations > 0


def test_native_search_uses_response_objective_not_only_final_filter(tmp_path):
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover
    from ml_platform_symbolic.expressions import evaluate

    constrained = request()
    control = request()
    control.limits["response_mode"] = "filter"
    unconstrained = request()
    unconstrained.limits["probes"] = []
    unconstrained.limits["response_checks"] = []
    results = [discover(r, DiscoveryContext(tmp_path/name))
               for r, name in ((unconstrained, "none"), (control, "filter"), (constrained, "search"))]
    slopes = []
    for result in results:
        assert result.candidates
        values = [evaluate(c.equation, np.array([[-0.01], [0.01]]), constrained.variables)
                  for c in result.candidates]
        slopes.append([float((v[1]-v[0])/0.02) for v in values])
    assert results[0].summary["response_search"]["response_objective_calls"] == 0
    assert results[1].summary["response_search"]["response_objective_calls"] == 0
    assert results[2].summary["response_search"]["response_objective_calls"] > 0
    assert min(abs(v+1) for v in slopes[0]) < 0.05
    assert min(abs(v+1) for v in slopes[1]) < 0.05
    assert min(abs(v) for v in slopes[2]) < 0.05
    # Deliberately contradictory data and monotonic assumption: this checks
    # search influence, not that the fitted law is scientifically true.


def test_native_wrapper_preserves_nonstandard_request_coordinates(tmp_path, monkeypatch):
    """Scaling is external to ParFam; native fit and probe loss share request units."""
    import parfam.train as native_train
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective
    from ml_platform_symbolic.response_search import compile_response_constraints

    r = request()
    r.X = np.linspace(4.0, 7.0, 30).reshape(-1, 1)
    r.y = 30.0 + 2.0 * r.X[:, 0]
    r.variables[0]["domain"] = [3.0, 8.0]
    r.limits["probes"] = [{"x": 6.0}]
    r.limits["response_bounds"] = [39.0, 45.0]
    r.limits["response_checks"][0]["points"] = [{"x": 6.0}]
    expected = compile_response_constraints(r)
    native_fit = native_train.hyperparameter_training
    objective_loss = ResponseObjective.model_loss
    seen = {"fit": 0, "loss": 0}

    from functools import wraps

    @wraps(native_fit)
    def check_fit(x, y, *args, **kwargs):
        np.testing.assert_array_equal(np.asarray(x), r.X)
        np.testing.assert_array_equal(np.asarray(y), r.y)
        seen["fit"] += 1
        return native_fit(x, y, *args, **kwargs)

    def check_loss(self, prediction, target, model, coefficients):
        assert self.compiled == expected
        assert np.isin(target.detach().cpu().numpy(), r.y).all()
        seen["loss"] += 1
        return objective_loss(self, prediction, target, model, coefficients)

    monkeypatch.setattr(native_train, "hyperparameter_training", check_fit)
    monkeypatch.setattr(ResponseObjective, "model_loss", check_loss)
    result = discover(r, DiscoveryContext(tmp_path / "coordinates"))
    assert result.candidates and seen["fit"] > 0 and seen["loss"] > 0
    assert result.summary["normalization"] is False


def test_native_constrained_refit_respects_mask_and_rejects_infeasible():
    import torch
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective
    from parfam.parfam_torch import Evaluator, ParFamTorch
    from parfam.train import train_lbfgs

    torch.set_default_dtype(torch.float64)
    model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=1,
                        functions=[], function_names=[], degree_input_denominator=0,
                        degree_output_denominator=0, width=1, device="cpu", stabilize_denominator=False)
    x = torch.linspace(-1, 1, 20).reshape(-1, 1)
    compiled = {"data_scale": 1.0, "points": [[0.0]], "terms": [
        {"indices": [0], "coefficients": [1.0], "lower": 1.0, "upper": 1.5, "scale": 1.0, "name": "value"}]}
    objective = ResponseObjective(compiled)
    objective.enforce = True
    evaluator = Evaluator(x, torch.full((20,), 3.0), model=model, lambda_0=0, lambda_1=0,
                          n_params=2, mask=torch.tensor([True, False]), custom_loss=objective)
    fitted = train_lbfgs(evaluator, 1, 1, torch.tensor([0.0]))
    assert fitted is not None and 1.0 <= fitted[0] <= 1.5
    assert fitted[0] > 1.49
    # The fixed-zero intercept cannot satisfy a positive value at x=0.
    evaluator = Evaluator(x, torch.full((20,), 3.0), model=model, lambda_0=0, lambda_1=0,
                          n_params=2, mask=torch.tensor([False, True]), custom_loss=objective)
    assert train_lbfgs(evaluator, 1, 1, torch.tensor([0.0])) is None
    assert objective.infeasible_solves > 0



def test_native_finite_only_probes_have_no_empty_scipy_constraint(tmp_path):
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover

    r = request()
    r.limits["response_checks"] = []
    result = discover(r, DiscoveryContext(tmp_path / "finite_only"))
    assert result.candidates
    assert result.summary["response_search"]["constrained_solves"] > 0


def test_invalid_native_probe_fails_locally_and_next_solve_can_continue(monkeypatch):
    import torch
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective
    from parfam.parfam_torch import Evaluator, ParFamTorch

    model = ParFamTorch(n_input=1, degree_input_numerator=1, degree_output_numerator=1,
                        functions=[], function_names=[], degree_input_denominator=0,
                        degree_output_denominator=0, width=1, device="cpu", stabilize_denominator=False)
    x = torch.linspace(-1, 1, 20, dtype=torch.float64).reshape(-1, 1)
    compiled = {"data_scale": 1.0, "points": [[0.0]], "terms": [
        {"indices": [0], "coefficients": [1.0], "lower": 1.0, "upper": 1.5, "scale": 1.0, "name": "value"}]}
    objective = ResponseObjective(compiled)
    evaluator = Evaluator(x, torch.full((20,), 3.0), model=model, lambda_0=0, lambda_1=0,
                          n_params=2, mask=torch.tensor([True, False]), custom_loss=objective)
    original = model.predict

    def invalid_probe(coefficients, points):
        if len(points) == 1:
            return torch.full((1,), float("nan"), dtype=torch.float64)
        return original(coefficients, points)

    with monkeypatch.context() as context:
        context.setattr(model, "predict", invalid_probe)
        assert objective.constrained_fit(evaluator, 1, 20, [0.0]) is None
    with monkeypatch.context() as context:
        context.setattr(model, "predict", lambda coefficients, points: torch.zeros((len(points), 1)))
        with pytest.raises(ValueError, match="prediction shape"):
            objective.constrained_fit(evaluator, 1, 20, [0.0])
    assert objective.invalid_response_solves == 1
    assert objective.infeasible_solves == 1
    fitted = objective.constrained_fit(evaluator, 1, 20, [0.0])
    assert fitted is not None and 1.0 <= fitted[0] <= 1.5


def test_native_refit_keeps_reinitialization_count(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    import torch
    from parfam.train import train_lbfgs

    starts = []

    def solve(evaluator, n_params, iterations, initial):
        starts.append(initial)
        return None if len(starts) == 1 else np.array([1.25])

    evaluator = SimpleNamespace(custom_loss=SimpleNamespace(enforce=True, constrained_fit=solve),
                                device="cpu", loss_func=lambda coefficients: 0.25)
    fitted = train_lbfgs(evaluator, 1, maxiter=3, coefficients=torch.tensor([0.0]))
    assert len(starts) == 3 and starts[0] is not None and starts[1:] == [None, None]
    assert fitted.tolist() == [1.25]


def test_native_wrapper_overrides_ini_cap_and_observes_actual_row_usage(tmp_path):
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover

    r = request()
    r.X = np.linspace(-1, 1, 700)[:, None]
    r.y = 1 + 2*r.X[:, 0]
    r.row_ids = np.arange(700)
    r.limits.update(response_mode="filter", probes=[], response_checks=[])
    r.options.update(maxiter=1, seconds=15)
    result = discover(r, DiscoveryContext(tmp_path))
    assert result.candidates
    for row in result.summary["family_row_usage"]:
        assert row["supplied_train_rows"] == 700
        assert row["coefficient_fit_rows"] == 560
        assert row["internal_validation_rows"] == 140
        assert row["subsampled_out_rows"] == 0
        assert row["max_dataset_length"] == 700
    for row in result.summary["finetuning_row_usage"]:
        assert row["max_dataset_length"] == 5000
        assert row["coefficient_fit_rows"] == (row["supplied_rows"] if row["optimizer_started"] else 0)


def test_native_winner_reduction_observes_actual_5000_row_cap(tmp_path, monkeypatch):
    import parfam.train as native_train
    from ml_platform_symbolic.engines.base import DiscoveryContext
    from ml_platform_symbolic.engines.parfam import discover

    r = request()
    r.X = np.linspace(-1, 1, 7000)[:, None]
    r.y = 1 + 2*r.X[:, 0]
    r.row_ids = np.arange(7000)
    r.limits.update(response_mode="filter", probes=[], response_checks=[])
    r.options.update(degree=2, maxiter=1, seconds=15)
    # Controlled family coefficients force the real native pruning branch.
    # The downstream native split/subsample/fit and observer remain unchanged.
    monkeypatch.setattr(native_train, "training", lambda *args, **kwargs: np.array([1., 2., 1e-6]))
    result = discover(r, DiscoveryContext(tmp_path))
    actual = [record for record in result.summary["finetuning_row_usage"] if record["optimizer_started"]]
    assert actual
    assert all(row["supplied_rows"] == 5600 and row["coefficient_fit_rows"] == 5000 for row in actual)
    assert result.summary["family_row_usage"][0]["coefficient_fit_rows"] == 5600


def test_solver_outcomes_do_not_confuse_feasible_best_with_success():
    import torch
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective
    from parfam.parfam_torch import Evaluator,ParFamTorch
    torch.set_default_dtype(torch.float64)
    model=ParFamTorch(n_input=1,degree_input_numerator=1,degree_output_numerator=1,functions=[],function_names=[],degree_input_denominator=0,degree_output_denominator=0,width=1,device="cpu",stabilize_denominator=False)
    compiled={"data_scale":1.,"points":[[0.]],"terms":[{"indices":[0],"coefficients":[1.],"lower":1.,"upper":1.5,"scale":1.,"name":"anchor"}]}
    objective=ResponseObjective(compiled)
    x=torch.linspace(-1,1,10).reshape(-1,1)
    evaluator=Evaluator(x,torch.full((10,),3.),model=model,lambda_0=0,lambda_1=0,n_params=2,mask=torch.tensor([True,False]),custom_loss=objective)
    fitted=objective.constrained_fit(evaluator,1,1,torch.tensor([1.2]),0)
    assert fitted is not None and fitted[0]==pytest.approx(1.2)
    assert objective.constrained_solves==1 and getattr(objective,'infeasible_solves',0)==0
    counts=objective.solver_outcomes['minimize']
    assert counts=={'returned':1,'success':0,'not_success':1,'success_not_reported':0,'stop_reasons':{'iteration_limit':1}}
    evaluator.mask=torch.tensor([False,True])
    assert objective.constrained_fit(evaluator,1,1,torch.tensor([1.2]),0) is None
    assert objective.infeasible_solves==1
    assert sum(v['returned'] for v in objective.solver_outcomes.values())==2
