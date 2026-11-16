"""Pinned native local-fit failures are not converged or completed candidates."""
import numpy as np
import pytest

pytestmark = pytest.mark.native


def runtime():
    torch = pytest.importorskip('torch')
    train = pytest.importorskip('parfam.train')
    tensor = pytest.importorskip('parfam.parfam_torch')
    assert train.symbolic_handoff_fit_failure_version == 1
    return torch, train, tensor


def evaluator(*, guarded=True, isolated=False):
    torch, train, tensor = runtime()
    from ml_platform_symbolic.engines.parfam_objective import ResponseObjective

    class Model:
        device = 'cpu'

        def predict(self, c, x):
            # sqrt(c^2) is finite at zero but its autodiff gradient is undefined.
            value = torch.sqrt(-c[0] ** 2 if isolated else c[0] ** 2)
            return value + c[1] * x[:, 0]

        def testing_mode(self):
            pass

        def prepare_input_monomials(self, x):
            pass

        def get_number_parameters(self):
            return 2

        def get_normalized_coefficients(self, c):
            return c

    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    objective = ResponseObjective({'data_scale': 1})
    objective.reject_nonfinite_gradients = guarded
    return tensor.Evaluator(x, torch.ones(2, dtype=torch.float64), Model(), 0, 0, 2,
                            custom_loss=objective)


def test_finite_value_nonfinite_gradient_raises_without_repair():
    torch, _, tensor = runtime()
    e = evaluator()
    point = np.array([0., 0.])
    assert e.loss_func(point) == 1
    with pytest.raises(tensor.NonfiniteGradientError):
        e.gradient(point)
    np.testing.assert_array_equal(e.predict().detach().numpy(), [0., 0.])
    ordinary = evaluator(guarded=False)
    ordinary.loss_func(point)
    assert np.isnan(ordinary.gradient(point)[0])  # Opt-in does not silently change ordinary native behavior.


def test_lbfgs_direct_backward_failure_is_not_swallowed():
    torch, train, tensor = runtime()
    e = evaluator()
    with pytest.raises(tensor.NonfiniteGradientError):
        e.fit_lbfgs(torch.zeros(2, dtype=torch.float64), verbose=False)
    assert train.train_lbfgs(e, 2, 1, coefficients=torch.zeros(2, dtype=torch.float64)) is None
    assert e.custom_loss.nonfinite_gradient_fits == 1


def test_basinhopping_retains_other_hops_and_all_invalid_returns_none():
    _, train, _ = runtime()
    e = evaluator()
    np.random.seed(17)
    result = train.training(e.model, e, e.x, e.y, 2, 20, 50000, 10000, False,
                            'basinhopping', 100, 'BFGS', x0=np.zeros(2))
    assert result is not None and np.isfinite(result).all()
    assert np.isfinite(e.loss_func(np.asarray(result)))
    assert e.custom_loss.nonfinite_gradient_fits >= 1
    bad = evaluator(isolated=True)
    result = train.training(bad.model, bad, bad.x, bad.y, 2, 0, 50000, 10000, False,
                            'basinhopping', 100, 'BFGS', x0=np.zeros(2))
    assert result is None and bad.custom_loss.nonfinite_gradient_fits == 1


def test_bfgs_and_lbfgs_continue_other_starts(monkeypatch):
    torch, train, _ = runtime()
    e = evaluator()
    starts = iter([np.zeros(2), np.array([.2, 0.])])
    monkeypatch.setattr(train.torch, 'randn', lambda *a, **kw: torch.tensor(next(starts), dtype=torch.float64))
    c = train.train_bfgs(e, 2, 2)
    assert c is not None and np.isfinite(c).all()
    assert e.custom_loss.nonfinite_gradient_fits == 1
    e = evaluator()
    monkeypatch.setattr(train.torch, 'randn', lambda *a, **kw: torch.tensor([.2, 0.], dtype=torch.float64))
    c = train.train_lbfgs(e, 2, 2, coefficients=torch.zeros(2, dtype=torch.float64))
    assert c is not None and torch.isfinite(c).all()
    assert e.custom_loss.nonfinite_gradient_fits == 1


@pytest.mark.parametrize('success,fun,x,accepted', [
    (False, 2., [1., 0.], True), (True, np.nan, [1., 0.], False),
    (True, 1., [np.nan, 0.], False), (True, np.inf, [1., 0.], False),
])
def test_local_result_validity_is_not_optimizer_success(monkeypatch, success, fun, x, accepted):
    _, train, _ = runtime()
    from scipy.optimize import OptimizeResult
    returned = OptimizeResult(success=success, fun=fun, x=np.array(x))
    monkeypatch.setattr(train, 'minimize', lambda *a, **kw: returned)
    e = evaluator()
    method = train._finite_minimizer(e, 'BFGS')
    result = method(lambda c: 0., np.array([1., 0.]))
    if accepted:
        assert result is returned and not result.success
    else:
        assert not result.success and np.isinf(result.fun)
        assert np.isfinite(result.x).all()
    assert (method.best_result is not None) == accepted
    assert getattr(e.custom_loss, 'nonfinite_result_fits', 0) == int(not accepted)


def family_settings(pruning):
    return dict(maxiter1=1, maxiter2=0, optimizer='basinhopping',
                maxiter_per_dim_local_minimizer=1, local_minimizer='BFGS', lambda_1=0, lambda_1_piecewise=0,
                lambda_1_cut=0, max_dataset_length=20, device='cpu', model='ParFamTorch',
                classifier=None, pruning_iterations=pruning, pruning_cut_off=.01)


def test_initial_family_failure_returns_non_candidate_tuple(monkeypatch):
    torch, train, _ = runtime()
    e = evaluator()
    monkeypatch.setattr(train, 'setup_model', lambda *a, **kw: e.model)
    monkeypatch.setattr(train, 'training', lambda *a, **kw: None)
    x = torch.arange(20, dtype=torch.float64).reshape(-1, 1)
    r = train.hyperparameter_training(x, torch.ones(20, dtype=torch.float64), '',
                                      family_settings(1), {}, 17, custom_loss=e.custom_loss)
    assert all(np.isinf(v) for v in r[:3])
    assert r[3] >= 0 and torch.isnan(r[4]).all() and r[5] == 0


def test_failed_pruning_preserves_completed_coefficients(monkeypatch):
    torch, train, _ = runtime()
    e = evaluator()
    monkeypatch.setattr(train, 'setup_model', lambda *a, **kw: e.model)
    fits = iter([np.array([1., 0.]), None])
    monkeypatch.setattr(train, 'training', lambda *a, **kw: next(fits))
    x = torch.arange(20, dtype=torch.float64).reshape(-1, 1)
    r = train.hyperparameter_training(x, torch.ones(20, dtype=torch.float64), '',
                                      family_settings(2), {}, 17, custom_loss=e.custom_loss)
    np.testing.assert_array_equal(r[4].numpy(), [1., 0.])
    assert all(np.isfinite(float(v)) for v in r[:3])


def test_all_failed_families_raise_specific_error(monkeypatch):
    torch, train, _ = runtime()
    e = evaluator()
    failed = (np.inf, np.inf, np.inf, .01, torch.full((2,), np.nan), 1)
    monkeypatch.setattr(train, 'hyperparameter_training', lambda *a, **kw: failed)
    with pytest.raises(train.NoFiniteFamilyError, match='no finite completed family'):
        train.run_jobs_sequentially(e.x, e.y, '', [{}, {}], 17,
                                    dict(time_limit=100, evaluations_limit=100), .001,
                                    custom_loss=e.custom_loss)


def test_runtime_patch_is_idempotent_and_rejects_unknown_source(tmp_path):
    _, train, tensor = runtime()
    from pathlib import Path

    from ml_platform_symbolic.engines.parfam_runtime import patch
    for module in (train, tensor):
        p = Path(module.__file__)
        (tmp_path / p.name).write_bytes(p.read_bytes())
    before = [(tmp_path / n).read_bytes() for n in ('train.py', 'parfam_torch.py')]
    patch(tmp_path)
    assert before == [(tmp_path / n).read_bytes() for n in ('train.py', 'parfam_torch.py')]
    (tmp_path / 'parfam_torch.py').write_bytes(before[1] + b'\n# changed marked source\n')
    with pytest.raises(ValueError, match='Unexpected pinned ParFam'):
        patch(tmp_path)


def test_failed_local_fit_records_actual_callback_counts():
    _, train, _ = runtime()
    e = evaluator()
    result = train._finite_minimizer(e, 'BFGS')(e.loss_func, np.zeros(2), jac=e.gradient)
    assert not result.success and result.nfev >= 1 and result.njev == 1
    assert result.nfev == e.evaluations


@pytest.mark.parametrize('failure', ['gradient', 'result'])
def test_lbfgs_later_failure_retains_prior_finite_fit(monkeypatch, failure):
    torch, train, tensor = runtime()
    e = evaluator()
    kept = torch.tensor([.5, 0.], dtype=torch.float64)
    calls = []
    def fit(c, verbose):
        calls.append(1)
        if len(calls) == 1:
            return kept, torch.tensor(.5)
        if failure == 'gradient':
            raise tensor.NonfiniteGradientError('controlled later failure')
        return torch.tensor([np.nan, 0.]), torch.tensor(np.nan)
    monkeypatch.setattr(e, 'fit_lbfgs', fit)
    result = train.train_lbfgs(e, 2, 2)
    assert result is kept and len(calls) == 2
    field = 'nonfinite_gradient_fits' if failure == 'gradient' else 'nonfinite_result_fits'
    assert getattr(e.custom_loss, field) == 1
