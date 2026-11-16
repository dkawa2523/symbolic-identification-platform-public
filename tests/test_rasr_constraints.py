"""Pinned native refinement receives constraints before native candidate ranking."""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.native


def test_native_rasr_response_fit_uses_original_coordinates(monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME','/work/pretrained/rasr/cache/xdg')
    monkeypatch.setenv('HF_HOME','/work/pretrained/rasr/cache/hf')
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    import simplipy
    from ml_platform_symbolic.engines.base import DiscoveryRequest
    from ml_platform_symbolic.engines.rasr_refinement import ResponseRefinement
    from src.rasr.inference.refiner import Refiner
    engine = simplipy.SimpliPyEngine.load('dev_7-3', install=True)
    x = np.linspace(10,12,20).reshape(-1,1)
    y = 50-(x[:,0]-10)
    request = DiscoveryRequest(x,y,np.arange(20),[{'id':'x0','column':'x','unit':'1','domain':[10,12]}],
        {'unit':'1'},['add','sub','mul','div'],
        {'probes':[{'x':12.0}],'response_bounds':[52.0,None],'response_checks':[],'response_mode':'search'},
        42,{})
    fitted = SimpleNamespace(X_mean_=np.array([10.]),X_std_=np.array([2.]),y_mean_=50.,y_std_scale_=5.)
    normalized_x=(x-10)/2
    normalized_y=(y-50)/5

    class CheckedRefinement(ResponseRefinement):
        def __call__(self, tokens, native_X, native_y, initial):
            # Pin native row order and normalization at the callback boundary.
            np.testing.assert_array_equal(native_X, normalized_x)
            np.testing.assert_array_equal(np.asarray(native_y).reshape(-1), normalized_y)
            return super().__call__(tokens, native_X, native_y, initial)

    solver = CheckedRefinement(request,fitted)
    constrained = Refiner(engine,n_variables=1)
    constrained.fit(['*','<constant>','x1'],normalized_x,normalized_y.reshape(-1,1),
        p0=np.array([-0.4]),p0_noise=None,n_restarts=1,coefficient_solver=solver)
    assert constrained.valid_fit
    coefficient=constrained._all_constants_values[0][0][0]
    assert 50.0 + 5.0 * coefficient >= 52.0 and coefficient < 0.401
    assert solver.records[0]['status']=='feasible'
    baseline = Refiner(engine,n_variables=1)
    baseline.fit(['*','<constant>','x1'],normalized_x,normalized_y.reshape(-1,1),
        p0=np.array([-0.4]),p0_noise=None,n_restarts=1)
    assert baseline._all_constants_values[0][0][0] == pytest.approx(-0.4)
    from src.rasr.inference.rasr import RASR
    dispatch = SimpleNamespace(coefficient_solver=solver, refiner_workers=0,
                               simplipy_engine_name='dev_7-3', verbose=False)
    payload = [{'tokens': ['*','<constant>','x1'], 'cand_id':0, 'log_prob':0.0,
                'n_restarts':1, 'init_noise_scale':0.0}]
    outputs = RASR._run_refinement(dispatch, payload, normalized_x, normalized_y)
    assert len(outputs) == 1 and 50.0 + 5.0 * outputs[0]['fits'][0][0][0] >= 52.0
    dispatch.refiner_workers = 1
    with pytest.raises(ValueError, match='serial'):
        RASR._run_refinement(dispatch, payload, normalized_x, normalized_y)
    request.limits['response_bounds']=[None,51.0]
    impossible = ResponseRefinement(request,fitted)
    fixed = Refiner(engine,n_variables=1)
    with pytest.raises(ValueError,match='feasible'):
        fixed.fit(['x1'],normalized_x,normalized_y.reshape(-1,1),p0_noise=None,n_restarts=1,
                  coefficient_solver=impossible)
    assert impossible.records[0]['status']=='rejected'
    # Native numeric leaves are fitted, while the exponent inside pow2 and
    # stored normalization quantities remain structural/fixed.
    from ml_platform_symbolic.engines.rasr import export_candidate
    typed = Refiner(engine,n_variables=1)
    _,symbols,initial = typed._identify_constants(['+','pow2','x1','1.0'])
    assert symbols == ['C_0'] and initial == [1.0]
    exported = export_candidate(SimpleNamespace(tokens=['+','pow2','x1','1.0'],constants=[1.0]),
                                fitted,request.variables,request.target)
    assert [k for k,v in exported['parameters'].items() if v['trainable']] == ['c0']
    request.limits['response_bounds']=[None,None]
    request.limits['probes']=[{'x':11.0}]
    invalid = ResponseRefinement(request,fitted)
    with pytest.raises(ValueError, match='Nonfinite common Train'):
        Refiner(engine,n_variables=1).fit(['exp','log','x1'],normalized_x,normalized_y.reshape(-1,1),
            p0_noise=None,n_restarts=1,coefficient_solver=invalid)
    assert invalid.records[0]['status']=='rejected'




@pytest.mark.parametrize('order,sign,tokens', [
    (1,-1,['*','<constant>','x1']),
    (2,1,['*','<constant>','pow2','x1']),
])
def test_native_rasr_derivative_conditions_on_different_families(monkeypatch,order,sign,tokens):
    monkeypatch.setenv('XDG_CACHE_HOME','/work/pretrained/rasr/cache/xdg')
    monkeypatch.setenv('HF_HOME','/work/pretrained/rasr/cache/hf')
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    import simplipy
    from ml_platform_symbolic.engines.base import DiscoveryRequest
    from ml_platform_symbolic.engines.rasr import export_candidate
    from ml_platform_symbolic.engines.rasr_refinement import ResponseRefinement
    from ml_platform_symbolic.expressions import evaluate
    from ml_platform_symbolic.response_search import response_feasible
    from src.rasr.inference.refiner import Refiner

    x=np.linspace(10,12,24).reshape(-1,1)
    z=(x[:,0]-10)/2
    y=50+5*(z if order==1 else -z*z)
    variables=[{'id':'x0','column':'x','unit':'1','domain':[10,12]}]
    request=DiscoveryRequest(x,y,np.arange(24),variables,{'unit':'1'},['add','sub','mul','div','pow_int'],
        {'probes':[],'response_bounds':[None,None],'response_checks':[{'feature':'x','points':[{'x':11.0}],
         'order':order,'sign':sign,'step':0.01,'tolerance':0.0}],'response_mode':'search'},42,{})
    fitted=SimpleNamespace(X_mean_=np.array([10.]),X_std_=np.array([2.]),y_mean_=50.,y_std_scale_=5.)
    solver=ResponseRefinement(request,fitted)
    refiner=Refiner(simplipy.SimpliPyEngine.load('dev_7-3',install=True),n_variables=1)
    refiner.fit(tokens,z[:,None],((y-50)/5)[:,None],p0=np.array([1.0 if order==1 else -1.0]),
                p0_noise=None,n_restarts=1,coefficient_solver=solver)
    assert refiner.valid_fit and solver.records[0]['status']=='feasible'
    equation=export_candidate(SimpleNamespace(tokens=tokens,constants=refiner._all_constants_values[0][0]),
                              fitted,variables,{'unit':'1'})
    values=evaluate(equation,np.asarray(solver.compiled['points']),variables)
    assert response_feasible(values,solver.compiled)
    assert abs(refiner._all_constants_values[0][0][0]) < 0.01
