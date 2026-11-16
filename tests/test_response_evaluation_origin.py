import numpy as np
from ml_platform_symbolic.constraints import response_evaluation, response_statistics

V=[{'id':'x0','column':'x','unit':'m','domain':[-2,2]}]
T={'unit':'m'}

def test_origin_overlap_and_invalid_denominators():
    c={'probes':[{'x':0},{'x':1},{'x':0},{'x':-1}], 'response_bounds':[0,None]}
    def predict(X):
        return np.where(X[:,0]==1,np.nan,X[:,0])
    rows=response_evaluation(predict,V,T,c,{'discovery_probes':2,'discovery_checks':0})
    first,extra=rows
    assert first['n_invalid']==1 and first['failure_rate']==.5
    assert first['violation_rate']==0 and first['max_violation'] is None
    assert extra['overlap_with_prepared_points']==1
    assert extra['n_violating']==1 and extra['violation_rate']==.5 and extra['max_violation']==1
    assert extra['response_unit']=='m' and 'not independent' in extra['data_role']
    assert extra['independence'].startswith('not_asserted')

def test_unspecified_is_not_evaluated_and_historical_not_assumed():
    def unused(X): raise AssertionError('must not predict')
    rows=response_evaluation(unused,V,T,{}, {'discovery_probes':0,'discovery_checks':0})
    assert [r['status'] for r in rows]==['not_evaluated']*2
    assert response_evaluation(unused,V,T,{},None)[0]['status']=='not_evaluated'

def test_derivative_extra_points_keep_units_and_meaning():
    request={'feature':'x','order':1,'sign':1,'step':.1,'tolerance':0,
             'points':[{'x':-.5},{'x':.5}]}
    rows=response_evaluation(lambda X:X[:,0]**2,V,T,{'response_checks':[request]},
                             {'discovery_probes':0,'discovery_checks':0})
    extra=rows[1]
    assert extra['n_violating']==1 and extra['violation_rate']==.5
    assert extra['max_violation'] > .99
    assert extra['response_unit']=='(m)/(m)^1'
    assert extra['points']==request['points']
