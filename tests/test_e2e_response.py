from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.engines.e2e import export_candidate,validate_request
from ml_platform_symbolic.engines.e2e_response import prepare_response_fit,fit_responses
from ml_platform_symbolic.engines.base import Candidate,DiscoveryRequest,DiscoveryContext
from ml_platform_symbolic.expressions import evaluate
from ml_platform_symbolic.model import save_model,load_model


def request(limits,options=None):
 X=np.linspace(-1,1,20)[:,None]
 return DiscoveryRequest(X,-2*X[:,0],np.arange(20),[{'id':'x0','column':'x','unit':'1','domain':[-2,2]}],{'column':'y','unit':'1'},['add','mul','sub','div'],{'max_nodes':40,'response_mode':'search',**limits},17,options or {})


def candidate(r,text,name='raw',kind='NoRef'):
 return Candidate(name,'e2e',17,export_candidate(text,r.variables,[0.],[1.]),{'passed':True,'refinement_type':kind})


def fit(r,pool,tmp_path):
 return fit_responses(pool,r,DiscoveryContext(tmp_path),prepare_response_fit(r))


def test_response_coefficient_search_changes_fit_not_native_and_saves(tmp_path):
 r=request({'response_checks':[{'feature':'x','order':1,'sign':1,'step':.1,'tolerance':1e-10,'points':[{'x':-.5},{'x':.5}]}]})
 pool=[candidate(r,'mul,-2.0,x_0')];original=deepcopy(pool)
 variants,meta=fit(r,pool,tmp_path)
 assert len(variants)==1 and meta['supplied']==meta['attempted']==1
 assert not meta['decoder_conditioned_on_response'] and pool==original
 v=variants[0]; assert v.equation['arithmetic']=='strict_real'
 assert v.native_check['finite_point_feasible']
 assert 'not native predictor parity' in v.native_check['kind']
 assert not np.allclose(evaluate(v.equation,r.X,r.variables),evaluate(pool[0].equation,r.X,r.variables))
 schema={'variables':r.variables,'target':r.target,'feature_ranges':[[-1,1]]}
 save_model(tmp_path/'derived.zip',v.equation,schema,{})
 np.testing.assert_allclose(load_model(tmp_path/'derived.zip').predict(pd.DataFrame({'x':r.X[:,0]})),evaluate(v.equation,r.X,r.variables))
 for name in ['x0_mean','x0_scale']:
  assert v.equation['parameters'][name]==pool[0].equation['parameters'][name]


def test_all_supplied_structures_attempted_and_impossible_fixed_retained(tmp_path):
 r=request({'probes':[{'x':0.}],'response_bounds':[1.,1.]});r.y[:]=0
 pool=[candidate(r,'0','best-raw'),candidate(r,'2.0','flexible','BFGS')]
 variants,meta=fit(r,pool,tmp_path)
 assert meta['supplied']==meta['attempted']==2 and meta['accepted']==1
 assert variants[0].native_check['parent_id']=='flexible'
 assert meta['attempts'][0]['status']=='no_accepted_fit'
 assert 'fixed-parameter' in meta['attempts'][0]['reason']
 assert pool[0].equation['ast']=={'const':0.0}


def test_contradiction_and_unbounded_probe_rejected_before_native(tmp_path):
 r=request({'probes':[{'x':0.}],'response_bounds':[2.,1.]})
 with pytest.raises(ValueError,match='Contradictory'):validate_request(r,DiscoveryContext(tmp_path))
 r.limits['response_bounds']=[None,None]
 with pytest.raises(NotImplementedError,match='finite bound'):validate_request(r,DiscoveryContext(tmp_path))


def test_feasible_best_not_reported_as_converged(tmp_path):
 r=request({'probes':[{'x':0.}],'response_bounds':[0.,2.]},{'response_max_nfev':1})
 variants,meta=fit(r,[candidate(r,'1.0')],tmp_path)
 assert len(variants)==1
 assert variants[0].native_check['finite_point_feasible'] is True
 assert variants[0].native_check['solver_converged'] is False


def test_fixed_feasible_has_no_solver_success(tmp_path):
 r=request({'probes':[{'x':0.}],'response_bounds':[0.,0.]})
 variants,meta=fit(r,[candidate(r,'0')],tmp_path)
 assert variants[0].native_check['solver_converged'] is None


def test_entire_stencil_must_be_inside_joint_domain():
 r=request({'response_checks':[{'feature':'x','order':1,'sign':1,'step':.1,'tolerance':0.,'points':[{'x':1.}]}]})
 r.domain={'kind':'box','columns':['x'],'bounds':[[-1,1]]}
 with pytest.raises(ValueError,match='stencil outside'):prepare_response_fit(r)


def test_filter_path_unchanged_and_deadline_does_not_hide_unattempted(tmp_path):
 r=request({'probes':[{'x':0.}],'response_bounds':[0.,0.]})
 r.limits['response_mode']='filter'
 assert prepare_response_fit(r) is None
 r.limits['response_mode']='search'
 pool=[candidate(r,'0','a'),candidate(r,'1.0','b')]
 original=deepcopy(pool)
 variants,meta=fit_responses(pool,r,DiscoveryContext(tmp_path,deadline=0.),prepare_response_fit(r))
 assert variants==[] and pool==original
 assert meta['supplied']==2 and meta['attempted']==0 and not meta['complete']
 assert all(x['status']=='not_attempted_deadline' for x in meta['attempts'])


@pytest.mark.parametrize("present", [False, True])
def test_optional_size_limit_matches_native_unbounded_semantics(tmp_path, present):
 r=request({'probes':[{'x':0.}],'response_bounds':[0.,0.]})
 if present: r.limits['max_nodes']=None
 else: r.limits.pop('max_nodes')
 variants,meta=fit(r,[candidate(r,'0')],tmp_path)
 assert len(variants)==1 and meta['accepted']==1
 assert variants[0].native_check['source_n_nodes']==1
