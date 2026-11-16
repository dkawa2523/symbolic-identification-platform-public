from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.data import write_json,read_json
from ml_platform_symbolic.expressions import parse_expression
from ml_platform_symbolic.ensemble import load_ensemble


def payload():
 variables=[{'id':'x0','column':'x','unit':'m','domain':None}]
 schema={'variables':variables,'target':{'column':'y','unit':'m'},'unit_coordinates':{'columns':{'x':{'source_unit':'cm','problem_unit':'m','scale':.01,'offset':0},'y':{'source_unit':'m','problem_unit':'m','scale':1.,'offset':0}}}}
 return {'schema_version':1,'kind':'structural_point_ensemble','schema':schema,'members':[{'weight':1.,'equation':parse_expression('x0',variables,trainable=False)}],'operators':[],'constraints':{'max_nodes':1,'require_units':False},'metadata':{}}


def test_artifact_marker_not_frozen_map_controls_historical_limit(tmp_path):
 p=payload();path=write_json(tmp_path/'old.json',p)
 old=load_ensemble(path)
 np.testing.assert_allclose(old.predict(pd.DataFrame({'x':[100.]})),[1.])
 assert 'no source-size guarantee' in old.metadata['complexity_scope']
 p['complexity_coordinate']='source';write_json(path,p)
 with pytest.raises(ValueError,match='constraints'):load_ensemble(path)
 p['constraints']['max_nodes']=3;write_json(path,p)
 np.testing.assert_allclose(load_ensemble(path).predict(pd.DataFrame({'x':[100.]})),[1.])
 p['complexity_coordinate']='guess';write_json(path,p)
 with pytest.raises(ValueError,match='Unknown stored'):load_ensemble(path)


@pytest.mark.parametrize('bound,status',[(2,'fail'),(3,'pass')])
def test_numerical_keeps_old_limit_but_checks_new_source_limit(synthetic_config,tmp_path,bound,status):
 from ml_platform_symbolic.equation_evaluation import create_study
 cfg=deepcopy(synthetic_config);cfg['data']['units']={'x':'cm','y':'m'};cfg['constraints']['max_nodes']=bound
 artifact=write_json(tmp_path/'old.json',payload())
 study=create_study({'problem':cfg,'purpose':'local_prediction','engineering_tolerance':100.,'sources':[{'ensemble':str(artifact)}]},run_dir=tmp_path/'study')
 row=read_json(study/'numerical.json')[0]
 old=next(c for c in row['checks'] if c['name']=='source/max_nodes')
 current=next(c for c in row['checks'] if c['name']=='max_nodes')
 assert (old['count'],old['status'])==(1,'pass')
 assert (current['count'],current['status'])==(3,status)
 assert row['n_nodes']==3 and row['working_n_nodes']==1


def test_runner_finite_pool_uses_source_size_before_discard(tmp_path):
 from ml_platform_symbolic.engines.base import Candidate,DiscoveryRequest,DiscoveryContext,DiscoveryResult
 from ml_platform_symbolic.engines.runner import _persist_result
 variables=[{'id':f'x{i}','column':f'w{i}','unit':'1','domain':[-10,10]} for i in range(2)]
 source=[{'id':f'x{i}','column':f'u{i}','unit':'1'} for i in range(2)]
 geometry={'source_variables':source,'map':{'kind':'affine_chart','source_columns':['u0','u1'],'target_columns':['w0','w1'],'forward':{'matrix':[[2,0],[0,1]],'offset':[1,0]}}}
 request=DiscoveryRequest(np.array([[3.,1.5],[3.,1.5]]),np.array([3.,3.]),np.arange(2),variables,{'column':'y','unit':'1'},['add'],{'max_nodes':20,'require_units':False,'probes':[]},17,{})
 candidates=[Candidate(name,'dictionary',17,parse_expression(expr,variables,trainable=False),{'passed':True}) for name,expr in [('A','x0'),('B','x1+x1')]]
 job={'engine':'dictionary','seed':17,'candidate_pool_policy':'pareto_diverse','max_export_candidates':1}
 _persist_result(DiscoveryResult(candidates,'ok',{}),tmp_path/'job.json',job,request,DiscoveryContext(tmp_path,provenance={'input_geometry':geometry}),None,['add'])
 rows=[json.loads(line) for line in (tmp_path/'candidates.jsonl').read_text().splitlines()]
 assert [r['candidate_id'] for r in rows]==['B']


