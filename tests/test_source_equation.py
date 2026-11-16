from copy import deepcopy
import numpy as np
import pytest
from ml_platform_symbolic.coordinates import source_equation,source_complexity_costs
from ml_platform_symbolic.expressions import parse_expression,evaluate,inspect_equation
from ml_platform_symbolic.engines.e2e import export_candidate

def context():
 return {'variables':[{'id':'x0','column':'length','unit':'m'}], 'target':{'column':'response','unit':'m'}, 'unit_coordinates':{'columns':{'length':{'source_unit':'cm','problem_unit':'m','scale':.01,'offset':0},'response':{'source_unit':'cm','problem_unit':'m','scale':.01,'offset':0}}}}

def test_source_ast_repeated_leaf_and_exact_budget_decomposition():
 c=context();e=parse_expression('x0+x0',c['variables'],trainable=False);p,v=source_equation(e,c)
 np.testing.assert_allclose(evaluate(p,np.array([[10.],[20.]]),v),[20.,40.])
 costs,outer=source_complexity_costs(c)
 assert costs=={'x0':3} and outer==2
 assert inspect_equation(p,v)['n_nodes']==1+2*costs['x0']+outer==9
 assert e['ast']=={'op':'add','args':[{'var':'x0'},{'var':'x0'}]}

def test_source_ast_affine_chart_simultaneous_not_recursive():
 c=context();source=deepcopy(c['variables']);c['variables']=[{'id':'x0','column':'chart','unit':'m'}]
 c['input_geometry']={'source_variables':source,'map':{'kind':'affine_chart','source_columns':['length'],'target_columns':['chart'],'forward':{'matrix':[[2]],'offset':[3]}}}
 e=parse_expression('x0',c['variables'],trainable=False);p,v=source_equation(e,c)
 np.testing.assert_allclose(evaluate(p,np.array([[10.],[20.]]),v),[320.,340.])
 costs,outer=source_complexity_costs(c);assert inspect_equation(p,v)['n_nodes']==costs['x0']+outer

def test_source_ast_temperature_offset_and_residual():
 c={'variables':[{'id':'x0','column':'temperature','unit':'K','coordinate_role':'absolute'}], 'target':{'column':'response','unit':'K','coordinate_role':'absolute'},'unit_coordinates':{'columns':{k:{'source_unit':'degC','problem_unit':'K','scale':1.,'offset':273.15} for k in ['temperature','response']}}}
 e=parse_expression('x0',c['variables'],trainable=False);p,v=source_equation(e,c)
 np.testing.assert_allclose(evaluate(p,np.array([[0.],[20.]]),v),[0.,20.],atol=1e-12)
 e=parse_expression('target-x0',c['variables']+[{**c['target'],'id':'target'}],trainable=False);p,v=source_equation(e,c,residual=True)
 np.testing.assert_allclose(evaluate(p,np.array([[0.,0.],[20.,30.]]),v),[0.,10.])
 assert source_complexity_costs(c,residual=True)[1]==0

def test_source_ast_e2e_normalization_once():
 c=context();e=export_candidate('add,x_0,1',c['variables'],[2.],[3.]);p,v=source_equation(e,c)
 np.testing.assert_allclose(evaluate(p,np.array([[100.],[200.]]),v),[200./3,100.])
 # E2E native variable costs 4 normalization nodes plus public variable expansion.
 cost,outer=source_complexity_costs(c)
 assert inspect_equation(p,v)['n_nodes']==1+(4+cost['x0'])+1+outer

@pytest.mark.parametrize('change',[{'scale':0},{'offset':float('nan')},{'problem_unit':'s'},{'scale':True}])
def test_source_ast_invalid_frozen_metadata_rejected(change):
 c=context();c['unit_coordinates']['columns']['length'].update(change)
 with pytest.raises(ValueError):source_complexity_costs(c)

def test_source_ast_missing_map_and_non_affine_rejected():
 c=context();c.pop('unit_coordinates');c['public_units']={'length':'cm'}
 with pytest.raises(ValueError,match='not frozen'):source_complexity_costs(c)
 c=context();c['input_geometry']={'map':{'kind':'nonlinear'}}
 with pytest.raises(NotImplementedError):source_complexity_costs(c)


def test_source_ast_preserves_large_valid_shared_graph():
 c={'variables':[{'id':'x0','column':'x','unit':'1'}],'target':{'column':'y','unit':'1'}}
 definitions={'a0':{'var':'x0'}}
 for i in range(1,13):definitions[f'a{i}']={'op':'add','args':[{'ref':f'a{i-1}'},{'ref':f'a{i-1}'}]}
 e={'schema_version':2,'ast':{'ref':'a12'},'definitions':definitions,'parameters':{}}
 assert inspect_equation(e,c['variables'])['n_nodes']==8191
 p,v=source_equation(e,c)
 assert len(p['definitions'])==13 and inspect_equation(p,v)['n_nodes']==8191
 np.testing.assert_allclose(evaluate(p,np.array([[2.]]),v),[8192.])
 from ml_platform_symbolic.equation_output import original_equation
 assert original_equation(e,c) is not None

def test_common_size_checks_and_output_use_source_not_working():
 from ml_platform_symbolic.coordinates import source_complexity
 from ml_platform_symbolic.constraints import check_constraints
 from ml_platform_symbolic.equation_output import equation_outputs
 from ml_platform_symbolic.evaluation import select_row
 c=context();eq=parse_expression('x0+x0',c['variables'],trainable=False)
 assert inspect_equation(eq,c['variables'])['n_nodes']==3
 assert source_complexity(eq,c)==9
 for bound,status in [(9,'pass'),(8,'fail')]:
  checks=check_constraints(eq,c['variables'],c['target'],['add'],{'max_nodes':bound,'require_units':False},context=c)
  size=next(x for x in checks if x['name']=='max_nodes')
  assert (size['count'],size['status'])==(9,status)
 output=equation_outputs(eq,c)
 assert output['n_nodes']==9 and output['working_n_nodes']==3
 rows=[{'candidate_id':'source-larger','eligible':True,'rmse':1,'n_nodes':9,'working_n_nodes':3,'n_parameters':0}, {'candidate_id':'source-smaller','eligible':True,'rmse':1,'n_nodes':7,'working_n_nodes':7,'n_parameters':0}]
 chosen,_=select_row(rows,{'selection_policy':'simplicity','acceptable_rmse':2})
 assert chosen['candidate_id']=='source-smaller'


def test_strict_source_size_refuses_missing_frozen_conversion():
 from ml_platform_symbolic.constraints import check_constraints
 c=context();del c['unit_coordinates'];c['public_units']={'length':'cm','response':'cm'}
 eq=parse_expression('x0',c['variables'],trainable=False)
 with pytest.raises(ValueError,match='not frozen'):
  check_constraints(eq,c['variables'],c['target'],[],{'max_nodes':30,'require_units':False},context=c)


def test_low_level_structure_api_does_not_claim_source_guarantee():
 from ml_platform_symbolic.constraints import check_constraints
 c=context();eq=parse_expression('x0+x0',c['variables'],trainable=False)
 checks=check_constraints(eq,c['variables'],{'unit':'m'},['add'],{'max_nodes':3,'require_units':False})
 size=next(x for x in checks if x['name']=='max_nodes')
 assert size['status']=='pass' and size['count']==3
 assert size['method']=='declared expression-coordinate AST; no source mapping supplied'
