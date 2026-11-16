from copy import deepcopy
import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.problem import problem_document,problem_config,problem_requirements,requirement_identity
from ml_platform_symbolic.input_review import review_inputs


def test_old_problem_identity_is_unchanged(synthetic_config):
    cfg=resolve_config(synthetic_config);doc=problem_document(cfg)
    assert 'unit_search' not in doc['requirements']
    legacy=deepcopy(cfg);legacy['constraints'].pop('unit_search', None)
    assert problem_document(legacy)==doc
    assert 'unit_search' not in resolve_config(doc)['constraints']
    assert problem_config(doc) == cfg

@pytest.mark.parametrize('engine,expected',[('pysr',{'unit_mode':'hard'}),('torchsisso',{'dimensional':True}),('physo',{})])
def test_common_request_resolves_native_options_without_mutating_input(synthetic_config,engine,expected):
    synthetic_config['search']={'engines':[engine]};synthetic_config['requirements']={'unit_search':True}
    original=deepcopy(synthetic_config);review=review_inputs(synthetic_config)
    item=review['engines'][engine]
    assert not item['declared_input_issues']
    assert item['constraint_application']['unit_search']['native_options']==expected
    assert review['requirements']['unit_search'] is True
    assert synthetic_config==original

@pytest.mark.parametrize('engine',['e2e','parfam','dictionary'])
def test_postcheck_adapter_does_not_start_as_unit_search(synthetic_config,engine):
    synthetic_config['search']={'engines':[engine]};synthetic_config['requirements']={'unit_search':True}
    r=review_inputs(synthetic_config);assert r['status']=='blocked'
    assert any('unit_search is unsupported' in i['reason'] for i in r['engines'][engine]['declared_input_issues'])

@pytest.mark.parametrize('engine,options',[('pysr',{'unit_mode':None}),('torchsisso',{'dimensional':False}),('torchsisso',{'dimensional':1})])
def test_explicit_native_off_is_not_silently_overridden(synthetic_config,engine,options):
    synthetic_config['search']={'engines':[engine],'engine_options':{engine:options}};synthetic_config['requirements']={'unit_search':True}
    r=review_inputs(synthetic_config);assert any('conflicts' in i['reason'] for i in r['engines'][engine]['declared_input_issues'])

def test_soft_native_use_is_retained_not_called_hard(synthetic_config):
    synthetic_config['search']={'engines':['pysr'],'engine_options':{'pysr':{'unit_mode':'soft'}}};synthetic_config['requirements']={'unit_search':True}
    r=review_inputs(synthetic_config);assert r['engines']['pysr']['constraint_application']['unit_search']['native_options']=={'unit_mode':'soft'}

@pytest.mark.parametrize('value',[1,None,'true'])
def test_invalid_common_flag_rejected(synthetic_config,value):
    synthetic_config['requirements']={'unit_search':value}
    with pytest.raises(ValueError,match='unit_search'):resolve_config(synthetic_config)

def test_unknown_unit_is_not_dimensionless(synthetic_config):
    synthetic_config['search']={'engines':['pysr']};synthetic_config['requirements']={'unit_search':True};synthetic_config['data']['units']['x']=None
    r=review_inputs(synthetic_config);assert any('known input' in i['reason'] for i in r['engines']['pysr']['declared_input_issues'])

@pytest.mark.parametrize('engine',['pysr'])
def test_template_conflict_rejected_before_native(engine):
    from ml_platform_symbolic.engines.runner import engine_input_issues
    from ml_platform_symbolic.engines.registry import get_engine
    cap=get_engine(engine).capabilities()
    job={'engine':engine,'limits':{'unit_search':True,'equation_template':{}},'options':cap['unit_search']['defaults'],'variables':[{'unit':'m'}],'target':{'unit':'m'},'operators':['add'],'device':'cpu'}
    assert any('unit_search does not support' in i['reason'] for i in engine_input_issues(job,cap))

@pytest.mark.parametrize('engine',['pysr','torchsisso','physo'])
@pytest.mark.parametrize('label',['rad','Hz*s'])
def test_angle_cycle_only_warns_native_dimensions_cannot_distinguish(synthetic_config,engine,label):
    synthetic_config['search']={'engines':[engine]};synthetic_config['requirements']={'unit_search':True};synthetic_config['data']['units']={'x':label,'y':label}
    r=review_inputs(synthetic_config);application=r['engines'][engine]['constraint_application']['unit_search']
    assert application['supported'] is True
    assert application['all_input_target_basic_dimensions_dimensionless'] is True
    assert 'not enforced' in application['warning'] and 'cannot distinguish' in application['warning']

def test_dimensional_units_do_not_get_angle_warning(synthetic_config):
    synthetic_config['search']={'engines':['pysr']};synthetic_config['requirements']={'unit_search':True};synthetic_config['data']['units']={'x':'m','y':'m'}
    a=review_inputs(synthetic_config)['engines']['pysr']['constraint_application']['unit_search']
    assert a['all_input_target_basic_dimensions_dimensionless'] is False
    assert a['warning'] is None
