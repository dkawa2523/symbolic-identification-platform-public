from types import SimpleNamespace
import numpy as np
import pytest
from ml_platform_symbolic.engines.base import native_input_indices


def request(allowed=None):
 return SimpleNamespace(variables=[{'id':'x0'},{'id':'x1'},{'id':'x2'}],limits={'allowed_variables':allowed},equation_template=None)

def test_native_columns_preserve_schema_order_and_no_request_mutation():
 r=request(['x2','x0']);assert native_input_indices(r)==[0,2];assert r.limits['allowed_variables']==['x2','x0']
 assert native_input_indices(request())==[0,1,2]

@pytest.mark.parametrize('allowed',[[],['unknown'],'x0'])
def test_native_columns_reject_empty_unknown_or_invalid(allowed):
 with pytest.raises((ValueError,NotImplementedError)):native_input_indices(request(allowed))

def test_native_columns_reject_unproven_template_or_input_map():
 r=request(['x0']);r.equation_template={'slots':{}}
 with pytest.raises(NotImplementedError):native_input_indices(r)
 r.equation_template=None
 with pytest.raises(NotImplementedError):native_input_indices(r,SimpleNamespace(provenance={'input_geometry':{'map':{'kind':'affine_chart'}}}))


@pytest.mark.parametrize('engine',['parfam','torchsisso','physo'])
def test_adapter_rejects_zero_native_columns_before_import(engine,tmp_path):
    from importlib import import_module
    from ml_platform_symbolic.engines.base import DiscoveryRequest,DiscoveryContext
    r=DiscoveryRequest(np.array([[1.,2.],[2.,3.]]),np.array([1.,2.]),np.arange(2),
        [{'id':'x0','column':'a','unit':'m','domain':[0,4]}, {'id':'x1','column':'b','unit':'s','domain':[0,4]}],
        {'column':'y','unit':'m'},['add','mul'],{'allowed_variables':[], 'max_nodes':20,'max_wall_seconds':60},17,{})
    with pytest.raises(NotImplementedError,match='at least one allowed'):
        import_module('ml_platform_symbolic.engines.'+engine).validate_request(r,DiscoveryContext(tmp_path))


def test_physo_coefficient_catalog_uses_only_allowed_input_units():
    from ml_platform_symbolic.engines.physo import coefficient_unit_catalog
    r=SimpleNamespace(variables=[{'unit':'m'},{'unit':'s'}],target={'unit':'m'},options={'unit_basis_degree':1})
    selected,_=coefficient_unit_catalog(r,[r.variables[0]])
    whole,_=coefficient_unit_catalog(r)
    assert all('s' not in unit for unit in selected.values())
    assert any('s' in unit for unit in whole.values())
