"""Pinned native reward integration: feasibility, coefficient writeback, duplicate safety."""
from types import SimpleNamespace
import numpy as np
import pytest
pytestmark = pytest.mark.native


def setup():
    import torch
    from physo.physym.library import Library
    from physo.physym.vect_programs import VectPrograms
    from ml_platform_symbolic.engines.physo import _export
    from ml_platform_symbolic.engines.physo_response import ResponseReward, prepare
    from physo.physym.reward import SquashedNRMSE
    from ml_platform_symbolic.engines.base import DiscoveryRequest
    torch.set_default_dtype(torch.float64)
    X=np.linspace(-1,1,21)[:,None]; y=-2*X[:,0]
    variables=[{'id':'x0','column':'x','unit':'1','domain':[-2,2]}]
    req=DiscoveryRequest(X,y,np.arange(len(y)),variables,{'column':'y','unit':'1'},['mul','neg'],
        {'response_mode':'search','response_enforcement':'hard','probes':[{'x':1}],'response_bounds':[1,None]},17,{'response_max_nfev':100})
    lib=Library(args_make_tokens=dict(op_names=['mul','neg'],use_protected_ops=False,input_var_ids={'x0':0},free_constants={'a'}))
    def programs(tokens):
        p=VectPrograms(batch_size=1,max_time_step=len(tokens),library=lib,n_realizations=1)
        p.set_programs(np.array([[lib.lib_name_to_idx[t] for t in tokens]]));return p
    controller=ResponseReward(req,prepare(req),lambda p:_export(p,variables,{'a':'1'}))
    callback=controller.factory(SquashedNRMSE,zero_out_duplicates=True)
    def run(p):return callback(p,torch.tensor(X.T),torch.tensor(y),torch.ones(len(y)),np.array([len(y)]),{})
    return req,controller,programs,run


def test_physo_response_changes_constants_before_native_reward():
    req,c,programs,run=setup();p=programs(['mul','a','x0']);r=run(p)
    assert r[0]>0
    assert float(p.free_consts.class_values[0,0])>=1
    assert c.counts['feasible']==1
    from ml_platform_symbolic.response_search import response_bounds
    assert c.examples[0]['response_values'][0]>=response_bounds(c.compiled)[0][0]
    assert c.summary()['native_reward_equal_duplicate_elimination'] is False


def test_physo_response_distinct_equal_data_rewards_not_duplicate_filtered():
    req,c,programs,run=setup()
    # These two forms have equal native data reward for x^2, but different feasibility.
    req.y[:]=req.X[:,0]**2
    assert run(programs(['x0']))[0]>0
    assert run(programs(['neg','x0']))[0]==0
    assert c.counts['no_accepted_fit']==1


def test_physo_response_factory_restored_on_exception():
    from physo.physym import reward
    from ml_platform_symbolic.engines.physo_response import reward_scope
    _,c,_,_=setup();original=reward.make_RewardsComputer
    with pytest.raises(RuntimeError,match='forced'):
        with reward_scope(c):
            assert reward.make_RewardsComputer != original
            raise RuntimeError('forced')
    assert reward.make_RewardsComputer is original
