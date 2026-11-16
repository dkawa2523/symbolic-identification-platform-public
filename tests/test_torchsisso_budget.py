import pytest
from types import SimpleNamespace
from ml_platform_symbolic.engines.torchsisso_budget import NodeBudget,prepare,feasible_combinations

@pytest.mark.native
def test_source_combination_boundary_and_column_mapping():
    import torch
    # Source descriptor costs 9 and 3, target overhead2: single small term B9.
    original=torch.arange(12,dtype=torch.float64).reshape(6,2)
    for total,allowed in [(9,True),(8,False)]:
        r=SimpleNamespace(node_budget=NodeBudget(total-2,lambda n:{'large':9,'small':3}[n]),names=['large','small'],x=original.clone(),dimension=3)
        if not allowed:
            with pytest.raises(ValueError,match='before fitting'):prepare(r)
        else:
            prepare(r)
            assert r.names==['small']
            assert torch.equal(r.x,original[:,1:2])
    r=SimpleNamespace(node_budget=NodeBudget(13,lambda _:3),node_costs=[3,4],node_budget_events=[])
    actual=feasible_combinations(r,torch.tensor([[0,0],[0,1]]))
    assert actual.tolist()==[[0,0]]
    assert r.node_budget_events==[{'proposed':2,'feasible':1}]
