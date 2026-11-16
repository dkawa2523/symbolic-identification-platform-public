"""Pinned native prior boundary tests; no fitting or benchmark data."""
import numpy as np
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.native


def fixtures():
    from physo.physym import library, vect_programs
    lib = library.Library(args_make_tokens={'op_names':['add','n2'], 'input_var_ids':{'x':0}})
    progs = vect_programs.VectPrograms(batch_size=1,max_time_step=12,library=lib,n_realizations=1)
    return lib, progs


def test_physo_omitted_requested_prior_fails_closed_and_restores():
    from physo.physym import prior
    from ml_platform_symbolic.engines.physo_budget import source_budget_prior
    lib, progs = fixtures()
    original = prior.make_PriorCollection
    invalid = [('HardLengthPrior', {'min_length':1,'max_length':1})]
    with pytest.warns(UserWarning):
        collection = original(lib,progs,invalid)
    assert 'HardLengthPrior' not in collection.priors_names
    with pytest.raises(ValueError,match='omitted requested prior'):
        with source_budget_prior(9,[3],2,{'add','n2'},[]):
            with pytest.warns(UserWarning):
                prior.make_PriorCollection(lib,progs,invalid)
    assert prior.make_PriorCollection is original


def test_physo_source_weighted_prefix_boundary_and_no_terminal():
    from physo.physym import prior
    from ml_platform_symbolic.engines.physo_budget import source_budget_prior
    lib, progs = fixtures()
    names = [t.name for t in lib.lib_tokens[:lib.n_choices]]
    original = prior.make_PriorCollection
    # n2(x) costs 2+3+output2 = 7; B-1 blocks it before sampling.
    for bound, allowed in [(7,1),(6,0)]:
        audit=[]
        with source_budget_prior(bound,[3],2,{'add','n2'},audit):
            collection=prior.make_PriorCollection(lib,progs,[])
            assert collection()[0,names.index('n2')] == allowed
            assert collection()[0,names.index('x')] == 1
        assert prior.make_PriorCollection is original
        assert audit[0]['token_costs']['n2'] == 2
    with pytest.raises(ValueError,match='cannot fit any terminal'):
        with source_budget_prior(4,[3],2,{'add','n2'},[]):
            prior.make_PriorCollection(lib,progs,[])
    assert prior.make_PriorCollection is original


def test_physo_completed_padding_respects_native_capacity_without_changing_tree():
    from physo.physym import library, vect_programs, prior
    from ml_platform_symbolic.engines.physo_budget import source_budget_prior
    lib = library.Library(args_make_tokens={'op_names':['add','neg'], 'input_var_ids':{'x':0}})
    progs = vect_programs.VectPrograms(batch_size=2,max_time_step=7,library=lib,n_realizations=1)
    # First complete at six tokens, second still needs one terminal at step six.
    seq = [['neg','add','x','neg','neg','x'], ['add','x','add','x','add','x']]
    for names in zip(*seq): progs.append(np.array([lib.lib_name_to_idx[n] for n in names]))
    before = progs.tokens.idx[0,:6].copy()
    with source_budget_prior(7,[1],0,{'add','neg'},[]):
        collection = prior.make_PriorCollection(lib,progs,[])
        mask = collection()
        assert mask[0,lib.lib_name_to_idx['add']] == 0
        assert mask[0,lib.lib_name_to_idx['x']] == 1
        assert mask[1,lib.lib_name_to_idx['add']] == 0
        assert mask[1,lib.lib_name_to_idx['neg']] == 0
        assert mask[1,lib.lib_name_to_idx['x']] == 1
        progs.append(np.array([lib.lib_name_to_idx['x']]*2))
    np.testing.assert_array_equal(before,progs.tokens.idx[0,:6])
    assert progs.n_lengths.tolist() == [6,7]
