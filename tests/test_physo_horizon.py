"""Optional PhySO token horizon, distinct from frozen source/slot AST limits."""
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
from ml_platform_symbolic.engines.physo import configure_horizon, validate_request


def config():
    return {'learning_config': {'max_time_step': 35}, 'priors_config': [('SoftLengthPrior', {'length_loc': 8, 'scale': 5})]}


def request(tmp_path, options):
    return DiscoveryRequest(np.ones((3, 1)), np.ones(3), np.arange(3),
                            [{'id': 'x', 'column': 'x', 'unit': '1'}], {'column': 'y', 'unit': '1'},
                            ['add', 'mul'], {'max_nodes': 12}, 17, options), DiscoveryContext(tmp_path)


def test_default_exact_native_config_compatibility():
    native = config()
    expected = deepcopy(native)
    expected['learning_config']['max_time_step'] = 256
    info = configure_horizon({}, 256, native)
    assert native == expected
    assert info['origin'] == 'legacy_common_source_max_nodes'
    assert info['preset_max_time_step'] == 35 and info['effective_max_time_step'] == 256


@pytest.mark.parametrize('value', [1, 5, 12, 30])
def test_explicit_value_not_clipped_and_source_cap_independent(value):
    native = config()
    info = configure_horizon({'max_time_step': value}, 12, native)
    assert native['learning_config']['max_time_step'] == value
    assert native['priors_config'][-1] == ('HardLengthPrior', {'min_length': 0 if value == 1 else 1, 'max_length': value})
    assert info['source_max_nodes'] == 12 and info['requested_max_time_step'] == value


@pytest.mark.parametrize('value', [True, False, None, 0, -1, 1.5, '5', [], float('inf')])
def test_invalid_horizon_rejected_without_mutation(tmp_path, value):
    native = config()
    original = deepcopy(native)
    with pytest.raises(ValueError, match='positive integer'):
        configure_horizon({'max_time_step': value}, 12, native)
    assert native == original
    r, c = request(tmp_path, {'max_time_step': value})
    with pytest.raises(ValueError, match='positive integer'):
        validate_request(r, c)


def test_public_datafree_validation_accepts_explicit_and_preserves_request(tmp_path):
    r, c = request(tmp_path, {'max_time_step': 5})
    original = deepcopy(r.limits)
    validate_request(r, c)
    assert r.limits == original and r.options == {'max_time_step': 5}


def test_known_slot_source_budget_not_changed(tmp_path):
    from ml_platform_symbolic.engines.physo_template import prepare
    from test_physo_template import request as known_request
    r, c = known_request(tmp_path)
    before = prepare(r, c).source_budget()
    limits, template = deepcopy(r.limits), deepcopy(r.equation_template)
    r.options['max_time_step'] = 5
    validate_request(r, c)
    assert prepare(r, c).source_budget() == before
    assert r.limits == limits and r.equation_template == template


@pytest.mark.native
@pytest.mark.parametrize('known', [False, True])
@pytest.mark.parametrize('horizon', [None, 5])
def test_actual_native_entry_receives_horizon_without_search(tmp_path, monkeypatch, known, horizon):
    import physo
    from ml_platform_symbolic.engines import physo as adapter
    from test_physo_template import request as known_request
    r, c = known_request(tmp_path)
    if not known:
        r.equation_template = None
    if horizon is not None:
        r.options['max_time_step'] = horizon
    original_limits = deepcopy(r.limits)
    class EntryObserved(RuntimeError):
        pass
    def observe(*args, **kwargs):
        config = kwargs['run_config']
        assert config['learning_config']['max_time_step'] == (horizon or r.limits['max_nodes'])
        hard = [p for p in config['priors_config'] if p[0] == 'HardLengthPrior']
        assert hard == ([] if horizon is None else [('HardLengthPrior', {'min_length': 1, 'max_length': horizon})])
        assert kwargs['epochs'] == 1 and kwargs['max_n_evaluations'] == 48
        assert (kwargs['candidate_wrapper'] is not None) == known
        raise EntryObserved('captured native entry, no sampling invoked')
    monkeypatch.setattr(physo, 'SR', observe)
    with pytest.raises(EntryObserved):
        adapter.discover(r, c)
    assert r.limits == original_limits


@pytest.mark.parametrize('value,expected', [(5, 'inputs_valid'), (True, 'blocked'), (0, 'blocked')])
def test_common_public_input_review(synthetic_config, value, expected):
    from ml_platform_symbolic.input_review import review_inputs
    synthetic_config['search'].update(engines=['physo'], engine_options={'physo': {'max_time_step': value}})
    assert review_inputs(synthetic_config)['status'] == expected


@pytest.mark.native
@pytest.mark.parametrize('horizon', [1, 2])
def test_real_native_prior_constructs_and_masks_capacity(horizon):
    from ml_platform_symbolic.engines.physo_budget import source_budget_prior
    from physo.physym import library, prior, vect_programs
    lib = library.Library(args_make_tokens={'op_names': ['add', 'neg'], 'input_var_ids': {'x': 0}})
    progs = vect_programs.VectPrograms(batch_size=1, max_time_step=horizon, library=lib, n_realizations=1)
    native = config()
    native['priors_config'] = []
    info = configure_horizon({'max_time_step': horizon}, 12, native)
    assert info['native_min_length'] == (0 if horizon == 1 else 1)
    assert info['effective_minimum_ast_length'] == 1
    original = prior.make_PriorCollection
    with source_budget_prior(12, [1], 0, {'add', 'neg'}, []):
        collection = prior.make_PriorCollection(lib, progs, native['priors_config'])
        assert 'HardLengthPrior' in collection.priors_names
        mask = collection()[0]
        assert mask[lib.lib_name_to_idx['x']] == 1
        assert mask[lib.lib_name_to_idx['add']] == 0
        assert mask[lib.lib_name_to_idx['neg']] == (0 if horizon == 1 else 1)
        progs.append(np.array([lib.lib_name_to_idx['x']]))
        assert progs.n_completed[0] == 1
    assert prior.make_PriorCollection is original
