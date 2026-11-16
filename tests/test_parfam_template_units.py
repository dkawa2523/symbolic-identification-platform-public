"""Explicit unit1 slot convention only; no ordinary export or generic unknown-unit inference."""
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.constraints import dimensions
from ml_platform_symbolic.engines.parfam_template import CompositeFamily
from ml_platform_symbolic.expressions import evaluate


class Native:
    def get_number_parameters(self):
        return 2

    def testing_mode(self):
        pass


def declaration(argument='1', output='1'):
    call = {'slot': 'h', 'args': [{'var': 'x'}]}
    return {'coordinate_space': 'working', 'ast': {'op': 'add', 'args': [call, deepcopy(call)]},
            'parameters': {}, 'components': {},
            'slots': {'h': {'arguments': [{'name': 'u', 'unit': argument}], 'unit': output,
                            'allowed_operators': ['add', 'mul'], 'max_nodes': 20}},
            'knowledge': {'kind': 'hypothesis', 'statement': 'Unit metadata mechanism test'}}


def exported(coefficients, unit=None, origin='native_coefficient_block'):
    return {'schema_version': 2, 'ast': {'op': 'add', 'args': [
        {'op': 'mul', 'args': [{'param': 'c0'}, {'var': 'u'}]}, {'param': 'c1'}]},
        'parameters': {f'c{i}': {'value': float(v), 'trainable': True, 'bounds': [None, None],
            'unit': unit, 'origin': origin, 'native_value': float(v),
            'normalization_group': 'denominator_0', 'normalization_norm': 5.0}
            for i, v in enumerate(coefficients)}}


def family(argument='1', output='1', unit=None, origin='native_coefficient_block'):
    return CompositeFamily(Native(), declaration(argument, output),
                           [{'id': 'x', 'column': 'x', 'unit': argument}],
                           lambda native, c, reduced=False: exported(c, unit, origin))


@pytest.mark.parametrize('origin', ['native_coefficient_block', 'native_normalized_denominator_coordinate',
                                  'native_reduction_inactive_zero'])
def test_explicit_dimensionless_preserves_numbers_ast_metadata_and_sharing(origin):
    c = np.array([3.0, 4.0])
    raw = exported(c, origin=origin)
    model = family(origin=origin)
    slot = model.slot_equation(c)
    assert slot['ast'] == raw['ast']
    for key, before in raw['parameters'].items():
        after = slot['parameters'][key]
        assert {k: v for k, v in after.items() if k not in {'unit', 'unit_origin', 'native_coefficient_index'}} == {
            k: v for k, v in before.items() if k != 'unit'}
        assert after['unit'] == '1'
        assert after['unit_origin'] == 'declared_dimensionless_template_slot_convention'
    assert dimensions(slot, [{'id': 'u', 'unit': '1'}], {'unit': '1'})[0] == 'pass'
    full = model.equation(c)
    assert len(full['parameters']) == 2  # Repeated calls remain one shared function.
    np.testing.assert_array_equal(evaluate(full, np.array([[1.0], [2.0]]), model.variables), [14., 20.])
    assert raw == exported(c, origin=origin)  # Exporter-owned source was not mutated.


@pytest.mark.parametrize(('argument', 'output'), [(None, '1'), ('1', None), ('m', '1'), ('1', 'm'),
                                                ('rad', '1'), ('1', 'rad'), ('percent', '1'), ('1', 'percent')])
def test_unknown_dimensional_angle_scaled_declarations_do_not_get_convention(argument, output):
    # Call only export helper, avoiding unrelated canonical-template validation.
    model = object.__new__(CompositeFamily)
    model.native = Native()
    model.template = declaration(argument, output)
    model.slot_name = 'h'
    model.export_slot = lambda native, c, reduced=False: exported(c)
    slot = model.slot_equation([3., 4.])
    assert all(p['unit'] is None and 'unit_origin' not in p for p in slot['parameters'].values())


@pytest.mark.parametrize('unit', ['1', 'm', 'rad', 'percent'])
def test_explicit_parameter_unit_is_never_overwritten(unit):
    model = object.__new__(CompositeFamily)
    model.native = Native()
    model.template = declaration()
    model.slot_name = 'h'
    model.export_slot = lambda native, c, reduced=False: exported(c, unit=unit)
    assert all(p['unit'] == unit and 'unit_origin' not in p for p in model.slot_equation([3., 4.])['parameters'].values())


def test_foreign_unknown_coefficient_origin_not_inferred():
    slot = family(origin='external_unknown').slot_equation([3., 4.])
    assert all(p['unit'] is None for p in slot['parameters'].values())


def test_fixed_zero_bounds_and_outer_coefficient_are_preserved():
    model = object.__new__(CompositeFamily)
    model.native = Native()
    model.template = declaration()
    model.template['parameters']['outer'] = {'value': 7., 'trainable': False, 'unit': '1'}
    model.slot_name = 'h'
    raw = exported([0., 4.], origin='native_reduction_inactive_zero')
    raw['parameters']['c0'].update(trainable=False, bounds=[0., 0.])
    model.export_slot = lambda native, c, reduced=False: deepcopy(raw)
    before = deepcopy(model.template)
    result = model.slot_equation([0., 4.], reduced=True)
    assert result['parameters']['c0']['trainable'] is False
    assert result['parameters']['c0']['bounds'] == [0., 0.]
    assert result['parameters']['c0']['value'] == 0.
    assert result['parameters']['c0']['origin'] == 'native_reduction_inactive_zero'
    assert model.template == before
    assert raw['parameters']['c0']['unit'] is None


def test_same_numeric_predictions_gradient_and_normalization_as_unknown_unit_control():
    torch = pytest.importorskip('torch')
    model, control = family(), family(output=None)
    x = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    c = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    d = c.detach().clone().requires_grad_()
    y, z = model.predict(c, x), control.predict(d, x)
    torch.testing.assert_close(y, z, rtol=0, atol=0)
    y.sum().backward()
    z.sum().backward()
    torch.testing.assert_close(c.grad, d.grad, rtol=0, atol=0)
    assert model.groups == control.groups == {'denominator_0': [0, 1]}
