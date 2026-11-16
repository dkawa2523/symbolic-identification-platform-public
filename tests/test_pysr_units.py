"""Exact export metadata, never physical identification or additional fitting."""
from copy import deepcopy

import numpy as np
import pytest

from ml_platform_symbolic.constraints import dimensions
from ml_platform_symbolic.engines.pysr import export_expression
from ml_platform_symbolic.engines.pysr_units import infer_coefficient_units
from ml_platform_symbolic.expressions import evaluate


@pytest.mark.parametrize('text,xunit,yunit,expected', [
    ('2.0*x0', 'm', 'm', {'c0': '1'}),
    ('x0+2.0', 'm', 'm', {'c0': 'm'}),
    ('2.0*x0+3.0', 'm', 'm', {'c0': '1', 'c1': 'm'}),
    ('exp(2.0*x0)', 'm', '1', {'c0': 'm^(-1)'}),
    ('2.0*square(x0)', 'm', 'm', {'c0': 'm^(-1)'}),
    ('2.0', 'm', 'm', {'c0': 'm'}),
    ('x0', 'm', 'm', {}),
])
def test_unique_structural_units_do_not_change_formula_or_prediction(text, xunit, yunit, expected):
    variables = [{'id': 'x0', 'column': 'x', 'unit': xunit}]
    equation = export_expression(text, variables)
    original = deepcopy(equation)
    result = infer_coefficient_units(equation, variables, {'unit': yunit})
    assert result['status'] == 'unique' and result['units'] == expected
    assert equation == original
    for name, label in result['units'].items():
        equation['parameters'][name]['unit'] = label
    assert dimensions(equation, variables, {'unit': yunit})[0] == 'pass'
    X = np.linspace(.1, 2, 21)[:, None]
    np.testing.assert_array_equal(evaluate(equation, X, variables), evaluate(original, X, variables))


@pytest.mark.parametrize('text,unit,status', [
    ('2.0*(3.0*x0)', 'm', 'underdetermined'),
    ('2.0*(3.0*x0)', '1', 'underdetermined'),
    ('x0+exp(2.0*x0)', 'm', 'inconsistent'),
    ('x0^2.0', 'm', 'unsupported'),
    ('x0^2.0', '1', 'unsupported'),
    ('2.0*x0', 'rad', 'unsupported'),
    ('2.0*x0', 'Hz*s', 'unsupported'),
    ('2.0*x0', 'degC', 'unsupported'),
    ('2.0*x0', 'cm', 'unsupported'),
    ('2.0*x0', None, 'unsupported'),
])
def test_inference_does_not_guess_unsupported_or_nonunique_units(text, unit, status):
    variables = [{'id': 'x0', 'column': 'x', 'unit': unit}]
    equation = export_expression(text, variables)
    original = deepcopy(equation)
    result = infer_coefficient_units(equation, variables, {'unit': unit})
    assert result['status'] == status
    assert equation == original


def test_known_parameter_units_are_constraints_not_overwritten():
    variables = [{'id': 'x0', 'column': 'x', 'unit': 'm'}]
    equation = export_expression('2.0*x0+3.0', variables)
    equation['parameters']['c0']['unit'] = 's'
    assert infer_coefficient_units(equation, variables, {'unit': 'm'})['status'] == 'inconsistent'
    assert equation['parameters']['c0']['unit'] == 's'


@pytest.mark.parametrize('mode', [None, 'hard', 'soft'])
def test_adapter_applies_metadata_only_when_native_unit_search_was_used(tmp_path, monkeypatch, mode):
    import sys
    from types import SimpleNamespace
    import pandas as pd
    from ml_platform_symbolic.engines import pysr as adapter
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest

    class NativeDouble:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y, **kwargs):
            assert ('X_units' in kwargs) == (mode is not None)
            self.equations_ = pd.DataFrame({'equation': ['x0 / -0.5'], 'loss': [0.0], 'complexity': [3]})
        def predict(self, X, index):
            return X[:, 0] / -0.5

    monkeypatch.setitem(sys.modules, 'pysr', SimpleNamespace(PySRRegressor=NativeDouble))
    monkeypatch.setattr(adapter, 'version', lambda _: 'test-double')
    X = np.array([[-1.0], [0.0], [1.0]])
    request = DiscoveryRequest(X, -2*X[:, 0], np.arange(3),
        [{'id': 'x0', 'column': 'x', 'unit': 'm'}], {'column': 'y', 'unit': 'm'},
        ['div'], {'max_nodes': 9, 'max_wall_seconds': 30}, 17, {'unit_mode': mode})
    result = adapter.discover(request, DiscoveryContext(tmp_path))
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.equation['parameters']['c0']['unit'] == (None if mode is None else '1')
    assert candidate.native_check['passed']
    assert bool(result.summary['coefficient_unit_inference']) == (mode is not None)
