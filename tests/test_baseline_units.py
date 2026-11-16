"""Known units of target statistics, without interpreting constructed coefficient units as a physical mechanism."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.constraints import dimensions
from ml_platform_symbolic.engines.dictionary import discover


@pytest.mark.parametrize("target_unit", ["m", "1/s"])
def test_mean_has_target_units_even_when_inputs_have_different_units(target_unit):
    request = SimpleNamespace(
        X=np.linspace(1, 4, 20)[:, None],
        y=np.linspace(3, 9, 20),
        seed=0,
        options={},
        variables=[{"id": "x0", "column": "temperature", "unit": "K", "domain": [1, 4]}],
        target={"unit": target_unit},
        operators=["add", "mul"],
        limits={},
    )
    result = discover(request, None, name="mean")
    assert result.candidates[0].native_check["passed"]
    assert dimensions(result.candidates[0].equation, request.variables, request.target)[0] == "pass"
    linear = discover(request, None, name="linear")
    parameters = linear.candidates[-1].equation["parameters"].values()
    assert any(p["unit"] == target_unit for p in parameters)
    assert all(p["unit"] is not None for p in parameters)
    assert dimensions(linear.candidates[-1].equation, request.variables, request.target)[0] == "pass"
    assert all(p.get("unit_origin") for p in parameters)
