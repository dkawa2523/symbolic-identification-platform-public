"""Actual native math regression; no search/checkpoint and no display-string oracle."""

import numpy as np
import pytest
from ml_platform_symbolic.engines.parfam import _export
from ml_platform_symbolic.expressions import evaluate

torch = pytest.importorskip("torch")


@pytest.mark.native
def test_native_rational_function_export_and_ordinary_division_factory():
    from parfam.train import setup_model

    torch.set_default_dtype(torch.float64)
    parameters = {
        "degree_input_polynomials": 1,
        "degree_output_polynomials": 1,
        "degree_input_denominator": 1,
        "degree_output_denominator": 1,
        "degree_output_polynomials_specific": [1],
        "degree_output_polynomials_denominator_specific": [1],
        "functions": ["sin"],
        "function_names": ["sin"],
        "maximal_potence": 2,
        "enforce_function": False,
        "stabilize_denominator": False,
    }
    model = setup_model(parameters, 1, "cpu", "ParFamTorch")
    assert not model.stabilize
    count = model.n_coefficients_first_layer + model.n_coefficients_last_layer
    coefficients = torch.linspace(0.2, 1.1, count)
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0.2, 2.0]}]
    equation = _export(model, coefficients.numpy(), variables, ["sin"])
    X = np.random.default_rng(18).uniform(0.2, 2.0, (17, 1))
    native = model.predict(coefficients, torch.as_tensor(X)).detach().numpy()
    np.testing.assert_allclose(evaluate(equation, X, variables), native, rtol=1e-12, atol=1e-12)
    assert any(p["trainable"] for p in equation["parameters"].values())
