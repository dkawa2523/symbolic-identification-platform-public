"""Real native custom-function roundtrip, including where upstream defaults differ."""

import numpy as np
import pytest


@pytest.mark.native
def test_parfam_public_function_extension_uses_ordinary_semantics(tmp_path):
    import parfam.parfamwrapper as wrapper
    import parfam.train as native
    import torch
    from ml_platform_symbolic.data import write_json
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.parfam import _export, discover
    from ml_platform_symbolic.expressions import evaluate

    original_extension = wrapper.extend_function_dict
    original_functions, original_names = native.function_dict, native.function_name_dict
    checks = []

    def observe(functions, names):
        result = original_extension(functions, names)
        for index, name in enumerate(("exp", "log", "sqrt")):
            alias = f"function_{index}"
            assert native.function_dict[alias] is getattr(torch, name)
            parameters = {
                "degree_input_polynomials": 1,
                "degree_output_polynomials": 1,
                "degree_input_denominator": 0,
                "degree_output_denominator": 0,
                "degree_output_polynomials_specific": [1],
                "degree_output_polynomials_denominator_specific": [1],
                "maximal_potence": 2,
                "functions": [alias],
                "function_names": [alias],
                "enforce_function": False,
                "stabilize_denominator": False,
            }
            model = native.setup_model(parameters, 1, "cpu", "ParFamTorch")
            coefficients = torch.ones(model.get_number_parameters(), dtype=torch.float64)
            X = np.array([[-2.0], [0.2], [12.0]])
            prediction = (
                model.predict(coefficients, torch.as_tensor(X, dtype=torch.float64)).detach().numpy().reshape(-1)
            )
            variables = [{"id": "x0", "column": "x", "unit": "1"}]
            equation = _export(model, coefficients.numpy(), variables, [name])
            portable = evaluate(equation, X, variables)
            assert np.array_equal(np.isfinite(prediction), np.isfinite(portable))
            finite = np.isfinite(prediction)
            np.testing.assert_allclose(prediction[finite], portable[finite], rtol=1e-12, atol=1e-12)
            if name == "exp":
                assert prediction[-1] > np.exp(10)  # No upstream exp clipping.
            else:
                assert not finite[0]  # No upstream sqrt(abs)/log(abs) substitution.
            checks.append(
                {
                    "function": name,
                    "native": [float(x) if np.isfinite(x) else None for x in prediction],
                    "same_invalid_mask": True,
                }
            )
        return result

    X = np.linspace(0.2, 2, 36).reshape(-1, 1)
    req = DiscoveryRequest(
        X,
        0.31 + 1.43 * np.exp(-0.81 * X[:, 0]),
        np.arange(len(X)),
        [{"id": "x0", "column": "x", "unit": "1", "domain": [0.2, 3.0]}],
        {"column": "y", "unit": "1"},
        ["add", "sub", "mul", "div", "exp", "log", "sqrt"],
        {"max_nodes": 60, "max_wall_seconds": 90},
        0,
        {"functions": ["exp", "log", "sqrt"], "max_active_functions": 1, "seconds": 8, "maxiter": 3},
    )
    wrapper.extend_function_dict = observe
    try:
        result = discover(req, DiscoveryContext(tmp_path))
    finally:
        wrapper.extend_function_dict = original_extension
    assert native.function_dict is original_functions and native.function_name_dict is original_names
    assert len(checks) == 3 and result.summary["native_fit_completed"]
    write_json(
        tmp_path / "ordinary_functions_control.json",
        {
            "checks": checks,
            "native_completed_family_fits": result.summary["native_completed_family_fits"],
            "faithful_exports": len(result.candidates),
            "scope": "real native function kernels and native continuous search, not evidence that every function family was searched or recovered",
        },
    )
