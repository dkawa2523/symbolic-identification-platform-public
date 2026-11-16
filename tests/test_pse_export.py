"""Full native coefficients, fixed exponents, and the observed C1/C10/x1/x10 bugs."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines.pse import _export_fit, stage_configuration
from ml_platform_symbolic.expressions import evaluate


def test_pse_export_preserves_native_slots_and_fixed_power():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    record = {"template": "C0 + C1*x0**2", "coefficients": [0.371234567891, -1.91234567891]}
    equation = _export_fit(record, variables)
    assert list(equation["parameters"]) == ["C0", "C1"]
    assert [p["value"] for p in equation["parameters"].values()] == record["coefficients"]
    X = np.array([[0.2], [1.3], [2.7]])
    np.testing.assert_allclose(
        evaluate(equation, X, variables), record["coefficients"][0] + record["coefficients"][1] * X[:, 0] ** 2
    )


@pytest.mark.native
def test_actual_psrn_fit_handles_multidigit_coefficients_and_variables():
    pytest.importorskip("psrn")
    from psrn.model.regressor import PSRN_Regressor, symbolic_handoff_constant_version
    from psrn.utils.data import expr_to_Y_pred

    assert symbolic_handoff_constant_version == 2
    rng = np.random.default_rng(313)
    raw = rng.normal(size=(64, 12))
    X, _ = np.linalg.qr(raw - raw.mean(axis=0))
    coefficients = np.linspace(-1.91, 1.37, 12)
    y = 0.371234567891 + X @ coefficients
    names = [f"x{i}" for i in range(12)]
    # Directly exercise the actual pinned native fitting routine, not a common
    # solver. Full enumeration/search is verified separately by the benchmark.
    state = SimpleNamespace(use_replace_expo=False, fitted_expr_c_set=set())
    values, loss, template, expression = PSRN_Regressor.fit_LS(
        state, "+".join(names), X, y[:, None], names, 1.0, True, False
    )
    assert values is not None and loss < 1e-12
    assert len(values) > 11  # Both C1 and C10 are genuinely substituted by native code.
    variables = [{"id": n, "column": n, "unit": "1"} for n in names]
    equation = _export_fit({"template": str(template), "coefficients": values.tolist()}, variables)
    probe = rng.normal(scale=0.2, size=(20, 12))
    np.testing.assert_allclose(
        evaluate(equation, probe, variables), expr_to_Y_pred(expression, probe, names).reshape(-1), atol=1e-12
    )
    np.testing.assert_allclose(evaluate(equation, probe, variables), 0.371234567891 + probe @ coefficients, atol=1e-6)


@pytest.mark.native
def test_native_pareto_observation_retains_identity_after_numeric_formatting():
    pytest.importorskip("psrn")
    from psrn.model.regressor import PSRN_Regressor

    ops = ["Add", "Mul", "Identity"]
    native = PSRN_Regressor(
        variables=["x0"],
        operators=ops,
        device="cpu",
        n_symbol_layers=1,
        n_inputs=2,
        n_sample_variables=1,
        use_const=True,
        use_dr_mask=False,
        token_generator_config={"base": {"has_const": True, "tokens": ops}},
        stage_config={"default": {"operators": ops, "n_psrn_inputs": 2}, "stages": [{}]},
    )
    native.pareto_frontier = []
    native.is_positive, native.use_strict_pareto, native.use_threshold, native.probe = False, True, False, None
    captured = []
    native._symbolic_record_pareto = lambda original, retained: captured.append((original, retained))
    original = "(-0.37996805682668755)+(0.15878907976192044)*x0+1.4366225694518202"
    native.pareto_update_and_check([(original, 0.9, 0.01, 3)])
    assert captured == [(original, native.get_pf()[0][0])]
    assert captured[0][0] != captured[0][1]  # Observe the changed native spelling, do not guess equivalence keys.


@pytest.mark.native
def test_real_gp_cosh_and_cube_match_native_tensor_and_portable_math():
    torch = pytest.importorskip("torch")
    pytest.importorskip("psrn")
    from ml_platform_symbolic.engines.pse import ordinary_gp_operators
    from ml_platform_symbolic.expressions import parse_expression
    from psrn.model.functions import Cosh, Pow3
    from psrn.model.token_generator.GP.model.expr_utils import utils

    original = utils.get_expression
    X = np.array([[-2.0], [-0.5], [0.3], [1.5]])
    variables = [{"id": "x0", "column": "x0", "unit": "1"}]
    with ordinary_gp_operators():
        # Exercise the actual GP dictionary constructor used by native set_input.
        definitions = utils.expression_dict(["Cosh", "Pow3"], 1, False)
        for name, tensor_class in [("Cosh", Cosh), ("Pow3", Pow3)]:
            expression = next(v for v in definitions.values() if v.str_name == name)
            prediction = expression.func(X)
            tensor = tensor_class(1, "cpu").forward(torch.as_tensor(X)).numpy()
            portable = evaluate(parse_expression(expression.str_func("x0"), variables, trainable=False), X, variables)
            np.testing.assert_allclose(prediction, tensor, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(prediction[:, 0], portable, rtol=1e-14, atol=1e-14)
    assert utils.get_expression is original
    with pytest.raises(RuntimeError, match="injected native failure"), ordinary_gp_operators():
        raise RuntimeError("injected native failure")
    assert utils.get_expression is original


def test_native_stages_resolve_only_declared_inputs_and_operators():
    request = SimpleNamespace(
        operators=["add", "mul", "exp", "pow_int"],
        variables=[{}, {}],
        limits={"max_wall_seconds": 90},
        options={
            "stages": [
                {"operators": ["add", "mul"], "seconds": 20},
                {"operators": ["add", "mul", "exp", "pow_int"], "slots": 4, "seconds": 30},
            ]
        },
    )
    stages = stage_configuration(request)
    assert stages[1]["operators"] == ["Add", "Mul", "Exp", "Pow2", "Pow3", "Identity"]
    assert stages[1]["n_sample_variables"] == 2
    request.options["stages"][1]["operators"].append("sin")
    with pytest.raises(ValueError, match="subset"):
        stage_configuration(request)
