"""Source-specific export regressions; no model download or native execution."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines.flash_ansr import export_candidate
from ml_platform_symbolic.expressions import evaluate

VARIABLES = [{"id": "x0", "column": "length", "unit": "1"}]


def native(tokens, slots=(), values=()):
    return SimpleNamespace(expression=tokens, slots=list(slots), constants=list(values))


def test_native_slots_keep_full_precision_and_unknown_units():
    equation = export_candidate(native(["*", "x1", "<constant>"], [2], [1.23456789123456]), VARIABLES)
    assert equation["parameters"]["c0"]["value"] == 1.23456789123456
    assert equation["parameters"]["c0"]["unit"] is None
    assert equation["parameters"]["c0"]["trainable"]


def test_native_division_hole_is_not_cancelled():
    equation = export_candidate(native(["/", "x1", "x1"]), VARIABLES)
    values = evaluate(equation, np.array([[0.0], [2.0]]), VARIABLES)
    assert np.isnan(values[0]) and values[1] == 1


@pytest.mark.parametrize(
    "candidate",
    [
        native(["rootn", "x1", "<constant>"], [2], [3]),
        native(["x2"]),
        native(["x1", "1"]),
    ],
)
def test_unportable_native_structure_is_rejected(candidate):
    with pytest.raises(ValueError):
        export_candidate(candidate, VARIABLES)


def test_fitted_real_exponent_and_signed_odd_root_are_preserved():
    equation = export_candidate(native(["pow", "x1", "<constant>"], [2], [1.37]), VARIABLES)
    assert equation["parameters"]["c0"]["trainable"]
    np.testing.assert_allclose(evaluate(equation, np.array([[0.2], [2.0]]), VARIABLES), np.array([0.2, 2.0]) ** 1.37)
    root = export_candidate(native(["rootn", "x1", "3"]), VARIABLES)
    np.testing.assert_allclose(evaluate(root, np.array([[-8.0], [27.0]]), VARIABLES), [-2.0, 3.0])


def test_ieee_export_roundtrip_keeps_real_domain_holes(tmp_path):
    import json

    from ml_platform_symbolic.expressions import evaluate_native

    equation = export_candidate(native(["inv", "inv", "x1"]), VARIABLES)
    path = tmp_path / "equation.json"
    path.write_text(json.dumps(equation))
    restored = json.loads(path.read_text())
    points = np.array([[0.0], [2.0], [-2.0]])
    assert restored["arithmetic"] == "numpy_ieee"
    np.testing.assert_array_equal(evaluate_native(restored, points, VARIABLES), [0.0, 2.0, -2.0])
    common = evaluate(restored, points, VARIABLES)
    assert np.isnan(common[0])
    np.testing.assert_array_equal(common[1:], [2.0, -2.0])


def test_context_operator_routes_do_not_claim_integer_power_generation():
    from ml_platform_symbolic.engines.flash_ansr import context_operator_routes
    routes = context_operator_routes(["pow_int", "sqrt"], True)
    assert not routes["pow"]["token_allowed"]
    assert not routes["rootn"]["token_allowed"]
    assert routes["masked_compatible_routes"] == ["pow", "rootn"]
    assert context_operator_routes(["pow_int"], False)["pow"]["token_allowed"]


@pytest.mark.native
def test_pinned_emission_default_and_explicit_format_leave_refiner_policy_unchanged():
    from flash_ansr import RefineConfig, SoftmaxSamplingConfig
    default = SoftmaxSamplingConfig(draws=8)
    explicit = SoftmaxSamplingConfig(draws=8, emission="fittable")
    assert default.to_kwargs() == explicit.to_kwargs()
    constants = SoftmaxSamplingConfig(draws=8, emission="constants").to_kwargs()
    assert {k: v for k, v in constants.items() if k != "emission"} == {
        k: v for k, v in default.to_kwargs().items() if k != "emission"}
    refinement = RefineConfig().as_dict()
    assert refinement["scope"] == "fittable"
    assert refinement["typed_spans"] == "freeze_then_free"
