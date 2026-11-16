"""Pure exact checks; no discovery, native search or benchmark/Final data."""

import math
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.domains import rational_domain_check
from ml_platform_symbolic.expressions import evaluate


def c(v):
    return {"const": v}


def x():
    return {"var": "x"}


def op(n, *a):
    return {"op": n, "args": list(a)}


def minus(v):
    return op("sub", x(), c(v))


def eq(ast, **extra):
    return {"schema_version": 2, "ast": ast, "parameters": {}, **extra}


def variables(a=0.0, b=4.0):
    return [{"id": "x", "column": "x", "unit": "1", "domain": [a, b]}]


def check(ast, a=0.0, b=4.0, **kw):
    return rational_domain_check(eq(ast), variables(a, b), **kw)


@pytest.mark.parametrize(
    "ast",
    [
        x(),
        op("div", c(1), op("add", op("pow_int", x(), c(2)), c(1))),
        op("div", c(1), minus(5)),
        op("pow_int", minus(1), c(2)),
        op("pow_int", minus(5), c(-2)),
    ],
)
def test_safe_is_divisor_nonzero_only(ast):
    r = check(ast)
    assert r["status"] == "safe" and "IEEE_finiteness" in r["not_certified"]


@pytest.mark.parametrize(
    "den",
    [
        minus(0),
        minus(4),
        minus(1),
        op("pow_int", minus(1), c(2)),
        op("mul", op("pow_int", minus(1), c(2)), minus(3)),
        op("sub", x(), x()),
        c(0),
    ],
)
def test_endpoint_even_multiplicity_and_zero_polynomial(den):
    assert check(op("div", c(1), den))["status"] == "unsafe"


@pytest.mark.parametrize(
    "ast",
    [
        op("div", minus(1), minus(1)),
        op("div", c(0), minus(1)),
        op("mul", c(0), op("div", c(1), minus(1))),
        op("pow_int", op("div", c(1), minus(1)), c(0)),
        op("pow_int", minus(1), c(-2)),
        op("div", c(1), op("div", c(1), minus(1))),
    ],
)
def test_all_intermediates_not_algebraically_cancelled(ast):
    assert check(ast)["status"] == "unsafe"


def test_singleton():
    assert check(op("div", c(1), minus(1)), 1.0, 1.0)["status"] == "unsafe"
    assert check(op("div", c(1), minus(2)), 1.0, 1.0)["status"] == "safe"


@pytest.mark.parametrize("bounds", [(None, 4.0), (0.0, None)])
def test_open_ended_scope(bounds):
    assert check(x(), *bounds)["status"] == "out_of_scope"


@pytest.mark.parametrize("bounds", [(4.0, 0.0), (0.0, math.nan), (0.0, math.inf)])
def test_invalid_bounds(bounds):
    assert check(x(), *bounds)["status"] == "unknown"


@pytest.mark.parametrize("value", [None, math.nan, math.inf, True])
def test_unknown_parameter(value):
    e = eq(op("div", c(1), {"param": "a"}))
    e["parameters"]["a"] = {"value": value, "trainable": True, "unit": None}
    assert rational_domain_check(e, variables())["status"] == "unknown"


def test_fixed_candidate_value_not_unit_inference_and_no_mutation():
    e = eq(op("div", c(1), op("add", x(), {"param": "a"})))
    e["parameters"]["a"] = {"value": 1.0, "trainable": True, "unit": None}
    before = deepcopy(e)
    assert rational_domain_check(e, variables())["status"] == "safe"
    e["parameters"]["a"]["value"] = -1.0
    assert rational_domain_check(e, variables())["status"] == "unsafe"
    e["parameters"]["a"]["value"] = 1.0
    assert e == before


def test_multidimensional_transcendental_and_hidden_unsupported():
    assert (
        rational_domain_check(eq(x()), variables() + [{"id": "z", "column": "z", "domain": [0, 4]}])["status"]
        == "out_of_scope"
    )
    for ast in [op("exp", x()), op("sqrt", x()), op("mul", c(0), op("exp", x()))]:
        assert check(ast)["status"] == "out_of_scope"


def test_region_box_not_affine_bound_hints():
    e = eq(op("div", c(1), minus(1)))
    assert (
        rational_domain_check(e, variables(), region={"kind": "box", "columns": ["x"], "bounds": [[2, 3]]})["status"]
        == "safe"
    )
    affine = {"kind": "affine", "columns": ["x"], "bounds": [[0, 4]], "inequalities": {"matrix": [[-1]], "rhs": [-2]}}
    assert rational_domain_check(e, variables(), region=affine)["status"] == "out_of_scope"


def test_schema2_refs_sparse_zero_terms_and_cycles():
    e = eq(op("sub", {"ref": "bad"}, {"ref": "bad"}), definitions={"bad": op("div", c(1), minus(1))})
    assert rational_domain_check(e, variables())["status"] == "unsafe"
    block = {"poly": [{"coefficient": c(0), "powers": [[0, 1]]}], "args": [op("div", c(1), minus(1))]}
    assert check(block)["status"] == "unsafe"
    assert rational_domain_check(eq({"ref": "s"}, definitions={"s": {"ref": "s"}}), variables())["status"] == "unknown"


@pytest.mark.parametrize("kw", [{"max_nodes": 2}, {"max_degree": 1}, {"max_integer_bits": 2}, {"max_operations": 2}])
def test_explicit_budgets_unknown(kw):
    assert check(op("div", c(1), op("add", op("pow_int", x(), c(2)), c(1))), **kw)["status"] == "unknown"


def test_math_safe_does_not_imply_ieee_safe():
    cases = [
        op("mul", c(1e308), x()),
        op("div", c(1), op("mul", c(1e-308), c(1e-308))),
        op("div", c(1), op("sub", op("add", c(1e16), x()), c(1e16))),
    ]
    for ast in cases:
        assert check(ast, 1.0, 2.0)["status"] == "safe"
        assert not np.isfinite(evaluate(eq(ast), np.array([[2.0], [1.0]]), variables(1.0, 2.0))).all()


def test_binary64_not_rounded_display():
    root = math.nextafter(1.0, 2.0)
    ast = op("div", c(1), minus(root))
    assert check(ast, 1.0, 1.0)["status"] == "safe"
    assert check(ast, root, root)["status"] == "unsafe"


def test_sturm_against_constructed_repeated_root_families():
    rng = np.random.default_rng(17019)
    for _ in range(40):
        roots = rng.integers(-5, 6, size=4).tolist()
        den = c(1)
        for root in roots:
            den = op("mul", den, op("pow_int", minus(root), c(2)))
        den = op("mul", den, op("add", op("pow_int", x(), c(2)), c(1)))
        assert check(op("div", c(1), den))["status"] == ("unsafe" if any(0 <= r <= 4 for r in roots) else "safe")


def test_invalid_budget_is_caller_error():
    with pytest.raises(ValueError):
        check(x(), max_operations=0)


def test_integer_literals_are_exact_math_not_silently_rounded():
    value = 2**53 + 1
    ast = op("div", c(1), minus(value))
    assert check(ast, value, value)["status"] == "unsafe"
    assert check(ast, float(value), float(value))["status"] == "safe"
    assert not np.isfinite(evaluate(eq(ast), np.array([[float(value)]]), variables(float(value), float(value)))).all()


def test_irrational_and_repeated_irrational_roots():
    base = op("sub", op("pow_int", x(), c(2)), c(2))
    assert check(op("div", c(1), base), 1.4, 1.5)["status"] == "unsafe"
    assert check(op("div", c(1), op("pow_int", base, c(2))), 1.4, 1.5)["status"] == "unsafe"
    assert check(op("div", c(1), base), 1.0, 1.4)["status"] == "safe"


def test_unknown_region_fields_cannot_be_ignored():
    result = rational_domain_check(
        eq(x()), variables(), region={"kind": "box", "columns": ["x"], "bounds": [[0, 4]], "unknown_restriction": 1}
    )
    assert result["status"] == "unknown"


def test_invalid_container_schemas_are_unknown_not_safe():
    assert rational_domain_check(None, variables())["status"] == "unknown"
    assert rational_domain_check(eq(x()), None)["status"] == "unknown"
    assert rational_domain_check(eq(x()), variables(), region=[])["status"] == "unknown"
    assert rational_domain_check({**eq(x()), "parameters": []}, variables())["status"] == "unknown"
    assert rational_domain_check({**eq(x()), "parameters": {"a": None}}, variables())["status"] == "unknown"
