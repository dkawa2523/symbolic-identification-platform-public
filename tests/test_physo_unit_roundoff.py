"""Real pinned PhySO dimensional-analysis boundary; no neural result is mocked."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.native


def unit_program(child_units, *, additive=False):
    from physo.physym import functions

    count = 3 if additive else 2
    size = 7
    units = np.zeros((1, count, size))
    units[0, 1, 0] = child_units
    if additive:
        units[0, 2, 0] = 1.0
    children = np.zeros((1, count, 2), dtype=int)
    children[0, 0] = [1, 2 if additive else 1]
    behavior = functions.UNIT_BEHAVIORS_DICT["BINARY_ADDITIVE_OP" if additive else "UNARY_DIMENSIONLESS_OP"].behavior_id
    tokens = SimpleNamespace(
        idx=np.arange(count).reshape(1, count),
        arity=np.array([[2, 0, 0] if additive else [1, 0]]),
        behavior_id=np.array([[behavior, 0, 0] if additive else [behavior, 0]]),
        phy_units=units,
        is_constraining_phy_units=np.ones((1, count), dtype=bool),
        children_pos=children,
    )
    return SimpleNamespace(tokens=tokens, dummy_idx=-1), np.array([[0], [0]]), np.array([[0], [count - 1]])


@pytest.mark.parametrize("residual", [0.0, 2.220446049250313e-16])
def test_dimensionless_child_accepts_machine_cancellation(residual):
    from physo.physym.dimensional_analysis import assign_units_bottom_up

    assign_units_bottom_up(*unit_program(residual))


@pytest.mark.parametrize("dimension", [1.0, 1e-6, np.nan])
def test_dimensionless_child_still_rejects_nonzero_or_unknown_units(dimension):
    from physo.physym.dimensional_analysis import assign_units_bottom_up

    with pytest.raises(AssertionError, match="Non-dimensionless token"):
        assign_units_bottom_up(*unit_program(dimension))


def test_additive_children_accept_only_roundoff_not_different_dimensions():
    from physo.physym.dimensional_analysis import assign_units_bottom_up

    assign_units_bottom_up(*unit_program(1.0 + 2.220446049250313e-16, additive=True))
    with pytest.raises(AssertionError, match="different physical units"):
        assign_units_bottom_up(*unit_program(2.0, additive=True))
