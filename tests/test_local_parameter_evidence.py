"""Local coefficient evidence is numerical, population-preserving, and not uniqueness."""
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic.equation_evaluation.numerical import local_parameter_evidence

VARS = [{"id": "x", "column": "x", "unit": "1"}]
TARGET = {"unit": "1"}


def op(name, *args):
    return {"op": name, "args": list(args)}


def record(ast, **parameters):
    return {
        "kind": "explicit",
        "equation": {"schema_version": 2, "ast": ast, "parameters": {
            name: {"value": value, "trainable": True, "unit": "1", "origin": "user-declared"}
            for name, value in parameters.items()
        }},
    }


def test_independent_and_redundant_coefficients_keep_source_immutable():
    X = np.arange(1., 8.)[:, None]
    source = record(op("add", op("mul", {"param": "a"}, {"var": "x"}), {"param": "b"}), a=2., b=3.)
    original = deepcopy(source)
    evidence = local_parameter_evidence(source, X, VARS, TARGET)
    assert evidence["status"] == "computed"
    assert evidence["rank"] == 2 and evidence["full_column_rank"]
    assert evidence["n_samples"] == len(X)
    assert evidence["parameter_names"] == ["a", "b"]
    assert [p["scale"] for p in evidence["parameters"]] == [2., 3.]
    assert all(p["step_plus"] > 0 and p["step_minus"] > 0 for p in evidence["parameters"])
    assert source == original
    source["equation"]["ast"] = op("mul", op("add", {"param": "a"}, {"param": "b"}), {"var": "x"})
    evidence = local_parameter_evidence(source, X, VARS, TARGET)
    assert evidence["rank"] == 1 and evidence["nullity"] == 1
    assert evidence["condition_number"] is None
    assert "global or structural" in evidence["interpretation_limit"]


def test_insensitive_and_underdetermined_designs():
    source = record(op("add", op("mul", {"param": "a"}, {"var": "x"}), {"param": "b"}), a=2., b=3.)
    evidence = local_parameter_evidence(source, np.zeros((5, 1)), VARS, TARGET)
    assert evidence["insensitive_parameters"] == ["a"]
    assert evidence["rank"] == 1
    evidence = local_parameter_evidence(source, np.ones((1, 1)), VARS, TARGET)
    assert evidence["nullity"] == 1 and not evidence["full_column_rank"]


def test_bound_probes_ties_and_fixed_parameters():
    source = record(op("add", {"param": "a"}, {"param": "alias"}), a=0., alias=0.)
    source["equation"]["parameters"]["alias"].update(trainable=False, tied_to="a", bounds=[0., 4.])
    original = deepcopy(source)
    evidence = local_parameter_evidence(source, np.ones((4, 1)), VARS, TARGET)
    assert evidence["status"] == "computed"
    assert evidence["parameter_names"] == ["a"]
    assert evidence["parameters"][0]["active_bound"] == "lower"
    assert evidence["parameters"][0]["step_minus"] == 0
    assert evidence["parameters"][0]["bounds"] == [0., 4.]
    assert evidence["column_rms_sensitivity"]["a"] == pytest.approx(8.)
    assert source == original
    source["equation"]["parameters"]["a"]["trainable"] = False
    assert local_parameter_evidence(source, np.ones((4, 1)), VARS, TARGET)["status"] == "not_applicable"


def test_invalid_rows_or_probes_are_not_dropped_and_missing_audit_is_explicit():
    source = record(op("sqrt", op("add", {"param": "a"}, {"var": "x"})), a=0.)
    evidence = local_parameter_evidence(source, np.array([[0.], [1.]]), VARS, TARGET)
    assert evidence["status"] == "unavailable"
    assert evidence["n_samples"] == 2 and evidence["n_valid_center"] == 2
    assert "probe invalid" in evidence["reason"]
    assert "rank" not in evidence
    evidence = local_parameter_evidence(source, np.array([[-1.], [1.]]), VARS, TARGET)
    assert evidence["status"] == "unavailable" and evidence["n_valid_center"] == 1
    del source["equation"]["parameters"]["a"]["origin"]
    assert "provenance" in local_parameter_evidence(source, np.ones((3, 1)), VARS, TARGET)["reason"]


@pytest.mark.parametrize("offset", [1e6, 1e8, 1e12])
@pytest.mark.parametrize("redundant", [False, True])
def test_output_roundoff_never_becomes_zero_or_spurious_full_rank(offset, redundant):
    coefficient = op("add", {"param": "a"}, {"param": "b"}) if redundant else {"param": "a"}
    source = record(op("add", {"const": offset}, op("mul", coefficient, {"var": "x"})), a=2., b=3.)
    evidence = local_parameter_evidence(source, np.arange(1., 11.)[:, None], VARS, TARGET)
    assert evidence["status"] == "unavailable"
    assert "numerical resolution" in evidence["reason"]
    assert evidence["roundoff_resolution"] > evidence["rank_threshold"]
    assert "rank" not in evidence and "full_column_rank" not in evidence
    assert "insensitive_parameters" not in evidence
    assert evidence["n_samples"] == 10


@pytest.mark.parametrize("frequency", [1., 1000., 10000., 100000.])
def test_half_step_check_catches_nonlinear_false_full_rank(frequency):
    source = record(op("sin", op("mul", {"const": frequency}, op("mul",
        op("add", {"param": "a"}, {"param": "b"}), {"var": "x"}))), a=2., b=3.)
    evidence = local_parameter_evidence(source, np.linspace(.1, 1., 10)[:, None], VARS, TARGET)
    if frequency == 1:
        assert evidence["status"] == "computed" and evidence["rank"] == 1
    else:
        assert evidence["status"] == "unavailable"
        assert evidence["half_step_difference"] > evidence["rank_threshold"]
        assert "rank" not in evidence
    assert all(p["verification_step_plus"] < p["step_plus"] for p in evidence["parameters"])


def test_exact_zero_output_and_zero_factor_design_remain_distinct_from_unresolved():
    source = record(op("mul", {"param": "a"}, {"var": "x"}), a=2.)
    evidence = local_parameter_evidence(source, np.zeros((5, 1)), VARS, TARGET)
    assert evidence["status"] == "computed" and evidence["rank"] == 0
    assert evidence["insensitive_parameters"] == ["a"]
    assert evidence["numerical_resolution"] == 0
