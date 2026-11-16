import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("benchmark_nist", Path(__file__).parents[1] / "scripts/benchmark_nist.py")
nist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nist)


@pytest.mark.parametrize("name", nist.CASES)
def test_certified_formula_reproduces_published_objective(name):
    parameters, X, y, rss, lines = nist.parse_nist((nist.DATA / f"{name}.dat").read_bytes())
    prediction = nist.evaluate(nist.equation(nist.CASES[name][0], parameters, 2), X, nist.VARIABLES)
    assert np.sum((prediction - y) ** 2) == pytest.approx(rss, rel=1e-9)
    assert lines["data"][1] - lines["data"][0] + 1 == len(y)


def test_parser_rejects_missing_observation():
    raw = (nist.DATA / "Misra1a.dat").read_bytes()
    with pytest.raises(ValueError):
        nist.parse_nist(b"\n".join(raw.splitlines()[:-1]))


def test_agreement_keeps_objective_and_prediction_distinct():
    metrics = nist.agreement(np.array([2.0, 3.0]), np.array([1.0, 2.0]), np.array([1.0, 3.0]), 1.0)
    assert metrics["rss_relative_error"] == 0
    assert metrics["prediction_relative_l2"] > 0


def test_scorer_preserves_actual_rows_for_readout():
    result = nist.run(max_nfev=1000)
    assert len(result["attempts"]) == 4
    for row in result["attempts"]:
        assert row["status"] == "agreement"
        saved = row["prediction_rows"]
        _, _, observed, _, _ = nist.parse_nist((nist.DATA / f"{row['dataset']}.dat").read_bytes())
        np.testing.assert_array_equal(saved["observed"], observed)
        np.testing.assert_allclose(saved["residual"], observed - np.array(saved["predicted"]))
        assert len(saved["valid"]) == row["rows"] and all(saved["valid"])
        assert "not held-out" in saved["partition"]
        assert "not a true physical law" in result["case_context"][row["dataset"]]["truth_status"]
