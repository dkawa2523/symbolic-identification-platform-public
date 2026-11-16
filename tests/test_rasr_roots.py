"""Actual pinned RASR numerical root semantics, without loading neural weights or fitting."""

from importlib import metadata
from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines.rasr import export_candidate
from ml_platform_symbolic.expressions import evaluate

pytestmark = pytest.mark.native


@pytest.mark.parametrize("degree", [3, 4, 5])
def test_real_rasr_root_export_preserves_negative_and_zero_boundaries(degree):
    try:
        version = metadata.version("simplipy")
    except metadata.PackageNotFoundError:
        pytest.skip("Pinned RASR image only")
    if version != "0.2.14":
        pytest.skip("Pinned RASR SimpliPy 0.2.14 only")
    import simplipy.operators as operators

    variables = [{"id": "x0", "column": "drive", "unit": "1", "domain": [-32.0, 32.0]}]
    candidate = SimpleNamespace(tokens=[f"pow1_{degree}", "x1"], constants=[])
    fitted = SimpleNamespace(X_mean_=np.zeros(1), X_std_=np.ones(1), y_mean_=0.0, y_std_scale_=1.0)
    equation = export_candidate(candidate, fitted, variables, {"unit": "1"})
    X = np.asarray([-32.0, -8.0, -1.0, -0.0, 0.0, 1.0, 8.0, 32.0])[:, None]
    with np.errstate(all="ignore"):
        native = getattr(operators, f"pow1_{degree}")(X[:, 0].astype(np.float32))
        portable = evaluate(equation, X, variables)
    np.testing.assert_array_equal(np.isfinite(native), np.isfinite(portable))
    np.testing.assert_allclose(native, portable, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert not any(p["trainable"] for p in equation["parameters"].values())
