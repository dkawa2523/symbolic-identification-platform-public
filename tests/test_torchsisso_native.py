"""One actual native sparse-descriptor roundtrip on X never supplied to fitting."""

import numpy as np
import pandas as pd
import pytest
from current_flow import explicit_choice, study_flow
from ml_platform_symbolic import finalize_hypothesis, load_model
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.equation_evaluation import load_study


@pytest.mark.native
def test_one_descriptor_export_failure_does_not_discard_other_models(tmp_path, monkeypatch):
    pytest.importorskip("TorchSisso")
    from ml_platform_symbolic.engines import torchsisso_oracle
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.torchsisso import discover

    X = np.random.default_rng(40).uniform(0.2, 3, (45, 3))
    y = 0.3 + 1.2 * X[:, 0] * X[:, 1] + 0.02 * X[:, 2]
    variables = [{"id": f"x{i}", "column": f"x{i}", "unit": "1", "domain": [0.1, 4]} for i in range(3)]
    request = DiscoveryRequest(
        X,
        y,
        np.arange(len(X)),
        variables,
        {"unit": "1"},
        ["add", "mul"],
        {"max_nodes": 100},
        17,
        {"n_expansion": 2, "n_term": 2, "k": 6},
    )
    original = torchsisso_oracle.descriptor_values
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError("Unreviewed descriptor")
        return original(*args, **kwargs)

    monkeypatch.setattr(torchsisso_oracle, "descriptor_values", fail_once)
    result = discover(request, DiscoveryContext(tmp_path))
    assert len(result.candidates) == 1
    assert result.summary["rejected_exports"] == [{"index": 0, "reason": "Unreviewed descriptor"}]
    assert result.candidates[0].candidate_id.endswith("/1")


@pytest.mark.native
def test_native_descriptor_oracle_keeps_singular_rows_and_batch_meaning():
    torch = pytest.importorskip("torch")
    pytest.importorskip("TorchSisso")
    from ml_platform_symbolic.engines.torchsisso import _equation
    from ml_platform_symbolic.engines.torchsisso_oracle import descriptor_values
    from ml_platform_symbolic.expressions import evaluate, evaluate_native

    variables = [{"id": "x0", "column": "x0"}, {"id": "x1", "column": "x1"}]
    X = np.array([[1.0, 2.0], [0.0, 0.0], [-1.0, 0.0], [2.0, -0.0], [2.0, 0.0], [3.0, 4.0]])
    names = ["(ln(x0))", "(x0/x1)", "(exp(ln(x0)))", "((x0/x1))**2"]
    actual = descriptor_values(names, X, variables).numpy()
    tensor = torch.as_tensor(X)
    expected = torch.stack(
        [
            torch.log(tensor[:, 0]),
            tensor[:, 0] / tensor[:, 1],
            torch.exp(torch.log(tensor[:, 0])),
            (tensor[:, 0] / tensor[:, 1]) ** 2,
        ],
        dim=1,
    ).numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    # Constant child columns and one-row batches must not disappear.
    for index in range(len(X)):
        np.testing.assert_allclose(
            descriptor_values(names, X[index : index + 1], variables).numpy(), actual[index : index + 1], equal_nan=True
        )
    equation = _equation([names[2]], [1.0], 0.0, variables)
    equation["schema_version"] = 2
    equation["arithmetic"] = "numpy_ieee"
    np.testing.assert_allclose(evaluate_native(equation, X, variables), actual[:, 2], equal_nan=True)
    assert actual[1, 2] == 0  # Native IEEE finite recovery is preserved only in raw parity.
    assert np.isnan(evaluate(equation, X, variables)[1])  # Not valid real-domain prediction.


@pytest.mark.native
def test_native_log_model_exports_without_claiming_zero_is_valid(tmp_path):
    pytest.importorskip("TorchSisso")
    x = np.linspace(0.2, 5, 80)
    frame = pd.DataFrame({"x": x, "z": np.sin(x) + 3, "y": 1.234 + 2.345 * np.log(x)})
    frame["sample_id"] = [f"r{i}" for i in range(len(frame))]
    frame.to_csv(tmp_path / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 50 + ["selection"] * 20 + ["final"] * 10).to_csv(
        tmp_path / "split.csv", index=False
    )
    _, discoveries, study = study_flow(
        {
            "data": {
                "local_path": str(tmp_path / "table.csv"),
                "target_column": "y",
                "feature_columns": ["x", "z"],
                "units": {k: "1" for k in ["x", "z", "y"]},
            },
            "split": {"file": str(tmp_path / "split.csv")},
            "search": {
                "engines": ["torchsisso"],
                "operators": ["add", "mul", "log"],
                "max_wall_seconds": 60,
                "engine_options": {"torchsisso": {"n_expansion": 2, "n_term": 1, "k": 4}},
            },
        },
        root=tmp_path / "run", purpose="local_prediction", engineering_tolerance=100,
    )
    _, decision = explicit_choice(study, tmp_path / "run")
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / "run/final-audit")
    _, records, rows, _ = load_study(study)
    selected = next(i for i, record in enumerate(records) if record["id"] == result.extra["selected_id"])
    model = load_model(result.artifacts["model_bundle"])
    summary = read_json(discoveries[0]["summary"])
    assert not summary["rejected_exports"]
    from ml_platform_symbolic.expressions import evaluate

    values = evaluate(
        model.equation,
        np.array([[1.0, 3.0], [0.0, 3.0], [-1.0, 3.0]]),
        [{"id": "x0", "column": "x"}, {"id": "x1", "column": "z"}],
    )
    np.testing.assert_allclose(values[0], 1.234, atol=1e-10)
    assert np.isnan(values[1:]).all()


@pytest.mark.native
def test_fresh_native_descriptors_match_reloaded_sparse_model(tmp_path):
    torch = pytest.importorskip("torch")
    constructor = pytest.importorskip("TorchSisso.FeatureSpaceConstruction").feature_space_construction
    rng = np.random.default_rng(1927)
    X = rng.uniform(0.2, 2.0, (72, 4))
    columns = [f"feature_{i}" for i in range(4)]
    frame = pd.DataFrame(X, columns=columns)
    frame["sample_id"] = [f"r{i}" for i in range(len(frame))]
    frame["y"] = 0.37654321 + 1.234567891 * X[:, 0] * X[:, 1] + rng.normal(0, 0.01, len(X))
    frame.to_csv(tmp_path / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 40 + ["selection"] * 12 + ["final"] * 20).to_csv(
        tmp_path / "split.csv", index=False
    )
    _, discoveries, study = study_flow(
        {
            "data": {
                "local_path": str(tmp_path / "table.csv"),
                "target_column": "y",
                "feature_columns": columns,
                "units": {name: "1" for name in [*columns, "y"]},
                "domain": {**{name: [0.2, 3] for name in columns}, "y": [0, 20]},
            },
            "split": {"file": str(tmp_path / "split.csv")},
            "search": {
                "engines": ["torchsisso"],
                "operators": ["add", "mul"],
                "max_wall_seconds": 60,
                "engine_options": {"torchsisso": {"n_expansion": 2, "n_term": 2, "k": 8}},
            },
        },
        root=tmp_path / "run", purpose="local_prediction", engineering_tolerance=100,
    )
    _, decision = explicit_choice(study, tmp_path / "run")
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / "run/final-audit")
    _, records, rows, _ = load_study(study)
    selected = next(i for i, record in enumerate(records) if record["id"] == result.extra["selected_id"])
    model = load_model(result.artifacts["model_bundle"])
    summary = read_json(discoveries[0]["summary"])
    assert summary["validation_probe_count"] > 0
    native_models = read_json(summary["native_artifacts"]["native_models"])["models"]
    rank = int(records[selected]["candidate_id"].split("/")[-1])
    native = native_models[rank]
    unseen = rng.uniform(0.2, 2.5, (17, 4))
    native_frame = pd.DataFrame(unseen, columns=[v["id"] for v in model.schema["variables"]])
    native_frame.insert(0, "target", np.arange(len(unseen), dtype=float))
    expanded, _, names = constructor(["+", "*"], native_frame, 2, "cpu").feature_space()
    features = expanded[:, [names.index(name) for name in native["descriptors"]]]
    expected = features @ torch.as_tensor(native["coefficients"], dtype=features.dtype) + native["intercept"]
    probe = pd.DataFrame(unseen, columns=columns)
    np.testing.assert_allclose(model.predict(probe), expected.numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.native
def test_dimensional_native_search_and_fresh_predictions(tmp_path):
    """Native dimensional expansion/regression; compound units and nuisance dimensions."""
    torch = pytest.importorskip("torch")
    sp = pytest.importorskip("sympy")
    constructor = pytest.importorskip("TorchSisso.DimensionalFeatureSpaceConstruction").feature_space_construction
    regressor = pytest.importorskip("TorchSisso.Regressor_dimension").Regressor
    rng = np.random.default_rng(48271)
    columns = ["speed", "duration", "offset", "mass", "temperature", "nuisance_length"]
    X = rng.uniform(0.2, 2.0, (100, len(columns)))
    frame = pd.DataFrame(X, columns=columns)
    frame["sample_id"] = [f"d{i}" for i in range(len(X))]
    frame["distance"] = 0.37 + 1.43 * X[:, 0] * X[:, 1] - 0.81 * X[:, 2] + rng.normal(0, 0.02, len(X))
    frame.to_csv(tmp_path / "table.csv", index=False)
    frame[["sample_id"]].assign(partition=["train"] * 60 + ["selection"] * 20 + ["final"] * 20).to_csv(
        tmp_path / "split.csv", index=False
    )
    _, discoveries, study = study_flow(
        {
            "data": {
                "local_path": str(tmp_path / "table.csv"),
                "target_column": "distance",
                "feature_columns": columns,
                "units": dict(zip([*columns, "distance"], ["m/s", "s", "m", "kg", "K", "m", "m"], strict=True)),
                "domain": {**{name: [0.2, 3.0] for name in columns}, "distance": [-5, 15]},
            },
            "split": {"file": str(tmp_path / "split.csv")},
            "search": {
                "engines": ["torchsisso"],
                "operators": ["add", "sub", "mul", "div"],
                "max_wall_seconds": 60,
                "engine_options": {"torchsisso": {"dimensional": True, "n_expansion": 2, "n_term": 2, "k": 8}},
            },
            "constraints": {"require_units": True},
        },
        root=tmp_path / "run", purpose="local_prediction", engineering_tolerance=100,
    )
    _, decision = explicit_choice(study, tmp_path / "run")
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / "run/final-audit")
    _, records, rows, _ = load_study(study)
    selected = next(i for i, record in enumerate(records) if record["id"] == result.extra["selected_id"])
    model = load_model(result.artifacts["model_bundle"])
    summary = read_json(discoveries[0]["summary"])
    native = read_json(summary["native_artifacts"]["native_models"])
    assert summary["dimensional"] and summary["native_candidates"] == 2
    assert {m["term_count"] for m in native["models"]} == {1, 2}
    assert model.equation["parameters"]["intercept"]["unit"] == "m"
    assert all(p["unit"] == "1" for k, p in model.equation["parameters"].items() if k != "intercept")
    assert any(c["name"] == "units" and c["status"] == "pass" for c in rows[selected]["checks"])

    # Independently invoke the actual dimensional constructor and regressor. No
    # adapter helper or shared AST computes the oracle; native display precision
    # is sufficient here as a SECOND check, not the production coefficient export.
    L, M, T, K = sp.symbols("L M T K", positive=True)
    units = [L / T, T, L, M, K, L]

    def features(values, target_values):
        df = pd.DataFrame(values, columns=[f"x{i}" for i in range(len(columns))])
        df.insert(0, "target", target_values)
        builder = constructor(
            df, operators=["+", "-", "*", "/"], no_of_operators=2, dimensionality=[str(u) for u in units]
        )
        builder.dimensionality = list(units)  # Native expansion appends descriptor dimensions in place.
        return builder.feature_expansion()

    expanded, y, names, dimensions = features(X[:60], frame.distance.to_numpy()[:60])
    search = regressor(expanded, y, names, dimensions, dimension=2, sis_features=8, output_dim=L)
    _, _, _, equations = search.regressor_fit()
    rank = int(records[selected]["candidate_id"].split("/")[-1])
    unseen = rng.uniform(0.25, 2.75, (17, len(columns)))
    fresh, _, fresh_names, _ = features(unseen, np.arange(len(unseen), dtype=float))
    state = native["models"][rank]
    selected = fresh[:, [fresh_names.index(name) for name in state["descriptors"]]]
    expected = selected @ torch.as_tensor(state["coefficients"], dtype=fresh.dtype) + state["intercept"]
    native_function = sp.lambdify(
        [sp.Symbol(f"x{i}") for i in range(len(columns))], sp.sympify(equations[rank]), "numpy"
    )
    np.testing.assert_allclose(expected.numpy(), native_function(*unseen.T), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        model.predict(pd.DataFrame(unseen, columns=columns)), expected.numpy(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.native
def test_torchsisso_saturated_generated_descriptor_does_not_break_higher_terms():
    torch = pytest.importorskip("torch")
    pytest.importorskip("TorchSisso")
    from ml_platform_symbolic.engines.torchsisso import usable_descriptors
    from TorchSisso.Regressor import Regressor

    torch.set_num_threads(1)
    temperature = torch.linspace(300, 360, 40, dtype=torch.float64)
    concentration = torch.as_tensor(np.random.default_rng(0).uniform(0.5, 2, 40), dtype=torch.float64)
    expanded = torch.stack([temperature, concentration, torch.tanh(temperature)], dim=1)
    target = 1.23456789 * temperature + 0.57123 * concentration + 0.3
    features, names, _, removed = usable_descriptors(expanded, ["x0", "x1", "tanh(x0)"])
    regressor = Regressor(features, target, names, 2, 2, "cpu", False)
    regressor.regressor_fit()
    assert removed == ["tanh(x0)"]
    assert len(regressor.portable_models_) == 2
    model = regressor.portable_models_[-1]
    prediction = (
        features[:, model["indices"]] @ torch.as_tensor(model["coefficients"], dtype=torch.float64) + model["intercept"]
    )
    np.testing.assert_allclose(prediction.numpy(), target.numpy(), rtol=1e-10, atol=1e-10)
