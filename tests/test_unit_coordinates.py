"""Scientific coordinate regressions: values, domains, immutable coefficients and signal."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.constraints import dimensions
from ml_platform_symbolic.data import load_partition, prepare, read_json
from ml_platform_symbolic.engines.base import DiscoveryRequest, checked_candidate
from ml_platform_symbolic.expressions import evaluate, inspect_equation, parse_expression
from ml_platform_symbolic.model import load_model, save_model
from ml_platform_symbolic.parameters import with_parameters
from ml_platform_symbolic.stage import run_task
from ml_platform_symbolic.units import (
    canonical_unit,
    conversion,
    convert_difference,
    convert_values,
    unit_dimensions,
    unit_spec,
)


@pytest.mark.parametrize(
    "label,alias", [("m/s2", "(m/s2)"), ("kg/m3", "(kg/m3)"), ("rad/s", "(rad/s)"), ("m^(1/3)", "(m**(1/3))")]
)
def test_aliases_compose_at_all_depths(label, alias):
    assert unit_spec(label) == unit_spec(alias)
    assert unit_dimensions(f"({label})/({alias})") == (0, 0, 0, 0)


def test_standard_value_difference_and_angle_coordinates():
    assert convert_values(1, "day", "s") == 86400
    assert convert_values(1, "MPa", "Pa") == 1e6
    assert convert_values(0, "degC", "K") == 273.15
    assert convert_difference(2, "degC", "K") == 2
    assert convert_values(2, "K", "delta_degC", role="difference") == 2
    assert np.sin(convert_values(90, "deg", "rad")) == pytest.approx(1)
    assert convert_values(1, "cm^(1/3)", "m^(1/3)") == pytest.approx(0.01 ** (1 / 3))
    with pytest.raises(ValueError, match="quantity"):
        conversion("Hz", "rad/s")
    assert canonical_unit("degC") == "K"
    assert canonical_unit("Hz") == "Hz"
    assert unit_dimensions(None) is None
    assert unit_dimensions("1") == (0, 0, 0, 0)


def test_fixed_scalar_exponents_not_free_fitted_values():
    variables = [{"id": "x0", "column": "length", "unit": "m"}]
    equation = parse_expression("x0**(1/3)", variables, trainable=False)
    assert dimensions(equation, variables, {"unit": "m^(1/3)"})[0] == "pass"
    assert unit_dimensions("m^(1/3)")[0] == Fraction(1, 3)
    # Runtime remains ordinary fractional power: no conversion into signed cbrt.
    assert np.isnan(evaluate(equation, np.array([[-8.0]]), variables)[0])
    for trainable, expected in [(False, "pass"), (True, "unknown")]:
        eq = parse_expression(
            "x0**n",
            variables,
            parameters={"n": {"value": 2, "trainable": trainable, "bounds": [1, 3], "unit": "1"}},
            trainable=False,
        )
        assert dimensions(eq, variables, {"unit": "m^2"})[0] == expected


def test_mixed_scale_and_degrees_do_not_pass_dimension_only_typing():
    variables = [{"id": "x0", "unit": "MPa"}, {"id": "x1", "unit": "Pa"}]
    equation = parse_expression("x0+x1", variables, trainable=False)
    assert dimensions(equation, variables, {"unit": "MPa"})[0] == "fail"
    frequencies = [{"id": "x0", "unit": "Hz"}, {"id": "x1", "unit": "rad/s"}]
    mixed = parse_expression("x0+x1", frequencies, trainable=False)
    assert dimensions(mixed, frequencies, {"unit": "Hz"})[0] == "fail"
    angle = [{"id": "x0", "unit": "deg"}]
    assert dimensions(parse_expression("sin(x0)", angle, trainable=False), angle, {"unit": "1"})[0] == "fail"
    assert (
        dimensions(
            parse_expression("asin(x0)", [{"id": "x0", "unit": "1"}], trainable=False),
            [{"id": "x0", "unit": "1"}],
            {"unit": "deg"},
        )[0]
        == "fail"
    )


def test_ties_require_identical_unit_coordinates_and_immutable_sources():
    p = {
        "a": {"value": 2.0, "trainable": True, "bounds": [1, 3], "unit": "m"},
        "b": {"value": 2.0, "trainable": False, "bounds": [1, 2.5], "unit": "s", "tied_to": "a"},
    }
    bad = {"schema_version": 1, "ast": {"param": "b"}, "parameters": p}
    with pytest.raises(ValueError, match="identical"):
        inspect_equation(bad, [])
    p["b"]["unit"] = "m"
    before = deepcopy(bad)
    changed = with_parameters(bad, ["a"], [2.25])
    assert changed["parameters"]["b"]["value"] == 2.25
    assert bad == before
    p["b"]["unit"] = "cm"
    with pytest.raises(ValueError, match="identical"):
        inspect_equation(bad, [])


def test_offset_dominated_fidelity_rejects_lost_signal():
    X = np.linspace(0, 1, 20)[:, None]
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0, 1]}]
    request = DiscoveryRequest(X, 1e9 + X[:, 0], np.arange(20), variables, {"unit": "1"}, ["add"], {}, 0, {})
    bad = parse_expression("1000000000", variables, trainable=False)
    with pytest.raises(ValueError, match="signal-scaled"):
        checked_candidate("test", 0, bad, request, lambda x: 1e9 + x[:, 0])
    good = parse_expression("1000000000+x0", variables, trainable=False)
    assert checked_candidate("test", 0, good, request, lambda x: 1e9 + x[:, 0]).native_check["passed"]
    from ml_platform_symbolic.numerical_fidelity import compare_predictions

    result = compare_predictions(np.array([1e15, 1e15 + 1]), np.array([1e15, 1e15 + 1]), scale=0.5)
    assert not result["passed"] and result["resolution_limited"]


@pytest.mark.parametrize("factor", [1e-200, 1e-20, 1.0, 1e20, 1e200])
def test_signal_fidelity_is_scale_metamorphic_without_physical_floor(factor):
    from ml_platform_symbolic.numerical_fidelity import signal_scale

    X = np.linspace(0.1, 1, 20)[:, None]
    variables = [{"id": "x0", "column": "x", "unit": "1", "domain": [0.1, 1]}]
    request = DiscoveryRequest(X, factor * X[:, 0], np.arange(20), variables, {"unit": "1"}, ["mul"], {}, 0, {})
    wrong = parse_expression("0", variables, trainable=False)
    with pytest.raises(ValueError, match="signal-scaled"):
        checked_candidate("test", 0, wrong, request, lambda x: factor * x[:, 0])
    good = parse_expression(
        "a*x0",
        variables,
        trainable=False,
        parameters={"a": {"value": factor, "trainable": False, "bounds": [factor, factor], "unit": "1"}},
    )
    assert checked_candidate("test", 0, good, request, lambda x: factor * x[:, 0]).native_check["passed"]
    assert signal_scale(request.y) / factor == pytest.approx(np.std(X[:, 0]))


def test_csv_units_freeze_before_split_and_roundtrip_prediction(synthetic_config, tmp_path):
    cfg = deepcopy(synthetic_config)
    frame = pd.read_csv(cfg["data"]["local_path"])
    frame["x"] = np.linspace(0, 1, len(frame))  # days
    frame["y"] = 2 * frame["x"]  # metres, velocity2m/day
    frame.to_csv(tmp_path / "days.csv", index=False)
    cfg["data"].update(
        local_path=str(tmp_path / "days.csv"), units={"x": "day", "y": "m"}, domain={"x": [0, 1], "y": [0, 2]}
    )
    cfg["constraints"].update(
        probes=[{"x": 0.5}],
        response_bounds=[0, 2],
        response_checks=[
            {"feature": "x", "order": 1, "sign": 1, "step": 0.1, "tolerance": 0.01, "points": [{"x": 0.5}]}
        ],
    )
    resolved = resolve_config(cfg)
    assert resolved["data"]["units"] == {"x": "s", "y": "m"}
    assert resolved["data"]["domain"]["x"] == [0, 86400]
    assert resolved["constraints"]["probes"] == [{"x": 43200}]
    assert resolved["constraints"]["response_checks"][0]["step"] == 8640
    assert resolved["constraints"]["response_checks"][0]["tolerance"] == pytest.approx(0.01 / 86400)
    assert resolve_config(resolved) == resolved
    refs = prepare(resolved, tmp_path / "prepared")
    metadata = read_json(refs["run"])
    train, _, ids = load_partition(refs["train"])
    assert len(ids) == metadata["partitions"]["train"]["n_rows"]
    assert train.max() <= 86400
    schema = {
        "variables": metadata["variables"],
        "target": metadata["target"],
        "feature_ranges": [[0, 86400]],
        "public_units": resolved["data"]["source_units"],
    }
    equation = parse_expression(
        "v*x0",
        schema["variables"],
        trainable=False,
        parameters={"v": {"value": 2 / 86400, "trainable": False, "bounds": [2 / 86400, 2 / 86400], "unit": "m/s"}},
    )
    path = save_model(tmp_path / "model.zip", equation, schema, {})
    model = load_model(path)
    assert model.predict(pd.DataFrame({"x": [0.5]}))[0] == pytest.approx(1)
    assert model.predict(pd.DataFrame({"x": [43200]}), units={"x": "s"}, output_unit="cm")[0] == pytest.approx(100)
    with pytest.raises(ValueError, match="domain"):
        model.predict(pd.DataFrame({"x": [2]}))
    with pytest.raises(ValueError, match="keys"):
        model.predict(pd.DataFrame({"x": [0.5]}), units={"xx": "day"})
    # CLI/stage honors explicitly supplied units, rather than ignoring day/s.
    pd.DataFrame({"x": [43200]}).to_csv(tmp_path / "infer.csv", index=False)
    result = run_task(
        {
            "task": "symbolic_infer",
            "model": {"local_model_path": str(path)},
            "data": {"local_path": str(tmp_path / "infer.csv"), "units": {"x": "s", "y": "cm"}},
        },
        run_dir=tmp_path / "infer",
    )
    assert pd.read_csv(result.tables["predictions"]).prediction.iloc[0] == pytest.approx(100)


def test_affine_inputs_and_unit_key_typos(synthetic_config, tmp_path):
    cfg = deepcopy(synthetic_config)
    cfg["data"]["units"] = {"xx": "m", "y": "m"}
    with pytest.raises(ValueError, match="keys"):
        resolve_config(cfg)
    cfg["data"]["units"] = {"x": "degC", "y": "MPa"}
    cfg["data"]["coordinate_roles"] = {"x": "value"}
    cfg["data"]["domain"] = {"x": [-10, 100], "y": [0, 3]}
    cfg["evaluation"].update(absolute_tolerance=0.1, acceptable_rmse=0.2)
    resolved = resolve_config(cfg)
    assert resolved["data"]["units"] == {"x": "K", "y": "Pa"}
    assert resolved["data"]["domain"]["x"] == [263.15, 373.15]
    assert resolved["data"]["domain"]["y"] == [0, 3e6]
    assert resolved["evaluation"]["absolute_tolerance"] == 1e5
    assert resolved["evaluation"]["acceptable_rmse"] == 2e5


def test_physical_speed_uses_day_input_once_and_radians_are_converted(tmp_path):
    variables = [{"id": "x0", "column": "time", "unit": "s", "domain": [0, 1e6]}]
    parameters = {"v": {"value": 2.0, "trainable": False, "bounds": [2, 2], "unit": "m/s"}}
    equation = parse_expression("v*x0", variables, parameters=parameters, trainable=False)
    schema = {
        "variables": variables,
        "target": {"column": "distance", "unit": "m"},
        "public_units": {"time": "day", "distance": "m"},
        "feature_ranges": [[0, 1e6]],
    }
    model = load_model(save_model(tmp_path / "speed.zip", equation, schema, {}))
    assert model.predict(pd.DataFrame({"time": [1.0]}))[0] == 172800
    variables = [{"id": "x0", "column": "angle", "unit": "rad", "domain": [0, np.pi]}]
    equation = parse_expression("sin(x0)", variables, trainable=False)
    schema = {
        "variables": variables,
        "target": {"column": "response", "unit": "1"},
        "public_units": {"angle": "deg", "response": "1"},
        "feature_ranges": [[0, np.pi]],
    }
    model = load_model(save_model(tmp_path / "sine.zip", equation, schema, {}))
    assert model.predict(pd.DataFrame({"angle": [90.0]}))[0] == pytest.approx(1.0)
    assert model.predict(pd.DataFrame({"angle": [np.pi / 2]}), units={"angle": "rad"})[0] == pytest.approx(1.0)
    bad = parse_expression("x0", variables, trainable=False)
    with pytest.raises(ValueError, match="mismatch"):
        save_model(tmp_path / "mislabeled.zip", bad, {**schema, "target": {"unit": "deg"}}, {})


def test_ensemble_and_implicit_use_the_same_saved_coordinate_boundary(tmp_path):
    from ml_platform_symbolic import load_ensemble, load_relation
    from ml_platform_symbolic.data import write_json

    variables = [{"id": "x0", "column": "temperature", "unit": "K", "domain": [250, 350], "coordinate_role": "value"}]
    schema = {
        "variables": variables,
        "target": {"column": "response", "unit": "K", "domain": [250, 350], "coordinate_role": "value"},
        "public_units": {"temperature": "degC", "response": "degC"},
        "feature_ranges": [[250, 350]],
    }
    first = parse_expression("x0", variables, trainable=False)
    second = parse_expression(
        "x0+d",
        variables,
        trainable=False,
        parameters={"d": {"value": 2, "trainable": False, "bounds": [2, 2], "unit": "delta_K"}},
    )
    path = write_json(
        tmp_path / "ensemble.json",
        {
            "schema_version": 1,
            "kind": "structural_point_ensemble",
            "schema": schema,
            "members": [{"weight": 0.5, "equation": first}, {"weight": 0.5, "equation": second}],
            "operators": ["add"],
            "constraints": {"max_nodes": 30, "require_units": True},
            "metadata": {"origin": "manual boundary example; no native Bayesian or uncertainty claim"},
        },
    )
    ensemble = load_ensemble(path)
    assert "no source-size guarantee" in ensemble.metadata["complexity_scope"]
    frame = pd.DataFrame({"temperature": [0.0]})
    assert ensemble.predict(frame)[0] == pytest.approx(1)
    np.testing.assert_allclose(ensemble.structural_quantiles(frame, [0.1, 0.9]), [[0, 2]])
    assert ensemble.predict(pd.DataFrame({"temperature": [273.15]}), units={"temperature": "K"}, output_unit="K")[
        0
    ] == pytest.approx(274.15)
    residual = parse_expression("target-x0", [*variables, {"id": "target", "unit": "K"}], trainable=False)
    path = write_json(
        tmp_path / "relation.json",
        {
            "schema_version": 1,
            "kind": "implicit_relations",
            "schema": schema,
            "members": [{"candidate_id": "identity", "residual": residual, "predictor": first}],
            "metadata": {"origin": "manual boundary example; not native search evidence"},
        },
    )
    relation = load_relation(path)
    assert relation.residual(frame, [0.0])[0] == 0
    assert (
        relation.residual(
            pd.DataFrame({"temperature": [273.15]}), [273.15], units={"temperature": "K"}, target_unit="K"
        )[0]
        == 0
    )
    assert relation.predict(frame)[0] == 0


@pytest.mark.parametrize("unit", ["K", "m", "s", "1"])
def test_point_difference_semantics_are_not_temperature_or_name_rules(unit):
    from ml_platform_symbolic.parameters import expression_coordinate_role
    from ml_platform_symbolic.units import Unit, coordinate_conversion, quantity_conversion, unit_spec

    parameters = {
        name: {"value": value, "unit": unit, "coordinate_role": "value", "trainable": True, "bounds": [0, 400]}
        for name, value in [("left", 300), ("right", 290)]
    }
    eq = parse_expression("left-right", [], parameters=parameters, trainable=False)
    assert expression_coordinate_role(eq, []) == "difference"
    if unit == "K":
        a, b = quantity_conversion(unit_spec("K"), unit_spec("degC"), "difference")
        assert evaluate(eq, np.empty((1, 0)), [])[0] * a + b == 10.0
        assert unit_spec("K") != unit_spec("delta_K")
        assert unit_spec("delta_K*1").role == "difference"
    source = unit_spec(unit)
    shifted = Unit(source.dimensions, scale=2.0, offset=100.0, angle=source.angle, cycle=source.cycle)
    a, b = coordinate_conversion(source, shifted, role="difference")
    assert evaluate(eq, np.empty((1, 0)), [])[0] * a + b == 5.0
    for parameter in eq["parameters"].values():
        parameter.pop("coordinate_role")
    assert expression_coordinate_role(eq, []) is None
    with pytest.raises(ValueError, match="explicit"):
        quantity_conversion(source, shifted, None)
    eq["parameters"]["left"]["coordinate_role"] = "value"
    eq["parameters"]["right"].update(value=300.0, coordinate_role="difference", trainable=False, tied_to="left")
    with pytest.raises(ValueError, match="point/difference"):
        inspect_equation(eq, [])


def test_generic_affine_values_increments_variance_derivatives_and_ratios():
    from ml_platform_symbolic.units import (
        ZERO,
        Unit,
        convert_derivative,
        convert_ratio,
        convert_variance,
        convert_width,
    )

    shifted = Unit(ZERO, scale=-2.0, offset=100.0)
    canonical = Unit(ZERO)
    assert convert_values(3.0, shifted, canonical) == 94.0
    assert convert_values(3.0, shifted, canonical, role="difference") == -6.0
    assert convert_difference(3.0, shifted, canonical) == -6.0
    assert convert_width(3.0, shifted, canonical) == 6.0
    assert convert_variance(3.0, shifted, canonical) == 12.0
    output = Unit((-1, 1, -2, 0), scale=3.0, offset=-7.0)
    pressure = Unit((-1, 1, -2, 0))
    assert convert_derivative(4.0, shifted, canonical, output, pressure) == -6.0
    assert convert_derivative(4.0, shifted, canonical, output, pressure, order=2) == 3.0
    with pytest.raises(ValueError, match="origin"):
        convert_ratio(2.0, (shifted, canonical), (shifted, canonical))
    assert convert_ratio(2.0, ("cm", "m"), ("s", "day")) == pytest.approx(1728.0)


def test_nonthermal_affine_data_roles_freeze_for_portable_inference(synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic.units import ATOMS, ZERO, Unit

    monkeypatch.setitem(ATOMS, "sensor_x", Unit(ZERO, scale=2.0, offset=10.0))
    monkeypatch.setitem(ATOMS, "sensor_y", Unit(ZERO, scale=3.0, offset=-7.0))
    raw = pd.read_csv(synthetic_config["data"]["local_path"])
    raw["x"] = np.linspace(1, 3, len(raw))
    raw["y"] = 2 * raw["x"]
    raw.to_csv(tmp_path / "sensor.csv", index=False)
    models = []
    for role in ("value", "difference"):
        cfg = deepcopy(synthetic_config)
        cfg["data"].update(
            local_path=str(tmp_path / "sensor.csv"),
            units={"x": "sensor_x", "y": "sensor_y"},
            coordinate_roles={"x": role, "y": role},
            domain={"x": [1, 3], "y": [2, 6]},
        )
        cfg["constraints"].update(
            probes=[{"x": 2}],
            response_bounds=[2, 6],
            response_checks=[
                {"feature": "x", "order": 1, "sign": 1, "step": 0.1, "tolerance": 0.2, "points": [{"x": 2}]}
            ],
        )
        resolved = resolve_config(cfg)
        refs = prepare(resolved, tmp_path / role)
        info = read_json(refs["run"])
        assert resolved["data"]["domain"]["x"] == ([12.0, 16.0] if role == "value" else [2.0, 6.0])
        assert resolved["constraints"]["response_checks"][0]["tolerance"] == pytest.approx(0.3)
        eq = parse_expression(
            "3*x0+b",
            info["variables"],
            trainable=False,
            parameters={
                "b": {
                    "value": -37.0 if role == "value" else 0.0,
                    "trainable": False,
                    "bounds": [None, None],
                    "unit": "1",
                }
            },
        )
        schema = {
            "variables": info["variables"],
            "target": info["target"],
            "public_units": resolved["data"]["source_units"],
            "unit_coordinates": info["unit_coordinates"],
            "feature_ranges": info["feature_ranges"],
        }
        path = save_model(
            tmp_path / f"model_{role}.zip",
            eq,
            schema,
            {"origin": "synthetic calibration boundary; no SR discovery claim"},
        )
        models.append(path)
    monkeypatch.delitem(ATOMS, "sensor_x")
    monkeypatch.delitem(ATOMS, "sensor_y")
    for path in models:
        model = load_model(path)
        np.testing.assert_allclose(model.predict(pd.DataFrame({"x": [1.0, 2.0, 3.0]})), [2.0, 4.0, 6.0])


@pytest.mark.parametrize(
    "label,expected",
    [("m/(s)^2", (1, 0, -2, 0)), ("1/(K)", (0, 0, 0, -1)), ("kg*m/s^2", (1, 1, -2, 0)), ("m^(1/2)", (0.5, 0, 0, 0))],
)
def test_explicit_unit_products_are_dimensions_not_conversion(label, expected):
    assert unit_dimensions(label) == expected


@pytest.mark.parametrize("label", [None, {}, "__import__('os')", "m^999", "m^(1/0)", "unknown", "2*m"])
def test_unknown_or_invalid_units_are_not_inferred(label):
    assert unit_dimensions(label) is None


def test_rational_root_dimensions_do_not_round_to_a_different_unit():
    variables = [{"id": "x0", "unit": "m"}]
    cube_root = {"schema_version": 1, "ast": {"op": "rootn", "args": [{"var": "x0"}, {"const": 3}]}, "parameters": {}}
    assert dimensions(cube_root, variables, {"unit": "m^(1/3)"})[0] == "pass"
    nested_root = {**cube_root, "ast": {"op": "sqrt", "args": [cube_root["ast"]]}}
    assert dimensions(nested_root, variables, {"unit": "m^(1/6)"})[0] == "pass"
