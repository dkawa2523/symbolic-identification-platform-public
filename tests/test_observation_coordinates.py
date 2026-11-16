"""CSV identity and representability at shared public coordinate boundaries."""

import gzip
import sys

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import model_matrix, model_output, prepare, read_observations
from ml_platform_symbolic.units import apply_coordinate, convert_values


@pytest.mark.parametrize("name", ["x", "y", "sample_id", "group"])
def test_duplicate_csv_headers_reject_read_prepare_and_cli(name, synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic.__main__ import main

    source = tmp_path / "ambiguous.csv"
    source.write_text(f"sample_id,x,y,group,{name}\na,1,2,g,3\n")
    with pytest.raises(ValueError, match="Duplicate CSV"):
        read_observations(source)
    synthetic_config["data"]["local_path"] = str(source)
    with pytest.raises(ValueError, match="Duplicate CSV"):
        prepare(resolve_config(synthetic_config), tmp_path / "prepared")
    monkeypatch.setattr(
        sys,
        "argv",
        ["symbolic", "predict", "unused.zip", "--data", str(source), "--run-dir", str(tmp_path / "prediction")],
    )
    with pytest.raises(ValueError, match="Duplicate CSV"):
        main()
    assert not (tmp_path / "prediction").exists()


@pytest.mark.parametrize("header", ["sample_id,,x", "sample_id, ,x", 'sample_id,"x', "sample_id,x\x00"])
def test_empty_or_malformed_csv_headers_reject(header, tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text(header + "\na,1,2\n")
    with pytest.raises(ValueError, match="header"):
        read_observations(source)


def test_quoted_unique_names_bom_and_extra_columns_remain_literal(tmp_path):
    source = tmp_path / "quoted.csv"
    source.write_text('\ufeffsample_id,"x,one","x""two",extra\n001,1,2,notes\n', encoding="utf-8")
    frame = read_observations(source)
    assert list(frame.columns) == ["sample_id", "x,one", 'x"two', "extra"]
    assert frame.sample_id.tolist() == ["001"]
    assert frame.extra.tolist() == ["notes"]


def schema():
    return {"variables": [{"id": "x0", "column": "x", "unit": "s"}], "target": {"column": "y", "unit": "s"}}


@pytest.mark.parametrize("frozen", [False, True])
def test_inference_input_output_and_widths_reject_loss_to_zero(frozen):
    saved = schema()
    if frozen:
        saved["unit_coordinates"] = {
            "columns": {
                "x": {"source_unit": "ms", "problem_unit": "s", "scale": 0.001, "offset": 0},
                "y": {"source_unit": "h", "problem_unit": "s", "scale": 3600, "offset": 0},
            }
        }
    with pytest.raises(ValueError, match="underflow"):
        model_matrix(pd.DataFrame({"x": [1e-322]}), saved, units={"x": "ms"})
    for difference in (False, True):
        with pytest.raises(ValueError, match="underflow"):
            model_output([1e-322], saved, "h", difference=difference)


@pytest.mark.parametrize("column", ["x", "y"])
def test_prepare_checks_feature_and_target_conversion(column, synthetic_config, tmp_path):
    source = synthetic_config["data"]["local_path"]
    frame = pd.read_csv(source)
    frame[column] = 1e-322
    frame.to_csv(source, index=False)
    synthetic_config["data"].update(units={"x": "s", "y": "s"}, source_units={column: "ms"})
    with pytest.raises(ValueError, match="underflow"):
        prepare(resolve_config(synthetic_config), tmp_path / "prepared")


def test_subnormal_affine_zero_cancellation_and_invalid_masks():
    assert convert_values(1e-310, "ms", "s") > 0
    assert convert_values(-273.15, "degC", "K") == 0
    assert convert_values(273.15, "K", "degC") == 0
    assert apply_coordinate(273.15, 1, 273.15, inverse=True) == 0
    assert apply_coordinate(-2.0, 0.5, 1.0) == 0
    result = model_output([0, np.nan, np.inf, -np.inf], schema(), "h")
    assert result[0] == 0 and np.isnan(result[1]) and np.isposinf(result[2]) and np.isneginf(result[3])
    with pytest.raises(ValueError, match="nonfinite"):
        convert_values(1e308, "h", "s")


def test_canonical_bounds_use_checked_conversion(synthetic_config):
    synthetic_config["data"].update(units={"x": "ms", "y": "s"}, domain={"x": [1e-322, 1]})
    with pytest.raises(ValueError, match="underflow"):
        resolve_config(synthetic_config)


def test_calibration_target_uses_the_same_coordinate_boundary():
    from types import SimpleNamespace

    from ml_platform_symbolic.prediction_intervals import _prediction_observations

    model = SimpleNamespace(schema=schema(), working_prediction=lambda frame, units: np.zeros(len(frame)))
    frame = pd.DataFrame({"x": [1.0], "y": [1e-322]})
    with pytest.raises(ValueError, match="underflow"):
        _prediction_observations(model, frame, {"y": "ms"})


def test_compressed_csv_preserves_unique_headers_and_rejects_duplicates(tmp_path):
    source = tmp_path / "observations.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write('   \nsample_id,"x,one",extra\n001,1,notes\n')
    frame = read_observations(source)
    assert list(frame.columns) == ["sample_id", "x,one", "extra"]
    assert frame.sample_id.tolist() == ["001"]
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write("sample_id,x,x\n001,1,2\n")
    with pytest.raises(ValueError, match="Duplicate CSV"):
        read_observations(source)
