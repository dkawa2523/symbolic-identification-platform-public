"""Release boundary counterexamples: one coordinate authority and data-only failure summaries."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import audit_prediction_interval, load_prediction_interval, run_task
from ml_platform_symbolic.data import digest, model_output, public_unit, read_json, write_json
from ml_platform_symbolic.hypotheses import save_prediction
from ml_platform_symbolic.implicit import ImplicitRelation
from ml_platform_symbolic.model import save_model


@pytest.mark.parametrize("case", ["scale_map", "scale_legacy", "affine", "difference", "affine_numeric"])
def test_interval_observation_and_summary_share_saved_coordinates(tmp_path, case):
    working, source_x, source_y, role = "m", "cm", "mm", "value"
    x, y, scale_x, scale_y, offset_x, offset_y, width = 100.0, 1000.0, 0.01, 0.001, 0.0, 0.0, 0.1
    if case in {"affine", "difference"}:
        working, source_x, source_y = "K", "degC", "degC"
        x = y = 20.0
        scale_x = scale_y = 1.0
        offset_x = offset_y = 273.15
        width = 2.0
        if case == "difference":
            role, source_x, source_y = "difference", "delta_degC", "delta_degC"
            offset_x = offset_y = 0.0
    elif case == "affine_numeric":
        # A saved affine coordinate, independent of a physical-domain name.
        working = source_x = source_y = "1"
        x, y, scale_x, scale_y, offset_x, offset_y, width = 3.0, -1.0, 2.0, 4.0, 10.0, 20.0, 8.0
    variables = [{"id": "x0", "column": "x", "unit": working, "coordinate_role": role, "domain": [0, 1000]}]
    target = {"column": "y", "unit": working, "coordinate_role": role, "domain": [0, 1000]}
    schema = {"variables": variables, "target": target, "feature_ranges": [[0, 1000]]}
    schema["unit_coordinates"] = {
        "columns": {
            "x": {"source_unit": source_x, "problem_unit": working, "scale": scale_x, "offset": offset_x},
            "y": {"source_unit": source_y, "problem_unit": working, "scale": scale_y, "offset": offset_y},
        }
    }
    equation = {"schema_version": 2, "ast": {"var": "x0"}, "parameters": {}, "definitions": {}}
    if case == "scale_legacy":
        schema.pop("unit_coordinates")
        schema["public_units"] = {"x": source_x, "y": source_y}
        equation = {"schema_version": 1, "ast": {"var": "x0"}, "parameters": {}}
    record = {"kind": "explicit", "equation": equation, "native_check": {"passed": True}}
    directory = tmp_path / "interval"
    point = save_prediction(directory / "prediction.json", record, schema, {})
    quantity, quantity_unit = (width * 100, "cm") if case == "scale_map" else (width, working)
    write_json(
        directory / "interval.json",
        {
            "schema_version": 1,
            "kind": "split_conformal_observation_interval",
            "prediction_sha256": digest(point),
            "quantile": quantity,
            "unit": quantity_unit,
            "alpha": 0.1,
            "group_column": None,
            "calibration_ids": ["old"],
            "prepared_problem_ids": [],
            "coverage_scope": "boundary fixture",
            "limits": "not a statistical coverage experiment",
            "observation_model": "fixed boundary fixture",
        },
    )
    model = load_prediction_interval(directory)
    frame = pd.DataFrame({"sample_id": ["new"], "y": [y], "unused": [99], "x": [x]})
    values = model.predict(frame)
    expected_width = width / abs(scale_y)
    np.testing.assert_allclose([a[0] for a in values], [y, y - expected_width, y + expected_width], atol=1e-10)
    explicit = model.predict(frame, units={"x": source_x}, output_unit=source_y)
    np.testing.assert_allclose(values, explicit, atol=1e-10)
    csv = tmp_path / "observed.csv"
    frame.to_csv(csv, index=False)
    audit = read_json(audit_prediction_interval(directory, csv, output=tmp_path / "audit.json"))
    assert audit["rmse"] == pytest.approx(0, abs=1e-10) and audit["unit_coverage"] == 1
    bundle = save_model(tmp_path / "model.zip", equation, schema, {})
    result = run_task(
        {"task": "symbolic_infer", "model": {"local_model_path": str(bundle)}, "data": {"local_path": str(csv)}},
        run_dir=tmp_path / "infer",
    )
    assert read_json(result.artifacts["result_summary"])["prediction_unit"] == source_y
    np.testing.assert_allclose(pd.read_csv(result.tables["predictions"]).prediction, [y], atol=1e-10)
    residual = {**equation, "ast": {"op": "sub", "args": [{"var": "target"}, {"var": "x0"}]}}
    relation = ImplicitRelation(residual, schema, {}, equation, None)
    np.testing.assert_allclose(relation.residual(frame), [0], atol=1e-10)
    assert model_output(width, schema, difference=True) == pytest.approx(expected_width)
    # Loading a second consumer does not mutate metadata to conceal the missing legacy field.
    assert read_json(point)["schema"] == schema


def test_public_unit_compatibility_is_one_boundary():
    definition = {"column": "y", "unit": "m"}
    assert public_unit({}, definition) == "m"
    assert public_unit({}, {"column": "unknown"}) is None
    assert public_unit({"public_units": {"y": "mm"}}, definition) == "mm"
    schema = {"unit_coordinates": {"columns": {"y": {"source_unit": "mm"}}}, "public_units": {"y": "m"}}
    with pytest.raises(ValueError, match="conflicts"):
        public_unit(schema, definition)


def test_discovery_summary_preserves_partial_and_timeout_without_final(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/container_stage.py"
    spec = spec_from_file_location("release_container_stage", path)
    stage = module_from_spec(spec)
    spec.loader.exec_module(stage)
    cfg = {"search": {"engines": ["mean", "physo", "pysr"], "seeds": [0]}}
    for name, status in [("mean", "partial"), ("physo", "timeout")]:
        write_json(
            tmp_path / "discover" / name / "0/engine_summary.json", {"status": status, "seed": 0, "reason": status}
        )
    provenance = {"config": cfg}
    before = deepcopy(provenance)
    stage.summarize_discovery(tmp_path, {}, provenance)
    result = read_json(tmp_path / "discovery_results.json")
    assert {x["status"] for x in result["results"]} == {"partial", "timeout", "not_run"}
    assert not result["final_accessed"] and not result["selection_performed"]
    assert not (tmp_path / "finalize").exists() and before == provenance


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "duplicate", "matching"])
def test_release_wheel_requires_complete_runtime_payload(tmp_path, mutation):
    import zipfile

    from scripts.build_internal_release import validate_wheel_payload

    source = tmp_path / "ml_platform_symbolic"
    source.mkdir()
    payload = {"__main__.py": b"print('ready')\n", "report_template.html": b"<html></html>", "backend.jl": b"nothing\n"}
    for name, content in payload.items():
        (source / name).write_bytes(content)
    members = dict(payload)
    if mutation == "missing":
        members.pop("__main__.py")
    elif mutation == "extra":
        members["obsolete.py"] = b"pass\n"
    elif mutation == "changed":
        members["__main__.py"] = b"print('stale')\n"
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr("ml_platform_symbolic/" + name, content)
        if mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("ml_platform_symbolic/__main__.py", payload["__main__.py"])
    with zipfile.ZipFile(wheel) as archive:
        if mutation == "matching":
            validate_wheel_payload(archive, source)
        else:
            with pytest.raises(ValueError, match="Candidate wheel"):
                validate_wheel_payload(archive, source)


def test_source_export_excludes_local_assets(tmp_path):
    path = Path(__file__).parents[1] / "scripts/export_source.py"
    spec = spec_from_file_location("source_export_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = [
        "docs/user_manual.md", "scripts/run_linux.py", "tests/test_example.py", "containers/Dockerfile",
        "examples/input/demo.csv", "pkgs/symbolic/src/ml_platform_symbolic/model.py",
        "data/partial_equations/frozen.csv", "data/private.csv", "data/unlisted.zip",
        "results/run/secrets.json", "build/private.py", "runtime/source/private.py",
        "pretrained/model.pt", "scripts/__pycache__/cached.py", "scripts/temporary.egg-info/file.py",
    ]
    for name in fixtures:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    paths = module.source_paths(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in paths}
    assert {"docs/figures/workflow.svg", "docs/figures/architecture.svg", "docs/figures/constraints.svg"} <= names
    assert set(fixtures[:7]) <= names
    assert not set(fixtures[7:]) & names
