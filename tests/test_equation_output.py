"""Presentation never invents equations, parameter meaning or evaluation data."""

import csv
from copy import deepcopy

import numpy as np
import pytest
from ml_platform_symbolic import compare_equations, create_study, export_outputs, export_result, load_result
from ml_platform_symbolic.data import digest, read_json
from ml_platform_symbolic.equation_output import equation_outputs, equation_rows, format_equation
from ml_platform_symbolic.expressions import evaluate, parse_expression
from ml_platform_symbolic.report_data import build_report_view
from ml_platform_symbolic.reporting import csv_cell, export_output_data, normalize_report_data


def context():
    return {"variables": [{"id": "x0", "column": "a", "unit": "1"}],
            "target": {"column": "y", "unit": "1"}}


def test_coefficients_keep_identity_precision_bounds_units_and_namespaces():
    schema = context()
    parameters = {
        "a": {"value": 1.2345678901234567, "trainable": True, "unit": None, "bounds": [0, 2]},
        "same": {"value": 1.2345678901234567, "trainable": False, "unit": None, "tied_to": "a"},
        "physical": {"value": 3.141592653589793, "trainable": False, "unit": "1", "origin": "known constant"},
    }
    equation = parse_expression("a*x0 + same + physical + x0**2", schema["variables"], parameters=parameters)
    before = deepcopy(equation)
    output = equation_outputs(equation, schema)
    assert output["equation"] == equation == before
    assert "coef['a']" in output["public_text"] and r"\theta_{\text{a}}" in output["public_latex"]
    assert {p["name"]: p["role"] for p in output["parameters"]} == {"a": "free", "same": "tied", "physical": "fixed"}
    assert output["parameters"][0]["unit"] is None
    assert output["parameters"][0]["value"] == parameters["a"]["value"]
    assert output["parameters"][0]["bounds"] == [0, 2]
    assert output["parameters"][1]["tied_to"] == "a"
    assert "2" not in {p["name"] for p in output["parameters"]}


def test_operator_semantics_and_domain_holes_are_not_simplified():
    schema = context()
    equations = [parse_expression(text, schema["variables"], trainable=False)
                 for text in ("x0/x0", "log(x0)**0", "x0**(1/3)")]
    root = {"schema_version": 2, "arithmetic": "numpy_ieee", "parameters": {},
            "ast": {"op": "rootn", "args": [{"var": "x0"}, {"const": 3}]}}
    assert evaluate(root, np.array([[-8.0]]), schema["variables"])[0] == -2
    assert np.isnan(evaluate(equations[-1], np.array([[-8.0]]), schema["variables"])[0])
    for equation in [*equations, root]:
        output = equation_outputs(equation, schema)
        assert output["equation"] == equation
        assert "intermediate" in " ".join(output["semantic_notes"])
    assert r"\frac" in equation_outputs(equations[0], schema)["latex"]
    assert "log" in equation_outputs(equations[1], schema)["latex"]
    assert r"\operatorname{pow}" in equation_outputs(equations[2], schema)["latex"]
    output = equation_outputs(root, schema)
    assert "rootn" in output["latex"] and output["arithmetic"] == "numpy_ieee"
    root["ast"]["op"] = "unknown_protected_div"
    with pytest.raises(ValueError, match="Unknown operator"):
        equation_outputs(root, schema)
    negative_base = {"schema_version": 1, "parameters": {},
                     "ast": {"op": "pow_int", "args": [{"const": -8}, {"const": 2}]}}
    assert "(-8)" in format_equation(negative_base, [])


def test_frozen_public_maps_apply_once_and_keep_residual_scale():
    schema = context()
    schema["variables"][0].update(unit="s", column="elapsed")
    schema["target"].update(unit="K", column="temperature")
    schema["unit_coordinates"] = {"columns": {
        "elapsed": {"scale": 86400.0, "offset": 0.0, "source_unit": "day", "problem_unit": "s"},
        "temperature": {"scale": 1.0, "offset": 273.15, "source_unit": "degC", "problem_unit": "K"},
    }}
    equation = parse_expression("2*x0+300", schema["variables"], trainable=False)
    output = equation_outputs(equation, schema)
    assert output["public_text"].count("273.15") == 1
    assert output["public_latex"].count("273.15") == 1
    assert "86400" in output["public_text"] and output["equation"] == equation
    variables = [*schema["variables"], {"id": "target", "column": "temperature"}]
    residual = parse_expression("target - 2*x0", variables, trainable=False)
    displayed = equation_outputs(residual, schema, residual=True)
    assert displayed["public_text"].endswith(" = 0")
    assert displayed["public_text"].count("273.15") == 1
    assert displayed["equation"] == residual


def test_compact_definitions_polynomials_and_tex_labels_are_preserved():
    schema = context()
    schema["variables"][0]["column"] = r"input_1%\write18{bad}"
    equation = {"schema_version": 2, "parameters": {},
                "definitions": {"part": {"op": "log", "args": [{"var": "x0"}]}},
                "ast": {"poly": [{"coefficient": {"const": 2.0}, "powers": [[0, 2]]}],
                        "args": [{"ref": "part"}]}}
    output = equation_outputs(equation, schema)
    assert output["equation"] == equation and "definitions" in output["equation"]
    assert output["latex"].count("log") == 2
    assert r"\write18" not in output["latex"] and r"\%" in output["latex"]
    with pytest.raises(ValueError, match="format"):
        format_equation(equation, schema["variables"], format="html")


def test_ensemble_and_implicit_components_remain_distinct_in_historical_views():
    view = {"candidates": [
        {"id": "ensemble", "kind": "ensemble", "execution_view": {"kind": "weighted_mean"},
         "uncertainty_scope": "Native posterior, not mechanism probabilities", "members": [
            {"index": 0, "weight": 1, "formula": "x", "parameters": []},
            {"index": 1, "weight": 0, "formula": "x+1", "parameters": []}]},
        {"id": "relation", "kind": "implicit", "residual": {"formula": "100*(y-x)=0", "parameters": []},
         "formula": None, "parameters": [{"name": "residual_scale", "value": 100}]},
    ]}
    rows = equation_rows(view)
    assert len(rows) == 4 and rows[1]["weight"] == 0
    assert rows[2]["component"] == "implicit_residual" and rows[2]["text"] == "100*(y-x)=0"
    assert rows[3]["component"] == "implicit_predictor" and rows[3]["parameters"] == []
    assert all(row["latex"] is None and row["equation"] is None for row in rows)
    assert rows[1]["execution_view"] == {"kind": "weighted_mean"}
    assert "not mechanism" in rows[0]["uncertainty_scope"]


@pytest.mark.parametrize("text", ["=FORMULA()", "+SUM(1)", "-name", "@something", " =formula", "\ttab"])
def test_csv_text_is_inert_without_changing_numeric_values(text):
    assert csv_cell(text) == "'" + text
    assert csv_cell(-1.2345678901234567) == -1.2345678901234567
    assert csv_cell(None) is None


def test_outputs_save_reload_without_model_evaluation_or_final(synthetic_config, tmp_path, monkeypatch):
    study = create_study({"problem": synthetic_config, "sources": [
        {"manual": {"id": "quadratic", "expression": "2*x*x+1"}},
        {"manual": {"id": "linear", "expression": "2*x+1"}},
    ]}, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    before = {str(p): digest(p) for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Output must not execute or evaluate a model")

    monkeypatch.setattr("ml_platform_symbolic.expressions.evaluate", forbidden)
    view = build_report_view(comparison)
    package = export_result(comparison, run_dir=tmp_path / "handoff")
    result = load_result(package, load_predictors=False)
    monkeypatch.setattr("ml_platform_symbolic.report_data.read_study_evidence", forbidden)
    monkeypatch.setattr("ml_platform_symbolic.model.load_model", forbidden)
    assert result["view"]["candidates"][0]["equation_output"]["equation"] == view["candidates"][0]["equation_output"]["equation"]
    output = export_outputs(package, output_dir=tmp_path / "files")
    assert {p.name for p in output.iterdir()} == {"README.txt", "equations.txt", "figures", "tables", "metadata"}
    assert list((output / "figures").glob("*.png"))
    assert {p.suffix for p in (output / "figures").rglob("*") if p.is_file()} == {".png", ".txt", ".svg"}
    assert (output / "metadata/plot_data.json").is_file()
    index = read_json(output / "metadata/output_index.json")
    assert all(digest(output / name) == sha for name, sha in index["files"].items())
    equations = read_json(output / "metadata/equations.json")
    assert len(equations) == 2 and all(row["equation"] and row["latex"] for row in equations)
    with (output / "tables/candidates.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert len(rows) == 2
        assert "coverage" not in reader.fieldnames
        assert "metric_reason" in reader.fieldnames
    assert all(digest(path) == sha for path, sha in before.items())
    assert not list(tmp_path.rglob("final_access.json"))
    with pytest.raises(ValueError, match="already exists"):
        export_outputs(package, output_dir=output)
    with pytest.raises(ValueError, match="detached result"):
        export_outputs(package, output_dir=tmp_path / "mix", representative_ids=["quadratic@manual"])
    snapshot = deepcopy(result["view"])
    export_output_data(snapshot, output_dir=tmp_path / "no-figures", figures=False)
    assert snapshot == result["view"]
    snapshot["candidates"][0]["source"].update(status="partial", wall_seconds=12.5)
    timed = export_output_data(snapshot, output_dir=tmp_path / "timed", figures=False)
    with (timed / "tables/candidates.csv").open(encoding="utf-8", newline="") as stream:
        timing_rows = list(csv.DictReader(stream))
    assert timing_rows[0]["run_status"] == "partial"
    assert timing_rows[0]["run_wall_seconds"] == "12.5"
    assert timing_rows[1]["run_wall_seconds"] == ""
    assert timing_rows[0]["partition"] == "Selection"
    assert read_json(tmp_path / "no-figures" / "metadata/report.json") == normalize_report_data(result["view"])


def test_saved_output_can_be_rerendered_without_sources_and_detects_tampering(tmp_path):
    view = {"question": "A saved equation", "candidates": [], "unit": "mm"}
    original = export_output_data(view, output_dir=tmp_path / "old", figures=False)
    assert {p.name for p in original.iterdir()} == {"README.txt", "equations.txt", "tables", "metadata"}
    replay = export_outputs(original, output_dir=tmp_path / "new", figures=False, include_html=True)
    assert (replay / "report.html").is_file()
    assert read_json(replay / "metadata/report.json") == view
    with pytest.raises(ValueError, match="frozen evidence"):
        export_outputs(original, output_dir=tmp_path / "mixed", figures=False, decision_dir="new-decision")
    (original / "metadata/report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        export_outputs(original, output_dir=tmp_path / "corrupt", figures=False)
    assert not (tmp_path / "corrupt").exists()


@pytest.mark.parametrize("subdirectory", ["", "details"])
def test_legacy_output_can_be_reexported_without_moving_original(tmp_path, subdirectory):
    from ml_platform_symbolic.data import write_json

    old = tmp_path / "flat"
    storage = old / subdirectory
    storage.mkdir(parents=True)
    view = {"question": "Historical saved evidence", "candidates": [], "unit": "1"}
    write_json(storage / "report.json", view)
    write_json(storage / "output_index.json", {"files": {(storage / "report.json").relative_to(old).as_posix(): digest(storage / "report.json")}})
    new = export_outputs(old, output_dir=tmp_path / "organized", figures=False)
    assert read_json(new / "metadata/report.json") == view
    assert (storage / "report.json").is_file()


def test_failed_render_does_not_leave_a_partial_result_directory(tmp_path, monkeypatch):
    view = {"question": "Saved", "candidates": [], "unit": "1"}

    def failed(*args, **kwargs):
        raise RuntimeError("Renderer unavailable")

    monkeypatch.setattr("ml_platform_symbolic.reporting.render_report_data", failed)
    with pytest.raises(RuntimeError, match="Renderer unavailable"):
        export_output_data(view, output_dir=tmp_path / "failed", figures=False, include_html=True)
    assert not list(tmp_path.iterdir())


def test_result_can_share_a_directory_with_calculation_records(tmp_path, monkeypatch):
    from ml_platform_symbolic.reporting import load_report_view
    view = {"question": "Saved", "candidates": [], "unit": "1"}
    root = tmp_path / "experiment"
    records = root / "calculation"
    records.mkdir(parents=True)
    evidence = records / "state.json"
    evidence.write_text('{"untouched": true}', encoding="utf-8")
    before = digest(evidence)
    def failed(*args, **kwargs):
        raise RuntimeError("Renderer unavailable")
    with monkeypatch.context() as patch:
        patch.setattr("ml_platform_symbolic.reporting.render_report_data", failed)
        with pytest.raises(RuntimeError, match="Renderer unavailable"):
            export_output_data(view, output_dir=root, figures=False, include_html=True)
    assert list(root.iterdir()) == [records]
    export_output_data(view, output_dir=root, figures=False)
    assert digest(evidence) == before
    assert load_report_view(root) == view
    with pytest.raises(ValueError, match="already exists"):
        export_output_data(view, output_dir=root, figures=False)


def test_graph_csv_is_in_tables_and_svg_references_its_exact_bytes(tmp_path):
    import hashlib
    import xml.etree.ElementTree as ET
    view = {"question": "Saved", "candidates": [], "unit": "1"}
    output = export_output_data(view, output_dir=tmp_path / "result")
    for svg in (output / "figures").glob("*.svg"):
        raw = svg.read_text(encoding="utf-8")
        ET.fromstring(raw)
        table = output / "tables" / (svg.stem + ".csv")
        assert table.is_file()
        assert "../tables/" + table.name in raw
        assert hashlib.sha256(table.read_bytes()).hexdigest() in raw


def test_representative_export_filters_tables_without_evaluating_or_changing_source(synthetic_config, tmp_path, monkeypatch):
    study = create_study({"problem": synthetic_config, "sources": [
        {"manual": {"id": "quadratic", "expression": "2*x*x+1"}},
        {"manual": {"id": "linear", "expression": "2*x+1"}},
    ]}, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    view = build_report_view(comparison, representative_ids=["quadratic@manual"])
    # A saved failed method must remain visible even without a representative.
    failed = deepcopy(view["candidates"][1])
    failed.update(id="failed/17@native_result", kind="native_result", method="failed", eligible=False,
                  reason="timeout", source={"status": "timeout", "wall_seconds": 20})
    view["candidates"].append(failed)
    original = deepcopy(view)

    def forbidden(*args, **kwargs):
        raise AssertionError("Compact export must not evaluate predictions")

    monkeypatch.setattr("ml_platform_symbolic.expressions.evaluate", forbidden)
    output = export_output_data(view, output_dir=tmp_path / "compact", representatives_only=True)
    stored = read_json(output / "metadata/report.json")
    assert {row["id"] for row in stored["candidates"]} == {"quadratic@manual", "failed/17@native_result"}
    assert stored["export_scope"]["original_count"] == 3
    assert stored["candidates"][0] == original["candidates"][0]
    assert not stored["pairs"]
    assert view == original
    default = export_output_data(view, output_dir=tmp_path / "full", figures=False)
    assert len(read_json(default / "metadata/report.json")["candidates"]) == 3


def test_saved_slot_structure_is_rendered_without_coordinate_invention(synthetic_config, tmp_path, monkeypatch):
    from ml_platform_symbolic.equation_output import template_rows, template_lines
    import xml.etree.ElementTree as ET

    study = create_study({"problem": synthetic_config, "sources": [{"manual": {"id": "f", "expression": "2*x+1"}}]},
                         run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    view = build_report_view(comparison)
    template = dict(status="computed", coordinates="canonical working; saved coordinates only",
                    declared_formula="known + h(x) + h(x)",
                    knowledge={"kind": "hypothesis", "statement": "User supplied relation"},
                    components=[dict(name="known", formula="k", unit="1", description="Declared fixed contribution")],
                    slots=[dict(name="h", arguments=[dict(name="u",unit="m")], unit="1", formula="c*u",
                                parameters=[dict(name="c",value=2.,unit="1/m",role="free")])],
                    dependent_parameters=[dict(name="b",value=4.,formula="2*c",unit="1/m")],
                    sharing="Repeated h calls share the function and coefficients.",
                    interpretation_limit="Declared structure is not a discovered physical law.")
    view["candidates"][0]["template"] = template
    before = deepcopy(view)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved named metadata is presentation only")
    monkeypatch.setattr("ml_platform_symbolic.expressions.evaluate", forbidden)
    output = export_output_data(view, output_dir=tmp_path / "output")
    assert view == before
    assert len([row for row in template_rows(template) if row["section"] == "fitted_slot"]) == 1
    text = (output / "equations.txt").read_text()
    assert "h(x) + h(x)" in text and "role=free" in text and "canonical working" in text
    assert "Repeated h calls share" in text and "dependent parameter: b = 2*c" in text
    visible = " ".join(" ".join(ET.parse(path).getroot().itertext())
                       for path in (output / "figures").glob("*/equation*.svg"))
    assert "fitted slot: h(u [m]) = c*u" in visible
    assert template_lines(None) == [] and template_rows(None) == []


def test_unavailable_template_keeps_reason_without_inventing_parts():
    from ml_platform_symbolic.equation_output import template_rows, template_lines
    metadata = {"status": "unavailable", "reason": "Missing saved slot binding"}
    rows = template_rows(metadata)
    assert rows == [{"coordinates": None, "status": "unavailable", "section": "status", "reason": metadata["reason"]}]
    for concise in (False, True):
        lines = template_lines(metadata, concise=concise)
        assert lines == ["Template readout: unavailable", "Reason: Missing saved slot binding"]
        assert not any("Declared structure" in line for line in lines)
