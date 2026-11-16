"""Stored figure evidence: gaps, source identity and machine-verifiable tables."""

import csv
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy

import pytest
from ml_platform_symbolic.figure_output import build_plot_data, render_figures


def saved_view():
    path = dict(id="same-name", display_id="source-a:0", steps=[0.0, 2.0, 1.0], path_unit="s", unit="mm",
                at={"x": 0, "fixed": 5}, direction={"x": 1, "fixed": 0},
                input_units={"x": "mm", "fixed": "1"}, source_columns=["x", "fixed"],
                source_points=[[0, 5], [2, 5], [1, 5]],
                prediction=[10.0, None, 12.0], valid=[True, False, True],
                scope="Saved mathematical path, not an intervention",
                reference_geometry=dict(status="computed", data_role="train", nearest_distance=[0.0, 2.0, 0.0],
                                        outside_marginal_range=[False, True, True],
                                        constant_feature_changed=[False, False, True]))
    cards = []
    for name, method in [("a", "native-a"), ("b", "native-b")]:
        cards.append(dict(id=name, method=method, eligible=name == "a", reason="failed requirement" if name == "b" else None,
                          complexity={"nodes": 5}, selection={"rmse": 1.5, "coverage": 0.5},
                          prediction_scatter={"observed": [10, 12], "predicted": [10, None], "valid": [True, False]},
                          residual_scatter={"y": [0, None], "valid": [True, False]},
                          interpretations=[dict(analysis_sha256="source-a", paths=[deepcopy(path)])]))
    pair = dict(axis="path_response_difference", analysis_sha256="source-a", path_display_id="source-a:0",
                left="a", right="b", steps=[0.0, 2.0, 1.0], difference=[0.0, None, -1.0],
                valid=[True, False, True], path_unit="s", unit="mm")
    return dict(candidates=cards, display_ids=["a", "b"], unit="mm", residual_axis={"x": [0, 1], "x_label": "x [mm]"},
                analysis_comparisons=[pair], study_sha256="study", comparison_sha256="comparison")


def test_plot_rows_preserve_values_invalid_rows_order_and_csv_binding(tmp_path):
    view = saved_view()
    original = deepcopy(view)
    data = build_plot_data(view)
    response = next(item for item in data["figures"] if item["kind"] == "response")
    assert [r["x"] for r in response["rows"][:3]] == [0, 2, 1]
    assert [r["plotted"] for r in response["rows"][:3]] == [True, False, True]
    assert [r["source_point"] for r in response["rows"][:3]] == [[0, 5], [2, 5], [1, 5]]
    paths = render_figures(data, tmp_path / "figures")
    assert view == original
    assert len(paths) == len(data["figures"]) * 4 + 1
    for figure in data["figures"]:
        png = (tmp_path / "figures" / (figure["id"] + ".png")).read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        assert width == 1500 and height > 700
        assert (tmp_path / "figures" / (figure["id"] + ".txt")).is_file()
        root = ET.parse(tmp_path / "figures" / (figure["id"] + ".svg")).getroot()
        csv_path = tmp_path / "figures" / root.attrib["data-source-csv"]
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == root.attrib["data-source-sha256"]
        with csv_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == len(figure["rows"])
        points = [element for element in root.iter() if "data-row" in element.attrib]
        assert len(points) == sum(row["plotted"] for row in figure["rows"])
        for point in points:
            row = rows[int(point.attrib["data-row"])]
            assert float(row["x"]) == float(point.attrib["data-x"])
            assert float(row["y"]) == float(point.attrib["data-y"])
            assert row["series"] == point.attrib["data-series"]
        if figure["kind"] in {"response", "response_difference"}:
            assert not [element for element in root.iter() if element.attrib.get("data-segment") == "true"]
    assert json.loads((tmp_path / "figures/plot_data.json").read_text(encoding="utf-8")) == data
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError, match="already exists"):
        render_figures(data, tmp_path / "figures")
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_different_sources_and_ambiguous_pairs_are_never_overlaid():
    view = saved_view()
    for card in view["candidates"]:
        second = deepcopy(card["interpretations"][0])
        second["analysis_sha256"] = "source-b"
        second["paths"][0].update(display_id="source-b:0", at={"x": 99, "fixed": 5}, prediction=[91, 92, 93])
        card["interpretations"].append(second)
    view["analysis_comparisons"].append({**view["analysis_comparisons"][0], "path_display_id": None})
    data = build_plot_data(view)
    responses = [f for f in data["figures"] if f["kind"] == "response"]
    assert len(responses) == 2
    assert {f["analysis_sha256"] for f in responses} == {"source-a", "source-b"}
    assert len([f for f in data["figures"] if f["kind"] == "response_difference"]) == 1
    assert any("not overlaid" in item["reason"] for item in data["unavailable"])


def test_missing_predictions_residuals_scores_and_support_are_not_invented(tmp_path):
    view = saved_view()
    del view["candidates"][0]["prediction_scatter"]
    del view["candidates"][0]["residual_scatter"]
    view["candidates"][0]["selection"] = {"rmse": None, "reason": "invalid domain", "coverage": 0}
    for card in view["candidates"]:
        del card["interpretations"][0]["paths"][0]["reference_geometry"]
    data = build_plot_data(view)
    metrics = next(f for f in data["figures"] if f["kind"] == "selection_tradeoff")["rows"]
    assert metrics[0]["y"] is None and metrics[0]["plotted"] is False
    assert metrics[0]["reason"] == "invalid domain"
    assert metrics[1]["eligible"] is False
    assert {item["kind"] for item in data["unavailable"]} >= {"observed_prediction", "residual", "input_geometry"}
    assert not [f for f in data["figures"] if f["kind"] == "input_geometry"]
    render_figures(data, tmp_path)


def test_train_geometry_preserves_constant_change_even_at_zero_distance(tmp_path):
    data = build_plot_data(saved_view())
    geometry = next(f for f in data["figures"] if f["kind"] == "input_geometry")
    assert geometry["rows"][2]["y"] == 0
    assert geometry["rows"][2]["constant_feature_changed"] is True
    assert geometry["rows"][2]["outside_marginal_range"] is True
    render_figures(data, tmp_path)
    root = ET.parse(tmp_path / (geometry["id"] + ".svg")).getroot()
    marker = next(e for e in root.iter() if e.attrib.get("data-row") == "2")
    assert marker.tag.endswith("path") and marker.attrib["stroke"] == "#222222"


@pytest.mark.parametrize("values", [[-1e308, 1e308], [0, 0], [1e-300, 1e-300],
                                  [sys.float_info.max, sys.float_info.max],
                                  [-sys.float_info.max, sys.float_info.max]])
def test_extreme_or_constant_finite_values_render_without_invalid_svg(tmp_path, values):
    view = saved_view()
    for card in view["candidates"]:
        card["prediction_scatter"] = {"observed": values, "predicted": values, "valid": [True, True]}
    data = build_plot_data(view)
    render_figures(data, tmp_path)
    scatter = next(f for f in data["figures"] if f["kind"] == "observed_prediction")
    content = (tmp_path / (scatter["id"] + ".svg")).read_text(encoding="utf-8")
    assert "nan" not in content.lower()
    assert "inf" not in content.lower()
    ET.fromstring(content)


def test_mask_or_coordinate_length_mismatch_is_not_silently_truncated():
    view = saved_view()
    view["candidates"][0]["prediction_scatter"]["valid"] = [True]
    with pytest.raises(ValueError, match="equal lengths"):
        build_plot_data(view)


def test_saved_order_keeps_nonmonotone_steps_without_sorting(tmp_path):
    view = saved_view()
    for card in view["candidates"]:
        card["interpretations"][0]["paths"][0].update(prediction=[0, 2, 1], valid=[True, True, True])
    data = build_plot_data(view)
    render_figures(data, tmp_path)
    response = next(f for f in data["figures"] if f["kind"] == "response")
    root = ET.parse(tmp_path / (response["id"] + ".svg")).getroot()
    assert [float(e.attrib["data-x"]) for e in root.iter() if e.attrib.get("data-series") == "a"] == [0, 2, 1]
    assert len([e for e in root.iter() if e.attrib.get("data-segment") == "true"]) == 4


def test_csv_formula_like_strings_are_escaped_and_raw_json_is_preserved(tmp_path):
    view = saved_view()
    payload = '=HYPERLINK("https://example.invalid", "open")'
    view["candidates"][0]["method"] = payload
    view["candidates"][0]["reason"] = "  @SUM(A1:A2)"
    data = build_plot_data(view)
    render_figures(data, tmp_path)
    csv_path = tmp_path / (data["figures"][0]["id"] + ".csv")
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["series"] == "'" + payload
    assert rows[0]["reason"] == "'  @SUM(A1:A2)"
    assert data["figures"][0]["rows"][0]["series"] == payload
    assert "apostrophe" in data["csv_string_policy"]


def test_all_invalid_or_nonfinite_evidence_is_retained_without_fake_points(tmp_path):
    view = saved_view()
    for card in view["candidates"]:
        card["prediction_scatter"] = {"observed": [10, 12], "predicted": [float("inf"), None], "valid": [True, False]}
    data = build_plot_data(view)
    figure = next(f for f in data["figures"] if f["kind"] == "observed_prediction")
    assert len(figure["rows"]) == 4
    assert all(not row["plotted"] for row in figure["rows"])
    assert figure["rows"][0]["plot_reason"] == "Missing or nonfinite saved coordinate"
    render_figures(data, tmp_path)
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    assert not [element for element in root.iter() if "data-row" in element.attrib]
    assert "No finite valid stored points" in " ".join(root.itertext())


def test_omitted_display_candidates_are_not_mislabeled_as_missing_evidence():
    view = saved_view()
    view["display_ids"] = ["a"]
    data = build_plot_data(view)
    pair = next(item for item in data["unavailable"] if item.get("right") == "b")
    assert "Outside displayed candidate set" in pair["reason"]


def test_figures_show_methods_formulas_units_and_precise_parameters(tmp_path):
    view = saved_view()
    card = view["candidates"][0]
    card.update(public_formula="y [mm] = coef['a'] * x", formula="coef['a'] * x_work",
                parameters=[dict(name="a", value=1.2345678901234567, unit=None)])
    data = build_plot_data(view)
    render_figures(data, tmp_path)
    figure = next(f for f in data["figures"] if f["kind"] == "observed_prediction")
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    visible = " ".join(e.text or "" for e in root.iter() if e.tag.endswith("text"))
    assert card["method"] in visible
    assert card["public_formula"] not in visible
    assert "1.23457" not in visible
    assert "[unit unknown]" not in visible  # Coefficients belong to separate sheets.
    assert "Observed [mm]" in visible and "Predicted [mm]" in visible
    assert "Saved analysis:" not in visible and "spreadsheet" not in visible
    complete = (tmp_path / (figure["id"] + ".txt")).read_text(encoding="utf-8")
    assert "1.2345678901234567" in complete
    assert "unit unknown" in complete
    assert card["id"] in complete


def test_long_equation_is_marked_as_excerpt_and_complete_in_text(tmp_path):
    view = saved_view()
    equation = "y [mm] = " + " + ".join(f"coef['c{i}'] * x" for i in range(150))
    view["candidates"][0]["public_formula"] = equation
    data = build_plot_data(view)
    render_figures(data, tmp_path)
    figure = next(f for f in data["figures"] if f["kind"] == "observed_prediction")
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    visible = " ".join(e.text or "" for e in root.iter() if e.tag.endswith("text"))
    assert "coef[" not in visible
    assert equation in (tmp_path / (figure["id"] + ".txt")).read_text(encoding="utf-8")
    assert figure["models"][0]["formula"] == equation


def test_implicit_predictor_never_borrows_residual_coefficients(tmp_path):
    view = saved_view()
    view["candidates"][0].update(kind="implicit", formula="native solver",
                                  parameters=[dict(name="residual_only", value=19, unit="m")])
    data = build_plot_data(view)
    model = data["figures"][0]["models"][0]
    assert model["formula"] == "native solver"
    assert not model["parameters"]


def test_ensemble_keeps_native_view_and_member_formulas_without_invented_mean(tmp_path):
    view = saved_view()
    view["candidates"][0].update(kind="ensemble", execution_view={"name": "native_median"},
                                 members=[dict(index=0, weight=0.4, public_formula="y = x"),
                                          dict(index=1, weight=0.6, public_formula="y = 2*x")])
    data = build_plot_data(view)
    render_figures(data, tmp_path)
    text = (tmp_path / (data["figures"][0]["id"] + ".txt")).read_text(encoding="utf-8")
    assert "native_median" in text and "y = x" in text and "y = 2*x" in text
    assert "weight=0.4" in text and "weight=0.6" in text


def test_compact_formula_preserves_operation_order_and_coordinate_units():
    from ml_platform_symbolic.equation_output import equation_outputs, format_equation
    from ml_platform_symbolic.expressions import parse_expression

    variables = [{"id": "x0", "column": "x", "unit": "m"},
                 {"id": "x1", "column": "z", "unit": "m"}]
    expressions = ["x0-(x1-x0)", "x0/(x1/x0)", "x0+(x1+x0)", "(x0+x1)*x0", "(-x0)**2"]
    for source in expressions:
        equation = parse_expression(source, variables, trainable=False)
        shown = format_equation(equation, variables, format="display")
        for x, z in [(1.25, 2.5), (4, -3)]:
            # Only these fixed test expressions, not user equations, are evaluated.
            expected = eval(source, {"x0": x, "x1": z})
            actual = eval(shown.replace("^", "**"), {"x": x, "z": z})
            assert actual == expected
    context = {"variables": variables[:1], "target": {"column": "y", "unit": "m"},
               "unit_coordinates": {"columns": {"x": {"scale": 0.01, "offset": 0.0, "source_unit": "cm", "problem_unit": "m"},
                                                   "y": {"scale": 0.001, "offset": 0.0, "source_unit": "mm", "problem_unit": "m"}}}}
    equation = parse_expression("2*x0+1", variables[:1], trainable=False)
    output = equation_outputs(equation, context)
    assert "0.01" in output["public_display_text"] and "0.001" in output["public_display_text"]
    assert output["equation"] == equation
    assert len(output["public_display_text"]) < len(output["public_text"])


def test_saved_fitted_candidate_identifies_its_estimator(synthetic_config, tmp_path):
    from ml_platform_symbolic import compare_equations, create_study, fit_hypotheses
    from ml_platform_symbolic.report_data import build_report_view

    original = create_study({"problem": synthetic_config, "sources": [{"manual": {
        "id": "curve", "expression": "a*x*x+b", "parameters": {
            "a": {"value": 1, "trainable": True, "unit": "1"},
            "b": {"value": 0, "trainable": True, "unit": "1"}}}}]}, run_dir=tmp_path / "initial")
    fitted = fit_hypotheses(original, ids=["curve@manual"], run_dir=tmp_path / "fit")
    artifacts = json.loads((fitted / "fits.json").read_text())["artifacts"]
    study = create_study({"problem": synthetic_config, "sources": [{"derived": path} for path in artifacts]},
                         run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    view = build_report_view(comparison)
    assert "least-squares" in view["candidates"][0]["method"].lower()
    assert view["candidates"][0]["source"]["fit"]["fit_role"] == "train"
    assert "least-squares" in build_plot_data(view)["figures"][0]["models"][0]["label"].lower()


def test_single_input_path_axis_uses_source_value_without_changing_saved_steps():
    view = saved_view()
    for card in view["candidates"]:
        card["interpretations"][0]["paths"][0]["path_unit"] = "mm"
    data = build_plot_data(view)
    response = next(figure for figure in data["figures"] if figure["kind"] == "response")
    assert response["x_label"] == "x [mm]"
    assert [row["x"] for row in response["rows"][:3]] == [0, 2, 1]


def test_eight_method_comparison_uses_distinct_series_colors(tmp_path):
    view = saved_view()
    template = deepcopy(view["candidates"][0])
    view["candidates"] = [{**deepcopy(template), "id": f"candidate-{i}", "method": f"method-{i}"} for i in range(8)]
    view["display_ids"] = [card["id"] for card in view["candidates"]]
    view["analysis_comparisons"] = []
    data = build_plot_data(view)
    figure = next(item for item in data["figures"] if item["kind"] == "observed_prediction")
    render_figures({**data, "figures": [figure]}, tmp_path)
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    colors = {point.attrib["fill"] for point in root.iter() if "data-row" in point.attrib}
    assert len(colors) == 8


def test_accuracy_shows_saved_display_equations_without_dropping_other_candidates(tmp_path):
    view = saved_view()
    template = deepcopy(view["candidates"][0])
    view["candidates"] = [{**deepcopy(template), "id": f"candidate-{i}", "method": f"method-{i // 6}",
                           "public_formula": f"y = formula_{i}(x)", "complexity": {"nodes": i + 1},
                           "selection": {"rmse": 45 - i}} for i in range(45)]
    representatives = [0, 6, 12, 18, 24, 30, 36, 44]
    view["display_ids"] = [f"candidate-{index}" for index in representatives]
    view["analysis_comparisons"] = []
    original = deepcopy(view)
    data = build_plot_data(view)
    figure = next(item for item in data["figures"] if item["kind"] == "selection_tradeoff")
    render_figures({**data, "figures": [figure]}, tmp_path)
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    visible = " ".join(e.text or "" for e in root.iter() if e.tag.endswith("text"))
    assert len([e for e in root.iter() if "data-row" in e.attrib]) == 45
    numbered = {int(number) for e in root.iter() for number in e.attrib.get("data-candidate-numbers", "").split(",")
                if number}
    assert not numbered  # IDs stay in the exact companion; overlapping point labels were removed.
    for index in representatives:
        assert f"formula_{index}(x)" not in visible
    assert "formula_1(x)" not in visible
    assert "equations below" not in visible
    complete = (tmp_path / (figure["id"] + ".txt")).read_text(encoding="utf-8")
    assert all(f"formula_{index}(x)" in complete for index in range(45))
    with (tmp_path / (figure["id"] + ".csv")).open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 45
    assert view == original


def test_pair_chart_never_silently_hides_pairs_and_keeps_saved_direction(tmp_path):
    from itertools import combinations

    view = saved_view()
    template = deepcopy(view["candidates"][0])
    view["candidates"] = [{**deepcopy(template), "id": f"candidate-{i}", "method": f"method-{i}"} for i in range(9)]
    # Reference is deliberately not first in the candidate archive, and it
    # appears on both sides of stored pairs. Display must not reverse signs.
    view["display_ids"] = ["candidate-4", *[f"candidate-{i}" for i in range(9) if i != 4]]
    view["analysis_comparisons"] = []
    for i, j in combinations(range(9), 2):
        view["analysis_comparisons"].append(dict(
            axis="path_response_difference", analysis_sha256="source-a", path_display_id="source-a:0",
            left=f"candidate-{i}", right=f"candidate-{j}", steps=[0.0, 2.0, 1.0],
            difference=[float(i - j), None, float(j - i)], valid=[True, False, True], path_unit="s", unit="mm"))
    original = deepcopy(view)
    data = build_plot_data(view)
    figure = next(item for item in data["figures"] if item["kind"] == "response_difference")
    render_figures({**data, "figures": [figure]}, tmp_path)
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    points = [item for item in root.iter() if "data-row" in item.attrib]
    expected_series = list(dict.fromkeys(row["series"] for row in figure["rows"]))
    assert len(expected_series) == 36 and len(points) == 72
    assert {point.attrib["data-series"] for point in points} == set(expected_series)
    for point in points:
        original_row = figure["rows"][int(point.attrib["data-row"])]
        assert float(point.attrib["data-y"]) == original_row["y"]
        assert float(point.attrib["data-x"]) == original_row["x"]
    assert not [item for item in root.iter() if item.attrib.get("data-segment") == "true"]
    visible = " ".join(item.text or "" for item in root.iter() if item.tag.endswith("text"))
    assert "5. method-4" in visible and "First saved display candidate" not in visible
    assert "7 / 36 saved pairs" not in visible
    assert "1. method-0 - 5. method-4" in visible and "5. method-4 - 6. method-5" in visible
    assert "Left minus right" in visible and "disagreement does not identify the better model" in visible
    assert "Prediction (left - right) [mm]" in visible
    with (tmp_path / (figure["id"] + ".csv")).open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 36 * 3
    saved = json.loads((tmp_path / "plot_data.json").read_text(encoding="utf-8"))
    assert len(saved["figures"][0]["rows"]) == 36 * 3
    full_text = (tmp_path / (figure["id"] + ".txt")).read_text(encoding="utf-8")
    assert "candidate-7 - candidate-8" in full_text and "valid=False" in full_text
    assert view == original


def test_method_outputs_are_separate_and_keep_exact_rows_and_equations(tmp_path):
    view = saved_view()
    view["candidates"][0].update(public_formula="y = coef['a'] * x", parameters=[dict(name="a", value=1.2345678901234567, unit=None)])
    data = build_plot_data(view)
    render_figures(data, tmp_path / "figures", table_dir=tmp_path / "tables", metadata_dir=tmp_path / "metadata", per_method=True)
    assert not (tmp_path / "figures/02_predictions.png").exists()
    for index, method in [(1, "native-a"), (2, "native-b")]:
        folder = tmp_path / "figures" / f"{index:02d}_{method}"
        assert (folder / "equation.png").is_file()
        assert (folder / "03_residuals.png").is_file()
        root = ET.parse(folder / "02_predictions.svg").getroot()
        points = [p for p in root.iter() if "data-row" in p.attrib]
        assert len(points) == 1
        assert {p.attrib["data-series"] for p in points} == {"a" if index == 1 else "b"}
        visible = " ".join(p.text or "" for p in root.iter() if p.tag.endswith("text"))
        assert "coef[" not in visible
        assert method in visible
        csv_path = folder / root.attrib["data-source-csv"]
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == root.attrib["data-source-sha256"]
    exact = (tmp_path / "figures/01_native-a/equation.txt").read_text(encoding="utf-8")
    assert "1.2345678901234567" in exact
    assert "unit unknown" in exact
    assert json.loads((tmp_path / "metadata/plot_data.json").read_text()) == data


def test_equation_pages_keep_long_formula_without_silent_omission():
    from ml_platform_symbolic.figure_output import _equation_sheets, _model_summaries
    view = saved_view()
    formula = "y = " + " + ".join(f"coefficient_{i} * x" for i in range(200))
    view["candidates"][0]["public_formula"] = formula
    model = _model_summaries(view["candidates"])["a"]
    exact, pages = _equation_sheets(model)
    assert len(pages) > 1 and formula in exact
    visible = " ".join(" ".join(ET.fromstring(p).itertext()) for p in pages)
    assert "coefficient_0" in visible and "coefficient_199" in visible


@pytest.mark.parametrize("legacy", [False, True])
def test_final_graphs_only_use_bound_saved_audit_arrays(synthetic_config, tmp_path, monkeypatch, legacy):
    import numpy as np
    from ml_platform_symbolic import (
        assess_audit_support,
        audit_hypotheses,
        choose_hypothesis,
        compare_equations,
        create_study,
    )
    from ml_platform_symbolic.data import digest, read_json, write_json
    from ml_platform_symbolic.report_data import build_report_view

    study = create_study({"problem": synthetic_config, "sources": [{"manual": {
        "id": "curve", "expression": "2*x+1"}}]}, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    decision = choose_hypothesis(comparison, run_dir=tmp_path / "choice", candidate_id=None,
                                 retain=["curve@manual"], reason="Fixed fixture before Final")
    audit = audit_hypotheses(study, decision, run_dir=tmp_path / "audit")
    if legacy:
        with np.load(audit / "predictions.npz") as stored:
            arrays = {key: stored[key] for key in stored.files if key not in {"observed", "c0_residual"}}
        np.savez_compressed(audit / "predictions.npz", **arrays)
        record = read_json(audit / "audit.json")
        record["predictions_sha256"] = digest(audit / "predictions.npz")
        write_json(audit / "audit.json", record)
    support = assess_audit_support(audit, decision, run_dir=tmp_path / "support")

    def forbidden(*args, **kwargs):
        raise AssertionError("A renderer must not reopen Final or predict")

    monkeypatch.setattr("ml_platform_symbolic.data.load_prepared", forbidden)
    monkeypatch.setattr("ml_platform_symbolic.equation_evaluation.numerical.predict_record", forbidden)
    view = build_report_view(comparison, decision_dir=decision, audit_dir=support)
    plots = build_plot_data(view)
    final_figures = [f for f in plots["figures"] if f.get("partition") == "Final"]
    if legacy:
        assert not final_figures
        assert any(item["kind"] == "final_prediction" for item in plots["unavailable"])
    else:
        assert len(final_figures) == 2
        final = view["candidates"][0]["final"]
        np.testing.assert_allclose(final["residual_scatter"]["y"],
                                   np.array(final["prediction_scatter"]["observed"]) - final["prediction_scatter"]["predicted"])
        render_figures(plots, tmp_path / "graphs", per_method=True)
        svg = next((tmp_path / "graphs").glob("*/*final_predictions.svg"))
        assert "Final data" in svg.read_text()


def test_display_subset_keeps_candidate_numbers_and_visible_identity(tmp_path):
    view = saved_view()
    view["display_ids"] = ["b"]
    data = build_plot_data(view)
    render_figures(data, tmp_path, per_method=True)
    assert not (tmp_path / "01_native-b").exists()
    folder = tmp_path / "02_native-b"
    for name in ("02_predictions.svg", "03_residuals.svg", "equation.svg"):
        visible = " ".join(ET.parse(folder / name).getroot().itertext())
        assert "2. native-b" in visible
        assert "Candidate: b" in visible or "Candidate b" in visible


@pytest.mark.parametrize("values,compressed", [([0.0, 1.0, 1e9], True), ([0.0, 0.0, 0.0], False), ([0.0, 1.0, 2.0], False), ([0.0, 1e-300, 1e300], True)])
def test_error_axis_preserves_zero_outliers_and_exact_values(values, compressed):
    from ml_platform_symbolic.figure_output import _svg
    view = saved_view()
    view["candidates"] = [deepcopy(view["candidates"][0]) for _ in values]
    for i, (card, value) in enumerate(zip(view["candidates"], values, strict=True)):
        card.update(id=str(i), selection={"rmse": value}, complexity={"nodes": i + 1})
    figure = build_plot_data(view)["figures"][0]
    original = deepcopy(figure)
    svg = _svg(figure, "accuracy.csv", "test")
    root = ET.fromstring(svg)
    points = [node for node in root.iter() if "data-row" in node.attrib]
    assert len(points) == len(values)
    assert [float(node.attrib["data-y"]) for node in points] == values
    axes = next(node for node in root.iter() if "data-y-scale" in node.attrib)
    assert (axes.attrib["data-y-scale"] != "linear") is compressed
    assert figure == original
    low, high = float(axes.attrib["data-plot-top"]), float(axes.attrib["data-plot-bottom"])
    assert all(low <= float(node.attrib["cy"]) <= high for node in points)
    if compressed:
        assert float(points[0].attrib["cy"]) > float(points[1].attrib["cy"]) > float(points[2].attrib["cy"])


def test_native_only_failure_is_not_presented_as_an_empty_prediction_series(tmp_path):
    view = dict(candidates=[dict(id="failed", method="example", eligible=False,
                subject={"view": "native_only"}, reason="No proposals met the declared grammar",
                complexity={"nodes": None}, selection={})], display_ids=[], unit="1")
    data = build_plot_data(view)
    render_figures(data, tmp_path / "figures")
    svg = (tmp_path / "figures/01_accuracy.svg").read_text(encoding="utf-8")
    assert "No candidate equations available." in svg
    assert "No proposals met the declared grammar" in svg
    assert data["figures"][0]["rows"][0]["y"] is None


def test_source_unit_label_changes_only_presentation_not_coordinates():
    view = saved_view()
    original = deepcopy(view)
    baseline = build_plot_data(view)
    view["unit_label"] = "MPa raw; model=1"
    labeled = build_plot_data(view)
    assert view["unit"] == original["unit"]
    for before, after in zip(baseline["figures"], labeled["figures"], strict=True):
        assert before["rows"] == after["rows"]
    prediction = next(f for f in labeled["figures"] if f["kind"] == "observed_prediction")
    assert "MPa raw; model=1" in prediction["x_label"] and "MPa raw; model=1" in prediction["y_label"]
    assert "MPa raw" not in str(baseline)


def test_per_method_output_also_renders_same_path_overlay_and_pair(tmp_path, monkeypatch):
    from ml_platform_symbolic.equation_evaluation import numerical

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering must not calculate predictions")

    monkeypatch.setattr(numerical, "predict_record", forbidden)
    data = build_plot_data(saved_view())
    before = deepcopy(data)
    render_figures(data, tmp_path / "figures", table_dir=tmp_path / "tables", per_method=True)
    comparison = tmp_path / "figures/comparison"
    assert len(list(comparison.glob("*.png"))) == 2
    difference = next(comparison.glob("*difference*.svg"))
    root = ET.parse(difference).getroot()
    text = " ".join(root.itertext())
    assert "native-a" in text and "native-b" in text and "fixed = 5" in text
    csv_path = comparison / root.attrib["data-source-csv"]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["y"] for r in rows] == ["0.0", "", "-1.0"]
    assert [r["plotted"] for r in rows] == ["True", "False", "True"]
    assert data == before


def test_nonzero_path_origin_changes_only_plot_axis_preserving_all_saved_evidence():
    view = saved_view()
    for card in view["candidates"]:
        path = card["interpretations"][0]["paths"][0]
        path.update(path_unit="mm", at={"x": 28, "fixed": 5}, source_points=[[28, 5], [30, 5], [29, 5]])
    view["analysis_comparisons"][0]["path_unit"] = "mm"
    original = deepcopy(view)
    data = build_plot_data(view)
    for kind in ("response", "response_difference", "input_geometry"):
        figure = next(f for f in data["figures"] if f["kind"] == kind)
        assert figure["x_label"] == "x [mm]"
        assert [r["x"] for r in figure["rows"][:3]] == [28, 30, 29]
        assert [r["path_step"] for r in figure["rows"][:3]] == [0, 2, 1]
    assert view == original


def test_representative_accuracy_is_separate_from_extreme_all_candidate_overview(tmp_path):
    view = saved_view()
    base = deepcopy(view['candidates'][0])
    view['candidates'] = []
    for i, (method, error, nodes) in enumerate([
        ('pysr', 0.2, 9), ('physo', 1.0, 9), ('torchsisso', 2.0, 15),
        ('parfam', 3.0, 70), ('e2e', 6.0, 12), ('parfam', 1e42, 95),
    ]):
        card = deepcopy(base)
        card.update(id=str(i), method=method, complexity={'nodes': nodes},
                    selection={'rmse': error, 'coverage': 1.0})
        view['candidates'].append(card)
    view['display_ids'] = [str(i) for i in range(5)]
    view['analysis_comparisons'] = []
    before = deepcopy(view)
    plots = build_plot_data(view)
    overview = next(f for f in plots['figures'] if f['kind'] == 'selection_tradeoff')
    representatives = next(f for f in plots['figures'] if f.get('candidate_scope') == 'representatives')
    assert len(overview['rows']) == 6
    assert max(r['y'] for r in overview['rows']) == 1e42
    assert len(representatives['rows']) == 5
    assert max(r['y'] for r in representatives['rows']) == 6.0
    assert {r['candidate_id'] for r in representatives['rows']} == set(view['display_ids'])
    render_figures(plots, tmp_path / 'figures', per_method=True)
    svg = (tmp_path / 'figures' / (representatives['id'] + '.svg')).read_text()
    assert 'Start with *_representatives.png' in (tmp_path / 'figures' / 'README.txt').read_text()
    assert 'Display representatives only' in svg
    assert 'All candidates with finite scores' not in svg
    assert (tmp_path / 'figures' / (representatives['id'] + '.png')).is_file()
    assert view == before


def test_selection_display_does_not_invent_missing_metric_coverage(tmp_path):
    view = saved_view()
    for card in view['candidates']:
        card['selection'].pop('coverage', None)
    plots = build_plot_data(view)
    overview = next(f for f in plots['figures'] if f['kind'] == 'selection_tradeoff')
    assert all('coverage' not in row for row in overview['rows'])
    render_figures(plots, tmp_path, per_method=True)
    with (tmp_path / 'displayed_models.csv').open() as file:
        table = csv.DictReader(file)
        assert 'selection_coverage' not in table.fieldnames
    readme = (tmp_path / '01_native-a' / 'README.txt').read_text()
    assert 'coverage:' not in readme
    assert 'Selection RMSE:' in readme
    assert 'Valid points:' in (tmp_path / '01_native-a' / '02_predictions.svg').read_text()


def test_linear_ticks_are_nice_in_original_not_normalized_coordinates():
    from ml_platform_symbolic.figure_output import _axis_layout

    assert _axis_layout([0.2, 9.0])[3] == pytest.approx([0, 2, 4, 6, 8])
    ticks = _axis_layout([-1.78e-15, 8.88e-16])[3]
    assert ticks == pytest.approx([-1e-15, 0, 1e-15], abs=1e-29)
    for values in ([3, 5, 7], [5], [0, 257]):
        ticks = _axis_layout(values, integer=True, nonnegative=True)[3]
        assert ticks and all(abs(tick - round(tick)) < 1e-10 for tick in ticks)


def test_parity_axes_have_same_limits_and_equal_physical_aspect():
    from ml_platform_symbolic.figure_output import _svg

    figure = next(f for f in build_plot_data(saved_view())["figures"] if f["kind"] == "observed_prediction")
    root = ET.fromstring(_svg(figure, "source.csv", "hash"))
    axes = next(node for node in root.iter() if "data-plot-left" in node.attrib)
    width = float(axes.attrib["data-plot-right"]) - float(axes.attrib["data-plot-left"])
    height = float(axes.attrib["data-plot-bottom"]) - float(axes.attrib["data-plot-top"])
    assert width == height
    ticks = {axis: [float(node.attrib["data-tick"]) for node in root.iter() if node.attrib.get("data-axis") == axis]
             for axis in ("x", "y")}
    assert ticks["x"] == ticks["y"]
    assert len([node for node in root.iter() if node.attrib.get("data-reference") == "identity"]) == 1


def test_single_series_is_monochrome_and_reference_is_not_a_fake_legend_point(tmp_path):
    view = saved_view()
    view["display_ids"] = ["a"]
    render_figures(build_plot_data(view), tmp_path, per_method=True)
    root = ET.parse(tmp_path / "01_native-a/02_predictions.svg").getroot()
    points = [node for node in root.iter() if "data-row" in node.attrib]
    assert {node.attrib["fill"] for node in points} == {"#222222"}
    assert len([node for node in root.iter() if node.tag.endswith("circle")]) == len(points)
    assert 'font-weight="bold"' not in ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("values,expected", [([1e-16, 10.0], "log"), ([0, 1e-16, 10.0], "symlog"),
                                            ([0, 0], "linear"), ([1, 2], "linear")])
def test_error_scale_is_explicit_and_never_moves_zero_to_an_epsilon(values, expected):
    from ml_platform_symbolic.figure_output import _svg

    view = saved_view()
    template = view["candidates"][0]
    view["candidates"] = [{**deepcopy(template), "id": str(i), "selection": {"rmse": value}}
                          for i, value in enumerate(values)]
    root = ET.fromstring(_svg(build_plot_data(view)["figures"][0], "source.csv", "hash"))
    assert next(node for node in root.iter() if "data-y-scale" in node.attrib).attrib["data-y-scale"] == expected
    assert [float(node.attrib["data-y"]) for node in root.iter() if "data-row" in node.attrib] == values
    if expected == "symlog":
        assert any(node.attrib.get("data-axis") == "y" and float(node.attrib["data-tick"]) == 0 for node in root.iter())


def test_high_dimensional_path_keeps_full_conditions_outside_image(tmp_path):
    view = saved_view()
    for card in view["candidates"]:
        path = card["interpretations"][0]["paths"][0]
        path["at"] = {f"feature_{i}": i for i in range(100)}
        path["direction"] = {f"feature_{i}": int(i == 0) for i in range(100)}
        path["input_units"] = {f"feature_{i}": "mm" for i in range(100)}
    data = build_plot_data(view)
    figure = next(f for f in data["figures"] if f["kind"] == "response")
    render_figures({**data, "figures": [figure]}, tmp_path)
    root = ET.parse(tmp_path / (figure["id"] + ".svg")).getroot()
    assert int(root.attrib["height"]) < 1000
    assert "100 inputs" in " ".join(root.itertext())
    caption = (tmp_path / (figure["id"] + ".txt")).read_text()
    assert '"feature_99": 99' in caption
    assert '"feature_99": "mm"' in caption


@pytest.mark.parametrize("values", [[1e12, 1e12 + 1], [1e-4, 1.00001e-4]])
def test_offset_axes_expose_small_changes_without_repeated_tick_labels(values):
    from ml_platform_symbolic.figure_output import _svg

    view = saved_view()
    view["display_ids"] = ["a"]
    view["candidates"][0]["prediction_scatter"] = {"observed": values, "predicted": values, "valid": [True, True]}
    figure = next(f for f in build_plot_data(view)["figures"] if f["kind"] == "observed_prediction")
    root = ET.fromstring(_svg(figure, "source.csv", "hash"))
    for axis in ("x", "y"):
        labels = [node.text for node in root.iter() if node.attrib.get("data-axis") == axis]
        assert len(labels) >= 2 and len(labels) == len(set(labels))
    assert "Offset +" in " ".join(root.itertext())
    assert [float(node.attrib["data-x"]) for node in root.iter() if "data-row" in node.attrib] == values


def test_multi_series_retains_method_colour_across_subset_and_dash_in_legend():
    from ml_platform_symbolic.figure_output import _svg

    view = saved_view()
    view["display_ids"] = ["b", "a"]
    figures = build_plot_data(view)["figures"]
    response = next(f for f in figures if f["kind"] == "response")
    root = ET.fromstring(_svg(response, "source.csv", "hash"))
    model_colours = {model["candidate_id"]: model["color"] for model in response["models"]}
    for node in root.iter():
        if "data-row" in node.attrib:
            assert node.attrib.get("fill", node.attrib.get("stroke")) == model_colours[node.attrib["data-series"]]
    legends = [node for node in root.iter() if node.attrib.get("data-legend-line") == "true"]
    assert len(legends) == 2 and len({node.attrib["stroke-dasharray"] for node in legends}) == 2


def test_long_axis_labels_wrap_or_are_explicitly_abbreviated_with_units_and_caption(tmp_path):
    from ml_platform_symbolic.figure_output import _figure_text, _svg

    figure = next(f for f in build_plot_data(saved_view())["figures"] if f["kind"] == "residual")
    figure["x_label"] = "measurements_with_a_long_but_meaningful_physical_variable_name_" * 3 + " [m]"
    figure["y_label"] = "Measured corrected pressure difference across the experimental apparatus [kPa]"
    figure["caption_path"] = "../../metadata/source.txt"
    root = ET.fromstring(_svg(figure, "source.csv", "hash"))
    texts = [node.text for node in root.iter() if node.tag.endswith("text")]
    assert any("… [m]" == value for value in texts)
    assert any("[kPa]" in value for value in texts)
    assert any("Full axis labels: ../../metadata/source.txt" == value for value in texts)
    assert figure["x_label"] in _figure_text(figure) and figure["y_label"] in _figure_text(figure)


def test_long_native_failure_reason_has_explicit_excerpt_notice():
    from ml_platform_symbolic.figure_output import _svg

    reason = "Native search could not produce a candidate. " * 40
    view = dict(candidates=[dict(id="failed", method="example", eligible=False,
                subject={"view": "native_only"}, reason=reason,
                complexity={"nodes": None}, selection={})], display_ids=[], unit="1")
    figure = build_plot_data(view)["figures"][0]
    svg = _svg(figure, "all_rows.csv", "hash")
    assert "Full reasons are preserved in CSV: all_rows.csv" in svg
    assert figure["rows"][0]["reason"] == reason
