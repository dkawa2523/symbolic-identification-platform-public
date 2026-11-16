"""Additional interpretation evidence must not replace or mix frozen sources."""

from copy import deepcopy

import pytest
from ml_platform_symbolic import compare_equations, create_study, evaluate_equations, export_result, load_result
from ml_platform_symbolic.data import read_json, write_json
from ml_platform_symbolic.equation_evaluation.backends.interpretation import _paths
from ml_platform_symbolic.report_data import build_report_view


def test_supplement_preserves_both_interpretations_and_binds_same_source_paths(synthetic_config, tmp_path):
    study = create_study(
        {
            "problem": synthetic_config,
            "sources": [{"manual": {"id": "a", "expression": "x"}}, {"manual": {"id": "b", "expression": "2*x"}}],
        },
        run_dir=tmp_path / "study",
    )
    analyses = []
    for index, at in enumerate((1.0, 2.0)):
        analysis = evaluate_equations(
            study, "interpretation", run_dir=tmp_path / f"analysis{index}",
            options={"reference": {"x": at}, "paths": [{
                "id": "response", "at": {"x": at}, "direction": {"x": 1.0},
                "steps": [0.0, 0.1], "path_unit": "1",
            }]},
        )
        assert read_json(analysis / "result.json")["outcome"] == "ok"
        analyses.append(analysis)
    primary = compare_equations(study, run_dir=tmp_path / "primary", evaluations=analyses[:1])
    supplement = compare_equations(study, run_dir=tmp_path / "supplement", evaluations=analyses[1:])
    original = {p: p.read_bytes() for directory in [study, *analyses, primary, supplement] for p in directory.iterdir() if p.is_file()}
    view = build_report_view(primary, supplemental_comparison=supplement)
    assert [a["analysis_sha256"] for a in view["analyses"]] == [a["analysis_sha256"] for a in view["candidates"][0]["interpretations"]]
    for card in view["candidates"]:
        assert [item["reference"]["x"] for item in card["interpretations"]] == [1.0, 2.0]
        assert card["interpretation"] == card["interpretations"][0]
        keys = [item["paths"][0]["display_id"] for item in card["interpretations"]]
        assert len(set(keys)) == 2
        assert keys == [item["path_display_id"] for item in view["analysis_comparisons"]]
    assert [item["difference"] for item in view["analysis_comparisons"]] == [[-1.0, -1.1], [-2.0, -2.1]]
    detached = export_result(primary, run_dir=tmp_path / "detached", supplemental_comparison=supplement)
    saved = load_result(detached, load_predictors=False)["view"]
    assert saved["candidates"] == view["candidates"]
    assert saved["analysis_comparisons"] == view["analysis_comparisons"]
    assert all(p.read_bytes() == before for p, before in original.items())

    # Historical comparisons do not record which analysis produced a path diff.
    # One source can be resolved from frozen definitions; two are not guessed.
    for directory in (primary, supplement):
        document = read_json(directory / "comparison.json")
        for item in document["analysis_comparisons"]:
            item.pop("analysis_sha256")
        write_json(directory / "comparison.json", document)
    legacy = build_report_view(primary)
    assert legacy["analysis_comparisons"][0]["path_display_id"] is not None
    legacy_combined = build_report_view(primary, supplemental_comparison=supplement)
    assert len(legacy_combined["analysis_comparisons"]) == 2
    assert all(item["path_display_id"] is None for item in legacy_combined["analysis_comparisons"])
    assert len(legacy_combined["candidates"][0]["interpretations"]) == 2


@pytest.mark.parametrize("ids", [[None], [""], ["  "], [123], ["repeated", "repeated"]])
def test_interpretation_paths_require_unambiguous_names(ids):
    schema = {"variables": [{"id": "x0", "column": "x", "unit": "1"}], "target": {"column": "y", "unit": "1"}}
    path = {"at": {"x": 1.0}, "direction": {"x": 1.0}, "steps": [0.0, 0.1], "path_unit": "1"}
    options = {"paths": [dict(path, **({"id": name} if name is not None else {})) for name in ids]}
    before = deepcopy(options)
    with pytest.raises(ValueError, match="unique nonempty string IDs"):
        _paths({}, options, schema)
    assert options == before


def test_report_preserves_invalid_metric_reasons_and_dataset_identity(synthetic_config, tmp_path, monkeypatch):
    study = create_study(
        {"problem": synthetic_config, "sources": [
            {"manual": {"id": "invalid_domain", "expression": "sqrt(x-2.5)"}},
            {"manual": {"id": "valid", "expression": "x"}},
        ]},
        run_dir=tmp_path / "study",
    )
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    original = {p: p.read_bytes() for directory in [study, comparison] for p in directory.iterdir() if p.is_file()}
    from ml_platform_symbolic import metrics
    from ml_platform_symbolic.equation_evaluation import numerical

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must not recompute predictions or metric scores")

    monkeypatch.setattr(numerical, "predict_record", forbidden)
    monkeypatch.setattr(metrics, "prediction_metrics", forbidden)
    view = build_report_view(comparison)
    invalid = next(c for c in view["candidates"] if c["id"] == "invalid_domain@manual")
    assert invalid["selection"]["rmse"] is None
    assert "invalid" in invalid["selection"]["reason"]
    assert invalid["eligible"] is False
    assert invalid["checks"]
    assert len(invalid["prediction_scatter"]["observed"]) == view["selection_rows"]
    assert len(invalid["prediction_scatter"]["predicted"]) == view["selection_rows"]
    assert not all(invalid["prediction_scatter"]["valid"])
    assert view["audit"]["provided"] is False
    context = read_json(study / "resolved.json")
    for key in ("data_sha256", "split_sha256", "group_column", "strata_column"):
        assert view["context"][key] == context.get(key)
    assert view["context"]["target"]["column"] == "y"
    assert all(p.read_bytes() == before for p, before in original.items())
