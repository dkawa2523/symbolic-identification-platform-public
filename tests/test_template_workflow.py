"""Template intent survives the same prepare/discover/study/save boundaries."""

import json
from copy import deepcopy

import pytest
from ml_platform_symbolic import compare_equations, create_study, render_report
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import prepare, read_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.problem import problem_config
from ml_platform_symbolic.report_data import build_report_view


def template_config(config, engine):
    config = deepcopy(config)
    config.pop("constraints", None)
    config["search"].update(engines=[engine], seeds=[7], max_wall_seconds=30)
    config["requirements"] = {
        "allowed_operators": ["add", "mul"],
        "require_units": True,
        "max_nodes": 30,
        "fixed_constants": {"k": {"value": 1.0, "unit": "1"}},
        "equation_template": {
            "coordinate_space": "working",
            "ast": {"op": "add", "args": [{"ref": "response"}, {"param": "k"}]},
            "components": {
                "response": {
                    "ast": {
                        "op": "mul",
                        "args": [{"param": "b"}, {"op": "mul", "args": [{"var": "x0"}, {"var": "x0"}]}],
                    },
                    "description": "Declared quadratic component",
                }
            },
            "parameters": {
                "a": {"value": 0.5, "trainable": True, "bounds": [0.1, 3.0], "unit": "1"},
                "b": {"derived_from": {"op": "mul", "args": [{"const": 2.0}, {"param": "a"}]}, "unit": "1"},
            },
            "slots": {},
            "knowledge": {"kind": "hypothesis", "statement": "Supplied quadratic family"},
        },
    }
    return resolve_config(config)


def test_template_complete_fit_and_saved_readout(synthetic_config, tmp_path, monkeypatch):
    config = template_config(synthetic_config, "known_form")
    refs = prepare(config, tmp_path / "prepare")
    provenance = read_json(refs["run"])
    assert problem_config(provenance) == config
    discovery = run_discovery(refs, provenance, "known_form", 7, tmp_path / "native")
    summary = read_json(discovery["summary"])
    assert summary["status"] == "ok"
    candidate = json.loads((tmp_path / "native/candidates.jsonl").read_text().splitlines()[0])
    assert candidate["equation"]["parameters"]["k"]["value"] == 1
    study = create_study(
        {
            "prepared_run": str(refs["run"]),
            "sources": [{"discovery": discovery["summary"]}],
            "parameter_evidence_ids": [candidate["candidate_id"] + "@native"],
            "engineering_tolerance": 0.01,
            "final_history": "known_exposed",
        },
        run_dir=tmp_path / "study",
    )
    saved = read_json(study / "numerical.json")[0]["local_parameter_identifiability"]
    assert saved["status"] == "computed" and saved["rank"] == 1, saved.get("reason")
    assert saved["parameter_names"] == ["a"]  # derived b and fixed k are not free coordinates
    assert saved["data_role"] == "selection"
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    def no_recomputation(*args, **kwargs):
        raise AssertionError("Reporting must consume preserved coefficient evidence")

    monkeypatch.setattr(
        "ml_platform_symbolic.equation_evaluation.numerical.local_parameter_evidence", no_recomputation
    )
    view = build_report_view(comparison)
    item = view["candidates"][0]
    assert item["local_parameter_identifiability"] == saved
    assert item["template"]["dependent_parameters"][0]["value"] == pytest.approx(2.0)
    assert item["template"]["components"][0]["name"] == "response"
    assert item["complexity"]["free"] == 1
    report = render_report(comparison, output=tmp_path / "report.html")
    assert report.is_file()
    from ml_platform_symbolic.reporting import export_output_data
    import csv
    import xml.etree.ElementTree as ET

    output = export_output_data(view, output_dir=tmp_path / "output")
    text = (output / "equations.txt").read_text()
    assert "declared component: response" in text and "dependent parameter: b" in text
    assert "canonical working" in text and "hypothesis" in text
    with (output / "tables/template_parts.csv").open(newline="") as stream:
        parts = list(csv.DictReader(stream))
    assert {row["section"] for row in parts} >= {"declared_outer", "declared_component", "dependent_parameter"}
    visible = " ".join(" ".join(ET.parse(path).getroot().itertext())
                       for path in (output / "figures").glob("*/equation*.svg"))
    assert "declared component: response" in visible and "dependent parameter: b" in visible



def test_unsupported_adapter_does_not_run_unconstrained(synthetic_config, tmp_path):
    config = template_config(synthetic_config, "dictionary")
    refs = prepare(config, tmp_path / "prepare")
    provenance = read_json(refs["run"])
    with pytest.raises(RuntimeError, match="unsupported"):
        run_discovery(refs, provenance, "dictionary", 7, tmp_path / "native")
    summary = read_json(tmp_path / "native/engine_summary.json")
    assert summary["status"] == "unsupported"
    assert summary["native_execution_status"] == "not_started"
    assert not (tmp_path / "native/candidates.jsonl").exists()
