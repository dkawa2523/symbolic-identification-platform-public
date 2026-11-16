from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ml_platform_symbolic import fit_hypotheses, render_report
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.equation_comparison import compare_equations
from ml_platform_symbolic.equation_evaluation import create_study, load_study
from ml_platform_symbolic.expressions import evaluate, inspect_equation, parse_expression
from ml_platform_symbolic.fitting import fit_coefficients
from ml_platform_symbolic.model import load_model, save_model
from ml_platform_symbolic.selection import choose_hypothesis, finalize_hypothesis


def plan(config):
    return {
        "problem": config,
        "purpose": "local_prediction",
        "engineering_tolerance": 0.01,
        "sources": [
            {
                "manual": {
                    "id": "quadratic",
                    "expression": "a*x*x+b",
                    "parameters": {
                        "a": {"value": 2.0, "trainable": True, "bounds": [0, 5], "unit": "1"},
                        "b": {"value": 1.0, "trainable": True, "bounds": [-5, 5], "unit": "1"},
                    },
                }
            },
            {"manual": {"id": "linear", "expression": "2*x+1"}},
        ],
    }


def test_manual_study_compare_choose_finalize(synthetic_config, tmp_path):
    native_input = plan(synthetic_config)
    original = deepcopy(native_input)
    study = create_study(native_input, run_dir=tmp_path / "study")
    assert native_input == original
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    info = read_json(comparison / "comparison.json")
    assert len(info["pairs"]) == 1
    assert info["pairs"][0]["prediction_rms_difference"] > 1
    assert not (study / "decision.json").exists()
    choice = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "choice",
        candidate_id="quadratic@manual",
        reason="The declared formula meets the predeclared local prediction tolerance",
    )
    result = finalize_hypothesis(study, choice, run_dir=tmp_path / "final")
    model = load_model(result.artifacts["model_bundle"])
    frame = pd.read_csv(synthetic_config["data"]["local_path"])
    np.testing.assert_allclose(model.predict(frame), frame.y)


def test_missing_support_defers_valid_equation(synthetic_config, tmp_path):
    config = plan(synthetic_config)
    config["required_evidence"] = ["likelihood_profiler"]
    study = create_study(config, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    choice = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "choice",
        candidate_id="quadratic@manual",
        reason="Keep the valid equation while evidence is not yet computed",
    )
    decision = read_json(choice / "decision.json")
    assert decision["status"] == "defer"
    assert read_json(comparison / "comparison.json")["candidates"][0]["eligible"]
    with pytest.raises(ValueError, match="adopted"):
        finalize_hypothesis(study, choice, run_dir=tmp_path / "final")


def test_final_exposure_is_shared_by_studies_of_one_prepared_context(synthetic_config, tmp_path):
    config = plan(synthetic_config)
    config["final_history"] = "declared_unseen"
    first = create_study(config, run_dir=tmp_path / "first")
    context = load_study(first)[0]
    prepared = context["prepared"]["run"]
    # Both studies exist before the first Final access, so finalization must re-read
    # the common authority, rather than trust either study's frozen history alone.
    second = create_study(
        {
            "prepared_run": prepared,
            "sources": config["sources"],
            "engineering_tolerance": 0.01,
            "final_history": "declared_unseen",
        },
        run_dir=tmp_path / "second",
    )
    for index, study in enumerate((first, second)):
        comparison = compare_equations(study, run_dir=tmp_path / f"comparison{index}")
        choice = choose_hypothesis(
            comparison,
            run_dir=tmp_path / f"choice{index}",
            candidate_id="quadratic@manual",
            reason="Local prediction under the frozen tolerance",
        )
        result = finalize_hypothesis(study, choice, run_dir=tmp_path / f"final{index}")
        role = load_model(result.artifacts["model_bundle"]).metadata["final_evidence_role"]
        assert role.startswith("holdout" if index == 0 else "post-selection audit")
    third = create_study(
        {"prepared_run": prepared, "sources": config["sources"], "final_history": "declared_unseen"},
        run_dir=tmp_path / "third",
    )
    assert load_study(third)[0]["final_exposure"]["same_data_exposed"]


def test_domain_hole_is_not_simplified_or_dropped(synthetic_config, tmp_path):
    config = plan(synthetic_config)
    config["problem"]["data"]["domain"] = {"x": [0.2, 3]}
    config["sources"] = [
        {"manual": {"id": "hole", "expression": "(x*x-1)/(x-1)"}},
        {"manual": {"id": "line", "expression": "x+1"}},
    ]
    study = create_study(config, run_dir=tmp_path / "study")
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    result = read_json(comparison / "comparison.json")
    assert result["pairs"][0]["equivalence"] == "domain counterexample"
    assert result["pairs"][0]["counterexample_X"] == [1.0]
    assert not result["candidates"][0]["eligible"]


def test_study_detects_changed_evidence(synthetic_config, tmp_path):
    study = create_study(plan(synthetic_config), run_dir=tmp_path / "study")
    with (study / "hypotheses.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="evidence changed"):
        compare_equations(study, run_dir=tmp_path / "comparison")


def test_added_axis_is_retained_but_new_context_cannot_reuse_it(synthetic_config, tmp_path):
    from ml_platform_symbolic.data import digest, write_json

    first = create_study(plan(synthetic_config), run_dir=tmp_path / "first")
    # A fixture for an extension's unavailable result, not a real backend claim.
    result = write_json(
        tmp_path / "axis.json",
        {
            "method": "future_axis",
            "result_kind": "future.axis/v1",
            "outcome": "unavailable",
            "results": [],
            "reason": "fixture, not computed",
            "study_sha256": digest(first / "study.json"),
        },
    )
    comparison = compare_equations(first, run_dir=tmp_path / "comparison", evaluations=[result])
    assert read_json(comparison / "comparison.json")["analyses"][0]["result"]["outcome"] == "unavailable"
    changed = plan(synthetic_config)
    changed["slices"] = {"different": {"feature": "x", "at": {}, "grid": [1, 2]}}
    second = create_study(changed, run_dir=tmp_path / "second")
    with pytest.raises(ValueError, match="context/pool changed"):
        compare_equations(second, run_dir=tmp_path / "stale", evaluations=[result])


def test_named_fit_and_readout_preserve_original_evidence(synthetic_config, tmp_path):
    config = plan(synthetic_config)
    config["question"] = "Local prediction <script>unsafe</script>"
    config["sources"][0]["manual"]["parameters"]["a"]["value"] = 0.5
    config["slices"] = {"operating": {"feature": "x", "at": {}, "grid": [0.5, 1, 2]}}
    original = create_study(config, run_dir=tmp_path / "source")
    records_before = read_json(original / "hypotheses.json")
    fits = fit_hypotheses(original, run_dir=tmp_path / "fit", ids=["quadratic@manual"])
    artifact = read_json(fits / "fits.json")["artifacts"][0]
    context = load_study(original)[0]
    new = create_study(
        {
            "prepared_run": context["prepared"]["run"],
            "sources": [{"derived": artifact}],
            "question": config["question"],
        },
        run_dir=tmp_path / "new",
    )
    assert load_study(new)[2][0]["metrics"]["selection"]["rmse"] < 1e-7
    assert read_json(original / "hypotheses.json") == records_before
    comparison = compare_equations(original, run_dir=tmp_path / "comparison")
    before = (comparison / "comparison.json").read_bytes()
    report = render_report(comparison, output=tmp_path / "report.html")
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in report.read_text()
    assert (comparison / "comparison.json").read_bytes() == before


def test_tied_coefficients_keep_identity_and_fixed_literals():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    parameters = {
        "a": {"value": 1.0, "trainable": True, "bounds": [0, 10], "unit": "1"},
        "b": {"value": 1.0, "trainable": False, "bounds": [0, 10], "unit": "1", "tied_to": "a"},
    }
    equation = parse_expression("b*x0+2*a", variables, trainable=False, parameters=parameters)
    original = deepcopy(equation)
    X = np.linspace(0.2, 2, 20).reshape(-1, 1)
    result = fit_coefficients(equation, X, 3 * X[:, 0] + 6, variables)
    assert inspect_equation(result, variables)["n_parameters"] == 1
    assert result["parameters"]["b"]["value"] == result["parameters"]["a"]["value"]
    np.testing.assert_allclose(evaluate(result, X, variables), 3 * X[:, 0] + 6)
    assert equation == original


def test_tied_alias_bounds_restrict_the_fitted_coordinate():
    variables = [{"id": "x0", "column": "x", "unit": "1"}]
    parameters = {
        "a": {"value": 0.5, "trainable": True, "bounds": [0, 10], "unit": "1"},
        "b": {"value": 0.5, "trainable": False, "bounds": [0, 1], "unit": "1", "tied_to": "a"},
    }
    equation = parse_expression("a*x0+b", variables, trainable=False, parameters=parameters)
    X = np.linspace(0.2, 2, 20).reshape(-1, 1)
    fitted = fit_coefficients(equation, X, 3 * X[:, 0] + 3, variables)
    assert fitted["parameters"]["a"]["value"] == pytest.approx(1, abs=1e-7)
    assert fitted["parameters"]["b"]["value"] <= 1


def test_bundle_restrictions_exposure_and_unknown_parity_are_preserved(synthetic_config, tmp_path):
    config = plan(synthetic_config)
    config["problem"]["data"]["domain"] = {"x": [0.5, 3], "y": [0, 100]}
    first = create_study(config, run_dir=tmp_path / "first")
    context, records, _, _ = load_study(first)
    provenance = read_json(context["prepared"]["run"])
    schema = {
        "variables": context["variables"],
        "target": {"column": "y", "unit": "1", "domain": [0, 25]},
        "feature_ranges": context["feature_ranges"],
    }
    bundle = save_model(
        tmp_path / "prior.zip",
        records[0]["equation"],
        schema,
        {"provenance": provenance, "final_metrics": {"rmse": 0.0}},
    )
    study = create_study(
        {
            "prepared_run": context["prepared"]["run"],
            "sources": [{"bundle": str(bundle)}],
            "engineering_tolerance": 0.01,
            "final_history": "declared_unseen",
            "slices": {"test": {"feature": "x", "at": {}, "grid": [4.0]}},
        },
        run_dir=tmp_path / "reanalysis",
    )
    ctx, imported, rows, arrays = load_study(study)
    assert rows[0]["native_fidelity"] == "unverified"
    assert not arrays["c0_slice_test_valid"][0]
    assert ctx["final_exposure"]["same_data_exposed"]
    comparison = compare_equations(study, run_dir=tmp_path / "comparison")
    choice = choose_hypothesis(
        comparison,
        run_dir=tmp_path / "choice",
        candidate_id=imported[0]["id"],
        reason="Portable local prediction only; prior Final exposure explicitly retained",
    )
    result = finalize_hypothesis(study, choice, run_dir=tmp_path / "audit")
    restored = load_model(result.artifacts["model_bundle"])
    assert restored.schema["target"]["domain"] == [0, 25]
    assert restored.metadata["final_evidence_role"].startswith("post-selection audit")
    with pytest.raises(ValueError, match="declared domain"):
        restored.predict(pd.DataFrame({"x": [4.0]}))


def test_representatives_share_one_rule_without_adoption_or_mutation():
    from copy import deepcopy

    from ml_platform_symbolic import select_representatives

    records = [{"id": name, "origin": {"engine": engine, "seed": seed}}
               for name, engine, seed in [("b", "a", 17), ("a", "a", 17), ("invalid", "a", 17),
                                           ("other", "a", 29), ("bad", "failed", 17)]]
    rows = [{"eligible": eligible, "metrics": {"selection": {"rmse": error}}}
            for eligible, error in [(True, 1.0), (True, 1.0), (False, 0.0), (True, 2.0), (True, float("inf"))]]
    before = deepcopy(records), deepcopy(rows)
    chosen = select_representatives(records, rows)
    assert {key: pair[0]["id"] for key, pair in chosen.items()} == {("a", 17): "a", ("a", 29): "other"}
    assert (records, rows) == before


def test_basic_example_is_selection_only_and_exports_graphs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    output = tmp_path / "basic"
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "examples/basic.py"),
                    "--output", str(output)], check=True)
    result = read_json(output / "calculation/summary.json")
    assert result["partition"] == "Selection" and not result["final_accessed"]
    assert {row["method"] for row in result["representatives"]} == {"mean", "linear", "dictionary"}
    assert (output / "equations.txt").is_file() and list((output / "figures").rglob("*.png"))
    assert (output / "tables/candidates.csv").is_file()
    assert not list(output.rglob("final_access.json")) and not (output / "calculation/final").exists()


@pytest.mark.parametrize("mode,history", [("development", "unknown"), ("historical", "declared_unseen"), ("development", "declared_unseen")])
def test_representative_test_scoring_preserves_indices_and_never_refits(synthetic_config, tmp_path, monkeypatch, mode, history):
    import runpy
    from pathlib import Path

    from ml_platform_symbolic import score_representatives
    from ml_platform_symbolic.equation_evaluation import numerical

    config = plan(synthetic_config)
    config["sources"].reverse()  # Winner has original index 1, not zero.
    config.update(mode=mode, final_history=history)
    study = create_study(config, run_dir=tmp_path / "study")
    _, records, _, _ = load_study(study)
    before = (study / "study.json").read_bytes()
    calls = []
    predict = numerical.predict_record

    def tracked(record, *args, **kwargs):
        calls.append(record["id"])
        return predict(record, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Final scoring must not fit")

    monkeypatch.setattr(numerical, "predict_record", tracked)
    monkeypatch.setattr("ml_platform_symbolic.fitting.fit_coefficients", forbidden)
    score = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/score.py"))["main"]
    output = tmp_path / "score"
    score(study, output)
    audit = read_json(output / "test/audit.json")
    assert calls == [records[1]["id"]]
    assert audit["results"][0]["source_index"] == 1
    expected = "holdout under explicit" if mode == "development" and history == "declared_unseen" else "post-selection audit"
    assert audit["final_evidence_role"].startswith(expected)
    assert not audit["selection_changed"] and not audit["refit"]
    with np.load(output / "test/predictions.npz") as arrays:
        assert "c1_prediction" in arrays and "c0_prediction" not in arrays
        from ml_platform_symbolic.reporting import load_report_view
        view = load_report_view(output / "output")
        assert set(view["audit"]["rows"]) == {records[1]["id"]}
        assert [card["id"] for card in view["candidates"]] == [records[1]["id"]]
        scatter = view["audit"]["rows"][records[1]["id"]]["prediction_scatter"]
        np.testing.assert_array_equal(scatter["predicted"], arrays["c1_prediction"])
        np.testing.assert_array_equal(scatter["observed"], arrays["observed"])
    assert (study / "study.json").read_bytes() == before
    assert (output / "summary.csv").is_file()
    assert all(row.get("reason") != "No eligible finite Selection candidate"
               for row in read_json(output / "summary.json")["attempts"] if row["status"] == "scored")
    assert list((output / "output/figures").rglob("*.png"))
    again = score_representatives(study, run_dir=tmp_path / "again")
    assert read_json(again / "audit.json")["final_evidence_role"].startswith("post-selection audit")


def test_representative_scoring_refuses_selection_as_test(tmp_path, monkeypatch):
    from ml_platform_symbolic import selection

    monkeypatch.setattr(selection, "load_study", lambda _: (
        {"prepared": {}}, [{"id": "one", "origin": {"engine": "linear", "seed": 17}}],
        [{"eligible": True, "metrics": {"selection": {"rmse": 0.0}}}], {}))
    with pytest.raises(ValueError, match="no Final partition"):
        selection.score_representatives(tmp_path, run_dir=tmp_path / "score")
    assert not (tmp_path / "score").exists()


def test_all_ineligible_benchmark_keeps_attempts_without_test_access(synthetic_config, tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    from ml_platform_symbolic import selection

    study = create_study(plan(synthetic_config), run_dir=tmp_path / "study")
    monkeypatch.setattr(selection, "select_representatives", lambda records, rows: {})
    from ml_platform_symbolic import equation_evaluation
    real_load = equation_evaluation.load_study

    def failed_load(path):
        context, records, rows, arrays = real_load(path)
        for record in records:
            record["origin"]["summary"] = {"status": "timeout", "reason": "Search exceeded 20 seconds", "wall_seconds": 20.1}
        return context, records, rows, arrays

    monkeypatch.setattr(equation_evaluation, "load_study", failed_load)
    score = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/score.py"))["main"]
    output = tmp_path / "score"
    score(study, output)
    summary = read_json(output / "summary.json")
    assert summary["attempts"] and all(row["status"] == "no_eligible_candidate" for row in summary["attempts"])
    assert summary["partition"] is None and not summary["final_accessed"]
    assert all(row["run_status"] == "timeout" and row["reason"] == "Search exceeded 20 seconds"
               and row["wall_seconds"] == 20.1 for row in summary["attempts"])
    assert not list(tmp_path.rglob("final_access.json"))
    assert not (output / "test").exists()


def test_benchmark_twenty_candidates_compares_all_five_frozen_representatives(synthetic_config, tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    from ml_platform_symbolic import selection, equation_comparison

    config = plan(synthetic_config)
    config["sources"] = [
        {"manual": {"id": f"m{method}_c{candidate}", "expression": f"2*x*x+{1 + candidate + method / 10}"}}
        for method in range(5) for candidate in range(4)
    ]
    study = create_study(config, run_dir=tmp_path / "study")
    real_load = load_study

    def method_origins(path):
        context, records, rows, arrays = real_load(path)
        for record in records:
            record["origin"].update(engine=record["id"].split("_")[0], seed=17)
        return context, records, rows, arrays

    # The fixture supplies five method origins; equations and stored predictions
    # are real common-study outputs, not substituted comparison results.
    monkeypatch.setattr(selection, "load_study", method_origins)
    monkeypatch.setattr(equation_comparison, "load_study", method_origins)
    score = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/score.py"))["main"]
    monkeypatch.setitem(score.__globals__, "load_study", method_origins)
    output = tmp_path / "score"
    score(study, output)
    frozen = read_json(output / "test/frozen_pool.json")["scored_ids"]
    comparison = read_json(output / "comparison/comparison.json")
    pairs = comparison["pairs"]
    assert len(frozen) == 5 and len(pairs) == 10
    assert {p[k] for p in pairs for k in ("left", "right")} == set(frozen)
    assert all("_c0@" in identifier for identifier in frozen)
    assert len(read_json(output / "summary.json")["attempts"]) == 5


@pytest.mark.parametrize("different_data", [False, True])
def test_multiple_prepared_sources_keep_common_identity_gate(synthetic_config, tmp_path, different_data):
    """Distinct prepared hashes may share one problem, never different observations."""
    from ml_platform_symbolic import select_representatives
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import digest, prepare
    from ml_platform_symbolic.engines.runner import run_discovery

    cfg = deepcopy(synthetic_config)
    table = pd.read_csv(cfg["data"]["local_path"])
    split = pd.read_csv(cfg["split"]["file"])
    split = split[split.partition != "final"]
    table = table[table.sample_id.isin(split.sample_id)]
    table.to_csv(cfg["data"]["local_path"], index=False)
    split.to_csv(cfg["split"]["file"], index=False)
    cfg["evaluation"]["mode"] = "exploratory"
    prepared_hashes, sources, prepared = [], [], []
    for method in ("mean", "linear"):
        local = deepcopy(cfg)
        if different_data and method == "linear":
            other = tmp_path / "different.csv"
            table.assign(x=table.x * 2).to_csv(other, index=False)
            local["data"]["local_path"] = str(other)
        local["search"].update(engines=[method], seeds=[17])
        refs = prepare(resolve_config(local), tmp_path / method / "prepare")
        prepared.append(refs)
        prepared_hashes.append(digest(refs["run"]))
        discovery = run_discovery(refs, read_json(refs["run"]), method, 17, tmp_path / method / "discover")
        sources.append({"discovery": str(discovery["summary"]), "prepared_run": str(refs["run"])})
    assert prepared_hashes[0] != prepared_hashes[1]
    request = {"prepared_run": prepared[0]["run"], "sources": sources,
               "purpose": "local_prediction", "final_history": "known_exposed"}
    output = tmp_path / "study"
    if different_data:
        with pytest.raises(ValueError, match="Discovery data/split/units/domain/fixed information differs"):
            create_study(request, run_dir=output)
        assert not (output / "study.json").exists()
    else:
        study = create_study(request, run_dir=output)
        context, records, rows, _ = load_study(study)
        assert set(select_representatives(records, rows)) == {("mean", 17), ("linear", 17)}
        assert "final" not in context["prepared"]
        assert {record["origin"]["prepared_run"] for record in records} == {str(refs["run"]) for refs in prepared}
    assert not list(tmp_path.rglob("final_access.json"))
    assert not list(tmp_path.rglob("final.npz"))
