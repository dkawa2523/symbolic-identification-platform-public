"""A supplied physical family is an explicit baseline, not symbolic discovery."""

import json
from copy import deepcopy
from pathlib import Path

from current_flow import explicit_choice, study_flow
from ml_platform_symbolic import finalize_hypothesis, load_model
from ml_platform_symbolic.equation_evaluation import load_study


def test_supplied_dimensional_family_fits_only_free_parameters(synthetic_config, tmp_path):
    reference = json.loads((Path(__file__).parents[1] / "benchmarks/settings/physo_reference.json").read_text())
    options = deepcopy(reference["search"]["engine_options"]["known_form"])
    options["equation"]["parameters"]["scale"]["bounds"] = [0, 1]
    cfg = deepcopy(synthetic_config)
    cfg["data"].update(units={"x": "s", "y": "m"}, fixed_constants={"g": {"value": 9.80665, "unit": "m/s2"}})
    cfg["search"].update(
        engines=["known_form"], operators=["add", "mul", "pow_int"], engine_options={"known_form": options}
    )
    cfg["constraints"]["require_units"] = True
    _, _, study = study_flow(cfg, tmp_path / "known-family", purpose="local_prediction", engineering_tolerance=1e-7)
    _, decision = explicit_choice(study, tmp_path / "known-family")
    result = finalize_hypothesis(study, decision, run_dir=tmp_path / "known-family/final-audit")
    _, records, rows, _ = load_study(study)
    selected = next(i for i, record in enumerate(records) if record["id"] == result.extra["selected_id"])
    assert result.metrics["final_rmse"] < 1e-7
    model = load_model(result.artifacts["model_bundle"])
    assert model.equation["parameters"]["g"] == {
        "value": 9.80665,
        "unit": "m/s2",
        "trainable": False,
        "bounds": [9.80665, 9.80665],
        "origin": "declared_fixed_constant",
    }
    assert 0 <= model.equation["parameters"]["scale"]["value"] <= 1
    assert model.metadata["origin"]["summary"]["method_kind"] == "user_supplied_family_reference"
    assert all(check["status"] == "pass" for check in rows[selected]["checks"] if check["required"])
