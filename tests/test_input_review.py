import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from ml_platform_symbolic.config import load_config, resolve_config
from ml_platform_symbolic.data import digest
from ml_platform_symbolic.input_review import engine_catalog, review_inputs


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("search", "engines", "linear", "search.engines"),
        ("search", "engines", ["pysr_typo"], "Unsupported engines"),
        ("search", "engine_options", {"dictionary": "max_degree=3"}, "engine_options"),
        ("search", "runtimes", {"dictionary": None}, "runtimes"),
        ("search", "seeds", "42", "seeds"),
        ("search", "max_wall_seconds", float("inf"), "max_wall_seconds"),
        ("search", "max_export_candidates", 2.5, "max_export_candidates"),
        ("search", "allow_partial_engines", "false", "allow_partial_engines"),
        ("data", "feature_columns", "x", "feature_columns"),
        ("data", "feature_columns", ["x", "y"], "exclude target"),
        ("data", "target_column", "sample_id", "identity"),
        ("constraints", "allowed_variables", "x", "allowed_variables"),
        ("constraints", "probes", [{"misspelled_x": 1}], "original input columns"),
        ("constraints", "probes", [{"x": None}], "finite"),
        ("constraints", "require_units", "false", "require_units"),
        ("evaluation", "absolute_tolerance", float("nan"), "absolute_tolerance"),
        ("split", "seed", -1, "split.seed"),
        ("split", "valid_size", "0.2", "valid_size"),
    ],
)
def test_invalid_input_is_rejected_before_preparation(synthetic_config, section, key, value, match):
    synthetic_config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match=match):
        resolve_config(synthetic_config)


@pytest.mark.parametrize(
    "text,match",
    [
        ('{"search":{"engines":["linear"],"engines":["pysr"]}}', "Duplicate JSON setting"),
        ('{"search":{"max_wall_seconds":NaN}}', "Nonfinite JSON"),
        ('{"search":{"max_wall_seconds":1e309}}', "Nonfinite JSON"),
    ],
)
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, text, match):
    path = tmp_path / "input.json"
    path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_review_reuses_units_and_partitions_without_mutating_or_running_models(synthetic_config, monkeypatch):
    from ml_platform_symbolic.engines import dictionary

    def forbidden(*args, **kwargs):
        raise AssertionError("Input review must never run discovery")

    monkeypatch.setattr(dictionary, "discover", forbidden)
    synthetic_config["data"]["units"] = {"x": "cm", "y": "mm"}
    before = deepcopy(synthetic_config)
    sources = [Path(synthetic_config["data"]["local_path"]), Path(synthetic_config["split"]["file"])]
    hashes = [digest(path) for path in sources]
    files = sorted(path.name for path in sources[0].parent.iterdir())
    review = review_inputs(synthetic_config)
    assert review["status"] == "inputs_valid"
    assert review["unit_coordinates"]["x"]["scale"] == 0.01
    assert review["unit_coordinates"]["y"]["problem_unit"] == "m"
    assert review["partitions"] == {
        "train": {"n_rows": 36, "n_groups": 12},
        "selection": {"n_rows": 12, "n_groups": 4},
        "final": {"n_rows": 12, "n_groups": 4},
    }
    assert synthetic_config == before
    assert [digest(path) for path in sources] == hashes
    assert sorted(path.name for path in sources[0].parent.iterdir()) == files
    assert not {"feature_ranges", "metrics", "predictions"} & review.keys()
    assert "not_checked" in review


def test_review_detects_group_leakage_using_real_preparation(synthetic_config):
    path = synthetic_config["split"]["file"]
    split = pd.read_csv(path)
    split.loc[0, "partition"] = "final"
    split.to_csv(path, index=False)
    with pytest.raises(ValueError, match="group crosses"):
        review_inputs(synthetic_config)


def test_declaration_blocker_is_not_misreported_as_readiness(synthetic_config):
    synthetic_config["search"] = {"engines": ["e2e"], "runtimes": {"e2e": {"device": "cuda"}}}
    review = review_inputs(synthetic_config)
    assert review["status"] == "blocked"
    reasons = [item["reason"] for item in review["engines"]["e2e"]["declared_input_issues"]]
    assert any("Unsupported device" in reason for reason in reasons)
    assert any("model_assets" in reason for reason in reasons)
    assert review["not_checked"]


def test_catalog_reads_current_registry_without_importing_native_libraries(monkeypatch):
    from ml_platform_symbolic.engines import registry

    monkeypatch.setitem(registry.ENGINES, "local_demo", ".dictionary")
    assert engine_catalog("local_demo")["local_demo"]["requires_assets"] is False
    assert "e2e" in engine_catalog()
    assert "symbolicregression" not in sys.modules


def test_validate_cli_is_readable_and_fails_closed(synthetic_config, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(synthetic_config), encoding="utf-8-sig")
    command = [sys.executable, "-m", "ml_platform_symbolic", "validate", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "x0=x" in result.stdout and "36 rows / 12 groups" in result.stdout
    assert "No search" in result.stdout and "Final metrics" in result.stdout
    synthetic_config["search"]["engines"] = ["e2e"]
    path.write_text(json.dumps(synthetic_config), encoding="utf-8")
    blocked = subprocess.run([*command, "--json"], capture_output=True, text=True, check=False)
    assert blocked.returncode == 2
    assert json.loads(blocked.stdout)["status"] == "blocked"
    path.write_text('{"data": []}', encoding="utf-8")
    malformed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert malformed.returncode == 2 and "data must be a mapping" in malformed.stderr
    assert "Traceback" not in malformed.stderr

@pytest.mark.parametrize("text", [
    "search:\n  engines: [pysr]\n  engines: [linear]\n",
    "search:\n  engine_options:\n    linear: {ridge: 0, ridge: 1}\n",
])
def test_duplicate_yaml_settings_are_rejected(tmp_path, text):
    path = tmp_path / "input.yaml"
    path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Duplicate YAML setting"):
        load_config(path)


@pytest.mark.parametrize("body", [
    "original001,id001,1,2\noriginal002,id002,3,4\n",
    "id001,1,2\nid002,3,4,extra\n",
])
def test_csv_extra_fields_cannot_shift_columns_or_disappear(tmp_path, body):
    from ml_platform_symbolic.data import read_observations

    path = tmp_path / "shifted.csv"
    path.write_text("sample_id,x,y\n" + body, encoding="utf-8")
    with pytest.raises(ValueError, match="header width"):
        read_observations(path)


def test_adapter_review_gets_only_prepared_train_and_exact_worker_limits(synthetic_config, monkeypatch):
    import numpy as np
    from ml_platform_symbolic.engines import dictionary

    seen = []
    def check(request, context):
        seen.append(request)
        assert len(request.X) == 36 and len(request.y) == 36
        assert request.limits["max_wall_seconds"] == synthetic_config["search"]["max_wall_seconds"]
        assert context.work_dir.is_dir()
    monkeypatch.setattr(dictionary, "validate_request", check)
    synthetic_config["data"]["units"] = {"x": "cm", "y": "mm"}
    review = review_inputs(synthetic_config)
    assert review["engines"]["dictionary"]["adapter_validation"] == "passed"
    before = seen[0].y.copy()
    frame = pd.read_csv(synthetic_config["data"]["local_path"])
    split = pd.read_csv(synthetic_config["split"]["file"])
    frame.loc[frame.sample_id.isin(split.loc[split.partition == "final", "sample_id"]), "y"] += 1000
    frame.to_csv(synthetic_config["data"]["local_path"], index=False)
    assert review_inputs(synthetic_config)["status"] == "inputs_valid"
    np.testing.assert_array_equal(before, seen[1].y)


def test_review_blocks_native_options_before_import_or_discovery(synthetic_config):
    synthetic_config["search"]["engine_options"] = {"dictionary": {"max_degre": 3}}
    review = review_inputs(synthetic_config)
    assert review["status"] == "blocked"
    assert review["engines"]["dictionary"]["adapter_validation"] == "blocked"
    assert "Unsupported native options" in review["engines"]["dictionary"]["adapter_input_issues"][0]["reason"]


def test_review_preserves_extension_without_optional_validation(synthetic_config, monkeypatch):
    from ml_platform_symbolic.engines import dictionary

    monkeypatch.delattr(dictionary, "validate_request")
    review = review_inputs(synthetic_config)
    assert review["status"] == "inputs_valid"
    assert review["engines"]["dictionary"]["adapter_validation"] == "not_available"


def test_review_rejects_asset_hash_mismatch_without_loading_checkpoint(synthetic_config, tmp_path):
    from ml_platform_symbolic.data import write_json

    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"not an executable checkpoint")
    manifest = write_json(tmp_path / "assets.json", {
        "engine": "e2e", "source": "local regression fixture", "source_revision": "fixed-test-v1",
        "directory": ".", "files": [{"path": asset.name, "size_bytes": asset.stat().st_size, "sha256": "0" * 64}],
    })
    synthetic_config["search"] = {"engines": ["e2e"]}
    synthetic_config["model_assets"] = {"e2e": {"manifest_file": str(manifest)}}
    review = review_inputs(synthetic_config)
    assert review["status"] == "blocked"
    assert review["engines"]["e2e"]["asset_verification"]["status"] == "blocked"
    assert "checksum/size mismatch" in review["engines"]["e2e"]["asset_input_issues"][0]["reason"]

@pytest.mark.parametrize("text,match", [
    ('{"engineering_tolerance":0.001,"engineering_tolerance":1000}', "Duplicate JSON"),
    ('{"threshold":1e309}', "Nonfinite JSON"),
    ('{"threshold":NaN}', "Nonfinite JSON"),
])
def test_public_plan_option_and_artifact_json_share_strict_parser(tmp_path, text, match):
    from ml_platform_symbolic.data import read_json
    from ml_platform_symbolic.equation_evaluation import create_study

    path = tmp_path / "plan.json"
    path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=match):
        read_json(path)
    with pytest.raises(ValueError, match=match):
        create_study(path, run_dir=tmp_path / "study")


def test_running_source_version_is_not_stale_installed_metadata(synthetic_config, monkeypatch):
    import tomllib

    from ml_platform_symbolic import _version

    source = Path(_version.__file__).parents[2] / "pyproject.toml"
    if not source.is_file():
        identity = _version.package_identity()
        assert identity["mode"] == "installed" and identity["version"] == identity["installed_metadata_version"]
        return
    expected = tomllib.loads(source.read_text(encoding="utf-8"))["project"]["version"]
    monkeypatch.setattr(_version, "version", lambda _: "0.0.0-stale-test")
    identity = _version.package_identity()
    assert identity == {"version": expected, "mode": "source", "installed_metadata_version": "0.0.0-stale-test"}
    assert review_inputs(synthetic_config)["package_identity"] == identity


def test_fixed_split_uses_the_same_strict_csv_boundary(synthetic_config):
    path = Path(synthetic_config["split"]["file"])
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n" + "\n".join("discarded," + line for line in lines[1:]) + "\n")
    with pytest.raises(ValueError, match="header width"):
        review_inputs(synthetic_config)
