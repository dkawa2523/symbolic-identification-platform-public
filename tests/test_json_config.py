import json

from ml_platform_symbolic.config import load_config, resolve_config


def test_json_scientific_numbers_remain_numeric(synthetic_config, tmp_path):
    synthetic_config["evaluation"] = {"absolute_tolerance": 1e-6}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(synthetic_config), encoding="utf-8")
    resolved = resolve_config(load_config(path))
    assert resolved["evaluation"]["absolute_tolerance"] == 0.000001


def test_new_problem_has_one_requirement_authority_and_a_nonmutating_legacy_reader(synthetic_config, tmp_path):
    from copy import deepcopy

    import pytest
    from ml_platform_symbolic.data import prepare, read_json
    from ml_platform_symbolic.problem import problem_config

    incoming = deepcopy(synthetic_config)
    incoming["requirements"] = {"allowed_operators": ["add", "mul"], "require_units": False}
    cfg = resolve_config(incoming)
    refs = prepare(cfg, tmp_path / "prepared")
    problem = read_json(refs["run"])
    assert problem["requirements"]["allowed_operators"] == ["add", "mul"]
    assert "operators" not in problem["config"]["search"]
    assert "fixed_constants" not in problem["config"]["data"]
    assert "constraints" not in problem["config"]
    before = deepcopy(problem)
    assert problem_config(problem) == cfg
    assert problem_config({"config": cfg}) == cfg
    assert problem == before
    corrupt = deepcopy(problem)
    corrupt["requirements"]["allowed_operators"] = ["div"]
    with pytest.raises(ValueError, match="changed"):
        problem_config(corrupt)
    incoming["search"]["operators"] = ["add", "mul"]
    with pytest.raises(ValueError, match="twice"):
        resolve_config(incoming)
