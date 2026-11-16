from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines.pysr_template import export_slot_tree, template_specification
from ml_platform_symbolic.expressions import evaluate


def literal(value):
    return SimpleNamespace(degree=0, constant=True, val=value)


def feature(index):
    return SimpleNamespace(degree=0, constant=False, feature=index)


def operator(index, *children):
    return SimpleNamespace(degree=len(children), op=index, l=children[0], r=children[-1])


def test_template_native_leaf_values_and_structural_integer_power():
    tree = operator(1, literal(1.2345678901234567), operator(1, feature(1)))
    equation = export_slot_tree(tree, [{"name": "u"}], {1: ["square"], 2: ["*"]})
    X = np.array([[-2.0], [0.0], [1.37]])
    np.testing.assert_allclose(evaluate(equation, X, [{"id": "u"}]), 1.2345678901234567 * X[:, 0] ** 2)
    assert len(equation["parameters"]) == 1
    assert equation["parameters"]["c0"]["value"] == 1.2345678901234567
    with pytest.raises(ValueError, match="formal arity"):
        export_slot_tree(feature(2), [{"name": "u"}], {1: [], 2: []})


def specification():
    return {
        "ast": {"op": "add", "args": [{"slot": "f", "args": [{"var": "x0"}]}, {"param": "b"}]},
        "slots": {"f": {"arguments": [{"name": "u", "unit": "1"}], "allowed_operators": ["add"]}},
        "parameters": {
            "a": {"value": 2.0, "trainable": True, "bounds": [None, None]},
            "b": {"derived_from": {"op": "mul", "args": [{"param": "a"}, {"const": 3.0}]}},
        },
    }


def test_template_renderer_expands_dependencies():
    template = specification()
    spec, mapping = template_specification(template, [{"id": "x0"}], ["add"], parameter_initialization="native")
    assert spec["parameters"] == {"parameter0": 1}
    assert "parameter0[1]" in spec["combine"]
    assert mapping["slots"] == {"f": "slot0"}


def test_native_slot_units_are_not_inferred_for_dimensioned_coefficients():
    unitless = export_slot_tree(literal(2.0), [{"name": "u", "unit": "1"}], {1: [], 2: []}, output_unit="1")
    assert unitless["parameters"]["c0"]["unit"] == "1"
    assert unitless["parameters"]["c0"]["unit_origin"] == "declared_dimensionless_template_slot_convention"
    dimensional = export_slot_tree(literal(2.0), [{"name": "u", "unit": "m"}], {1: [], 2: []}, output_unit="m")
    assert dimensional["parameters"]["c0"]["unit"] is None


def test_named_component_renders_its_ast():
    template = specification()
    template["components"] = {"term": {"ast": template["ast"], "description": "shared form"}}
    template["ast"] = {"ref": "term"}
    spec, _ = template_specification(template, [{"id": "x0"}], ["add"], parameter_initialization="native")
    assert "slot0(input0)" in spec["combine"]


def test_unverified_native_version_rejects_before_model_construction(monkeypatch, tmp_path):
    import sys

    from ml_platform_symbolic.engines import pysr as adapter

    constructed = []
    monkeypatch.setitem(sys.modules, "pysr", SimpleNamespace(PySRRegressor=lambda **kwargs: constructed.append(kwargs)))
    monkeypatch.setattr(adapter, "version", lambda name: "2.8.0")
    request = SimpleNamespace(
        options={"template_parameter_initialization": "native"},
        operators=["add"],
        limits={"max_wall_seconds": 100},
        equation_template=specification(),
        variables=[{"id": "x0"}],
    )
    for slot in request.equation_template["slots"].values():
        slot["max_nodes"] = 8  # Normalized requests declare this search limit.
    with pytest.raises(NotImplementedError, match="2.7.2"):
        adapter.discover(request, SimpleNamespace(work_dir=tmp_path))
    assert not constructed


@pytest.mark.parametrize("restriction", ["bounds", "initialization", "slot_operators", "complete_form"])
def test_unsupported_template_rejects_before_dependency_access(monkeypatch, tmp_path, restriction):
    import builtins

    from ml_platform_symbolic.engines import pysr as adapter

    request = unsupported_request(restriction)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"pysr", "juliacall"}:
            pytest.fail("Unsupported request initialized an optional native runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(adapter, "version", lambda name: pytest.fail("Preflight accessed native dependency metadata"))
    with pytest.raises(NotImplementedError):
        adapter.discover(request, SimpleNamespace(work_dir=tmp_path))


def unsupported_request(restriction):
    template = specification()
    options = {"template_parameter_initialization": "native"}
    if restriction == "bounds":
        template["parameters"]["a"]["bounds"] = [0, None]
    elif restriction == "initialization":
        options = {}
    elif restriction == "slot_operators":
        template["slots"]["f"]["allowed_operators"] = ["mul"]
    elif restriction == "complete_form":
        template["slots"] = {}
    return SimpleNamespace(
        options=options,
        operators=["add"],
        limits={"max_wall_seconds": 30, "max_nodes": 20},
        equation_template=template,
        variables=[{"id": "x0"}],
    )


@pytest.mark.parametrize(
    "engine,restriction",
    [
        ("pysr", "bounds"),
        ("pysr", "initialization"),
        ("pysr", "slot_operators"),
        ("pysr", "complete_form"),
        ("known_form", "slots"),
    ],
)
def test_unsupported_template_subprocess_records_not_started(tmp_path, engine, restriction):
    import os
    import subprocess
    import sys

    from ml_platform_symbolic.data import read_json, write_json

    request = unsupported_request(restriction)
    # A native import would fail even if the optional dependency were installed.
    (tmp_path / "pysr.py").write_text("raise RuntimeError('native import reached')\n")
    train = tmp_path / "train.npz"
    np.savez(train, X=np.ones((2, 1)), y=np.ones(2), row_ids=np.array(["a", "b"]))
    job = write_json(
        tmp_path / "job.json",
        {
            "train": str(train),
            "variables": request.variables,
            "target": {"id": "y"},
            "operators": request.operators,
            "limits": {**request.limits, "equation_template": request.equation_template},
            "seed": 7,
            "options": request.options if engine == "pysr" else {},
            "engine": engine,
        },
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    result = subprocess.run(
        [sys.executable, "-m", "ml_platform_symbolic.engines.worker", str(job)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = read_json(tmp_path / "engine_summary.json")
    assert summary["status"] == "unsupported"
    assert summary["native_execution_status"] == "not_started"
    assert "native import reached" not in summary["reason"]
    assert not (tmp_path / "candidates.jsonl").exists()


def test_template_capability_distinguishes_search_size_gate_from_unit_check():
    from ml_platform_symbolic.engines.pysr import capabilities

    description = capabilities()["constraints"]["equation_template"]
    assert "source AST" in description and "local slot size" in description
    assert "search fitness before prediction" in description
    assert "units" in description and "after search" in description
    assert "slot size post-filtered" not in description


def test_template_capability_is_used_in_common_input_application():
    from ml_platform_symbolic.engines.pysr import capabilities
    from ml_platform_symbolic.response_search import constraint_application

    capability = capabilities()
    applied = constraint_application({"engine": "pysr", "limits": {"equation_template": specification()}}, capability)
    assert applied["equation_template"] == capability["constraints"]["equation_template"]
