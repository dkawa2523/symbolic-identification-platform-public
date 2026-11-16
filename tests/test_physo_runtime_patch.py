"""Exact installed native sources are read-only; all patch writes use tmp_path."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.native
CONTAINERS = Path(__file__).resolve().parents[1] / "containers"
sys.path.insert(0, str(CONTAINERS))
try:
    import patch_physo_runtime as runtime
    import patch_physo_unit_roundoff as unit
finally:
    sys.path.remove(str(CONTAINERS))


def installed_bytes():
    spec = importlib.util.find_spec("physo")
    if spec is None:
        pytest.skip("Pinned PhySO is not installed")
    assert spec.origin, "PhySO package has no concrete source origin"
    root = Path(spec.origin).parent
    data = {name: (root / name).read_bytes() for name in runtime.BEFORE}
    hashes = {name: runtime.digest(value) for name, value in data.items()}
    assert hashes in (runtime.RAW, runtime.BEFORE, runtime.AFTER), (
        "Unknown or mixed installed native source"
    )
    if hashes == runtime.AFTER:
        for name, replacements in runtime.REPLACEMENTS.items():
            text = data[name].decode("utf-8")
            for before, after in reversed(replacements):
                assert text.count(after) == 1
                text = text.replace(after, before, 1)
            data[name] = text.encode("utf-8")
        assert {
            name: runtime.digest(value) for name, value in data.items()
        } == runtime.BEFORE
    return data


def setup(tmp_path, raw=False):
    root = tmp_path / "physo"
    for name, value in installed_bytes().items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    dim = root / runtime.DIM
    text = dim.read_text()
    for left, right in [
        ("child_phy_units", "np.zeros(Tok.UNITS_VECTOR_SIZE)"),
        ("phy_units", "np.zeros(Tok.UNITS_VECTOR_SIZE)"),
        ("child1_phy_units", "child0_phy_units"),
    ]:
        text = text.replace(
            f"np.allclose({left}, {right}, rtol=0., atol=1e-12)",
            f"np.array_equal({left}, {right})",
        )
    dim.write_text(text)
    assert runtime.digest(dim.read_bytes()) == unit.UPSTREAM_SHA256
    if not raw:
        unit.apply(root)
    return root


@pytest.mark.parametrize("raw", [True, False])
def test_exact_outputs_chain_idempotence(tmp_path, raw):
    root = setup(tmp_path, raw)
    untouched = root / "learn"
    untouched.mkdir()
    (untouched / "learn.py").write_text("untouched sampler")
    record = runtime.apply(root)
    for n, h in runtime.AFTER.items():
        assert runtime.digest((root / n).read_bytes()) == h
    marker = root / "physym/symbolic_unit_roundoff_patch.json"
    assert record["unit_stage"]["patched_sha256"] == runtime.BEFORE[runtime.DIM]
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert runtime.apply(root) == record
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert (untouched / "learn.py").read_text() == "untouched sampler"
    with pytest.raises(ValueError):
        unit.apply(root)  # historical guard not relaxed
    assert json.loads(marker.read_text()) == record["unit_stage"]


@pytest.mark.parametrize(
    "kind",
    ["mixed", "unknown", "missing_unit", "bad_unit", "stale_runtime", "raw_stale_unit"],
)
def test_reject_without_changes(tmp_path, kind):
    root = setup(tmp_path, raw=kind == "raw_stale_unit")
    marker = root / "physym/symbolic_unit_roundoff_patch.json"
    if kind == "mixed":
        path = root / "task/fit.py"
        text = path.read_text()
        for before, after in runtime.REPLACEMENTS["task/fit.py"]:
            assert text.count(before) == 1
            text = text.replace(before, after, 1)
        path.write_text(text)
        assert runtime.digest(path.read_bytes()) == runtime.AFTER["task/fit.py"]
    if kind == "unknown":
        (root / "physym/prior.py").write_text("unknown")
    if kind == "missing_unit":
        marker.unlink()
    if kind in ("bad_unit", "raw_stale_unit"):
        marker.write_text("{}")
    if kind == "stale_runtime":
        (root / "symbolic_runtime_patch.json").write_text("{}")
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.apply(root)
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("tamper", ["missing", "runtime", "unit"])
def test_final_marker_required(tmp_path, tamper):
    root = setup(tmp_path)
    runtime.apply(root)
    if tamper == "missing":
        (root / "symbolic_runtime_patch.json").unlink()
    elif tamper == "runtime":
        (root / "symbolic_runtime_patch.json").write_text("{}")
    else:
        (root / "physym/symbolic_unit_roundoff_patch.json").write_text("{}")
    with pytest.raises(ValueError):
        runtime.apply(root)
