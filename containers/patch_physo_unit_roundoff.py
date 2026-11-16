"""Narrow compatibility repair for PhySO1.1.11 unit-vector cancellation.

The native prior permits floating unit arithmetic, but bottom-up assertions use
bitwise equality. Preserve all assertions and constraining flags, comparing only
their three unit vectors with atol1e-12, rtol0 (stricter than the native prior).
"""

import argparse
import hashlib
import json
from pathlib import Path

UPSTREAM_SHA256 = "01a8836ac52da61a5a365b6f336383952b5de9005eb7102f801962ba959b3f0c"
TOLERANCE = 1e-12


def apply(package_dir):
    target = Path(package_dir) / "physym/dimensional_analysis.py"
    marker = target.with_name("symbolic_unit_roundoff_patch.json")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if digest == metadata["patched_sha256"]:
            print(json.dumps(metadata))
            return
    if digest != UPSTREAM_SHA256:
        raise ValueError("Unexpected PhySO unit-analysis source; inspect instead of applying a guessed patch")
    source = target.read_text()
    pairs = [
        ("child_phy_units", "np.zeros(Tok.UNITS_VECTOR_SIZE)"),
        ("phy_units", "np.zeros(Tok.UNITS_VECTOR_SIZE)"),
        ("child1_phy_units", "child0_phy_units"),
    ]
    for left, right in pairs:
        old = f"np.array_equal({left}, {right})"
        if source.count(old) != 1:
            raise ValueError(f"Expected one pinned unit assertion: {old}")
        source = source.replace(old, f"np.allclose({left}, {right}, rtol=0., atol={TOLERANCE!r})")
    compile(source, str(target), "exec")
    target.write_text(source)
    metadata = {
        "upstream": "PhySO1.1.11",
        "original_sha256": UPSTREAM_SHA256,
        "patched_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "absolute_unit_exponent_tolerance": TOLERANCE,
        "relative_tolerance": 0,
        "scope": "Three bottom-up unit-vector assertions; no unit labels, data, search settings, rewards, fitting or native prior changed",
    }
    marker.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    apply(parser.parse_args().package_dir)
