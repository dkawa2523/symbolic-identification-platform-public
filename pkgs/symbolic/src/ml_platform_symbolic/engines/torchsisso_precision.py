"""Observe unrounded native TorchSISSO 0.1.8 results without changing its search.

Apply only to the explicitly supplied isolated package directory. This is a small
instrumentation patch to the reviewed official code, not a replacement algorithm.
"""

import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "a14560c8cea5d001914bf13986ba6f629a68ccb36f974db2f3d89d49992f7a63"


def instrument(source):
    replacements = [
        ("class Regressor:", "class Regressor:\n    symbolic_handoff_precision_version = 1"),
        ("        final_equations = []", "        final_equations = []\n        self.portable_models_ = []"),
        (
            "        terms = []",
            "        self.portable_models_.append({\n"
            "            'indices': indices_min.reshape(-1).detach().cpu().tolist(),\n"
            "            'coefficients': non_std_coeff.reshape(-1).detach().cpu().tolist(),\n"
            "            'intercept': float(non_std_intercept),\n"
            "        })\n\n        terms = []",
        ),
        (
            "                if intercept > 0:",
            "                self.portable_models_.append({\n"
            "                    'indices': [int(selected_index)],\n"
            "                    'coefficients': [(coef[1]/self.x_std[int(selected_index)]).item()],\n"
            "                    'intercept': float(intercept),\n"
            "                })\n\n                if intercept > 0:",
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Native source differs from reviewed instrumentation context")
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    args = parser.parse_args()
    path = args.package_dir / "Regressor.py"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Requires unmodified official TorchSisso 0.1.8 Regressor.py; not overwriting")
    changed = instrument(data.decode("utf-8")).encode("utf-8")
    path.with_suffix(".py.original").write_bytes(data)
    path.write_bytes(changed)
    report = {
        "upstream": "PaulsonLab/TorchSISSO",
        "version": "0.1.8",
        "scope": "retain native selected indices and unrounded original-scale coefficients/intercept",
        "original_sha256": ORIGINAL_SHA256,
        "instrumented_sha256": hashlib.sha256(changed).hexdigest(),
    }
    (args.package_dir / "symbolic_handoff_precision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
