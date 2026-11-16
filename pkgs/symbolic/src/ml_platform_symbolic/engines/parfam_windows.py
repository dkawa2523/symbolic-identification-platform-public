"""Pinned ParFam portability fix: use Windows peak working set for memory display only."""

import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_LF_SHA256 = "97725b53d899fbc3f8eb90f5a8ed94e04e24f2bde12119f237fcf68dd914e4a3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    args = parser.parse_args()
    file = args.package_dir / "utils.py"
    source = file.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_LF_SHA256:
        raise ValueError("Requires the reviewed unmodified ParFam e10379a source; not overwriting")
    original = source
    source = source.replace(
        "import resource\n",
        "try:\n    import resource\nexcept ModuleNotFoundError:\n    resource = None  # Windows has no POSIX resource module.\n",
        1,
    )
    source = source.replace(
        "usage=resource.getrusage(resource.RUSAGE_SELF)",
        "usage = (resource.getrusage(resource.RUSAGE_SELF) if resource is not None\n             else (None, None, psutil.Process().memory_info().peak_wset / 1024.0))",
        1,
    )
    if source == original or "usage=resource.getrusage" in source:
        raise ValueError("Unexpected source context")
    file.with_suffix(".py.original").write_bytes(file.read_bytes())
    file.write_text(source, encoding="utf-8", newline="\n")
    info = {
        "source_revision": "e10379a0e19238b50178492c48da0d0444d4898f",
        "original_lf_sha256": ORIGINAL_LF_SHA256,
        "patched_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "scope": "POSIX-only memory-display import/function made portable; no optimizer/prediction/resource-limit changes",
    }
    (args.package_dir / "symbolic_windows_patch.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
