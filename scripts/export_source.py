"""Export current source into a new build directory, without Git history or local assets."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

TOP = (
    ".gitignore", ".gitattributes", ".dockerignore", "AGENTS.md", "pyproject.toml", "README.md",
    "pkgs/symbolic/pyproject.toml", "pkgs/symbolic/README.md",
    "pkgs/spdml_layer/src/ml_platform_spdml/symbolic.py", "tests/README.md", "data/README.md",
    "docs/figures/workflow.svg", "docs/figures/architecture.svg", "docs/figures/constraints.svg",
    "data/nist/Misra1a.dat", "data/nist/Chwirut2.dat", "data/nist/LICENSE-NIST-PD.txt",
)
# Explicit source roots only. Never walk results, build, runtime, weights, or private data.
ROOTS = ("docs", "examples", "benchmarks", "scripts", "pkgs/symbolic/src", "containers", "tests")
SUFFIXES = {".py", ".jl", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".html", ".cjs", ".sh", ".ps1"}


def source_paths(project):
    paths = {project / name for name in TOP}
    for directory in ROOTS:
        paths.update(
            p for p in (project / directory).rglob("*")
            if p.is_file() and not p.is_symlink()
            and not {"__pycache__", "build", "dist"} & set(p.relative_to(project).parts)
            and not any(part.endswith((".egg-info", ".dist-info")) for part in p.relative_to(project).parts)
            and (p.suffix in SUFFIXES or p.name.startswith("Dockerfile"))
        )
    # Pinned public fixtures support data-dependent regressions; no other data roots travel.
    paths.update(
        p for p in (project / "data/partial_equations").rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix in {".json", ".csv", ".names", ".zip"}
    )
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to((project / "build").resolve()):
        raise ValueError("Export must remain inside this project's build directory")
    paths = source_paths(project)
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
            raise ValueError(f"Invalid source entry: {path.relative_to(project)}")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in paths:
        rel = path.relative_to(project)
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        hashes[rel.as_posix()] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (output / "SOURCE_MANIFEST.json").write_text(
        json.dumps({
            "scope": "Current internal source; no Git history, commit, publication or remote created",
            "rights": "Internal use/copy/handoff authority requires confirmation before distribution",
            "files": hashes,
            "excluded": ["results", "build outputs", "runtime", "private data", "weights/indexes", "wheelhouse", "venv/cache", "secrets"],
        }, indent=2), encoding="utf-8",
    )
    print(f"Exported {len(hashes)} source files to {output}")


if __name__ == "__main__":
    main()
