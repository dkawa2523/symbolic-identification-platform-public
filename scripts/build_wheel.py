"""Build from a disposable source copy because the project mount is read-only."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, default=Path("build/package"))
root = parser.parse_args().directory
if not root.resolve().is_relative_to((Path(__file__).resolve().parents[1] / "build").resolve()):
    raise ValueError("Build directory must remain in this project's build directory")
root.mkdir(parents=True, exist_ok=False)
source = root / "source"
shutil.copytree("pkgs/symbolic", source, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist"))
subprocess.run(
    [sys.executable, "-m", "build", "--wheel", "--no-isolation", str(source), "--outdir", str(root / "dist")],
    check=True,
)
