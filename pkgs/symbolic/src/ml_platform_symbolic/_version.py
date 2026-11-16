"""Identify the executing package rather than stale metadata in a mounted runtime."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_identity():
    package = Path(__file__).resolve().parent
    try:
        installed = version("ml-platform-symbolic")
    except PackageNotFoundError:
        installed = None
    source = package.parent.parent / "pyproject.toml"
    if package.parent.name == "src" and source.is_file():
        project = tomllib.loads(source.read_text(encoding="utf-8"))["project"]
        if project.get("name") == "ml-platform-symbolic":
            return {"version": project["version"], "mode": "source", "installed_metadata_version": installed}
    if installed is None:
        raise PackageNotFoundError("Executing symbolic package has neither source metadata nor an installed distribution")
    return {"version": installed, "mode": "installed", "installed_metadata_version": installed}
