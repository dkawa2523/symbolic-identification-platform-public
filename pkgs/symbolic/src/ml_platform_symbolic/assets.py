"""Validate pre-positioned native assets; no SDK, model download or shared cache."""

import re
from pathlib import Path, PurePosixPath

from .config import load_config
from .data import digest


def verify_assets(manifest_file, engine):
    manifest_path = Path(manifest_file).resolve()
    manifest = load_config(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Asset manifest must be an object")
    if manifest.get("engine") != engine or not isinstance(manifest.get("source"), str) or not manifest["source"].strip():
        raise ValueError("Asset manifest engine/source mismatch")
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or revision.strip() in {"main", "master", "latest", ""}:
        raise ValueError("Asset source_revision must identify a fixed upstream version")
    location = manifest.get("directory")
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Asset manifest directory must be a nonempty local path")
    directory = (manifest_path.parent / location).resolve()
    entries = manifest.get("files")
    if not directory.is_dir() or not isinstance(entries, list) or not entries:
        raise ValueError("Asset directory and nonempty file manifest are required")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each asset file entry must be an object with path, size_bytes and sha256")
        name = entry.get("path")
        if not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Asset file path must be a nonempty relative POSIX path without traversal")
        file = (directory / name).resolve()
        size, checksum = entry.get("size_bytes"), entry.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Asset {name} requires a nonnegative integer size_bytes and lowercase SHA256")
        identity = str(PurePosixPath(name))
        if identity in seen or not file.is_relative_to(directory) or not file.is_file():
            raise ValueError(f"Invalid/missing/duplicate asset path: {name}")
        seen.add(identity)
        if file.stat().st_size != size or digest(file) != checksum:
            raise ValueError(f"Asset checksum/size mismatch: {name}")
    return directory, {**manifest, "manifest_sha256": digest(manifest_path)}
