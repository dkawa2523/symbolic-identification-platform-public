from copy import deepcopy

import pytest
from ml_platform_symbolic.assets import verify_assets
from ml_platform_symbolic.data import digest, write_json


@pytest.mark.parametrize("case", ["not_object", "files_object", "entry_scalar", "path_missing", "size_bool", "bad_hash", "duplicate", "traversal", "valid"])
def test_asset_manifest_rejects_ambiguous_shape_and_identity(tmp_path, case):
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"fixture data, never deserialized")
    manifest = {"engine": "e2e", "source": "local fixture", "source_revision": "fixed-v1", "directory": ".",
                "files": [{"path": asset.name, "size_bytes": asset.stat().st_size, "sha256": digest(asset)}]}
    if case == "not_object":
        manifest = []
    elif case == "files_object":
        manifest["files"] = {"weights.bin": "ignored"}
    elif case == "entry_scalar":
        manifest["files"] = ["weights.bin"]
    elif case == "path_missing":
        manifest["files"][0].pop("path")
    elif case == "size_bool":
        manifest["files"][0]["size_bytes"] = True
    elif case == "bad_hash":
        manifest["files"][0]["sha256"] = "not-a-hash"
    elif case == "duplicate":
        manifest["files"].append(deepcopy(manifest["files"][0]))
        manifest["files"][1]["path"] = "./weights.bin"
    elif case == "traversal":
        manifest["files"][0]["path"] = "../weights.bin"
    path = write_json(tmp_path / "assets.json", manifest)
    if case == "valid":
        directory, verified = verify_assets(path, "e2e")
        assert directory == tmp_path and verified["manifest_sha256"] == digest(path)
    else:
        with pytest.raises(ValueError):
            verify_assets(path, "e2e")


def test_verified_asset_snapshot_symlink_and_blob_may_share_storage(tmp_path):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"shared verified checkpoint bytes")
    snapshot = tmp_path / "snapshot.bin"
    snapshot.symlink_to(blob)
    manifest = {"engine": "rasr", "source": "local HF cache fixture", "source_revision": "fixed-v1",
                "directory": ".", "files": [
                    {"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in (blob, snapshot)]}
    path = write_json(tmp_path / "assets.json", manifest)
    directory, verified = verify_assets(path, "rasr")
    assert directory == tmp_path and len(verified["files"]) == 2


def test_asset_symlink_must_still_resolve_inside_declared_directory(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not in assets root")
    directory = tmp_path / "assets"
    directory.mkdir()
    link = directory / "link.bin"
    link.symlink_to(outside)
    manifest = {"engine": "rasr", "source": "local fixture", "source_revision": "fixed-v1", "directory": "assets",
                "files": [{"path": "link.bin", "size_bytes": outside.stat().st_size, "sha256": digest(outside)}]}
    path = write_json(tmp_path / "assets.json", manifest)
    with pytest.raises(ValueError, match="Invalid/missing/duplicate"):
        verify_assets(path, "rasr")
