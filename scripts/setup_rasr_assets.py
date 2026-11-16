"""Explicit pinned RASR checkpoint and complete native simplifier asset setup."""

import argparse
import json
import os
from pathlib import Path

from ml_platform_symbolic.data import digest, write_json

MODEL = "davidotte/RASR-94M-L"
REVISION = "13c53e64e0456d516f45212659f82c97f4bc8ec1"
SIMPLIFIER_REVISION = "b08c7595e3380ac386754263c441ff08f1dc246d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="pretrained/rasr", type=Path)
    root = parser.parse_args().directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(XDG_CACHE_HOME=str(root / "cache/xdg"), HF_HOME=str(root / "cache/hf"), WANDB_MODE="disabled")
    from huggingface_hub import hf_hub_download, snapshot_download
    from simplipy.asset_manager import install_asset

    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=root / "model")
    manifest_path = Path(
        hf_hub_download(repo_id="psaegert/simplipy-assets", repo_type="dataset", filename="manifest.json")
    )
    native_manifest = json.loads(manifest_path.read_text())
    native_asset = native_manifest["dev_7-3"]
    if manifest_path.parent.name != SIMPLIFIER_REVISION:
        raise ValueError("Native simplifier manifest changed; review the new immutable source first")
    # This older asset has no revision/digests in its native manifest. Fetch its
    # exact published snapshot explicitly; common manifest binds observed bytes.
    snapshot_download(
        repo_id="psaegert/simplipy-assets",
        repo_type="dataset",
        revision=SIMPLIFIER_REVISION,
        allow_patterns=["engines/dev_7-3/config.yaml", "engines/dev_7-3/rules.json"],
        local_dir=root / "cache/xdg/simplipy",
    )
    if not install_asset("dev_7-3"):
        raise RuntimeError("Official RASR simplifier setup failed")
    files = [
        {"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "manifest.json" and not p.name.endswith(".lock")
    ]
    write_json(
        root / "manifest.json",
        {
            "engine": "rasr",
            "source": "https://huggingface.co/" + MODEL,
            "source_revision": REVISION,
            "directory": ".",
            "license": "MIT",
            "simplipy_manifest_revision": manifest_path.parent.name,
            "simplipy_asset": native_asset,
            "simplipy_asset_revision": SIMPLIFIER_REVISION,
            "files": files,
        },
    )
    print(f"Verified checkpoint/files={len(files)}; manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
