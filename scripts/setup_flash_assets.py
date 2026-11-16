"""Explicit setup of the pinned public Flash model and complete SimpliPy offline assets."""

import argparse
import os
from pathlib import Path

from ml_platform_symbolic.data import digest, write_json

MODEL = "psaegert/flash-ansr-v25.0-T8-3M"
REVISION = "d1324395d53191833e4b5927d3e5322aea902ade"
MANIFEST_REVISION = "b08c7595e3380ac386754263c441ff08f1dc246d"
EXPECTED = {
    "model/model.yaml": (943, "0ff5eb2b29bff5615f806c325fa1a418b11a10dbd332c0da52fa27ff84b240ad"),
    "model/tokenizer.yaml": (2900, "1696af4c85fab980fb3f910d746141fd2ee86dcc3f1817141650a39791c7ef65"),
    "model/model.safetensors": (14051872, "6e5a6c1c927f17bcd564701ad925e0568e9fbcdc30127fa8c8bd220f73185d2e"),
    "config.yaml": (3006, "590a21dc22c13bded11927e854459f0112d1c4af0787f4374c82eec580743b03"),
    "rules.json": (8284816, "7a0ee807e1f09b93285dacfe81113542fdfa06d972e684f078fc99d73d0853f8"),
    "rules_real.json": (23952428, "c31e378dbc91754f5627c5380da2e6c812ca6801a6c0628f043d3037333d544e"),
    "rules_corpus.json": (24961989, "ad48b0c3aaccc8773daa07ab6c076fdd3493f824aebd9cc3a301408fbde33e10"),
    "rules.json.provenance.json": (2758697, "b5faa6b21c51b23265ccba90faa15287ad8d64a43da89fd9ea8ee084b23143e2"),
    "mine.yaml": (3398, "91cf7bdeadba52a9be3097f4ee8e622e1f66de05237aa07b191da346f4eed452"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="pretrained/flash", type=Path)
    root = parser.parse_args().directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(XDG_CACHE_HOME=str(root / "cache/xdg"), HF_HOME=str(root / "cache/hf"), WANDB_MODE="disabled")
    from huggingface_hub import hf_hub_download, snapshot_download
    from simplipy.asset_manager import install_asset

    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=root / "model")
    # Native resolver uses main. Keep the supported cached ref, but require its known fixed commit.
    manifest = Path(hf_hub_download(repo_id="psaegert/simplipy-assets", repo_type="dataset", filename="manifest.json"))
    if manifest.parent.name != MANIFEST_REVISION:
        raise ValueError("Upstream SimpliPy manifest changed; review new asset identity before using it")
    if not install_asset("acj-5-4-llm"):
        raise RuntimeError("Official SimpliPy asset installation failed")
    for name, (size, checksum) in EXPECTED.items():
        path = root / name if name.startswith("model/") else root / "cache/xdg/simplipy/engines/acj-5-4-llm" / name
        if path.stat().st_size != size or digest(path) != checksum:
            raise ValueError(f"Pinned Flash/SimpliPy asset changed: {name}")
    # Include the cached HF manifest too; only exclude this root's output manifest on repeated setup.
    files = [
        {"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "manifest.json" and not p.name.endswith(".lock")
    ]
    write_json(
        root / "manifest.json",
        {
            "engine": "flash_ansr",
            "source": "https://huggingface.co/" + MODEL,
            "source_revision": REVISION,
            "directory": ".",
            "license": "MIT",
            "simplipy_revision": "9f79374ff6905188161fdcbf38bf0dffbcff4d09",
            "simplipy_manifest_revision": MANIFEST_REVISION,
            "files": files,
        },
    )
    print(f"Verified {len(files)} files; local offline manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
