"""Fetch the exact official Meta E2E checkpoint outside discovery; verify before use."""

import argparse
import urllib.request
from pathlib import Path

from ml_platform_symbolic.assets import verify_assets
from ml_platform_symbolic.data import digest, write_json
from ml_platform_symbolic.engines.e2e import MODEL_SHA256, SOURCE_REVISION

URL = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
SIZE = 374775977


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("pretrained/e2e"))
    root = parser.parse_args().directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "model1.pt"
    if not checkpoint.is_file():
        partial = checkpoint.with_suffix(".pt.partial")
        urllib.request.urlretrieve(URL, partial)
        if partial.stat().st_size != SIZE or digest(partial) != MODEL_SHA256:
            raise ValueError("Official E2E checkpoint size/hash mismatch; partial preserved, not loaded")
        partial.replace(checkpoint)
    if checkpoint.stat().st_size != SIZE or digest(checkpoint) != MODEL_SHA256:
        raise ValueError("Existing E2E checkpoint differs from reviewed official release; not overwritten")
    manifest = write_json(
        root / "manifest.json",
        {
            "engine": "e2e",
            "source": URL,
            "source_revision": MODEL_SHA256,
            "code_source": "https://github.com/facebookresearch/symbolicregression",
            "code_revision": SOURCE_REVISION,
            "directory": ".",
            "code_license": "Apache-2.0",
            "model_license": "no separate weights license confirmed; not redistributed",
            "format": "official whole-model PyTorch pickle; only this exact reviewed digest may be loaded",
            "files": [{"path": checkpoint.name, "size_bytes": SIZE, "sha256": MODEL_SHA256}],
        },
    )
    verify_assets(manifest, "e2e")
    print(f"Verified official checkpoint: {manifest}")


if __name__ == "__main__":
    main()
