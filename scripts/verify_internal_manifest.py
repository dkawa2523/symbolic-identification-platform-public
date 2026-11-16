"""Check the delivered file hashes before installation; no network or file mutations."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
    root = parser.parse_args().directory.resolve()
    files = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for name, checksum in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/unsafe delivered file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Delivered file checksum mismatch: {name}")
    print(f"Verified {len(files)} delivered files")


if __name__ == "__main__":
    main()
