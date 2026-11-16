"""Narrow compatibility patch for the pinned 2022 E2E source on PyTorch 2.x."""

import argparse
from pathlib import Path

OLD = "generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)"
NEW = (
    "generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)"
    "  # Modified by symbolic-handoff: PyTorch 2.x requires a boolean EOS mask."
)


def patch(package_dir):
    path = Path(package_dir) / "model/transformer.py"
    source = path.read_text(encoding="utf-8")
    if source.count(OLD) == 1 and NEW not in source:
        path.write_text(source.replace(OLD, NEW), encoding="utf-8")
    elif source.count(NEW) != 1 or OLD in source:
        raise ValueError("E2E source differs from the reviewed PyTorch EOS-mask compatibility patch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    patch(parser.parse_args().package_dir)
