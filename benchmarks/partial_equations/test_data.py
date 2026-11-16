"""Offline data integrity checks. This does not test model fitting performance."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "partial_equations"


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class DataIntegrity(unittest.TestCase):
    def test_reproduction_is_byte_stable(self):
        environment = json.loads((DATA / "generator_environment.json").read_text())
        if (platform.python_version(), platform.system()) != (environment["python"], environment["system"]):
            self.skipTest(
                "Byte regeneration pinned to Windows Python 3.12.8; frozen CSV integrity is checked separately"
            )
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA).rglob("*") if p.is_file()}
        subprocess.run([sys.executable, str(ROOT / "prepare_data.py")], check=True)
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA).rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_pinned_csv_hashes(self):
        expected = json.loads((DATA / "files_sha256.json").read_text())
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), digest)

    def test_abalone_partition_and_quality_preservation(self):
        folder = DATA / "abalone"
        rows = read(folder / "table.csv")
        split = {r["sample_id"]: r["partition"] for r in read(folder / "split.csv")}
        self.assertEqual(len(rows), 4177)
        self.assertEqual(len(split), len(rows))
        groups = {}
        for row in rows:
            partition = split[row["sample_id"]]
            self.assertEqual(partition == "final_extrapolation", float(row["length"]) > 0.65)
            groups.setdefault(row["group_id"], set()).add(partition)
        self.assertTrue(all(len(parts) == 1 for parts in groups.values()))
        self.assertEqual(sum(float(row["height"]) == 0 for row in rows), 2)
        self.assertEqual(len(read(folder / "quality_flags.csv")), 3)
        self.assertFalse(json.loads((folder / "metadata.json").read_text())["performance_evaluated"])

    def test_synthetic_partitions_and_negative_controls(self):
        folders = sorted(p for p in (DATA / "synthetic").glob("*/*") if p.is_dir())
        self.assertEqual(len(folders), 24)
        for folder in folders:
            rows = read(folder / "table.csv")
            split = {r["sample_id"]: r["partition"] for r in read(folder / "split.csv")}
            self.assertEqual(len(rows), 400)
            self.assertEqual(len(split), 400)
            self.assertEqual(set(split), {r["sample_id"] for r in rows})
            self.assertEqual(set(split), {r["sample_id"] for r in read(folder / "oracle_scoring_only.csv")})
            train = [r for r in rows if split[r["sample_id"]] == "train"]
            extra = [r for r in rows if split[r["sample_id"]] == "final_extrapolation"]
            self.assertTrue(all(abs(float(r["z"])) <= 1 for r in train))
            self.assertTrue(all(float(r["z"]) >= 1.2 for r in extra))
            if folder.parent.name == "confounded":
                self.assertTrue(all(r["x"] == r["z"] for r in train))
                deconfounded = [r for r in rows if split[r["sample_id"]] == "final_deconfounding"]
                self.assertTrue(all(r["x"] != r["z"] for r in deconfounded))
            if folder.parent.name == "zero_factor":
                self.assertTrue(all(float(r["x"]) == 0 for r in train))
            if folder.parent.name == "multiplicative":
                self.assertTrue(any(float(r["x"]) == 0 for r in train))


if __name__ == "__main__":
    unittest.main(verbosity=2)
