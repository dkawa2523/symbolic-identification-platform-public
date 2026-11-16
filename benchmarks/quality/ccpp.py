"""Audit the official CCPP workbook; select Sheet1, never combine its shuffles.

No split, model fit, target statistics or Final scoring is performed.
"""

import argparse
import csv
import hashlib
import io
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
HEADERS = ["AT", "V", "AP", "RH", "PE"]
URL = "https://archive.ics.uci.edu/static/public/294/combined%2Bcycle%2Bpower%2Bplant.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sheets(payload):
    """Read this fixed numeric workbook with explicit cells; reject formulas/gaps."""
    with zipfile.ZipFile(io.BytesIO(payload)) as book:
        shared = ["".join(node.itertext()) for node in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        relationships = {
            r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        }
        definitions = ET.fromstring(book.read("xl/workbook.xml")).find("s:sheets", NS)
        output = {}
        for definition in definitions:
            member = posixpath.normpath(posixpath.join("xl", relationships[definition.attrib[RID]]))
            rows = []
            for index, row in enumerate(ET.fromstring(book.read(member)).find("s:sheetData", NS), 1):
                cells = row.findall("s:c", NS)
                expected = [f"{letter}{index}" for letter in "ABCDE"]
                if [c.attrib["r"] for c in cells] != expected:
                    raise ValueError(f"Missing/extra/reordered cells in {member} row {index}")
                values = []
                for cell in cells:
                    if cell.find("s:f", NS) is not None:
                        raise ValueError("Formula cell is not an original numeric observation")
                    value = cell.findtext("s:v", namespaces=NS)
                    if value is None:
                        raise ValueError("Missing value")
                    values.append(shared[int(value)] if cell.attrib.get("t") == "s" else value)
                if index == 1:
                    if values != HEADERS:
                        raise ValueError(f"Unexpected columns: {values}")
                else:
                    numbers = tuple(Decimal(v) for v in values)
                    if not all(v.is_finite() for v in numbers):
                        raise ValueError("Nonfinite observation")
                    rows.append(numbers)
            output[definition.attrib["name"]] = rows
        return output


def multiset_hash(rows):
    counter = Counter(rows)
    body = [[*[str(v.normalize()) for v in row], count] for row, count in sorted(counter.items())]
    return sha(json.dumps(body, separators=(",", ":")).encode())


def audit(root):
    source = (root / "source.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        payload = archive.read("CCPP/Folds5x2_pp.xlsx")
        readme = archive.read("CCPP/Readme.txt")
    tables = sheets(payload)
    if list(tables) != [f"Sheet{i}" for i in range(1, 6)]:
        raise ValueError("Expected the five original shuffles")
    selected = tables["Sheet1"]
    base = Counter(selected)
    summaries = []
    for name, rows in tables.items():
        equal = Counter(rows) == base
        summaries.append(
            {
                "sheet": name,
                "rows": len(rows),
                "multiset_sha256": multiset_hash(rows),
                "same_multiset_as_sheet1": equal,
                "same_order_as_sheet1": rows == selected,
            }
        )
        if len(rows) != 9568 or not equal:
            raise ValueError(f"Unexpected row count or different observations: {name}")
    groups = defaultdict(list)
    for row in selected:
        groups[row[:4]].append(row[4])
    group_ids = {key: f"x{i:05d}" for i, key in enumerate(sorted(groups))}
    table = root / "table.csv"
    if table.exists():
        raise FileExistsError("Existing audit outputs are not overwritten; use a new directory")
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", *HEADERS, "group_id"])
        for i, row in enumerate(selected, 1):
            writer.writerow([f"ccpp_{i:05d}", *[str(v) for v in row], group_ids[row[:4]]])
    metadata = {
        "dataset": "CCPP",
        "uci_id": 294,
        "doi": "10.24432/C5002N",
        "citation": "Tfekci, P. & Kaya, H. (2014). Combined Cycle Power Plant [Dataset]. UCI Machine Learning Repository.",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_page": "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
        "source_url": URL,
        "audited_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sha(source),
        "workbook_sha256": sha(payload),
        "table_sha256": sha(table.read_bytes()),
        "selected_sheet": "Sheet1",
        "selection_rule": "First workbook sheet, fixed before model evaluation; other sheets audit only",
        "features": HEADERS[:4],
        "target": "PE",
        "units": {"AT": "degC", "V": "cmHg", "AP": "mbar", "RH": "%", "PE": "MW"},
        "rows": len(selected),
        "missing_cells": 0,
        "nonfinite_cells": 0,
        "sheets": summaries,
        "unique_complete_rows": len(base),
        "exact_duplicate_excess_rows": len(selected) - len(base),
        "complete_duplicate_groups": sum(n > 1 for n in base.values()),
        "unique_input_groups": len(groups),
        "repeated_input_groups": sum(len(v) > 1 for v in groups.values()),
        "input_groups_with_different_targets": sum(len(set(v)) > 1 for v in groups.values()),
        "maximum_input_group_size": max(map(len, groups.values())),
        "group_definition": "Exact AT,V,AP,RH tuple; duplicate-leakage guard only, not experiment/plant/time ID",
        "duplicate_policy": "Keep every source row; same-input rows must remain in one future partition",
        "target_access": "Parsed only for schema, finiteness, duplicate and sheet multiset checks; no target statistics or model evaluation",
        "split_status": "Not generated; benchmark plan is separate",
        "pretraining_overlap": "Unknown; public dataset may be in pretrained corpora",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (root / "source.sha256").write_text(sha(source) + "\n", encoding="ascii")
    (root / "source_readme.txt").write_bytes(readme)
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "rows",
                    "unique_complete_rows",
                    "exact_duplicate_excess_rows",
                    "unique_input_groups",
                    "input_groups_with_different_targets",
                    "source_sha256",
                ]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ccpp"))
    audit(parser.parse_args().root)
