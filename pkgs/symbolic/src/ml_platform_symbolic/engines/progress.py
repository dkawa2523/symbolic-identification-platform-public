"""One atomic, data-only completed-native-iteration snapshot. No optimizer resume."""

import base64
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from ..data import digest
from ..expressions import inspect_equation
from .base import Candidate, DiscoveryResult


def save_progress(job_path, result, provenance):
    directory = Path(job_path).parent.resolve()
    if len(result.native_artifacts) > 32:
        raise ValueError("Too many native progress artifacts")
    files = {}
    for name, filename in result.native_artifacts.items():
        source = Path(filename).resolve()
        if not source.is_relative_to(directory) or source.stat().st_size > 20_000_000:
            raise ValueError("Native progress artifacts must be bounded local data")
        files[name] = {"name": source.name, "bytes": base64.b64encode(source.read_bytes()).decode("ascii")}
    payload = {
        "job_sha256": digest(job_path),
        "candidates": [asdict(c) for c in result.candidates],
        "summary": result.summary,
        "artifacts": files,
        "provenance": provenance,
    }
    content = json.dumps(payload, allow_nan=False, sort_keys=True)
    record = {"payload": payload, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    target = directory / "completed_progress.json"
    temporary = target.with_suffix(".pending")
    temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    os.replace(temporary, target)


def load_progress(job_path, variables):
    directory = Path(job_path).parent
    target = directory / "completed_progress.json"
    if not target.exists():
        return None
    if target.stat().st_size > 64_000_000:
        raise ValueError("Completed progress exceeds data resource limit")
    record = json.loads(target.read_text(encoding="utf-8"))
    payload = record["payload"]
    content = json.dumps(payload, allow_nan=False, sort_keys=True)
    if hashlib.sha256(content.encode()).hexdigest() != record["sha256"] or payload["job_sha256"] != digest(job_path):
        raise ValueError("Completed progress identity/checksum mismatch")
    candidates = [Candidate(**v) for v in payload["candidates"]]
    for candidate in candidates:
        inspect_equation(candidate.equation, variables)
        if candidate.native_check.get("passed") is not True:
            raise ValueError("Progress candidate has no passed native parity evidence")
    artifacts = {}
    for index, (name, item) in enumerate(payload["artifacts"].items()):
        # Never trust incoming path names or execute any saved content.
        output = directory / "recovered_native" / f"{index}_{Path(item['name']).name}"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(base64.b64decode(item["bytes"], validate=True))
        artifacts[name] = str(output)
    result = DiscoveryResult(candidates, "partial" if candidates else "timeout", payload["summary"], artifacts)
    return result, payload["provenance"], record["sha256"]
