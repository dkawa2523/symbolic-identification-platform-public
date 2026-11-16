"""Lossless study storage: candidate state is separate from shared source context.

Only whole, explicit context fields are shared. Equations, coefficients, member
weights, candidate IDs, failure reasons and order are never deduplicated. The
single file is the existing study checksum/atomic-write boundary, with no paths
to chase or external context dependencies. Readers return ordinary independent
dicts, retaining the historical public API and its mutation isolation.
"""

import hashlib
import json

from .data import read_json, write_json

_CONTEXT_FIELDS = frozenset(
    {
        "origin",
        "variables",
        "target",
        "source_schema",
        "native_metadata",
        "source_constraints",
        "source_operators",
        "input_geometry",
        "execution_domain",
    }
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_records(records):
    if not isinstance(records, list):
        raise ValueError("Hypothesis records must be a list")
    ids = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("Each hypothesis needs a nonempty string id")
        if record["id"] in ids:
            raise ValueError("Duplicate hypothesis id")
        ids.add(record["id"])


def write_hypotheses(path, records):
    """Atomically write every record, factoring identical source context once.

    Index by canonical JSON bytes, not Python equality (which conflates True,
    1 and 1.0). Check the bytes too, so a digest collision cannot merge context.
    No candidate is dropped or changed, including rejected/native-only output.
    """
    _check_records(records)
    contexts, entries, index = [], [], {}
    for record in records:
        context = {key: value for key, value in record.items() if key in _CONTEXT_FIELDS}
        encoded = _encoded(context)
        checksum = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        matches = index.setdefault(checksum, [])
        reference = next((i for i, prior in matches if prior == encoded), None)
        if reference is None:
            reference = len(contexts)
            contexts.append(context)
            matches.append((reference, encoded))
        entries.append(
            {
                "context": reference,
                "record": {key: value for key, value in record.items() if key not in _CONTEXT_FIELDS},
            }
        )
    return write_json(
        path,
        {
            "schema_version": 1,
            "kind": "hypothesis_pool",
            "contexts": contexts,
            "records": entries,
        },
        indent=None,
    )


def read_hypotheses(path):
    """Read legacy flat lists or the normalized pool without rewriting either.

    Reconstruct independent context objects using the JSON decoder. It is
    faster than Python deepcopy for large provenance ledgers, and deliberately
    does not expose mutable aliases between candidates or caller-owned inputs.
    Full eager reconstruction still costs memory proportional to the legacy
    public record shape; the file and hashing no longer multiply that peak.
    """
    payload = read_json(path)
    if isinstance(payload, list):
        _check_records(payload)
        return payload
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "kind", "contexts", "records"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "hypothesis_pool"
        or not isinstance(payload["contexts"], list)
        or not isinstance(payload["records"], list)
    ):
        raise ValueError("Unsupported hypothesis storage format")
    contexts = payload["contexts"]
    if any(not isinstance(context, dict) or set(context) - _CONTEXT_FIELDS for context in contexts):
        raise ValueError("Invalid shared hypothesis context")
    encoded_contexts = [_encoded(context) for context in contexts]
    records, used = [], set()
    for entry in payload["records"]:
        if not isinstance(entry, dict) or set(entry) != {"context", "record"}:
            raise ValueError("Invalid hypothesis entry")
        reference, candidate = entry["context"], entry["record"]
        if type(reference) is not int or not 0 <= reference < len(contexts):
            raise ValueError("Invalid hypothesis context reference")
        if not isinstance(candidate, dict) or set(candidate) & _CONTEXT_FIELDS:
            raise ValueError("Candidate cannot override shared context")
        records.append({**json.loads(encoded_contexts[reference]), **candidate})
        used.add(reference)
    if used != set(range(len(contexts))):
        raise ValueError("Unreferenced hypothesis context")
    _check_records(records)
    return records
