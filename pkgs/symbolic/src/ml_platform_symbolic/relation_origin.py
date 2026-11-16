"""Resolve source provenance once; model checksums never imply native parity."""


def canonical_origin(origin, native_metadata=None):
    result = dict(origin or {})
    metadata = result.get("metadata", {})
    # source_origin is the explicit historical bundle field, not an ID/name
    # heuristic. Outer evaluated provenance remains authoritative when present.
    native = native_metadata or {}
    candidates = [
        result,
        result.get("source_origin", {}),
        metadata.get("origin", {}),
        metadata.get("source_origin", {}),
        metadata,
        native.get("source_origin", {}),
        native,
    ]
    for key in ("engine", "seed", "summary", "profile_evidence", "prepared_run"):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get(key) is not None:
                result[key] = candidate[key]
                break
    return result
