"""Freeze row identity and partitions before exposing training arrays to engines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import warnings
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from ._version import package_identity
from .coordinates import freeze_input_geometry, map_input_values
from .domains import validate_input_domain, within_bounds
from .problem import problem_document
from .units import apply_coordinate, quantity_conversion, unit_spec


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value, *, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".pending", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                value,
                stream,
                indent=indent,
                allow_nan=False,
                separators=(",", ":") if indent is None else None,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def read_json(path):
    """One strict boundary for configurations, plans, options and saved artifacts."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON setting: {key}; keep one explicit value")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid_constant(value)
        return number

    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object,
        parse_constant=invalid_constant, parse_float=finite_float,
    )


def read_observations(path, *, identifiers=("sample_id",), optional_identifiers=()):
    """IDs are literal strings, including leading zeroes; never infer them from numbers."""
    # Read the header as data, before pandas can rename ambiguous columns.
    # Let pandas own compression, encoding and source handling for both reads.
    # Its Python parser preserves NULs and rejects malformed header quoting.
    try:
        rows = pd.read_csv(path, header=None, nrows=1, dtype=str, keep_default_na=False, engine="python")
        header = rows.iloc[0].tolist() if len(rows) else []
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Malformed CSV header: {exc}") from exc
    if not header or any(not name.strip() or "\x00" in name for name in header):
        raise ValueError("CSV header names must be nonempty")
    if len(set(header)) != len(header):
        raise ValueError("Duplicate CSV header names")
    identifiers = tuple(dict.fromkeys(identifiers))
    try:
        with warnings.catch_warnings():
            # pandas otherwise infers an index or drops extra fields, silently
            # shifting IDs/features/targets when data rows exceed header width.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                path, dtype=dict.fromkeys((*identifiers, *optional_identifiers), str),
                keep_default_na=False, index_col=False,
            )
    except (pd.errors.ParserError, pd.errors.ParserWarning) as exc:
        raise ValueError(f"Malformed CSV rows: fields must match the header width: {exc}") from exc
    if any(name not in frame for name in identifiers):
        raise ValueError("Missing declared identity/group column")
    for name in dict.fromkeys((*identifiers, *(name for name in optional_identifiers if name in frame))):
        if frame[name].isna().any() or frame[name].str.strip().eq("").any():
            raise ValueError(f"Missing/empty identifier: {name}")
    return frame


def observation_index(provenance):
    """Frozen target-free identities. Legacy raw reads require the original hash."""
    saved = provenance.get("observation_index")
    if saved is not None:
        if digest(saved["file"]) != saved["sha256"]:
            raise ValueError("Frozen observation identities changed")
        with np.load(saved["file"], allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    cfg = provenance["config"]
    source = cfg["data"]["local_path"]
    if digest(source) != provenance["data_sha256"]:
        raise ValueError("Legacy source metadata changed; frozen identities unavailable")
    group = cfg["split"].get("group_column")
    frame = read_observations(source, identifiers=["sample_id", *([group] if group else [])])
    return {
        "sample_id": frame.sample_id.to_numpy(dtype=str),
        "group": frame[group].to_numpy(dtype=str) if group else frame.sample_id.to_numpy(dtype=str),
    }


def frozen_labels(provenance, column):
    """Freeze labels selected by a study; never use a mutable CSV during Final."""
    source = provenance["config"]["data"]["local_path"]
    if digest(source) != provenance["data_sha256"]:
        raise ValueError("Source label metadata changed")
    frame = read_observations(source, identifiers=["sample_id", column])
    ids = observation_index(provenance)["sample_id"]
    values = frame.set_index("sample_id")[column]
    return {"sample_id": ids, "label": values.loc[ids].to_numpy(dtype=str)}


def numeric_matrix(frame, columns):
    missing = [c for c in columns if c not in frame.columns]
    if missing or len(set(frame.columns)) != len(frame.columns):
        raise ValueError(f"Missing or duplicate columns: {missing}")
    try:
        values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Expected numeric columns {columns}: {exc}") from exc
    invalid = np.argwhere(~np.isfinite(values))
    if len(invalid):
        row, col = invalid[0]
        raise ValueError(f"Nonfinite input at row {row}, column {columns[col]}")
    return values


def column_to_working(values, schema, definition, source):
    record = (schema.get("unit_coordinates") or {}).get("columns", {}).get(definition["column"], {})
    if record and source == record["source_unit"] and definition.get("unit") == record["problem_unit"]:
        return apply_coordinate(values, record["scale"], record["offset"])
    a, b = quantity_conversion(source, definition.get("unit"), definition.get("coordinate_role"))
    return apply_coordinate(values, a, b)


def public_unit(schema, definition):
    """Frozen coordinate map is authoritative; public_units is legacy metadata."""
    column = definition.get("column")
    frozen = (schema.get("unit_coordinates") or {}).get("columns", {}).get(column)
    legacy = schema.get("public_units", {})
    if frozen:
        public = frozen["source_unit"]
        if column in legacy:
            scale, offset = quantity_conversion(legacy[column], public, definition.get("coordinate_role"))
            if not np.isclose(scale, 1.0, rtol=1e-14, atol=0) or offset != 0:
                raise ValueError("Saved public unit metadata conflicts with the frozen coordinate map")
        return public
    return legacy.get(column, definition.get("unit"))


def model_output(values, schema, output_unit=None, *, difference=False):
    """Map working values or nonnegative widths to the saved/requested coordinate."""
    target = schema.get("target", {})
    public = public_unit(schema, target)
    wanted = output_unit if output_unit is not None else public
    record = (schema.get("unit_coordinates") or {}).get("columns", {}).get(target.get("column"), {})
    values = np.asarray(values, dtype=float)
    if record and wanted == record["source_unit"] and target.get("unit") == record["problem_unit"]:
        return apply_coordinate(
            values,
            abs(record["scale"]) if difference else record["scale"],
            0.0 if difference else record["offset"],
            inverse=True,
        )
    a, b = quantity_conversion(target.get("unit"), wanted, target.get("coordinate_role"))
    return apply_coordinate(values, abs(a) if difference else a, 0.0 if difference else b)


def input_variables(schema):
    return (schema.get("input_geometry") or {}).get("source_variables", schema["variables"])


def model_matrix(frame, schema, units=None):
    """One saved-coordinate boundary shared by point, ensemble and implicit inference."""
    from .units import validate_unit_mapping

    variables = input_variables(schema)
    columns = [v["column"] for v in variables]
    supplied = {v["column"]: public_unit(schema, v) for v in variables}
    if units is not None:
        validate_unit_mapping(units, columns)
        supplied.update(units)
    X = numeric_matrix(frame, columns)
    for i, v in enumerate(variables):
        X[:, i] = column_to_working(X[:, i], schema, v, supplied[v["column"]])
    if not np.isfinite(X).all():
        raise ValueError("Unit conversion produced nonfinite input")
    if schema.get("input_geometry"):
        X = map_input_values(X, schema["input_geometry"])
    return X


def _partitions(frame, cfg):
    split = cfg["split"]
    if (
        "sample_id" not in frame
        or frame.sample_id.isna().any()
        or frame.sample_id.astype(str).str.strip().eq("").any()
        or frame.sample_id.duplicated().any()
    ):
        raise ValueError("sample_id must be unique, nonempty and present")
    groups = frame[split["group_column"]] if split.get("group_column") else frame.sample_id
    if groups.isna().any():
        raise ValueError("Missing split group")
    if split["method"] == "fixed":
        source = read_observations(split["file"], identifiers=("sample_id", "partition"))
        if list(source.columns) != ["sample_id", "partition"] or source.sample_id.duplicated().any():
            raise ValueError("split.csv requires unique sample_id,partition columns")
        if set(source.sample_id) != set(frame.sample_id.astype(str)):
            raise ValueError("Split IDs must cover input IDs exactly, without unknown/missing IDs")
        labels = source.set_index("sample_id").partition.loc[frame.sample_id.astype(str)].to_numpy()
    else:
        order = sorted(groups.astype(str).unique())
        np.random.default_rng(split["seed"]).shuffle(order)
        n_final = int(np.ceil(len(order) * split["valid_size"])) if cfg["evaluation"]["mode"] == "holdout" else 0
        n_selection = int(np.ceil((len(order) - n_final) * split["selection_size"]))
        assignments = {
            g: "final" if i < n_final else "selection" if i < n_final + n_selection else "train"
            for i, g in enumerate(order)
        }
        labels = groups.astype(str).map(assignments).to_numpy()
    expected = {"train", "selection", "final"} if cfg["evaluation"]["mode"] == "holdout" else {"train", "selection"}
    if set(labels) != expected:
        raise ValueError(f"Expected nonempty partitions {sorted(expected)}, got {set(labels)}")
    check = pd.DataFrame({"group": groups.astype(str), "partition": labels})
    if (check.groupby("group").partition.nunique() > 1).any():
        raise ValueError("A group crosses partitions")
    return labels, groups


def prepare(cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    data = cfg["data"]
    group_column = cfg["split"].get("group_column")
    frame = read_observations(data["local_path"], identifiers=["sample_id", *([group_column] if group_column else [])])
    features = data["feature_columns"]
    excluded = {data["target_column"], "sample_id", cfg["split"].get("group_column"), *data["id_columns"]}
    if not features or len(set(features)) != len(features) or set(features) & excluded:
        raise ValueError("Features must be distinct and exclude target, IDs and group columns")
    X = numeric_matrix(frame, features)
    y = numeric_matrix(frame, [data["target_column"]])[:, 0]
    transforms = {}
    for index, column in enumerate([*features, data["target_column"]]):
        unit = data["units"].get(column)
        source = data["source_units"].get(column, unit)
        role = data["coordinate_roles"].get(column)
        scale, offset = quantity_conversion(source, unit, role)
        transforms[column] = {
            "coordinate_role": role,
            "source_unit": source,
            "problem_unit": unit,
            "scale": scale,
            "offset": offset,
            "dimension_status": "known" if unit_spec(unit) is not None else "unknown",
        }
        if index < len(features):
            X[:, index] = apply_coordinate(X[:, index], scale, offset)
        else:
            y = apply_coordinate(y, scale, offset)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Unit conversion produced nonfinite data")
    variables = [
        {
            "id": f"x{i}",
            "column": c,
            "unit": data["units"].get(c),
            "domain": data["domain"].get(c),
            "coordinate_role": data["coordinate_roles"].get(c),
        }
        for i, c in enumerate(features)
    ]
    validate_input_domain(X, variables)
    if not within_bounds(y, data["domain"].get(data["target_column"])).all():
        raise ValueError("Observed target outside declared domain")
    if cfg["constraints"]["probes"]:
        validate_input_domain(np.asarray([[p[c] for c in features] for p in cfg["constraints"]["probes"]]), variables)
    source_X = X.copy()
    geometry, variables = freeze_input_geometry(cfg, variables)
    X = map_input_values(X, geometry)
    labels, groups = _partitions(frame, cfg)
    identity_file = directory / "observation_index.npz"
    np.savez_compressed(
        identity_file,
        sample_id=frame.sample_id.to_numpy(dtype=str),
        group=groups.to_numpy(dtype=str),
        partition=np.asarray(labels, dtype=str),
    )
    paths, counts = {}, {}
    for partition in sorted(set(labels)):
        mask = labels == partition
        path = directory / f"{partition}.npz"
        np.savez_compressed(
            path, X=X[mask], source_X=source_X[mask], y=y[mask], row_ids=frame.sample_id.to_numpy(dtype=str)[mask]
        )
        paths[partition] = str(path.resolve())
        counts[partition] = {"n_rows": int(mask.sum()), "n_groups": int(groups[mask].nunique())}
    versions = {"python": platform.python_version()}
    for package in ["ml-platform-symbolic", "ml-platform-core", "numpy", "pandas", "scipy", *cfg["search"]["engines"]]:
        with suppress(PackageNotFoundError):
            versions[package] = version(package)
    identity = package_identity()
    versions["ml-platform-symbolic"] = identity["version"]
    source = Path(__file__).resolve().parent
    git = shutil.which("git")
    revision = None
    if git and identity["mode"] == "source":
        revision = subprocess.run(  # noqa: S603 -- fixed read-only Git command; no shell or user arguments.
            [git, "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=False
        ).stdout.strip()
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*.py")))
    ).hexdigest()
    provenance = {
        "config": cfg,
        "observation_index": {"file": str(identity_file.resolve()), "sha256": digest(identity_file)},
        "unit_coordinates": {
            "policy": "canonical problem; explicit CSV conversion; all domains/constraints/tolerances in problem units",
            "columns": transforms,
        },
        "variables": variables,
        "input_geometry": geometry,
        "target": {
            "coordinate_role": data["coordinate_roles"].get(data["target_column"]),
            "column": data["target_column"],
            "unit": data["units"].get(data["target_column"]),
            "domain": data["domain"].get(data["target_column"]),
        },
        "fixed_constants": data["fixed_constants"],
        "partitions": counts,
        "partition_hashes": {k: digest(v) for k, v in paths.items()},
        "data_sha256": digest(data["local_path"]),
        "versions": versions,
        "package_identity": identity,
        "code_revision": revision or None,
        "source_sha256": source_hash,
        "feature_ranges": [
            [float(a), float(b)]
            for a, b in zip(X[labels == "train"].min(axis=0), X[labels == "train"].max(axis=0), strict=True)
        ],
    }
    if cfg["split"]["method"] == "fixed":
        provenance["split_sha256"] = digest(cfg["split"]["file"])
    else:
        provenance["split_sha256"] = hashlib.sha256(
            json.dumps(list(zip(frame.sample_id, labels, strict=True))).encode()
        ).hexdigest()
    metadata_path = Path(data["local_path"]).with_name("metadata.json")
    if metadata_path.is_file():
        provenance["source_metadata"] = read_json(metadata_path)
    provenance.pop("fixed_constants", None)
    provenance.update(problem_document(cfg))
    run = write_json(directory / "run.json", provenance)
    return {"run": str(run.resolve()), **paths}


def load_partition(path):
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"], data["row_ids"]


def load_prepared(refs, provenance, partition):
    if digest(refs[partition]) != provenance["partition_hashes"][partition]:
        raise ValueError(f"Prepared {partition} partition checksum mismatch")
    return load_partition(refs[partition])


def value_digest(value):
    """Stable existing JSON identity; no model execution or semantic simplification."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
