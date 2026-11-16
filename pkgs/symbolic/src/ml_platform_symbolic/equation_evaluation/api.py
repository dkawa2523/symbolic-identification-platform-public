"""Frozen equation/context inputs and common numerical evidence, without selection."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from ml_platform_core.config import deep_merge

from ..config import resolve_config
from ..coordinates import execution_region, intersect_input_geometry
from ..data import digest, frozen_labels, load_prepared, observation_index, prepare, read_json, write_json
from ..domains import validate_bounds
from ..execution_views import bind_view
from ..expressions import inspect_equation, parse_expression
from ..hypotheses import (
    artifact_record,
    prediction_equations,
)
from ..hypothesis_storage import read_hypotheses, write_hypotheses
from ..model import load_model
from ..problem import problem_config, problem_document, requirement_reference
from ..profile_evidence import profile_evidence
from ..relation_origin import canonical_origin
from ..worker_process import stop_worker, worker_process
from .numerical import _numerical

EVALUATION_AXES = {
    "reggression": "structure",
    "effector": "behavior",
    "likelihood_profiler": "evidence",
    "interpretation": "interpretation",
}
METHODS = {name: f"{__package__}.backends.{axis}" for name, axis in EVALUATION_AXES.items()}
OPTIONS = {
    "interpretation": {"ids", "max_wall_seconds", "paths", "reference"},
    "reggression": {"ids", "max_wall_seconds", "patterns"},
    "effector": {"ids", "max_wall_seconds", "kind", "features", "bins", "centering", "grid_points"},
    "likelihood_profiler": {
        "ids",
        "max_wall_seconds",
        "sigma",
        "sigma_unit",
        "parameter_units",
        "confidence_level",
        "parameter_ranges",
        "quantities",
        "step",
        "maxiters",
    },
}


def register_evaluator(name, *, axis, module, options):
    """Explicit installed evaluator hook; no artifact imports or plugin discovery."""
    if not name.isidentifier() or name in METHODS or not isinstance(axis, str) or not axis.isidentifier():
        raise ValueError("New evaluator needs a unique name and a declared analysis axis")
    if not isinstance(module, str) or not module or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("Evaluator module must be an explicitly installed Python module name")
    METHODS[name] = module
    EVALUATION_AXES[name] = axis
    OPTIONS[name] = set(options) | {"ids", "max_wall_seconds"}


def new_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output must be new/empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manual(definition, variables):
    equation = deepcopy(definition.get("equation"))
    if equation is None:
        parameters = deepcopy(definition.get("parameters", {}))
        names = {v["column"] for v in variables}
        if names & parameters.keys():
            raise ValueError("Manual parameter names conflict with original feature columns")
        public = [{**v, "id": v["column"]} for v in variables]
        equation = parse_expression(definition["expression"], public, trainable=False, parameters=parameters)
        ids = {v["column"]: v["id"] for v in variables}

        def convert(node):
            if "var" in node:
                node["var"] = ids[node["var"]]
            for child in node.get("args", []):
                convert(child)

        convert(equation["ast"])
    for parameter in equation["parameters"].values():
        parameter.setdefault("origin", "user_declared")
    inspect_equation(equation, variables)
    return {
        "id": definition["id"] + "@manual",
        "candidate_id": definition["id"],
        "variant": "manual",
        "kind": "explicit",
        "equation": equation,
        "origin": {"engine": "manual", "knowledge": definition.get("knowledge", "user supplied hypothesis")},
        "native_check": {"kind": "manual declaration; no native engine execution", "passed": True},
    }


def _collect(sources, base, refs, provenance):
    records, bindings = [], {}
    for source in sources:
        if set(source) == {"manual"}:
            records.append(_manual(source["manual"], provenance["variables"]))
            continue
        if set(source) in ({"discovery"}, {"discovery", "prepared_run"}):
            path = (base / source["discovery"]).resolve()
            summary = read_json(path)
            source_profile = profile_evidence(path)
            bindings[str(path)] = digest(path)
            source_run = (base / source["prepared_run"]).resolve() if "prepared_run" in source else Path(refs["run"])
            if summary.get("prepared_sha256") != digest(source_run):
                raise ValueError("Discovery prepared run identity mismatch")
            native_context = read_json(source_run)
            if any(
                native_context.get(key) != provenance.get(key)
                for key in ("data_sha256", "split_sha256", "variables", "target", "input_geometry")
            ):
                raise ValueError("Discovery data/split/units/domain/fixed information differs from this study")
            if (
                problem_config(native_context)["data"]["fixed_constants"]
                != problem_config(provenance)["data"]["fixed_constants"]
            ):
                raise ValueError("Discovery fixed information differs from this study")
            if native_context["partition_hashes"]["train"] != provenance["partition_hashes"]["train"]:
                raise ValueError("Discovery Train partition differs from this study")
            bindings[str(source_run)] = digest(source_run)
            candidate_path = path.with_name("candidates.jsonl")
            if (
                summary["status"] not in {"ok", "partial"}
                or summary.get("portable_status", "candidates") != "candidates"
            ):
                records.append(
                    {
                        "id": f"{summary['engine']}/{summary['seed']}@native_result",
                        "kind": "native_result",
                        "origin": {
                            **summary,
                            "summary": summary,
                            "profile_evidence": source_profile,
                            "prepared_run": str(source_run),
                        },
                        "native_payload": deepcopy(summary),
                        "unavailable_reason": summary.get("reason", "No portable point equation"),
                    }
                )
                continue
            if digest(candidate_path) != summary["candidates_sha256"]:
                raise ValueError("Discovery candidates checksum mismatch")
            bindings[str(candidate_path)] = digest(candidate_path)
            for line in candidate_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                records.append(
                    {
                        "id": item["candidate_id"] + "@native",
                        "candidate_id": item["candidate_id"],
                        "variant": "native",
                        "kind": "explicit",
                        "equation": item["equation"],
                        "origin": {
                            "engine": item["engine"],
                            "seed": item["seed"],
                            "summary": summary,
                            "profile_evidence": source_profile,
                            "prepared_run": str(source_run),
                        },
                        "native_check": item["native_check"],
                    }
                )
            continue
        if set(source) in ({"ensemble"}, {"relation", "candidate_id"}):
            key = "ensemble" if "ensemble" in source else "relation"
            path = (base / source[key]).resolve()
            if path.stat().st_size > 20_000_000:
                raise ValueError("Native artifact exceeds import size limit")
            record = artifact_record(read_json(path), candidate_id=source.get("candidate_id"))
            from ..units import quantity_role

            for actual, wanted in zip(record["variables"], provenance["variables"], strict=True):
                if (actual["id"], actual["column"], actual.get("unit"), quantity_role(actual)) != (
                    wanted["id"],
                    wanted["column"],
                    wanted.get("unit"),
                    quantity_role(wanted),
                ):
                    raise ValueError("Native artifact input coordinates/roles differ; no silent reinterpretation")
            if any(record["target"].get(k) != provenance["target"].get(k) for k in ("column", "unit")) or quantity_role(
                record["target"]
            ) != quantity_role(provenance["target"]):
                raise ValueError("Native artifact target coordinates/roles differ")
            bindings[str(path)] = digest(path)
            record.update(
                id=path.stem + "/" + (source.get("candidate_id") or digest(path)[:12]),
                origin={
                    "artifact": str(path),
                    "artifact_sha256": digest(path),
                    "engine": record["native_metadata"].get("engine", "imported_native"),
                    "seed": record["native_metadata"].get("seed"),
                },
            )
            for equation in prediction_equations(record):
                inspect_equation(equation, provenance["variables"])
            records.append(record)
            continue
        if set(source) == {"bundle"}:
            path = (base / source["bundle"]).resolve()
            model = load_model(path)
            wanted = [(v["column"], v["unit"]) for v in provenance["variables"]]
            if [(v["column"], v["unit"]) for v in model.schema["variables"]] != wanted:
                raise ValueError("Bundle feature order/units differ; explicit conversion required")
            from ..units import quantity_role

            if any(
                quantity_role(a) != quantity_role(b)
                for a, b in zip(model.schema["variables"], provenance["variables"], strict=True)
            ) or quantity_role(model.schema["target"]) != quantity_role(provenance["target"]):
                raise ValueError(
                    "Bundle quantity roles differ or are unverified; no silent point/difference reinterpretation"
                )
            if model.schema["target"]["unit"] != provenance["target"]["unit"]:
                raise ValueError("Bundle target units differ")
            bindings[str(path)] = digest(path)
            records.append(
                {
                    "id": path.stem + "/" + digest(path)[:12] + "@bundle",
                    "candidate_id": path.stem,
                    "variant": "bundle",
                    "kind": "explicit",
                    "equation": model.equation,
                    "variables": model.schema["variables"],
                    "input_geometry": model.schema.get("input_geometry"),
                    "target": model.schema["target"],
                    "origin": {"bundle": str(path), "metadata": model.metadata},
                    "native_check": model.metadata.get("selected_native_check")
                    or {"passed": None, "kind": "native parity unverified; checksum is not numerical parity"},
                }
            )
            continue
        if set(source) == {"derived"}:
            path = (base / source["derived"]).resolve()
            item = read_json(path)
            if (
                item.get("kind") != "derived_equation"
                or item["data_sha256"] != provenance["data_sha256"]
                or item["split_sha256"] != provenance["split_sha256"]
            ):
                raise ValueError(
                    "Derived fit must bind the same data/split; use a provenance-preserving bundle for a new problem"
                )
            inspect_equation(item["equation"], provenance["variables"])
            bindings[str(path)] = digest(path)
            records.append(
                {
                    "id": item["parent"]["id"] + "/" + item["variant"],
                    "candidate_id": item["parent"]["id"],
                    "variant": item["variant"],
                    "kind": "explicit",
                    "equation": item["equation"],
                    "variables": item["variables"],
                    "input_geometry": item.get("input_geometry"),
                    "target": item["target"],
                    "origin": {"derived": str(path), "parent": item["parent"], "fit": item},
                    "native_check": {
                        "passed": None,
                        "kind": "named derived point fit; source native remains unchanged",
                    },
                }
            )
            continue
        raise ValueError("Each source is manual, discovery, bundle, derived, ensemble, or relation with candidate_id")
    for record in records:
        record["origin"] = canonical_origin(record.get("origin"), record.get("native_metadata"))
    ids = [item["id"] for item in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Equations require unique nonempty IDs; duplicate hypotheses are not votes")
    return records, bindings


def final_exposure(refs, provenance, records, *, history="unknown"):
    evidence = []
    task = Path(refs["run"]).parent.parent
    tasks = {task} | {
        Path(record["origin"]["prepared_run"]).parent.parent
        for record in records
        if record.get("origin", {}).get("prepared_run")
    }
    paths = []
    for source_task in tasks:
        paths.extend(source_task / name for name in ("model_info.json", "result_summary.json", "final_access.json"))
        paths.append(source_task / "finalize/result_summary.json")
        if (source_task / "finalize").is_dir():
            paths.extend((source_task / "finalize").glob("*/result_summary.json"))
    for path in paths:
        if path.is_file():
            item = read_json(path)
            if path.name == "final_access.json":
                evidence.append(
                    {
                        "source": str(path),
                        "data_sha256": item.get("data_sha256"),
                        "split_sha256": item.get("split_sha256"),
                    }
                )
            elif item.get("final_metrics") is not None:
                evidence.append(
                    {
                        "source": str(path),
                        "data_sha256": provenance["data_sha256"],
                        "split_sha256": provenance["split_sha256"],
                    }
                )
    for record in records:
        metadata = record.get("origin", {}).get("metadata", {})
        old = metadata.get("provenance", {})
        if metadata.get("final_metrics") is not None:
            evidence.append(
                {
                    "source": record["origin"].get("bundle"),
                    "data_sha256": old.get("data_sha256"),
                    "split_sha256": old.get("split_sha256"),
                }
            )
    if history == "known_exposed":
        evidence.append(
            {
                "source": "explicit known-exposed history declaration",
                "data_sha256": provenance["data_sha256"],
                "split_sha256": provenance["split_sha256"],
            }
        )
    exposed = any(item["data_sha256"] == provenance["data_sha256"] for item in evidence)
    return {
        "declared_history": history,
        "status": "known prior Final exposure on these data" if exposed else "prior exposure unknown",
        "same_data_exposed": exposed,
        "evidence": evidence,
    }


def create_study(plan, *, run_dir):
    """Resolve and evaluate equations on Train/Selection. Never load Final targets or select."""
    base = Path.cwd()
    if isinstance(plan, (str, Path)):
        base = Path(plan).resolve().parent
        plan = read_json(plan)
    plan = deepcopy(plan)
    allowed = {
        "prepared_run",
        "problem",
        "sources",
        "question",
        "purpose",
        "engineering_tolerance",
        "regions",
        "slices",
        "derivative_steps",
        "derivatives",
        "data",
        "mode",
        "required_evidence",
        "final_history",
        "data_kind",
        "quantity_tolerances",
        "parameter_target",
        "additional_constraints",
        "hypothesis_groups",
        "strata_column",
        "execution_views",
        "response_validation",
        "parameter_evidence_ids",
    }
    if set(plan) - allowed:
        raise ValueError(f"Unknown study settings: {set(plan) - allowed}")
    parameter_evidence_ids = plan.get("parameter_evidence_ids", [])
    if (
        not isinstance(parameter_evidence_ids, list)
        or any(not isinstance(item, str) or not item.strip() for item in parameter_evidence_ids)
        or len(set(parameter_evidence_ids)) != len(parameter_evidence_ids)
    ):
        raise ValueError("parameter_evidence_ids must be a list of unique nonempty record IDs")
    directory = new_directory(run_dir)
    if ("prepared_run" in plan) == ("problem" in plan):
        raise ValueError("Supply one prepared_run or one problem config (preparation only, no implicit search)")
    if "problem" in plan:
        cfg = resolve_config(plan["problem"])
        refs = prepare(cfg, directory / "prepare")
    else:
        path = (base / plan["prepared_run"]).resolve()
        refs = {
            "run": str(path),
            **{
                name: str(path.with_name(name + ".npz"))
                for name in ("train", "selection", "final")
                if path.with_name(name + ".npz").is_file()
            },
        }
    provenance = read_json(refs["run"])
    evaluation_config = problem_config(provenance)
    additional = plan.get("additional_constraints", {})
    if additional:
        data = evaluation_config["data"]
        if (set(additional) & {"probes", "response_bounds", "response_checks"}
                and data.get("source_units") != data.get("units")):
            import warnings
            warnings.warn(
                f"additional_constraints uses prepared canonical units {data['units']}, "
                f"not CSV/source units {data.get('source_units')}; input geometry is applied afterwards",
                UserWarning, stacklevel=2,
            )
        original = evaluation_config["constraints"]
        if set(additional) - {
            "require_units",
            "max_nodes",
            "max_free_parameters",
            "probes",
            "response_bounds",
            "response_checks",
        }:
            raise ValueError("Additional study constraints accept units, size and sampled response requirements only")
        if original["require_units"] and additional.get("require_units") is False:
            raise ValueError("Additional constraints cannot weaken a source requirement")
        if additional.get("max_nodes", original["max_nodes"]) > original["max_nodes"]:
            raise ValueError("Additional constraints cannot increase the source node cap")
        old_cap = original.get("max_free_parameters")
        if old_cap is not None and (
            additional.get("max_free_parameters", old_cap) is None
            or additional.get("max_free_parameters", old_cap) > old_cap
        ):
            raise ValueError("Additional constraints cannot increase the source parameter cap")
        merged = deep_merge(original, additional)
        for key in ("probes", "response_checks"):
            merged[key] = original.get(key, []) + additional.get(key, [])
        old_bounds, new_bounds = original["response_bounds"], additional.get("response_bounds", [None, None])
        merged["response_bounds"] = [
            max((v for v in (old_bounds[0], new_bounds[0]) if v is not None), default=None),
            min((v for v in (old_bounds[1], new_bounds[1]) if v is not None), default=None),
        ]
        validate_bounds(merged["response_bounds"])
        evaluation_config = resolve_config({**evaluation_config, "constraints": merged})
    roles = {"evidence_fit": "train", "comparison": "selection", "behavior_reference": "train", **plan.get("data", {})}
    if set(roles) != {"evidence_fit", "comparison", "behavior_reference"} or roles["evidence_fit"] != "train":
        raise ValueError("Likelihood fitting uses Train only; data roles are explicit and unique")
    if roles["comparison"] != "selection" or roles["behavior_reference"] not in {"train", "selection"}:
        raise ValueError("Comparison uses Selection; Final is unavailable before an explicit frozen choice")
    arrays = {}
    for partition in ("train", "selection"):
        X, y, ids = load_prepared(refs, provenance, partition)
        arrays.update(
            {partition + "_X": X, partition + "_y": y, partition + "_ids": ids, partition + "_weights": np.ones(len(y))}
        )
    identities = observation_index(provenance)
    group_column = provenance["config"]["split"].get("group_column")
    if group_column:
        groups = dict(zip(identities["sample_id"], identities["group"], strict=True))
        for partition in ("train", "selection"):
            arrays[partition + "_groups"] = np.asarray([groups[key] for key in arrays[partition + "_ids"]], dtype=str)
    strata_column = plan.get("strata_column")
    if strata_column:
        if strata_column == provenance["target"]["column"]:
            raise ValueError("A stratum label cannot read held-out target values as metadata")
        labels = frozen_labels(provenance, strata_column)
        arrays["strata_ids"], arrays["frozen_strata"] = labels["sample_id"], labels["label"]
        lookup = dict(zip(labels["sample_id"], labels["label"], strict=True))
        for partition in ("train", "selection"):
            arrays[partition + "_strata"] = np.asarray([lookup[key] for key in arrays[partition + "_ids"]], dtype=str)
    variables = provenance["variables"]
    columns = {v["column"] for v in variables}
    regions = plan.get("regions", {"all": {}})
    for bounds in regions.values():
        if set(bounds) - columns:
            raise ValueError("Region names must be original feature columns")
        for limits in bounds.values():
            validate_bounds(limits)
    steps = plan.get("derivative_steps", {})
    if set(steps) - columns:
        raise ValueError("Derivative steps use original feature columns")
    resolved_steps = [
        float(steps.get(v["column"], max(np.ptp(arrays["train_X"][:, i]) * 1e-5, 1e-7)))
        for i, v in enumerate(variables)
    ]
    if not np.isfinite(resolved_steps).all() or min(resolved_steps) <= 0:
        raise ValueError("Finite-difference steps must be finite positive execution-coordinate increments")
    parameter_target = plan.get("parameter_target")
    if parameter_target not in {None, "conditional_model", "latent_relation"}:
        raise ValueError(
            "parameter_target is conditional_model or latent_relation, not inferred from coefficient names"
        )
    observation = provenance["config"].get("observation_model") or {}
    if (
        plan.get("purpose") == "parameter_interpretation"
        and (observation.get("errors") or {}).get("inputs")
        and parameter_target is None
    ):
        raise ValueError(
            "Known input error: explicitly choose parameter_target conditional_model or latent_relation; fixed-X intervals do not identify latent physical coefficients"
        )
    tolerance = plan.get("engineering_tolerance")
    if isinstance(tolerance, dict):
        from ..units import convert_width

        if set(tolerance) != {"value", "unit"}:
            raise ValueError("Engineering tolerance quantity needs value and unit")
        tolerance = float(convert_width(tolerance["value"], tolerance["unit"], provenance["target"].get("unit")))
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError(
            "Engineering tolerance must be positive and representable; numbers use working target units, or supply value/unit"
        )
    if "derivatives" in plan and type(plan["derivatives"]) is not bool:
        raise ValueError("derivatives is an explicit boolean request")
    response_validation = deepcopy(plan.get("response_validation"))
    if response_validation is not None:
        from ..units import convert_width

        if (
            set(response_validation) != {"pairs", "tolerance", "scope"}
            or not isinstance(response_validation["scope"], str)
            or not response_validation["scope"].strip()
        ):
            raise ValueError(
                "Response validation fixes ordered Selection ID pairs, tolerance quantity and interpretation scope"
            )
        pairs = response_validation["pairs"]
        ids = set(arrays["selection_ids"].tolist())
        if (
            not isinstance(pairs, list)
            or not pairs
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(i, str) or i not in ids for i in pair)
                or pair[0] == pair[1]
                for pair in pairs
            )
        ):
            raise ValueError("Response pairs must name two distinct existing Selection IDs, never Final rows")
        if len({tuple(pair) for pair in pairs}) != len(pairs):
            raise ValueError("Duplicate response pairs are not repeated evidence")
        quantity = response_validation["tolerance"]
        if not isinstance(quantity, dict) or set(quantity) != {"value", "unit"}:
            raise ValueError("Response RMSE tolerance needs an explicit value/unit difference quantity")
        threshold = float(convert_width(quantity["value"], quantity["unit"], provenance["target"].get("unit")))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Response tolerance must be positive and representable")
        response_validation.update(
            max_rmse=threshold,
            unit=provenance["target"].get("unit"),
            data_role="selection",
            direction="second observation minus first",
        )
    for requirement in plan.get("quantity_tolerances", {}).values():
        if (
            not {"max_width", "unit"} <= set(requirement)
            or set(requirement) - {"max_width", "unit", "coordinate_role", "request"}
            or requirement.get("coordinate_role") not in {None, "value", "difference"}
            or not requirement["unit"]
            or not np.isfinite(requirement["max_width"])
            or requirement["max_width"] <= 0
        ):
            raise ValueError("Each quantity tolerance needs a positive max_width and explicit unit")
    for requirement in plan.get("quantity_tolerances", {}).values():
        if "request" in requirement:
            from ..question_results import question_signature

            if (
                question_signature(
                    {
                        "kind": "conditional_interval",
                        "estimand": parameter_target or "conditional_model",
                        "unit": requirement["unit"],
                        "coordinate_role": requirement.get("coordinate_role"),
                        "request": requirement["request"],
                    }
                )
                is None
            ):
                raise ValueError("Precision request must freeze quantity definition, confidence and observation model")
    # Historical width-only requests remain readable but cannot establish new
    # precision adoption; missing_precision reports the unresolved meaning.
    if not set(plan.get("required_evidence", [])) <= METHODS.keys():
        raise ValueError("Unknown required evidence method")
    slices = {}
    for name, definition in plan.get("slices", {}).items():
        if not name.isidentifier() or definition["feature"] not in columns:
            raise ValueError("Slice name must be an identifier and feature an original column")
        at = definition.get("at", {})
        if set(at) != columns - {definition["feature"]}:
            raise ValueError("Specify all other slice conditions explicitly; no hidden median recipe")
        points = [
            [x if v["column"] == definition["feature"] else at[v["column"]] for v in variables]
            for x in definition["grid"]
        ]
        if not points or not np.isfinite(points).all():
            raise ValueError("Slice needs finite points")
        slices[name] = {**definition, "X": points, "support": "user requested slice; joint support not established"}
    records, bindings = _collect(plan["sources"], base, refs, provenance)
    if set(parameter_evidence_ids) - {record["id"] for record in records}:
        raise ValueError("parameter_evidence_ids must refer to immutable record IDs in this study")
    for record in records:
        source_geometry = record.get("input_geometry", record.get("source_schema", {}).get("input_geometry"))
        record["input_geometry"] = intersect_input_geometry(source_geometry, provenance.get("input_geometry"))
        record["execution_domain"] = execution_region(record["input_geometry"], variables)
    views = plan.get("execution_views", {})
    if set(views) - {r["id"] for r in records}:
        raise ValueError("Execution views refer to explicit immutable relation IDs")
    records = [bind_view(record, views.get(record["id"], record.get("execution_view"))) for record in records]
    groups = plan.get("hypothesis_groups", {})
    if not set(groups) <= {r["id"] for r in records}:
        raise ValueError("Hypothesis annotations use explicit record IDs")
    for group in groups.values():
        if set(group) != {"label", "rationale"} or not all(isinstance(v, str) and v.strip() for v in group.values()):
            raise ValueError("Each scientific hypothesis annotation needs label and rationale; syntax is not meaning")
    history = plan.get("final_history", "unknown")
    if history not in {"unknown", "declared_unseen", "known_exposed"}:
        raise ValueError("final_history is unknown, declared_unseen or known_exposed")
    context = {
        "schema_version": 2 if response_validation is not None else 1,
        "hypothesis_groups": groups,
        "prepared": refs,
        "prepared_sha256": digest(refs["run"]),
        "variables": variables,
        "target": provenance["target"],
        "unit_coordinates": provenance.get("unit_coordinates"),
        "display_units": provenance["config"]["data"].get("display_units", {}),
        "input_geometry": provenance.get("input_geometry"),
        "config": evaluation_config,
        "response_point_origin": {
            "additional_input_units": evaluation_config["data"]["units"],
            "additional_input_coordinates": "canonical problem coordinates before input geometry",
            "discovery_probes": len(problem_config(provenance)["constraints"].get("probes", [])),
            "discovery_checks": len(problem_config(provenance)["constraints"].get("response_checks", [])),
        },
        "constraint_application": {
            "kind": "postcheck only; no claim these additions were enforced in native search",
            "discovery_constraints": problem_config(provenance)["constraints"],
            "discovery_requirements_ref": requirement_reference(provenance),
            "additional": additional,
            "candidate_pool": "the immutable source pool, including its recorded export cap and truncation",
        },
        "data_sha256": provenance["data_sha256"],
        "split_sha256": provenance["split_sha256"],
        "data_roles": roles,
        "observation_model": provenance["config"].get("observation_model"),
        "regions": regions,
        "slices": slices,
        "derivative_steps": resolved_steps,
        "compute_derivatives": bool(plan.get("derivatives", False)),
        "group_column": group_column,
        "strata_column": strata_column,
        "derivative_method": "execution-coordinate finite differences; original physical direction requires a declared source path; not analytic",
        "question": plan.get("question", "Compare equation hypotheses"),
        "purpose": plan.get("purpose", "local_prediction"),
        "parameter_target": parameter_target or "conditional_model",
        "engineering_tolerance": tolerance,
        "response_validation": response_validation,
        "parameter_evidence_ids": parameter_evidence_ids,
        "required_evidence": plan.get("required_evidence", []),
        "quantity_tolerances": plan.get("quantity_tolerances", {}),
        "mode": plan.get("mode", "development"),
        "data_kind": plan.get("data_kind", "unspecified; not asserted to be equipment measurements"),
        "final_exposure": final_exposure(refs, provenance, records, history=history),
        "final_history": history,
        "support": "marginal Train ranges only; not a joint-feasibility or safety guarantee",
        "feature_ranges": provenance["feature_ranges"],
        "source_bindings": bindings,
        "runtime": {
            "python": platform.python_version(),
            "image": os.getenv("SYMBOLIC_CONTAINER_IMAGE_ID"),
            "common_source_sha256": {
                name: digest(Path(__file__).parents[1] / name)
                for name in (
                    "equation_evaluation/api.py",
                    "equation_evaluation/numerical.py",
                    "estimation.py",
                    "execution_views.py",
                    "question_results.py",
                    "metrics.py",
                    "operators.py",
                    "hypotheses.py",
                    "hypothesis_storage.py",
                    "evaluation.py",
                    "units.py",
                    "coordinates.py",
                    "numerical_fidelity.py",
                    "expressions.py",
                    "constraints.py",
                    "domains.py",
                    "parameters.py",
                )
            },
        },
    }
    rows = _numerical(context, records, arrays)
    write_json(directory / "request.json", plan)
    stored_context = {**context, **problem_document(evaluation_config)}
    write_json(directory / "resolved.json", stored_context)
    write_hypotheses(directory / "hypotheses.json", records)
    write_json(directory / "numerical.json", rows)
    np.savez_compressed(directory / "numerical.npz", **arrays)
    write_json(
        directory / "study.json",
        {
            "schema_version": 3,
            "files": {
                name: digest(directory / name)
                for name in ("resolved.json", "hypotheses.json", "numerical.json", "numerical.npz")
            },
        },
    )
    return directory


def read_study_evidence(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "study.json")
    if manifest.get("schema_version") not in {1, 2, 3} or set(manifest["files"]) != {
        "resolved.json",
        "hypotheses.json",
        "numerical.json",
        "numerical.npz",
    }:
        raise ValueError("Unsupported study manifest")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Study evidence changed: {name}")
    context = read_json(directory / "resolved.json")
    context["config"] = problem_config(context)
    records = read_hypotheses(directory / "hypotheses.json")
    for record in records:
        record["origin"] = canonical_origin(record.get("origin"), record.get("native_metadata"))
    rows = read_json(directory / "numerical.json")
    with np.load(directory / "numerical.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return context, records, rows, arrays


def load_study(directory):
    context, records, rows, arrays = read_study_evidence(directory)
    if digest(context["prepared"]["run"]) != context["prepared_sha256"]:
        raise ValueError("Prepared context changed")
    for name, expected in context["source_bindings"].items():
        if digest(name) != expected:
            raise ValueError("Source equation artifacts changed")
    return context, records, rows, arrays


def evaluate_equations(study_dir, method, *, run_dir, options=None):
    """Explicit optional analysis; imports no search libraries and never chooses/refits source candidates."""
    _, records, _, _ = load_study(study_dir)
    if method not in METHODS:
        raise ValueError(f"Unknown evaluation method: {method}")
    options = deepcopy(options or {})
    if options.get("ids") and not set(options["ids"]) <= {r["id"] for r in records}:
        raise ValueError("Evaluation IDs must belong to the frozen hypothesis pool")
    if set(options) - OPTIONS[method]:
        raise ValueError(f"Unknown {method} options: {set(options) - OPTIONS[method]}")
    seconds = options.get("max_wall_seconds", 120)
    if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("Evaluation wall budget must be finite and in (0,3600]")
    directory = new_directory(run_dir)
    from ..domains import runtime_extensions

    job = write_json(
        directory / "job.json",
        {
            "runtime_extensions": runtime_extensions(),
            "study": str(Path(study_dir).resolve()),
            "method": method,
            "directory": str(directory),
            "options": options,
            "axis": EVALUATION_AXES[method],
            "backend_module": METHODS[method],
        },
    )
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        started = time.monotonic()
        with worker_process(
            [sys.executable, "-m", f"{__package__}.worker", str(job)],
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process:
            try:
                process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                stop_worker(process)
                write_json(directory / "result.json", {"outcome": "failed", "reason": "worker timeout", "results": []})
    result = (
        read_json(directory / "result.json")
        if (directory / "result.json").is_file()
        else {"outcome": "failed", "reason": f"Worker exit {process.returncode}; see worker.log", "results": []}
    )
    result.update(
        {
            "schema_version": 1,
            "method": method,
            "study_sha256": digest(Path(study_dir) / "study.json"),
            "options": options,
            "runtime_image": os.getenv("SYMBOLIC_CONTAINER_IMAGE_ID"),
            "worker_exit_code": process.returncode,
            "wall_seconds": time.monotonic() - started,
        }
    )
    write_json(directory / "result.json", result)
    return directory
