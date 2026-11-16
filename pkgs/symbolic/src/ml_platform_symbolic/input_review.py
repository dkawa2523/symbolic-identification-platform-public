"""Read-only input review using the same preparation and adapter boundaries as a run."""

from pathlib import Path
from tempfile import TemporaryDirectory

from .assets import verify_assets
from .config import resolve_config
from .data import prepare, read_json
from .engines.base import DiscoveryContext
from .engines.registry import ENGINES, get_engine
from .engines.runner import discovery_job, discovery_request, engine_input_issues
from .response_search import constraint_application


def engine_catalog(name=None):
    names = [name] if name is not None else sorted(ENGINES)
    return {engine: get_engine(engine).capabilities() for engine in names}


def review_inputs(config):
    """Validate data integrity, never fit/search/score or retain a prepared dataset."""
    cfg = resolve_config(config)
    if cfg["task"] == "symbolic_infer":
        raise ValueError("validate reviews discovery inputs; use predict with a saved model for inference")
    data, search = cfg["data"], cfg["search"]
    if not data["local_path"]:
        raise ValueError("data.local_path must point to the input CSV")
    # Reuse the exact preparation path, including unit/domain and group checks.
    # Final values are checked for integrity, never scored or shown in the review.
    with TemporaryDirectory(prefix="symbolic-input-review-") as temporary:
        refs = prepare(cfg, Path(temporary) / "prepared")
        prepared = read_json(refs["run"])
        engines = {}
        for engine in search["engines"]:
            module = get_engine(engine)
            capability = module.capabilities()
            runtime = search["runtimes"].get(engine, {})
            asset = cfg["model_assets"].get(engine, {})
            job = discovery_job(refs, prepared, engine, search["seeds"][0])
            issues = engine_input_issues(job, capability)
            adapter_issues, asset_issues = [], []
            assets_dir = None
            asset_info = None
            manifest = job["asset_manifest"]
            if manifest:
                try:
                    assets_dir, asset_info = verify_assets(manifest, capability.get("asset_engine", engine))
                except (ValueError, OSError) as exc:
                    asset_issues.append({"status": "failed", "reason": str(exc)})
            validation = "not_available"
            validator = getattr(module, "validate_request", None)
            if callable(validator):
                try:
                    # Exactly the worker's transformed coordinates, limits and
                    # TRAIN arrays; Selection and Final never reach an adapter.
                    request = discovery_request(job)
                    request.operators = [op for op in request.operators if op in capability["operators"]]
                    validator(request, DiscoveryContext(Path(temporary), job["device"], assets_dir, {
                        # Use the same frozen source/working geometry as _worker.
                        # Empty provenance can falsely validate or reject mapped input.
                        "public_units": job.get("public_units", {}),
                        "unit_coordinates": job.get("unit_coordinates"),
                        "input_geometry": job.get("input_geometry"),
                    }))
                    validation = "passed"
                except (ValueError, TypeError, NotImplementedError) as exc:
                    validation = "blocked"
                    adapter_issues.append({
                        "status": "unsupported" if isinstance(exc, NotImplementedError) else "failed",
                        "reason": str(exc),
                    })
            engines[engine] = {
                "declared_input_issues": issues,
                "adapter_input_issues": adapter_issues,
                "adapter_validation": validation,
                "asset_input_issues": asset_issues,
                "asset_verification": {
                    "status": "blocked" if asset_issues else "passed" if asset_info else "not_requested",
                    "manifest_sha256": asset_info["manifest_sha256"] if asset_info else None,
                    "files_verified": len(asset_info["files"]) if asset_info else 0,
                },
                "options": search["engine_options"].get(engine, {}),
                "runtime": runtime,
                "assets": asset,
                "effective_operator_subset": [op for op in search["operators"] if op in capability["operators"]],
                "capabilities": capability,
                "constraint_application": constraint_application(job, capability),
                "response_mode": search["response_mode"],
            }
    columns = prepared["unit_coordinates"]["columns"]
    return {
        "status": "blocked" if any(
            item["declared_input_issues"] or item["adapter_input_issues"] or item["asset_input_issues"]
            for item in engines.values()
        ) else "inputs_valid",
        "package_identity": prepared["package_identity"],
        "path_base": str(Path.cwd()),
        "data_file": str(Path(data["local_path"]).absolute()),
        "identity_column": "sample_id",
        "group_column": cfg["split"]["group_column"],
        "feature_columns": data["feature_columns"],
        "target_column": data["target_column"],
        "unit_coordinates": columns,
        "working_domains": data["domain"],
        "working_variables": prepared["variables"],
        "target": prepared["target"],
        "split": cfg["split"],
        "partitions": prepared["partitions"],
        "search_budget": {
            key: search[key]
            for key in (
                "seeds",
                "max_wall_seconds",
                "candidate_pool_policy",
                "max_export_candidates",
                "allow_partial_engines",
            )
        },
        "requirements": prepared["requirements"],
        "engines": engines,
        "not_checked": [
            "Only adapter-provided static input checks ran. An extension without validate_request is explicitly not_available.",
            "Native libraries, runtime startup, GPU availability and checkpoint deserialization were not tested; verified file hashes and input checks are not native readiness evidence.",
            "No search, fitting, predictions, Selection metrics or Final metrics were performed. No input files were changed.",
        ],
    }


def format_input_review(review):
    lines = [
        f"Input review: {review['status']}",
        f"CSV: {review['data_file']}",
        f"Relative path base: {review['path_base']}",
    ]
    lines.append(f"Identity: sample_id; group: {review['group_column'] or '(not declared)'}")
    lines.append("Columns (CSV unit -> working unit, working = scale * CSV + offset):")
    for column in [*review["feature_columns"], review["target_column"]]:
        item = review["unit_coordinates"][column]
        role = "target" if column == review["target_column"] else "input"
        lines.append(
            f"  {role} {column}: {item['source_unit'] or 'unknown'} -> {item['problem_unit'] or 'unknown'}; scale={item['scale']}, offset={item['offset']}; role={item['coordinate_role'] or 'unknown'}"
        )
        if column in review["working_domains"]:
            lines.append(
                f"    working domain: {review['working_domains'][column]} {item['problem_unit'] or '(unknown unit)'}"
            )
    lines.append("Execution variables: " + ", ".join(f"{v['id']}={v['column']}" for v in review["working_variables"]))
    lines.append(
        "Partitions: "
        + "; ".join(
            f"{name}: {item['n_rows']} rows / {item['n_groups']} groups" for name, item in review["partitions"].items()
        )
    )
    lines.append(f"Budget: {review['search_budget']}")
    requirements = review["requirements"]
    responses = requirements["required_responses"]
    lines.append(
        f"Requirements: max_nodes={requirements['max_nodes']}, require_units={requirements['require_units']}, unit_search={requirements.get('unit_search', False)}, template={'present' if requirements['equation_template'] else 'none'}, fixed_constants={list(requirements['fixed_constants'])}"
    )
    lines.append(
        f"Response conditions: {len(responses['points'])} probes, bounds={responses['bounds']} {review['target']['unit'] or '(unknown unit)'} (working target unit), {len(responses['checks'])} derivative checks (sampled, not global guarantees)"
    )
    lines.append("Selected engines (static input checks only; native readiness is not certified):")
    for name, item in review["engines"].items():
        lines.append(
            f"  {name}: declared options={item['options'] or '(adapter defaults)'}; runtime={item['runtime'] or '(current Python, cpu)'}; assets={item['assets'] or '(none)'}"
        )
        for condition, application in item["constraint_application"].items():
            lines.append(f"    {condition}: {application}")
        lines.append(f"    response constraints: {item['response_mode']}; search support={item['capabilities'].get('search_constraints', [])}")
        lines.append(f"    effective allowed operators: {', '.join(item['effective_operator_subset'])}")
        lines.append(
            f"    adapter input checks: {item['adapter_validation']}; asset integrity: {item['asset_verification']['status']}"
        )
        for issue in [*item["declared_input_issues"], *item["adapter_input_issues"], *item["asset_input_issues"]]:
            lines.append(f"    BLOCKED: {issue['reason']}")
    lines.extend(["Scope:", *(f"  {note}" for note in review["not_checked"])])
    lines.append("Use --json for the complete canonical requirements and capability declarations.")
    return "\n".join(lines)
