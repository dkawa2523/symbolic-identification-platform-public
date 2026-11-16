"""One subprocess per engine/seed, with a hard wall limit and validated completed-iteration progress on interruption."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..assets import verify_assets
from ..candidate_pool import allocate_candidates
from ..constraints import check_constraints
from ..coordinates import execution_constraints, execution_region
from ..data import digest, load_partition, read_json, write_json
from ..domains import validation_constraints, within_bounds
from ..expressions import evaluate, inspect_equation
from ..metrics import prediction_metrics
from ..problem import problem_config, requirement_reference
from ..response_search import constraint_application, requested_response_constraints, response_enforcement
from ..worker_process import stop_worker, worker_process
from .base import DiscoveryContext, DiscoveryRequest
from .progress import load_progress, save_progress
from .registry import get_engine


def engine_input_issues(job, capabilities):
    """Check declared adapter compatibility without starting native code or loading assets."""
    issues = []
    if job["limits"].get("unit_search", False):
        support = capabilities.get("unit_search")
        reason = None
        if support is None:
            reason = "Requested unit_search is unsupported; post-check-only units do not satisfy search-time use"
        elif job["limits"].get("equation_template") is not None and not support.get("templates", False):
            reason = "Requested unit_search does not support this equation template"
        else:
            from ..units import unit_dimensions
            if any(unit_dimensions(item.get("unit")) is None for item in [*job["variables"], job["target"]]):
                reason = "Requested unit_search needs known input and target dimensions; unknown is not dimensionless"
            for key, allowed in support.get("choices", {}).items():
                value = job["options"].get(key)
                if not any(type(value) is type(item) and value == item for item in allowed):
                    reason = f"Requested unit_search conflicts with {key}={value!r}; expected {allowed}"
        if reason:
            issues.append({"status": "unsupported", "reason": reason + "; no native execution or fallback"})
    requested = requested_response_constraints(job["limits"])
    if job["limits"].get("response_mode", "search") == "search":
        missing = set(requested) - set(capabilities.get("search_constraints", []))
        if missing:
            issues.append({"status": "unsupported", "reason":
                f"Constraints are not used during this adapter's search: {sorted(missing)}. "
                "No filter-only fallback; use response_mode=filter only for an explicit control experiment."})
    try:
        response_enforcement(job["limits"], capabilities)
    except (ValueError, NotImplementedError) as error:
        issues.append({"status": "unsupported", "reason": str(error)})
    template = job["limits"].get("equation_template")
    if template is not None and not capabilities.get("equation_template", False):
        issues.append(
            {
                "status": "unsupported",
                "reason": "Required equation template is unsupported by this adapter; no native execution or unconstrained fallback",
            }
        )
    device = job.get("device", "cpu")
    if device.split(":")[0] not in capabilities.get("devices", ["cpu"]):
        issues.append({"status": "failed", "reason": f"Unsupported device {device} for {job['engine']}"})
    if not set(job["operators"]) & set(capabilities["operators"]):
        issues.append({"status": "failed", "reason": "Engine supports none of the allowed operators"})
    fixed_support = capabilities.get("fixed_constants")
    if job.get("fixed_constants") and not (
        fixed_support is True or (fixed_support == "template_only" and template is not None)
    ):
        issues.append(
            {
                "status": "unsupported",
                "reason": "Required fixed constants are unsupported by this adapter; no native execution",
            }
        )
    if job.get("asset_manifest") and not capabilities["requires_assets"]:
        issues.append({"status": "failed", "reason": "This engine does not consume model assets"})
    if capabilities["requires_assets"] and not job.get("asset_manifest"):
        issues.append({"status": "failed", "reason": "This engine requires a fixed local model_assets manifest"})
    return issues


def discovery_request(job):
    """Build the adapter request from Train only; shared by review and execution."""
    X, y, ids = load_partition(job["train"])
    return DiscoveryRequest(
        X,
        y,
        ids,
        job["variables"],
        job["target"],
        job["operators"],
        job["limits"],
        job["seed"],
        job["options"],
        job.get("fixed_constants", {}),
        domain=job.get("execution_domain"),
        equation_template=job["limits"].get("equation_template"),
    )


def _worker(job_path, *, recover_pysr=False):
    job = read_json(job_path)
    from ..domains import initialize_runtime_extensions

    start = time.monotonic()
    summary_path = Path(job_path).with_name("pysr_recovery_status.json" if recover_pysr else "engine_summary.json")
    try:
        initialize_runtime_extensions(job.get("runtime_extensions", []))
        request = discovery_request(job)
        module = get_engine(job["engine"])
        capabilities = module.capabilities()
        issues = engine_input_issues(job, capabilities)
        if issues:
            write_json(
                summary_path,
                {**issues[0], "native_execution_status": "not_started", "engine": job["engine"], "seed": job["seed"]},
            )
            return
        device = job.get("device", "cpu")
        requested_operators = request.operators
        request.operators = [op for op in requested_operators if op in capabilities["operators"]]
        asset_directory, asset_info = None, None
        if job.get("asset_manifest"):
            asset_directory, asset_info = verify_assets(job["asset_manifest"], capabilities.get("asset_engine", job["engine"]))
        context = DiscoveryContext(
            Path(job_path).parent,
            device,
            asset_directory,
            {
                "prepared_sha256": job.get("prepared_sha256"),
                "split_seed": job.get("split_seed"),
                "public_units": job.get("public_units", {}),
                "unit_coordinates": job.get("unit_coordinates"),
                "input_geometry": job.get("input_geometry"),
                "runtime_image_id": os.environ.get("SYMBOLIC_CONTAINER_IMAGE_ID"),
                "assets": asset_info,
            },
        )
        context.deadline = (
            start + request.limits["max_wall_seconds"] - min(5.0, request.limits["max_wall_seconds"] * 0.1)
        )
        context.checkpoint = lambda result: save_progress(job_path, result, context.provenance)
        if recover_pysr:
            if job["engine"] != "pysr":
                raise ValueError("CSV recovery is specific to PySR")
            result = module.discover(request, context, recover=True)
            if result.candidates:
                context.checkpoint(result)
            summary = {"status": "partial" if result.candidates else "failed",
                       "recovered_candidates": len(result.candidates), "details": result.summary}
        else:
            result = (
                module.discover(request, context, name=job["engine"])
                if job["engine"] in {"mean", "linear", "dictionary"}
                else module.discover(request, context)
            )
            summary = _persist_result(result, job_path, job, request, context, asset_info, requested_operators)
    except Exception as exc:
        traceback.print_exc()
        summary = {
            "status": "unsupported" if isinstance(exc, NotImplementedError) else "failed",
            "native_execution_status": "not_started" if isinstance(exc, NotImplementedError) else "failed",
            "reason": _failure_reason(exc),
            "engine": job["engine"],
            "seed": job["seed"],
        }
    summary["wall_seconds"] = time.monotonic() - start
    write_json(summary_path, summary)


def _persist_result(result, job_path, job, request, context, asset_info, requested_operators):
    summary_path = Path(job_path).with_name("engine_summary.json")
    X, y = request.X, request.y
    module = get_engine(job["engine"])
    records = [asdict(c) for c in result.candidates]
    policy = job["candidate_pool_policy"]
    path = Path(job_path).with_name("candidates.jsonl")
    archive = path if policy == "all" else path.with_name("portable_candidates.jsonl")
    if policy != "all":
        archive.write_text("".join(json.dumps(c, allow_nan=False) + "\n" for c in records), encoding="utf-8")
    # Apply the SAME required task checks before allocating the finite
    # comparison pool. An invalid low-loss formula must not crowd out every
    # valid one. No Selection/Final data or coefficient refit is used here;
    # compare still owns final eligibility and all selection decisions.
    constraints = validation_constraints(
        {"constraints": request.limits, "split": {"seed": job.get("split_seed", request.seed)}},
        request.variables,
        request.target,
        X,
        domain=request.domain,
    )
    qualities = {}

    def rank(candidate):
        equation = candidate["equation"]
        prediction = evaluate(equation, X, request.variables)
        metrics = prediction_metrics(y, prediction)
        checks = check_constraints(equation, request.variables, request.target, requested_operators, constraints, context={**(context.provenance or {}), "variables": request.variables, "target": request.target})
        eligible = (
            all(c["status"] == "pass" for c in checks if c["required"])
            and np.isfinite(prediction).all()
            and within_bounds(prediction, request.target.get("domain")).all()
        )
        stats = inspect_equation(equation, request.variables)
        qualities[candidate["candidate_id"]] = {
            "train_checks_pass": bool(eligible),
            "train_rmse": metrics["rmse"] if metrics["rmse"] is not None else float("inf"),
            "nodes": next(check["count"] for check in checks if check["name"] == "max_nodes"),
            "working_nodes": stats["n_nodes"],
            "free_parameters": stats["n_parameters"],
        }

    if policy != "all":
        for record in records:
            rank(record)
    records = allocate_candidates(records, qualities, job["max_export_candidates"], policy)
    path.write_text("".join(json.dumps(c, allow_nan=False) + "\n" for c in records), encoding="utf-8")
    summary = {
        **result.summary,
        "assets": asset_info,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "device": context.device,
        "allowed_operators": requested_operators,
        "native_operators": request.operators,
        "unavailable_search_operators": sorted(set(requested_operators) - set(request.operators)),
        "operator_policy": "unavailable_search_operators is the requested/export-capability difference, not decoder reachability; generation enforcement and context-dependent routes are specified by constraint_support and native metadata",
        "status": result.status,
        "native_execution_status": result.summary.get("native_execution_status", "completed"),
        "native_artifacts": {k: str(v) for k, v in result.native_artifacts.items()},
        "native_artifact_sha256": {k: digest(v) for k, v in result.native_artifacts.items()},
        "portable_status": result.summary.get("portable_status", "candidates" if records else "unavailable"),
        "fixed_constants_native_support": module.capabilities().get("fixed_constants", "unsupported"),
        "exported_candidates": len(result.candidates),
        "selection_candidates": len(records),
        "prelimit_constraint_eligible": sum(q["train_checks_pass"] for q in qualities.values()) if qualities else None,
        "portable_candidates": str(archive),
        "portable_candidates_sha256": digest(archive),
        "candidate_pool": {
            "policy": policy,
            "emitted": len(result.candidates),
            "retained": len(records),
            "limit": None if policy == "all" else job["max_export_candidates"],
            "discarded_from_comparison_ids": sorted(
                {c.candidate_id for c in result.candidates} - {r["candidate_id"] for r in records}
            ),
            "archive_scope": "all faithfully exported native candidates, not all internal search states",
            "quality_scope": "Train/shared probes only when explicit bounded allocation is requested; compare owns final eligibility",
        },
        "truncation": "none; all faithfully emitted candidates reach common evaluation" if policy == "all" else policy,
        "engine": job["engine"],
        "seed": job["seed"],
    }
    write_json(summary_path, summary)
    return summary


def _recover_completed(job_path, job, directory, original, termination, python=None):
    """One recovery boundary for a timeout, crash or explicit native failure."""
    completed = load_progress(job_path, job["variables"])
    if completed is None and job["engine"] == "pysr" and (directory / "pysr_run.json").is_file():
        # The caller has stopped/reaped the writer before this boundary. Native
        # imports run in their own bounded process, never in the orchestrator.
        recovery_limit = 60
        with (directory / "pysr_recovery.log").open("w", encoding="utf-8") as log:
            with worker_process([str(Path(python or sys.executable).absolute()), "-m",
                                 f"{__package__}.worker", str(Path(job_path).resolve()), "--recover-pysr"],
                                stdout=log, stderr=subprocess.STDOUT) as recovery:
                try:
                    recovery.wait(timeout=recovery_limit)
                except subprocess.TimeoutExpired:
                    stop_worker(recovery)
                    original["native_recovery"] = {"status": "timeout", "wall_limit": recovery_limit}
                    return None
        recovery_status = directory / "pysr_recovery_status.json"
        original["native_recovery"] = read_json(recovery_status) if recovery_status.is_file() else {
            "status": "failed", "reason": f"Recovery worker exited {recovery.returncode}"}
        completed = load_progress(job_path, job["variables"])
    if completed is None:
        return None
    native, native_provenance, checkpoint_sha256 = completed
    request = discovery_request(job)
    request.operators = [op for op in job["operators"] if op in get_engine(job["engine"]).capabilities()["operators"]]
    context = DiscoveryContext(directory, job["device"], provenance=native_provenance)
    original_path = write_json(directory / "interrupted_execution.json", original)
    summary = _persist_result(
        native, job_path, job, request, context, native_provenance.get("assets"), job["operators"]
    )
    summary.update(
        native_execution_status="interrupted" if termination == "hard_wall_limit" else "failed",
        termination_reason=termination,
        reason="Validated saved native hypotheses; search did not complete: "
        + original.get("reason", "unknown exit"),
        completed_progress_sha256=checkpoint_sha256,
        original_execution={"artifact": str(original_path), "sha256": digest(original_path), **original},
        checkpoint_scope="data-only hypotheses; no optimizer/RNG resume guarantee",
        work_scope="Counts stop at the validated snapshot; in-flight work and optimizer state are not reconstructed",
    )
    return summary


def discovery_job(refs, provenance, engine, seed):
    """One canonical prepared job for input review and the actual worker."""
    cfg = problem_config(provenance)
    get_engine(engine)  # Configuration error, not a partial engine failure.
    runtime = cfg["search"].get("runtimes", {}).get(engine, {})
    asset = cfg.get("model_assets", {}).get(engine, {})
    from ..domains import runtime_extensions

    job = {
        "runtime_extensions": runtime_extensions(),
        "prepared_sha256": digest(refs["run"]),
        "split_seed": cfg["split"]["seed"],
        "public_units": cfg["data"].get("source_units", {}),
        "unit_coordinates": provenance.get("unit_coordinates"),
        "input_geometry": provenance.get("input_geometry"),
        "execution_domain": execution_region(provenance.get("input_geometry"), provenance["variables"]),
        "fixed_constants": cfg["data"]["fixed_constants"],
        "requirements_ref": requirement_reference(provenance),
        "train": refs["train"],
        "variables": provenance["variables"],
        "target": provenance["target"],
        "operators": cfg["search"]["operators"],
        "limits": {
            **execution_constraints(cfg["constraints"], provenance.get("input_geometry"), provenance["variables"]),
            "max_wall_seconds": cfg["search"]["max_wall_seconds"],
            "response_mode": cfg["search"].get("response_mode", "filter"),
        },
        "seed": seed,
        "engine": engine,
        "options": cfg["search"]["engine_options"].get(engine, {}),
        "max_export_candidates": cfg["search"]["max_export_candidates"],
        "candidate_pool_policy": cfg["search"]["candidate_pool_policy"],
        "device": runtime.get("device", "cpu"),
        "asset_manifest": str(Path(asset["manifest_file"]).resolve()) if asset else None,
    }
    if job["limits"].get("unit_search", False):
        # Derive native settings only for a new explicit common request. Never
        # alter a stored request's options, or silently override explicit choices.
        support = get_engine(engine).capabilities().get("unit_search", {})
        job["options"] = {**support.get("defaults", {}), **job["options"]}
    # Add domain validity points without widening explicitly declared response points.
    # Selection/Final never enter search. Domain-only jobs remain unchanged.
    if requested_response_constraints(job["limits"]):
        train_X, _, _ = load_partition(refs["train"])
        job["limits"] = validation_constraints(
            {"constraints": job["limits"], "split": cfg["split"]},
            job["variables"], job["target"], train_X, domain=job["execution_domain"],
        )
    return job


def run_discovery(refs, provenance, engine, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    cfg = problem_config(provenance)
    if engine not in cfg["search"]["engines"] or seed not in cfg["search"]["seeds"]:
        raise ValueError("Discovery engine/seed is not in the prepared plan")
    if digest(refs["train"]) != provenance["partition_hashes"]["train"]:
        raise ValueError("Prepared train partition checksum mismatch")
    runtime = cfg["search"].get("runtimes", {}).get(engine, {})
    job = discovery_job(refs, provenance, engine, seed)
    job_path = write_json(directory / "job.json", job)
    start = time.monotonic()
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        with worker_process(  # noqa: S603 -- fixed worker module and prepared local JSON, shell=False.
            [
                # Resolving symlinks bypasses a POSIX venv and loses its packages.
                str(Path(runtime.get("python", sys.executable)).absolute()),
                "-m",
                f"{__package__}.worker",
                str(job_path.resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process:
            try:
                process.wait(timeout=cfg["search"]["max_wall_seconds"])
            except subprocess.TimeoutExpired:
                stop_worker(process)
                recovery_started = time.monotonic()
                summary = {
                    "status": "timeout",
                    "native_execution_status": "interrupted",
                    "reason": "Hard wall limit; no complete native progress snapshot",
                    "engine": engine,
                    "seed": seed,
                }
                try:
                    recovered = _recover_completed(job_path, job, directory, summary, "hard_wall_limit", runtime.get("python"))
                    if recovered is not None:
                        summary = recovered
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    summary["progress_error"] = _failure_reason(exc)
                summary.update(
                    wall_seconds=recovery_started - start, recovery_seconds=time.monotonic() - recovery_started
                )
                write_json(directory / "engine_summary.json", summary)
    summary_path = directory / "engine_summary.json"
    if not summary_path.is_file():
        write_json(
            summary_path,
            {
                "status": "failed",
                "reason": f"Worker exited {process.returncode}; see worker.log",
                "engine": engine,
                "seed": seed,
            },
        )
    summary = read_json(summary_path)
    if summary["status"] == "failed" and summary.get("native_execution_status") != "completed":
        recovery_started = time.monotonic()
        try:
            recovered = _recover_completed(job_path, job, directory, summary, "native_failure", runtime.get("python"))
            if recovered is not None:
                summary = recovered
                summary["recovery_seconds"] = time.monotonic() - recovery_started
        except (ValueError, KeyError, TypeError, OSError) as exc:
            summary["progress_error"] = _failure_reason(exc)
    summary["runtime_image_id"] = os.environ.get("SYMBOLIC_CONTAINER_IMAGE_ID")
    if summary["status"] == "failed" and summary.get("native_execution_status") == "completed":
        summary.setdefault("reason", "Native execution completed but no faithfully portable candidates survived export")
    summary["prepared_sha256"] = job["prepared_sha256"]
    summary["job_sha256"] = digest(job_path)
    summary["requested_constraints"] = job["limits"]
    summary["requirements_ref"] = job["requirements_ref"]
    summary["resolved_native_profile"] = {
        key: job[key] for key in ("options", "operators", "device", "candidate_pool_policy")
    }
    summary["response_mode"] = job["limits"].get("response_mode", "filter")
    summary["capability_declaration"] = get_engine(engine).capabilities()
    summary["constraint_application"] = constraint_application(job, summary["capability_declaration"])
    summary["response_enforcement"] = summary["constraint_application"].get("response_enforcement", {"effective": "not_requested"})
    summary["readiness_scope"] = (
        "this profile/run only; execution, fidelity, common admissibility and scientific adequacy are distinct"
    )
    if summary["status"] in {"ok", "partial"}:
        summary["candidates_sha256"] = digest(directory / "candidates.jsonl")
    summary["wall_seconds"] = time.monotonic() - start
    write_json(summary_path, summary)
    if summary["status"] != "ok" and not cfg["search"]["allow_partial_engines"]:
        raise RuntimeError(
            f"{engine}/{seed} {summary['status']}: {summary.get('reason', 'no portable candidates')}; {summary_path}"
        )
    result = {"summary": str(summary_path.resolve()), "candidates": str((directory / "candidates.jsonl").resolve())}
    archive = summary.get("portable_candidates")
    if archive is not None and Path(archive).is_file():
        result["all_candidates"] = str(Path(archive).resolve())
    return result


def _failure_reason(exc):
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:
        return f"{type(exc).__name__}: exception message unavailable (native runtime); inspect worker.log"
