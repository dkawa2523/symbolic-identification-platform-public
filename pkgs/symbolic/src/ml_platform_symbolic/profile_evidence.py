"""Run-scoped profile evidence. A completed call is not a scientific readiness verdict."""

from pathlib import Path

from .data import digest, read_json
from .problem import problem_config, requirement_reference


def profile_evidence(summary_path):
    path = Path(summary_path)
    summary = read_json(path)
    job = read_json(path.with_name("job.json")) if path.with_name("job.json").is_file() else {}
    if summary.get("job_sha256") and digest(path.with_name("job.json")) != summary["job_sha256"]:
        raise ValueError("Resolved native job changed after execution")
    return {
        "runtime_image_id": summary.get("runtime_image_id"),
        "native_effective_work": {
            k: v
            for k, v in summary.items()
            if k
            in {
                "smc_observation",
                "particles",
                "mcmc_moves_per_step",
                "native_candidates",
                "search_seconds",
                "native_search_operators",
                "tempering_final",
                "top_structure_mass",
                "distinct_native_command_stacks",
                "native_work",
                "effective_optimizer_options",
                "optimizer_work",
                "checkpoint_seconds",
                "termination_reason",
                "work_scope",
            }
        },
        "engine": summary["engine"],
        "seed": summary["seed"],
        "summary_path": str(path.resolve()),
        "summary_sha256": digest(path),
        "profile": {key: job.get(key) for key in ("options", "operators", "limits", "device", "max_wall_seconds")},
        "native_execution_status": summary.get("native_execution_status", "unverified historical execution"),
        "status": summary["status"],
        "reason": summary.get("reason"),
        "wall_seconds": summary.get("wall_seconds"),
        "hard_wall_budget": job.get("limits", {}).get("max_wall_seconds"),
        "exported_candidates": summary.get("exported_candidates"),
        "retained_candidates": summary.get("selection_candidates"),
        "candidate_pool": summary.get("candidate_pool"),
        "constraint_application": summary.get("constraint_application"),
        "response_enforcement": summary.get("response_enforcement", {"effective": "historical_not_recorded"}),
        "search_response_evidence": summary.get("response_search", summary.get("response_fit")),
        "constraint_support": summary.get("constraint_support", "native enforcement not established by this artifact"),
        "native_artifacts": summary.get("native_artifacts", {}),
        "assets": summary.get("assets"),
        "runtime": summary.get("python"),
        "scope": "evidence for this resolved profile and run; not all-library readiness or adequacy on other problems",
    }


def constraint_application(record, checks, context):
    origin = record.get("origin", {})
    native = origin.get("summary", {}).get("constraint_support") or origin.get("summary", {}).get(
        "capability_declaration", {}
    ).get("constraints", {})
    return {
        "requirements_ref": requirement_reference(context),
        "requested": problem_config(context)["constraints"],
        "search_application": origin.get("summary", {}).get("constraint_application", "historical run: not recorded"),
        "additional_evaluation_conditions": context.get("constraint_application", {}).get("additional", {}),
        "evaluation_scope": "Selection-stage checks; additions do not establish independent final validation if used to choose candidates",
        "native_declaration": native or "not established; do not infer enforcement from a common pass",
        "evaluation": checks,
        "meaning": "common checks are postchecks on the retained output; sampled passes are not global enforcement",
    }

