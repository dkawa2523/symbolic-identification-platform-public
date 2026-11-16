"""Score frozen Selection representatives on an existing fixed test partition.

No discovery, refit, calibration, deployment choice or Final-based selection.
Study history must describe actual prior exposure; this entry never upgrades it.
"""
import argparse
import csv
from pathlib import Path

from ml_platform_symbolic import compare_equations, export_outputs, score_representatives
from ml_platform_symbolic.data import read_json, write_json
from ml_platform_symbolic.equation_evaluation import load_study


def main(study, output, *, evaluations=()):
    output.mkdir(parents=True, exist_ok=False)
    audit_dir = score_representatives(study, run_dir=output / "test")
    context, records, _, _ = load_study(study)
    audit = read_json(audit_dir / "audit.json") if audit_dir is not None else {
        "results": [], "final_evidence_role": "Not scored: no eligible Selection representative",
        "final_history": context.get("final_history", "unknown"),
        "history_scope": "Final was not accessed"}
    scored = {row["id"]: row for row in audit["results"]}
    comparison = compare_equations(study, run_dir=output / "comparison", pair_ids=list(scored), evaluations=evaluations)
    export_outputs(comparison, output_dir=output / "output", audit_dir=audit_dir,
                   representative_ids=list(scored), representatives_only=True)
    # Keep every attempted method/seed, including failures without a representative.
    attempts = {}
    for record in records:
        origin = record.get("origin", {})
        key = (origin.get("engine"), origin.get("seed"))
        source = origin.get("summary", {})
        attempts.setdefault(key, {"method": key[0], "seed": key[1], "status": "no_eligible_candidate",
                                  "run_status": source.get("status"), "wall_seconds": source.get("wall_seconds"),
                                  "reason": source.get("reason") or "No eligible finite Selection candidate"})
        if record["id"] in scored:
            result = scored[record["id"]]
            attempts[key] = {**attempts[key], "status": "scored", "reason": source.get("reason"),
                             "candidate_id": record["id"], "coverage": result.get("coverage"),
                             **(result.get("target_metrics") or {})}
    summary = {"partition": "Final" if audit_dir is not None else None, "role": audit["final_evidence_role"],
               "history": audit["final_history"], "history_scope": audit["history_scope"],
               "final_accessed": audit_dir is not None,
               "selection_changed": False, "refit": False, "attempts": list(attempts.values())}
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in summary["attempts"] for k in row}))
        writer.writeheader()
        writer.writerows(summary["attempts"])
    print(f"Test results: {output}; {summary['role']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", default=[],
                        help="Existing same-study interpretation result with fixed shared paths; repeatable")
    args = parser.parse_args()
    main(args.study, args.output, evaluations=args.evaluation)
