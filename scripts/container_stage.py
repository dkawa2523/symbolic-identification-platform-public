"""Thin container entrypoint: call the same domain stages across native runtime families."""

import argparse
import os
from pathlib import Path

import pandas as pd
from ml_platform_symbolic.config import load_config, resolve_config
from ml_platform_symbolic.data import prepare, read_json, write_json
from ml_platform_symbolic.engines.runner import run_discovery
from ml_platform_symbolic.problem import problem_config


def discovery_records(root, cfg):
    discoveries, summaries = [], {}
    for engine in cfg["search"]["engines"]:
        summaries[engine] = []
        for seed in cfg["search"]["seeds"]:
            directory = root / "discover" / engine / str(seed)
            summary_file = directory / "engine_summary.json"
            if summary_file.exists():
                record = read_json(summary_file)
                refs = {"summary": str(summary_file), "candidates": str(directory / "candidates.jsonl")}
                discoveries.append(refs)
            else:
                record = {"seed": seed, "status": "not_run", "reason": "Discovery did not complete"}
            summaries[engine].append({**record, "summary_path": str(summary_file)})
    return discoveries, summaries


def summarize_discovery(root, state, provenance):
    """Persist returned work and failures without scoring, selecting or reading Final."""
    _, summaries = discovery_records(root, problem_config(provenance))
    rows = [
        {
            "engine": engine,
            **{
                key: record.get(key)
                for key in (
                    "seed",
                    "status",
                    "native_execution_status",
                    "reason",
                    "wall_seconds",
                    "exported_candidates",
                    "summary_path",
                )
            },
        }
        for engine, records in summaries.items()
        for record in records
    ]
    pd.DataFrame(rows).to_csv(root / "discovery_results.csv", index=False)
    write_json(
        root / "discovery_results.json", {"results": rows, "selection_performed": False, "final_accessed": False}
    )


def compare_selection(root, state, provenance):
    """Reuse public comparison/output APIs; no adoption, refit or Final access."""
    from ml_platform_symbolic import compare_equations, create_study, export_outputs, select_representatives
    from ml_platform_symbolic.equation_evaluation import load_study

    if root.name != "calculation":
        raise ValueError("Comparison expects a calculation directory beside its presentation files")
    discoveries, summaries = discovery_records(root, problem_config(provenance))
    missing = [f"{engine}/{row['seed']}" for engine, rows in summaries.items()
               for row in rows if row["status"] == "not_run"]
    if missing:
        raise ValueError("Cannot compare missing discoveries: " + ", ".join(missing))
    study = create_study(
        {"prepared_run": state["prepare"]["run"],
         "sources": [{"discovery": ref["summary"]} for ref in discoveries],
         "purpose": "local_prediction",
         "question": "Compare equations on fixed Selection rows, not independent Final evidence."},
        run_dir=root / "study")
    _, records, rows, _ = load_study(study)
    representatives = select_representatives(records, rows)
    ids = [record["id"] for record, _ in representatives.values()]
    comparison = compare_equations(study, run_dir=root / "comparison", pair_ids=ids)
    output = export_outputs(comparison, output_dir=root.parent, representative_ids=ids)
    write_json(root / "comparison_summary.json", {
        "partition": "Selection", "final_accessed": False, "adoption_performed": False,
        "rule": "Minimum eligible finite Selection RMSE per method/seed; exact ties by ID",
        "representative_ids": ids, "output": str(output),
        "all_candidates": str(study),
        "discovery_complete": all(row["status"] == "ok" for rows in summaries.values() for row in rows),
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "discover", "summarize", "compare"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config")
    parser.add_argument("--engine")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.stage == "prepare":
        if root.exists() and any(p.name != "container_execution.json" for p in root.iterdir()):
            raise ValueError("Container run directory must be new/empty")
        cfg = resolve_config(load_config(args.config))
        if cfg["task"] != "symbolic_pipeline" or any("python" in r for r in cfg["search"]["runtimes"].values()):
            raise ValueError("Container task mode accepts per-engine device; Python is chosen by its image")
        refs = prepare(cfg, root / "prepare")
        write_json(root / "state.json", {"prepare": refs, "search": cfg["search"]})
        return
    state = read_json(root / "state.json")
    provenance = read_json(state["prepare"]["run"])
    if args.stage == "discover":
        directory = root / "discover" / args.engine / str(args.seed)
        refs = run_discovery(state["prepare"], provenance, args.engine, args.seed, directory)
        summary = read_json(refs["summary"])
        summary["runtime_image_id"] = os.environ.get("SYMBOLIC_CONTAINER_IMAGE_ID")
        write_json(refs["summary"], summary)
        write_json(directory / "refs.json", refs)
    elif args.stage == "summarize":
        summarize_discovery(root, state, provenance)
    elif args.stage == "compare":
        compare_selection(root, state, provenance)


if __name__ == "__main__":
    main()
