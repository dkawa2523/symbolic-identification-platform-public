"""Render saved identification scores only: python scripts/summarize_identification.py SCORES OUTPUT_DIR."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric_summary(rows, key):
    draws = defaultdict(list)
    missing = invalid = 0
    for row in rows:
        draw = str(row.get("data_seed")) if row.get("data_seed") is not None else row["case"]
        draws[draw]  # Retain draws with no usable metric.
        value = row.get(key)
        if value is None:
            missing += 1
        elif not finite(value):
            invalid += 1
        else:
            draws[draw].append(value)
    medians = {key: median(values) if values else None for key, values in draws.items()}
    values = [value for values in draws.values() for value in values]
    centers = [value for value in medians.values() if value is not None]
    return {
        "median_of_data_draw_solver_medians": median(centers) if centers else None,
        "worst_individual_run": (min(values) if key.endswith((".r2", "coverage")) else max(values))
        if values else None,
        "finite_count": len(values), "attempt_count": len(rows), "missing_count": missing,
        "invalid_count": invalid, "data_draw_solver_medians": medians,
    }


def summarize(evidence):
    rows = [flatten(row) for row in evidence["results"]]
    groups = defaultdict(list)
    for row in rows:
        case = row["case"]
        group = "proton" if case == "proton_form_factor" else (
            "confounded" if case.startswith("saturation_confounded_") else "independent"
        )
        if group == "independent" and not case.startswith("saturation_independent_"):
            raise ValueError(f"Unknown case: {case}")
        groups[group, row["engine"]].append(row)
    summaries = []
    for (group, engine), attempts in groups.items():
        keys = {"selection_rmse", "selection_mae", "selection_r2", "test_metrics.rmse",
                "test_metrics.mae", "test_metrics.r2", "coverage"}
        if group != "proton":
            keys.add("x_response_rmse")
        keys.update(key for row in attempts for key in row
                    if key.startswith("strata.") and key.endswith((".rmse", ".mae", ".r2", ".coverage")))
        draws = defaultdict(list)
        for row in attempts:
            draws[str(row.get("data_seed")) if group != "proton" else row["case"]].append(row.get("solver_seed"))
        metrics = {key: metric_summary(attempts, key) for key in sorted(keys)}
        coverages = [row[key] for row in attempts for key in row
                     if key.endswith("coverage") and finite(row[key])]
        summaries.append({
            "group": group, "engine": engine, "attempt_count": len(attempts),
            "eligible_attempt_count": sum(bool(row.get("selected_id")) and row.get("eligible_count", 0) > 0
                                          for row in attempts),
            "data_draws" if group != "proton" else "public_datasets": len(draws),
            "solver_seeds_by_dataset": dict(draws),
            "statuses": {key: dict(Counter(str(row.get(key)) for row in attempts))
                         for key in ("native_status", "native_execution_status", "score_status")},
            "min_observed_coverage": min(coverages) if coverages else None, "metrics": metrics,
        })
    return rows, {"declared_attempts": evidence["attempts"], "stored_attempts": len(rows),
                  "unrepresented_attempts": evidence["attempts"] - len(rows), "groups": summaries}


def render(summary):
    lines = ["# Frozen identification evidence", "", NOTES, "",
             f"Stored attempts: {summary['stored_attempts']}/{summary['declared_attempts']}; "
             f"unrepresented: {summary['unrepresented_attempts']}.", ""]
    for group in summary["groups"]:
        lines += [f"## {group['group']} / {group['engine']}",
                  f"Eligible choices: {group['eligible_attempt_count']}/{group['attempt_count']} attempts; "
                  f"minimum observed coverage (overall and strata): {group['min_observed_coverage']}.",
                  f"Dataset -> solver seeds: {json.dumps(group['solver_seeds_by_dataset'])}",
                  f"Preserved statuses: {json.dumps(group['statuses'])}", "",
                  "Metric | Nested median | Worst run | Finite/attempts | Missing | Invalid",
                  "--- | ---: | ---: | ---: | ---: | ---:"]
        for key, value in group["metrics"].items():
            if key.endswith("coverage") or key.endswith((".mae", "_mae", "_r2")):
                continue
            center, worst = value["median_of_data_draw_solver_medians"], value["worst_individual_run"]
            def fmt(x):
                return f"{x:.6g}" if x is not None else "missing"
            lines.append(f"{key} | {fmt(center)} | {fmt(worst)} | "
                         f"{value['finite_count']}/{value['attempt_count']} | "
                         f"{value['missing_count']} | {value['invalid_count']}")
        lines.append("")
    return "\n".join(lines)


NOTES = (
    "Reporting only; no prediction, refit, reselection or adoption/pass criterion. Engineering tolerance is absent. "
    "Synthetic arms have three paired data draws; parfam has three solver seeds nested within each draw, "
    "not nine independent datasets. Controls have one solver seed per draw. Summaries take the median of "
    "per-draw solver medians, retaining missing/invalid denominators; finite-only summaries are conditional "
    "on available evidence, not failure penalties. Worst means maximum error, minimum R2/coverage. "
    "Proton is one public dataset: its median describes solver variation only, not data uncertainty. "
    "Minimum observed coverage does not impute missing coverage. Selection MAE/R2 were not stored and remain missing. "
    "X-response error covers 101 declared interior finite-difference points at z=0.7, not global equivalence. "
    "All stored fields and reasons are in attempts.csv; all metric counts and per-draw medians are in summary.json."
)


def write_report(source, destination):
    raw = Path(source).read_bytes()
    rows, summary = summarize(json.loads(raw))
    summary.update(source_sha256=hashlib.sha256(raw).hexdigest(),
                   choices_sha256=json.loads(raw).get("choices_sha256"), interpretation=NOTES)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    fields = sorted({key for row in rows for key in row} | {"selection_mae", "selection_r2"})
    with (destination / "attempts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "summary.md").write_text(render(summary), encoding="utf-8")
    if __package__:
        from .benchmark_readout import render as render_html
    else:
        from benchmark_readout import render as render_html

    (destination / "evidence.json").write_bytes(raw)
    (destination / "report.html").write_text(
        render_html(json.loads(raw), source_name=Path(source).name, source_sha256=summary["source_sha256"]),
        encoding="utf-8",
    )
    print("Open report:", destination / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_report(args.scores, args.output_dir)

