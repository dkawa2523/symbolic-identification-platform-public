"""Readable files and optional interactive reports from saved calculation evidence."""

import csv
import html
import json
import math
import tempfile
from copy import deepcopy
from pathlib import Path

from .report_data import build_report_view


def load_report_view(
    comparison_dir,
    *,
    decision_dir=None,
    representative_ids=None,
    prediction_interval=None,
    coverage=None,
    audit_dir=None,
    supplemental_comparison=None,
    display_units=None,
):
    """Read the existing evidence boundary, without loading predictor runtimes."""
    if (Path(comparison_dir) / "result_manifest.json").is_file():
        if any(
            value is not None
            for value in (
                decision_dir,
                representative_ids,
                prediction_interval,
                coverage,
                audit_dir,
                supplemental_comparison,
                display_units,
            )
        ):
            raise ValueError("A detached result keeps its frozen evidence; do not mix new decision/audit sources")
        from .portable_results import load_result

        return load_result(comparison_dir, load_predictors=False)["view"]
    # Presentation bundles are replayable without loading original data or runtimes.
    root = Path(comparison_dir).resolve()
    for index_name, report_name in (("metadata/output_index.json", "metadata/report.json"),
                                    ("details/output_index.json", "details/report.json"),
                                    ("output_index.json", "report.json")):
        if not (root / index_name).is_file():
            continue
        if any(value is not None for value in (decision_dir, representative_ids, prediction_interval,
                                               coverage, audit_dir, supplemental_comparison, display_units)):
            raise ValueError("A saved output keeps its frozen evidence; do not mix new decision/audit sources")
        from .data import digest, read_json

        index = read_json(root / index_name)
        if report_name not in index.get("files", {}):
            raise ValueError("Saved output index does not identify its report data")
        for name, expected in index["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
                raise ValueError(f"Saved output artifact changed or missing: {name}")
        return read_json(root / report_name)
    return build_report_view(
        comparison_dir,
        decision_dir=decision_dir,
        representative_ids=representative_ids,
        prediction_interval=prediction_interval,
        coverage=coverage,
        audit_dir=audit_dir,
        supplemental_comparison=supplemental_comparison,
        display_units=display_units,
    )


def render_report(comparison_dir, *, output, **view_options):
    output = Path(output)
    if output.exists():
        raise ValueError("Report output already exists; preserve earlier readout")
    return render_report_data(load_report_view(comparison_dir, **view_options), output=output)


def normalize_report_data(value):
    """JSON display convention only; validity/evidence remains explicitly stored."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: normalize_report_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_report_data(item) for item in value]
    return value


def render_report_data(view, *, output):
    """Render an already materialized view without opening data/model sources."""
    output = Path(output)
    if output.exists():
        raise ValueError("Report output already exists; preserve earlier readout")
    # Data cannot terminate the script element or inject markup.

    payload = (
        json.dumps(normalize_report_data(view), ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    result = template.replace("@@QUESTION@@", html.escape(view["question"])).replace("@@DATA@@", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    return output


def csv_cell(value):
    """Spreadsheet-safe display cell; exact original values stay in JSON."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and (value.startswith(("\t", "\r", "\n")) or
                                   value.lstrip().startswith(("=", "+", "-", "@"))):
        return "'" + value
    return value


def _write_csv(path, rows, fields):
    """Nested evidence stays JSON; blanks mean missing, never numeric zero."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(value) for key, value in row.items()})


def export_output_data(view, *, output_dir, figures=True, include_html=False, representatives_only=False):
    """Publish views beside an existing calculation, without changing its evidence.

    A calculation-only directory is allowed. Existing presentation files are never
    overwritten. All rendering finishes first; failed publication rolls back only
    the newly published files, leaving calculation records untouched.
    """
    if representatives_only:
        view = deepcopy(view)
        selected = set(view.get("display_ids", []))
        represented = {(c.get("method"), c.get("seed")) for c in view["candidates"] if c["id"] in selected}
        original_count = len(view["candidates"])
        view["candidates"] = [c for c in view["candidates"] if c["id"] in selected
                              or c["kind"] == "native_result"
                              or (not c.get("eligible") and (c.get("method"), c.get("seed")) not in represented)]
        retained = {c["id"] for c in view["candidates"]}
        view["pairs"] = [pair for pair in view.get("pairs", [])
                         if pair.get("left") in retained and pair.get("right") in retained]
        view["analysis_comparisons"] = [pair for pair in view.get("analysis_comparisons", [])
                                        if pair.get("left") in retained and pair.get("right") in retained]
        if view.get("audit", {}).get("rows"):
            view["audit"]["rows"] = {key: value for key, value in view["audit"]["rows"].items() if key in retained}
        view["export_scope"] = {"candidates": "Explicit display representatives and failed/unrepresented method rows",
                                "original_count": original_count, "retained_count": len(retained),
                                "omitted_candidates": "Unchanged in the source study; not re-evaluated or deleted"}
        view["display_rule"] = "Frozen display representatives only; failed/unrepresented methods retained. Full candidate pool remains in the source study."
    directory = Path(output_dir)
    def check_destination():
        if directory.exists() and (not directory.is_dir() or directory.is_symlink() or
                                   any(p.name not in {"calculation", "comparisons"} or
                                       not p.is_dir() or p.is_symlink() for p in directory.iterdir())):
            raise ValueError("Output directory already exists with result files; preserve earlier readout")
    check_destination()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".result-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "result"
        _write_output_data(view, output_dir=staged, figures=figures, include_html=include_html)
        check_destination()
        if not directory.exists():
            staged.rename(directory)
        else:
            published = []
            try:
                for path in staged.iterdir():
                    target = directory / path.name
                    if target.exists():
                        raise ValueError("Output file already exists; preserve earlier readout")
                    path.rename(target)
                    published.append(target)
            except BaseException:
                for path in reversed(published):
                    path.rename(staged / path.name)
                raise
    return directory


def _write_output_data(view, *, output_dir, figures=True, include_html=False):
    """Export a materialized readout to reusable files, never to a new model.

    Same view as HTML/detached results. Adding a renderer does not change the
    evaluator, comparison, decision or persisted predictor contract.
    """
    from .data import digest, write_json
    from .equation_output import equation_rows, latex_name, template_rows, template_lines

    directory = Path(output_dir)
    if directory.exists():
        raise ValueError("Output directory already exists; preserve earlier readout")
    view = normalize_report_data(view)
    equations = equation_rows(view)
    candidate_numbers = {card["id"]: index for index, card in enumerate(view["candidates"], 1)}
    plot_data = None
    if figures:
        from .figure_output import build_plot_data

        plot_data = build_plot_data(view)
    directory.mkdir(parents=True, exist_ok=False)
    tables = directory / "tables"
    details = directory / "metadata"
    tables.mkdir()
    details.mkdir()
    write_json(details / "report.json", view)
    if include_html:
        render_report_data(view, output=directory / "report.html")
    write_json(details / "equations.json", equations)
    equation_fields = ["candidate_id", "method", "kind", "component", "member_index", "member_id", "weight",
                       "eligible", "reason", "variant", "display_status", "text", "public_text", "display_text", "public_display_text", "latex",
                       "public_latex", "arithmetic", "operators", "semantic_notes", "subject", "execution_view",
                       "uncertainty_scope", "branch", "prediction_unavailable_reason", "native_result"]
    _write_csv(tables / "equations.csv", [{**row, "candidate_number": candidate_numbers[row["candidate_id"]]} for row in equations], [*equation_fields, "candidate_number"])
    parameters = [{**{key: row.get(key) for key in equation_fields[:8]}, **parameter}
                  for row in equations for parameter in row["parameters"]]
    _write_csv(tables / "parameters.csv", parameters,
               [*equation_fields[:8], "name", "value", "unit", "unit_origin", "coordinate_role", "role",
                "tied_to", "bounds", "origin"])
    candidates = []
    constraints = []
    for card in view["candidates"]:
        metrics = card.get("selection") or {}
        candidates.append({
            **{key: card.get(key) for key in ("id", "method", "kind", "variant", "seed", "eligible", "reason",
                                             "native_check", "source", "complexity", "syntax", "hypothesis")},
            "candidate_number": candidate_numbers[card["id"]],
            "run_status": (card.get("source") or {}).get("status"),
            "run_wall_seconds": (card.get("source") or {}).get("wall_seconds"),
            "response_enforcement": ((card.get("source") or {}).get("response_enforcement") or {}).get("effective", "historical_not_recorded"),
            "response_enforcement_requested": ((card.get("source") or {}).get("response_enforcement") or {}).get("requested"),
            "response_enforcement_stage": ((card.get("source") or {}).get("response_enforcement") or {}).get("stage"),
            "partition": "Selection", "unit": view.get("unit"), "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"), "r2": metrics.get("r2"),
            "metric_reason": metrics.get("reason"), "metrics": metrics,
            "selected": card["id"] == (view.get("decision") or {}).get("selected_id"),
            "decision_status": (view.get("decision") or {}).get("status"),
        })
        for index, check in enumerate(card.get("checks", [])):
            constraints.append(dict(candidate_id=card["id"], check_index=index, evidence=check))
    _write_csv(tables / "candidates.csv", candidates,
               ["id", "method", "kind", "variant", "seed", "eligible", "reason", "native_check", "source",
                "complexity", "syntax", "hypothesis", "partition", "unit", "rmse", "mae", "r2",
                "metric_reason", "metrics", "selected", "decision_status", "candidate_number",
                "run_status", "run_wall_seconds", "response_enforcement", "response_enforcement_requested", "response_enforcement_stage"])
    _write_csv(tables / "constraints.csv", constraints, ["candidate_id", "check_index", "evidence"])
    response_rows = [{"candidate_id": card["id"], "engine": card.get("method"), **check}
                     for card in view["candidates"] for check in card.get("response_evaluation", [])]
    _write_csv(tables / "response_checks.csv", response_rows,
               ["candidate_id", "engine", "method", "point_origin", "name", "status", "reason", "data_role",
                "independence", "coordinates", "point_units", "response_bounds", "feature", "direction", "working_direction", "path_unit", "order", "sign", "bounds_scope", "n_points", "n_valid", "n_violating",
                "n_invalid", "violation_rate", "failure_rate", "rate_denominator", "max_violation",
                "max_finite_violation", "response_unit", "overlap_with_prepared_points", "points",
                "worst_point", "worst_response", "worst_prediction", "step", "tolerance"])
    templates = {card["id"]: card.get("template") for card in view["candidates"]}
    named_rows = [{"candidate_id": card["id"], "method": card.get("method"), **row}
                  for card in view["candidates"] for row in template_rows(card.get("template"))]
    if named_rows:
        _write_csv(tables / "template_parts.csv", named_rows,
                   ["candidate_id", "method", "section", "name", "formula", "declared_formula", "arguments",
                    "unit", "value", "parameters", "knowledge", "description", "coordinates", "status", "reason"])

    pairs = view.get("pairs", [])
    _write_csv(tables / "pair_comparisons.csv", pairs,
               list(dict.fromkeys(["left", "right", *[key for pair in pairs for key in pair]])))
    text = [str(view["question"]), "", "Stored equations and coefficients; presentation only.",
            "All candidates are retained. Eligibility, display order and scientific adoption are distinct.",
            "Missing values remain missing. No metrics, responses, uncertainty or predictions were recomputed.", ""]
    tex = ["% LaTeX fragments require amsmath; labels are escaped text.",
           "% Equations are display only. See equations.json for exact AST, units and arithmetic semantics.",
           "% Parameters refer to working coordinates. Unknown units are not dimensionless."]
    for row in equations:
        identity = f"{row['candidate_id']} / {row['component']}"
        if row["member_index"] is not None:
            identity += f" / member {row['member_index']} (weight={row['weight']!r})"
        text.extend([identity, f"Method: {row['method']}; eligible: {row['eligible']}; {row['display_status']}",
                     "Public: " + (row.get("public_text") or "unavailable"),
                     "Working: " + (row.get("text") or "unavailable")])
        for field in ("execution_view", "subject", "uncertainty_scope", "branch", "prediction_unavailable_reason"):
            if row.get(field) is not None:
                text.append(f"{field}: " + json.dumps(row[field], ensure_ascii=False, allow_nan=False))
        tex.append(r"\[" + latex_name(identity) + r"\]")
        for field in ("execution_view", "uncertainty_scope", "branch", "prediction_unavailable_reason"):
            if row.get(field) is not None:
                tex.append(r"\noindent " + latex_name(field + ": " + json.dumps(row[field], ensure_ascii=False)) + r"\par")
        for coordinate in ("public_latex", "latex"):
            if row.get(coordinate):
                tex.append(r"\[" + row[coordinate] + r"\]")
        if not row.get("latex"):
            tex.append(r"\[" + latex_name(row["display_status"]) + r"\]")
        for parameter in row["parameters"]:
            line = (f"  {parameter['name']} = {parameter['value']!r}; unit={parameter.get('unit') or 'unknown'}; "
                    f"role={parameter.get('role')}; tied_to={parameter.get('tied_to')}; "
                    f"bounds={parameter.get('bounds')}; origin={parameter.get('origin')}")
            text.append(line)
            tex.append(r"\[" + latex_name(line.strip()) + r"\]")
        text.extend(template_lines(templates.get(row["candidate_id"])))
        text.extend([*row["semantic_notes"], ""])
        tex.extend(r"\noindent " + latex_name(note) + r"\par" for note in row["semantic_notes"])
    (details / "equations.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    (details / "equations.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    if plot_data is not None:
        from .figure_output import render_figures

        render_figures(plot_data, directory / "figures", table_dir=tables, metadata_dir=details, per_method=True)
    concise = [str(view["question"]), ""]
    for row in equations:
        concise.extend([f"{row['method']} | {row['candidate_id']} | {row['component']}",
                        row.get("public_display_text") or row.get("public_text") or row.get("display_text") or row.get("text") or "Equation unavailable"])
        if row["member_index"] is not None:
            concise.append(f"Member {row['member_index']}; weight = {row['weight']!r}")
        concise.extend(f"  {p['name']} = {p['value']!r} [{p.get('unit') or 'unit unknown'}]"
                       for p in row["parameters"])
        concise.extend(template_lines(templates.get(row["candidate_id"])))
        if row.get("reason"):
            concise.append("Status: " + str(row["reason"]))
        concise.append("")
    concise.append("Coefficient values use the saved working coordinates. Details: metadata/equations.txt")
    (directory / "equations.txt").write_text("\n".join(concise) + "\n", encoding="utf-8")
    readme = [str(view["question"]), "", "Results",
              "  equations.txt  Methods, equations and coefficients",
              "  tables/  Candidate scores, constraints and graph data"]
    if figures:
        readme.append("  figures/  Method folders with separate graphs and equation sheets")
    if include_html:
        readme.append("  report.html  Optional interactive view")
    readme += ["", "Reproduce or inspect",
               "  calculation/  Computation records, when generated in this run",
               "  comparisons/  Additional evaluations, when requested",
               "  metadata/  Machine-readable display data and file integrity index", "",
               f"Candidates: {len(view['candidates'])}", f"Output unit: {view.get('unit') or 'not recorded'}",
               f"Decision: {(view.get('decision') or {}).get('status') or 'not selected'}"]
    readme += ["", "Graphs and tables display saved calculation values without refitting or reselection.",
               "response_checks.csv: prepared/additional point checks, violation rates and units. Additional checks affect Selection, not independent Test.",
               "Invalid points are counted separately; absent evaluation points mean not evaluated, never globally valid."]
    readme += ["candidates.csv: Selection scores compare candidates; they are not independent Final scores.",
               "response_enforcement is the search-time effective mode, not permission to violate final admissibility checks.",
                 "Different adapter defaults/stages are not equivalent guarantees; historical_not_recorded is not inferred.",
                 "run_status describes execution, not equation quality. run_wall_seconds is the entire discovery run",
               "including worker startup, search, export and recovery, excluding common Selection evaluation.",
               "It repeats for candidates from the same run: do not sum candidate rows. Blank means not recorded."]
    (directory / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    files = {path.relative_to(directory).as_posix(): digest(path) for path in sorted(directory.rglob("*"))
             if path.is_file()}
    write_json(details / "output_index.json", dict(
        layout_version=3,
        scope="Presentation files from one saved readout; not a new evaluation, selection or predictor bundle.",
        study_sha256=view.get("study_sha256"), comparison_sha256=view.get("comparison_sha256"),
        decision_sha256=view.get("decision_sha256"), files=files,
        unavailable=(plot_data or {}).get("unavailable", []),
        coefficient_scope="Values, bounds and units are the original working-coordinate coefficients.",
        latex_scope="Historical views without AST retain their original text; no formula parsing or reconstruction.",
        csv_scope="Potential spreadsheet-formula text is apostrophe-prefixed; exact original strings remain in JSON.",
        formula_size_scope="Display expands bounded local definitions/polynomial blocks; large formulas may need manual layout in TeX.",
    ))
    return directory


def export_outputs(comparison_dir, *, output_dir, figures=True, include_html=False, representatives_only=False, **view_options):
    """Comparison or verified detached result to text/LaTeX/tables/plot files."""
    return export_output_data(load_report_view(comparison_dir, **view_options), output_dir=output_dir,
                              figures=figures, include_html=include_html, representatives_only=representatives_only)
