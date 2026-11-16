"""Static figures and their exact source tables from a frozen report view.

No model execution, metric calculation, selection, or access to source datasets.
PNG, SVG and CSV use the same ordered rows, including invalid and unavailable values.
"""

import csv
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

_COLORS = (
    "#196787", "#b74c28", "#75559c", "#28806b", "#916a12", "#aa497b",
    "#395ac8", "#d08700", "#cb3694", "#719727", "#00a2ae", "#6e4939", "#323c46",
)
_PATH_KEYS = ("id", "at", "direction", "steps", "path_unit", "input_units", "source_points", "unit")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _aligned(*values):
    if not all(isinstance(value, (list, tuple)) for value in values) or len({len(v) for v in values}) != 1:
        raise ValueError("Stored plot coordinates and validity masks must have equal lengths")


def _rows(series, x, y, valid, **extra):
    _aligned(x, y, valid)
    return [dict(series=series, point_index=i, x=a, y=b, valid=ok,
                 plotted=ok is True and _finite(a) and _finite(b),
                 plot_reason=None if ok is True and _finite(a) and _finite(b) else
                 "Stored invalid or unavailable row" if ok is not True else "Missing or nonfinite saved coordinate",
                 **extra)
            for i, (a, b, ok) in enumerate(zip(x, y, valid, strict=True))]



def _model_summaries(cards):
    """Copy saved formulas and their parameters; never reconstruct a predictor."""
    result = {}
    methods = list(dict.fromkeys(card.get("method") or "unknown method" for card in cards))
    for index, card in enumerate(cards, 1):
        method = card.get("method") or "unknown method"
        output = card.get("equation_output") or {}
        public = output.get("public_text") or card.get("public_formula")
        text = public or output.get("text") or card.get("formula")
        parameters = output.get("parameters", card.get("parameters", []))
        if card.get("kind") == "implicit" and not output:
            # Residual coefficients cannot be attached to an unrelated predictor.
            parameters = []
        components = []
        if card.get("kind") == "ensemble":
            text = "Native ensemble; execution view: " + str((card.get("execution_view") or {}).get("name", "not saved"))
            parameters = []
            for member in card.get("members", []):
                item = member.get("equation_output") or {}
                components.append(dict(index=member.get("index"), weight=member.get("weight"),
                                       formula=item.get("public_text") or member.get("public_formula")
                                       or item.get("text") or member.get("formula"),
                                       parameters=item.get("parameters", member.get("parameters", []))))
        result[card["id"]] = dict(
            candidate_id=card["id"], number=index, method=method,
            label=f"{index}. {method}", color=_COLORS[methods.index(method) % len(_COLORS)],
            formula=text, display_formula=output.get("public_display_text") or (output.get("display_text") if not public else None), coordinate="public" if public else "working", parameters=parameters,
            mappings=card.get("mappings", []), components=components, template=deepcopy(card.get("template")),
            eligible=card.get("eligible"), reason=card.get("reason"),
            variant=card.get("variant"), seed=card.get("seed"), selection=card.get("selection") or {},
            final_metrics=(card.get("final") or {}).get("target_metrics") or {},
        )
    return result


def _figure_text(figure):
    """Complete, unabridged companion text for a figure, including long equations."""
    lines = [figure["title"], f"X: {figure['x_label']}", f"Y: {figure['y_label']}", ""]
    for model in figure.get("models", []):
        lines.extend([model["label"], "Candidate: " + model["candidate_id"],
                      f"Variant: {model.get('variant')}; seed: {model.get('seed')}",
                      f"Eligible: {model.get('eligible')}; reason: {model.get('reason')}",
                      "Equation (" + model["coordinate"] + "): " + (model["formula"] or "not saved")])
        for parameter in model["parameters"]:
            lines.append(f"  coef[{parameter['name']!r}] = {parameter.get('value')!r} "
                         f"[{parameter.get('unit') or 'unit unknown'}]")
        lines.extend(model["mappings"])
        for member in model["components"]:
            lines.append(f"Member {member['index']}; weight={member['weight']!r}: {member['formula'] or 'not saved'}")
            for parameter in member["parameters"]:
                lines.append(f"  coef[{parameter['name']!r}] = {parameter.get('value')!r} "
                             f"[{parameter.get('unit') or 'unit unknown'}]")
        lines.append("")
    if figure["kind"] == "response_difference":
        lines.extend(["All saved pair differences (left - right; original direction):", ""])
        current = None
        for row in figure["rows"]:
            if row["series"] != current:
                current = row["series"]
                lines.append(current)
            lines.append(f"  step={row['x']!r}; difference={row['y']!r}; valid={row['valid']!r}")
        lines.append("")
    if figure.get("conditions"):
        lines.append("Path conditions: " + json.dumps(figure["conditions"], ensure_ascii=False))
    lines.extend(["Scope: " + figure["scope"], "", "Exact rows and provenance: matching CSV and plot_data.json."])
    return "\n".join(lines) + "\n"


def build_plot_data(view):
    """Return presentation-only plot rows; missing evidence is never synthesized.

    Candidate scatter/path plots follow the saved display IDs. The complexity
    comparison keeps every candidate, including ineligible or unscored records.
    No ranking, coefficient adjustment, residual or pair difference is computed.
    """
    figures, unavailable = [], []
    # Optional presentation label: no coordinate conversion or dimensional claim.
    unit = view.get("unit_label") or view.get("unit") or "unknown unit"
    cards = view.get("candidates", [])
    ids = view.get("display_ids", [])
    by_id = {card["id"]: card for card in cards}
    models = _model_summaries(cards)
    shown = [by_id[name] for name in ids if name in by_id]

    def missing(kind, reason, **context):
        unavailable.append(dict(kind=kind, reason=reason, **context))

    def add(kind, title, x_label, y_label, rows, *, scope, **metadata):
        if not rows:
            missing(kind, "No saved rows for this figure")
            return
        candidate_ids = list(dict.fromkeys(row[key] for row in rows
                                           for key in ("candidate_id", "left", "right") if key in row))
        if kind == "selection_tradeoff":
            metadata["display_ids"] = [name for name in ids if name in candidate_ids]
        names = dict(selection_tradeoff="accuracy", observed_prediction="predictions", residual="residuals",
                     response="response", response_difference="difference", input_geometry="input_distance")
        figure_name = ("final_" if metadata.get("partition") == "Final" else "") + names[kind]
        if metadata.get("candidate_scope") == "representatives":
            figure_name = "representatives"
        figures.append(dict(id=f"{len(figures) + 1:02d}_{figure_name}", kind=kind, title=title,
                            x_label=x_label, y_label=y_label, rows=rows, scope=scope,
                            models=[models[name] for name in candidate_ids if name in models], **metadata))

    metric_rows = []
    for card in cards:
        score = card.get("selection") or {}
        value, complexity = score.get("rmse"), (card.get("complexity") or {}).get("nodes")
        metric_rows.extend(_rows(card.get("method", "unknown method"), [complexity], [value],
                                 [_finite(value) and _finite(complexity)], candidate_id=card["id"],
                                 eligible=card.get("eligible"), reason=score.get("reason") or card.get("reason"),
                                 subject=card.get("subject"),
                                 checks=card.get("checks"), native_check=card.get("native_check")))
    add("selection_tradeoff", "Prediction error and equation size", "Expression nodes [count]",
        f"Selection RMSE [{unit}]", metric_rows, scope="All frozen candidates; no ranking or new score. "
        "Ineligible candidates are marked with crosses. Missing scores remain in CSV, not zero.")

    prediction, residual = [], []
    axis = view.get("residual_axis") or {}
    for card in shown:
        saved = card.get("prediction_scatter")
        if saved is None:
            missing("observed_prediction", "Selection predictions were not saved", candidate_id=card["id"])
        else:
            prediction.extend(_rows(card["id"], saved["observed"], saved["predicted"], saved["valid"],
                                    candidate_id=card["id"], method=card.get("method"),
                                    eligible=card.get("eligible")))
        saved = card.get("residual_scatter")
        if saved is None or "x" not in axis:
            missing("residual", "Selection residuals or input coordinates were not saved", candidate_id=card["id"])
        else:
            residual.extend(_rows(card["id"], axis["x"], saved["y"], saved["valid"],
                                  candidate_id=card["id"], method=card.get("method"), eligible=card.get("eligible")))
    add("observed_prediction", "Observed and predicted values", f"Observed [{unit}]",
        f"Predicted [{unit}]", prediction, reference="identity", scope="Stored Selection rows, actual covariates. "
        "Invalid rows remain in the source table. No Final data or recalculated predictions.")
    add("residual", "Prediction residuals", axis.get("x_label", "Input [unit unknown]"),
        f"Observed - predicted [{unit}]", residual, reference="zero", scope="Selection rows at actual covariates; "
        "not a fixed-condition response, time sequence or causal effect.")

    if (view.get("audit") or {}).get("provided"):
        final_prediction, final_residual = [], []
        for card in shown:
            final = card.get("final") or {}
            saved = final.get("prediction_scatter")
            if saved is None:
                missing("final_prediction", "Final observations/predictions were not saved", candidate_id=card["id"])
                continue
            final_prediction.extend(_rows(card["id"], saved["observed"], saved["predicted"], saved["valid"],
                                           candidate_id=card["id"], method=card.get("method")))
            residual = final.get("residual_scatter")
            if residual is not None:
                final_residual.extend(_rows(card["id"], saved["predicted"], residual["y"], residual["valid"],
                                            candidate_id=card["id"], method=card.get("method")))
        role = (view.get("audit") or {}).get("role") or "Saved Final audit; independence not established"
        add("observed_prediction", "Final: observed and predicted values", f"Observed [{unit}]",
            f"Predicted [{unit}]", final_prediction, reference="identity", partition="Final", scope=role)
        add("residual", "Final: prediction residuals", f"Predicted [{unit}]",
            f"Observed - predicted [{unit}]", final_residual, reference="zero", partition="Final", scope=role)

    paths = {}
    for card in shown:
        interpretations = card.get("interpretations")
        if interpretations is None:
            interpretations = [card["interpretation"]] if card.get("interpretation") else []
        found = False
        for analysis in interpretations:
            for path in analysis.get("paths", []):
                source = analysis.get("analysis_sha256")
                display_id = path.get("display_id")
                if not source or not display_id:
                    missing("response", "Saved path source identity is absent; no cross-source overlay", candidate_id=card["id"],
                            path_id=path.get("id"))
                    continue
                definition = {key: path.get(key) for key in _PATH_KEYS}
                key = (source, display_id, json.dumps(definition, sort_keys=True, ensure_ascii=False, allow_nan=False))
                entry = paths.setdefault(key, dict(definition=path, rows=[], candidate_ids=[]))
                rows = _rows(card["id"], path["steps"], path["prediction"], path["valid"],
                             candidate_id=card["id"], method=card.get("method"), eligible=card.get("eligible"),
                             analysis_sha256=source, path_display_id=display_id)
                points = path.get("source_points")
                if points is not None:
                    _aligned(points, rows)
                    for row, point in zip(rows, points, strict=True):
                        row["source_point"] = point
                entry["rows"].extend(rows)
                entry["candidate_ids"].append(card["id"])
                found = True
        if not found:
            missing("response", "No source-bound saved response path for displayed candidate", candidate_id=card["id"])

    bound_differences = set()
    for (source, display_id, _), entry in paths.items():
        path = entry["definition"]
        metadata = dict(analysis_sha256=source, path_display_id=display_id, path_id=path.get("id"),
                        conditions={key: path.get(key) for key in (*_PATH_KEYS, "source_columns", "step_widths")})
        label = f"Path step [{path.get('path_unit') or 'unknown unit'}]"
        changing = [name for name, slope in (path.get("direction") or {}).items() if slope != 0]
        axis_offset = None
        response_title = f"Model response: {path['id']}"
        if len(changing) == 1:
            name = changing[0]
            response_title = f"Model response along {name}"
            if (path["direction"][name] == 1 and path.get("path_unit")
                    and (path.get("input_units") or {}).get(name) == path["path_unit"]):
                origin = (path.get("at") or {}).get(name)
                if _finite(origin):
                    axis_offset = float(origin)
                    label = f"{name} [{path['path_unit']}]"
                    metadata["x_coordinate"] = {"feature": name, "origin": axis_offset,
                                                "meaning": "saved source input = origin + path step"}

        def axis_rows(rows, axis_offset=axis_offset):
            if axis_offset is None:
                return rows
            return [{**row, "path_step": row["x"], "x": row["x"] + axis_offset} for row in rows]

        scope = str(path.get("scope") or "Saved user-declared path")
        scope += ". Saved grid only; line segments connect adjacent valid grid rows, not validated interpolation."
        path_unit = unit if path.get("unit") == view.get("unit") else (path.get("unit") or unit)
        add("response", response_title, label, f"Prediction [{path_unit}]",
            axis_rows(entry["rows"]), line=True, scope=scope, **metadata)
        differences = []
        for index, pair in enumerate(view.get("analysis_comparisons", [])):
            if (pair.get("axis") != "path_response_difference" or pair.get("path_display_id") != display_id
                    or pair.get("analysis_sha256") not in (None, source)
                    or not {pair.get("left"), pair.get("right")} <= set(entry["candidate_ids"])):
                continue
            if any(pair.get(key) != path.get(key) for key in ("steps", "path_unit", "unit")):
                continue
            differences.extend(_rows(f"{pair['left']} - {pair['right']}", pair["steps"], pair["difference"],
                                     pair["valid"], left=pair["left"], right=pair["right"],
                                     analysis_sha256=source, path_display_id=display_id))
            bound_differences.add(index)
        if differences:
            add("response_difference", f"Prediction differences: {path['id']}", label,
                f"Prediction (left - right) [{path_unit}]", axis_rows(differences), line=True, reference="zero",
                scope="Same source-bound path and saved difference values. Disagreement is not evidence of which "
                "candidate is correct, causal, safe, or the best next experiment.", **metadata)
        else:
            missing("response_difference", "No stored pair difference bound to this exact saved path", **metadata)

        geometry = path.get("reference_geometry") or {}
        if geometry.get("status") != "computed" or geometry.get("data_role") != "train":
            missing("input_geometry", geometry.get("reason") or "Train input geometry not saved", **metadata)
            continue
        distances = geometry.get("nearest_distance", [])
        outside, changed = geometry.get("outside_marginal_range", []), geometry.get("constant_feature_changed", [])
        _aligned(path["steps"], distances, outside, changed)
        support = _rows("Nearest Train input", path["steps"], distances, [_finite(v) for v in distances],
                        analysis_sha256=source, path_display_id=display_id)
        for row, out, constant in zip(support, outside, changed, strict=True):
            row.update(outside_marginal_range=out, constant_feature_changed=constant)
        add("input_geometry", f"Distance from training inputs: {path['id']}", label,
            "Range-scaled input distance [1]", axis_rows(support), line=False, reference_geometry=geometry,
            scope="Input-only geometry, not prediction accuracy, joint support, safety or causality. No OOD cutoff. "
            "Squares: outside a marginal range. Crosses: changed a Train-constant input (omitted from distance).",
            **metadata)
    for index, pair in enumerate(view.get("analysis_comparisons", [])):
        if pair.get("axis") == "path_response_difference" and index not in bound_differences:
            reason = "Pair not bound to an exact displayed path; not overlaid or recomputed"
            if not {pair.get("left"), pair.get("right")} <= set(ids):
                reason = "Outside displayed candidate set; stored pair retained in the full comparison"
            missing("response_difference", reason,
                    left=pair.get("left"), right=pair.get("right"), path_display_id=pair.get("path_display_id"))
    if ids:
        add("selection_tradeoff", "Representative equations: error and size", "Expression nodes [count]",
            f"Selection RMSE [{unit}]", [row for row in metric_rows if row["candidate_id"] in ids],
            candidate_scope="representatives", scope="Saved display representatives only. "
            "All candidates and extreme errors remain in the accuracy overview and candidate table.")

    return _clean(deepcopy(dict(schema_version=1, figures=figures, unavailable=unavailable,
                               csv_string_policy="Strings beginning with =, +, -, @ or whitespace controls are "
                               "prefixed with an apostrophe for spreadsheet safety. JSON retains exact raw strings.",
                               displayed_candidate_ids=[c["id"] for c in shown],
                               provenance={key: view.get(key) for key in
                                           ("study_sha256", "comparison_sha256", "decision_sha256", "purpose", "unit", "display_units", "display_unit_scope")},
                               scope="Frozen report evidence only; no new prediction, fit, score, selection or Final access.")))


def _csv_bytes(rows):
    from .reporting import csv_cell

    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_cell(value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def _axis_layout(values, *, integer=False, nonnegative=False):
    """Stable linear coordinates and readable 1/2/5 ticks, including float extremes."""
    low, high = min(values), max(values)
    scale = max(abs(low), abs(high)) or 1.0
    lo, hi = low / scale, high / scale
    pad = (hi - lo) * .05 if lo != hi else .05
    limit = sys.float_info.max / scale
    lo, hi = max(lo - pad, -limit), min(hi + pad, limit)
    if nonnegative:
        lo = max(0.0, lo)
    # Work in scaled coordinates so subtracting opposite float extremes is safe.
    raw_step = (hi - lo) / 5
    log_step = math.log10(raw_step) + math.log10(scale)
    exponent = math.floor(log_step)
    fraction = 10 ** (log_step - exponent)
    multiple = next(v for v in (1, 2, 5, 10) if v >= fraction - 1e-12)
    step = multiple * 10 ** (exponent - math.log10(scale))
    if integer:
        step = max(step, 1 / scale)
        if step * scale < sys.float_info.max:
            step = math.ceil(step * scale - 1e-12 * max(step * scale, 1)) / scale
    first, last = math.ceil(lo / step), math.floor(hi / step)
    ticks = [i * step * scale for i in range(first, last + 1)]
    ticks = list(dict.fromkeys(v for v in ticks if math.isfinite(v)))
    return lo, hi, scale, ticks


def _axis_offset(values):
    """Expose small variation around a large baseline without repeated labels."""
    low, high = min(values), max(values)
    span = high - low
    return low if math.isfinite(span) and span > 0 and abs(low) / span >= 100 else 0.0


def _tick_label(value):
    if value == 0:
        return "0"
    if abs(value) < .001 or abs(value) >= 100000:
        return f"{value:.3g}"
    return f"{value:.6g}"


def _svg(figure, csv_name, csv_sha256):
    """One result per axes, from saved rows only; no fitted or selected values.

    Single-series plots are monochrome. Colour and line styles are used only
    to identify overlaid series; annotations and reference lines stay neutral.
    """
    rows = figure["rows"]
    series = list(dict.fromkeys(row["series"] for row in rows))
    valid = [row for row in rows if row["plotted"]]
    models = {model["candidate_id"]: model for model in figure.get("models", [])}
    kind = figure["kind"]
    caption_path = figure.get("caption_path", "metadata/" + figure["id"] + ".txt")
    labels, colors = {}, {}
    for index, name in enumerate(series):
        pair = next((r for r in rows if r["series"] == name and "left" in r), None)
        labels[name] = models[name]["label"] if name in models else name
        if pair:
            labels[name] = " - ".join(models.get(pair[key], {}).get("label", pair[key]) for key in ("left", "right"))
        model = models.get(name)
        if kind == "selection_tradeoff":
            model = next((m for m in models.values() if m["method"] == name), None)
        colors[name] = ("#222222" if len(series) == 1 else
                        model["color"] if model else _COLORS[index % len(_COLORS)])

    # Context belongs in a small caption, not inside the plotting rectangle.
    notes = []
    if figure.get("candidate_id"):
        notes.append("Candidate: " + figure["candidate_id"])
    if figure.get("conditions"):
        c = figure["conditions"]
        at, direction, units = c.get("at") or {}, c.get("direction") or {}, c.get("input_units") or {}
        axis_feature = (figure.get("x_coordinate") or {}).get("feature")
        conditions = [f"{name} = {value!r}" + (f" + ({direction[name]!r}) * t" if direction.get(name) else "")
                      + f" [{units.get(name) or 'unit unknown'}]" for name, value in at.items() if name != axis_feature]
        if len(conditions) <= 4:
            notes.append(("Fixed inputs: " if axis_feature else "Path: ") + "; ".join(conditions))
        else:
            notes.append(f"Declared path with {len(at)} inputs; full coordinates and units: {caption_path}")
        notes.append("Saved path points; lines join adjacent valid points, not verified interpolation."
                     if figure.get("line") else "Saved path points; not prediction-error evidence.")
    else:
        notes.append(figure.get("partition", "Selection") + " data" if kind != "input_geometry" else "Training inputs")
    if figure.get("partition") == "Final":
        notes.append(figure["scope"])
    if len(valid) != len(rows):
        notes.append(f"Valid points: {len(valid)} / {len(rows)}; unavailable rows remain in CSV.")
    if kind == "selection_tradeoff":
        notes.append("Display representatives only; fixed using Selection, not Final."
                     if figure.get("candidate_scope") == "representatives" else
                     "All candidates with finite scores; missing scores are not zero.")
    if kind == "response_difference":
        notes.append("Left minus right, in legend order; disagreement does not identify the better model.")
    if kind == "input_geometry":
        notes.append("Input distance only, not accuracy or joint support. Square: outside a marginal range; x: changed constant input.")
    elif any(row.get("eligible") is False for row in rows):
        notes.append("x: excluded by saved checks (not a search-failure classification).")

    title_lines = textwrap.wrap(figure["title"], 85, break_long_words=True)
    top = 62 + 24 * (len(title_lines) - 1)
    left, right, bottom = 125.0, 925.0, top + 435.0
    # Identical limits AND physical aspect are necessary for a 45-degree identity.
    if figure.get("reference") == "identity":
        left, right, bottom = 245.0, 745.0, top + 500.0
    def axis_label_lines(label, width):
        lines = textwrap.wrap(label, width, break_long_words=True)
        if len(lines) <= 2:
            return lines
        unit = re.search(r"\[[^\[\]]*\]\s*$", label)
        if not any(note.startswith("Full axis labels:") for note in notes):
            notes.append("Full axis labels: " + caption_path)
        return [lines[0], "… " + (unit.group() if unit else "(full label in caption)")]

    x_labels = axis_label_lines(figure["x_label"], 48 if figure.get("reference") == "identity" else 78)
    y_labels = axis_label_lines(figure["y_label"], 44)
    legends = []
    if len(series) > 1 or (len(series) == 1 and not figure.get("candidate_id")):
        legends = [(colors[name], labels[name], index) for index, name in enumerate(series)]
    # Horizontal legend rows below the axes leave the data full width. Never
    # truncate series or pairs: large comparisons use separate output figures.
    legend_lines = [(color, textwrap.wrap(label, 100, break_long_words=True), index)
                    for color, label, index in legends]
    note_lines = [line for note in notes for line in textwrap.wrap(note, 118, break_long_words=True)]
    legend_y = bottom + 78 + 21 * max(0, len(x_labels) - 1)
    note_y = legend_y + sum(len(chunks) * 18 + 5 for _, chunks, _ in legend_lines) + 10
    height = math.ceil(note_y + len(note_lines) * 18 + 18)
    def esc(value):
        return html.escape(str(value), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" '
             f'role="img" aria-labelledby="title desc" data-source-csv="{esc(csv_name)}" data-source-sha256="{csv_sha256}">',
             f'<title id="title">{esc(figure["title"])}</title><desc id="desc">{esc(figure["scope"])}</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#222222">']

    def text(x, y, value, size=15, **attrs):
        extra = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
        parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" {extra}>{esc(value)}</text>')

    for index, line in enumerate(title_lines):
        text(125, 27 + index * 24, line, 19)
    if valid:
        xs, ys = [r["x"] for r in valid], [r["y"] for r in valid]
        if figure.get("reference") == "identity":
            xs = ys = xs + ys
        if figure.get("reference") == "zero":
            ys = ys + [0]
        positive = [v for v in ys if v > 0]
        wide = (kind == "selection_tradeoff" and min(ys) >= 0 and positive
                and math.log10(max(positive)) - math.log10(min(positive)) >= 2)
        scale_kind, knee = "linear", None
        if wide:
            scale_kind = "log" if min(ys) > 0 else "symlog"
            knee = min(positive)

        xoffset = _axis_offset(xs)
        yoffset = _axis_offset(ys) if scale_kind == "linear" else 0.0

        def transform(value):
            if scale_kind == "linear":
                return value - yoffset
            if scale_kind == "symlog" and value <= knee:
                return value / knee
            return math.log10(value) - math.log10(knee) + (1 if scale_kind == "symlog" else 0)

        x0, x1, sx, xticks = _axis_layout([v - xoffset for v in xs], integer=kind == "selection_tradeoff", nonnegative=kind == "selection_tradeoff" and not xoffset)
        y0, y1, sy, yticks = _axis_layout([transform(v) for v in ys],
                                        nonnegative=kind in {"selection_tradeoff", "input_geometry"} and scale_kind != "log" and not yoffset)
        if scale_kind != "linear":
            first, last = math.floor(math.log10(min(positive))), math.ceil(math.log10(max(positive)))
            step = max(1, math.ceil((last - first) / 4))
            powers = sorted(set([*range(first, last + 1, step), last]))
            yticks = [10.0 ** power for power in powers if -323 <= power <= 308]
            if scale_kind == "log" and yticks:
                y0, y1, sy, _ = _axis_layout([transform(v) for v in [*ys, *yticks]])
            if scale_kind == "symlog":
                yticks = [0, knee] + [v for v in yticks if math.log10(v) - math.log10(knee) >= .5]
            text(right, top - 12, "Log scale" if scale_kind == "log" else f"Linear to {_tick_label(knee)}; log above", 13,
                 text_anchor="end")
        ytick_labels = {value + yoffset: _tick_label(value) for value in yticks} if scale_kind == "linear" else {}
        if scale_kind == "linear":
            yticks = [value + yoffset for value in yticks]
        if xoffset:
            text(right, bottom + 46, f"Offset + ({xoffset:.17g})", 12, text_anchor="end")
        if yoffset:
            text(left, top - 12, f"Offset + ({yoffset:.17g})", 12)
        parts.append(f'<g data-x-offset="{xoffset}" data-y-offset="{yoffset}" data-y-scale="{scale_kind}" data-y-linear-threshold="{knee}" '
                     f'data-plot-left="{left}" data-plot-right="{right}" data-plot-top="{top}" data-plot-bottom="{bottom}">')
        def px(value):
            return left + ((value - xoffset) / sx - x0) / (x1 - x0) * (right - left)

        def py(value):
            return bottom - (transform(value) / sy - y0) / (y1 - y0) * (bottom - top)
        for tick in xticks:
            value = tick + xoffset
            x = px(value)
            parts.append(f'<path d="M {x:.3f} {bottom:.3f} v 5" stroke="#555555"/>')
            text(x, bottom + 25, _tick_label(tick), text_anchor="middle", data_axis="x", data_tick=value)
        for value in yticks:
            y = py(value)
            if not top - .01 <= y <= bottom + .01:
                continue
            parts.append(f'<path d="M {left-5:.3f} {y:.3f} h 5" stroke="#555555"/>')
            text(left - 12, y + 5, ytick_labels.get(value, _tick_label(value)), text_anchor="end", data_axis="y", data_tick=value)
        if figure.get("reference") == "identity":
            parts.append(f'<path data-reference="identity" d="M {left:.3f} {bottom:.3f} L {right:.3f} {top:.3f}" '
                         'stroke="#777777" stroke-dasharray="6 5" fill="none"/>')
        elif figure.get("reference") == "zero":
            parts.append(f'<path data-reference="zero" d="M {left:.3f} {py(0):.3f} H {right:.3f}" '
                         'stroke="#777777" stroke-dasharray="6 5" fill="none"/>')
        for series_index, name in enumerate(series):
            color, previous = colors[name], None
            dash = ("", "8 4", "2 4", "9 3 2 3", "6 2 2 2 2 2")[series_index % 5]
            for index, row in enumerate(rows):
                if row["series"] != name:
                    continue
                if not row["plotted"]:
                    previous = None
                    continue
                x, y = px(row["x"]), py(row["y"])
                if figure.get("line") and previous is not None:
                    parts.append(f'<path data-segment="true" d="M {previous[0]:.3f} {previous[1]:.3f} L {x:.3f} {y:.3f}" '
                                 f'stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}" fill="none"/>')
                attrs = (f'data-row="{index}" data-series="{esc(name)}" data-x="{row["x"]!r}" '
                         f'data-y="{row["y"]!r}"')
                tip = esc(json.dumps(row, ensure_ascii=False, allow_nan=False))
                if row.get("constant_feature_changed") or row.get("eligible") is False:
                    parts.append(f'<path {attrs} d="M {x-4:.3f} {y-4:.3f} l 8 8 M {x-4:.3f} {y+4:.3f} l 8 -8" '
                                 f'stroke="{color}" stroke-width="1.6"><title>{tip}</title></path>')
                elif row.get("outside_marginal_range"):
                    parts.append(f'<rect {attrs} x="{x-4:.3f}" y="{y-4:.3f}" width="8" height="8" '
                                 f'fill="white" stroke="{color}"><title>{tip}</title></rect>')
                else:
                    parts.append(f'<circle {attrs} cx="{x:.3f}" cy="{y:.3f}" r="3.4" fill="{color}" '
                                 f'fill-opacity="0.72"><title>{tip}</title></circle>')
                previous = x, y
        parts.append('</g>')
    else:
        native_only = bool(rows) and all((row.get("subject") or {}).get("view") == "native_only" for row in rows)
        text(left + 20, top + 80, "No candidate equations available." if native_only else
             "No finite valid stored points. See CSV and unavailable evidence.", 16)
        if native_only:
            reason = "; ".join(dict.fromkeys(row["reason"] for row in rows if row.get("reason")))
            reason_lines = textwrap.wrap(reason, 85)
            for index, line in enumerate(reason_lines[:8]):
                text(left + 20, top + 110 + index * 20, line, 14)
            if len(reason_lines) > 8:
                text(left + 20, top + 280, "… Full reasons are preserved in CSV: " + csv_name, 13)
    parts.append(f'<path d="M {left:.3f} {top:.3f} V {bottom:.3f} H {right:.3f}" stroke="#555555" fill="none"/>')
    for index, label in enumerate(x_labels):
        text((left + right) / 2, bottom + 58 + index * 21, label, 17, text_anchor="middle")
    for index, label in enumerate(y_labels):
        x = left - 94 + (index - (len(y_labels) - 1) / 2) * 21
        text(x, (top + bottom) / 2, label, 17, text_anchor="middle",
             transform=f"rotate(-90 {x} {(top+bottom)/2})")
    y = legend_y
    for color, chunks, index in legend_lines:
        if figure.get("line"):
            dash = ("", "8 4", "2 4", "9 3 2 3", "6 2 2 2 2 2")[index % 5]
            parts.append(f'<path data-legend-line="true" d="M 125 {y-5} h 30" '
                         f'stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}"/>')
        else:
            parts.append(f'<circle cx="130" cy="{y-5}" r="3.4" fill="{color}"/>')
        for chunk in chunks:
            text(165 if figure.get("line") else 145, y, chunk, 14)
            y += 18
        y += 5
    for index, line in enumerate(note_lines):
        text(125, note_y + index * 18, line, 13)
    parts.append('</g></svg>')
    return "\n".join(parts)


def _equation_sheets(model):
    """Readable pages from saved display text, with exact text kept alongside."""
    from .equation_output import template_lines

    title = model["label"] + " | equation"
    equation = model.get("display_formula") or model.get("formula") or "Equation not saved"
    lines = ["Coordinates: " + model["coordinate"], "", *textwrap.wrap(equation, 76, break_on_hyphens=False), ""]
    if model["parameters"]:
        lines.append("Coefficients (working coordinates; rounded on image)")
        for p in model["parameters"]:
            value = p.get("value")
            shown = f"{value:.8g}" if _finite(value) else repr(value)
            lines.extend(textwrap.wrap(f"{p['name']} = {shown} [{p.get('unit') or 'unit unknown'}]; role={p.get('role', 'not recorded')}", 76))
    for member in model["components"]:
        lines.extend(textwrap.wrap(f"Member {member['index']}; weight={member['weight']!r}: {member['formula']}", 76))
    for line in template_lines(model.get("template"), concise=True):
        lines.extend(textwrap.wrap(line, 76, break_on_hyphens=False) or [""])
    pages = [lines[i:i + 24] for i in range(0, len(lines), 24)] or [[]]
    sheets = []
    for page, page_lines in enumerate(pages, 1):
        height = max(280, 140 + 27 * len(page_lines))
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#203342">',
                 f'<text x="40" y="40" font-size="24" font-weight="bold">{html.escape(title)}</text>',
                 f'<text x="40" y="68" font-size="13">Candidate {html.escape(model["candidate_id"])} | page {page}/{len(pages)}</text>']
        for i, line in enumerate(page_lines):
            parts.append(f'<text x="40" y="{110 + 27 * i}" font-size="20">{html.escape(line)}</text>')
        parts.append('</g></svg>')
        sheets.append("\n".join(parts))
    exact = [title, "Candidate: " + model["candidate_id"], "Coordinates: " + model["coordinate"],
             model.get("formula") or "Equation not saved", ""]
    exact.extend(f"{p['name']} = {p.get('value')!r} [{p.get('unit') or 'unit unknown'}]" for p in model["parameters"])
    exact.extend(model["mappings"])
    exact.extend(template_lines(model.get("template")))
    for member in model["components"]:
        exact.append(f"Member {member['index']}; weight={member['weight']!r}: {member['formula']}")
        exact.extend(f"{p['name']} = {p.get('value')!r} [{p.get('unit') or 'unit unknown'}]" for p in member["parameters"])
    return "\n".join(exact) + "\n", sheets


def render_figures(plot_data, output_dir, *, table_dir=None, metadata_dir=None, per_method=False):
    """Write PNG images, SVG/CSV evidence and complete equation text.

    Existing files are refused before writing. SVG metadata names and hashes the
    exact CSV; data-row attributes point into its zero-based data row order.
    """
    # Rasterize the exact SVG, rather than maintain a second plotting algorithm.
    # resvg is a self-contained wheel; no browser, network, model or TeX runtime.
    from resvg_py import svg_to_bytes

    output_dir = Path(output_dir)
    table_dir = Path(table_dir) if table_dir is not None else output_dir
    metadata_dir = Path(metadata_dir) if metadata_dir is not None else output_dir
    files = [(metadata_dir / "plot_data.json",
              (json.dumps(_clean(plot_data), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))]
    names = set()
    figures = plot_data["figures"]
    # The complete evidence tables are retained once. User-facing images show
    # one saved candidate at a time, rather than hiding errors in overlays.
    for figure in figures:
        name = figure["id"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Figure IDs must be safe filenames")
        if name in names:
            raise ValueError("Figure IDs must be unique")
        names.add(name)
        raw = _csv_bytes(figure["rows"])
        files.append((table_dir / (name + ".csv"), raw))
        files.append((metadata_dir / (name + ".txt"), _figure_text(figure).encode("utf-8")))
        if not per_method or figure["kind"] in {"selection_tradeoff", "input_geometry"}:
            csv_link = Path(os.path.relpath(table_dir / (name + ".csv"), output_dir)).as_posix()
            svg = _svg({**figure, "caption_path": Path(os.path.relpath(metadata_dir / (figure["id"] + ".txt"), output_dir)).as_posix()}, csv_link, hashlib.sha256(raw).hexdigest())
            files.extend([(output_dir / (name + ".svg"), svg.encode("utf-8")),
                          (output_dir / (name + ".png"), svg_to_bytes(svg_string=svg, zoom=1.5, font_family="DejaVu Sans"))])
    if per_method:
        # Cross-method evidence is rendered separately, never reconstructed from
        # per-method images. Each saved pair gets a readable chart, not a hidden
        # subset of a crowded all-pairs overlay.
        directory = output_dir / "comparison"
        comparison_index = ["Start with the response overlay. Open a pair difference only when needed.",
                            "All pairs are saved; none is a model-selection decision.", ""]
        for figure in figures:
            if figure["kind"] not in {"response", "response_difference"}:
                continue
            groups = [(figure["id"], figure)]
            if figure["kind"] == "response_difference":
                groups = []
                for index, series in enumerate(dict.fromkeys(r["series"] for r in figure["rows"]), 1):
                    rows = [r for r in figure["rows"] if r["series"] == series]
                    ids = {rows[0]["left"], rows[0]["right"]}
                    models = [m for m in figure.get("models", []) if m["candidate_id"] in ids]
                    groups.append((f"{figure['id']}_{index:02d}", {**figure, "rows": rows, "models": models}))
            for name, saved in groups:
                raw = _csv_bytes(saved["rows"])
                csv_path = table_dir / "comparison" / (name + ".csv")
                link = Path(os.path.relpath(csv_path, directory)).as_posix()
                svg = _svg({**saved, "caption_path": Path(os.path.relpath(metadata_dir / (figure["id"] + ".txt"), directory)).as_posix()}, link, hashlib.sha256(raw).hexdigest())
                identities = "; ".join(f"{m['label']} [{m['candidate_id']}]" for m in saved.get("models", []))
                plotted = sum(row["plotted"] for row in saved["rows"])
                comparison_index.append(f"{name}.png | {identities} | plotted {plotted}/{len(saved['rows'])} saved rows")
                files.extend([(csv_path, raw), (directory / (name + ".svg"), svg.encode("utf-8")),
                              (directory / (name + ".png"), svg_to_bytes(svg_string=svg, zoom=1.5, font_family="DejaVu Sans"))])
        if len(comparison_index) > 3:
            files.append((directory / "README.txt", ("\n".join(comparison_index) + "\n").encode("utf-8")))
    if per_method:
        models = {m["candidate_id"]: m for f in figures for m in f.get("models", [])}
        ids = plot_data.get("displayed_candidate_ids", [])
        summary = []
        for candidate_id in ids:
            model = models.get(candidate_id)
            if model is None:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", model["method"].lower()).strip("-")[:40] or "method"
            folder = f"{model['number']:02d}_{slug}"
            directory = output_dir / folder
            score = model.get("selection") or {}
            summary.append(dict(method=model["method"], candidate_id=candidate_id, figures=folder,
                                eligible=model.get("eligible"), reason=model.get("reason"),
                                selection_rmse=score.get("rmse"),
                                final_rmse=(model.get("final_metrics") or {}).get("rmse")))
            generated = []
            for figure in figures:
                if figure.get("candidate_scope") == "representatives":
                    continue
                if figure["kind"] not in {"selection_tradeoff", "observed_prediction", "residual", "response"}:
                    continue
                method_ids = {key for key, value in models.items() if value["method"] == model["method"]}
                is_accuracy = figure["kind"] == "selection_tradeoff"
                rows = [r for r in figure["rows"] if r.get("candidate_id") in
                        (method_ids if is_accuracy else {candidate_id})]
                if not rows:
                    continue
                single = {**figure, "rows": rows, "models": [models[key] for key in models if key in method_ids] if is_accuracy else [model],
                          "display_ids": [candidate_id], "candidate_id": candidate_id,
                          "title": (model["method"] if is_accuracy else model["label"]) + " | " + figure["title"]}
                name = figure["id"]
                raw = _csv_bytes(rows)
                csv_path = table_dir / folder / (name + ".csv")
                csv_link = Path(os.path.relpath(csv_path, directory)).as_posix()
                svg = _svg({**single, "caption_path": Path(os.path.relpath(metadata_dir / (figure["id"] + ".txt"), directory)).as_posix()}, csv_link, hashlib.sha256(raw).hexdigest())
                files.extend([(csv_path, raw), (directory / (name + ".svg"), svg.encode("utf-8")),
                              (directory / (name + ".png"), svg_to_bytes(svg_string=svg, zoom=1.5, font_family="DejaVu Sans"))])
                generated.append(name + ".png")
            # A dedicated equation sheet is never mixed with chart axes or points.
            equation_text, sheets = _equation_sheets(model)
            files.append((directory / "equation.txt", equation_text.encode("utf-8")))
            for page, svg in enumerate(sheets, 1):
                name = "equation" if len(sheets) == 1 else f"equation_{page:02d}"
                files.extend([(directory / (name + ".svg"), svg.encode("utf-8")),
                              (directory / (name + ".png"), svg_to_bytes(svg_string=svg, zoom=1.5, font_family="DejaVu Sans"))])
            readme = [model["method"], "Candidate: " + candidate_id,
                      "Saved display representative; not a final model decision.",
                      "Accuracy graph includes all candidates of this method; crosses are excluded candidates.",
                      "Reasons and full candidate scores: ../../tables/candidates.csv",
                      f"Selection RMSE: {score.get('rmse')}",
                      f"Final RMSE: {(model.get('final_metrics') or {}).get('rmse')} (None means not saved)",
                      f"Eligible: {model.get('eligible')}; reason: {model.get('reason')}", "",
                      *generated, "equation*.png / equation.txt: separate formula and coefficients", "",
                      "No saved graph is invented when evidence is unavailable."]
            files.append((directory / "README.txt", ("\n".join(readme) + "\n").encode("utf-8")))
        files.append((table_dir / "displayed_models.csv", _csv_bytes(summary)))
        files.append((output_dir / "README.txt", (b"Open a method folder for its prediction, residual and response graphs.\n"
                      b"Equations are separate equation*.png and equation.txt files.\n"
                      + (b"comparison/ contains saved same-path response overlays and one chart per saved pair.\n"
                       if any(f["kind"] in {"response", "response_difference"} for f in figures) else
                       b"No shared response-path evidence was supplied. Evaluate fixed common paths on this study and pass that result to compare_equations(evaluations=...). Selection-pair metrics do not supply response curves.\n")
                      + (b"Start with *_representatives.png for the displayed methods; use *_accuracy.png for all candidates and extreme errors. Both show Selection, not Final.\n"
                         if any(f.get("candidate_scope") == "representatives" for f in figures) else
                         b"The top-level accuracy chart compares saved Selection scores; it is not Final performance.\n")
                      + b"All candidate and pair data remain in tables/ and metadata/.\n")))
    if any(path.exists() for path, _ in files):
        raise ValueError("Figure output already exists; preserve earlier evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return [path for path, _ in files]
