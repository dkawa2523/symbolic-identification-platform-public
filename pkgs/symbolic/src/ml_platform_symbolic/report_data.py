"""Read-only presentation view of stored scientific evidence.

This module formats coordinates, checks artifact identities and assembles values.
It never predicts, fits, scores or decides scientific adoption.
"""

from pathlib import Path

from .data import digest, input_variables, model_output, public_unit, read_json, value_digest
from .equation_evaluation.api import read_study_evidence
from .equation_output import equation_outputs, format_equation, original_equation
from .parameters import parameter_table
from .question_results import resolved_questions
from .units import EVIDENCE_COORDINATE_POLICY, conversion


def _display_unit_labels(schema, display_units):
    """Label public-coordinate values only; never convert or establish dimensions.

    Raw labels can supplement unknown/unit-1 identity coordinates. For physical
    coordinates the declared label must name the SAME numeric coordinate, not
    merely a dimensionally compatible unit. Frozen source maps remain authority.
    """
    definitions = [*input_variables(schema), schema["target"]]
    by_column = {item["column"]: item for item in definitions}
    if not isinstance(display_units, dict) or set(display_units) - set(by_column):
        raise ValueError("Display units must map known public columns to source labels")
    labels = {}
    for column, label in display_units.items():
        if not isinstance(label, str) or not label.strip() or "\n" in label:
            raise ValueError("Display units require nonempty single-line source labels")
        definition = by_column[column]
        public = public_unit(schema, definition)
        frozen = (schema.get("unit_coordinates") or {}).get("columns", {}).get(column, {})
        if public in (None, "1"):
            if (frozen.get("scale", 1) != 1 or frozen.get("offset", 0) != 0
                    or definition.get("unit") not in (None, "1")):
                raise ValueError(f"Display label for {column} cannot relabel transformed coordinates")
            labels[column] = f"{label} source label; model={public or 'unknown'}"
        else:
            scale, offset = conversion(label, public)
            if abs(scale - 1) > 1e-14 or offset != 0:
                raise ValueError(f"Display label for {column} differs from plotted coordinate {public}")
            labels[column] = label
    return labels


def _formula(equation, variables):
    if equation is None:
        return None
    try:
        return format_equation(equation, variables, parameter_names=True)
    except (ValueError, KeyError, TypeError) as exc:
        return f"Display conversion unavailable: {exc}; authoritative native artifact retained"


def _original_formula(equation, context, *, residual=False):
    return original_equation(equation, context, residual=residual)


def _metrics(values, schema):
    if values is None:
        return None
    return {
        **values,
        **{
            key: float(model_output(values[key], schema, difference=True))
            for key in ("rmse", "mae")
            if values.get(key) is not None
        },
    }


def _audit_quantities(row, schema):
    """One working-to-public difference conversion for all audit displays."""
    support = row.get("prediction_support")
    if support is not None:
        support = {**support, "criteria": dict(support.get("criteria", {}))}
        if support.get("observed_rmse") is not None:
            support["observed_rmse"] = float(model_output(support["observed_rmse"], schema, difference=True))
        if support["criteria"].get("rmse_max") is not None:
            support["criteria"]["rmse_max"] = float(
                model_output(support["criteria"]["rmse_max"], schema, difference=True)
            )
        support["unit"] = public_unit(schema, schema["target"])
    scatter = row.get("prediction_scatter")
    residual = row.get("residual_scatter")
    converted = {}
    if scatter is not None:
        converted["prediction_scatter"] = {**scatter,
            "observed": model_output(scatter["observed"], schema).tolist(),
            "predicted": model_output(scatter["predicted"], schema).tolist()}
    if residual is not None:
        converted["residual_scatter"] = {**residual,
            "y": model_output(residual["y"], schema, difference=True).tolist()}
    return {
        **row,
        **converted,
        "target_metrics": _metrics(row.get("target_metrics"), schema),
        "strata": {name: _metrics(value, schema) for name, value in row.get("strata", {}).items()},
        "prediction_support": support,
    }


def _public_mapping(context):
    schema = {
        key: context[key] for key in ("variables", "target", "unit_coordinates", "input_geometry") if key in context
    }
    source = input_variables(schema)
    lines, columns = [], []
    frozen = (context.get("unit_coordinates") or {}).get("columns", {})
    for variable in source:
        name = variable["column"]
        unit = public_unit(schema, variable)
        item = frozen.get(name, {})
        a, b = item.get("scale", 1.0), item.get("offset", 0.0)
        lines.append(
            f"{name}_work [{variable.get('unit') or 'unknown'}] = {a:g} * {name} [{unit or 'unknown'}] + ({b:g})"
        )
        columns.append(
            dict(
                column=name,
                public_unit=unit,
                working_unit=variable.get("unit"),
                scale=a,
                offset=b,
                coordinate_role=variable.get("coordinate_role"),
            )
        )
    mapping = (context.get("input_geometry") or {}).get("map")
    if mapping:
        if mapping.get("kind") != "affine_chart":
            lines.append("Public formula conversion unavailable for this map; no physical equation invented")
        else:
            for name, row, b in zip(
                mapping["target_columns"], mapping["forward"]["matrix"], mapping["forward"]["offset"], strict=True
            ):
                expression = " + ".join(f"({a:g}) * {v['column']}_work" for a, v in zip(row, source, strict=True) if a)
                lines.append(f"{name}_work = {expression or '0'} + ({b:g})")
    target = context["target"]
    item = frozen.get(target["column"], {})
    a, b = item.get("scale", 1.0), item.get("offset", 0.0)
    target_unit = public_unit(schema, target)
    lines.append(
        f"{target['column']}_work [{target.get('unit') or 'unknown'}] = {a:g} * {target['column']} [{target_unit or 'unknown'}] + ({b:g})"
    )
    output = f"{target['column']} [{target_unit or 'unknown'}] = (F_work - ({b:g})) / ({a:g})"
    return schema, columns, lines, output


def _audit_view(directory, comparison, decision, decision_hash, records):
    if directory is None:
        return {"provided": False, "label": "Final監査未読込。この表示からFinal支持を判断できません。", "rows": {}}
    directory = Path(directory)
    support = directory / "audit_support.json"
    raw = directory / "audit.json"
    if support.exists() or raw.exists():
        data = read_json(support) if support.exists() else None
        if data is None:
            audit = read_json(raw)
            pool = read_json(directory / "frozen_pool.json")
            if pool.get("decision_sha256") is not None or "scored_indices" not in pool:
                raise ValueError("Full-pool audit requires its explicit support assessment")
            if pool.get("scored_ids") != [pool["ids"][i] for i in pool["scored_indices"]]:
                raise ValueError("Frozen representative indices differ")
            if pool["scored_ids"] != [row["id"] for row in audit["results"]]:
                raise ValueError("Scored representatives differ from frozen choice")
            data = {"study_sha256": audit["study_sha256"], "decision_sha256": decision_hash,
                    "audit_path": str(raw), "audit_sha256": digest(raw), "results": audit["results"],
                    "scope": "Frozen Selection representatives: test metrics only, not adoption"}
            support = raw
        if data["study_sha256"] != comparison["study_sha256"] or data["decision_sha256"] != decision_hash:
            raise ValueError("Final support belongs to a different study/decision")
        if digest(data["audit_path"]) != data["audit_sha256"]:
            raise ValueError("Final audit changed after support assessment")
        audit = read_json(data["audit_path"])
        if audit["study_sha256"] != comparison["study_sha256"]:
            raise ValueError("Final audit study differs")
        if digest(Path(data["audit_path"]).with_name("frozen_pool.json")) != audit["frozen_pool_sha256"]:
            raise ValueError("Frozen audit pool changed")
        rows = {row["id"]: row for row in data["results"]}
        prediction_path = Path(data["audit_path"]).with_name("predictions.npz")
        if digest(prediction_path) != audit["predictions_sha256"]:
            raise ValueError("Final audit predictions changed")
        import numpy as np

        pool = read_json(Path(data["audit_path"]).with_name("frozen_pool.json"))
        with np.load(prediction_path, allow_pickle=False) as arrays:
            # Older audits did not retain observations. Never reopen Final data
            # or infer missing residuals in the presentation layer.
            if "observed" in arrays:
                observed = arrays["observed"]
                sample_ids = arrays["sample_id"]
                for index, candidate_id in enumerate(pool["ids"]):
                    prefix = f"c{index}"
                    if candidate_id not in rows or f"{prefix}_prediction" not in arrays:
                        continue
                    prediction, valid = arrays[f"{prefix}_prediction"], arrays[f"{prefix}_valid"]
                    if not len(observed) == len(prediction) == len(valid) == len(sample_ids):
                        raise ValueError("Stored Final plot coordinates have different lengths")
                    rows[candidate_id]["prediction_scatter"] = dict(
                        observed=observed.tolist(), predicted=prediction.tolist(), valid=valid.tolist(),
                        sample_id=sample_ids.tolist(), scope=audit["final_evidence_role"])
                    if f"{prefix}_residual" in arrays:
                        residual = arrays[f"{prefix}_residual"]
                        if len(residual) != len(observed):
                            raise ValueError("Stored Final residual has a different length")
                        rows[candidate_id]["residual_scatter"] = dict(y=residual.tolist(), valid=valid.tolist())
        if not set(rows) <= {record["id"] for record in records}:
            raise ValueError("Audit contains different hypotheses")
        return dict(
            provided=True,
            label=data["scope"],
            role=audit["final_evidence_role"],
            rows=rows,
            source=str(support.resolve()),
            source_sha256=digest(support),
        )
    path = directory / "result_summary.json"
    if not path.exists():
        raise ValueError(
            "Use a finalized result or explicitly assessed audit_support; raw audit metrics are not an adoption decision"
        )
    data = read_json(path)
    if decision is None:
        raise ValueError("Final result display requires the explicitly bound decision")
    if data.get("decision") != decision or data.get("selected_id") != decision.get("selected_id"):
        raise ValueError("Final result and displayed decision differ")
    if digest(directory / "frozen_prediction.json") != data["frozen_prediction_sha256"]:
        raise ValueError("Frozen prediction changed")
    row = dict(
        id=data["selected_id"],
        target_metrics=data.get("final_metrics"),
        coverage=(data.get("final_metrics") or {}).get("coverage"),
        prediction_support=data.get("prediction_support"),
        strata={},
    )
    return dict(
        provided=True,
        label="Frozen predictor audit; Selection choice is unchanged",
        role=data.get("final_evidence_role"),
        rows={row["id"]: row},
        source=str(path.resolve()),
        source_sha256=digest(path),
    )


def _interpretation_views(analysis_sources, comparisons):
    """Preserve each saved analysis and bind only unambiguous same-source paths.

    Legacy comparisons omit their analysis identity. Resolve them only when the
    stored definitions identify one source; do not recompute or guess a curve.
    """
    interpretations, sources = {}, {}
    for source in analysis_sources.values():
        analysis = source["result"]
        if analysis.get("result_kind") != "interpretation.readout/v1":
            continue
        fingerprint = source["sha256"]
        sources[fingerprint] = {}
        for item in analysis.get("results", []):
            saved = {
                **item,
                "analysis_sha256": fingerprint,
                "paths": [
                    {**path, "display_id": f"{fingerprint}:{index}"}
                    for index, path in enumerate(item.get("paths", []))
                ],
            }
            interpretations.setdefault(item["id"], []).append(saved)
            sources[fingerprint][item["id"]] = saved
    bound = []
    definition_keys = ("id", "at", "direction", "steps", "path_unit", "input_units", "source_points", "unit")
    for item in comparisons:
        if item.get("axis") != "path_response_difference":
            bound.append(item)
            continue
        matches = []
        for fingerprint, records in sources.items():
            if item.get("analysis_sha256") not in (None, fingerprint):
                continue
            left = records.get(item["left"], {}).get("paths", [])
            right = records.get(item["right"], {}).get("paths", [])
            for a in left:
                if a.get("id") != item.get("path_id") or any(
                    a.get(key) != item.get(key) for key in ("steps", "path_unit", "unit")
                ):
                    continue
                for b in right:
                    if all(a.get(key) == b.get(key) for key in definition_keys):
                        matches.append(a["display_id"])
        bound.append({
            **item,
            "path_display_id": matches[0] if len(matches) == 1 else None,
            "path_binding": "same saved analysis" if len(matches) == 1 else
            "Stored path source is ambiguous or unavailable; retained without overlay",
        })
    return interpretations, bound


def build_report_view(
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
    path = Path(comparison_dir) / "comparison.json"
    comparison = read_json(path)
    context, records, rows, arrays = read_study_evidence(comparison["study"])
    if digest(Path(comparison["study"]) / "study.json") != comparison["study_sha256"]:
        raise ValueError("Comparison context changed")
    if digest(Path(comparison_dir) / "pair_arrays.npz") != comparison["pair_arrays_sha256"]:
        raise ValueError("Pair evidence changed")
    for analysis in comparison["analyses"]:
        if digest(analysis["path"]) != analysis["sha256"]:
            raise ValueError("Analysis evidence changed")
    decision, decision_hash = None, None
    if decision_dir is not None:
        decision_path = Path(decision_dir) / "decision.json"
        decision = read_json(decision_path)
        decision_hash = digest(decision_path)
        if decision["comparison_sha256"] != digest(path):
            raise ValueError("Decision belongs to different comparison evidence")
        selected = next((r for r in records if r["id"] == decision.get("selected_id")), None)
        if selected is not None and selected.get("subject") != decision.get("subject"):
            raise ValueError("Displayed decision execution view differs")
    audit = _audit_view(audit_dir, comparison, decision, decision_hash, records)
    schema, columns, mappings, output_rule = _public_mapping(context)
    unit = public_unit(schema, context["target"])
    # Explicit override allows old studies to be presented without rewriting them.
    display = context.get("display_units", {}) if display_units is None else display_units
    labels = _display_unit_labels(schema, display)
    audit = {
        **audit,
        "rows": {key: _audit_quantities(row, schema) for key, row in audit["rows"].items()},
        "unit": unit,
        "quantity_coordinate": "public",
    }
    interval = None
    if prediction_interval is None and decision and decision.get("interval_sha256"):
        prediction_interval = decision.get("interval_path")
        if prediction_interval is None:
            raise ValueError("Provide the calibration artifact assessed by the displayed use decision")
    if coverage is not None and prediction_interval is None:
        raise ValueError("Coverage requires its named prediction interval")
    if prediction_interval is not None:
        from .prediction_intervals import read_interval_evidence

        interval, _, fingerprint = read_interval_evidence(
            prediction_interval,
            decision=decision,
            decision_sha256=decision_hash,
        )
        if interval.get("study_sha256") != comparison["study_sha256"]:
            raise ValueError("Prediction interval belongs to another immutable study")
        interval = {**interval, "artifact_sha256": fingerprint, "heldout": read_json(coverage) if coverage else None}
        if coverage and interval["heldout"]["interval_sha256"] != fingerprint:
            raise ValueError("Coverage belongs to another calibration")
    supplemental = None
    if supplemental_comparison is not None:
        supplemental_path = Path(supplemental_comparison) / "comparison.json"
        supplemental = read_json(supplemental_path)
        if (
            supplemental["study_sha256"] != comparison["study_sha256"]
            or supplemental["candidates"] != comparison["candidates"]
        ):
            raise ValueError("Supplement must preserve the exact frozen hypothesis/numerical pool")
        if digest(Path(supplemental_comparison) / "pair_arrays.npz") != supplemental["pair_arrays_sha256"]:
            raise ValueError("Supplementary comparison evidence changed")
        for evidence in supplemental["analyses"]:
            if digest(evidence["path"]) != evidence["sha256"]:
                raise ValueError("Supplementary analysis changed")
    # A supplement adds evidence. It never replaces the evidence that supported
    # the frozen decision or suppresses an existing analysis/limitation.
    analysis_sources = {a["sha256"]: a for a in [*comparison["analyses"], *(supplemental or {}).get("analyses", [])]}
    # Frozen primary evidence remains authoritative for overlapping pairs;
    # same-pool supplementary comparisons add the missing requested pairs.
    display_pairs = {
        (pair["left"], pair["right"]): pair for pair in [*(supplemental or {}).get("pairs", []), *comparison["pairs"]]
    }
    analyses = [{**a["result"], "analysis_sha256": a["sha256"]} for a in analysis_sources.values()]
    analysis_comparisons = {
        value_digest(item): item
        for item in [*comparison.get("analysis_comparisons", []), *(supplemental or {}).get("analysis_comparisons", [])]
    }
    interpretations, analysis_comparisons = _interpretation_views(analysis_sources, analysis_comparisons.values())
    questions = [q for analysis in analyses for q in resolved_questions(analysis)]
    native_variables = [{**v, "column": v["column"] + "_work"} for v in context["variables"]]
    from .coordinates import source_input_values

    points = source_input_values(arrays["selection_X"], context.get("input_geometry"))
    variable = input_variables(schema)[0]
    x_schema = {**schema, "target": variable}
    residual_axis = {
        "x": model_output(points[:, 0], x_schema).tolist(),
        "x_label": f"{variable['column']} [{labels.get(variable['column'], public_unit(schema, variable))}]",
        "scope": "Selection rows at their actual covariates; not fixed-condition causal response",
    }
    cards = []
    for record, row, compared in zip(records, rows, comparison["candidates"], strict=True):
        origin = record.get("origin", {})
        summary = origin.get("summary", origin) if isinstance(origin, dict) else {}
        estimator = (origin.get("fit") or {}).get("method")
        method = origin.get("engine")
        if estimator:
            fitted = {"least_squares": "Least-squares fit", "linear_deming": "Deming regression fit"}.get(
                estimator, str(estimator) + " fit")
            method = f"{method} / {fitted}" if method else fitted
        audit_row = audit["rows"].get(record["id"])
        final = audit_row  # Already resolved once; shared by header, table and card.
        card = dict(
            id=record["id"],
            kind=record["kind"],
            subject=record.get("subject"),
            method=method or "unverified source",
            seed=origin.get("seed", record.get("native_metadata", {}).get("seed")),
            variant=record.get("variant"),
            syntax=compared.get("family"),
            hypothesis=context.get("hypothesis_groups", {}).get(record["id"]),
            eligible=row["eligible"],
            reason=row.get("reason"),
            native_check=record.get("native_check"),
            selection=_metrics(row.get("metrics", {}).get("selection"), schema),
            complexity=dict(nodes=row.get("n_nodes"), free=row.get("n_parameters")),
            checks=row.get("checks", []),
            response_evaluation=row.get("response_evaluation", [{"status": "not_evaluated", "reason": "Historical result lacks point-origin evaluation"}]),
            selection_strata={
                name: _metrics(value, schema) for name, value in row.get("strata", {}).get("selection", {}).items()
            },
            final=final,
            interpretation=next(iter(interpretations.get(record["id"], [])), None),
            interpretations=interpretations.get(record["id"], []),
            template=row.get("template"),
            local_parameter_identifiability=row.get("local_parameter_identifiability"),
            questions=[q for q in questions if q["candidate_id"] == record["id"]],
            execution_view=record.get("execution_view"),
            source=dict(
                status=summary.get("status", "not recorded"),
                native_execution_status=summary.get("native_execution_status"),
                wall_seconds=summary.get("wall_seconds"),
                response_enforcement=summary.get("response_enforcement", {"effective": "historical_not_recorded"}),
                recovery_seconds=summary.get("recovery_seconds"),
                reason=summary.get("reason", record.get("unavailable_reason")),
                termination=summary.get("termination_reason"),
                native_work=summary.get("native_work"),
                scope=summary.get("checkpoint_scope", summary.get("work_scope")),
                fit=origin.get("fit"),
                knowledge=origin.get("knowledge"),
                artifacts=summary.get("native_artifacts", origin.get("artifact")),
                effective_profile=summary.get("resolved_native_profile"),
                profile_evidence=origin.get("profile_evidence"),
            ),
            parameters=[],
            formula=None,
            public_formula=None,
            public_rule=output_rule,
            mappings=mappings,
            residual=None,
            members=[],
        )
        if record["kind"] == "explicit":
            card.update(
                formula=_formula(record["equation"], native_variables),
                public_formula=_original_formula(record["equation"], context),
                parameters=parameter_table(record["equation"]),
                equation_output=equation_outputs(record["equation"], context),
            )
        elif record["kind"] == "ensemble":
            card["formula"] = "Native ensemble: " + str(record.get("execution_view", record.get("prediction_view")))
            card["uncertainty_scope"] = record.get("uncertainty_scope")
            # Exact serialized-state groups only. They are not algebraic or mechanism groups.
            groups = {}
            for index, member in enumerate(record["members"]):
                group = value_digest(member["equation"])
                groups.setdefault(group, []).append(index)
            for index, member in enumerate(record["members"]):
                group = value_digest(member["equation"])
                indices = groups[group]
                card["members"].append(
                    dict(
                        index=index,
                        id=member.get("candidate_id", member.get("id", str(index))),
                        weight=member["weight"],
                        exact_state_group=group[:12],
                        group_count=len(indices),
                        group_mass=sum(record["members"][i]["weight"] for i in indices),
                        participates="positive native mass" if member["weight"] > 0 else "retained zero-mass member",
                        formula=_formula(member["equation"], native_variables),
                        public_formula=_original_formula(member["equation"], context),
                        parameters=parameter_table(member["equation"]),
                        equation_output=equation_outputs(member["equation"], context),
                        native_check=member.get("native_check"),
                    )
                )
        elif record["kind"] == "implicit":
            card["residual"] = dict(
                public_formula=_original_formula(record["residual"], context, residual=True),
                formula=_formula(
                    record["residual"],
                    [
                        *native_variables,
                        {"id": "target", **context["target"], "column": context["target"]["column"] + "_work"},
                    ],
                )
                + " = 0",
                parameters=parameter_table(record["residual"]),
                equation_output=equation_outputs(record["residual"], context, residual=True),
                evidence=row.get("residual_evidence"),
                branch={k: v for k, v in record.get("native_member", {}).items() if k not in {"residual", "predictor"}},
                prediction_unavailable_reason=record.get("prediction_unavailable_reason"),
            )
            card["formula"] = _formula(record.get("predictor"), native_variables)
            card["public_formula"] = _original_formula(record.get("predictor"), context)
            card["parameters"] = parameter_table(record["residual"])
            card["equation_output"] = equation_outputs(record.get("predictor"), context)
        elif record["kind"] == "native_result":
            card["native_result"] = {
                "native_execution_status": summary.get("native_execution_status", "unverified"),
                "portable_status": summary.get("portable_status", "not recorded"),
                "prediction_status": "unavailable",
                "required_semantics": "not evaluated without a faithful executable view; unavailable is not a failed law",
                "retained_native_record": record.get("native_payload", summary),
                "scope": "Saved native result fields only. Referenced binaries/assets are not executed or implicitly bundled; no portable equation is invented.",
            }
        if row.get("response_validation") is not None:
            response = row["response_validation"]
            request = context["response_validation"]
            card["response_validation"] = {
                **response,
                "pairs": request["pairs"],
                "unit": unit,
                "metrics": _metrics(response["metrics"], schema),
                "max_rmse": float(model_output(request["max_rmse"], schema, difference=True)),
                **{
                    key: [
                        float(model_output(v, schema, difference=True)) if v is not None else None
                        for v in response[key]
                    ]
                    for key in ("observed_difference", "predicted_difference")
                },
            }
        if record.get("source_schema"):
            card["source_schema"] = record["source_schema"]
        # Stored Selection residual scatter, formatted in public coordinates only.
        i = len(cards)
        if f"c{i}_selection_prediction" in arrays:
            card["prediction_scatter"] = dict(
                observed=model_output(arrays["selection_y"], schema).tolist(),
                predicted=model_output(arrays[f"c{i}_selection_prediction"], schema).tolist(),
                valid=arrays[f"c{i}_selection_valid"].tolist(),
                scope="Stored Selection rows; no Final data opened or predictions recomputed",
            )
        if f"c{i}_selection_residual" in arrays:
            residual = model_output(arrays[f"c{i}_selection_residual"], schema, difference=True)
            card["residual_scatter"] = dict(
                y=residual.tolist(),
                valid=arrays[f"c{i}_selection_valid"].tolist(),
                y_label=f"observed - prediction [{unit}]",
                scope="Selection rows at their actual covariates; not fixed-condition causal response",
            )
        cards.append(card)
    if representative_ids is not None and not set(representative_ids) <= {c["id"] for c in cards}:
        raise ValueError("Display IDs must belong to this pool")
    if representative_ids is None:
        # Display only: sort stored Selection error, preserve different syntax,
        # and keep the actual chosen subject visible independently of adoption.
        chosen, seen = [], set()
        for card in sorted(
            cards,
            key=lambda c: (
                (c["selection"] or {}).get("rmse") is None,
                (c["selection"] or {}).get("rmse") or 0,
                c["id"],
            ),
        ):
            family = card["syntax"] or card["kind"]
            if card["eligible"] and family not in seen:
                chosen.append(card["id"])
                seen.add(family)
            if len(chosen) == 3:
                break
    else:
        chosen = list(representative_ids)
    selected_id = (decision or {}).get("selected_id")
    if selected_id and selected_id not in chosen:
        chosen.insert(0, selected_id)
    marker = Path(context["prepared"]["run"]).parent.parent / "final_access.json"
    return dict(
        schema_version=1,
        question=context["question"],
        purpose=context["purpose"],
        data_kind=context.get("data_kind"),
        unit=unit,
        unit_label=labels.get(context["target"]["column"]),
        display_units=dict(display),
        display_unit_scope="Source labels only; no conversion or dimensional validation" if display else None,
        working_unit=context["target"].get("unit"),
        engineering_tolerance=float(model_output(context["engineering_tolerance"], schema, difference=True))
        if context["engineering_tolerance"] is not None
        else None,
        columns=columns,
        mappings=mappings,
        geometry=context.get("input_geometry"),
        observation_model=context.get("observation_model"),
        study_sha256=comparison["study_sha256"],
        comparison_sha256=digest(path),
        selection_rows=len(arrays["selection_y"]),
        decision=decision,
        decision_sha256=decision_hash,
        audit=audit,
        interval=interval,
        exposure=dict(
            selection_snapshot={
                **context["final_exposure"],
                "declared_history": context.get("final_history", "unknown"),
            },
            current_access_record=marker.exists(),
            current_record=read_json(marker) if marker.exists() else None,
        ),
        residual_axis=residual_axis,
        candidates=cards,
        display_ids=chosen,
        display_rule="Display only: explicit IDs or lowest stored Selection-error representatives of different syntax; selected subject pinned. Syntax diversity is not mechanism diversity. All methods/candidates remain in the table.",
        pair_display_scope="Frozen primary pairs plus explicitly requested same-pool supplemental pairs. Primary retained on overlap; supplemental results are display evidence, not a changed adoption decision.",
        pairs=[
            {
                **pair,
                "prediction_rms_difference": float(
                    model_output(pair["prediction_rms_difference"], schema, difference=True)
                )
                if pair.get("prediction_rms_difference") is not None
                else None,
                "maximum_prediction_difference": float(
                    model_output(pair["maximum_prediction_difference"], schema, difference=True)
                )
                if pair.get("maximum_prediction_difference") is not None
                else None,
                "target_unit": unit,
            }
            for pair in display_pairs.values()
        ],
        analysis_comparisons=analysis_comparisons,
        analyses=[
            {
                **{
                    k: v
                    for k, v in analysis.items()
                    if k != "results" or analysis.get("result_kind") != "interpretation.readout/v1"
                },
                "results_location": "candidate interpretation cards; original frozen evaluator artifact retained"
                if analysis.get("result_kind") == "interpretation.readout/v1"
                else "native analysis payload",
                "display_coordinate_verified": analysis.get("result_kind") != "evidence.profile/v1"
                or analysis.get("coordinate_policy") == EVIDENCE_COORDINATE_POLICY,
            }
            for analysis in analyses
        ],
        evidence_notes=[
            "Historical unit coordinates unverified: raw intervals retained but not certified for precision adoption"
            for analysis in analyses
            if analysis.get("result_kind") == "evidence.profile/v1"
            and analysis.get("coordinate_policy") != EVIDENCE_COORDINATE_POLICY
        ],
        context={
            key: context.get(key)
            for key in (
                "variables", "target", "unit_coordinates", "data_roles", "constraint_application", "runtime",
                "data_sha256", "split_sha256", "group_column", "strata_column",
            )
        },
        supplemental_comparison_sha256=digest(Path(supplemental_comparison) / "comparison.json")
        if supplemental_comparison
        else None,
    )
