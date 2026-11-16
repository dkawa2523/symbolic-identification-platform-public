"""Declared iid-Gaussian likelihood and real Julia profiles, separate from source candidates."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from ...constraints import equation_unit
from ...data import digest, read_json, write_json
from ...data import value_digest as identity
from ...domains import region_mask, within_bounds
from ...expression_graph import expanded_equation
from ...expressions import inspect_equation, parse_expression
from ...numerical_fidelity import compare_predictions, signal_scale
from ...parameters import (
    expression_coordinate_role,
    free_parameter_bounds,
    free_parameter_names,
    parameter_coordinate_role,
    with_parameters,
)
from ...question_results import conditional_request, profile_questions
from ...units import (
    EVIDENCE_COORDINATE_POLICY,
    convert_width,
    quantity_conversion,
    quantity_role,
    unit_spec,
)
from ..numerical import intersect_schema, prerequisite_failure


def _parameter_scale(value, bounds):
    if all(v is not None for v in bounds):
        span = bounds[1] - bounds[0]
        if np.isfinite(span) and span > 0:
            return span
        span = max(abs(v) for v in bounds)  # Overflow-safe coordinate for extreme finite bounds.
        if np.isfinite(span) and span > 0:
            return span
    if np.isfinite(value) and value != 0:
        return abs(value)
    raise ValueError(
        "Likelihood conditioning needs finite bounds or a nonzero initial parameter scale; declare physical bounds"
    )


def annotate_crossings(profile, baseline, increment):
    """Native Identifiable can stop at a coarse scan tolerance, short of the NLL level."""
    endpoint_source = profile.get("endpoint_refinement", profile)
    tolerance = max(1e-4, increment * 1e-3)
    accepted, deltas = {}, {}
    for side in ("left", "right"):
        value = endpoint_source.get("native_endpoints", {}).get(side)
        candidates = [i for i, x in enumerate(endpoint_source.get("x", [])) if value is not None and x == value]
        objective = endpoint_source.get("objective", [])
        delta = next(
            (objective[i] - baseline for i in candidates if i < len(objective) and objective[i] is not None), None
        )
        verified = (
            endpoint_source.get("retcodes", {}).get(side) == "Identifiable"
            and delta is not None
            and np.isfinite(delta)
            and abs(delta - increment) <= tolerance
        )
        accepted[side] = value if verified else None
        deltas[side] = delta
    profile["accepted_endpoints"] = accepted
    profile["endpoint_validation"] = {
        "all_verified": all(value is not None for value in accepted.values()),
        "delta_nll": deltas,
        "threshold_increment": increment,
        "objective_tolerance": tolerance,
    }
    profile["crossing_status"] = (
        "verified threshold crossings; conditional interval"
        if profile["endpoint_validation"]["all_verified"]
        else "threshold crossing not verified in all directions; interval unavailable, native coordinates retained"
    )


def run(context, records, rows, arrays, directory, options):
    executable = shutil.which("julia") or os.getenv("PYTHON_JULIAPKG_EXE")
    project = os.getenv("SYMBOLIC_LIKELIHOOD_PROJECT")
    if not executable or not project:
        return {
            "outcome": "unavailable",
            "reason": "Use the equation-evidence local Docker profile",
            "results": [],
            "result_kind": "evidence.profile/v1",
            "coordinate_policy": EVIDENCE_COORDINATE_POLICY,
        }
    sigma = float(
        convert_width(
            options["sigma"], options.get("sigma_unit", context["target"].get("unit")), context["target"].get("unit")
        )
    )
    confidence = float(options.get("confidence_level", 0.95))
    if not np.isfinite(sigma) or sigma <= 0 or not 0 < confidence < 1:
        raise ValueError("Known Gaussian sigma is positive in original target units; confidence in(0,1)")
    requested_ranges = options.get("parameter_ranges", {})
    increment = float(chi2.ppf(confidence, 1) / 2)  # NLL, NOT -2logL.
    results = []
    directory = Path(directory)
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        if options.get("ids") and record["id"] not in options["ids"]:
            continue
        if record["kind"] != "explicit":
            results.append(
                {
                    "id": record["id"],
                    "outcome": "unavailable",
                    "reason": "This evaluator requires an explicit parameterized equation; native output remains available to prediction/behavior comparison",
                }
            )
            continue
        failure = prerequisite_failure(record, row)
        if failure:
            results.append({"id": record["id"], "outcome": "unavailable", "reason": failure})
            continue
        try:
            record = {**record, "equation": expanded_equation(record["equation"])}
            names = free_parameter_names(record["equation"], require_origin=True)
            if not names or not set(requested_ranges) <= set(names):
                raise ValueError("Profiles require audited free parameters and explicit compatible scan ranges")
            parameters = record["equation"]["parameters"]
            bounds = dict(zip(names, free_parameter_bounds(record["equation"], names), strict=True))
            output_units = options.get("parameter_units", {})
            if set(output_units) - set(requested_ranges):
                raise ValueError("Parameter output units require a named requested profile")
            transforms = {
                name: quantity_conversion(
                    parameters[name].get("unit"),
                    output_units.get(name, parameters[name].get("unit")),
                    parameter_coordinate_role(parameters, name),
                )
                for name in requested_ranges
            }
            ranges = {
                name: [(v - transforms[name][1]) / transforms[name][0] for v in limits]
                for name, limits in requested_ranges.items()
            }
            for name, limits in ranges.items():
                low, high = bounds[name]
                if (
                    len(limits) != 2
                    or not np.isfinite(limits).all()
                    or limits[0] >= limits[1]
                    or low is not None
                    and limits[0] < low
                    or high is not None
                    and limits[1] > high
                ):
                    raise ValueError("Profile scan must be finite, ordered and within physical parameter bounds")
            quantities = []
            for quantity in options.get("quantities", []):
                low, high = quantity["range"]
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ValueError("Functional scan range must be finite and ordered")
                scan_tolerance = quantity.get("scan_tolerance", (high - low) * 1e-10)
                if not np.isfinite(scan_tolerance) or not 0 < scan_tolerance < high - low:
                    raise ValueError(
                        "Functional scan_tolerance is positive in quantity units and smaller than its range"
                    )
                if "prediction_at" in quantity:
                    point = [quantity["prediction_at"][v["column"]] for v in context["variables"]]
                    variables, _ = intersect_schema(record, context)
                    if (
                        not np.isfinite(point).all()
                        or not region_mask(np.asarray([point]), record.get("execution_domain"))[0]
                        or any(
                            not within_bounds(np.asarray([x]), v.get("domain"))[0]
                            for x, v in zip(point, variables, strict=True)
                        )
                    ):
                        raise ValueError("Functional prediction point is outside the common/source input domain")
                    ast_node = record["equation"]["ast"]
                else:
                    point = []
                    equation = parse_expression(quantity["expression"], [], trainable=False, parameters=parameters)
                    ast_node = equation["ast"]
                source_unit, errors = equation_unit(
                    {"schema_version": 1, "ast": ast_node, "parameters": parameters},
                    context["variables"] if point else [],
                )
                if point:
                    # A prediction is in its declared saved target coordinates.
                    # Arithmetic dimension inference is not a second quantity
                    # authority (e.g. coherent angular factors need not infer
                    # a unique physical output kind). Eligibility already owns
                    # unit constraints; do not relabel an arbitrary functional AST.
                    source_unit = unit_spec(context["target"].get("unit"))
                inferred_role = (
                    quantity_role(context["target"])
                    or expression_coordinate_role(record["equation"], context["variables"])
                    if point
                    else expression_coordinate_role({"ast": ast_node, "parameters": parameters}, [])
                )
                requested_role = quantity.get("coordinate_role")
                if requested_role is not None and requested_role not in {"value", "difference"}:
                    raise ValueError("Functional coordinate_role must be value or difference")
                if requested_role is not None and inferred_role is not None and requested_role != inferred_role:
                    raise ValueError("Functional role conflicts with known point/difference arithmetic")
                effective_role = requested_role if requested_role is not None else inferred_role
                wanted = unit_spec(quantity.get("unit"))
                if effective_role is None and wanted is not None and wanted.role is not None:
                    effective_role = wanted.role
                if errors:
                    raise ValueError("Functional quantity has invalid unit coordinates: " + "; ".join(errors))
                status = "unknown" if source_unit is None or wanted is None else "pass"
                if status == "unknown":
                    # No conversion can be established. Numerical result is explicitly untyped.
                    if quantity.get("unit") is not None:
                        raise ValueError("Functional quantity unit conversion is unknown; no relabeling allowed")
                    transform = (1.0, 0.0)
                else:
                    transform = quantity_conversion(source_unit, wanted, effective_role)
                quantities.append(
                    {
                        "id": quantity["id"],
                        "request": conditional_request(
                            {
                                "prediction_at": quantity["prediction_at"],
                                "input_units": {v["column"]: v.get("unit") for v in context["variables"]},
                            }
                            if "prediction_at" in quantity
                            else {"expression": quantity["expression"]},
                            confidence_level=confidence,
                            observation_model={
                                "family": "iid_gaussian_known_sigma",
                                "sigma": sigma,
                                "unit": context["target"].get("unit"),
                            },
                        ),
                        "ast": ast_node,
                        "X": point,
                        "range": quantity["range"],
                        "scan_tolerance": scan_tolerance,
                        "unit": quantity.get("unit"),
                        "unit_status": status,
                        "coordinate_role": effective_role,
                        "coordinate_conversion": {
                            "scale": transform[0],
                            "offset": transform[1],
                            "meaning": "requested quantity = source AST quantity * scale + role-dependent offset; range and endpoints in requested unit",
                        },
                        "definition": quantity.get(
                            "definition", "user-defined functional; no physical characteristic certified"
                        ),
                    }
                )
            if not ranges and not quantities:
                raise ValueError("Request named parameter ranges or functional quantities explicitly")
            job = {
                "equation": record["equation"],
                "free_names": names,
                "variable_ids": [v["id"] for v in context["variables"]],
                "X": arrays["train_X"].tolist(),
                "y": arrays["train_y"].tolist(),
                "sigma": sigma,
                "target_domain": intersect_schema(record, context)[1].get("domain"),
                "lower": [bounds[name][0] for name in names],
                "upper": [bounds[name][1] for name in names],
                "parameter_scales": [_parameter_scale(parameters[name]["value"], bounds[name]) for name in names],
                "parameter_ranges": ranges,
                "parameter_steps": {name: options.get("step", 0.1) / transforms[name][0] for name in ranges},
                "quantities": quantities,
                "threshold_increment": increment,
                "step": options.get("step", 0.1),
                "maxiters": options.get("maxiters", 500),
            }
            job_path = write_json(directory / f"job_{index}.json", job)
            raw_path = directory / f"native_{index}.json"
            subprocess.run(
                [
                    executable,
                    "--startup-file=no",
                    f"--project={project}",
                    str(Path(__file__).with_name("likelihood_worker.jl")),
                    str(job_path),
                    str(raw_path),
                ],
                check=True,
            )
            native = read_json(raw_path)
            if native["version"] != "1.6.1":
                raise ValueError("Unverified LikelihoodProfiler runtime version")
            source_prediction = np.asarray(native["source_predictions"], dtype=float)
            fidelity = compare_predictions(
                source_prediction,
                arrays[f"c{index}_train_prediction"],
                scale=signal_scale(arrays["train_y"]),
                relative_tolerance=1e-9,
            )
            if not fidelity["passed"]:
                raise ValueError("Julia/source signal-scaled numerical fidelity failed")
            native["source_fidelity"] = fidelity
            if native["outcome"] != "ok":
                results.append({"id": record["id"], "outcome": "failed", "native": native})
                continue
            for profile in native["profiles"] + native["quantities"]:
                if "objective" in profile:
                    profile["delta_nll"] = [
                        None if v is None else float(v - native["objective_min"]) for v in profile["objective"]
                    ]
                    annotate_crossings(profile, native["objective_min"], increment)
            for profile in native["profiles"]:
                name = profile["name"]
                scale, offset = transforms[name]
                profile["source_unit"] = parameters[name].get("unit")
                profile["coordinate_role"] = parameter_coordinate_role(parameters, name)
                profile["unit"] = output_units.get(name, parameters[name].get("unit"))
                profile["coordinate_conversion"] = {
                    "scale": scale,
                    "offset": offset,
                    "native_artifact": "unchanged native_* file; nuisance parameter arrays retain source coordinates",
                }
                for part in [profile, profile.get("endpoint_refinement", {})]:
                    if "x" in part:
                        part["x"] = [None if v is None else v * scale + offset for v in part["x"]]
                    for key in ("native_endpoints", "accepted_endpoints"):
                        if key in part:
                            part[key] = {
                                side: None if v is None else v * scale + offset for side, v in part[key].items()
                            }
                    if "endpoint_scan_tolerance" in part:
                        part["endpoint_scan_tolerance"] *= abs(scale)
            for profile, definition in zip(native["quantities"], quantities, strict=True):
                profile["unit_status"] = definition["unit_status"]
                profile["coordinate_role"] = definition["coordinate_role"]
                profile["unit"] = definition["unit"]
                profile["definition"] = definition["definition"]
                profile["request"] = definition["request"]
                profile["coordinate_conversion"] = definition["coordinate_conversion"]
            derived = with_parameters(record["equation"], names, native["optimum"])
            inspect_equation(derived, context["variables"])
            fit = {
                "kind": "derived_equation",
                "equation": derived,
                "variant": "likelihood_fit",
                "parent": {
                    "id": record["id"],
                    "equation_sha256": identity(record["equation"]),
                    "study_sha256": context["study_sha256"],
                },
                "variables": record.get("variables", context["variables"]),
                "target": intersect_schema(record, context)[1],
                "data_sha256": context["data_sha256"],
                "split_sha256": context["split_sha256"],
                "fit_role": "train",
                "native_artifact": str(raw_path),
                "native_sha256": digest(raw_path),
                "likelihood": {
                    "family": "iid_gaussian_known_sigma",
                    "sigma": sigma,
                    "normalization": "full negative log likelihood in original target units",
                },
            }
            fit_path = write_json(directory / f"fit_{index}.json", fit)
            results.append(
                {
                    "id": record["id"],
                    "outcome": "ok"
                    if any("objective" in p for p in native["profiles"] + native["quantities"])
                    else "unavailable",
                    "reason": None
                    if any("objective" in p for p in native["profiles"] + native["quantities"])
                    else "Native fit completed, but requested profiles/endpoints are unavailable",
                    "native": native,
                    "fit_artifact": str(fit_path),
                    "fit_sha256": digest(fit_path),
                    "confidence_level": confidence,
                    "threshold_increment": increment,
                    "unit": context["target"]["unit"],
                    "meaning": "conditional on this structure, declared noise, bounds and Train data; not structure probability",
                }
            )
        except subprocess.CalledProcessError as exc:
            results.append({"id": record["id"], "outcome": "failed", "reason": str(exc)})
        except ValueError as exc:
            results.append({"id": record["id"], "outcome": "unavailable", "reason": str(exc)})
    for item in results:
        item["questions"] = profile_questions(item, coordinate_policy=EVIDENCE_COORDINATE_POLICY)
    return {
        "outcome": "ok"
        if any(r["outcome"] == "ok" for r in results)
        else "failed"
        if any(r["outcome"] == "failed" for r in results)
        else "unavailable",
        "result_kind": "evidence.profile/v1",
        "coordinate_policy": EVIDENCE_COORDINATE_POLICY,
        "results": results,
        "assumptions": "declared iid Gaussian known sigma; intervals conditional/asymptotic unless analytically justified",
        "interval_policy": "native endpoints/statuses preserved; absent crossing is not a proof of unboundedness",
        "license": "MIT",
        "julia_worker_sha256": digest(Path(__file__).with_name("likelihood_worker.jl")),
        "julia_manifest_sha256": digest(Path(project) / "Manifest.toml")
        if (Path(project) / "Manifest.toml").is_file()
        else None,
    }
