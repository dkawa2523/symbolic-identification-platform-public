"""Actual Effector0.6 effects with explicit common frames, centering and absolute level."""

from importlib.metadata import version

import numpy as np

from ...domains import within_bounds
from ..numerical import finite_jacobian, predict_record, prerequisite_failure


def run(context, records, rows, arrays, directory, options):
    import effector

    if version("effector") != "0.6.0":
        raise ValueError("This verified bridge requires effector0.6.0")
    kind = options.get("kind", "rhale")
    if kind not in {"ale", "rhale"}:
        raise ValueError("Effector kind is ale or rhale")
    center = options.get("centering", "zero_start")
    if center not in {False, "zero_start", "zero_integral"}:
        raise ValueError("Centering is false, zero_start or zero_integral")
    bins, grid_points = options.get("bins", 8), options.get("grid_points", 33)
    if type(bins) is not int or not 2 <= bins <= 64 or type(grid_points) is not int or not 2 <= grid_points <= 256:
        raise ValueError("Explicit Fixed bins2..64 and grid_points2..256 required")
    role = context["data_roles"]["behavior_reference"]
    X = arrays[role + "_X"]
    variables, target = context["variables"], context["target"]
    columns = [v["column"] for v in variables]
    features = options.get("features", columns)
    if not set(features) <= set(columns):
        raise ValueError("Features use original column names")
    limits = np.vstack([X.min(axis=0), X.max(axis=0)])
    if (limits[0] == limits[1]).any():
        return {
            "outcome": "unavailable",
            "reason": "Constant reference feature; no continuous effect frame",
            "results": [],
            "result_kind": "behavior.effects/v1",
        }
    schema = {"feature_names": columns, "feature_types": ["continuous"] * len(columns), "target_name": target["column"]}
    results = []
    for record, row in zip(records, rows, strict=True):
        if options.get("ids") and record["id"] not in options["ids"]:
            continue
        failure = prerequisite_failure(record, row)
        if failure:
            results.append({"id": record["id"], "outcome": "unavailable", "reason": failure})
            continue

        if row.get("prediction_available") is False:
            results.append(
                {"id": record["id"], "outcome": "unavailable", "reason": "Behavior effects require an X-only prediction view"}
            )
            continue

        def model(points, record=record):
            prediction, valid = predict_record(record, points, variables, target)
            return np.where(valid, prediction, np.nan)

        def jacobian(points, record=record):
            return finite_jacobian(record, points, variables, target, context["derivative_steps"])

        kwargs = {
            "data": X,
            "model": model,
            "nof_instances": "all",
            "axis_limits": limits,
            "schema": schema,
            "random_state": 42,
        }
        engine = (
            effector.RHALE(**kwargs, model_jac=jacobian, data_effect=jacobian(X))
            if kind == "rhale"
            else effector.ALE(**kwargs)
        )
        if not np.array_equal(engine.indices, np.arange(len(X))):
            raise ValueError("Native axis limits/ingestion changed rows; mask alignment not established")
        curves, errors = [], []
        for feature in features:
            k = columns.index(feature)
            grid = np.linspace(limits[0, k], limits[1, k], grid_points)
            engine.fit(
                k,
                centering=center,
                binning_method=effector.axis_partitioning.Fixed(nof_bins=bins, min_points_per_bin=0),
            )
            engine.eval(k, grid, centering=center)  # RHALE bins are materialized lazily.
            # Pinned0.6.0 cache exposes the EFFECTIVE fixed frame, including endpoint epsilon.
            summaries = [v for v in engine._summaries.values() if isinstance(v, dict) and "limits" in v]
            native_limits = summaries[-1]["limits"]
            for name, bounds in context["regions"].items():
                mask = np.ones(len(X), dtype=bool)
                for j, variable in enumerate(variables):
                    mask &= within_bounds(X[:, j], bounds.get(variable["column"]))
                if mask.sum() < 2:
                    errors.append(f"{feature}/{name}: fewer than two supported reference rows")
                    continue
                effect = engine.eval(k, grid, centering=center, mask=mask)
                heter = engine.eval_heter(k, grid, mask=mask)
                counts = np.histogram(X[mask, k], bins=native_limits)[0]
                if not np.isfinite(effect).all() or not np.isfinite(heter).all():
                    errors.append(f"{feature}/{name}: nonfinite native effect, no rows dropped")
                    continue
                curves.append(
                    {
                        "feature": feature,
                        "region": name,
                        "x": grid.tolist(),
                        "effect": effect.tolist(),
                        "heterogeneity": heter.tolist(),
                        "native_bin_limits": native_limits.tolist(),
                        "points_per_bin": counts.tolist(),
                        "n_reference_rows": int(mask.sum()),
                        "centering": center,
                        "effect_unit": target["unit"],
                        "input_unit": variables[k]["unit"],
                        "heterogeneity_unit": f"({target['unit']}/{variables[k]['unit']})^2",
                        "heterogeneity_meaning": "conditional slope variance, not coefficient CI or predictive uncertainty",
                        "empty_bins": "native interpolation may fill empty bins; counts0 remain unsupported",
                    }
                )
        prediction = model(X)
        results.append(
            {
                "id": record["id"],
                "outcome": "ok" if curves and not errors else "unavailable",
                "reason": "; ".join(errors) or None,
                "curves": curves,
                "absolute_reference_X": X[0].tolist(),
                "absolute_reference_prediction": float(prediction[0]),
                "mean_prediction": float(np.mean(prediction)),
                "reference_role": role,
                "jacobian_method": context["derivative_method"] if kind == "rhale" else "native ALE finite changes",
            }
        )
    return {
        "outcome": "ok" if any(r["outcome"] == "ok" for r in results) else "unavailable",
        "result_kind": "behavior.effects/v1",
        "method_variant": kind,
        "version": version("effector"),
        "license": "MIT",
        "axis_limits": limits.tolist(),
        "binning": {"kind": "Fixed", "count": bins},
        "reference_role": role,
        "results": results,
        "scope": "conditional effects on declared reference data; no causal or global physics guarantee",
    }
