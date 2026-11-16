"""Example fourth axis over stored Selection arrays, not a noise/UQ model."""

import numpy as np


def run(context, records, rows, arrays, directory, options):
    results = []
    weights = arrays["selection_weights"]
    if not np.all(weights == weights[0]):
        return dict(
            outcome="unavailable",
            results=[],
            reason="This small example is unweighted; weighted empirical quantiles not implemented",
            result_kind="residual.empirical/v1",
        )
    for i, (record, _row) in enumerate(zip(records, rows, strict=True)):
        if options.get("ids") and record["id"] not in options["ids"]:
            continue
        key = f"c{i}_selection_residual"
        valid = arrays.get(f"c{i}_selection_valid")
        if key not in arrays or valid is None or not valid.all():
            results.append(
                dict(
                    id=record["id"],
                    outcome="unavailable",
                    reason="Incomplete/no target prediction; no residual rows removed",
                    questions=[],
                )
            )
            continue
        values = arrays[key]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite residual data")
        q = np.quantile(values, [0.25, 0.5, 0.75])
        results.append(
            dict(
                id=record["id"],
                outcome="ok",
                questions=[
                    dict(
                        kind="empirical_residual_quantiles",
                        status="established",
                        data_role="selection",
                        n_rows=len(values),
                        values=dict(q25=float(q[0]), median=float(q[1]), q75=float(q[2])),
                        unit=context["target"].get("unit"),
                        meaning="Empirical signed observed-minus-predicted residuals on stored Selection rows; not a confidence interval, noise distribution identification, or a new score for selection.",
                    )
                ],
            )
        )
    return dict(
        outcome="ok" if any(r["outcome"] == "ok" for r in results) else "unavailable",
        result_kind="residual.empirical/v1",
        results=results,
    )
