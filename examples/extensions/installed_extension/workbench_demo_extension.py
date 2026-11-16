"""Installed extension example: disconnected region and real NumPy polynomial fit.

This is a declared polynomial family, not general symbolic discovery. Initialization
is explicitly requested by the application/CLI; model loading never imports it.
"""

import numpy as np

__version__ = "1.0.0"


def contains(X, settings):
    return np.logical_or.reduce([(X[:, 0] >= a) & (X[:, 0] <= b) for a, b in settings["intervals"]])


def validation_points(seed, settings):
    return [[value] for interval in settings["intervals"] for value in interval]


def initialize():
    from ml_platform_symbolic.domains import register_region
    from ml_platform_symbolic.engines.registry import register_engine
    from ml_platform_symbolic.equation_evaluation.api import register_evaluator

    register_region("demo_disconnected", __version__, contains=contains, validation_points=validation_points)
    register_engine("numpy_polynomial_demo", __name__)
    register_evaluator("region_membership_demo", axis="domain_membership", module=__name__, options=set())


def capabilities():
    return dict(
        operators=["add", "mul", "pow_int"],
        constraints={"units": "postchecked", "domain": "provided feasible points", "max_nodes": "postchecked"},
        requires_assets=False,
        devices=["cpu"],
    )


def discover(request, context):
    from ml_platform_symbolic.data import write_json
    from ml_platform_symbolic.engines.base import DiscoveryResult, checked_candidate, validate_options

    validate_options(request.options, {"degree"})
    degree = request.options.get("degree", 2)
    if request.X.shape[1] != 1 or type(degree) is not int or not 1 <= degree <= 4:
        raise ValueError("Example supports one input and an explicitly chosen degree1..4")
    # The example deliberately has a narrow, truthful unit boundary. A production
    # adapter can use the common dimensional helpers for its own native basis.
    if request.variables[0].get("unit") != "1" or request.target.get("unit") != "1":
        raise ValueError("Demo polynomial coefficients are dimensionless only")
    native = np.polynomial.Polynomial.fit(request.X[:, 0], request.y, degree)
    coefficients = native.convert().coef
    terms, parameters = [], {}
    for power, value in enumerate(coefficients):
        name = f"c{power}"
        parameters[name] = dict(
            value=float(value), unit="1", trainable=True, origin="numpy_polynomial_fit", bounds=[None, None]
        )
        term = {"param": name}
        if power:
            feature = {"var": request.variables[0]["id"]}
            if power > 1:
                feature = {"op": "pow_int", "args": [feature, {"const": power}]}
            term = {"op": "mul", "args": [term, feature]}
        terms.append(term)
    ast = terms[0]
    for term in terms[1:]:
        ast = {"op": "add", "args": [ast, term]}
    equation = dict(schema_version=1, ast=ast, parameters=parameters)
    candidate = checked_candidate("numpy_polynomial_demo", 0, equation, request, lambda X: native(X[:, 0]))
    artifact = write_json(
        context.work_dir / "numpy_native_fit.json",
        dict(
            algorithm="numpy.polynomial.Polynomial.fit",
            numpy_version=np.__version__,
            degree=degree,
            native_coefficients=native.coef.tolist(),
            domain=native.domain.tolist(),
            window=native.window.tolist(),
            converted_coefficients=coefficients.tolist(),
            assumption="Declared polynomial family and fixed-X least squares; no unknown-family or EIV claim",
        ),
    )
    return DiscoveryResult(
        [candidate],
        "ok",
        dict(native_execution_status="completed", effective_degree=degree),
        {"native_fit": str(artifact)},
    )


def run(context, records, rows, arrays, directory, options):
    from ml_platform_symbolic.coordinates import execution_region
    from ml_platform_symbolic.domains import region_mask

    region = execution_region(context.get("input_geometry"), context["variables"])
    mask = region_mask(arrays["domain_X"], region)
    return dict(
        outcome="ok",
        result_kind="domain.membership/v1",
        results=[
            dict(
                id=r["id"],
                outcome="ok",
                n_points=len(mask),
                all_feasible=bool(mask.all()),
                questions=[],
                scope="Explicit initialized region evaluated in this separate worker",
            )
            for r in records
        ],
    )
