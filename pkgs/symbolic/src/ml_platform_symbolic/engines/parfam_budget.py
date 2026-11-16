"""Extend bounded native degree families and gate source size before fitting."""

import copy
import inspect
import itertools
from contextlib import contextmanager


def polynomial_cost(powers, features):
    if not powers:
        return 1
    return (
        2 * len(powers)
        - 1
        + sum(sum(int(exponent) * (1 + features[i]) for i, exponent in enumerate(term)) for term in powers)
    )


def family_cost(model, variable_costs, target_overhead=0):
    if len(variable_costs) != model.n_input:
        raise ValueError("ParFam variable complexity mapping mismatch")
    x = list(variable_costs)
    hidden = x[:]
    for _ in range(len(model.function_names) * model.width):
        argument = polynomial_cost(model.monomial_list_input_numerator, x)
        if model.degree_input_denominator:
            argument += 1 + polynomial_cost(model.monomial_list_input_denominator, x)
        hidden.append(1 + argument)
    result = polynomial_cost(model.monomial_list_output_numerator, hidden)
    if model.degree_output_denominator:
        result += 1 + polynomial_cost(model.monomial_list_output_denominator, hidden)
    return result + target_overhead


def lower_degree_families(parameters, mode, repetitions):
    """Append missing lower-degree topologies; never collapse native repetitions.

    Function sets, width, function-specific powers, enforce_function and maximal
    potence are unchanged. This is a finite degree-grid extension, not all formulas.
    Fixed-family mode No remains exact. Existing native order is retained first.
    """
    if mode == "No":
        return [(p, "native") for p in parameters]
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("ParFam repetitions must be a positive integer")
    if len(parameters) % repetitions:
        raise ValueError("Unexpected pinned ParFam repetition layout")
    base = parameters[: len(parameters) // repetitions]
    if parameters != base * repetitions:
        raise ValueError("Unexpected pinned ParFam repetition order")

    def key(value):
        if isinstance(value, dict):
            return tuple((k, key(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(key(v) for v in value)
        return value

    seen = {key(p) for p in base}
    extra = []
    fields = (
        "degree_input_polynomials",
        "degree_output_polynomials",
        "degree_input_denominator",
        "degree_output_denominator",
    )
    for parent in base:
        ranges = []
        for field in fields:
            maximum = parent[field]
            if type(maximum) is not int or maximum < 0:
                raise ValueError("Unsupported ParFam degree specification")
            # Positive numerator degrees match the pinned native search domain.
            minimum = 1 if field.endswith("polynomials") and maximum else 0
            ranges.append(range(minimum, maximum + 1))
        for degrees in itertools.product(*ranges):
            child = copy.deepcopy(parent)
            child.update(zip(fields, degrees, strict=True))
            identity = key(child)
            if identity not in seen:
                seen.add(identity)
                extra.append(child)
    return [(p, "native") for p in parameters] + [(p, "lower_degree_extension") for p in extra] * repetitions


@contextmanager
def family_budget(native_train, max_nodes, variable_costs, target_overhead, *, audit):
    """Isolated-worker scoped hook; preserve enumeration order/repetitions."""
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    original = native_train.get_model_parameter_list
    setup = native_train.setup_model
    signature = inspect.signature(original)
    if list(signature.parameters) != ["model_parameters_max", "config", "training_parameters", "x", "y"]:
        raise ValueError("Unsupported pinned ParFam family enumeration signature")

    def bounded(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        training = bound["training_parameters"]
        mode = bound["config"]["META"]["model_parameter_search"]
        if (
            training["model"] != "ParFamTorch"
            or training["normalization"]
            or mode not in {"No", "Complete", "Max"}
            or training["parallel"]
        ):
            raise ValueError("Unsupported native ParFam complexity pathway")
        retained = []
        for parameters, origin in lower_degree_families(original(*args, **kwargs), mode, training["repetitions"]):
            model = setup(copy.deepcopy(parameters), bound["x"].shape[1], "cpu", "ParFamTorch")
            count = (
                model.source_complexity(max_nodes)
                if hasattr(model, "source_complexity")
                else family_cost(model, variable_costs, target_overhead)
            )
            audit.append(
                {
                    "nodes": count,
                    "included": count <= max_nodes,
                    "origin": origin,
                    "degrees": {
                        k: parameters[k]
                        for k in (
                            "degree_input_polynomials",
                            "degree_output_polynomials",
                            "degree_input_denominator",
                            "degree_output_denominator",
                        )
                    },
                }
            )
            if count <= max_nodes:
                retained.append(parameters)
        if not retained:
            raise ValueError("No configured ParFam family fits max_nodes before coefficient fitting")
        return retained

    native_train.get_model_parameter_list = bounded
    try:
        yield
    finally:
        native_train.get_model_parameter_list = original
