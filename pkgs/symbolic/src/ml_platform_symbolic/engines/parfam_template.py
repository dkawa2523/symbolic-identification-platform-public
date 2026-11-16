"""Composite native coefficient families for a single shared template slot.

The portable compiler owns parameter identity and expansion. This module only
provides differentiable strict-real evaluation of that same expression. Native
family enumeration, optimization and coefficient reduction remain in ParFam.
"""

from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from ..equation_templates import instantiate_template
from ..expression_graph import expanded_equation

SUPPORTED_OPERATORS = {
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow_int",
    "sqrt",
    "log",
    "exp",
    "abs",
    "sin",
    "cos",
    "tanh",
    "sinh",
    "cosh",
}


def validate_template_scope(template, context=None):
    """Reject valid common features this adapter cannot execute, before imports."""
    if context is not None and (context.provenance.get("input_geometry") or {}).get("map"):
        raise NotImplementedError("ParFam template mapped-coordinate search is not supported")
    if len(template["slots"]) != 1:
        raise NotImplementedError("ParFam templates currently require exactly one shared slot")
    if any(p.get("trainable") for p in template["parameters"].values()):
        raise NotImplementedError("ParFam templates do not yet optimize outer free parameters")

    def visit(node, inside_slot=False):
        if "slot" in node:
            if inside_slot:
                raise NotImplementedError("ParFam templates do not support nested slot calls")
            inside_slot = True
        if "op" in node and node["op"] not in SUPPORTED_OPERATORS:
            raise NotImplementedError("Unsupported ParFam template operator: " + node["op"])
        for child in node.get("args", []):
            visit(child, inside_slot)
        if "ref" in node:
            visit(template["components"][node["ref"]]["ast"], inside_slot)

    visit(template["ast"])
    for parameter in template["parameters"].values():
        if "derived_from" in parameter:
            visit(parameter["derived_from"])
    return next(iter(template["slots"]))


def strict_torch(ast, parameters, X, variables):
    """Return values and propagated validity; never repair/drop invalid rows.

    Safe operands prevent NaN gradients from unused invalid branches. The mask
    still marks those rows invalid, so they cannot become a finite loss or win.
    """
    import torch

    zero = torch.zeros(len(X), dtype=X.dtype, device=X.device)
    true = torch.ones(len(X), dtype=torch.bool, device=X.device)
    indices = {v["id"]: i for i, v in enumerate(variables)}

    def ev(node):
        if "var" in node:
            value = X[:, indices[node["var"]]]
            return value, torch.isfinite(value)
        if "const" in node or "param" in node:
            value = node["const"] if "const" in node else parameters[node["param"]]
            value = zero + value
            return value, torch.isfinite(value)
        pairs = [ev(a) for a in node["args"]]
        valid = true.clone()
        for _, mask in pairs:
            valid = valid & mask
        args = [torch.where(mask, value, torch.zeros_like(value)) for value, mask in pairs]
        op, a = node["op"], args[0]
        if op == "add":
            result = a + args[1]
        elif op == "sub":
            result = a - args[1]
        elif op == "mul":
            result = a * args[1]
        elif op == "div":
            ok = args[1] != 0
            valid = valid & ok
            result = a / torch.where(ok, args[1], torch.ones_like(a))
        elif op == "pow_int":
            b = args[1]
            ok = (b == torch.round(b)) & ~((a == 0) & (b < 0))
            valid = valid & ok
            result = torch.pow(torch.where(ok, a, torch.ones_like(a)), torch.where(ok, b, torch.ones_like(b)))
        elif op in {"log", "sqrt"}:
            ok = a > 0 if op == "log" else a >= 0
            valid = valid & ok
            result = getattr(torch, op)(torch.where(ok, a, torch.ones_like(a)))
        elif op == "neg":
            result = -a
        elif op in SUPPORTED_OPERATORS:
            result = getattr(torch, op)(a)
        else:
            raise ValueError("Unsupported composite operator: " + op)
        return result, valid & torch.isfinite(result)

    return ev(ast)


class CompositeFamily:
    """Native model protocol with shared coefficients and full-template loss."""

    def __init__(self, native, template, variables, export_slot, coordinate_context=None):
        self.coordinate_context = coordinate_context
        self.native = native
        self.template = deepcopy(template)
        self.variables = variables
        self.export_slot = export_slot
        self.slot_name = validate_template_scope(template)
        sample = self.slot_equation(np.ones(native.get_number_parameters()))
        self.structure, _ = instantiate_template(template, {self.slot_name: sample}, variables=variables)
        self.ast = expanded_equation(self.structure)["ast"]
        self.groups = {}
        for record in self.structure["parameters"].values():
            group = record.get("normalization_group")
            if group is not None:
                self.groups.setdefault(group, []).append(record["native_coefficient_index"])
        self.testing_mode()

    def __getattr__(self, name):
        return getattr(self.native, name)

    def slot_equation(self, coefficients, reduced=False):
        slot = self.export_slot(self.native, coefficients, reduced=reduced)
        slot["arithmetic"] = "strict_real"
        declaration = self.template["slots"][self.slot_name]
        dimensionless = declaration["unit"] == "1" and all(
            argument["unit"] == "1" for argument in declaration["arguments"]
        )
        for key, record in slot["parameters"].items():
            record["native_coefficient_index"] = int(key[1:])
            if dimensionless and record.get("unit") is None and record.get("origin") in {
                "native_coefficient_block", "native_normalized_denominator_coordinate", "native_reduction_inactive_zero"
            }:
                record.update(unit="1", unit_origin="declared_dimensionless_template_slot_convention")
        return slot

    def equation(self, coefficients, reduced=False):
        return instantiate_template(
            self.template, {self.slot_name: self.slot_equation(coefficients, reduced)}, variables=self.variables
        )[0]

    def testing_mode(self):
        # No sample-shaped caches are retained across datasets or slot calls.
        for key in (
            "input_monomials_numerator",
            "input_monomials_denominator",
            "output_monomials_dict",
            "output_monomials_denominator_dict",
        ):
            setattr(self, key, None)
        self.native.testing_mode()

    def prepare_input_monomials(self, x):
        self.testing_mode()

    def predict(self, coefficients, x, symbolic=False):
        import torch

        if symbolic:
            raise ValueError("Use the portable equation export for symbolic composition")
        parameters = {}
        for name, record in self.structure["parameters"].items():
            index = record.get("native_coefficient_index")
            value = record["value"] if index is None else coefficients[index]
            group = record.get("normalization_group")
            if group is not None:
                norm = torch.linalg.vector_norm(coefficients[self.groups[group]])
                value = value / norm
            parameters[name] = value
        value, valid = strict_torch(self.ast, parameters, x, self.variables)
        return torch.where(valid, value, torch.full_like(value, float("nan")))

    def get_formula(self, coefficients, decimals=3, verbose=False):
        import sympy as sp

        equation = self.equation(coefficients.detach().cpu().numpy())

        def expr(node):
            if "var" in node:
                return sp.Symbol(node["var"])
            if "const" in node:
                return sp.Float(node["const"])
            if "param" in node:
                return sp.Float(equation["parameters"][node["param"]]["value"])
            a = [expr(child) for child in node["args"]]
            op = node["op"]
            if op == "add":
                return a[0] + a[1]
            if op == "sub":
                return a[0] - a[1]
            if op == "mul":
                return a[0] * a[1]
            if op == "div":
                return a[0] / a[1]
            if op == "pow_int":
                return a[0] ** a[1]
            if op == "neg":
                return -a[0]
            return getattr(sp, "Abs" if op == "abs" else op)(a[0])

        return expr(expanded_equation(equation)["ast"])

    def source_complexity(self, max_nodes):
        from ..expressions import inspect_equation

        slot = self.template["slots"][self.slot_name]
        sample = self.slot_equation(np.ones(self.get_number_parameters()))
        formal = [{"id": a["name"], "unit": a["unit"]} for a in slot["arguments"]]
        stats = inspect_equation(sample, formal)
        if stats["n_nodes"] > slot["max_nodes"] or not stats["operators"] <= set(slot["allowed_operators"]):
            return max_nodes + 1
        if self.coordinate_context is not None:
            from ..coordinates import source_complexity

            return source_complexity(self.structure, self.coordinate_context)
        return inspect_equation(self.structure, self.variables)["n_nodes"]


@contextmanager
def native_template_scope(request, context, export):
    if request.equation_template is None:
        yield
        return
    template = request.equation_template
    slot_name = validate_template_scope(template, context)
    import parfam.parfamwrapper as wrapper
    import parfam.train as train

    slot = template["slots"][slot_name]
    formal = [{"id": a["name"], "unit": a["unit"]} for a in slot["arguments"]]
    original, wrapper_original = train.setup_model, wrapper.setup_model

    def setup(parameters, n_input, device, model):
        parameters = {**parameters, "stabilize_denominator": False}
        raw = original(parameters, len(formal), device, model)

        def export_slot(native, coefficients, reduced=False):
            return export(
                native,
                coefficients,
                formal,
                [getattr(f, "__name__", str(f)) for f in native.function_names],
                reduced=reduced,
            )

        return CompositeFamily(
            raw,
            template,
            request.variables,
            export_slot,
            {**context.provenance, "variables": request.variables, "target": {"column": "target", **request.target}},
        )

    train.setup_model = wrapper.setup_model = setup
    try:
        yield
    finally:
        train.setup_model, wrapper.setup_model = original, wrapper_original
