"""Same-loss affine shared-slot reduction for pinned native TorchSISSO.

Only b(x) + a*f(h(x)), a fixed nonzero scalar, is reducible here. This is not
nonlinear outer fitting or independent per-slot regression. Original template
AST, strict domain, fixed parameters and source budget survive export.
"""
from copy import deepcopy

import numpy as np

from ..coordinates import source_complexity
from ..equation_templates import instantiate_template
from ..expressions import evaluate, inspect_equation, parse_expression


def op(name, *args):
    return {"op": name, "args": list(args)}


class LinearSlot:
    def __init__(self, request, context):
        self.request, self.context = request, context
        self.template = deepcopy(request.equation_template)
        if not request.options.get("dimensional", False):
            raise NotImplementedError("TorchSISSO template requires dimensional=true; unknown coefficient units unsupported")
        if (context.provenance.get("input_geometry") or {}).get("map"):
            raise NotImplementedError("TorchSISSO template mapped inputs unsupported")
        if len(self.template["slots"]) != 1:
            raise NotImplementedError("TorchSISSO template requires one shared slot")
        if any(p.get("trainable") for p in self.template["parameters"].values()):
            raise NotImplementedError("TorchSISSO template outer free parameters unsupported")
        self.name, self.slot = next(iter(self.template["slots"].items()))
        self.variables = [{"id": a["name"], "column": a["name"], "unit": a["unit"]} for a in self.slot["arguments"]]
        self.target = {"unit": self.slot["unit"]}
        from ..units import require_canonical, unit_dimensions
        for v in [*self.variables, self.target]:
            require_canonical(v["unit"])
            if unit_dimensions(v["unit"]) is None:
                raise NotImplementedError("TorchSISSO template requires known canonical slot units")
        self.calls = []

        def resolve(node):
            if "ref" in node:
                return resolve(self.template["components"][node["ref"]]["ast"])
            if "param" in node:
                p = self.template["parameters"][node["param"]]
                return resolve(p["derived_from"]) if "derived_from" in p else deepcopy(node)
            return {"op": node["op"], "args": [resolve(a) for a in node["args"]]} if "op" in node else deepcopy(node)

        def scalar(node):
            if any(k in node for k in ["var", "slot"]):
                raise NotImplementedError("TorchSISSO slot multiplier must be a fixed scalar, not input dependent")
            for child in node.get("args", []):
                scalar(child)
            value = self.values(node, np.empty((1, len(request.variables))))[0]
            if not np.isfinite(value):
                raise ValueError("Nonfinite TorchSISSO template scalar")
            return float(value)

        def split(node):
            if "slot" in node:
                if node["slot"] != self.name or any(self.has_slot(a) for a in node["args"]):
                    raise NotImplementedError("TorchSISSO nested slots unsupported")
                args = [resolve(a) for a in node["args"]]
                if self.calls and args != self.calls[0]:
                    raise NotImplementedError("TorchSISSO shared slot calls require identical argument ASTs")
                self.calls.append(args)
                return 1., {"const": 0.}
            if not self.has_slot(node):
                return 0., node
            children = node.get("args", [])
            name = node.get("op")
            if name in {"add", "sub"}:
                a, b = split(children[0])
                c, d = split(children[1])
                return (a+c if name == "add" else a-c), op(name, b, d)
            if name == "neg":
                a, b = split(children[0])
                return -a, op("neg", b)
            if name == "mul":
                i = 0 if self.has_slot(children[0]) else 1
                if self.has_slot(children[1-i]):
                    raise NotImplementedError("TorchSISSO nonlinear outer slot products unsupported")
                factor = scalar(children[1-i])
                a, b = split(children[i])
                return factor*a, op("mul", children[1-i], b)
            if name == "div" and not self.has_slot(children[1]):
                factor = scalar(children[1])
                if factor == 0:
                    raise ValueError("TorchSISSO template division by zero")
                a, b = split(children[0])
                return a/factor, op("div", b, children[1])
            raise NotImplementedError("TorchSISSO supports only fixed affine outer slot structure")

        self.ast = resolve(self.template["ast"])
        self.scale, self.offset = split(self.ast)
        if not self.calls or not np.isfinite(self.scale) or self.scale == 0:
            raise NotImplementedError("TorchSISSO template needs a finite nonzero shared-slot scale")
        self.arguments = self.calls[0]
        if len(self.arguments) > 1 and any(not set(a) <= {"var", "const", "param"} for a in self.arguments):
            raise NotImplementedError(
                "TorchSISSO multi-argument composed calls unsupported: unused argument domain is not monotone under subset pruning"
            )
        self.coordinate_context = {**context.provenance, "variables": request.variables,
                                   "target": {"column": "target", **request.target}}
        self.operators = self.slot["allowed_operators"]
        if not set(self.operators) <= set(request.operators):
            raise ValueError("TorchSISSO slot operators differ from common operators")
        # Preflight every Train/declared-domain validation point without dropping rows.
        from ..domains import region_points
        points = np.vstack([request.X, region_points(request.X, request.variables, request.seed, request.domain)])
        self.validation_points = points
        self.transform(points)
        if not np.isfinite(self.offset_values(points)).all():
            raise ValueError("Known TorchSISSO template offset undefined on required points")
        self.X = self.transform(request.X)
        self.y = (request.y-self.offset_values(request.X))/self.scale
        if not np.isfinite(self.y).all():
            raise ValueError("TorchSISSO affine reduction overflows")
        from ..numerical_fidelity import compare_predictions, signal_scale
        self.loss_scale = signal_scale(self.y)
        restored = self.restore(self.y, request.X)
        self.reduction_check = compare_predictions(request.y, restored, scale=signal_scale(request.y),
                                                   relative_tolerance=1e-10)
        recovered = (restored-self.offset_values(request.X))/self.scale
        self.reduced_resolution_check = compare_predictions(self.y, recovered, scale=self.loss_scale,
                                                            relative_tolerance=1e-10)
        self.reduction_check["exact_observed_roundtrip"] = bool(np.array_equal(request.y, restored))
        self.reduced_resolution_check["exact_observed_roundtrip"] = bool(np.array_equal(self.y, recovered))
        if not all(check["passed"] or check["exact_observed_roundtrip"]
                   for check in [self.reduction_check, self.reduced_resolution_check]):
            raise ValueError("Float64 cannot resolve TorchSISSO affine loss reduction; change coordinates/known outer scale")
        if np.ptp(self.y) == 0 or np.any(np.ptp(self.X, axis=0) == 0):
            raise ValueError("TorchSISSO reduced target/inputs have zero Train variance")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale_squared = self.loss_scale*self.loss_scale
            native_energy = np.sum((self.y-self.y.mean())**2)
            input_std = np.std(self.X, axis=0)
        if not np.isfinite([scale_squared, native_energy]).all() or scale_squared <= 0 or native_energy <= 0:
            raise ValueError("TorchSISSO reduced target outside native Float64 quadratic range; change coordinates")
        if not np.isfinite(input_std).all() or np.any(input_std <= 0):
            raise ValueError("TorchSISSO reduced inputs outside native Float64 standardization range")

    @staticmethod
    def has_slot(node):
        return "slot" in node or any(LinearSlot.has_slot(a) for a in node.get("args", []))

    def values(self, ast, X):
        return evaluate({"schema_version": 2, "ast": ast, "parameters": {n: p for n, p in self.template["parameters"].items() if "derived_from" not in p}},
                        X, self.request.variables)

    def transform(self, X):
        values = np.column_stack([self.values(a, X) for a in self.arguments])
        if not np.isfinite(values).all():
            raise ValueError("TorchSISSO template input transform undefined; no rows dropped")
        return values

    def offset_values(self, X):
        return self.values(self.offset, X)

    def restore(self, prediction, X):
        # Preserve actual outer operation order, including repeated shared calls;
        # the reduced linear matrix is mathematical, not a floating-point rewrite.
        key = "__torchsisso_slot_value"
        ids = {v["id"] for v in self.request.variables}
        while key in ids:
            key += "_"
        def bind(node):
            if "slot" in node:
                return {"var": key}
            return {"op": node["op"], "args": [bind(a) for a in node["args"]]} if "op" in node else node
        parameters = {n: p for n, p in self.template["parameters"].items() if "derived_from" not in p}
        return evaluate({"schema_version": 2, "ast": bind(self.ast), "parameters": parameters},
                        np.column_stack([X, prediction]), [*self.request.variables, {"id": key, "column": key}])

    def export(self, equation):
        full, checks = instantiate_template(self.template, {self.name: equation},
                                             variables=self.request.variables, target=self.request.target)
        failed = [x for x in checks["checks"] if x["required"] and x["status"] != "pass"]
        if failed:
            raise ValueError("TorchSISSO template structure/units failed: " + str(failed))
        return full

    def validate_response(self, compiled):
        if compiled is not None:
            points = np.asarray(compiled["points"])
            self.validation_points = np.vstack([self.validation_points, points])
            self.transform(points)
            if not np.isfinite(self.offset_values(points)).all():
                raise ValueError("Known TorchSISSO offset undefined on response stencil")

    def response(self, compiled, device, iterations):
        from ..response_search import response_feasible
        from .torchsisso_oracle import descriptor_values
        from .torchsisso_response import ResponseSearch
        self.validate_response(compiled)
        original_points = np.asarray(compiled["points"])
        fit = deepcopy(compiled)
        fit["points"] = self.transform(original_points).tolist()
        offset = self.offset_values(original_points)
        for term in fit["terms"]:
            shift = float(np.dot(offset[term["indices"]], term["coefficients"]))
            term["coefficients"] = [self.scale*c for c in term["coefficients"]]
            for key in ["lower", "upper"]:
                if term[key] is not None:
                    term[key] -= shift
            if not np.isfinite(shift) or not np.isfinite(term["coefficients"]).all():
                raise ValueError("Unrepresentable TorchSISSO template response transformation")
            if any(term[key] is not None and not np.isfinite(term[key]) for key in ["lower", "upper"]):
                raise ValueError("Unrepresentable TorchSISSO template response bounds")
        template = self

        class FullResponse(ResponseSearch):
            def feasible_model(self, ids, intercept, coefficients):
                from .torchsisso import _equation
                slot = _equation([self.reg.names[int(i)] for i in ids], coefficients, intercept,
                                 template.variables, template.target["unit"])
                full = template.export(slot)
                return response_feasible(evaluate(full, original_points, template.request.variables), compiled)

        result = FullResponse(fit, lambda names: descriptor_values(names, np.asarray(fit["points"]),
                              self.variables, device=device).detach().cpu().numpy(), iterations)
        result.objective_scale = self.loss_scale
        return result

    def budget(self):
        return TemplateBudget(self)


class TemplateBudget:
    """Native subset gate: slot affine size AND fully expanded original AST."""
    def __init__(self, template):
        self.template = template
        self.max_nodes = template.slot["max_nodes"]

    def costs(self, names):
        self.names = names
        points = self.template.transform(self.template.validation_points)
        equations = [parse_expression(n.replace("ln(", "log("), self.template.variables, trainable=False) for n in names]
        self.valid = [bool(np.isfinite(evaluate(eq, points, self.template.variables)).all()) for eq in equations]
        return [inspect_equation(eq, self.template.variables)["n_nodes"] for eq in equations]

    def select(self, indices):
        self.names = [self.names[i] for i in indices]
        self.valid = [self.valid[i] for i in indices]

    def admissible(self, indices, costs):
        from .torchsisso import _equation
        if not all(self.valid[int(i)] for i in indices):
            return False
        if 1+sum(costs[int(i)]+3 for i in indices) > self.max_nodes:
            return False
        limit = self.template.request.limits.get("max_free_parameters")
        if limit is not None and len(indices)+1 > limit:
            return False
        try:
            slot = _equation([self.names[int(i)] for i in indices], [1.]*len(indices), 0.,
                             self.template.variables, self.template.target["unit"])
            full = self.template.export(slot)
            return source_complexity(full, self.template.coordinate_context) <= self.template.request.limits["max_nodes"]
        except ValueError:
            return False


def prepare(request, context):
    return LinearSlot(request, context) if request.equation_template is not None else None
