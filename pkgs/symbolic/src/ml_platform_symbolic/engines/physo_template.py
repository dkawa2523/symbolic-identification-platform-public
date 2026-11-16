"""Pinned PhySO single-program shared-slot extension, in an isolated worker.

Native RNN, unit prior, LBFGS and reward remain native. Mask propagation and
nonfinite-fit rejection are explicit strict-real extensions, not protected ops.
"""

import hashlib
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

from ..equation_templates import instantiate_template
from ..units import require_canonical, unit_dimensions

OUTER_OPERATORS = {
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
    "tan",
    "asin",
    "acos",
    "atan",
    "tanh",
    "sinh",
    "cosh",
}


class SharedSlot:
    def __init__(self, request, context):
        self.template = deepcopy(request.equation_template)
        self.variables, self.target = request.variables, request.target
        if (context.provenance.get("input_geometry") or {}).get("map"):
            raise NotImplementedError("PhySO template mapped inputs are unsupported")
        if len(self.template["slots"]) != 1:
            raise NotImplementedError("PhySO template requires exactly one shared slot, not independent slots")
        if any(p.get("trainable") for p in self.template["parameters"].values()):
            raise NotImplementedError("PhySO template outer free parameters are unsupported")
        self.name, self.slot = next(iter(self.template["slots"].items()))
        self.formal = [{"id": a["name"], "column": a["name"], "unit": a["unit"]} for a in self.slot["arguments"]]
        if [(a["id"], a["unit"]) for a in self.formal] != [(v["id"], v["unit"]) for v in self.variables]:
            raise NotImplementedError("PhySO template requires identity formal input schema; no input subset")
        for record in [*self.formal, {"unit": self.slot["unit"]}]:
            require_canonical(record["unit"])
            if unit_dimensions(record["unit"]) is None:
                raise ValueError("PhySO template slot requires known canonical dimensions")
        self.calls = []

        def visit(node, inside=False):
            if "slot" in node:
                if inside:
                    raise NotImplementedError("PhySO template nested slot calls are unsupported")
                if any(set(a) != {"var"} for a in node["args"]):
                    raise NotImplementedError(
                        "PhySO template calls require variable-only arguments; composed arguments unsupported"
                    )
                self.calls.append(node["args"])
                inside = True
            if "op" in node and node["op"] not in OUTER_OPERATORS:
                raise NotImplementedError("Unsupported PhySO template outer operator: " + node["op"])
            for child in node.get("args", []):
                visit(child, inside)
            if "ref" in node:
                visit(self.template["components"][node["ref"]]["ast"], inside)

        visit(self.template["ast"])
        for p in self.template["parameters"].values():
            if "derived_from" in p:
                visit(p["derived_from"])
        if not self.calls:
            raise ValueError("PhySO template has no slot calls")
        if not set(self.slot["allowed_operators"]) <= set(request.operators):
            raise ValueError("PhySO slot operators differ from common operator set")
        if "rootn" in self.slot["allowed_operators"]:
            raise NotImplementedError("PhySO unprotected cbrt is not signed-real; template rootn unsupported")
        self.operators = self.slot["allowed_operators"]
        self.coordinate_context = {
            **context.provenance,
            "variables": request.variables,
            "target": {"column": "target", **request.target},
        }
        self.audit = {"strict_execute_calls": 0, "nonfinite_fit_failures": 0}

    def export(self, slot):
        # Native name is preserved through common parameter namespacing.
        for name, record in slot["parameters"].items():
            if record.get("trainable"):
                record["physo_native_name"] = name
        eq, checks = instantiate_template(
            self.template, {self.name: slot}, variables=self.variables, target=self.target
        )
        failed = [r for r in checks["checks"] if r["required"] and r["status"] != "pass"]
        if failed:
            raise ValueError("PhySO template portable structure/units failed: " + str(failed))
        return eq

    def source_budget(self):
        from ..coordinates import source_complexity

        def cost(ast):
            slot = {"schema_version": 1, "ast": ast, "parameters": {}}
            eq, _ = instantiate_template(self.template, {self.name: slot}, variables=self.variables)
            return source_complexity(eq, self.coordinate_context)

        # One slot leaf per call; all outer nodes and frozen transforms remain.
        overhead = cost({"const": 1.0}) - len(self.calls)
        costs = [cost({"var": v["id"]}) - overhead for v in self.formal]
        return costs, overhead, len(self.calls)

    def __call__(self, func, X):
        import torch

        indices = {v["id"]: i for i, v in enumerate(self.variables)}
        zero = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)

        def ev(node):
            if "var" in node:
                value = X[indices[node["var"]]]
            elif "const" in node:
                value = zero + node["const"]
            elif "param" in node:
                p = self.template["parameters"][node["param"]]
                value = ev(p["derived_from"]) if "derived_from" in p else zero + p["value"]
            elif "ref" in node:
                value = ev(self.template["components"][node["ref"]]["ast"])
            elif "slot" in node:
                value = func(torch.stack([ev(a) for a in node["args"]]))
            else:
                args = [ev(a) for a in node["args"]]
                op = node["op"]
                if op == "add":
                    value = args[0] + args[1]
                elif op == "sub":
                    value = args[0] - args[1]
                elif op == "mul":
                    value = args[0] * args[1]
                elif op == "div":
                    value = args[0] / args[1]
                elif op == "pow_int":
                    value = args[0] ** args[1]
                elif op == "neg":
                    value = -args[0]
                else:
                    value = getattr(torch, {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}.get(op, op))(args[0])
                valid = torch.isfinite(value)
                for arg in args:
                    valid = valid & torch.isfinite(arg)
                value = torch.where(valid, value, torch.full_like(value, float("nan")))
            return zero + value

        return ev(self.template["ast"])


def prepare(request, context):
    return SharedSlot(request, context) if request.equation_template is not None else None


@contextmanager
def strict_native_scope(template):
    if template is None:
        yield
        return
    import torch
    from physo.physym import batch_execute, execute, free_const, program

    expected = [
        (execute, "33ad14ce641c2434fa0434628ab9b510212bc86b18589deaa6fbd1d3800f4e31"),
        (program, "1baf9556796b3d06ab01ef457441d81a029174d16d3cdcbfe22bdf302bc80229"),
        (free_const, "b888642d8f338f47c20f80397744b514fc1d9ef0d371b1eb1f86dc501e58cb83"),
        (batch_execute, "dc0a57e69f7e6824bd714b90bd758f752d710f8de476193777fae37c2a73f1e9"),
    ]
    for module, digest in expected:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise ValueError("Unreviewed PhySO template source: " + module.__name__)
    original_execute, original_optimizer = execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"]

    def strict_execute(input_var_data, program_tokens, class_free_consts_vals=None, spe_free_consts_vals=None):
        template.audit["strict_execute_calls"] += 1
        zero = torch.zeros(input_var_data.shape[1], dtype=input_var_data.dtype, device=input_var_data.device)
        stack = []
        for token in reversed(program_tokens):
            if token.arity:
                args = stack[-token.arity :][::-1]
                del stack[-token.arity :]
                value = token.function(*args)
                valid = torch.isfinite(value)
                for arg in args:
                    valid = valid & torch.isfinite(arg)
                value = torch.where(valid, value, torch.full_like(value, float("nan")))
            elif token.var_type == 1:
                value = input_var_data[token.var_id]
            elif token.var_type == 3:
                value = zero + token.fixed_const
            elif token.var_type == 20:
                value = zero + class_free_consts_vals[token.var_id]
            elif token.var_type == 21:
                value = zero + spe_free_consts_vals[token.var_id]
            else:
                raise ValueError("Unsupported strict PhySO token: " + token.name)
            stack.append(value)
        if len(stack) != 1:
            raise ValueError("Incomplete strict PhySO program")
        return zero + stack[0]

    def guarded_optimizer(params, f, **kwargs):
        parameters = params if isinstance(params, list) else [params]
        hooks = []

        def objective(p):
            value = f(p)
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("Nonfinite PhySO template objective")
            # Hook rejects undefined autograd without changing/zeroing the gradient.
            if value.requires_grad and not hooks:

                def gradient_hook(gradient):
                    if not bool(torch.isfinite(gradient).all()):
                        raise FloatingPointError("Nonfinite PhySO template gradient")
                    return gradient

                for parameter in parameters:
                    if parameter.requires_grad:
                        hooks.append(parameter.register_hook(gradient_hook))
            return value

        try:
            result = original_optimizer(params, objective, **kwargs)
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Nonfinite PhySO template fitted coefficients")
            return result
        except FloatingPointError:
            template.audit["nonfinite_fit_failures"] += 1
            # Native catches local fit errors; poison failed coefficients explicitly
            # so unchanged finite initial values cannot receive a success reward.
            with torch.no_grad():
                for p in parameters:
                    p.fill_(float("nan"))
            raise
        finally:
            for hook in hooks:
                hook.remove()

    execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"] = strict_execute, guarded_optimizer
    try:
        yield
    finally:
        execute.ExecuteProgram, free_const.OPTIMIZERS["LBFGS"] = original_execute, original_optimizer


def local_budget_scope(template, operators, audit):
    if template is None:
        return nullcontext()
    from .physo_budget import source_budget_prior

    return source_budget_prior(template.slot["max_nodes"], [1] * len(template.formal), 0, operators, audit)
