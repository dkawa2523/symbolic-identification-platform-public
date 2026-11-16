"""Differentiable sampled constraints used by native ParFam family optimization."""
from __future__ import annotations


class _InvalidResponseTrial(Exception):
    """A numerical trial is undefined; never replace or discard its points."""


class ResponseObjective:
    """The pinned runtime supplies the current native model and full parameters."""

    def __init__(self, compiled):
        self.compiled = compiled
        self.calls = 0
        self.solver_outcomes = {}
        self.no_free_parameter_solves = 0

    def _record_solver(self, kind, result):
        """Count returned optimizer results, never infer a candidate's convergence."""
        counts = self.solver_outcomes.setdefault(kind, {"returned": 0, "success": 0,
            "not_success": 0, "success_not_reported": 0, "stop_reasons": {}})
        counts["returned"] += 1
        success = result.get("success")
        counts["success_not_reported" if success is None else "success" if bool(success) else "not_success"] += 1
        message = str(result.get("message", "")).lower().rstrip('.')
        known = {"optimization terminated successfully": "success",
                 "iteration limit reached": "iteration_limit",
                 "positive directional derivative for linesearch": "line_search_failure"}
        status = result.get("status")
        reason = known.get(message, "solver_status_" + str(int(status)) if status is not None else "not_reported")
        counts["stop_reasons"][reason] = counts["stop_reasons"].get(reason, 0) + 1

    def model_loss(self, prediction, target, model, coefficients):
        import torch

        self.calls += 1
        zero = coefficients.sum() * 0.0
        if not torch.isfinite(prediction).all():
            return zero + float("inf")
        data_loss = torch.mean(((prediction - target) / self.compiled["data_scale"]) ** 2)
        return data_loss

    def constrained_fit(self, evaluator, n_params, iterations, initial=None, hops=0):
        """Native family coefficients, solved with explicit sampled constraints."""
        import numpy as np
        import torch
        from scipy.optimize import NonlinearConstraint, basinhopping, minimize

        from ..response_search import inward_response_bounds, response_bounds, response_feasible, response_values

        def complete(active):
            active = torch.as_tensor(active, dtype=torch.float64, device=evaluator.device)
            if evaluator.mask is None:
                return active
            full = torch.zeros(evaluator.n_params, dtype=active.dtype, device=active.device)
            full[evaluator.mask] = active
            return full

        def predict(active, points):
            model = evaluator.model
            names = ("input_monomials_numerator", "input_monomials_denominator",
                     "output_monomials_dict", "output_monomials_denominator_dict")
            saved = {name: getattr(model, name) for name in names}
            try:
                model.testing_mode()
                return model.predict(complete(active), torch.as_tensor(
                    points, dtype=torch.float64, device=evaluator.device)).detach().cpu().numpy()
            finally:
                for name, value in saved.items():
                    setattr(model, name, value)

        def loss(active):
            self.calls += 1
            evaluator.evaluations += 1
            values = predict(active, evaluator.x)
            target = evaluator.y.detach().cpu().numpy()
            if not np.isfinite(values).all():
                return float("inf")
            return float(np.mean(((values - target) / self.compiled["data_scale"]) ** 2))

        def values(active):
            prediction = predict(active, self.compiled["points"])
            if prediction.shape != (len(self.compiled["points"]),):
                raise ValueError("Unexpected native response prediction shape")
            if not np.isfinite(prediction).all():
                raise _InvalidResponseTrial("Nonfinite native response prediction")
            # Check numerical overflow explicitly; malformed terms, dimensions,
            # and programming errors still propagate rather than being hidden.
            with np.errstate(over="ignore", invalid="ignore"):
                terms = np.asarray([np.dot(prediction[t["indices"]], t["coefficients"]) / t["scale"]
                                    for t in self.compiled["terms"]])
            if not np.isfinite(terms).all():
                raise _InvalidResponseTrial("Nonfinite native response combination")
            return response_values(prediction, self.compiled)

        lower, upper = response_bounds(self.compiled)
        active = np.isfinite(lower) | np.isfinite(upper)
        constraint = (NonlinearConstraint(lambda c: values(c)[active], lower[active], upper[active])
                      if active.any() else ())
        start = np.random.randn(int(n_params)) if initial is None else np.asarray(initial.detach().cpu() if isinstance(initial, torch.Tensor) else initial)
        options = {"method": "SLSQP", "constraints": constraint,
                   "options": {"maxiter": int(iterations), "ftol": 1e-12}}
        best = []

        def record(active, *_):
            if response_feasible(predict(active, self.compiled["points"]), self.compiled):
                score = loss(active)
                if np.isfinite(score):
                    best.append((score, np.array(active, copy=True)))

        steps = [0]

        def local_step(active):
            steps[0] += 1
            record(active)

        options["callback"] = local_step
        record(start)
        try:
            if int(n_params) == 0:
                self.no_free_parameter_solves += 1
            elif hops:
                result = basinhopping(loss, start, niter=int(hops), minimizer_kwargs=options, callback=record)
                self._record_solver("basinhopping_best_local", result.lowest_optimization_result)
                record(result.x)
            else:
                result = minimize(loss, start, **options)
                self._record_solver("minimize", result)
                record(result.x)
            remaining = int(iterations) * (int(hops) + 1) - steps[0]
            if int(n_params) and remaining > 0 and not response_feasible(
                    predict(result.x, self.compiled["points"]), self.compiled):
                low, high = inward_response_bounds(lower, upper)
                if not (np.array_equal(low, lower) and np.array_equal(high, upper)):
                    options["constraints"] = NonlinearConstraint(lambda c: values(c)[active], low[active], high[active])
                    options["options"]["maxiter"] = remaining
                    retry = minimize(loss, result.x, **options)
                    self._record_solver("minimize_retry", retry)
                    record(retry.x)
        except _InvalidResponseTrial:
            self.invalid_response_solves = getattr(self, "invalid_response_solves", 0) + 1
            # Keep any genuinely feasible iterate already visited. Otherwise the
            # caller records failure and moves to another family/reinitialization.
        self.constrained_solves = getattr(self, "constrained_solves", 0) + 1
        if not best:
            self.infeasible_solves = getattr(self, "infeasible_solves", 0) + 1
            return None
        return min(best, key=lambda item: item[0])[1]

    def score_loss(self, prediction, target, model, coefficients):
        """Native family/reduction selection must not reward infeasible trials."""
        if not getattr(self, "enforce", False):
            return self.model_loss(prediction, target, model, coefficients)
        import torch

        from ..response_search import response_feasible
        names = ("input_monomials_numerator", "input_monomials_denominator",
                 "output_monomials_dict", "output_monomials_denominator_dict")
        saved = {name: getattr(model, name) for name in names}
        try:
            model.testing_mode()
            values = model.predict(coefficients, torch.as_tensor(self.compiled["points"],
                dtype=coefficients.dtype, device=coefficients.device)).detach().cpu().numpy()
        finally:
            for name, value in saved.items():
                setattr(model, name, value)
        if not response_feasible(values, self.compiled):
            return coefficients.sum() * 0.0 + float("inf")
        return self.model_loss(prediction, target, model, coefficients)
