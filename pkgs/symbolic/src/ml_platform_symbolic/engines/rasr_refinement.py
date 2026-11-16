"""Common constrained coefficient fitting inside pinned RASR native refinement."""
from types import SimpleNamespace

import numpy as np

from ..expressions import evaluate
from ..fitting import fit_coefficients
from ..response_search import compile_response_constraints, response_bounds, response_feasible, response_values


class ResponseRefinement:
    def __init__(self, request, fitted, record_file=None):
        self.request, self.fitted = request, fitted
        self.compiled = compile_response_constraints(request)
        self.records = []
        self.record_file = record_file

    def save(self):
        if self.record_file is not None:
            from ..data import write_json
            write_json(self.record_file, {"stage": "native candidate coefficient refinement before native ranking", "attempts": self.records})

    def __call__(self, tokens, native_X, native_y, initial):
        from .rasr import export_candidate
        r = self.request
        record = {"tokens": list(tokens), "parameters": len(initial), "status": "started"}
        self.records.append(record)
        self.save()
        try:
            if len(native_X) != len(r.X) or len(native_y) != len(r.y):
                raise ValueError("Native refinement dropped Train rows")
            equation = export_candidate(SimpleNamespace(tokens=tokens, constants=list(initial)),
                                        self.fitted, r.variables, r.target)
            points = np.asarray(self.compiled['points'], dtype=float)
            def values(equation):
                return response_values(evaluate(equation, points, r.variables), self.compiled)
            lower, upper = response_bounds(self.compiled)
            diagnostics = {}
            result = fit_coefficients(equation, r.X, r.y, r.variables,
                data_scale=self.compiled['data_scale'], constraints=(values, lower, upper),
                max_nfev=1000, diagnostics=diagnostics)
            if not np.isfinite(evaluate(result, r.X, r.variables)).all():
                raise ValueError("Nonfinite common Train prediction")
            if not response_feasible(evaluate(result, points, r.variables), self.compiled):
                raise ValueError("No strictly feasible native coefficient refinement")
            constants = np.asarray([result['parameters'][f'c{i}']['value'] for i in range(len(initial))])
            record.update(status="feasible", diagnostics=diagnostics)
            self.save()
            return constants, np.array([])
        except ValueError as error:
            record.update(status="rejected", reason=str(error))
            self.save()
            raise
