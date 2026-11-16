"""Finite-point coefficient search over every supplied E2E structure, not decoder conditioning."""
from copy import deepcopy
import time
import numpy as np
from ..coordinates import source_complexity
from ..domains import region_mask, validate_input_domain
from ..expressions import evaluate
from ..fitting import fit_coefficients
from ..parameters import free_parameter_names
from ..response_search import (compile_response_constraints, requested_response_constraints,
                               response_bounds, response_values, response_feasible)
from .base import checked_candidate


def prepare_response_fit(request):
    from .e2e import capabilities
    from ..response_search import response_enforcement
    response_enforcement(request.limits, capabilities())
    if request.limits.get('response_mode', 'search') != 'search' or not requested_response_constraints(request.limits):
        return None
    if request.domain is not None and request.domain['columns'] != [v['column'] for v in request.variables]:
        raise ValueError('E2E response domain column order differs')
    validate_input_domain(request.X, request.variables)
    if not region_mask(request.X, request.domain).all():
        raise ValueError('E2E Train outside declared joint domain')
    compiled = compile_response_constraints(request)
    if not compiled['terms']:
        raise ValueError('E2E response search has no bounded terms')
    points = np.asarray(compiled['points'], dtype=float)
    if not region_mask(points, request.domain).all():
        raise ValueError('E2E response stencil outside declared joint domain')
    lower, upper = response_bounds(compiled)  # Contradictions fail before loading native weights.
    if not np.any(np.isfinite(lower) | np.isfinite(upper)):
        raise NotImplementedError("E2E response coefficient search requires a finite bound/check; unbounded validity probes alone are not constrained optimization")
    return compiled


def fit_responses(candidates, request, context, compiled):
    """Native candidates remain immutable; each accepted variant has separate lineage."""
    points = np.asarray(compiled['points'], dtype=float)
    lower, upper = response_bounds(compiled)
    budget = request.options.get('response_max_nfev', 1000)
    coordinate_context = {**context.provenance, 'variables': request.variables, 'target': request.target}
    from .e2e_domain import prepare_admission
    admission = prepare_admission(request, context, compiled)
    derived, attempts = [], []
    start = time.monotonic()
    for parent in candidates:
        row = {'parent_id': parent.candidate_id,
               'parent_refinement_type': parent.native_check.get('refinement_type'),
               'status': 'no_accepted_fit', 'solver': {}}
        attempts.append(row)
        if admission is not None:
            admission.begin()
        if context.deadline is not None and time.monotonic() >= context.deadline:
            row.update(status='not_attempted_deadline', reason='Outer search deadline reached; no candidate ranking used')
            continue
        if admission is not None and admission.unavailable_reason() is not None:
            row.update(status='not_attempted_gate_budget', reason=admission.unavailable_reason())
            continue
        try:
            equation = deepcopy(parent.equation)
            for name in free_parameter_names(equation, require_origin=True):
                if equation['parameters'][name]['origin'] != 'native_decimal_coefficient':
                    raise ValueError('Unverified free E2E coefficient origin')
            equation['arithmetic'] = 'strict_real'
            fitted = fit_coefficients(
                equation, request.X, request.y, request.variables,
                constraints=(lambda eq: response_values(evaluate(eq, points, request.variables), compiled), lower, upper),
                data_scale=compiled['data_scale'], diagnostics=row['solver'], max_nfev=budget,
                iterate_acceptance=admission,
            )
            prediction = evaluate(fitted, request.X, request.variables)
            if not np.isfinite(prediction).all() or not response_feasible(evaluate(fitted, points, request.variables), compiled):
                raise ValueError('Returned fit failed full-Train/finite-point feasibility recheck')
            nodes = source_complexity(fitted, coordinate_context)
            max_nodes = request.limits.get('max_nodes')
            if max_nodes is not None and nodes > max_nodes:
                raise RuntimeError('E2E response fit changed source complexity beyond search budget')
            candidate = checked_candidate('e2e', 'response-fit', fitted, request,
                lambda X: evaluate(fitted, X, request.variables), native_ieee=False)
            candidate.candidate_id = parent.candidate_id + '/response-fit'
            candidate.native_check.update(
                kind='portable_constrained_coefficient_fit; not native predictor parity',
                parent_id=parent.candidate_id, parent_refinement_type=row['parent_refinement_type'],
                refinement_type='response-fit', finite_point_feasible=True,
                solver_converged=row['solver'].get('success'), source_n_nodes=nodes,
            )
            if admission is not None:
                candidate.native_check['rational_domain_admission'] = admission.certificate()
            derived.append(candidate)
            row.update(status='accepted', candidate_id=candidate.candidate_id, finite_point_feasible=True,
                       solver_converged=row['solver'].get('success'), train_mse=float(np.mean((prediction-request.y)**2)))
        except (ValueError, FloatingPointError, OverflowError) as error:
            row['reason'] = str(error)
        finally:
            if admission is not None:
                row['rational_domain_admission'] = admission.last_result()
    return derived, {
        'stage': 'coefficient_optimization', 'solver': 'SLSQP', 'constraint_kind': 'explicit finite-point bounds, not penalty',
        'scope': 'declared finite points only; no global physical or derivative guarantee',
        'coordinates': 'prepared working coordinates before E2E internal normalization; frozen source mapping retained',
        'arithmetic': 'strict_real derived variants; native numpy_ieee equations unchanged',
        'decoder_conditioned_on_response': False, 'prefit_ranking': False, 'candidate_cap': None,
        'supplied': len(candidates), 'attempted': sum(not r['status'].startswith('not_attempted_') for r in attempts),
        'accepted': len(derived), 'complete': all(not r['status'].startswith('not_attempted_') for r in attempts) and (admission is None or admission.budget_denials == 0),
        'train_rows': len(request.X), 'constraint_points': len(points), 'max_nfev_per_candidate': budget,
        'elapsed_seconds': time.monotonic()-start, 'attempts': attempts,
        **({'rational_domain_admission': admission.summary()} if admission is not None else {}),
    }
