"""Finite-point constrained coefficients before native PhySO reward/RNN update.

Scoped factory replacement is restricted to the independent discovery worker.
Native generation and RNN training remain native; coefficient fitting is an extension.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np
from ..expressions import evaluate
from ..fitting import fit_coefficients
from ..response_search import response_bounds, response_values, response_feasible


def prepare(request):
    from .physo import capabilities
    from ..response_search import response_enforcement, requested_response_constraints, compile_response_constraints
    from ..domains import validate_input_domain, region_mask
    response_enforcement(request.limits, capabilities())
    if request.limits.get('response_mode', 'search') != 'search' or not requested_response_constraints(request.limits):
        return None
    if request.domain is not None and request.domain['columns'] != [v['column'] for v in request.variables]:
        raise ValueError('PhySO response domain column order differs')
    validate_input_domain(request.X, request.variables)
    if not region_mask(request.X, request.domain).all():
        raise ValueError('PhySO Train outside declared joint domain')
    compiled = compile_response_constraints(request)
    if not compiled['terms']:
        raise ValueError('PhySO response search has no bounded terms')
    if not region_mask(np.asarray(compiled['points']), request.domain).all():
        raise ValueError('PhySO response stencil outside declared joint domain')
    lower, upper = response_bounds(compiled)
    if not np.any(np.isfinite(lower) | np.isfinite(upper)):
        raise NotImplementedError('PhySO constrained coefficient fitting needs finite bounds')
    return compiled


class ResponseReward:
    def __init__(self, request, compiled, export):
        self.request, self.compiled, self.export = request, compiled, export
        from .base import native_input_indices
        self.feature_indices = native_input_indices(request)
        self.counts = Counter()
        self.examples = []
        self.batches = []
        self.factory_calls = 0

    def factory(self, reward_function, zero_out_unphysical=False, zero_out_duplicates=False,
                keep_lowest_complexity_duplicate=False, parallel_mode=False, n_cpus=None):
        if parallel_mode:
            raise ValueError('Constrained PhySO reward requires independent sequential worker')
        self.factory_calls += 1
        self.native_duplicate_requested = bool(zero_out_duplicates)
        self.reward_function = reward_function
        self.zero_out_unphysical = zero_out_unphysical
        return self.compute

    def compute(self, programs, X, y_target, y_weights, n_samples_per_dataset, free_const_opti_args):
        import torch
        if len(n_samples_per_dataset) != 1 or int(n_samples_per_dataset[0]) != len(self.request.X):
            raise ValueError('PhySO response fit supports the full single Train realization only')
        if not np.array_equal(X.detach().cpu().numpy().T, self.request.X[:, self.feature_indices]):
            raise ValueError('PhySO reward Train coordinates changed')
        if not np.array_equal(y_target.detach().cpu().numpy(), self.request.y):
            raise ValueError('PhySO reward Train target changed')
        if not bool(torch.all(torch.as_tensor(y_weights) == 1)):
            raise NotImplementedError('PhySO response fit does not support weighted Train objectives')
        rewards = np.zeros(programs.batch_size)
        points = np.asarray(self.compiled['points'], float)
        lower, upper = response_bounds(self.compiled)
        batch_counts = Counter()
        for i in range(programs.batch_size):
            row = {'index': i, 'solver': {}}
            if not programs.is_complete[i] or (self.zero_out_unphysical and not programs.is_physical[i]):
                batch_counts['excluded_incomplete_or_unphysical'] += 1
                continue
            program = programs.get_prog(i)
            try:
                equation = self.export(program)
                fitted = fit_coefficients(equation, self.request.X, self.request.y, self.request.variables,
                    constraints=(lambda eq: response_values(evaluate(eq, points, self.request.variables), self.compiled), lower, upper),
                    data_scale=self.compiled['data_scale'], diagnostics=row['solver'],
                    max_nfev=self.request.options.get('response_max_nfev', 200))
                # Only native class constants used by this program are changed. Fixed tokens stay immutable.
                with torch.no_grad():
                    for token in program.tokens:
                        if token.var_type == 20:
                            matches = [record for name, record in fitted['parameters'].items()
                                       if record.get('physo_native_name', name) == token.name and record.get('trainable')]
                            if len(matches) != 1:
                                raise ValueError('PhySO template free coefficient identity changed')
                            program.free_consts.class_values[0, token.var_id] = matches[0]['value']
                pred = program(X).detach().cpu().numpy()
                probe = program(torch.as_tensor(points[:, self.feature_indices].T, dtype=X.dtype, device=X.device)).detach().cpu().numpy()
                portable = evaluate(fitted, self.request.X, self.request.variables)
                portable_probe = evaluate(fitted, points, self.request.variables)
                if not (np.allclose(pred, portable, rtol=1e-8, atol=1e-10) and np.allclose(probe, portable_probe, rtol=1e-8, atol=1e-10)):
                    raise ValueError('PhySO constrained coefficient native/portable prediction mismatch')
                if not np.isfinite(pred).all() or not response_feasible(probe, self.compiled):
                    raise ValueError('PhySO constrained native prediction failed finite-point recheck')
                reward = float(self.reward_function(y_target=y_target, y_pred=torch.as_tensor(pred, dtype=y_target.dtype, device=y_target.device), y_weights=y_weights))
                if not np.isfinite(reward) or reward <= 0:
                    raise ValueError('PhySO feasible expression has no finite positive native reward')
                rewards[i] = reward
                batch_counts['feasible'] += 1
                batch_counts['solver_converged' if row['solver'].get('success') else 'feasible_without_convergence_claim'] += 1
                row.update(status='feasible', reward=reward, equation=fitted, response_values=response_values(probe, self.compiled).tolist())
            except (ValueError, FloatingPointError, OverflowError) as exc:
                batch_counts['no_accepted_fit'] += 1
                row.update(status='no_accepted_fit', reason=str(exc))
            # Bounded examples, not a new all-candidate ledger.
            if sum(r['status'] == row['status'] for r in self.examples) < 3:
                self.examples.append(row)
        self.counts.update(batch_counts)
        self.batches.append({'batch': len(self.batches), **dict(batch_counts), 'positive_rewards': int(np.count_nonzero(rewards > 0))})
        return rewards

    def summary(self):
        return {'stage': 'coefficient_fit_before_native_reward_and_RNN_update', 'enforcement': 'hard',
                'scope': 'declared finite points only; generated structures and physical-unit prior unchanged',
                'solver': 'common SLSQP coefficient extension, not native LBFGS',
                'coordinates': 'prepared working coordinates; no repeated source conversion',
                'native_reward_equal_duplicate_elimination': False,
                'native_duplicate_setting_overridden': getattr(self, 'native_duplicate_requested', None),
                'factory_calls': self.factory_calls, 'counts': dict(self.counts),
                'count_scope': 'program visits across reward batches; repeated structures count again',
                'batches': self.batches, 'examples': self.examples,
                'max_nfev_per_program': self.request.options.get('response_max_nfev', 200)}


@contextmanager
def reward_scope(controller):
    if controller is None:
        yield
        return
    import physo
    from physo.physym import reward
    root = Path(physo.__file__).parent
    expected = {'physym/reward.py': '02d2357953ea0bd77988ddc1af43b4b1a93d7aab23599207dad5eabf85fba6b7',
                'task/args_handler.py': '44256664f6442e5f064ab66ebf7b7df9890de98eff648b4ace6779962b0165be'}
    for path, digest in expected.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Unreviewed PhySO response reward integration source: ' + path)
    original = reward.make_RewardsComputer
    reward.make_RewardsComputer = controller.factory
    try:
        yield
        if not controller.factory_calls or not controller.batches:
            raise RuntimeError('PhySO did not use the required constrained reward callback')
    finally:
        reward.make_RewardsComputer = original
