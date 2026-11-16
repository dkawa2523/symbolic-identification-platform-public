"""Pinned native flag plumbing only; invoked during isolated Docker image setup."""

import argparse
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "612c57fae9a1a83cfb4cd42b47b723e77c24904e582a52fad409f41b00f4a14a"
MARKER = "symbolic_handoff_division_version = 1"


def _patch_division(package_dir):
    path = Path(package_dir) / "train.py"
    source = path.read_bytes()
    if MARKER.encode() in source:
        return hashlib.sha256(source).hexdigest()
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Unexpected pinned ParFam train.py; division plumbing was not applied")
    text = source.decode("utf-8")
    old = "enforce_function=model_parameters['enforce_function'])"
    text = text.replace(
        old,
        "enforce_function=model_parameters['enforce_function'],\n                            stabilize_denominator=model_parameters.get('stabilize_denominator', True))",
        1,
    )
    anchor = "        model_parameter_list = [model_parameters_perfect] * training_parameters['repetitions']\n    t_0 = time()"
    if text.count(anchor) != 1:
        raise ValueError("Pinned ParFam candidate-list anchor mismatch")
    text = text.replace(
        anchor,
        "        model_parameter_list = [model_parameters_perfect] * training_parameters['repetitions']\n    for parameters in model_parameter_list:\n        parameters['stabilize_denominator'] = training_parameters.get('stabilize_denominator', True)\n    t_0 = time()",
    )
    text += "\n" + MARKER + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


OBJECTIVE_MARKER = "symbolic_handoff_objective_version = 3"
PATCHED_TRAIN_SHA256 = "9a9c3eddfc2b4e0841b9535785ed48bc2b3dbfd1d37a10689f27fe981d78283d"
TORCH_SHA256 = "265a222a9b5302273b7a691bfc8658f37235b5a4de92185899fd45536c8b4c0f"


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Pinned ParFam objective anchor mismatch: " + old[:80])
    return text.replace(old, new, 1)


def patch(package_dir):
    """Pass full models/coefficients to opt-in objectives in every fitting stage."""
    _patch_division(package_dir)
    train_path = Path(package_dir) / "train.py"
    torch_path = Path(package_dir) / "parfam_torch.py"
    train, tensor = train_path.read_text(), torch_path.read_text()
    if OBJECTIVE_MARKER in train and OBJECTIVE_MARKER in tensor:
        return _patch_fit_failure(package_dir)
    if (hashlib.sha256(train_path.read_bytes()).hexdigest() != PATCHED_TRAIN_SHA256
            or hashlib.sha256(torch_path.read_bytes()).hexdigest() != TORCH_SHA256):
        raise ValueError("Unexpected pinned ParFam objective source; no changes made")
    # Native validation used theta=None. Preserve old callbacks but supply the
    # actual model and complete coefficients to our explicit model_loss protocol.
    import ast
    tree = ast.parse(train)
    lines = train.splitlines(keepends=True)
    replacements = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "custom_loss"):
                continue
            if len(node.args) != 3 or not isinstance(node.args[2], ast.Constant) or node.args[2].value is not None:
                raise ValueError("Unexpected native validation custom_loss signature")
            coeff = "coefficients" if function.name == "evaluate" else "best_coefficients"
            if function.name == "evaluate_test_iterative_finetuning":
                # The inner reduction trial must score its trial coefficients.
                prior = "".join(lines[max(function.lineno - 1, node.lineno - 3):node.lineno - 1])
                if "model.predict(coefficients, x_val)" in prior:
                    coeff = "coefficients"
            if function.name not in {"evaluate", "evaluate_test", "evaluate_test_iterative_finetuning"}:
                raise ValueError("Unexpected native objective caller " + function.name)
            replacements.append((node.lineno - 1, node.col_offset, node.end_col_offset,
                "_model_loss(custom_loss, " + ast.unparse(node.args[0]) + ", "
                + ast.unparse(node.args[1]) + ", model, " + coeff + ")"))
    if len(replacements) != 7:
        raise ValueError("Unexpected pinned native validation call count")
    for row, start, end, value in sorted(replacements, reverse=True):
        lines[row] = lines[row][:start] + value + lines[row][end:]
    train = "".join(lines)
    train = _replace_once(train,
        "n_params=n_params, lambda_1_cut=lambda_1_cut, lambda_1_piecewise=lambda_1_piecewise)",
        "n_params=n_params, lambda_1_cut=lambda_1_cut, lambda_1_piecewise=lambda_1_piecewise, custom_loss=custom_loss)")
    train = _replace_once(train, "                              n_params=n_params)\n",
        "                              n_params=n_params, custom_loss=custom_loss)\n")
    train = _replace_once(train,
        "    lw = [-10] * n_params_active",
        """    if getattr(evaluator.custom_loss, 'enforce', False):
        if optimizer != 'basinhopping':
            raise ValueError('Constrained ParFam requires native basinhopping')
        initial = np.random.randn(n_params_active) if x0 is None else x0
        result = evaluator.custom_loss.constrained_fit(evaluator, n_params_active,
            maxiter_per_dim_local_minimizer * x.shape[1], initial, hops=int(maxiter / 10))
        model.testing_mode()
        return initial if result is None else result
    lw = [-10] * n_params_active""")
    train = _replace_once(train,
        "def train_lbfgs(evaluator, n_params_active, maxiter, coefficients=None):",
        """def train_lbfgs(evaluator, n_params_active, maxiter, coefficients=None):
    if getattr(evaluator.custom_loss, 'enforce', False):
        best, best_loss = None, float('inf')
        for _ in range(maxiter):
            result = evaluator.custom_loss.constrained_fit(evaluator, n_params_active,
                200 * max(1, int(n_params_active)), coefficients)
            if result is not None:
                loss = evaluator.loss_func(result)
                if loss < best_loss:
                    best, best_loss = result, loss
                    if loss < 1e-5:
                        break
            coefficients = None
        return None if best is None else torch.tensor(best, dtype=torch.double, device=evaluator.device)""")
    anchor = "    return best_relative_l2_distance_train, best_relative_l2_distance_val, best_r_squared_val, best_training_time, "
    if train.count(anchor) != 1:
        raise ValueError("Unexpected native family selection return")
    train = train.replace(anchor,
        "    if best_model_parameters is None and getattr(custom_loss, 'enforce', False):\n"
        "        raise ValueError('ParFam found no feasible family under declared response constraints')\n" + anchor)
    train += "\n" + OBJECTIVE_MARKER + "\n" + '''
def _model_loss(callback, prediction, target, model, coefficients):
    if hasattr(callback, "score_loss"):
        return callback.score_loss(prediction, target, model, coefficients)
    if hasattr(callback, "model_loss"):
        return callback.model_loss(prediction, target, model, coefficients)
    return callback(prediction, target, None)
'''
    anchor = "loss = self.custom_loss(y_pred, self.y, self.coefficients_current)"
    if tensor.count(anchor) != 2:
        raise ValueError("Unexpected pinned native optimizer call count")
    tensor = tensor.replace(anchor, "loss = self._custom_model_loss(y_pred)")
    method = '''    def _custom_model_loss(self, prediction):
        self.evaluations += 1
        if hasattr(self.custom_loss, "model_loss"):
            coefficients = self.coefficients_current
            if self.mask is not None:
                complete = torch.zeros(self.n_params, device=self.device, dtype=torch.double)
                complete[self.mask] = coefficients
                coefficients = complete
            return self.custom_loss.model_loss(prediction, self.y, self.model, coefficients)
        return self.custom_loss(prediction, self.y, self.coefficients_current)

'''
    tensor = _replace_once(tensor, "    def gradient(self, _):", method + "    def gradient(self, _):")
    tensor += "\n" + OBJECTIVE_MARKER + "\n"
    # Validate both edits before touching either source file.
    compile(train, str(train_path), "exec")
    compile(tensor, str(torch_path), "exec")
    train_path.write_text(train, encoding="utf-8")
    torch_path.write_text(tensor, encoding="utf-8")
    return _patch_fit_failure(package_dir)



FIT_FAILURE_MARKER = "symbolic_handoff_fit_failure_version = 1"
OBJECTIVE_TRAIN_SHA256 = "300acae924415c3cb3e37406162ecb462bdc9c2aba7d550ac77c1895b22f3c2c"
OBJECTIVE_TORCH_SHA256 = "1fba9c0d7f8e32260316e72d02830875abb07d0c6c702d323701fc16729ecf61"
FIT_TRAIN_SHA256 = "791ca284393fccdd0776239914661b810ffa91772820a64a57f9d26007306449"
FIT_TORCH_SHA256 = "b0b5ae9c164c97fb589f3f5b2c50a355a82c6fc5c2fc5198de1772594b4c1efa"


def _patch_fit_failure(package_dir):
    """Opt-in local-fit failure, with no altered objective or gradient fallback."""
    train_path, tensor_path = (Path(package_dir) / n for n in ('train.py', 'parfam_torch.py'))
    train, tensor = train_path.read_text(), tensor_path.read_text()
    if FIT_FAILURE_MARKER in train and FIT_FAILURE_MARKER in tensor:
        if (hashlib.sha256(train_path.read_bytes()).hexdigest() != FIT_TRAIN_SHA256
                or hashlib.sha256(tensor_path.read_bytes()).hexdigest() != FIT_TORCH_SHA256):
            raise ValueError("Unexpected pinned ParFam marked fit-failure source")
        return hashlib.sha256(train_path.read_bytes()).hexdigest()
    if (hashlib.sha256(train_path.read_bytes()).hexdigest() != OBJECTIVE_TRAIN_SHA256
            or hashlib.sha256(tensor_path.read_bytes()).hexdigest() != OBJECTIVE_TORCH_SHA256):
        raise ValueError('Unexpected pinned ParFam fit-failure source; no changes made')
    tensor = _replace_once(tensor, 'class Evaluator:',
        'class NonfiniteGradientError(FloatingPointError):\n'
        '    """The current local fit has an undefined autodiff gradient."""\n\n\nclass Evaluator:')
    tensor = _replace_once(tensor, '    def gradient(self, _):', '''
    def _check_gradient(self):
        if (getattr(self.custom_loss, 'reject_nonfinite_gradients', False)
                and (self.coefficients_current.grad is None
                     or not torch.isfinite(self.coefficients_current.grad).all())):
            raise NonfiniteGradientError('Nonfinite ParFam local-fit autograd gradient')

    def gradient(self, _):''')
    tensor = _replace_once(tensor, '        self.loss_current.backward()\n        return convert_to_module',
        '        self.loss_current.backward()\n        self._check_gradient()\n        return convert_to_module')
    tensor = _replace_once(tensor, '                    loss.backward()\n                return loss',
        '                    loss.backward()\n                self._check_gradient()\n                return loss')
    tensor = _replace_once(tensor, '                optimizer.step(closure)\n            except Exception as e:',
        '                optimizer.step(closure)\n            except NonfiniteGradientError:\n                raise\n            except Exception as e:')
    train = _replace_once(train, 'from .parfam_torch import ParFamTorch, Evaluator',
        'from .parfam_torch import ParFamTorch, Evaluator, NonfiniteGradientError')
    train = _replace_once(train, '    lw = [-10] * n_params_active',
        '    guarded = getattr(evaluator.custom_loss, "reject_nonfinite_gradients", False)\n'
        '    if guarded:\n        local_minimizer = _finite_minimizer(evaluator, local_minimizer)\n'
        '    lw = [-10] * n_params_active')
    # Both stochastic SciPy paths use the callable local method; other branches keep their own fit loop.
    train = _replace_once(train, '    model.testing_mode()\n    return coefficients\n',
        '    if guarded and optimizer in {"basinhopping", "dual_annealing"}:\n'
        '        if not np.isfinite(ret.fun) or not np.isfinite(ret.x).all():\n'
        '            best = local_minimizer.best_result\n'
        '            coefficients = best.x if best is not None else None\n'
        '    model.testing_mode()\n    return coefficients\n')
    train = _replace_once(train,
        "        ret = minimize(evaluator.loss_func, x0=coefficients, method='BFGS', jac=evaluator.gradient,\n                       options={'maxiter': 200 * n_params_active})",
        "        method = (_finite_minimizer(evaluator, 'BFGS') if getattr(evaluator.custom_loss,\n"
        "                  'reject_nonfinite_gradients', False) else 'BFGS')\n"
        "        ret = minimize(evaluator.loss_func, x0=coefficients, method=method, jac=evaluator.gradient,\n"
        "                       options={'maxiter': 200 * n_params_active})\n"
        "        if getattr(evaluator.custom_loss, 'reject_nonfinite_gradients', False) and not np.isfinite(ret.fun):\n"
        "            coefficients = None\n            continue")
    left, right = train.split('def train_lbfgs(', 1)
    left = _replace_once(left, "    if best_coefficients is None:\n        logging.warning(f'best_coefficients is None, something that should not happen. \\n'",
        "    if best_coefficients is None and getattr(evaluator.custom_loss, 'reject_nonfinite_gradients', False):\n"
        "        return None\n    if best_coefficients is None:\n        logging.warning(f'best_coefficients is None, something that should not happen. \\n'")
    train = left + 'def train_lbfgs(' + right
    train = _replace_once(train, '        coefficients, relative_l2_distance = evaluator.fit_lbfgs(coefficients, verbose=False)',
        '        try:\n            coefficients, relative_l2_distance = evaluator.fit_lbfgs(coefficients, verbose=False)\n'
        '        except NonfiniteGradientError:\n            _record_gradient_failure(evaluator)\n'
        '            coefficients = None\n            continue\n'
        '        if getattr(evaluator.custom_loss, "reject_nonfinite_gradients", False):\n'
        '            if not torch.isfinite(coefficients).all() or not torch.isfinite(relative_l2_distance):\n'
        '                _record_nonfinite_result(evaluator)\n'
        '                coefficients = None\n                continue')
    # Second all-starts-failed fallback belongs to LBFGS, whose pruning caller accepts None.
    anchor="    if best_coefficients is None:\n        logging.warning(f'best_coefficients is None, something that should not happen. \\n'"
    # First occurrence was preserved in the earlier edit; select the LBFGS function only.
    left, right = train.split('def train_lbfgs(', 1)
    right = _replace_once(right, anchor,
        "    if best_coefficients is None and getattr(evaluator.custom_loss, 'reject_nonfinite_gradients', False):\n"
        "        return None\n" + anchor)
    train = left + 'def train_lbfgs(' + right
    train = _replace_once(train, "    logging.info(f'Number of evaluations: {evaluator.evaluations}')",
        "    if coefficients_ is None and getattr(custom_loss, 'reject_nonfinite_gradients', False):\n"
        "        failed = torch.full((n_params,), float('nan'), device=training_parameters['device'], dtype=torch.double)\n"
        "        return np.inf, np.inf, np.inf, time() - t_0, failed, evaluator.evaluations\n"
        "    logging.info(f'Number of evaluations: {evaluator.evaluations}')")
    train = _replace_once(train, "        coefficients = torch.zeros(n_params, device=training_parameters['device'], dtype=torch.double)\n        coefficients[mask] = torch.tensor(coefficients_, device=training_parameters['device'], dtype=torch.double)",
        "        if coefficients_ is None and getattr(custom_loss, 'reject_nonfinite_gradients', False):\n            break  # Keep the last completed coefficient fit.\n"
        "        coefficients = torch.zeros(n_params, device=training_parameters['device'], dtype=torch.double)\n        coefficients[mask] = torch.tensor(coefficients_, device=training_parameters['device'], dtype=torch.double)")
    train = _replace_once(train, "    if best_model_parameters is None and getattr(custom_loss, 'enforce', False):",
        "    if best_model_parameters is None and getattr(custom_loss, 'reject_nonfinite_gradients', False):\n"
        "        raise NoFiniteFamilyError('ParFam found no finite completed family fit')\n"
        "    if best_model_parameters is None and getattr(custom_loss, 'enforce', False):")
    train += '''

class NoFiniteFamilyError(ValueError):
    """Every attempted family lacked a finite completed fit."""


def _record_gradient_failure(evaluator):
    objective = evaluator.custom_loss
    if objective is not None:
        objective.nonfinite_gradient_fits = getattr(objective, 'nonfinite_gradient_fits', 0) + 1


def _record_nonfinite_result(evaluator):
    objective = evaluator.custom_loss
    if objective is not None:
        objective.nonfinite_result_fits = getattr(objective, 'nonfinite_result_fits', 0) + 1


def _finite_minimizer(evaluator, method):
    """Keep other basin hops/restarts; an undefined local fit is not a candidate."""
    from scipy.optimize import OptimizeResult

    def local(fun, x0, args=(), jac=None, hess=None, hessp=None, bounds=None,
              constraints=(), callback=None, **options):
        point = np.asarray(x0, dtype=float).copy()
        if not np.isfinite(point).all():
            raise ValueError('ParFam local-fit start must be finite')
        calls = {'fun': 0, 'jac': 0}
        def counted_fun(x, *values):
            calls['fun'] += 1
            return fun(x, *values)
        def counted_jac(x, *values):
            calls['jac'] += 1
            return jac(x, *values)
        try:
            result = minimize(counted_fun, point, args=args, method=method,
                              jac=counted_jac if callable(jac) else jac, hess=hess,
                              hessp=hessp, bounds=bounds, constraints=constraints,
                              callback=callback, options=options)
        except NonfiniteGradientError:
            _record_gradient_failure(evaluator)
            return OptimizeResult(x=point, fun=np.inf, success=False,
                                  message='nonfinite gradient', nfev=calls['fun'], njev=calls['jac'])
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            _record_nonfinite_result(evaluator)
            return OptimizeResult(x=point, fun=np.inf, success=False,
                                  message='nonfinite local-fit result', nfev=calls['fun'],
                                  njev=calls['jac'])
        # Lack of convergence alone does not discard a finite returned candidate.
        if local.best_result is None or result.fun < local.best_result.fun:
            local.best_result = result
        return result
    local.best_result = None
    return local
'''
    train += '\n' + FIT_FAILURE_MARKER + '\n'
    tensor += '\n' + FIT_FAILURE_MARKER + '\n'
    compile(train, str(train_path), 'exec')
    compile(tensor, str(tensor_path), 'exec')
    train_path.write_text(train, encoding='utf-8')
    tensor_path.write_text(tensor, encoding='utf-8')
    return hashlib.sha256(train_path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    print(patch(parser.parse_args().package_dir))
