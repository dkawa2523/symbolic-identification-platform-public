"""Pinned RASR optional refinement and pre-sampling operator-mask hooks."""
import argparse
import hashlib
from pathlib import Path

HASHES = {"rasr.py": "9681e3f50e971bbe22ce1dc1456fb88e02b5dfc90f0674539be0969c55ec0354",
          "../model/model.py": "1899edd7c2896976bf6fe107ebe6c3e3f7092d465aa53549eead2f6b77a4c1e2",
          "refiner.py": "fe6b454e04b61fdaffbb7a09959dbbee8f9b35b7a044eddfd80d739f3eef4afb"}
MARKER = "symbolic_response_refinement_version = 1"


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Pinned RASR refinement anchor mismatch: " + before[:70])
    return text.replace(before, after, 1)



def progress_hooks(text):
    """Reuse the exact native completed-candidate checks, without rerunning optimizers."""
    first = text.index("        results: list[CandidateResult] = []\n", text.index("    def fit("))
    last = text.index("        if not results:\n", first)
    block = text[first:last]
    returns = "results, n_invalid_fit, n_nonfinite_pred, n_nonfinite_score, n_lambda_failed"
    helper = ("    def _completed_candidates(self, refined_payloads, X_standardized, y_standardized,\n"
              "                              X_val_standardized=None, y_val_standardized=None):\n"
              + block + "        return " + returns + "\n\n")
    text = text[:first] + "        " + returns + " = self._completed_candidates(\n            refined_payloads, X_standardized, y_standardized, X_val_standardized, y_val_standardized)\n\n" + text[last:]
    text = replace(text, "    def fit(\n", helper + "    def fit(\n")
    text = replace(text, '        if (X_val is None) != (y_val is None):',
                   '        if getattr(self, "symbolic_progress", None) is not None and (X_val is not None or y_val is not None):\n'
                   '            raise ValueError("Completed-refinement checkpoints currently require Train-only native ranking")\n'
                   '        if (X_val is None) != (y_val is None):')
    text = replace(text, "        sampled_ids, sampled_log_probs = self.model.sample(",
                   '        progress = getattr(self, "symbolic_progress", None)\n'
                   '        if progress is not None:\n            progress("decode_started")\n'
                   "        sampled_ids, sampled_log_probs = self.model.sample(")
    text = replace(text, "        token_id_candidates = sampled_ids[0]",
                   '        if progress is not None:\n            progress("decode_completed", generated=len(sampled_ids[0]))\n'
                   "        token_id_candidates = sampled_ids[0]")
    text = replace(text, "        refined_payloads = self._run_refinement(",
                   '        if progress is not None:\n            progress("refine_started", parsed=len(candidate_payloads), parse_failed=n_parse_failed, outside_variables=n_too_many_variables)\n'
                   "        refined_payloads = self._run_refinement(")
    text = replace(text, '                result = _refine_candidate_worker(payload)\n                if result is not None:',
                   '                progress = getattr(self, "symbolic_progress", None)\n'
                   '                if progress is not None:\n                    progress("candidate_started", candidate_id=payload["cand_id"])\n'
                   '                result = _refine_candidate_worker(payload)\n'
                   '                if progress is not None:\n'
                   '                    completed = self._completed_candidates([result], X, y)[0] if result is not None else []\n'
                   '                    progress("candidate_completed", candidate_id=payload["cand_id"], candidate=completed[0] if completed else None)\n'
                   '                if result is not None:')
    return text + "\nsymbolic_progress_version = 1\n"


def patch(directory):
    root = Path(directory)
    texts = {name: (root/name).read_text() for name in HASHES}
    if all(MARKER in text for text in texts.values()):
        return
    for name, digest in HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Unexpected pinned RASR source: " + name)
    text = texts["rasr.py"]
    text = replace(text, '            converge_error="ignore",\n        )',
                   '            converge_error="ignore",\n            coefficient_solver=payload.get("coefficient_solver"),\n        )')
    text = replace(text, '        if not candidate_payloads:\n            return []', '''        if not candidate_payloads:
            return []
        solver = getattr(self, "coefficient_solver", None)
        if solver is not None:
            if self.refiner_workers != 0 or not perform_refinement:
                raise ValueError("Response refinement requires serial native coefficient fitting")
            candidate_payloads = [{**payload, "coefficient_solver": solver} for payload in candidate_payloads]''')
    text = replace(text, '            unique=True,\n        )',
                   '            unique=True,\n            token_mask=getattr(self, "token_mask", None),\n        )')
    texts['rasr.py'] = text
    text = texts['refiner.py']
    text = replace(text, '        converge_error: Literal["raise", "ignore"] = "ignore",\n    ) -> "Refiner":',
                   '        converge_error: Literal["raise", "ignore"] = "ignore",\n        coefficient_solver=None,\n    ) -> "Refiner":')
    text = replace(text, '            diff = pred_function(X) - y[:, 0]', '''            if coefficient_solver is not None:
                coefficient_solver(expression, X, y[:, 0], np.array([]))
            diff = pred_function(X) - y[:, 0]''')
    text = replace(text, '''                    popt, pcov = curve_fit(
                        pred_function, X_valid, y_valid, p0=p0_run, method=curve_fit_method, **optimizer_kwargs
                    )''', '''                    if coefficient_solver is None:
                        popt, pcov = curve_fit(
                            pred_function, X_valid, y_valid, p0=p0_run, method=curve_fit_method, **optimizer_kwargs
                        )
                    else:
                        popt, pcov = coefficient_solver(expression, X_valid, y_valid, p0_run)''')
    texts['refiner.py'] = text
    text = texts['../model/model.py']
    text = replace(text, '        extract_expressions: bool = True,\n        unique: bool = False,\n    )',
                   '        extract_expressions: bool = True,\n        unique: bool = False,\n        token_mask=None,\n    )')
    text = replace(text, '                    filtered_logits = next_token_logits.clone()',
                   '                    filtered_logits = next_token_logits.clone()\n'
                   '                    if token_mask is not None:\n'
                   '                        filtered_logits = token_mask(decoder_input_ids, filtered_logits, max_len - current_len + 1)')
    texts['../model/model.py'] = text + '\nsymbolic_operator_mask_version = 1\n'
    texts['rasr.py'] += '\nsymbolic_operator_mask_version = 1\n'
    texts["rasr.py"] = progress_hooks(texts["rasr.py"])
    for name,text in texts.items():
        texts[name] = text + '\n' + MARKER + '\n'
        compile(texts[name],name,'exec')
    for name,text in texts.items():
        (root/name).write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    patch(parser.parse_args().directory)
