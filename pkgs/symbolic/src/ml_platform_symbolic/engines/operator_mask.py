"""Opt-in native operator-token restriction, not a syntax or semantic grammar."""


class OperatorMask:
    def __init__(self, token_ids, operator_tokens, token_operators, operators):
        allowed = set(operators)
        names = set(operator_tokens)
        self.forbidden_tokens = sorted(
            token for token in token_ids
            if token in names and token_operators.get(token) not in allowed
        )
        self.forbidden = sorted({token_ids[token] for token in self.forbidden_tokens})
        self.steps = 0

    def __call__(self, prefixes, logits, remaining=None):
        import torch

        # Only authoritative operator IDs are changed. Prefix/numeric/EOS
        # syntax and expanded portable grammar remain separate checks.
        result = logits.clone()
        result[:, self.forbidden] = -float("inf")
        if not torch.isfinite(result).any(dim=1).all():
            raise ValueError("Operator mask leaves no finite next-token logits")
        self.steps += len(result)
        return result


def metadata(mask):
    return {
        "enabled": mask is not None,
        "forbidden_native_operator_tokens": mask.forbidden_tokens if mask else [],
        "processed_rows": mask.steps if mask else 0,
        "scope": "native operator tokens only; not arity, numeric syntax, variable, EOS, node count or semantic constraints",
        "distribution": ("operator-restricted pretrained next-token distribution; weights unchanged" if mask else
                         "operator-token distribution not restricted by this mask; independent grammar, variable or size masks may still apply"),
        "portable_scope": "fixed normalization retained; full output grammar/size checked separately without exemption",
    }
