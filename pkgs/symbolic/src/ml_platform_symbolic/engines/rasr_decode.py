"""RASR operator-token restrictions before sampling, not a syntax/semantic grammar."""

TOKEN_OPERATORS = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "inv": "div",
    "neg": "neg", "abs": "abs", "exp": "exp", "log": "log",
    "sin": "sin", "cos": "cos", "tanh": "tanh",
    "pow2": "pow_int", "pow3": "pow_int", "pow4": "pow_int", "pow5": "pow_int",
    "pow1_2": "sqrt", "pow1_3": "rootn", "pow1_4": "rootn", "pow1_5": "rootn",
}


def OperatorMask(fitted, operators):
    from .operator_mask import OperatorMask as CommonOperatorMask
    return CommonOperatorMask(
        {token: index for index, token in fitted.tokenizer.index_to_token.items()},
        [token for token, arity in fitted.operator_arity.items() if arity > 0],
        TOKEN_OPERATORS, operators,
    )
