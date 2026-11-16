"""Portable semantic IDs. Native command mappings stay in their adapters.

pow, pow_int and rootn intentionally differ on negative/zero bases. Protected
native arithmetic is represented explicitly; unknown operators never fall back.
"""

import numpy as np

ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "neg": 1, "pow_int": 2, "sqrt": 1, "log": 1, "exp": 1, "abs": 1}
ARITY.update(sin=1, cos=1, tanh=1, sinh=1, cosh=1)
UNARY_FUNCTIONS = {
    "neg": np.negative,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
}
ARITY.update({name: 1 for name in UNARY_FUNCTIONS}, pow=2, rootn=2)
