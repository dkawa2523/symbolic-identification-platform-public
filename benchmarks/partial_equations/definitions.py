"""Shared fixed partial-equation definitions; no campaign execution or scoring."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "partial_equations"
OPERATORS = ["add", "sub", "mul", "log"]

def op(name, *args):
    return {"op": name, "args": list(args)}

def var(index):
    return {"var": f"x{index}"}

def slot(name, *args):
    return {"slot": name, "args": list(args)}

def template(case, unrestricted=False):
    x, z = var(0), var(1)
    definitions = {}
    parameters = {}
    arities = {"h": 1}
    if case in ("additive", "wrong_prior", "confounded"):
        definitions = {
            "base": {"ast": op("mul", {"param": "fixed_slope"}, x), "description": "Declared fixed linear term"}
        }
        parameters = {
            "fixed_slope": {
                "value": 2.0,
                "trainable": False,
                "unit": "1",
                "bounds": [2.0, 2.0],
                "origin": "user_declared",
            }
        }
        ast = op("add", {"ref": "base"}, slot("h", z))
    elif case in ("multiplicative", "zero_factor"):
        ast = op("mul", x, slot("h", z))
    elif case == "shared_component":
        ast = op("add", slot("h", x), slot("h", z))
    elif case == "gauge_nonidentifiable":
        ast = op("add", slot("h", x), slot("g", x))
        arities["g"] = 1
    elif case == "positive_domain":
        ast = op("add", op("log", slot("h", z)), x)
    elif case == "abalone":
        volume = op("mul", op("mul", var(0), var(1)), var(2))
        arguments = [var(i) for i in range(5)] if unrestricted else [var(3), var(4)]
        ast = op("mul", volume, slot("h", *arguments))
        arities = {"h": len(arguments)}
    else:
        raise ValueError(case)
    return {
        "coordinate_space": "working",
        "ast": ast,
        "components": definitions,
        "parameters": parameters,
        "slots": {
            name: {
                "arguments": [{"name": f"u{i}", "unit": "1"} for i in range(arity)],
                "unit": "1",
                "allowed_operators": OPERATORS,
                "max_nodes": 18,
                "description": "Unknown function; arguments enforce dependency scope",
            }
            for name, arity in arities.items()
        },
        "knowledge": {
            "kind": "hypothesis",
            "statement": f"Frozen {case} structural hypothesis; predictive performance does not prove mechanistic truth",
        },
    }
