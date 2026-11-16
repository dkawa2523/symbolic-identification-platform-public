"""Real rEGGression frozen graph indexing; known-broken native patterns are unavailable."""

import os
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ...data import write_json
from ...expression_graph import expanded_equation
from ...expressions import equation_text
from ..numerical import prerequisite_failure


def frozen_tir(equation, variables):
    """Native TIR accepts signed numeric atoms, but not common neg(...) syntax."""

    equation = expanded_equation(equation)

    def freeze(node):
        if "param" in node:
            return {"const": equation["parameters"][node["param"]]["value"]}
        if "op" not in node:
            return dict(node)
        args = [freeze(a) for a in node["args"]]
        if node["op"] == "neg":
            if "const" not in args[0]:
                raise ValueError("TIR unary negation of a nonconstant expression is not faithfully supported")
            return {"const": -args[0]["const"]}
        return {"op": node["op"], "args": args}

    frozen = {"ast": freeze(equation["ast"]), "parameters": {}}
    return equation_text(frozen, [{**v, "column": v["id"]} for v in variables]).replace("**", "^")


def run(context, records, rows, arrays, directory, options):
    from reggression import Reggression

    if version("reggression") != "2.4.0":
        raise ValueError("This verified bridge requires reggression2.4.0")
    directory = Path(directory)
    os.chdir(directory)  # Native temporary files remain task-owned.
    variables = context["variables"]
    frame = pd.DataFrame(arrays["train_X"], columns=[v["id"] for v in variables])
    frame["target"] = arrays["train_y"]
    frame.to_csv(directory / "native_data.csv", index=False)
    engine = Reggression(
        dataset=str(directory / "native_data.csv"), parseParams=False, refit=False, db=str(directory / "structure.db")
    )
    results = []
    for record, row in zip(records, rows, strict=True):
        if options.get("ids") and record["id"] not in options["ids"]:
            continue
        if record["kind"] != "explicit":
            results.append(
                {
                    "id": record["id"],
                    "outcome": "unavailable",
                    "reason": "This evaluator requires an explicit parameterized equation; native output remains available to prediction/behavior comparison",
                }
            )
            continue
        failure = prerequisite_failure(record, row)
        if failure:
            results.append({"id": record["id"], "outcome": "unavailable", "reason": failure})
            continue
        if row.get("metrics", {}).get("train", {}).get("rmse") is None:
            results.append(
                {"id": record["id"], "outcome": "unavailable", "reason": "Graph registration requires finite common Train RMSE"}
            )
            continue
        try:
            text = frozen_tir(record["equation"], variables)
            eid = int(engine.dbInsert(expr=text, alg="TIR"))
            # Registration uses already computed common Train error, never native refitting.
            engine.dbSetFit(eid=eid, fitness=-row["metrics"]["train"]["rmse"])
            expressions = engine.getNExpressions(eid, 3).to_dict(orient="records")
            results.append(
                {
                    "id": record["id"],
                    "outcome": "ok",
                    "native_eclass": eid,
                    "frozen_input": text,
                    "backend_representations": expressions,
                    "meaning": "backend graph index only; not domain-certified equivalence or replacement",
                }
            )
        except (ValueError, RuntimeError) as exc:
            results.append({"id": record["id"], "outcome": "unavailable", "reason": str(exc)})
    tokens = (
        engine.distributionOfTokens().to_dict(orient="records") if any(r["outcome"] == "ok" for r in results) else []
    )
    write_json(directory / "native_tokens.json", tokens)
    requested_pattern = bool(options.get("patterns"))
    return {
        "outcome": "unavailable" if requested_pattern else "ok" if tokens else "unavailable",
        "reason": "Native2.4.0 pattern matcher does not distinguish operators/repeated wildcards"
        if requested_pattern
        else None,
        "result_kind": "structure.pool/v1",
        "version": version("reggression"),
        "license": "GPL-3.0",
        "results": results,
        "native_token_distribution": tokens,
        "native_pattern_queries": "unsupported: actual sin/exp and add/mul counterexample verified",
        "parameter_policy": "frozen instantiated literals and constant negation via dbInsert; authoritative AST unchanged; no CSV parameter extraction, eqsat or optimize",
        "scope": "graph/subexpression sharing on this pool; authoritative parameter identity remains external",
    }
