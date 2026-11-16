"""Presentation of saved equations. No evaluation, simplification or refitting.

The AST and parameter table remain authoritative. Public-coordinate expressions
substitute only frozen affine maps; coefficients keep their working meanings.
"""

from copy import deepcopy

from .data import input_variables, public_unit
from .expression_graph import expanded_equation
from .expressions import inspect_equation
from .parameters import parameter_table


def latex_name(value):
    """Treat arbitrary labels as text, never as executable TeX commands."""
    escapes = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_",
        "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
        "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    }
    return r"\text{" + "".join(escapes.get(char, char) for char in str(value)) + "}"


def _number(value, format):
    text = repr(value)
    if format == "latex" and "e" in text.lower():
        mantissa, exponent = text.lower().split("e")
        return mantissa + r"\times 10^{" + str(int(exponent)) + "}"
    return text


def _text_name(value):
    return value if value.isidentifier() else "input[" + repr(value) + "]"


def format_equation(equation, variables, *, format="text", parameter_names=True, variable_expressions=None):
    """Render a validated AST without algebraic changes; do not parse display text."""
    if format not in {"text", "latex", "display"}:
        raise ValueError("Equation format must be text, latex or display")
    inspect_equation(equation, variables)
    names = {v["id"]: _text_name(v["column"]) if format != "latex" else latex_name(v["column"]) for v in variables}
    names.update(variable_expressions or {})
    tree = expanded_equation(equation)

    def display(node):
        # Minimal parentheses preserve the stored operation tree, including
        # right-nested operations. No cancellation or coefficient rounding.
        if "var" in node:
            return names[node["var"]], 100
        if "const" in node:
            return repr(node["const"]), 100 if node["const"] >= 0 else 25
        if "param" in node:
            value = equation["parameters"][node["param"]]["value"]
            return (("\u03b8[" + node["param"] + "]", 100) if parameter_names else
                    (repr(value), 100 if value >= 0 else 25))
        op = node["op"]
        args = [display(child) for child in node["args"]]
        signs = {"add": ("+", 10), "sub": ("-", 10), "mul": ("*", 20), "div": ("/", 20),
                 "pow_int": ("^", 30)}
        if op in signs:
            sign, precedence = signs[op]
            left, right = args
            a = "(" + left[0] + ")" if left[1] < precedence or (op == "pow_int" and left[1] == precedence) else left[0]
            b = "(" + right[0] + ")" if right[1] <= precedence else right[0]
            return f"{a} {sign} {b}", precedence
        if op == "neg":
            value, precedence = args[0]
            return "-" + ("(" + value + ")" if precedence <= 25 else value), 25
        return f"{op}({', '.join(arg[0] for arg in args)})", 100

    if format == "display":
        return display(tree["ast"])[0]

    def visit(node):
        if "var" in node:
            return names[node["var"]]
        if "const" in node:
            return _number(node["const"], format)
        if "param" in node:
            if not parameter_names:
                return _number(equation["parameters"][node["param"]]["value"], format)
            return ("coef[" + repr(node["param"]) + "]" if format == "text" else
                    r"\theta_{" + latex_name(node["param"]) + "}")
        args = [visit(child) for child in node["args"]]
        op = node["op"]
        if format == "text":
            if op == "pow_int":
                return f"(({args[0]}) ** ({args[1]}))"
            signs = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            return (f"({args[0]} {signs[op]} {args[1]})" if op in signs else
                    f"{op}({', '.join(args)})")
        if op in {"add", "sub", "mul"}:
            sign = {"add": "+", "sub": "-", "mul": r"\cdot"}[op]
            return r"\left(" + args[0] + " " + sign + " " + args[1] + r"\right)"
        if op == "div":
            return r"\frac{" + args[0] + "}{" + args[1] + "}"
        if op == "pow_int":
            return r"\left(" + args[0] + r"\right)^{" + args[1] + "}"
        if op == "neg":
            return r"-\left(" + args[0] + r"\right)"
        if op == "abs":
            return r"\left|" + args[0] + r"\right|"
        if op == "sqrt":
            return r"\sqrt{" + args[0] + "}"
        # Keep pow/rootn explicit: a mathematical fractional-power glyph could
        # suggest a signed rational root which numpy.power does not implement.
        return r"\operatorname{" + op + r"}\left(" + ", ".join(args) + r"\right)"

    return visit(tree["ast"])


def original_equation(equation, context, *, residual=False, format="text", parameter_names=True):
    """Render the same frozen-coordinate AST used for numerical node accounting."""
    from .coordinates import source_equation

    if equation is None:
        return None
    mapping = (context.get("input_geometry") or {}).get("map")
    if mapping and mapping.get("kind") != "affine_chart":
        return None
    public, variables = source_equation(equation, context, residual=residual)
    expression = format_equation(public, variables, format=format, parameter_names=parameter_names)
    if residual:
        return expression + " = 0"
    target = context["target"]
    unit = public_unit(context, target) or "unknown"
    label = (f"{target['column']} [{unit}]" if format != "latex" else
             latex_name(target["column"]) + r"\;[" + latex_name(unit) + "]")
    return label + " = " + expression


def equation_outputs(equation, context, *, residual=False):
    """Data-only renderings plus the original AST and coefficient identities."""
    from .coordinates import source_complexity
    if equation is None:
        return None
    variables = [{**v, "column": v["column"] + "_work"} for v in context["variables"]]
    if residual:
        variables.append({**context["target"], "id": "target", "column": context["target"]["column"] + "_work"})
    stats = inspect_equation(equation, variables)
    suffix = " = 0" if residual else ""
    notes = [
        "Display only; no simplification, fitting, coefficient conversion or model selection.",
        "Common real-domain validity requires every intermediate and result to be finite; domain holes remain.",
        "Coefficient values/units refer to the stored working equation; unknown units are not dimensionless.",
        "coef['name'] / theta_name denote coefficients, distinct from input and target column names.",
    ]
    if "rootn" in stats["operators"]:
        notes.append("rootn(x,n): sign(x)*abs(x)**(1/n) for odd integer n; x**(1/n) for even n. n is fixed and nonzero.")
    if stats["operators"] & {"pow", "pow_int"}:
        notes.append("pow uses real numpy.power; pow_int retains its fixed integer exponent. No cancellation of domain holes.")
    return dict(
        equation=deepcopy(equation),
        n_nodes=source_complexity(equation, context, residual=residual),
        working_n_nodes=stats["n_nodes"],
        complexity_scope="expanded source-coordinate AST",
        text=format_equation(equation, variables) + suffix,
        latex=format_equation(equation, variables, format="latex") + suffix,
        public_text=original_equation(equation, context, residual=residual),
        public_latex=original_equation(equation, context, residual=residual, format="latex"),
        display_text=format_equation(equation, variables, format="display") + suffix,
        public_display_text=original_equation(equation, context, residual=residual, format="display"),
        parameters=parameter_table(equation),
        arithmetic=stats["arithmetic"],
        operators=sorted(stats["operators"]),
        semantic_notes=notes,
    )


def equation_rows(view):
    """All stored equation components, including zero-mass and residual views.

    Historical readouts keep their original text. They are never parsed back
    into an AST or given guessed LaTeX/operator/parameter semantics.
    """
    rows = []

    def add(card, component, item, *, member=None):
        output = item.get("equation_output")
        row = dict(candidate_id=card["id"], method=card.get("method"), kind=card["kind"],
                   component=component, member_index=member, member_id=item.get("id") if member is not None else None,
                   weight=item.get("weight") if member is not None else None,
                   eligible=card.get("eligible"), reason=card.get("reason"), variant=card.get("variant"),
                   subject=deepcopy(card.get("subject")), execution_view=deepcopy(card.get("execution_view")),
                   uncertainty_scope=deepcopy(card.get("uncertainty_scope")),
                   branch=deepcopy((card.get("residual") or {}).get("branch")),
                   prediction_unavailable_reason=(card.get("residual") or {}).get("prediction_unavailable_reason"),
                   native_result=deepcopy(card.get("native_result")))
        if output is not None:
            row.update(deepcopy(output), display_status="stored AST")
        else:
            row.update(text=item.get("formula"), public_text=item.get("public_formula"), latex=None,
                       public_latex=None, parameters=deepcopy(item.get("parameters", [])), equation=None,
                       arithmetic=None, operators=None,
                       display_status="AST not retained; original text only" if item.get("formula") else
                       "portable equation unavailable",
                       semantic_notes=["Missing equation/LaTeX is not reconstructed from text or model execution."])
        if card["kind"] == "ensemble":
            row["semantic_notes"].append(
                "Native ensemble members/weights retain the saved execution_view and uncertainty scope; "
                "weights are not inferred probabilities of physical mechanisms. No single formula is invented."
            )
        rows.append(row)

    for card in view["candidates"]:
        if card["kind"] == "ensemble":
            for index, member in enumerate(card.get("members", [])):
                add(card, "ensemble_member", member, member=member.get("index", index))
            if not card.get("members"):
                add(card, "ensemble", card)
        elif card["kind"] == "implicit":
            add(card, "implicit_residual", card.get("residual") or {})
            # Residual parameters do not describe the predictor's coefficients.
            add(card, "implicit_predictor", {**card, "parameters": []})
        else:
            add(card, "prediction" if card["kind"] == "explicit" else "native_result", card)
    return rows



def template_rows(template):
    """Named saved structure only; never infer physical terms or change coordinates."""
    if not template:
        return []
    common = {"coordinates": template.get("coordinates"), "status": template.get("status")}
    if template.get("status") != "computed":
        return [{**common, "section": "status", "reason": template.get("reason") or "not recorded"}]
    rows = [{**common, "section": "declared_outer", "name": "relation",
             "formula": template.get("declared_formula"), "knowledge": template.get("knowledge"),
             "description": template.get("interpretation_limit")}]
    for section, key in (("declared_component", "components"), ("fitted_slot", "slots"),
                         ("dependent_parameter", "dependent_parameters")):
        rows.extend({**common, **deepcopy(item), "section": section} for item in template.get(key, []))
    if template.get("sharing"):
        rows.append({**common, "section": "sharing", "name": "slot_calls", "description": template["sharing"]})
    return rows


def template_lines(template, *, concise=False):
    """Readable rendering shared by text exports and separate equation sheets."""
    rows = template_rows(template)
    if not rows:
        return []
    if template.get("status") != "computed":
        return ["Template readout: " + str(template.get("status") or "not recorded"),
                "Reason: " + str(template.get("reason") or "not recorded")]
    lines = ["Declared structure and fitted slots", "Coordinates: " + str(template.get("coordinates") or "not recorded")]
    for row in rows:
        if concise and row["section"] == "sharing":
            lines.append("Shared slot name = shared function and coefficients")
            continue
        label = row["section"].replace("_", " ") + ": " + str(row.get("name", ""))
        if row.get("arguments") is not None:
            label += "(" + ", ".join(f"{a['name']} [{a.get('unit') or 'unit unknown'}]" for a in row["arguments"]) + ")"
        lines.append(label + (" = " + str(row["formula"]) if row.get("formula") is not None else ""))
        if row.get("declared_formula") and (not concise or row["declared_formula"] != row.get("formula")):
            lines.append("  Declared: " + row["declared_formula"])
        if row.get("unit") is not None:
            lines.append("  Unit: " + str(row["unit"]))
        if "value" in row:
            lines.append("  Saved value: " + repr(row["value"]))
        for parameter in row.get("parameters", []):
            if concise:
                value = parameter.get("value")
                shown = f"{value:.8g}" if isinstance(value, (int, float)) else repr(value)
                tied = f"; tied to {parameter['tied_to']}" if parameter.get("tied_to") else ""
                lines.append(f"  {parameter['name']} = {shown}; {parameter.get('role', 'role unknown')}; [{parameter.get('unit') or 'unit unknown'}]{tied}")
                continue
            lines.append(f"  {parameter['name']} = {parameter.get('value')!r}; role={parameter.get('role', 'not recorded')}; "
                         f"unit={parameter.get('unit') or 'unknown'}; tied_to={parameter.get('tied_to')}")
        if row.get("knowledge") is not None and not concise:
            lines.append("  Declared knowledge: " + str(row["knowledge"]))
        if row.get("description") and not concise:
            lines.append("  " + str(row["description"]))
    return lines
