"""Exact coefficient dimensions for this adapter's bounded scalar grammar.

This is dimensional bookkeeping, not physical parameter identification. Radian
quantity meaning is not inferred; the common unit checker treats radians as
canonical dimensionless numerical coordinates. Cycle counts remain distinct.
"""

from fractions import Fraction

from ..constraints import dimensions
from ..units import Unit, coherent_unit_label, unit_spec


def assign_coefficient_units(equation, slot):
    """Reject inconsistent or nonunique dimensional systems instead of guessing.

    For an explicitly all-unit1 slot, preserve the historical prior that every
    coefficient is dimensionless, including otherwise underdetermined products.
    In dimensional slots no free dimensional gauge is silently fixed.
    """
    formal = [{"id": a["name"], "unit": a["unit"]} for a in slot["arguments"]]
    all_unit1 = slot["unit"] == "1" and all(a["unit"] == "1" for a in slot["arguments"])
    names = list(equation["parameters"])
    if not all_unit1:
        n = len(names)
        rows = []

        def constant(label):
            spec = unit_spec(label)
            return [Fraction(0)] * n, list(map(Fraction, (*spec.dimensions, spec.cycle)))

        def combine(left, right, factor=1):
            return tuple([a + factor * b for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True))

        def constrain(left, right):
            weights, known = combine(left, right, -1)
            rows.append([*weights, *[-v for v in known]])

        variables = {a["name"]: constant(a["unit"]) for a in slot["arguments"]}
        one = constant("1")

        def visit(node):
            if "var" in node:
                return variables[node["var"]]
            if "const" in node:
                return one
            if "param" in node:
                weights = [Fraction(int(name == node["param"])) for name in names]
                return weights, [Fraction(0)] * 5
            args = [visit(a) for a in node["args"]]
            op = node["op"]
            if op in {"add", "sub"}:
                constrain(args[0], args[1])
                return args[0]
            if op in {"mul", "div"}:
                return combine(args[0], args[1], 1 if op == "mul" else -1)
            if op in {"neg", "abs"}:
                return args[0]
            if op in {"sqrt", "pow_int"}:
                power = Fraction(1, 2) if op == "sqrt" else Fraction(node["args"][1]["const"])
                return tuple([v * power for v in part] for part in args[0])
            constrain(args[0], one)
            return one

        constrain(visit(equation["ast"]), constant(slot["unit"]))
        pivot_row = 0
        pivots = []
        for column in range(n):
            pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][column]), None)
            if pivot is None:
                continue
            rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
            factor = rows[pivot_row][column]
            rows[pivot_row] = [v / factor for v in rows[pivot_row]]
            for i, row in enumerate(rows):
                if i != pivot_row:
                    factor = row[column]
                    rows[i] = [a - factor * b for a, b in zip(row, rows[pivot_row], strict=True)]
            pivots.append(column)
            pivot_row += 1
        if any(not any(row[:n]) and any(row[n:]) for row in rows):
            raise ValueError("Inconsistent slot dimensions; no coefficient-unit assignment")
        if len(pivots) != n:
            raise ValueError("Underdetermined coefficient dimensions; no arbitrary unit assignment")
        for row, column in zip(rows, pivots, strict=False):
            values = row[n:]
            label = coherent_unit_label(Unit(tuple(values[:4]), cycle=values[4]))
            if unit_spec(label) is None:
                raise ValueError("Inferred coefficient unit exceeds the portable unit grammar")
            equation["parameters"][names[column]].update(
                unit=label,
                unit_origin="unique_exact_numeric_dimension_solution; physical quantity kind not inferred",
            )
    status, reasons = dimensions(equation, formal, {"unit": slot["unit"]})
    if status != "pass":
        raise ValueError("Slot unit assignment failed common check: " + "; ".join(reasons))
