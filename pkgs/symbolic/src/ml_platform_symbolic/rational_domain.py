"""Bounded exact divisor-zero checks, not IEEE or monotonicity certificates."""

from fractions import Fraction


class _Unknown(Exception):
    pass


class _Outside(Exception):
    pass


class _Unsafe(Exception):
    def __init__(self, evidence):
        self.evidence = evidence


class _Arithmetic:
    def __init__(self, degree, bits, operations):
        self.degree, self.bits, self.limit = degree, bits, operations
        self.operations = 0

    def charge(self):
        self.operations += 1
        if self.operations > self.limit:
            raise _Unknown("arithmetic_operation_budget")

    def integer_size(self, bits):
        if bits > self.bits:
            raise _Unknown("integer_bit_budget")

    def scalar(self, value):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _Unknown("unknown_or_invalid_coefficient_value")
        try:
            result = Fraction(value)
        except (ValueError, OverflowError, TypeError) as error:
            raise _Unknown("unknown_or_nonfinite_coefficient_value") from error
        self.integer_size(max(result.numerator.bit_length(), result.denominator.bit_length()))
        return result

    def add(self, a, b):
        self.charge()
        if not a:
            return b
        if not b:
            return a
        # Bound the integer intermediates BEFORE Fraction constructs/reduces them.
        self.integer_size(
            max(
                a.numerator.bit_length() + b.denominator.bit_length(),
                b.numerator.bit_length() + a.denominator.bit_length(),
            )
            + 1
        )
        self.integer_size(a.denominator.bit_length() + b.denominator.bit_length())
        return self.checked(a + b)

    def multiply(self, a, b):
        self.charge()
        if not a or not b:
            return Fraction(0)
        self.integer_size(a.numerator.bit_length() + b.numerator.bit_length())
        self.integer_size(a.denominator.bit_length() + b.denominator.bit_length())
        return self.checked(a * b)

    def divide(self, a, b):
        self.charge()
        if not b:
            raise _Unknown("internal_polynomial_division_by_zero")
        self.integer_size(a.numerator.bit_length() + b.denominator.bit_length())
        self.integer_size(a.denominator.bit_length() + b.numerator.bit_length())
        return self.checked(a / b)

    def checked(self, value):
        self.integer_size(max(value.numerator.bit_length(), value.denominator.bit_length()))
        return value

    def polynomial(self, coefficients):
        result = list(coefficients)
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        if len(result) - 1 > self.degree:
            raise _Unknown("polynomial_degree_budget")
        return tuple(result)

    def sum(self, left, right, sign=1):
        return self.polynomial(
            self.add(left[i] if i < len(left) else Fraction(0), sign * right[i] if i < len(right) else Fraction(0))
            for i in range(max(len(left), len(right)))
        )

    def product(self, left, right):
        if left == (0,) or right == (0,):
            return (Fraction(0),)
        if len(left) + len(right) - 2 > self.degree:
            raise _Unknown("polynomial_degree_budget")
        result = [Fraction(0)] * (len(left) + len(right) - 1)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                result[i + j] = self.add(result[i + j], self.multiply(a, b))
        return self.polynomial(result)

    def power(self, value, exponent):
        result = (Fraction(1),)
        for _ in range(exponent):
            result = self.product(result, value)
        return result

    def value(self, polynomial, point):
        result = Fraction(0)
        for coefficient in reversed(polynomial):
            result = self.add(self.multiply(result, point), coefficient)
        return result

    def remainder(self, numerator, denominator):
        result = list(numerator)
        while len(result) >= len(denominator) and any(result):
            factor = self.divide(result[-1], denominator[-1])
            offset = len(result) - len(denominator)
            for i, value in enumerate(denominator):
                result[offset + i] = self.add(result[offset + i], -self.multiply(factor, value))
            result = list(self.polynomial(result))
        return self.polynomial(result)

    def roots(self, polynomial, low, high):
        """Existence on CLOSED [low, high]; count distinct interior roots only."""
        if polynomial == (0,):
            return {"kind": "identically_zero", "roots": "all_points"}
        if self.value(polynomial, low) == 0:
            return {"kind": "endpoint", "endpoint": "lower", "roots": "at_least_one"}
        if self.value(polynomial, high) == 0:
            return {"kind": "endpoint", "endpoint": "upper", "roots": "at_least_one"}
        if low == high or len(polynomial) == 1:
            return None
        # Exact Horner interval exclusion avoids costly Sturm sequences when a
        # conservative rational enclosure already proves the polynomial nonzero.
        lower = upper = Fraction(0)
        for coefficient in reversed(polynomial):
            products = [self.multiply(a, b) for a in (lower, upper) for b in (low, high)]
            lower = upper = products[0]
            for product in products[1:]:
                if self.add(product, -lower) < 0:
                    lower = product
                if self.add(product, -upper) > 0:
                    upper = product
            lower, upper = self.add(lower, coefficient), self.add(upper, coefficient)
        if lower > 0 or upper < 0:
            return None
        derivative = tuple(self.multiply(Fraction(i), polynomial[i]) for i in range(1, len(polynomial)))
        sequence = [polynomial, self.polynomial(derivative)]
        while True:
            remainder = self.remainder(sequence[-2], sequence[-1])
            if remainder == (0,):
                break
            sequence.append(tuple(-x for x in remainder))

        def variations(point):
            values = [self.value(p, point) for p in sequence]
            signs = [1 if x > 0 else -1 for x in values if x]
            return sum(a != b for a, b in zip(signs, signs[1:], strict=False))

        count = variations(low) - variations(high)
        if count < 0:
            raise _Unknown("internal_sturm_inconsistency")
        return {"kind": "interior", "roots": count} if count else None


def check(
    equation, variables, *, region=None, max_nodes=256, max_degree=32, max_integer_bits=4096, max_operations=20000
):
    """Check every rational AST divisor on its declared 1D closed box.

    The mathematical numbers are exact stored integers and the exact binary
    values of stored floats, NOT rounded display strings. Trainable coefficients are
    treated as fixed at their present candidate values. No fitting occurs.
    'safe' proves only that all intermediate rational divisions/negative powers
    are defined there. It does not certify IEEE finite arithmetic, conditioning,
    positivity, monotonicity, units, physical truth or coefficient uncertainty.
    Unknown/unsupported results must never be interpreted as safe.
    """
    from .domains import check_region_definition, domain_variables, validate_bounds
    from .expression_graph import expanded_equation
    from .expressions import inspect_equation

    budgets = {
        "max_nodes": max_nodes,
        "max_degree": max_degree,
        "max_integer_bits": max_integer_bits,
        "max_operations": max_operations,
    }
    if any(type(v) is not int or v < 1 for v in budgets.values()):
        raise ValueError("Rational-domain budgets must be positive integers")
    arithmetic = _Arithmetic(max_degree, max_integer_bits, max_operations)
    checked = []
    result = {
        "scope": "all intermediate rational divisors on one declared closed interval",
        "method": "exact_Fraction_interval_exclusion_and_Sturm",
        "budgets": budgets,
        "coefficient_semantics": "exact stored integers and exact binary values of stored floats; fixed candidate",
        "not_certified": ["IEEE_finiteness", "conditioning", "monotonicity", "physical_validity"],
        "checked_divisors": checked,
    }
    try:
        if not isinstance(equation, dict) or not isinstance(variables, list):
            raise _Unknown("invalid_equation_or_variable_schema")
        if region is not None and not isinstance(region, dict):
            raise _Unknown("invalid_region_schema")
        if not isinstance(equation.get("parameters", {}), dict):
            raise _Unknown("invalid_parameter_schema")
        if len(variables) != 1:
            raise _Outside("requires_exactly_one_input_variable")
        if region is not None and (
            region.get("kind") != "box"
            or region.get("columns") != [variables[0]["column"]]
            or region.get("equalities")
            or region.get("inequalities")
        ):
            raise _Outside("requires_declared_1D_box_not_axis_bound_hints")
        if region is not None:
            check_region_definition(region, [variables[0]["column"]])
        variable = domain_variables(variables, region)[0]
        bounds = variable.get("domain")
        if bounds is None or any(x is None for x in bounds):
            raise _Outside("requires_finite_closed_interval")
        validate_bounds(bounds)
        low, high = map(arithmetic.scalar, bounds)
        result["interval"] = list(bounds)
        for parameter in equation.get("parameters", {}).values():
            if not isinstance(parameter, dict):
                raise _Unknown("invalid_parameter_schema")
            arithmetic.scalar(parameter.get("value"))
        stats = inspect_equation(equation, variables)
        if stats["n_nodes"] > max_nodes:
            raise _Unknown("expanded_AST_node_budget")
        tree = expanded_equation(equation)["ast"]
        one = (Fraction(1),)

        def divisor(polynomial, path, operation):
            proof = arithmetic.roots(polynomial, low, high)
            entry = {
                "path": path,
                "operation": operation,
                "degree": len(polynomial) - 1,
                "zero_set": proof if proof else "empty_on_closed_interval",
            }
            checked.append(entry)
            if proof:
                raise _Unsafe(entry)

        def visit(node, path):
            if "const" in node:
                return (arithmetic.scalar(node["const"]),), one
            if "param" in node:
                return (arithmetic.scalar(equation["parameters"][node["param"]]["value"]),), one
            if "var" in node:
                return (Fraction(0), Fraction(1)), one
            # Visit ALL child expressions before simplification, including x**0.
            children = [visit(child, path + [i]) for i, child in enumerate(node["args"])]
            op = node["op"]
            p, q = children[0]
            if op == "neg":
                return tuple(-x for x in p), q
            if op not in {"add", "sub", "mul", "div", "pow_int"}:
                raise _Outside("non_rational_operator:" + op)
            r, s = children[1]
            if op in {"add", "sub"}:
                return arithmetic.sum(
                    arithmetic.product(p, s), arithmetic.product(r, q), 1 if op == "add" else -1
                ), arithmetic.product(q, s)
            if op == "mul":
                return arithmetic.product(p, r), arithmetic.product(q, s)
            if op == "div":
                divisor(r, path + [1], op)
                return arithmetic.product(p, s), arithmetic.product(q, r)
            exponent = int(node["args"][1]["const"])  # inspect_equation validates fixed literal -8..8.
            if exponent < 0:
                divisor(p, path + [0], "negative_integer_power")
                p, q = q, p
            return arithmetic.power(p, abs(exponent)), arithmetic.power(q, abs(exponent))

        visit(tree, [])
        result.update(status="safe", reason="every_intermediate_divisor_proved_nonzero")
    except _Unsafe as error:
        result.update(status="unsafe", reason="proved_intermediate_divisor_zero", witness=error.evidence)
    except _Outside as error:
        result.update(status="out_of_scope", reason=str(error))
    except _Unknown as error:
        result.update(status="unknown", reason=str(error))
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as error:
        result.update(status="unknown", reason="invalid_or_unresolved_common_input:" + str(error))
    result["arithmetic_operations"] = arithmetic.operations
    return result
