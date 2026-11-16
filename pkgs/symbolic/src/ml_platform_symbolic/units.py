"""Bounded unit grammar and explicit numerical coordinate conversions.

Dimensions alone never authorize relabeling numbers. Unknown labels remain
unknown. Search priors and learned standardization do not belong here.
"""

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

ZERO = (Fraction(0),) * 4  # length, mass, time, temperature
EVIDENCE_COORDINATE_POLICY = "model_output_coordinates_and_quantity_roles/v3"


@dataclass(frozen=True)
class Unit:
    dimensions: tuple
    scale: float = 1.0
    offset: float = 0.0  # canonical = value * scale + offset
    angle: Fraction = Fraction(0)
    cycle: Fraction = Fraction(0)
    role: str | None = None
    kind_known: bool = True  # Numerical coordinates can be known without a quantity interpretation.

    def __post_init__(self):
        if len(self.dimensions) != 4 or self.role not in {None, "value", "difference"}:
            raise ValueError("Invalid unit dimensions/coordinate role")
        if (
            not isinstance(self.scale, (int, float))
            or not isinstance(self.offset, (int, float))
            or not math.isfinite(self.scale)
            or not math.isfinite(self.offset)
            or self.scale == 0
        ):
            raise ValueError("Unit coordinate scale must be finite/nonzero and offset finite")

    def power(self, exponent):
        if self.offset:
            raise ValueError("Affine units cannot occur in products or powers; convert to K first")
        return Unit(
            tuple(d * exponent for d in self.dimensions),
            self.scale ** float(exponent),
            angle=self.angle * exponent,
            cycle=self.cycle * exponent,
            role=self.role if exponent == 1 else None,
            kind_known=self.kind_known,
        )

    def product(self, other, sign=1):
        if self.offset or other.offset:
            raise ValueError("Affine units cannot occur in products; convert to K first")
        return Unit(
            tuple(a + sign * b for a, b in zip(self.dimensions, other.dimensions, strict=True)),
            self.scale * other.scale**sign,
            angle=self.angle + sign * other.angle,
            cycle=self.cycle + sign * other.cycle,
            role=self.role if other == ONE else other.role if sign == 1 and self == ONE else None,
            kind_known=self.kind_known and other.kind_known,
        )


ATOMS = {
    "m": Unit((1, 0, 0, 0)),
    "cm": Unit((1, 0, 0, 0), 0.01),
    "mm": Unit((1, 0, 0, 0), 0.001),
    "kg": Unit((0, 1, 0, 0)),
    "g": Unit((0, 1, 0, 0), 0.001),
    "mg": Unit((0, 1, 0, 0), 1e-6),
    "s": Unit((0, 0, 1, 0)),
    "ms": Unit((0, 0, 1, 0), 0.001),
    "min": Unit((0, 0, 1, 0), 60.0),
    "h": Unit((0, 0, 1, 0), 3600.0),
    "day": Unit((0, 0, 1, 0), 86400.0),
    "K": Unit((0, 0, 0, 1)),
    "degC": Unit((0, 0, 0, 1), offset=273.15),
    "delta_K": Unit((0, 0, 0, 1), role="difference"),
    "delta_degC": Unit((0, 0, 0, 1), role="difference"),
    "Pa": Unit((-1, 1, -2, 0)),
    "kPa": Unit((-1, 1, -2, 0), 1e3),
    "MPa": Unit((-1, 1, -2, 0), 1e6),
    "N": Unit((1, 1, -2, 0)),
    "kN": Unit((1, 1, -2, 0), 1e3),
    "bar": Unit((-1, 1, -2, 0), 1e5),
    "mbar": Unit((-1, 1, -2, 0), 100.0),
    "W": Unit((2, 1, -3, 0)),
    "Wh": Unit((2, 1, -2, 0), 3600.0),
    "MWh": Unit((2, 1, -2, 0), 3.6e9),
    "pct": Unit(ZERO, 0.01),
    "rad": Unit(ZERO, angle=Fraction(1)),
    "deg": Unit(ZERO, math.pi / 180, angle=Fraction(1)),
    # Cycles and angular frequency are different quantities. No silent 2*pi conversion.
    "Hz": Unit((0, 0, -1, 0), cycle=Fraction(1)),
}
ONE = Unit(ZERO)


def unit_spec(label):
    """Products/division and exact bounded rational powers; no source eval."""
    if isinstance(label, Unit):
        return label
    if not isinstance(label, str) or len(label) > 128:
        return None
    # Legacy shorthand tokens compose identically at every nesting depth.
    text = re.sub(r"\bs2\b", "(s**2)", label)
    text = re.sub(r"\bm3\b", "(m**3)", text).replace("^", "**")

    def exponent(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * exponent(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return exponent(node.left) / exponent(node.right)
        raise ValueError("Invalid unit exponent")

    def visit(node):
        if isinstance(node, ast.Name) and node.id in ATOMS:
            return ATOMS[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 1:
            return ONE
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            return visit(node.left).product(visit(node.right), 1 if isinstance(node.op, ast.Mult) else -1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            power = exponent(node.right)
            if abs(power) > 8 or power.denominator > 16:
                raise ValueError("Unit power exceeds bounded grammar")
            return visit(node.left).power(power)
        raise ValueError("Unknown unit expression")

    try:
        expression = ast.parse(text, mode="eval")
        if len(list(ast.walk(expression))) > 40:
            return None
        return visit(expression.body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


def unit_dimensions(label):
    spec = unit_spec(label)
    return None if spec is None else spec.dimensions


def same_coordinate(a, b, *, quantity=True):
    return (
        a.dimensions == b.dimensions
        and (not quantity or (a.angle == b.angle and a.cycle == b.cycle))
        and math.isclose(a.scale, b.scale, rel_tol=1e-14)
        and a.offset == b.offset
    )


def conversion(source, target, *, role="value"):
    """Return scale/offset for target-coordinate values; incompatible/unknown fail closed."""
    a, b = unit_spec(source), unit_spec(target)
    if a is None or b is None:
        if source == target:
            return 1.0, 0.0  # Explicitly identical unknown coordinates, not dimension assurance.
        raise ValueError(f"Cannot convert unknown units {source!r} -> {target!r}")
    return coordinate_conversion(a, b, role=role)


def coordinate_conversion(a, b, *, role="value"):
    if role not in {"value", "difference"}:
        raise ValueError("Coordinate role must be value or difference; unknown meanings require declaration")
    if not a.kind_known or not b.kind_known:
        raise ValueError("Numerical unit coordinates are known but physical quantity meaning is undetermined")
    if a.dimensions != b.dimensions or a.angle != b.angle or a.cycle != b.cycle:
        raise ValueError("Incompatible unit/quantity coordinates")
    return a.scale / b.scale, (a.offset - b.offset) / b.scale if role == "value" else 0.0


def quantity_role(definition):
    role = definition.get("coordinate_role")
    if role is not None and role not in {"value", "difference"}:
        raise ValueError("coordinate_role must be value/difference or unknown")
    spec = unit_spec(definition.get("unit"))
    if spec is not None and spec.role is not None:
        if role is not None and role != spec.role:
            raise ValueError("Declared coordinate role conflicts with the explicit difference-unit definition")
        return spec.role
    return role


def quantity_conversion(source, target, role):
    source_spec, target_spec = unit_spec(source), unit_spec(target)
    for spec in (source_spec, target_spec):
        if spec is not None and spec.role is not None and role != spec.role:
            raise ValueError("Quantity meaning is unknown or conflicts with the explicit difference-unit role")
    a, b = conversion(source, target)
    if role is None and b != 0:
        raise ValueError("Affine-origin conversion needs explicit coordinate_role: value or difference")
    if role not in {None, "value", "difference"}:
        raise ValueError("Unknown coordinate_role")
    return a, 0.0 if role == "difference" else b


def apply_coordinate(values, scale, offset=0.0, *, inverse=False):
    """Apply float64 coordinates, rejecting loss to zero but allowing cancellation.

    Representable subnormals and pre-existing invalid evidence retain their masks.
    Inversion subtracts the origin before division to preserve affine zeroes.
    """
    values = np.asarray(values, dtype=float)
    scale, offset = np.asarray(scale, dtype=float), np.asarray(offset, dtype=float)
    if np.any(~np.isfinite(scale) | (scale == 0)) or np.any(~np.isfinite(offset)):
        raise ValueError("Coordinate scale must be finite/nonzero and offset finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        operand = values - offset if inverse else values
        scaled = operand / scale if inverse else operand * scale
        result = scaled if inverse else scaled + offset
    valid = np.isfinite(values)
    if np.any(valid & ~np.isfinite(result)):
        raise ValueError("Coordinate conversion is nonfinite")
    if np.any(valid & (operand != 0) & (scaled == 0) & (result == 0)):
        raise ValueError("Coordinate conversion underflow: nonzero quantity became zero")
    return float(result) if result.ndim == 0 else result


def convert_values(values, source, target, *, role="value"):
    scale, offset = quantity_conversion(source, target, role)
    return apply_coordinate(values, scale, offset)


def convert_difference(values, source, target):
    """Signed increments; no absolute-coordinate offset."""
    scale, _ = conversion(source, target)
    return apply_coordinate(values, scale)


def convert_width(values, source, target):
    """Widths/sigma/error magnitudes use absolute scale; never an offset."""
    scale, _ = conversion(source, target)
    return apply_coordinate(values, abs(scale))


def convert_variance(values, source, target):
    scale, _ = conversion(source, target)
    return apply_coordinate(values, scale**2)


def convert_derivative(values, input_source, input_target, output_source, output_target, *, order=1):
    if order not in {1, 2}:
        raise ValueError("Only current first/second derivative questions are supported")
    x_scale, _ = conversion(input_source, input_target)
    y_scale, _ = conversion(output_source, output_target)
    return apply_coordinate(values, y_scale / x_scale**order)


def convert_ratio(values, numerator, denominator):
    """A stored ratio cannot be origin-independently converted across affine shifts."""
    a, b = conversion(*numerator)
    c, d = conversion(*denominator)
    if b != 0 or d != 0:
        raise ValueError("Ratio depends on coordinate origin; convert underlying values and request a new evaluation")
    return apply_coordinate(values, a / c)


def canonical_unit(label):
    spec = unit_spec(label)
    if spec is None:
        return label
    if spec.scale == 1 and spec.offset == 0:
        return label  # Preserve coherent public spellings; the parser owns aliases.
    coherent = Unit(spec.dimensions, angle=spec.angle, cycle=spec.cycle, kind_known=spec.kind_known)
    return coherent_unit_label(coherent) or label


def require_canonical(label):
    """Problem numbers/constraints use coherent coordinates; CSV source_units may differ."""
    spec = unit_spec(label)
    if spec is not None and (spec.scale != 1 or spec.offset != 0):
        raise ValueError(
            f"Problem unit {label!r} is not canonical; declare e.g. Pa/K/rad/s and put the CSV unit in data.source_units. Domains, constants and tolerances use problem coordinates."
        )


def validate_unit_mapping(mapping, columns):
    if not isinstance(mapping, dict) or set(mapping) - set(columns):
        raise ValueError("Unit keys must name declared input/target columns exactly")
    if any(not isinstance(label, (str, type(None))) or label == "" for label in mapping.values()):
        raise ValueError("Unit labels must be nonempty strings or null (unknown)")


def coherent_unit_label(spec):
    """Serialize known coherent dimensions; never drop scale or invent a quantity kind."""
    if spec is None or not spec.kind_known or spec.scale != 1 or spec.offset != 0:
        return None
    for symbol, reference in ATOMS.items():
        if (
            reference.scale == 1
            and reference.offset == 0
            and (spec.dimensions, spec.angle, spec.cycle) == (reference.dimensions, reference.angle, reference.cycle)
        ):
            return symbol
    powers = list(spec.dimensions)
    powers[2] += spec.cycle
    items = list(zip(("m", "kg", "s", "K"), powers, strict=True))
    items += [("rad", spec.angle), ("Hz", spec.cycle)]
    return "*".join(symbol if power == 1 else f"{symbol}^({power})" for symbol, power in items if power) or "1"
