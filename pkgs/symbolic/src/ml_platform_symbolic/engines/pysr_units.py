"""Recover uniquely implied coefficient dimensions from ordinary PySR exports.

This records structural dimensional consequences, not native-exported units or
physical identification. No fit, numerical value, expression or candidate changes.
Unsupported semantics and free dimensional gauges remain unknown.
"""

import sympy as sp

from ..units import Unit, coherent_unit_label, unit_spec


def infer_coefficient_units(equation, variables, target):
    """Return exact metadata only when every unknown coefficient has a unique unit.

    Inputs must use coherent basic units. Angle/cycle, affine coordinates, explicit
    quantity roles and general/learned powers are intentionally outside this scope.
    Existing known parameter units are equations, never overwritten assumptions.
    """
    names = [name for name, spec in equation['parameters'].items() if spec.get('unit') is None]
    n = len(names)
    rows = []

    def known(label):
        unit = unit_spec(label)
        if unit is None or unit.scale != 1 or unit.offset or unit.angle or unit.cycle or unit.role:
            raise ValueError('Unknown, noncoherent or physical-kind unit semantics')
        return sp.zeros(1, n), sp.Matrix(1, 4, list(unit.dimensions))

    def combine(left, right, sign=1):
        return left[0] + sign * right[0], left[1] + sign * right[1]

    def equal(left, right):
        weights, constant = combine(left, right, -1)
        rows.append((weights, -constant))

    try:
        if any(v.get('quantity_role') or v.get('coordinate_role') for v in [*variables, target]):
            raise ValueError('Explicit quantity roles are not inferred')
        one = known('1')
        inputs = {v['id']: known(v.get('unit')) for v in variables}

        def visit(node):
            if 'var' in node:
                return inputs[node['var']]
            if 'const' in node:
                return one
            if 'param' in node:
                key = node['param']
                if key not in names:
                    return known(equation['parameters'][key]['unit'])
                return sp.Matrix(1, n, [int(k == key) for k in names]), sp.zeros(1, 4)
            op = node['op']
            args = [visit(a) for a in node['args']]
            if op in {'add', 'sub'}:
                equal(*args)
                return args[0]
            if op in {'mul', 'div'}:
                return combine(*args, 1 if op == 'mul' else -1)
            if op in {'neg', 'abs'}:
                return args[0]
            if op == 'sqrt':
                return tuple(value / 2 for value in args[0])
            if op == 'pow_int':
                exponent = node['args'][1].get('const')
                if type(exponent) is not int:
                    raise ValueError('Not a structural integer exponent')
                equal(args[1], one)
                return tuple(exponent * value for value in args[0])
            if op in {'log', 'exp', 'sin', 'cos', 'tanh', 'sinh', 'cosh'}:
                equal(args[0], one)
                return one
            raise ValueError('Unsupported dimensional inference operator: ' + op)

        equal(visit(equation['ast']), known(target.get('unit')))
        matrix = sp.Matrix.vstack(*(row[0] for row in rows))
        rhs = sp.Matrix.vstack(*(row[1] for row in rows))
        if matrix.rank() != matrix.row_join(rhs).rank():
            return {'status': 'inconsistent'}
        if matrix.rank() < n:
            return {'status': 'underdetermined'}
        solution = matrix.gauss_jordan_solve(rhs)[0]
        labels = {name: coherent_unit_label(Unit(tuple(solution[i, :]))) for i, name in enumerate(names)}
        if any(label is None or unit_spec(label) is None for label in labels.values()):
            raise ValueError('Inferred dimensions exceed the portable unit grammar')
        return {'status': 'unique', 'units': labels,
                'basis': 'unique exact structural dimension solution; not physical parameter identification'}
    except ValueError as exc:
        return {'status': 'unsupported', 'reason': str(exc)}
