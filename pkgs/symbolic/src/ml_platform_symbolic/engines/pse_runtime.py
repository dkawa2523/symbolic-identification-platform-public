"""Token-safe substitutions in pinned PSRN; applied only at Docker image setup."""

import argparse
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = "7a21bb12a5acc2e27e83d26a96e090c4fd4c13e01095dafbef60b8a68d0f9d28"
MARKER = "symbolic_handoff_constant_version = 2"


def patch(package_dir):
    path = Path(package_dir) / "model" / "regressor.py"
    source = path.read_bytes()
    if MARKER.encode() in source:
        return hashlib.sha256(source).hexdigest()
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Unexpected pinned PSRN source; constant substitution patch was not applied")
    text = source.decode("utf-8").replace("\r\n", "\n")
    replacements = {
        'r"(?<![a-zA-Z]){}(?![a-zA-Z])".format(variable)': 'r"(?<![a-zA-Z0-9_]){}(?![a-zA-Z0-9_])".format(re.escape(variable))',
        '            for i, c in enumerate(C):\n                expr_c_temp = expr_c_temp.replace("C{}".format(i), str(c))': '            expr_c_temp = re.sub(r"\\bC(\\d+)\\b", lambda m: "(" + repr(float(C[int(m.group(1))])) + ")", expr_c_temp)',
        '        for i, c in enumerate(best_C):\n            final_c = final_c.replace("C{}".format(i), str(c))': '        final_c = re.sub(r"\\bC(\\d+)\\b", lambda m: "(" + repr(float(best_C[int(m.group(1))])) + ")", final_c)',
        "            fitted_c_expr = str(fitted_c)": '            fitted_c_expr = str(fitted_c) if self.prun_const else repr(float(fitted_c))\n            if not self.prun_const and hasattr(self, "_symbolic_record_initial"):\n                self._symbolic_record_initial("mean", fitted_c_expr, [float(fitted_c)])',
        "            fitted_linear_expr = str(reg.intercept_[0])": "            fitted_linear_expr = str(reg.intercept_[0]) if self.prun_const else repr(float(reg.intercept_[0]))",
        '" + " + str(reg.coef_[0][i]) + "*" + str(self.variables[i])': '" + " + (str(reg.coef_[0][i]) if self.prun_const else repr(float(reg.coef_[0][i]))) + "*" + str(self.variables[i])',
        "            fitted_linear_expr = str(\n                prun_constant(sympy.sympify(fitted_linear_expr), self.prun_ndigit)\n            )": '            if self.prun_const:\n                fitted_linear_expr = str(prun_constant(sympy.sympify(fitted_linear_expr), self.prun_ndigit))\n            elif hasattr(self, "_symbolic_record_initial"):\n                self._symbolic_record_initial("linear", fitted_linear_expr, [float(reg.intercept_[0]), *map(float, reg.coef_[0])])',
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError("Pinned PSRN substitution anchor mismatch")
        text = text.replace(old, new)
    anchor = "                    expr = str(expr_sympy)"
    if text.count(anchor) != 1:
        raise ValueError("Pinned PSRN Pareto observation anchor mismatch")
    text = text.replace(
        anchor,
        anchor
        + '\n                    if hasattr(self, "_symbolic_record_pareto"):\n                        self._symbolic_record_pareto(sample[0], expr)',
    )
    text += "\n" + MARKER + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    print(patch(parser.parse_args().package_dir))
