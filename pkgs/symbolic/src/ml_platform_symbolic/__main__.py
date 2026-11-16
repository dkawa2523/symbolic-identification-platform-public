import argparse
import json
import sys
from pathlib import Path

from ._version import package_identity
from .config import load_config
from .data import read_json
from .equation_comparison import compare_equations
from .equation_evaluation import create_study, evaluate_equations
from .estimation import fit_hypotheses
from .reporting import render_report
from .selection import assess_prediction_use, choose_hypothesis, finalize_hypothesis
from .stage import run_task


def main():
    commands = {
        "study",
        "fit",
        "evaluate",
        "compare",
        "choose",
        "finalize",
        "report",
        "outputs",
        "prepare",
        "predict",
        "calibrate",
        "assess-use",
        "export",
        "validate",
        "engines",
    }
    parser = argparse.ArgumentParser(description="Scientific equation hypotheses: evaluate, compare, choose explicitly")
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_identity()['version']}")
    if len(sys.argv) == 1 or sys.argv[1] in commands | {"--help", "-h", "--version"}:
        sub = parser.add_subparsers(dest="command", required=True)
        for name in sorted(commands):
            item = sub.add_parser(name)
            item.add_argument(
                "--runtime-extensions", help="Explicit trusted module/version JSON list; never read from a model"
            )
            if name == "engines":
                item.add_argument("input", nargs="?", help="Optional engine name; list all declarations if omitted")
            else:
                item.add_argument("input", help="Plan/config file or prior stage directory")
            if name == "validate":
                item.add_argument("--json", action="store_true", help="Complete input/coordinate/capability review")
            if name in {"report", "outputs"}:
                item.add_argument("--output", required=True)
                item.add_argument("--decision")
                item.add_argument("--audit", help="Finalized result or stored audit_support directory")
                item.add_argument("--supplement", help="Additional comparison bound to the unchanged frozen pool")
                item.add_argument("--show", action="append", help="Display up to three explicit IDs; no reselection")
                if name == "outputs":
                    item.add_argument("--html", action="store_true", help="Also write the optional interactive HTML report")
                    item.add_argument("--no-figures", action="store_true", help="Export equations/tables without plot files")
                    item.add_argument("--interval", help="Stored observation interval, never computed by output")
                    item.add_argument("--coverage", help="Stored matching interval coverage evidence")
            elif name not in {"validate", "engines"}:
                item.add_argument("--run-dir", required=True)
            if name == "predict":
                item.add_argument("--data", required=True)
                item.add_argument("--units", help="JSON mapping of feature input units")
                item.add_argument("--output-unit")
            if name == "export":
                item.add_argument("--decision")
                item.add_argument("--audit")
                item.add_argument("--prediction", help="Explicit saved SAME-subject zip/json; never a native binary")
                item.add_argument("--interval", help="Saved SAME-subject observation interval directory")
                item.add_argument("--coverage", help="Stored matching interval coverage evidence")
            if name == "fit":
                item.add_argument("--id", action="append", required=True)
                item.add_argument("--estimator", choices=("least_squares", "linear_deming"), default="least_squares")
                item.add_argument(
                    "--options", help="JSON estimator-specific mapping; observations/errors belong to the problem"
                )
            if name == "calibrate":
                item.add_argument("--decision", required=True)
                item.add_argument("--data", required=True)
                item.add_argument(
                    "--options",
                    required=True,
                    help="JSON alpha/exchangeability/group/observation assumptions/optional units",
                )
            if name == "assess-use":
                item.add_argument("--interval", required=True)
            if name == "evaluate":
                item.add_argument(
                    "--method",
                    required=True,
                    help="Registered evaluator name; checked after explicit runtime initialization",
                )
                item.add_argument("--options", help="JSON method options; no executable code")
            if name == "compare":
                item.add_argument("--evaluation", action="append", default=[])
                item.add_argument(
                    "--pair-id",
                    action="append",
                    help="Explicit pair-comparison subjects; all candidate evaluations remain",
                )
            if name == "choose":
                item.add_argument("--id")
                item.add_argument("--retain", action="append", default=[])
                item.add_argument("--reason", required=True)
                item.add_argument(
                    "--calibration-plan",
                    help="JSON predeclared policy; freezes predictor rather than adopting before calibration",
                )
            if name == "finalize":
                item.add_argument("--decision", required=True)
        args = parser.parse_args()
        if args.runtime_extensions:
            from .domains import initialize_runtime_extensions

            initialize_runtime_extensions(read_json(args.runtime_extensions))
        if args.command in {"validate", "engines"}:
            from .input_review import engine_catalog, format_input_review, review_inputs

            try:
                if args.command == "engines":
                    print(
                        json.dumps(
                            {
                                "scope": "Declared capabilities only; not installation or native execution evidence",
                                "engines": engine_catalog(args.input),
                            },
                            indent=2,
                        )
                    )
                    return
                review = review_inputs(load_config(args.input))
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
            print(json.dumps(review, indent=2) if args.json else format_input_review(review))
            if review["status"] == "blocked":
                raise SystemExit(2)
            return
        if args.command == "predict":
            import pandas as pd

            from .data import read_observations
            from .equation_evaluation.api import new_directory
            from .hypotheses import load_prediction
            from .model import load_model
            from .prediction_intervals import load_prediction_interval

            frame = read_observations(args.data, identifiers=(), optional_identifiers=("sample_id",))
            directory = new_directory(args.run_dir)
            units = read_json(args.units) if args.units else None
            if Path(args.input).is_dir():
                if (Path(args.input) / "result_manifest.json").is_file():
                    from .portable_results import load_result

                    result = load_result(args.input)
                    model = result["interval"] or result["prediction"]
                    if model is None:
                        raise ValueError(
                            "Result package is readout/native-only; it contains no reusable target predictor"
                        )
                    has_interval = result["interval"] is not None
                else:
                    model = load_prediction_interval(args.input)
                    has_interval = True
                if has_interval:
                    point, low, high = model.predict(frame, units=units, output_unit=args.output_unit)
                    result_frame = pd.DataFrame({"prediction": point, "lower": low, "upper": high})
                else:
                    result_frame = pd.DataFrame(
                        {"prediction": model.predict(frame, units=units, output_unit=args.output_unit)}
                    )
            else:
                model = load_model(args.input) if Path(args.input).suffix == ".zip" else load_prediction(args.input)
                result_frame = pd.DataFrame(
                    {"prediction": model.predict(frame, units=units, output_unit=args.output_unit)}
                )
            if "sample_id" in frame:
                result_frame.insert(0, "sample_id", frame.sample_id.to_numpy())
            result_frame.to_csv(directory / "predictions.csv", index=False)
            result = directory
        elif args.command == "study":
            result = create_study(args.input, run_dir=args.run_dir)
        elif args.command == "export":
            from .portable_results import export_result

            result = export_result(
                args.input,
                run_dir=args.run_dir,
                decision_dir=args.decision,
                audit_dir=args.audit,
                prediction=args.prediction,
                prediction_interval=args.interval,
                coverage=args.coverage,
            )
        elif args.command == "fit":
            result = fit_hypotheses(
                args.input,
                run_dir=args.run_dir,
                ids=args.id,
                estimator=args.estimator,
                options=read_json(args.options) if args.options else None,
            )
        elif args.command == "evaluate":
            result = evaluate_equations(
                args.input, args.method, run_dir=args.run_dir, options=read_json(args.options) if args.options else None
            )
        elif args.command == "compare":
            result = compare_equations(
                args.input, run_dir=args.run_dir, evaluations=args.evaluation, pair_ids=args.pair_id
            )
        elif args.command == "choose":
            result = choose_hypothesis(
                args.input,
                run_dir=args.run_dir,
                candidate_id=args.id,
                reason=args.reason,
                retain=args.retain,
                calibration_plan=read_json(args.calibration_plan) if args.calibration_plan else None,
            )
        elif args.command == "calibrate":
            from .prediction_intervals import calibrate_prediction

            result = calibrate_prediction(
                args.input, args.decision, args.data, run_dir=args.run_dir, **read_json(args.options)
            )
        elif args.command == "assess-use":
            result = assess_prediction_use(args.input, args.interval, run_dir=args.run_dir)
        elif args.command == "finalize":
            result = finalize_hypothesis(args.input, args.decision, run_dir=args.run_dir)
        elif args.command == "outputs":
            from .reporting import export_outputs

            result = export_outputs(
                args.input,
                output_dir=args.output,
                figures=not args.no_figures,
                include_html=args.html,
                decision_dir=args.decision,
                audit_dir=args.audit,
                supplemental_comparison=args.supplement,
                representative_ids=args.show,
                prediction_interval=args.interval,
                coverage=args.coverage,
            )
        elif args.command == "report":
            result = render_report(
                args.input,
                output=args.output,
                decision_dir=args.decision,
                audit_dir=args.audit,
                supplemental_comparison=args.supplement,
                representative_ids=args.show,
            )
        else:
            cfg = load_config(args.input)
            cfg["task"] = "symbolic_stage"
            cfg.setdefault("run", {}).update(stage="prepare")
            result = run_task(cfg, run_dir=Path(args.run_dir))
    else:
        parser.error(
            "Use an explicit command; automatic selection/Final execution is not supported."
        )
    if isinstance(result, Path):
        print(json.dumps({"output": str(result)}, indent=2))
        return
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "metrics": result.metrics,
                "artifacts": {k: str(v) for k, v in result.artifacts.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
