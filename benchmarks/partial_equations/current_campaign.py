"""Fixed-seed partial-equation development check using the common study/test scorer."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_symbolic import create_study, run_task
from ml_platform_symbolic.data import digest, read_json, write_json
from ml_platform_symbolic.domains import region_mask


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
definitions = module(HERE / "definitions.py", "partial_definitions")
oracle = module(HERE / "prepare_data.py", "partial_data")
scorer = module(HERE.parent / "score.py", "common_benchmark_score")
CASES = ["additive", "multiplicative", "shared_component", "positive_domain", "wrong_prior", "zero_factor"]


def prepare_cases(root, cases, engines, seconds, structures):
    root.mkdir(parents=True, exist_ok=False)
    # Declared before data generation. Extrapolation is outside observation,
    # not outside the permissible prediction domain. No retrospective expansion.
    protocol = {"permitted_bounds": [[-1., 2.], [-1., 2.]], "train_selection_bounds": [-1., 1.],
                "test_regions": {"interpolation": [-1., 1.], "extrapolation": [1.2, 2.]},
                "search_seed": 17, "test_seed": 981173, "test_rows_per_region": 32,
                "final_history": "known_exposed", "scope": "Historical disclosed synthetic families/seeds; development regression, not independent law discovery"}
    write_json(root / "protocol.json", protocol)
    prepared = {}
    for case in cases:
        index = CASES.index(case)
        folder = root / case
        folder.mkdir()
        source = definitions.DATA / "synthetic" / case / "1729"
        split = pd.read_csv(source / "split.csv")
        split = split[split.partition.isin(["train", "selection"])].copy()
        frame = pd.read_csv(source / "table.csv")
        frame = frame[frame.sample_id.isin(split.sample_id)].copy()
        frame["region"] = "observation"
        final_rows = []
        for ri, (region, (low, high)) in enumerate(protocol["test_regions"].items()):
            rng = np.random.default_rng(np.random.SeedSequence([protocol["test_seed"], index, ri]))
            points = rng.uniform(low, high, (protocol["test_rows_per_region"], 2))
            final_rows.extend({"sample_id": f"test_{region}_{i}", "x": x, "z": z,
                               "y": oracle.CASES[case][2](x, z), "region": region}
                              for i, (x, z) in enumerate(points))
        final = pd.DataFrame(final_rows)
        frame = pd.concat([frame, final], ignore_index=True)
        split = pd.concat([split, final[["sample_id"]].assign(partition="final")], ignore_index=True)
        domain = {"kind": "box", "columns": ["x", "z"], "bounds": protocol["permitted_bounds"]}
        assert region_mask(frame[["x", "z"]].to_numpy(), domain).all()
        assert (final.loc[final.region == "extrapolation", ["x", "z"]].to_numpy() > 1.).all()
        assert not frame.sample_id.duplicated().any()
        frame.to_csv(folder / "data.csv", index=False)
        split.to_csv(folder / "split.csv", index=False)
        template = definitions.template(case)
        for slot in template["slots"].values():
            slot["max_nodes"] = 7
        cfg = {"data": {"local_path": str(folder / "data.csv"), "feature_columns": ["x", "z"],
                        "target_column": "y", "units": {"x": "1", "z": "1", "y": "1"},
                        "domain": {"x": [-1., 2.], "z": [-1., 2.]}},
               "split": {"file": str(folder / "split.csv")},
               "search": {"engines": engines, "seeds": [17], "operators": definitions.OPERATORS,
                          "max_wall_seconds": seconds, "allow_partial_engines": True,
                          "engine_options": {"template_enumeration": {"objective": "mse", "order": "balanced",
                              "max_structures": structures, "max_nfev": 50, "max_slot_nodes": 7, "starts": [1., -1.]},
                              "pysr": {"objective": "mse", "niterations": 5, "seconds": seconds * .5,
                                       "populations": 2, "population_size": 20, "ncycles_per_iteration": 10}}},
               "constraints": {"max_nodes": 45, "equation_template": template},
               "evaluation": {"mode": "holdout", "coefficient_fit": "none"}}
        cfg["search"]["engine_options"] = {e: cfg["search"]["engine_options"][e] for e in engines}
        write_json(folder / "problem.json", cfg)
        result = run_task({**cfg, "task": "symbolic_stage", "run": {"stage": "prepare"}}, run_dir=folder / "prepare")
        prepared[case] = result.extra
    write_json(root / "campaign.json", {"protocol_sha256": digest(root / "protocol.json"), "cases": cases,
               "engines": engines, "prepared": prepared, "history": protocol["final_history"], "scope": protocol["scope"]})


def execute(root, stage):
    plan = read_json(root / "campaign.json")
    if "prepared" not in plan or "protocol_sha256" not in plan:
        raise ValueError("Historical campaign layout: preserve its results and prepare a new common-flow directory")
    if digest(root / "protocol.json") != plan["protocol_sha256"]:
        raise ValueError("Protocol changed after prepare")
    for case in plan["cases"]:
        folder, refs = root / case, plan["prepared"][case]
        if stage == "discover":
            for engine in plan["engines"]:
                run_task({"task": "symbolic_stage", "run": {"stage": "discover", "engine": engine, "seed": 17},
                          "stage_inputs": {"prepare": refs}}, run_dir=folder / "calculation" / engine)
        elif stage == "select":
            sources = [{"discovery": str(folder / "calculation" / e / "discover" / "engine_summary.json")} for e in plan["engines"]]
            create_study({"prepared_run": refs["run"], "sources": sources, "purpose": "local_prediction",
                          "question": plan["scope"], "final_history": plan["history"], "strata_column": "region"},
                         run_dir=folder / "study")
        else:
            # Common scorer freezes Selection representatives before reading Final;
            # existing strata reporting separates interpolation and extrapolation.
            scorer.main(folder / "study", folder / "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "discover", "select", "score"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["additive"])
    parser.add_argument("--engines", nargs="+", choices=["template_enumeration", "pysr"], default=["template_enumeration"])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--structures", type=int, default=32)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_cases(args.root.resolve(), args.cases, args.engines, args.seconds, args.structures)
    else:
        execute(args.root.resolve(), args.stage)


if __name__ == "__main__":
    main()
