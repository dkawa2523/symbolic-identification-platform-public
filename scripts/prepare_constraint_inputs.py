"""Freeze one data/split/schema bundle per maintained case; no discovery or Final scoring."""

import argparse
import shutil
from pathlib import Path

from benchmark_symbolic import MAINTAINED_CASES, external, synthetic
from ml_platform_symbolic.data import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=MAINTAINED_CASES, default=list(MAINTAINED_CASES))
    parser.add_argument("--data-seed", type=int, default=2026102601)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for case in args.cases:
        folder = root / case
        folder.mkdir()
        if case in {"yacht", "airfoil", "concrete"}:
            config = external(case)
            source = Path(config["data"]["local_path"])
            shutil.copyfile(source, folder / "table.csv")
            shutil.copyfile(config["split"]["file"], folder / "split.csv")
            if source.with_name("metadata.json").is_file():
                shutil.copyfile(source.with_name("metadata.json"), folder / "metadata.json")
            config["data"]["local_path"] = str(folder / "table.csv")
            config["split"]["file"] = str(folder / "split.csv")
            history = "public data previously exposed in this project; prospective audit, not untouched confirmation"
        else:
            config = synthetic(folder, case, 0, data_seed=args.data_seed)
            history = "new generated points/noise; known mathematical family; no pretraining-novelty guarantee"
        write_json(folder / "problem.json", config)
        manifest[case] = {
            "problem": str(folder / "problem.json"),
            "history": history,
            "files": {p.name: digest(p) for p in folder.iterdir() if p.is_file()},
        }
        print(case, manifest[case]["files"]["table.csv"], flush=True)
    write_json(root / "inputs.json", {"data_seed": args.data_seed, "cases": manifest})


if __name__ == "__main__":
    main()
