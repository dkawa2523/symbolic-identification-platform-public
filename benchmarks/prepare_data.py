"""Freeze complementary new inputs/criteria before any discovery or Final scoring."""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import ml_platform_symbolic
import numpy as np
import pandas as pd
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.data import prepare, write_json

parser = argparse.ArgumentParser(
    description="Reproduce four complementary research inputs using the existing workbench; no automatic search/Final."
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--raw-root", type=Path, required=True)
parser.add_argument(
    "--fetch",
    action="store_true",
    help="Explicitly download only official UCI316/551 archives; CC BY4 attribution preserved",
)
parser.add_argument("--seed", type=int, default=489077)
args = parser.parse_args()
out = args.output.resolve()
raw = args.raw_root.resolve()
if out.exists():
    raise ValueError("Use a new output directory; preserve data/Final history")
out.mkdir(parents=True)
raw.mkdir(parents=True, exist_ok=True)
archives = {
    "316": (
        "condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants",
        "91a3815da80b5ab7e2d5b82ac82f1c2cbf89182c7a65bcdf240db1e014423cb9",
    ),
    "551": (
        "gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset",
        "55fdb1acc25f05bd9c77aac285c424e9154ffb0dbf1bbadb56b69a1142231295",
    ),
}
sources = {}
for key, (slug, sha) in archives.items():
    path = raw / f"uci{key}.zip"
    url = f"https://archive.ics.uci.edu/static/public/{key}/{slug}.zip"
    if not path.exists():
        if not args.fetch:
            raise ValueError(f"Supply {path} or explicitly choose --fetch; no automatic downloads")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(20_000_000)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(
                "Official archive bytes changed; review a new dataset version rather than silently continuing"
            )
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("Raw archive SHA mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failure")
    sources[key] = dict(
        url=url,
        sha256=sha,
        page=f"https://archive.ics.uci.edu/dataset/{key}",
        license="CC BY4 on official dataset page; cite dataset and associated paper; raw files excluded from Git",
    )
write_json(out / "sources.json", sources)
inputs = out / "inputs"
inputs.mkdir()
rng = np.random.default_rng(args.seed)
cases = {}
operators = ["add", "sub", "mul", "div", "neg", "abs", "sqrt", "pow_int", "exp", "log", "sin", "cos", "tanh"]
profiles = {
    "pysr": dict(
        niterations=100, populations=6, population_size=40, ncycles_per_iteration=100, seconds=45, maxdepth=12
    ),
    "parfam": dict(
        family="rational",
        functions=["sin", "cos"],
        max_active_functions=1,
        degree=2,
        input_degree=2,
        denominator_degree=1,
        maxiter=50,
        repetitions=1,
        seconds=60,
        candidate_scope="completed_families",
    ),
    "pysips": dict(
        particles=100,
        mcmc_samples=10,
        max_complexity=16,
        seconds=40,
        search_operators=["add", "sub", "mul", "div", "exp", "log", "abs", "sin", "cos", "pow_int"],
        opt_restarts=1,
        param_init_bounds=[-5, 5],
    ),
    "mean": {},
    "linear": {},
    "dictionary": dict(max_degree=3, max_terms=8),
}


def save_case(
    name,
    frame,
    features,
    target,
    units,
    *,
    engines,
    group=None,
    config=None,
    tolerance,
    description,
    final_strata,
    observation_model=None,
    coordinate_roles=None,
):
    table = inputs / (name + ".csv")
    split = inputs / (name + "_split.csv")
    frame.drop(columns=["partition"]).to_csv(table, index=False)
    frame[["sample_id", "partition"]].to_csv(split, index=False)
    cfg = {
        "data": dict(
            local_path=str(table),
            feature_columns=features,
            target_column=target,
            units=units,
            coordinate_roles=coordinate_roles or {},
        ),
        "split": dict(file=str(split), group_column=group),
        "search": dict(
            engines=["mean", "linear", "dictionary", *engines],
            operators=operators,
            seeds=[0, 1, 2],
            max_wall_seconds=180,
            max_export_candidates=100,
            candidate_pool_policy="all",
            allow_partial_engines=True,
            engine_options={e: profiles[e] for e in ["mean", "linear", "dictionary", *engines]},
        ),
        "requirements": dict(max_nodes=80, max_free_parameters=18, require_units=False),
        "domain": dict(kind="observed", columns=features, space_id=name + "_observed"),
        "observation_model": observation_model,
    }
    if config:
        cfg.update(config)
    path = write_json(inputs / (name + "_config.json"), cfg)
    prepared = prepare(resolve_config(cfg), out / "prepared" / name)
    cases[name] = dict(
        config=str(path),
        prepared=prepared,
        features=features,
        target=target,
        description=description,
        engineering_tolerance=tolerance,
        units=units,
        group=group,
        final_strata=final_strata,
        counts=frame.partition.value_counts().to_dict(),
        split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
    )


# Independent composition recipes, never a Cartesian cube of impossible mixtures.
parts = []
for role, n, alpha in [("train", 60, 3.0), ("selection", 30, 3.0), ("final", 60, 3.0), ("final", 60, 0.35)]:
    mixtures = rng.dirichlet([alpha] * 3, size=n)
    for x in mixtures:
        group = f"composition-{len(parts)}"
        truth = 1.7 + 2.1 * x[0] + 1.3 * x[1] + 4.2 * x[0] * x[1] + 0.6 * x[0] / (0.3 + x[2])
        group_error = rng.normal(0, 0.01)
        for rep in range(3):
            parts.append(
                dict(
                    sample_id=f"{group}-{rep}",
                    recipe_id=group,
                    a=x[0],
                    b=x[1],
                    c=x[2],
                    response=truth + group_error + rng.normal(0, 0.015),
                    truth=truth,
                    partition=role,
                    stratum="edge_transport" if alpha < 1 else "interior",
                )
            )
frame = pd.DataFrame(parts)
domain = dict(
    kind="affine",
    columns=["a", "b", "c"],
    space_id="composition",
    bounds=[[0, 1]] * 3,
    equalities=dict(matrix=[[1, 1, 1]], rhs=[1]),
)
chart = dict(
    kind="affine_chart",
    space_id="composition_chart",
    target_columns=["u", "v"],
    target_units={"u": "1", "v": "1"},
    forward=dict(matrix=[[1, 0, 0], [0, 1, 0]], offset=[0, 0]),
    inverse=dict(matrix=[[1, 0], [0, 1], [-1, -1]], offset=[0, 0, 1]),
)
save_case(
    "composition",
    frame,
    ["a", "b", "c"],
    "response",
    dict(a="1", b="1", c="1", response="MPa"),
    engines=["pysr", "parfam", "pysips"],
    group="recipe_id",
    config={"domain": domain, "input_map": chart},
    tolerance=0.12e6,
    description="Fresh rational mixture response on a simplex; chart and feasible replacement sensitivity, unseen edge recipes. Synthetic, truth never an input.",
    final_strata=["interior", "edge_transport"],
)

# Observed-input predictive objective differs from latent physical slope recovery.
parts = []
for role, n, limits in [
    ("train", 400, (0.1, 2.0)),
    ("selection", 200, (0.1, 2.0)),
    ("final", 400, (0.1, 2.0)),
    ("final", 200, (2.2, 3.2)),
]:
    latent = rng.uniform(*limits, n)
    x = latent + rng.normal(0, 0.22, n)
    truth = 3.6 * latent + 0.8
    y = truth + rng.normal(0, 0.08, n)
    for i in range(n):
        parts.append(
            dict(
                sample_id=f"eiv-{len(parts)}",
                x=x[i],
                response=y[i],
                latent_x=latent[i],
                truth=truth[i],
                partition=role,
                stratum="latent_ood" if limits[0] > 2 else "id",
            )
        )
frame = pd.DataFrame(parts)
observation = dict(
    target_meaning="observed_response",
    errors=dict(
        kind="independent_additive_known_sigma",
        inputs={"x": dict(value=0.22, unit="m")},
        target=dict(value=0.08, unit="N"),
    ),
    assumptions="Independently generated additive Gaussian x/y errors, known standard deviations, same latent affine law; blind discovery is not supplied that structure.",
)
save_case(
    "eiv",
    frame,
    ["x"],
    "response",
    dict(x="m", response="N"),
    engines=["pysr"],
    tolerance=0.8,
    description="Fresh noisy-input mechanical calibration; compare blind predictive equations with explicitly informed affine LS/Deming estimators, not blanket superiority.",
    final_strata=["id", "latent_ood"],
    observation_model=observation,
)

with zipfile.ZipFile(raw / "uci316.zip") as z:
    array = np.loadtxt(io.BytesIO(z.read("UCI CBM Dataset/data.txt")))
names = [
    "lever",
    "speed",
    "GTT",
    "GTn",
    "GGn",
    "Ts",
    "Tp",
    "T48",
    "T1",
    "T2",
    "P48",
    "P1",
    "P2",
    "Pexh",
    "TIC",
    "mf",
    "kMc",
    "kMt",
]
naval = pd.DataFrame(array, columns=names)
naval["state"] = naval.kMc.map(lambda v: f"{v:.3f}") + "_" + naval.kMt.map(lambda v: f"{v:.3f}")
states = rng.permutation(naval.state.unique())
labels = {s: "train" if i < 300 else "selection" if i < 400 else "final" for i, s in enumerate(states[:500])}
naval = naval[naval.state.isin(labels)].copy()
naval["partition"] = naval.state.map(labels)
# Highest speed absent from all Train/Selection. Final speed effect is conditional
# on ALSO unseen states; no claim that this isolates a speed causal intervention.
naval = naval[(naval.partition == "final") | (naval.speed < 27)].copy()
naval["sample_id"] = ["naval-" + str(i) for i in naval.index]
naval["stratum"] = np.where(naval.speed >= 27, "new_state_and_speed", "new_state_seen_speed")
naval = naval.rename(columns={"kMc": "response"})
save_case(
    "naval",
    naval,
    ["T48", "T2", "P48", "P2", "mf"],
    "response",
    dict(T48="degC", T2="degC", P48="bar", P2="bar", mf="kg/s", response="1"),
    engines=["pysr", "parfam"],
    group="state",
    tolerance=0.004,
    description="Official UCI316 simulation. 500 preselected state-pairs; whole state groups. Propeller torque ambiguous units excluded; degradation labels and IDs excluded from X. Highest speed only Final, confounded with unseen-state transport.",
    final_strata=["new_state_seen_speed", "new_state_and_speed"],
    observation_model=dict(
        target_meaning="latent_state",
        assumptions="Supervised simulator state label from steady-state sensors; inverse regression, not equation coefficient identification.",
    ),
    coordinate_roles={"T48": "value", "T2": "value"},
)

years = []
with zipfile.ZipFile(raw / "uci551.zip") as z:
    for year in range(2011, 2016):
        f = pd.read_csv(io.BytesIO(z.read(f"gt_{year}.csv")))
        f["year"] = year
        f["row_in_year"] = np.arange(len(f))
        f["sample_id"] = [f"gt-{year}-{i}" for i in range(len(f))]
        if year <= 2012:
            f = f.iloc[np.linspace(0, len(f) - 1, 1000, dtype=int)].copy()
            f["partition"] = "train"
        elif year == 2013:
            f = f.iloc[np.linspace(0, len(f) // 2 - 1, 800, dtype=int)].copy()
            f["partition"] = "selection"
        else:
            f = f.iloc[np.linspace(0, len(f) - 1, 1000, dtype=int)].copy()
            f["partition"] = "final"
        f["stratum"] = str(year)
        years.append(f)
gas = pd.concat(years).rename(columns={"NOX": "response"})
if "response" not in gas:
    gas = gas.rename(columns={"NOx": "response"})
save_case(
    "gas_turbine",
    gas,
    ["AT", "AP", "AH", "TIT", "TAT", "TEY"],
    "response",
    dict(AT="degC", AP="mbar", AH="pct", TIT="degC", TAT="degC", TEY="MWh", response="mg/m3"),
    engines=["pysr", "parfam"],
    tolerance=8e-6,
    description="Official UCI551 real plant, chronological year-blocked reduced sample:2011/12 Train, early2013 Selection,2014/15 Final. No timestamps inferred, no random-row split, no causal emission law. CDP excluded due unit-scale ambiguity. No exchangeability coverage guarantee.",
    final_strata=["2014", "2015"],
    observation_model=dict(
        target_meaning="observed_response",
        assumptions="Hourly aggregate observations, temporal dependence and year shift; fixed-X likelihood or exchangeable-row conformal coverage is not presumed.",
    ),
    coordinate_roles={"AT": "value", "TIT": "value", "TAT": "value"},
)

jobs = []
for name, case in cases.items():
    cfg = json.loads(Path(case["config"]).read_text())
    for engine in cfg["search"]["engines"]:
        for seed in [0] if engine in ("mean", "linear", "dictionary") else [0, 1, 2]:
            profile = (
                "posterior"
                if engine == "pysips"
                else "core"
                if engine in ("mean", "linear", "dictionary")
                else "classic"
            )
            jobs.append(
                dict(
                    id=len(jobs),
                    case=name,
                    engine=engine,
                    seed=seed,
                    profile=profile,
                    prepared=case["prepared"],
                    run_dir=str(out / "runs" / name / engine / str(seed)),
                )
            )

package = Path(ml_platform_symbolic.__file__).parent
source = {
    str(f.relative_to(package)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in package.rglob("*")
    if f.suffix in (".py", ".jl")
}
plan = dict(
    schema_version=1,
    data_seed=args.seed,
    cases=cases,
    jobs=jobs,
    profiles=profiles,
    source=source,
    policy="Input/protocol reference only. This generator does not execute jobs. All generation and Train estimator variants, all per-method/pooled choices and calibration plans freeze before any scientific Final. Baselines reused seed0; 3 actual native seeds. No re-tuning on Final. Technical geometry-example-02 is already exposed and excluded.",
    selection_rule="Per method/seed: eligible candidate minimum Selection RMSE then common nodes; pooled shortlist retains different user-annotated hypotheses, with syntax never equated to scientific law. Tolerance controls prediction use, not proof of correct physics.",
    interpretation="Public real data have no known true formula. Synthetic function agreement at new points is not symbolic proof. EIV Deming reference is informed by affine family and known measurement variances, never blind recovery. All no-candidate/partial/native unavailability retained.",
    uncertainty="Only explicitly frozen predictors on new independent synthetic calibration may receive group/row split-conformal intervals. Temporal gas data: empirical residuals, no exchangeability guarantee. PySIPS native structural weights conditional on fitted constants, not observation UQ.",
    oracle="Synthetic truth/latent_x columns and source law are scorer-only, excluded from feature columns and native requests.",
    expected_work="24 native +12 reused baseline jobs, sequential Docker,180s maximum/job including worker startup; this is an upper bound, not an ETA. No all-method/all-case campaign.",
)
write_json(out / "plan.json", plan)
print(
    json.dumps(
        {
            "cases": {k: v["counts"] for k, v in cases.items()},
            "jobs": len(jobs),
            "plan_sha256": hashlib.sha256((out / "plan.json").read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
