"""Public declarations belong to the domain; SDK transport belongs to spdml."""

from ml_platform_core.contracts import PackageManifest, ParameterSpec, StageSpec, TaskSpec

DEFAULTS = {
    "task": "symbolic_pipeline",
    "_sampling_provenance": [],
    "domain": None,
    "input_map": None,
    "observation_model": None,
    "run": {"stage": None, "engine": None, "seed": 42},
    "data": {
        "local_path": None,
        "target_column": None,
        "feature_columns": [],
        "id_columns": ["sample_id"],
        "units": {},
        "source_units": {},
        "display_units": {},
        "coordinate_roles": {},
        "domain": {},
        "fixed_constants": {},
    },
    "split": {
        "method": "fixed",
        "file": None,
        "group_column": None,
        "seed": 42,
        "valid_size": 0.2,
        "selection_size": 0.2,
    },
    "search": {
        "engines": ["dictionary"],
        "operators": ["add", "sub", "mul", "div"],
        "seeds": [42],
        "max_wall_seconds": 300,
        "max_export_candidates": 100,
        "candidate_pool_policy": "all",
        "allow_partial_engines": False,
        "response_mode": "search",
        "engine_options": {},
        "runtimes": {},
    },
    "constraints": {
        "equation_template": None,
        "max_nodes": 30,
        "max_free_parameters": None,
        "require_units": False,
        "unit_search": False,
        "allowed_variables": None,
        "probes": [],
        "response_bounds": [None, None],
        "response_checks": [],
        "response_enforcement": None,
    },
    "evaluation": {
        "mode": "holdout",
        "coefficient_fit": "none",
        "selection_policy": "balanced",
        "relative_tolerance": 0.05,
        "absolute_tolerance": 0.0,
        "acceptable_rmse": None,
    },
    "model": {"local_model_path": None},
    "model_assets": {},
    "stage_inputs": {},
}
_VALUE_TYPES = {bool: "bool", int: "int", float: "float", str: "str", dict: "dict", list: "list", type(None): "str"}
PARAMETERS = (
    ParameterSpec("Run/task", "str", ("task",), "Prepared configuration, explicit stage or portable model inference"),
    ParameterSpec("Constraints/probe_sampling", "dict", ("constraints", "probe_sampling"),
                  "Optional finite Sobol box points: domain, points (power of two), corners budget, seed; not a global guarantee"),
    ParameterSpec("ModelAssets/manifests", "dict", ("model_assets",), "Fixed local per-engine asset manifests"),
    *tuple(
        ParameterSpec(
            f"{'Input' if section == 'data' else section.title()}/{key}",
            _VALUE_TYPES[type(value)],
            (section, key),
            ("Explicit response checks; each supports points and/or optional finite-box sampling"
             if section == "constraints" and key == "response_checks" else f"Symbolic {section}.{key}"),
        )
        for section, values in DEFAULTS.items()
        if isinstance(values, dict)
        for key, value in values.items()
    ),
)
ARTIFACTS = {
    "prepare": ("run", "train", "selection", "final"),
    "discover": ("candidates", "engine_summary"),
    "infer": ("predictions", "result_summary"),
}
SYMBOLIC_MANIFEST = PackageManifest(
    tasks=tuple(TaskSpec(k, PARAMETERS) for k in ("symbolic_stage", "symbolic_infer")),
    stages=tuple(StageSpec(k, PARAMETERS, v) for k, v in ARTIFACTS.items()),
)
