import math
from copy import deepcopy
from pathlib import Path

from ml_platform_core.config import deep_merge

from .coordinates import canonical_problem
from .domains import validate_bounds
from .expression_graph import MAX_STORED_NODES
from .expressions import ARITY
from .manifest import DEFAULTS
from .problem import unpack_requirements
from .response_sampling import materialize_response_sampling
from .units import quantity_role, require_canonical, unit_spec, validate_unit_mapping


def load_config(path):
    """Preserve JSON numbers (YAML 1.1 treats JSON 1e-6 as a string)."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        import yaml

        class UniqueLoader(yaml.SafeLoader):
            def construct_mapping(self, node, deep=False):
                self.flatten_mapping(node)
                mapping = {}
                for key_node, value_node in node.value:
                    key = self.construct_object(key_node, deep=deep)
                    try:
                        duplicate = key in mapping
                    except TypeError as exc:
                        raise ValueError("YAML setting keys must be scalar values") from exc
                    if duplicate:
                        raise ValueError(f"Duplicate YAML setting: {key}; keep one explicit value")
                    mapping[key] = self.construct_object(value_node, deep=deep)
                return mapping

        try:
            return yaml.load(path.read_text(encoding="utf-8-sig"), Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {exc}") from exc

    from .data import read_json

    return read_json(path)


def _string_list(value, name, *, nonempty=True):
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}list of distinct nonempty strings")


def _finite_number(value, name, *, minimum=0, strict=False):
    if type(value) not in {int, float} or not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be a finite number {'>' if strict else '>='} {minimum}")


def _validate_columns(cfg):
    data = cfg["data"]
    _string_list(data["feature_columns"], "data.feature_columns")
    _string_list(data["id_columns"], "data.id_columns", nonempty=False)
    target = data["target_column"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("data.target_column must name one nonempty response column")
    group = cfg["split"]["group_column"]
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError("split.group_column must be a nonempty column name or null")
    if set(data["feature_columns"]) & {target, "sample_id", group, *data["id_columns"]}:
        raise ValueError("Features must be distinct and exclude target, IDs and group columns")
    if target in {"sample_id", group, *data["id_columns"]}:
        raise ValueError("data.target_column must exclude identity and group columns")


def _check_unknown(config):
    for section, values in config.items():
        if section not in DEFAULTS:
            raise ValueError(f"Unknown setting: {section}")
        if isinstance(DEFAULTS[section], dict):
            if not isinstance(values, dict):
                raise ValueError(f"{section} must be a mapping")
            if section not in {"stage_inputs", "model_assets"}:
                unknown = set(values) - set(DEFAULTS[section])
                if unknown:
                    raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")


def _validate_split(cfg):
    if cfg["split"]["method"] not in {"fixed", "group"}:
        raise ValueError("split.method must be fixed or group")
    if cfg["split"]["method"] == "fixed" and not cfg["split"]["file"]:
        raise ValueError("Fixed split requires split.file")
    if cfg["split"]["method"] == "group" and not cfg["split"]["group_column"]:
        raise ValueError("Group split requires split.group_column")
    for key in ("valid_size", "selection_size"):
        value = cfg["split"][key]
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"split.{key} must be in (0,1)")
    if type(cfg["split"]["seed"]) is not int or cfg["split"]["seed"] < 0:
        raise ValueError("split.seed must be a nonnegative integer")


def _validate_search(search):
    if search["response_mode"] not in {"search", "filter"}:
        raise ValueError("search.response_mode must be search or explicit filter-only control")

    from .engines.registry import ENGINES

    _string_list(search["engines"], "search.engines")
    unknown = set(search["engines"]) - set(ENGINES)
    if unknown:
        raise ValueError(f"Unsupported engines {sorted(unknown)}; available: {sorted(ENGINES)}")
    _string_list(search["operators"], "search.operators")
    if (
        not isinstance(search["seeds"], list)
        or not search["seeds"]
        or any(type(seed) is not int or seed < 0 for seed in search["seeds"])
        or len(set(search["seeds"])) != len(search["seeds"])
    ):
        raise ValueError("Engines and nonnegative integer seeds must be nonempty and unique")
    if set(search["operators"]) - set(ARITY) or not search["operators"]:
        raise ValueError("Unsupported/empty operators")
    if type(search["max_export_candidates"]) is not int or search["max_export_candidates"] < 1:
        raise ValueError("search.max_export_candidates must be a positive integer")
    _finite_number(search["max_wall_seconds"], "search.max_wall_seconds", strict=True)
    if type(search["allow_partial_engines"]) is not bool:
        raise ValueError("search.allow_partial_engines must be true or false")
    for key in ("engine_options", "runtimes"):
        if not isinstance(search[key], dict) or any(not isinstance(v, dict) for v in search[key].values()):
            raise ValueError(f"search.{key} must map selected engine names to settings objects")
    if search["candidate_pool_policy"] not in {"all", "pareto_diverse"}:
        raise ValueError("candidate_pool_policy must be all or pareto_diverse")
    if set(search["engine_options"]) - set(search["engines"]):
        raise ValueError("engine_options contains an unselected engine")
    if set(search["runtimes"]) - set(search["engines"]):
        raise ValueError("runtimes contains an unselected engine")
    for runtime in search["runtimes"].values():
        if not isinstance(runtime, dict) or set(runtime) - {"python", "device"}:
            raise ValueError("Each search.runtimes entry accepts only python and device")
        if "python" in runtime and (not isinstance(runtime["python"], str) or not Path(runtime["python"]).is_file()):
            raise ValueError("Runtime python must be an existing local interpreter path")
        if "device" in runtime and not isinstance(runtime["device"], str):
            raise ValueError("Runtime device must be a string")


def _validate_constraints(constraints):
    enforcement = constraints.get("response_enforcement")
    if enforcement is not None and (not isinstance(enforcement, str) or enforcement not in {"hard", "soft"}):
        raise ValueError("Response enforcement must be null, hard or soft")

    if type(constraints["max_nodes"]) is not int or not 1 <= constraints["max_nodes"] <= MAX_STORED_NODES:
        raise ValueError("Invalid node limit")
    if type(constraints["require_units"]) is not bool:
        raise ValueError("constraints.require_units must be true or false")
    if type(constraints["unit_search"]) is not bool:
        raise ValueError("constraints.unit_search must be true or false")
    for key in ("probes", "response_checks"):
        if not isinstance(constraints[key], list):
            raise ValueError(f"constraints.{key} must be a list")
    validate_bounds(constraints["response_bounds"])
    if constraints["response_bounds"] != [None, None] and not constraints["probes"]:
        raise ValueError("response_bounds requires explicit probes")
    cap = constraints["max_free_parameters"]
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("max_free_parameters must be a nonnegative integer or null")
    for request in constraints["response_checks"]:
        required = {"order", "sign", "step", "tolerance", "points"}
        if (
            not isinstance(request, dict)
            or not required <= request.keys()
            or request.keys() - required - {"name", "knowledge", "feature", "direction", "path_unit"}
            or (("feature" in request) == ("direction" in request))
            or ("direction" in request and "path_unit" not in request)
        ):
            raise ValueError(
                "Each response check needs feature OR explicit direction/path_unit plus order/sign/step/tolerance/points"
            )
        if type(request["order"]) is not int or request["order"] not in {1, 2} or request["sign"] not in {-1, 1}:
            raise ValueError("Response checks support order 1/2 and sign -1/+1")
        if not request["points"]:
            raise ValueError("Response checks require explicit nonempty condition points")
        for key in ("step", "tolerance"):
            value = request[key]
            validate_bounds([value, value])
            if value is None or value < 0 or (key == "step" and value == 0):
                raise ValueError("Response step must be positive and tolerance nonnegative")


def _validate_evaluation(evaluation):
    if evaluation["mode"] not in {"holdout", "exploratory"} or evaluation["coefficient_fit"] not in {
        "none",
        "bounded_least_squares",
    }:
        raise ValueError("Unsupported evaluation mode / coefficient_fit")
    if evaluation["selection_policy"] not in {"balanced", "accuracy", "simplicity"}:
        raise ValueError("Unsupported selection policy")
    if evaluation["selection_policy"] == "simplicity" and evaluation["acceptable_rmse"] is None:
        raise ValueError("simplicity requires evaluation.acceptable_rmse")
    for key in ("relative_tolerance", "absolute_tolerance"):
        _finite_number(evaluation[key], f"evaluation.{key}")
    if evaluation["acceptable_rmse"] is not None:
        _finite_number(evaluation["acceptable_rmse"], "evaluation.acceptable_rmse")


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON/YAML object with data, split and search settings")
    if "problem_schema_version" in config:
        from .problem import problem_config

        config = problem_config(config)
    if "_prepared_requirements_ref" in config:
        raise ValueError("Prepared config is not standalone; pass its complete problem document or use load_problem")
    config = materialize_response_sampling(unpack_requirements(config))
    _check_unknown(config)
    cfg = deep_merge(deepcopy(DEFAULTS), dict(config))
    if cfg["task"] not in {"symbolic_pipeline", "symbolic_stage", "symbolic_infer"}:
        raise ValueError("Unsupported symbolic task")
    if cfg["task"] == "symbolic_infer":
        if not cfg["model"]["local_model_path"] or not cfg["data"]["local_path"]:
            raise ValueError("Inference requires model.local_model_path and data.local_path")
        for key in ("units", "source_units"):
            if not isinstance(cfg["data"][key], dict):
                raise ValueError("Inference units/source_units must be mappings; columns are checked against the model")
        if cfg["data"]["units"] and cfg["data"]["source_units"]:
            raise ValueError("Inference accepts data.units OR data.source_units for CSV coordinates, not both")
        if cfg["data"]["coordinate_roles"]:
            raise ValueError(
                "Inference cannot override saved quantity coordinate_roles; declare meaning before fitting"
            )
        return cfg
    observation = cfg.get("observation_model")
    _validate_columns(cfg)
    display = cfg["data"]["display_units"]
    columns = {*cfg["data"]["feature_columns"], cfg["data"]["target_column"]}
    if (not isinstance(display, dict) or set(display) - columns
            or any(not isinstance(label, str) or not label.strip() or "\n" in label
                   for label in display.values())):
        raise ValueError("data.display_units maps declared columns to nonempty single-line source labels; no conversion")
    if observation is not None:
        if not isinstance(observation, dict) or set(observation) - {"target_meaning", "errors", "assumptions"}:
            raise ValueError("observation_model declares target_meaning, optional errors and assumptions")
        if observation.get("target_meaning") not in {"observed_response", "latent_state", "latent_relation_response"}:
            raise ValueError(
                "Declare whether the target is an observed response, latent-state label or latent-relation response"
            )
        errors = observation.get("errors")
        if errors is not None:
            if (
                not isinstance(errors, dict)
                or errors.get("kind") != "independent_additive_known_sigma"
                or set(errors) != {"kind", "inputs", "target"}
            ):
                raise ValueError(
                    "Supported declared error metadata is independent_additive_known_sigma; other likelihoods remain explicit evaluator settings"
                )
            if set(errors["inputs"]) - set(cfg["data"]["feature_columns"]):
                raise ValueError("Input-error metadata names original observed columns")
            from .units import convert_width

            for column, declaration in [*errors["inputs"].items(), (cfg["data"]["target_column"], errors["target"])]:
                if set(declaration) != {"value", "unit"} or declaration["value"] <= 0:
                    raise ValueError("Known standard deviations need positive values and explicit units")
                convert_width(declaration["value"], declaration["unit"], cfg["data"]["units"].get(column))
    _validate_split(cfg)
    columns = [*cfg["data"]["feature_columns"], cfg["data"]["target_column"]]
    for key in ("units", "source_units"):
        validate_unit_mapping(cfg["data"][key], columns)
    roles = cfg["data"]["coordinate_roles"]
    if (
        not isinstance(roles, dict)
        or set(roles) - set(columns)
        or any(role not in {None, "value", "difference"} for role in roles.values())
    ):
        raise ValueError("data.coordinate_roles uses declared columns and value/difference (or unknown null)")

    if not isinstance(cfg["data"]["domain"], dict) or not isinstance(cfg["data"]["fixed_constants"], dict):
        raise ValueError("data.domain and data.fixed_constants must be mappings")
    declared_columns = {*cfg["data"]["feature_columns"], cfg["data"]["target_column"]}
    for column, bounds in cfg["data"]["domain"].items():
        if column not in declared_columns:
            raise ValueError(f"Domain refers to an unknown feature/target: {column}")
        validate_bounds(bounds)
    for name, constant in cfg["data"]["fixed_constants"].items():
        if not isinstance(name, str) or not name.isidentifier() or name in declared_columns:
            raise ValueError("Fixed constant names must be identifiers distinct from columns")
        if (
            not isinstance(constant, dict)
            or not {"value", "unit"} <= set(constant)
            or set(constant) - {"value", "unit", "coordinate_role"}
        ):
            raise ValueError("Each fixed constant requires value and unit")
        validate_bounds([constant["value"], constant["value"]])
        if constant["value"] is None or not isinstance(constant["unit"], (str, type(None))):
            raise ValueError("Fixed constants require a finite value and a unit label or null")

    _validate_search(cfg["search"])
    if cfg["constraints"].get("response_enforcement") is not None and cfg["search"]["response_mode"] != "search":
        raise ValueError("response_enforcement requires response_mode=search; filter is post-check only")
    if set(cfg["model_assets"]) - set(cfg["search"]["engines"]):
        raise ValueError("model_assets contains an unselected engine")
    for asset in cfg["model_assets"].values():
        if (
            not isinstance(asset, dict)
            or set(asset) != {"manifest_file"}
            or not isinstance(asset["manifest_file"], str)
        ):
            raise ValueError("Each model_assets entry requires only manifest_file")
    if not isinstance(cfg["_sampling_provenance"], list):
        raise ValueError("Internal sampling provenance must be a list")
    _validate_constraints(cfg["constraints"])
    columns = cfg["data"]["feature_columns"]
    for point in cfg["constraints"]["probes"]:
        if not isinstance(point, dict) or set(point) != set(columns):
            raise ValueError("Constraint probes must contain exactly the original input columns")
        for value in point.values():
            validate_bounds([value, value])
            if value is None:
                raise ValueError("Constraint probe coordinates must be finite")
    for request in cfg["constraints"]["response_checks"]:
        if "feature" in request and request["feature"] not in columns:
            raise ValueError("Response feature must use an original input column")
        if "direction" in request:
            direction = request["direction"]
            if not isinstance(direction, dict) or set(direction) != set(columns):
                raise ValueError("Response direction names every original input column, including zeros")
            for value in direction.values():
                validate_bounds([value, value])
            if not any(direction.values()):
                raise ValueError("Response direction must be nonzero")
            if unit_spec(request["path_unit"]) is None:
                raise ValueError("Response path parameter needs a known unit")
        for point in request["points"]:
            if not isinstance(point, dict) or set(point) != set(columns):
                raise ValueError("Response points must contain exactly the original input columns")
            for column, value in point.items():
                validate_bounds([value, value])
                if value is None:
                    raise ValueError("Response point coordinates must be finite")
                direction = request.get("direction", {})
                delta = request["step"] * (direction[column] if direction else float(column == request["feature"]))
                bounds = cfg["data"]["domain"].get(column, [None, None])
                if (bounds[0] is not None and value - abs(delta) < bounds[0]) or (
                    bounds[1] is not None and value + abs(delta) > bounds[1]
                ):
                    raise ValueError("Response condition or finite step exceeds declared input domain")
    allowed = cfg["constraints"].get("allowed_variables")
    if allowed is not None:
        _string_list(allowed, "constraints.allowed_variables", nonempty=False)
        names = {column: f"x{i}" for i, column in enumerate(cfg["data"]["feature_columns"])}
        ids = set(names.values())
        if any(name not in names and name not in ids for name in allowed):
            raise ValueError("allowed_variables requires original feature columns (or unambiguous legacy IDs)")
        if any(name in names and name in ids and names[name] != name for name in allowed):
            raise ValueError("Ambiguous original column/legacy ID in allowed_variables")
        cfg["constraints"]["allowed_variables"] = [names.get(name, name) for name in allowed]
    _validate_evaluation(cfg["evaluation"])
    if (cfg["search"]["engine_options"].get("e2e", {}).get("rational_domain_admission") is not None
            and cfg["evaluation"]["coefficient_fit"] != "none"):
        raise NotImplementedError(
            "E2E rational_domain_admission requires evaluation.coefficient_fit=none; "
            "automatic refitting would invalidate the fixed-coefficient proof"
        )
    cfg = canonical_problem(cfg)
    for label in cfg["data"]["units"].values():
        require_canonical(label)
    for constant in cfg["data"]["fixed_constants"].values():
        require_canonical(constant["unit"])
        quantity_role(constant)
    for column, label in cfg["data"]["units"].items():
        quantity_role({"unit": label, "coordinate_role": cfg["data"]["coordinate_roles"].get(column)})
    template = cfg["constraints"].get("equation_template")
    if template is not None:
        from .equation_templates import validate_template

        variables = [
            {
                "id": f"x{i}",
                "column": column,
                "unit": cfg["data"]["units"].get(column),
                "coordinate_role": cfg["data"]["coordinate_roles"].get(column),
                "domain": cfg["data"]["domain"].get(column),
            }
            for i, column in enumerate(cfg["data"]["feature_columns"])
        ]
        if cfg.get("input_map") is not None:
            from .coordinates import freeze_input_geometry

            # 'working' means the model's actual execution coordinates. Validate
            # against the same frozen chart used by preparation and inference,
            # not source-column positions or manually duplicated target units.
            _, variables = freeze_input_geometry(cfg, variables)
        target_column = cfg["data"]["target_column"]
        target = {
            "column": target_column,
            "unit": cfg["data"]["units"].get(target_column),
            "coordinate_role": cfg["data"]["coordinate_roles"].get(target_column),
        }
        cfg["constraints"]["equation_template"] = validate_template(
            template, variables, target, cfg["search"]["operators"], cfg["data"]["fixed_constants"]
        )
    # Keep historical resolved/prepared shapes identical unless explicitly enabled.
    if not cfg["constraints"].get("unit_search", False):
        cfg["constraints"].pop("unit_search", None)
    return cfg
