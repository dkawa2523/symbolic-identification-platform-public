"""The same finite conditions guide search and remain independently testable."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.constraints import prediction_constraints
from ml_platform_symbolic.response_search import compile_response_constraints, response_penalty


def request(sign=1, order=1):
    return SimpleNamespace(
        X=np.arange(5, dtype=float)[:, None], y=np.arange(5, dtype=float),
        variables=[{"id": "x0", "column": "x", "unit": "1", "domain": [-5, 5]}],
        target={"unit": "1", "domain": [-20, 20]},
        limits={"probes": [{"x": 0}], "response_bounds": [-1, 1],
                "response_checks": [{"feature": "x", "order": order, "sign": sign,
                    "step": 0.1, "tolerance": 1e-8, "points": [{"x": 1}, {"x": 2}]}]})


@pytest.mark.parametrize("sign,order", [(1, 1), (-1, 1), (1, 2), (-1, 2)])
def test_search_penalty_matches_existing_checker_at_same_points(sign, order):
    r = request(sign, order)
    compiled = compile_response_constraints(r)
    for slope in (-1, 1):
        def predict(x, slope=slope):
            return slope * x[:, 0]**order
        checks = prediction_constraints(predict, r.variables, r.target, r.limits)
        loss = response_penalty(predict(np.asarray(compiled["points"])), compiled)
        assert (loss == 0) == all(c["status"] == "pass" for c in checks)


def test_invalid_points_never_improve_search_score():
    compiled = compile_response_constraints(request())
    values = np.zeros(len(compiled["points"]))
    values[-1] = np.nan
    assert np.isinf(response_penalty(values, compiled))


def test_search_scale_does_not_use_selection_or_final():
    r = request()
    compiled = compile_response_constraints(r)
    assert compiled["data_scale"] == pytest.approx(np.std(r.y))
    assert all(t["scale"] > 0 for t in compiled["terms"])


def test_unsupported_search_constraints_are_not_silently_filtered():
    from ml_platform_symbolic.engines.runner import engine_input_issues
    job = {"engine": "example", "operators": ["add"], "limits": {
        "response_mode": "search", "probes": [{"x": 0}], "response_bounds": [0, None]}}
    capability = {"operators": ["add"], "requires_assets": False}
    assert engine_input_issues(job, capability)[0]["status"] == "unsupported"
    job["limits"]["response_mode"] = "filter"
    assert engine_input_issues(job, capability) == []
    job["limits"]["response_mode"] = "search"
    capability["search_constraints"] = ["probes", "response_bounds"]
    assert engine_input_issues(job, capability) == []


def test_discovery_and_common_check_share_domain_points(synthetic_config, tmp_path):
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import load_partition, prepare, read_json
    from ml_platform_symbolic.domains import validation_constraints
    from ml_platform_symbolic.engines.runner import discovery_job

    synthetic_config["data"]["domain"] = {"x": [0, 4]}
    synthetic_config["constraints"].update(probes=[{"x": 0.5}], response_bounds=[0, 40])
    refs = prepare(resolve_config(synthetic_config), tmp_path / "prepared")
    provenance = read_json(refs["run"])
    job = discovery_job(refs, provenance, "dictionary", 42)
    X, _, _ = load_partition(refs["train"])
    checked = validation_constraints(
        {"constraints": job["limits"], "split": {"seed": job["split_seed"]}},
        job["variables"], job["target"], X, domain=job["execution_domain"],
    )
    search_points = {tuple(row.values()) for row in job["limits"]["probes"]}
    check_points = {tuple(row.values()) for row in checked["probes"]}
    assert search_points == check_points
    assert search_points == {(0.5,)}
    assert job["limits"]["domain_probes"] == checked["domain_probes"]
    validity_points = {tuple(row.values()) for row in checked["domain_probes"]}
    assert (0.0,) in validity_points and (4.0,) in validity_points


@pytest.mark.parametrize("engine", ["flash_ansr"])
def test_pretrained_response_search_is_explicitly_unsupported(engine):
    from ml_platform_symbolic.engines.registry import get_engine
    from ml_platform_symbolic.engines.runner import engine_input_issues

    job = {"engine": engine, "operators": ["add"], "asset_manifest": "not_loaded.json",
           "limits": {"response_mode": "search", "probes": [{"x": 0}], "response_bounds": [0, 1]}}
    issues = engine_input_issues(job, get_engine(engine).capabilities())
    assert issues and issues[0]["status"] == "unsupported"
    assert "No filter-only fallback" in issues[0]["reason"]
    job["limits"]["response_mode"] = "filter"
    assert engine_input_issues(job, get_engine(engine).capabilities()) == []


def test_e2e_response_search_capability_is_coefficient_only():
    from ml_platform_symbolic.engines.registry import get_engine
    from ml_platform_symbolic.engines.runner import engine_input_issues

    capabilities = get_engine("e2e").capabilities()
    job = {"engine": "e2e", "operators": ["add"], "asset_manifest": "not_loaded.json",
           "limits": {"response_mode": "search", "probes": [{"x": 0}], "response_bounds": [0, 1]}}
    assert engine_input_issues(job, capabilities) == []
    assert "coefficient search" in capabilities["constraints"]["probes"]
    assert "decoder not response-conditioned" in capabilities["constraints"]["probes"]
    job["limits"]["response_mode"] = "filter"
    assert engine_input_issues(job, capabilities) == []


def test_finite_condition_pass_does_not_hide_violation_between_points():
    from ml_platform_symbolic.response_search import response_feasible

    r = request()
    r.target["domain"] = None
    r.limits = {"probes": [{"x": -1.0}, {"x": 1.0}], "response_bounds": [0, None]}
    declared = compile_response_constraints(r)

    def narrow_dip(points):
        return 1.0 - 2.0 * np.exp(-np.square(points[:, 0] / 0.01))

    assert response_feasible(narrow_dip(np.asarray(declared["points"])), declared)
    # An evaluation-only point detects an unobserved violation. It does not
    # retroactively make the declared finite checks a global guarantee.
    r.limits = {**r.limits, "probes": [{"x": 0.0}]}
    evaluation = compile_response_constraints(r)
    assert not response_feasible(narrow_dip(np.asarray(evaluation["points"])), evaluation)


def test_derivative_stencil_outside_declared_domain_is_rejected():
    r = request()
    r.variables[0]["domain"] = [0, 2]
    r.limits["response_checks"][0]["points"] = [{"x": 0}]
    with pytest.raises(ValueError):
        compile_response_constraints(r)


def test_rasr_native_response_refinement_supported_but_template_is_not():
    from ml_platform_symbolic.engines.rasr import capabilities, validate_request
    from ml_platform_symbolic.engines.runner import engine_input_issues

    job = {"engine": "rasr", "operators": ["add"], "asset_manifest": "not_loaded.json",
           "limits": {"response_mode": "search", "probes": [{"x": 0}], "response_bounds": [0, 1]}}
    assert engine_input_issues(job, capabilities()) == []
    r = request()
    r.options = {}
    r.equation_template = None
    assert validate_request(r, None) is None
    for location in ("request", "limits"):
        r.equation_template = {"relation": "f(x)"} if location == "request" else None
        r.limits["equation_template"] = {"relation": "f(x)"} if location == "limits" else None
        with pytest.raises(ValueError, match="does not implement equation templates"):
            validate_request(r, None)


@pytest.mark.parametrize("target_domain", [None, [-2, 2]])
def test_explicit_response_scope_is_not_widened_by_domain_validity(target_domain):
    from ml_platform_symbolic.domains import validation_constraints
    from ml_platform_symbolic.engines.runner import engine_input_issues
    from ml_platform_symbolic.response_search import requested_response_constraints
    r = request()
    r.target["domain"] = target_domain
    explicit = {"probes": [{"x": 0}], "response_bounds": [-1, 1], "response_checks": []}
    r.limits = validation_constraints({"constraints": explicit, "split": {"seed": 17}},
                                     r.variables, r.target, r.X)
    assert r.limits["probes"] == explicit["probes"] == [{"x": 0}]
    assert r.limits["response_checks"] == []
    assert len(r.limits["domain_probes"]) > 1
    compiled = compile_response_constraints(r)
    points = np.asarray(compiled["points"])
    # x exceeds the RESPONSE bound elsewhere but is finite over the domain.
    checks = prediction_constraints(lambda x: x[:, 0], r.variables, r.target, r.limits)
    assert next(c for c in checks if c['name'] == 'probe_response')['status'] == 'pass'
    assert (response_penalty(points[:, 0], compiled) == 0) is (target_domain is None)
    assert next(c for c in checks if c['name'] == 'domain_prediction_validity')['status'] == 'pass'
    # Automatic finite probes still reject an invalid result, without imposing
    # an unknown response bound there. No point is discarded.
    def invalid(x):
        return np.where(abs(x[:, 0]) >= 5, np.nan, x[:, 0])
    assert np.isinf(response_penalty(invalid(points), compiled))
    assert next(c for c in prediction_constraints(invalid, r.variables, r.target, r.limits)
                if c['name'] == 'domain_prediction_validity')['status'] == 'fail'
    # Domain-only jobs do not become unsupported response-search requests.
    domain_only = {"domain_probes": r.limits['domain_probes'], "response_mode": "search"}
    assert requested_response_constraints(domain_only) == []
    assert engine_input_issues({'engine': 'test', 'operators': ['add'], 'limits': domain_only},
                               {'operators': ['add'], 'requires_assets': False}) == []


def test_box_sampling_is_data_free_deterministic_and_resolve_idempotent(synthetic_config):
    from copy import deepcopy

    from ml_platform_symbolic.config import resolve_config
    cfg = deepcopy(synthetic_config)
    cfg['data'].update(local_path='never-read.csv', units={'x': 'cm', 'y': 'm'}, domain={'x': [0, 100]})
    spec = {'domain': {'x': [0, 100]}, 'points': 8, 'corners': 2, 'seed': 17}
    cfg['constraints'] = {'probes': [{'x': 50}, {'x': 50}], 'probe_sampling': spec,
        'response_checks': [{'feature': 'x', 'order': 1, 'sign': 1, 'step': 10,
                             'tolerance': 0, 'sampling': spec, 'points': [{'x': 50}]}]}
    resolved = resolve_config(cfg)
    assert cfg['constraints']['probes'] == [{'x': 50}, {'x': 50}]
    assert resolved['constraints']['probes'][:2] == [{'x': .5}, {'x': .5}]
    assert len(resolved['constraints']['probes']) == 12
    check = resolved['constraints']['response_checks'][0]
    assert check['step'] == .1
    assert min(p['x'] for p in check['points']) == pytest.approx(.1)
    assert max(p['x'] for p in check['points']) == pytest.approx(.9)
    assert resolve_config(resolved) == resolved
    assert resolve_config(cfg) == resolved
    assert 'sampling' not in check and 'probe_sampling' not in resolved['constraints']
    assert resolved['_sampling_provenance'][0]['declaration'] == spec
    assert resolved['_sampling_provenance'][0]['coordinate_units']['x'] == 'cm'
    assert len(resolved['_sampling_provenance']) == 2


def test_sampling_keeps_feature_order_and_full_directional_stencils():
    from ml_platform_symbolic.response_sampling import materialize_response_sampling
    spec = {'domain': {'a': [-2, 2], 'b': [10, 20]}, 'points': 8, 'corners': 4, 'seed': 2}
    check = {'direction': {'a': -2, 'b': 1}, 'path_unit': 's', 'step': .25, 'sampling': spec}
    cfg = {'data': {'feature_columns': ['b', 'a']}, 'constraints': {'response_checks': [check]}}
    points = materialize_response_sampling(cfg)['constraints']['response_checks'][0]['points']
    assert len(points) == 12 and list(points[0]) == ['b', 'a']
    for p in points:
        for sign in (-1, 1):
            assert -2 <= p['a'] + sign * .25 * -2 <= 2
            assert 10 <= p['b'] + sign * .25 <= 20


@pytest.mark.parametrize('change', [{'points': 3}, {'points': 8192}, {'corners': 1},
    {'seed': -1}, {'domain': {'wrong': [0, 1]}}, {'domain': {'x': [0, .1]}}])
def test_sampling_rejects_invalid_budgets_and_empty_stencil_box(change):
    from ml_platform_symbolic.response_sampling import materialize_response_sampling
    spec = {'domain': {'x': [0, 1]}, 'points': 8, 'corners': 2, 'seed': 0, **change}
    cfg = {'data': {'feature_columns': ['x']}, 'constraints': {'response_checks': [
        {'feature': 'x', 'step': .1, 'sampling': spec}]}}
    with pytest.raises(ValueError):
        materialize_response_sampling(cfg)


def test_required_response_sampling_alias_resolves_to_same_points(synthetic_config):
    from copy import deepcopy

    from ml_platform_symbolic.config import resolve_config
    cfg = deepcopy(synthetic_config)
    spec = {'domain': {'x': [-1, 1]}, 'points': 4, 'corners': 2, 'seed': 4}
    cfg['constraints'] = {'probe_sampling': spec}
    direct = resolve_config(cfg)
    cfg['constraints'] = {}
    cfg['requirements'] = {'required_responses': {'sampling': spec}}
    assert resolve_config(cfg) == direct


def test_sampling_provenance_survives_in_memory_prepare(synthetic_config, tmp_path):
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import prepare, read_json
    from ml_platform_symbolic.problem import problem_config
    synthetic_config['constraints'] = {'probe_sampling': {
        'domain': {'x': [-1, 1]}, 'points': 4, 'corners': 2, 'seed': 8}}
    cfg = resolve_config(synthetic_config)
    refs = prepare(cfg, tmp_path/'sampled')
    stored = problem_config(read_json(refs['run']))
    assert stored['_sampling_provenance'] == cfg['_sampling_provenance']
    assert stored['constraints']['probes'] == cfg['constraints']['probes']
    assert resolve_config(stored) == cfg


def test_sampling_affine_temperature_coordinates_preserve_stencil(synthetic_config):
    from ml_platform_symbolic.config import resolve_config
    synthetic_config['data'].update(units={'x': 'degC', 'y': 'm'}, domain={'x': [0, 100]}, coordinate_roles={'x': 'value'})
    synthetic_config['constraints'] = {'response_checks': [{'feature': 'x', 'order': 1,
        'sign': 1, 'step': 10, 'tolerance': 0, 'sampling': {
            'domain': {'x': [0, 100]}, 'points': 4, 'corners': 2, 'seed': 1}}]}
    cfg = resolve_config(synthetic_config)
    check = cfg['constraints']['response_checks'][0]
    assert cfg['data']['units']['x'] == 'K'
    assert check['step'] == 10
    assert min(p['x'] for p in check['points']) == pytest.approx(283.15)
    assert max(p['x'] for p in check['points']) == pytest.approx(363.15)
    assert resolve_config(cfg) == cfg


@pytest.mark.parametrize('box,step', [([-1e308, 1e308], .1), ([0, .2], .1)])
def test_sampling_rejects_unrepresentable_stencil_or_overflow_box(box, step):
    from ml_platform_symbolic.response_sampling import materialize_response_sampling
    cfg = {'data': {'feature_columns': ['x']}, 'constraints': {'response_checks': [
        {'feature': 'x', 'step': step, 'sampling': {
            'domain': {'x': box}, 'points': 4, 'corners': 2, 'seed': 0}}]}}
    with pytest.raises(ValueError):
        materialize_response_sampling(cfg)


def test_sampling_rounds_generated_center_inward_without_changing_explicit_points():
    import math

    from ml_platform_symbolic.response_sampling import materialize_response_sampling
    explicit = {'x': -.9 + .3}
    cfg = {'data': {'feature_columns': ['x']}, 'constraints': {'response_checks': [
        {'feature': 'x', 'step': .3, 'points': [explicit], 'sampling': {
            'domain': {'x': [-.9, 2]}, 'points': 4, 'corners': 2, 'seed': 0}}]}}
    result = materialize_response_sampling(cfg)['constraints']['response_checks'][0]
    assert result['points'][0] == explicit and result['step'] == .3
    generated = result['points'][1:]
    assert len(generated) == 6
    assert min(p['x'] for p in generated) == math.nextafter(-.9 + .3, math.inf)
    assert all(p['x'] - .3 >= -.9 and p['x'] + .3 <= 2 for p in generated)


@pytest.mark.parametrize("engine", ["e2e", "flash_ansr", "rasr"])
@pytest.mark.parametrize("enabled", [False, True])
def test_constraint_application_reports_configured_decoder_mode(engine, enabled):
    from ml_platform_symbolic.response_search import constraint_application

    job = {"engine": engine, "limits": {}, "options": {"operator_mask": enabled}}
    before = deepcopy(job)
    result = constraint_application(job, {"constraints": {"operators": "optional decoder mask"}})
    assert f"operator_mask={enabled}" in result["operators"]
    assert ("runtime evidence required" if enabled else "masking disabled") in result["operators"]
    assert job == before


@pytest.mark.parametrize("engine", ["e2e", "flash_ansr", "rasr"])
def test_input_width_capability_is_not_variable_restriction(engine):
    from ml_platform_symbolic.response_search import constraint_application
    job = {"engine": engine, "limits": {"allowed_variables": ["x0"]}, "options": {}}
    result = constraint_application(job, {"constraints": {"variables": "accepts input width up to 17"}})
    assert result["allowed_variables"].startswith("post-filtered only")
    assert "17" not in result["allowed_variables"]


def test_explicit_variable_restriction_capability_is_used():
    from ml_platform_symbolic.response_search import constraint_application
    from ml_platform_symbolic.engines import dictionary, pysr
    for engine, module in [("dictionary", dictionary), ("pysr", pysr)]:
        result = constraint_application({"engine": engine, "limits": {"allowed_variables": ["x0"]}}, module.capabilities())
        assert result["allowed_variables"] == module.capabilities()["constraints"]["allowed_variables"]
