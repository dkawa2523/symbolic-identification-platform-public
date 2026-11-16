"""Native objective integration, not only string generation or post-selection."""

from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_symbolic.engines.pysr_objective import native_response_objective


def request(*, mode="search", template=None):
    return SimpleNamespace(
        X=np.array([[-1.0], [0.0], [1.0]]), y=np.array([1.0, 0.0, -1.0]),
        variables=[{"id": "x0", "column": "x", "unit": "1", "domain": [-2, 2]}],
        target={"unit": "1"}, options={}, equation_template=template,
        limits={"response_mode": mode, "probes": [{"x": -1.0}, {"x": 1.0}],
                "response_bounds": [0.0, None], "response_checks": []},
    )


def test_pysr_filter_is_explicit_and_template_receives_full_expression_objective():
    options, evidence = native_response_objective(request(mode="filter"))
    assert options == {} and evidence["mode"] == "filter"
    options, evidence = native_response_objective(request(template={"slots": {}}))
    assert "loss_function_expression" in options and evidence["search_response_constraints"]


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True])
def test_pysr_constraint_weight_is_not_silently_ignored(weight):
    req = request()
    req.options["constraint_weight"] = weight
    with pytest.raises(ValueError, match="constraint_weight"):
        native_response_objective(req)


@pytest.mark.native
@pytest.mark.parametrize("order", [None, 1, 2])
def test_julia_objective_changes_search_score_for_constraint_violation(order):
    import pysr  # Initialize Julia through PySR before accessing juliacall.
    from juliacall import Main as jl

    assert pysr.__version__ == "2.7.2"
    req = request()
    if order is not None:
        req.limits["response_checks"] = [{"feature": "x", "order": order, "sign": 1,
                                           "step": 0.1, "tolerance": 0.0, "points": [{"x": 0.0}]}]
    source, evidence = native_response_objective(req)
    assert evidence["search_response_constraints"]
    jl.seval("using SymbolicRegression")
    jl.seval(source["loss_function"])
    values = jl.seval('''begin
        opts = Options(binary_operators=[+, *], unary_operators=[-])
        data = Dataset(reshape([-1.0, 0.0, 1.0], 1, :), [1.0, 0.0, -1.0])
        x = Node{Float64}(feature=1)
        decreasing = Node{Float64}(op=1, l=x)
        zero_expr = Node{Float64}(val=0.0)
        (constrained_loss(decreasing, data, opts), constrained_loss(zero_expr, data, opts))
    end''')
    from ml_platform_symbolic.response_search import response_penalty

    compiled = {"points": evidence["points"], "terms": evidence["terms"]}
    expected_decreasing = response_penalty(-np.asarray(evidence["points"])[:, 0], compiled)
    expected_zero = np.mean((req.y / evidence["data_scale"])**2)
    np.testing.assert_allclose(values, [expected_decreasing, expected_zero])
    assert values[0] > 0  # Perfect data fit still pays the requested response penalty.
    assert np.isfinite(values).all()

@pytest.mark.native
@pytest.mark.parametrize("with_template", [False, True])
def test_pysr_native_fit_uses_response_objective_and_exports_same_prediction(tmp_path, with_template):
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.pysr import discover

    original = request()
    req = DiscoveryRequest(
        X=original.X, y=original.y, row_ids=np.array(["a", "b", "c"]),
        variables=original.variables, target=original.target, operators=["add", "mul", "neg"],
        limits={**original.limits, "max_nodes": 8, "max_wall_seconds": 60}, seed=17,
        options={"seconds": 20, "niterations": 5, "populations": 2, "population_size": 40,
                 "ncycles_per_iteration": 50, "constraint_weight": 10},
    )
    if with_template:
        from ml_platform_symbolic.equation_templates import validate_template

        req.equation_template = {
            "coordinate_space": "working",
            "knowledge": {"kind": "requirement", "statement": "Known additive offset with unknown function"},
            "ast": {"op": "add", "args": [{"slot": "f", "args": [{"var": "x0"}]}, {"const": 1.0}]},
            "slots": {"f": {"arguments": [{"name": "u", "unit": "1"}],
                              "unit": "1", "allowed_operators": req.operators, "max_nodes": 1}},
            "parameters": {},
        }
        req.equation_template = validate_template(req.equation_template, req.variables, req.target, req.operators, {})
        req.limits["equation_template"] = req.equation_template
    result = discover(req, DiscoveryContext(work_dir=tmp_path))
    assert result.summary["response_search"]["search_response_constraints"]
    assert result.candidates
    assert all(candidate.native_check["passed"] for candidate in result.candidates)
    if with_template:
        assert not result.summary["rejected_exports"]
        assert result.summary["response_search"]["slot_max_nodes"] == {"f": 1}


@pytest.mark.native
@pytest.mark.parametrize("mode", ["search", "filter"])
def test_pysr_explicit_mse_preserves_scale_in_control_conditions(mode):
    import pysr
    from juliacall import Main as jl

    assert pysr.__version__ == "2.7.2"
    req = request(mode=mode)
    req.options["objective"] = "mse"
    # Search without any constraints is the none condition.
    if mode == "search":
        req.limits.update(probes=[], response_checks=[])
    options, evidence = native_response_objective(req)
    assert evidence["objective"] == "train_scaled_mse"
    assert not evidence["search_response_constraints"]
    jl.seval("using SymbolicRegression")
    jl.seval(options["loss_function"])
    value = jl.seval("""begin
        opts = Options(binary_operators=[+, *], unary_operators=[-])
        data = Dataset(reshape([-1.0, 0.0, 1.0], 1, :), [1.0, 0.0, -1.0])
        constrained_loss(Node{Float64}(val=0.0), data, opts)
    end""")
    np.testing.assert_allclose(value, np.mean((req.y / evidence["data_scale"])**2))


@pytest.mark.native
@pytest.mark.parametrize("maximum", [2, 3])
def test_native_template_slot_gate_counts_power_literal_and_preserves_default_mse(maximum):
    import pysr
    from juliacall import Main as jl

    assert pysr.__version__ == "2.7.2"
    from ml_platform_symbolic.equation_templates import validate_template
    req = request(mode="filter")
    req.equation_template = validate_template({
        "coordinate_space": "working", "knowledge": {"kind": "requirement", "statement": "Known additive offset"},
        "ast": {"op": "add", "args": [{"slot": "f", "args": [{"var": "x0"}]}, {"const": 1.0}]},
        "slots": {"f": {"arguments": [{"name": "u", "unit": "1"}], "unit": "1",
                          "allowed_operators": ["add", "pow_int"], "max_nodes": maximum}}, "parameters": {}
    }, req.variables, req.target, ["add", "pow_int"], {})
    options, evidence = native_response_objective(req)
    assert evidence["objective"] == "native_mse"
    assert not evidence["search_response_constraints"]
    jl.seval("using SymbolicRegression")
    jl.seval(options["loss_function_expression"])
    losses = jl.seval("""begin
        opts = Options(binary_operators=[+], unary_operators=[square])
        structure = TemplateStructure{(:slot0,)}(((; slot0), (x,)) -> slot0(x) + 1.0;
                                                num_features=(slot0=1,))
        x = Node{Float64}(feature=1)
        squared = Node{Float64}(op=1, l=x)
        expressions = map((x, squared)) do node
            inner = ComposableExpression(node; operators=opts.operators, variable_names=["u"])
            TemplateExpression((slot0=inner,); structure, operators=opts.operators, variable_names=["x"])
        end
        data = Dataset(reshape([-1.0, 0.0, 1.0], 1, :), [1.0, 0.0, -1.0])
        map(ex -> constrained_loss(ex, data, opts), expressions)
    end""")
    assert losses[0] == pytest.approx(11 / 3)
    if maximum == 2:
        assert np.isinf(losses[1])  # Native two-node square exports three AST nodes.
    else:
        assert losses[1] == pytest.approx(11 / 3)


@pytest.mark.parametrize("options,expected", [
    ({}, {"parallelism": "serial", "deterministic": True}),
    ({"parallelism": "multithreading"}, {"parallelism": "multithreading", "deterministic": False}),
    ({"parallelism": "multiprocessing", "procs": 2},
     {"parallelism": "multiprocessing", "deterministic": False, "procs": 2}),
])
def test_explicit_pysr_execution_options(options, expected):
    from ml_platform_symbolic.engines.pysr import execution_options
    assert execution_options(options) == expected


@pytest.mark.parametrize("options", [
    {"parallelism": []}, {"parallelism": {}},
    {"parallelism": "multithreading", "deterministic": True}, {"procs": 2},
    {"parallelism": "multiprocessing", "procs": True}, {"parallelism": "multiprocessing", "procs": 0},
])
def test_incompatible_pysr_execution_options_are_rejected(options):
    from ml_platform_symbolic.engines.pysr import execution_options
    with pytest.raises(ValueError):
        execution_options(options)


@pytest.mark.parametrize("allowed,indices", [(None, [0, 1, 2]), (["x2"], [2]), (["x2", "x0"], [0, 2])])
def test_pysr_native_input_subset_keeps_original_order_and_ids(allowed, indices):
    from ml_platform_symbolic.engines.pysr import native_feature_indices
    r = SimpleNamespace(variables=[{"id": f"x{i}"} for i in range(3)],
                        limits={"allowed_variables": allowed}, equation_template=None)
    assert native_feature_indices(r) == indices
    assert [v["id"] for v in r.variables] == ["x0", "x1", "x2"]


@pytest.mark.parametrize("limits,template", [
    ({"allowed_variables": []}, None),
    ({"allowed_variables": ["x2"]}, {"fixed": "template"}),
])
def test_pysr_variable_subset_blocks_unsupported_combinations(limits, template):
    from ml_platform_symbolic.engines.pysr import native_feature_indices
    r = SimpleNamespace(variables=[{"id": f"x{i}"} for i in range(3)],
                        limits=limits, equation_template=template)
    with pytest.raises(NotImplementedError):
        native_feature_indices(r)


def test_variable_subset_projects_only_native_points_not_response_meaning():
    from ml_platform_symbolic.response_search import compile_response_constraints, response_feasible
    req = request()
    req.X = np.array([[-1., 10.], [0., 20.], [1., 30.]])
    req.variables = [{"id": "x0", "column": "unused", "unit": "1", "domain": [-2, 2]},
                     {"id": "x2", "column": "kept", "unit": "1", "domain": [0, 40]}]
    req.limits.update(allowed_variables=["x2"], probes=[], response_bounds=[None, None],
        response_checks=[{"feature": "unused", "order": 1, "sign": 1, "step": .1,
                          "tolerance": 0., "points": [{"unused": 0., "kept": 20.}]}])
    compiled = compile_response_constraints(req)
    options, evidence = native_response_objective(req)
    assert evidence["points"] == compiled["points"]
    assert evidence["terms"] == compiled["terms"]
    assert evidence["native_variable_ids"] == ["x2"]
    assert evidence["native_points"] == [[20.], [20.], [20.]]
    assert 'reshape(T[20.0, 20.0, 20.0], 1, 3)' in options["loss_function"]
    assert response_feasible(np.array([20., 20., 20.]), compiled)  # Nondecreasing, not strictly increasing.
    req.limits["response_checks"][0]["points"][0]["unused"] = 2.
    with pytest.raises(ValueError):  # Projection must not conceal an invalid original stencil.
        native_response_objective(req)


def test_subset_response_rejects_full_space_coupled_domain_violation():
    req = request()
    req.X = np.array([[-1., 10.], [0., 20.], [1., 30.]])
    req.variables = [{"id": "x0", "column": "unused", "unit": "1", "domain": [-2, 2]},
                     {"id": "x2", "column": "kept", "unit": "1", "domain": [0, 40]}]
    req.limits.update(allowed_variables=["x2"], probes=[{"unused": 1., "kept": 20.}])
    req.domain = {"kind": "affine", "columns": ["unused", "kept"],
                  "equalities": {"matrix": [[1., 1.]], "rhs": [20.]}}
    with pytest.raises(ValueError, match="original declared input domain"):
        native_response_objective(req)
    req.limits["probes"][0]["unused"] = 0.
    _, evidence = native_response_objective(req)
    assert evidence["points"] == [[0., 20.]] and evidence["native_points"] == [[20.]]

def test_source_weighted_native_size_is_not_postfilter_only():
    from ml_platform_symbolic.engines.pysr import source_search_budget
    req=request(mode='filter');req.target={'column':'y','unit':'m'}
    req.variables=[{'id':'x0','column':'x','unit':'m'}]
    req.limits['max_nodes']=14
    ctx=SimpleNamespace(provenance={'unit_coordinates':{'columns':{c:{'source_unit':'cm','problem_unit':'m','scale':.01,'offset':0} for c in ['x','y']}}})
    schema,native_size,costs,overhead=source_search_budget(req,ctx)
    assert native_size==12 and costs==[3] and overhead==2
    req.limits['max_nodes']=8
    with pytest.raises(ValueError,match='at least 7'):source_search_budget(req,ctx)
    req.limits['max_nodes']=14
    req.equation_template={'slots':{}}
    assert source_search_budget(req,ctx)[1]==12


def test_template_source_size_counts_shared_calls_and_composite_arguments():
    from ml_platform_symbolic.engines.pysr_objective import template_size_expression
    call={'slot':'f','args':[{'op':'add','args':[{'var':'x0'},{'var':'x0'}]}]}
    t={'slots':{'f':{}},'ast':{'op':'add','args':[call,call]},'parameters':{}}
    text=template_size_expression(t,{'x0':3})
    assert text.count('get_contents(tree).slot0')==2
    assert text.count('[(1 + 3 + 3)]')==2


def test_common_response_enforcement_preserves_defaults_and_rejects_downgrade():
    from ml_platform_symbolic.engines import pysr, parfam, e2e
    from ml_platform_symbolic.response_search import response_enforcement, constraint_application
    from ml_platform_symbolic.engines.runner import engine_input_issues
    req = request()
    for name, engine, default in [("pysr",pysr,"soft"),("parfam",parfam,"hard"),("e2e",e2e,"hard")]:
        cap=engine.capabilities()
        assert response_enforcement(req.limits,cap)["effective"]==default
        job={"engine":name,"limits":{**req.limits,"response_enforcement":"soft"},"operators":["add"],"asset_manifest":"weights.json" if name=="e2e" else None}
        issues=engine_input_issues(job,cap)
        if name=="pysr": assert not issues
        else:
            assert any("no conversion" in issue["reason"] for issue in issues)
            assert constraint_application(job,cap)["response_enforcement"]["effective"]=="blocked"
    req.limits.update(response_mode="filter",response_enforcement="hard")
    with pytest.raises(ValueError,match="filter"):native_response_objective(req)


def test_hard_response_uses_same_terms_without_hidden_tolerance():
    req=request();soft,original=native_response_objective(req)
    req.limits["response_enforcement"]="hard"
    hard,evidence=native_response_objective(req)
    assert evidence["terms"]==original["terms"]
    assert evidence["response_enforcement"]["effective"]=="hard"
    assert evidence["weight"] is None
    assert "response < T(0.0) && return L(Inf)" in hard["loss_function"]
    assert "penalty +=" not in hard["loss_function"]
    req.options["constraint_weight"]=2
    with pytest.raises(ValueError,match="only to soft"):native_response_objective(req)
    req.options={};req.limits["response_bounds"]=[2,1]
    with pytest.raises(ValueError,match="Contradictory"):native_response_objective(req)


def test_common_enforcement_config_roundtrip_and_filter_conflict():
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.problem import problem_document, problem_config
    cfg={"data":{"feature_columns":["x"],"target_column":"y"},"split":{"file":"split.csv"},"requirements":{"required_responses":{"enforcement":"hard"}}}
    resolved=resolve_config(cfg)
    assert problem_config(problem_document(resolved))["constraints"]["response_enforcement"]=="hard"
    assert problem_document(resolved)["requirements"]["required_responses"]["enforcement"]=="hard"
    cfg["search"]={"response_mode":"filter"}
    with pytest.raises(ValueError,match="requires response_mode"):resolve_config(cfg)


@pytest.mark.native
def test_julia_hard_gate_vs_soft_and_explicit_interval_tolerance():
    import pysr
    from juliacall import Main as jl
    jl.seval("using SymbolicRegression")
    req=request();values={}
    for mode in ("soft","hard"):
        req.limits["response_enforcement"]=mode
        source,evidence=native_response_objective(req);jl.seval(source["loss_function"])
        values[mode]=jl.seval("begin; opts=Options(binary_operators=[+, *], unary_operators=[-]); data=Dataset(reshape([-1.,0.,1.],1,:),[1.,0.,-1.]); x=Node{Float64}(feature=1); neg=Node{Float64}(op=1,l=x); zero_expr=Node{Float64}(val=0.); (constrained_loss(neg,data,opts),constrained_loss(zero_expr,data,opts)); end")
    assert np.isfinite(values['soft']).all()
    assert np.isinf(values['hard'][0]) and np.isfinite(values['hard'][1])
    for bounds,allowed in [([1.,1.],False),([.999999,1.000001],True)]:
        req.limits.update(probes=[{'x':0.}],response_bounds=bounds)
        source,_=native_response_objective(req);jl.seval(source['loss_function'])
        value=jl.seval('constrained_loss(Node{Float64}(val=1.00000001),data,opts)')
        assert bool(np.isfinite(value))==allowed


@pytest.mark.parametrize("include_valid", [False,True])
def test_nonfinite_native_hall_rows_are_recorded_without_losing_valid_candidate(tmp_path,monkeypatch,include_valid):
    import sys,json,pandas as pd
    from ml_platform_symbolic.engines import pysr as adapter
    from ml_platform_symbolic.engines.base import DiscoveryRequest,DiscoveryContext
    seen=[]
    class FakeNative:
        def __init__(self,**kwargs):pass
        def fit(self,X,y,**kwargs):
            losses=[float('inf'),float('-inf'),float('nan')]+([0.] if include_valid else [])
            self.equations_=pd.DataFrame({'equation':['x0']*len(losses),'loss':losses,'complexity':[1]*len(losses)})
        def predict(self,X,index):
            assert index==3;seen.append(index);return X[:,0]
    monkeypatch.setitem(sys.modules,'pysr',SimpleNamespace(PySRRegressor=FakeNative))
    monkeypatch.setattr(adapter,'version',lambda _: 'test-double')
    r=request(mode='filter')
    req=DiscoveryRequest(r.X,r.y,np.arange(3),r.variables,{'column':'y','unit':'1'},['add'],{**r.limits,'max_nodes':8,'max_wall_seconds':60},17,{})
    result=adapter.discover(req,DiscoveryContext(tmp_path))
    assert len(result.candidates)==int(include_valid)
    assert result.status==('ok' if include_valid else 'failed')
    assert len(result.summary['rejected_exports'])==3
    rows=json.loads((tmp_path/'pysr_native.json').read_text())['equations']
    assert [r['loss_status'] for r in rows[:3]]==['inf','-inf','nan']
    assert all(r['loss'] is None for r in rows[:3])
    assert bool(seen)==include_valid


def test_enforcement_single_authority_and_legacy_problem_hash():
    from copy import deepcopy
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.problem import problem_document,problem_config,requirement_identity
    cfg={'data':{'feature_columns':['x'],'target_column':'y'},'split':{'file':'split.csv'},'requirements':{'required_responses':{'enforcement':'hard'}}}
    duplicated=deepcopy(cfg);duplicated['constraints']={'response_enforcement':'hard'}
    with pytest.raises(ValueError,match='declared twice'):resolve_config(duplicated)
    misplaced=deepcopy(cfg);misplaced['search']={'response_enforcement':'hard'}
    with pytest.raises(ValueError,match='Unknown search'):resolve_config(misplaced)
    document=problem_document(resolve_config(cfg))
    before=document['requirements_sha256']
    document['requirements']['required_responses'].pop('enforcement')
    old_hash=requirement_identity(document['requirements'])
    document['requirements_sha256']=old_hash;document['config']['_prepared_requirements_ref']=old_hash
    assert problem_config(document)['constraints']['response_enforcement'] is None
    assert document['requirements_sha256']==old_hash and old_hash!=before


def test_native_unit_profile_is_explicit_and_preserves_column_mapping():
    from ml_platform_symbolic.engines.pysr import unit_profile
    req = request()
    assert unit_profile(req) == ({}, {}, {"mode": "post_check_only", "native_units": False})
    req.options = {"unit_mode": "hard", "dimensionless_constants_only": True}
    req.variables = [{"id": "x0", "unit": "m"}, {"id": "x1", "unit": "s"}]
    req.target = {"unit": "m/s"}
    req.limits["allowed_variables"] = ["x1"]
    model, fit, info = unit_profile(req)
    assert model["dimensional_constraint_penalty"] == float("inf")
    assert fit == {"X_units": ["s^(1/1)"], "y_units": "m^(1/1) * s^(-1/1)"}
    assert info["native_variable_ids"] == ["x1"]
    req.variables[1]["unit"] = None
    with pytest.raises(ValueError, match="known canonical"):
        unit_profile(req)
    req.variables[1]["unit"] = "cm"
    with pytest.raises(ValueError, match="not canonical"):
        unit_profile(req)


@pytest.mark.parametrize("options", [
    {"unit_mode": []}, {"unit_mode": "strict"},
    {"dimensionless_constants_only": False},
    {"unit_mode": "hard", "dimensional_constraint_penalty": 1},
    {"unit_mode": "soft", "dimensional_constraint_penalty": float("inf")},
    {"unit_mode": "soft", "dimensional_constraint_penalty": 0},
    {"unit_mode": "soft", "dimensionless_constants_only": 1},
])
def test_native_unit_profile_rejects_ambiguous_options(options):
    from ml_platform_symbolic.engines.pysr import unit_profile
    req = request()
    req.options = options
    with pytest.raises(ValueError):
        unit_profile(req)


def test_native_unit_template_preserves_legacy_but_rejects_requested_search():
    from ml_platform_symbolic.engines.pysr import unit_profile
    req = request(template={"slots": {}})
    assert not unit_profile(req)[2]["native_units"]
    req.options["unit_mode"] = "hard"
    with pytest.raises(NotImplementedError, match="templates"):
        unit_profile(req)


def test_native_unit_custom_loss_and_application_distinguish_modes():
    from ml_platform_symbolic.engines.pysr import capabilities
    from ml_platform_symbolic.response_search import constraint_application
    req = request()
    for mode in ("soft", "hard"):
        req.options = {"unit_mode": mode}
        compiled, evidence = native_response_objective(req)
        assert "violates_dimensional_constraints" in compiled["loss_function"]
        assert ("unit_violation && return L(Inf)" in compiled["loss_function"]) == (mode == "hard")
        assert evidence["unit_search"]["mode"] == mode
        display = constraint_application({"engine": "pysr", "limits": req.limits, "options": req.options}, capabilities())
        assert f"unit_mode={mode}" in display["units"]
    req.options = {}
    assert "violates_dimensional_constraints" not in native_response_objective(req)[0]["loss_function"]


@pytest.mark.native
@pytest.mark.parametrize("unit_mode", ["soft", "hard"])
@pytest.mark.parametrize("custom", [False, True])
def test_native_unit_loss_dispatch_rejects_or_penalizes_same_numeric_bad_dimension(unit_mode, custom):
    import pysr
    from juliacall import Main as jl
    from ml_platform_symbolic.engines.pysr import unit_profile
    req = request()
    req.X = np.array([[1., 1.], [2., 2.], [3., 3.]])
    req.y = np.array([1., 2., 3.])
    req.variables = [{"id": "x0", "column": "x", "unit": "m", "domain": [0, 4]},
                     {"id": "x1", "column": "t", "unit": "s", "domain": [0, 4]}]
    req.target = {"unit": "m"}
    req.limits = {"response_mode": "search", "probes": [], "response_checks": []}
    req.options = {"unit_mode": unit_mode, "dimensionless_constants_only": True}
    if unit_mode == "soft":
        req.options["dimensional_constraint_penalty"] = 7.
    if custom:
        req.limits.update(probes=[{"x": 1., "t": 1.}], response_bounds=[0., None], response_enforcement="hard")
    model, fit, _ = unit_profile(req)
    compiled, _ = native_response_objective(req)
    jl.seval("using SymbolicRegression")
    penalty = "Inf" if unit_mode == "hard" else "7.0"
    options = f"dimensional_constraint_penalty={penalty}, dimensionless_constants_only=true"
    if custom:
        jl.seval(compiled["loss_function"])
        options += ", loss_function=constrained_loss"
    else:
        assert compiled == {}
    values = list(jl.seval("""begin
        data=Dataset([1. 2. 3.;1. 2. 3.],[1.,2.,3.]; X_units=["m^(1/1)","s^(1/1)"],y_units="m^(1/1)")
        opts=Options(binary_operators=[+,*],OPTIONS)
        [SymbolicRegression.LossFunctionsModule.eval_loss(Node{Float64}(feature=i),data,opts) for i in 1:2]
    end""".replace("OPTIONS", options)))
    assert values == [0., model["dimensional_constraint_penalty"]]


def test_new_common_responses_freeze_one_enforcement_for_all_main_engines():
    from copy import deepcopy
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.problem import problem_document, problem_config
    from ml_platform_symbolic.engines.registry import get_engine
    from ml_platform_symbolic.response_search import response_enforcement
    cfg = {"data": {"feature_columns": ["x"], "target_column": "y"},
           "split": {"file": "split.csv"},
           "requirements": {"required_responses": {"points": [{"x": 1}], "bounds": [0, None]}}}
    original = deepcopy(cfg)
    doc = problem_document(resolve_config(cfg))
    limits = {**problem_config(doc)["constraints"], "response_mode": "search"}
    assert cfg == original
    assert doc["requirements"]["required_responses"]["enforcement"] == "hard"
    for name in ("pysr", "parfam", "torchsisso", "e2e", "physo"):
        assert response_enforcement(limits, get_engine(name).capabilities())["effective"] == "hard"
    # Explicit soft remains soft, and unsupported engines must reject it.
    cfg["requirements"]["required_responses"]["enforcement"] = "soft"
    limits = {**resolve_config(cfg)["constraints"], "response_mode": "search"}
    assert response_enforcement(limits, get_engine("pysr").capabilities())["effective"] == "soft"
    with pytest.raises(NotImplementedError, match="soft"):
        response_enforcement(limits, get_engine("e2e").capabilities())
    # Explicit post-filter control and legacy input retain their old meanings.
    cfg = deepcopy(original); cfg["search"] = {"response_mode": "filter"}
    assert resolve_config(cfg)["constraints"]["response_enforcement"] is None
    cfg = deepcopy(original); responses = cfg.pop("requirements")["required_responses"]
    cfg["constraints"] = {"probes": responses["points"], "response_bounds": responses["bounds"]}
    assert resolve_config(cfg)["constraints"]["response_enforcement"] is None


def test_public_response_default_reaches_prepared_pysr_objective(tmp_path, synthetic_config):
    from ml_platform_symbolic.config import resolve_config
    from ml_platform_symbolic.data import prepare, read_json
    from ml_platform_symbolic.engines.runner import discovery_job, discovery_request
    cfg = dict(synthetic_config)
    cfg.pop("constraints")
    cfg["search"] = {"engines": ["pysr"], "seeds": [17]}
    cfg["requirements"] = {"required_responses": {"points": [{"x": 1.0}], "bounds": [0, None]}}
    refs = prepare(resolve_config(cfg), tmp_path / "prepared")
    job = discovery_job(refs, read_json(refs["run"]), "pysr", 17)
    request = discovery_request(job)
    objective, evidence = native_response_objective(request)
    assert request.limits["response_enforcement"] == "hard"
    assert evidence["response_enforcement"]["effective"] == "hard"
    assert "return L(Inf)" in objective["loss_function"]
    assert "penalty +=" not in objective["loss_function"]
