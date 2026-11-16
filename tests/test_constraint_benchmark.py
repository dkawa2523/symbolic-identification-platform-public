"""Boundary checks for the paired constraint experiment, not native performance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.problem import problem_config


def benchmark():
    path = Path(__file__).parents[1] / "benchmarks/constraints/run.py"
    spec = importlib.util.spec_from_file_location("constraint_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_inputs_are_identical_and_independent_checks_are_not_search_points(tmp_path):
    module = benchmark()
    root = tmp_path / "campaign"
    module.prepare_run(root, ["pysr", "parfam"], [17], 60)
    plan = read_json(root / "plan.json")
    prepared = {arm: read_json(refs["run"]) for arm, refs in plan["prepared"].items()}
    for partition in ("train", "selection"):
        assert len({p["partition_hashes"][partition] for p in prepared.values()}) == 1
    cfg = {arm: problem_config(p) for arm, p in prepared.items()}
    assert not cfg["none"]["constraints"]["probes"]
    assert cfg["filter"]["constraints"] == cfg["search"]["constraints"]
    train_points = {p["x"] for p in cfg["search"]["constraints"]["probes"]}
    final_points = {p["x"] for p in plan["independent_conditions"]["probes"]}
    assert train_points.isdisjoint(final_points)
    with np.load(plan["prepared"]["search"]["train"]) as data:
        assert data["X"].max() < 0.65
    assert "test.csv" not in str(prepared)


def test_existing_experiment_and_frozen_selection_are_not_overwritten(tmp_path):
    module = benchmark()
    root = tmp_path / "campaign"
    module.prepare_run(root, ["pysr"], [17], 60)
    original = (root / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.prepare_run(root, ["pysr"], [17], 60)
    assert (root / "plan.json").read_bytes() == original
    (root / "choices.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        module.select(root)


def test_fresh_holdout_stays_out_of_plan_and_known_relation_is_shared(tmp_path):
    module = benchmark()
    root = tmp_path / "fresh"
    module.prepare_run(root, ["pysr", "template_enumeration"], [17], 60,
                       test_seed=34091, known_relation=True)
    plan = read_json(root / "plan.json")
    assert "independent_conditions" not in plan
    assert plan["test_conditions_sha256"]
    assert read_json(root / "test_lock.json")["state"] == "generated_not_reviewed"
    assert not (root / "test_access.json").exists()
    configs = [problem_config(read_json(refs["run"])) for refs in plan["prepared"].values()]
    templates = [cfg["constraints"]["equation_template"] for cfg in configs]
    assert templates[0] == templates[1] == templates[2]
    assert all(cfg["search"]["engines"] == ["pysr", "template_enumeration"] for cfg in configs)
    assert "test_conditions" not in str(configs)


def test_partial_common_protocol_keeps_extrapolation_inside_predeclared_domain(tmp_path):
    import pandas as pd
    from ml_platform_symbolic.domains import region_mask
    path = Path(__file__).parents[1] / "benchmarks/partial_equations/current_campaign.py"
    spec = importlib.util.spec_from_file_location("partial_current", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "partial"
    module.prepare_cases(root, ["additive"], ["template_enumeration"], 30, 32)
    plan = read_json(root / "campaign.json")
    refs = plan["prepared"]["additive"]
    prepared = read_json(refs["run"])
    cfg = problem_config(prepared)
    assert "final" in refs and cfg["evaluation"]["mode"] == "holdout"
    assert plan["history"] == "known_exposed"
    frame = pd.read_csv(root / "additive/data.csv")
    split = pd.read_csv(root / "additive/split.csv")
    extra = frame[frame.region == "extrapolation"]
    assert len(extra) == 32 and (extra[["x", "z"]].to_numpy() > 1.).all()
    domain = {"kind": "box", "columns": ["x", "z"], "bounds": [[-1., 2.], [-1., 2.]]}
    assert region_mask(extra[["x", "z"]].to_numpy(), domain).all()
    assert not region_mask(extra[["x", "z"]].to_numpy(), {**domain, "bounds": [[-1., 1.], [-1., 1.]]}).any()
    with np.load(refs["train"]) as arrays:
        assert np.abs(arrays["X"]).max() <= 1.
    assert not set(extra.sample_id) & set(split[split.partition.isin(["train", "selection"])].sample_id)
    with pytest.raises(FileExistsError):
        module.prepare_cases(root, ["additive"], ["template_enumeration"], 30, 32)


def test_partial_common_flow_rejects_old_layout_and_changed_protocol(tmp_path):
    from ml_platform_symbolic.data import write_json
    path = Path(__file__).parents[1] / "benchmarks/partial_equations/current_campaign.py"
    spec = importlib.util.spec_from_file_location("partial_current_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = tmp_path / "old"
    old.mkdir()
    write_json(old / "campaign.json", {"cases": ["additive"], "attempts": 28})
    with pytest.raises(ValueError, match="Historical campaign"):
        module.execute(old, "score")
    root = tmp_path / "new"
    module.prepare_cases(root, ["additive"], ["template_enumeration"], 30, 32)
    write_json(root / "protocol.json", {"changed": True})
    with pytest.raises(ValueError, match="Protocol changed"):
        module.execute(root, "score")
    assert not list(root.rglob("final_access.json"))


def test_main_five_share_split_and_explicit_hard_intent(tmp_path):
    module=benchmark()
    settings=read_json(Path(__file__).parents[1]/'benchmarks/constraints/settings-main.json')
    root=tmp_path/'main'
    module.prepare_run(root,list(settings['methods']),[17],600,test_seed=17001,settings=settings)
    plan=read_json(root/'plan.json')
    assert set(plan['prepared']) == {'none','search'}
    none,search=[read_json(plan['prepared'][a]['run']) for a in ['none','search']]
    assert none['partition_hashes']==search['partition_hashes']
    cfg=problem_config(search)
    assert cfg['constraints']['response_enforcement']=='hard'
    assert cfg['constraints']['max_nodes']==64
    assert set(cfg['search']['engines'])=={'pysr','parfam','torchsisso','e2e','physo'}
    assert 'independent_conditions' not in plan
    assert not (root/'test_access.json').exists()

def test_scoring_shared_study_keeps_each_method_predictions(tmp_path, monkeypatch):
    import pandas as pd
    from ml_platform_symbolic.data import write_json, digest
    module = benchmark()
    root = tmp_path / 'paired'
    module.prepare_run(root, ['pysr', 'parfam'], [17], 30, test_seed=17)
    plan = read_json(root / 'plan.json')
    provenance = read_json(plan['prepared']['none']['run'])
    study = root / 'shared'; study.mkdir()
    write_json(study / 'study.json', {})
    choices = [{'arm':'none','engine':engine,'seed':17,'selected_id':str(i),
                'study':str(study),'study_sha256':digest(study/'study.json')}
               for i,engine in enumerate(['pysr','parfam'])]
    write_json(root/'choices.json', {'plan_sha256':digest(root/'plan.json'),'choices':choices})
    monkeypatch.setattr(module, 'load_study', lambda _: (provenance,[{'id':'0'},{'id':'1'}],[],{}))
    monkeypatch.setattr(module, 'predict_record', lambda r,X,*_: (np.full(len(X),1+int(r['id'])),np.ones(len(X),dtype=bool)))
    module.score(root)
    rows=read_json(root/'scores.json')['attempts']
    paths=[Path(row['predictions_file']) for row in rows]
    assert len(set(paths)) == 2
    assert [pd.read_csv(p).predicted.iloc[0] for p in paths] == [1,2]
    with pytest.raises(FileExistsError): module.score(root)
