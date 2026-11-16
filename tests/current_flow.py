"""Explicit test stages. No automatic Final, selection, or legacy result facade."""
from pathlib import Path

from ml_platform_symbolic import compare_equations, create_study, run_task, select_representatives
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.equation_evaluation import load_study


def prepare_discover(config, root):
    root = Path(root)
    prepared = run_task({**config, 'task': 'symbolic_stage', 'run': {'stage': 'prepare'}}, run_dir=root / 'prepare')
    refs = prepared.extra
    search = read_json(refs['run'])['config']['search']
    discoveries = []
    for engine in search['engines']:
        for seed in search['seeds']:
            result = run_task({'task': 'symbolic_stage', 'run': {'stage': 'discover', 'engine': engine, 'seed': seed},
                               'stage_inputs': {'prepare': refs}}, run_dir=root / 'discover' / engine / str(seed))
            discoveries.append(result.extra)
    return refs, discoveries


def make_study(refs, discoveries, directory, **plan):
    return create_study({'prepared_run': refs['run'], 'sources': [{'discovery': str(d['summary'])} for d in discoveries],
                         **plan}, run_dir=directory)


def study_flow(config, root, **plan):
    """Stop after Train/Selection; callers explicitly choose any subsequent step."""
    refs, discoveries = prepare_discover(config, root)
    study = make_study(refs, discoveries, Path(root) / 'study', **plan)
    return refs, discoveries, study


def explicit_choice(study, root):
    """Explicit test-only adoption based on supplied study's predeclared tolerance."""
    from ml_platform_symbolic import choose_hypothesis
    _, records, rows, _ = load_study(study)
    representatives = select_representatives(records, rows)
    if not representatives:
        raise ValueError('No eligible Selection representatives')
    selected = min(representatives.values(), key=lambda item: (item[1]['metrics']['selection']['rmse'], item[0]['id']))[0]
    comparison = compare_equations(study, run_dir=Path(root) / 'comparison', pair_ids=[])
    decision = choose_hypothesis(comparison, run_dir=Path(root) / 'decision', candidate_id=selected['id'],
                                 reason='Explicit test choice under predeclared local prediction tolerance')
    return comparison, decision
