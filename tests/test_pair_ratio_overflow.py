import json
import numpy as np
from ml_platform_symbolic import create_study, compare_equations
from ml_platform_symbolic.data import read_json
from ml_platform_symbolic.report_data import build_report_view
from ml_platform_symbolic.equation_evaluation import load_study


def test_finite_prediction_ratio_overflow_retains_rms_and_other_pairs(synthetic_config, tmp_path):
    study = create_study({
        'problem': synthetic_config, 'engineering_tolerance': .1,
        'sources': [{'manual': {'id': name, 'expression': expr}} for name, expr in
                    [('good', '3*x'), ('other', '3*x+1'), ('huge', '1e308')]],
    }, run_dir=tmp_path/'study')
    _, records, _, _ = load_study(study)
    huge = next(r['id'] for r in records if r['id'].startswith('huge'))
    comparison = compare_equations(study, run_dir=tmp_path/'comparison', pair_ids=[r['id'] for r in records])
    data = read_json(comparison/'comparison.json')
    pairs = data['pairs']
    assert len(pairs) == 3
    overflow = [p for p in pairs if huge in (p['left'], p['right'])]
    assert len(overflow) == 2
    for pair in overflow:
        assert np.isfinite(pair['prediction_rms_difference'])
        assert pair['prediction_rms_difference'] == 1e308
        assert pair['engineering_scale_ratio'] is None
        assert 'overflow' in pair['engineering_scale_ratio_reason']
    normal = next(p for p in pairs if p not in overflow)
    assert normal['engineering_scale_ratio'] == 10
    assert normal['engineering_scale_ratio_reason'] is None
    view = build_report_view(comparison)
    assert len(view['pairs']) == 3
    json.dumps(view, allow_nan=False)
    assert sum(bool(p['engineering_scale_ratio_reason']) for p in view['pairs']) == 2
