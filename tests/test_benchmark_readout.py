"""The readout renders preserved evidence, without reading raw data or running a model."""

import json
from copy import deepcopy

import pytest

from scripts.benchmark_readout import render, scatter, write_report


def test_html_retains_formulas_failures_partitions_and_exact_source(tmp_path):
    evidence = {"attempts": 3, "results": [
        {"case": "unknown_case", "engine": "method", "equation": "y = a*x + b", "data_seed": 10,
         "solver_seed": 20, "test_metrics": {"rmse": 2, "mae": 1, "r2": -3}, "coverage": 0.5,
         "prediction_rows": {"partition": "Final", "observed": [1, 2, 3], "predicted": [1, None, 4],
                             "residual": [0, None, -1], "valid": [True, False, True]}},
        {"case": "unknown_case", "engine": "failed", "status": "failed", "reason": "<script>alert(1)</script>",
         "final": {"extrapolation": {"rmse": None, "reason": "invalid predictions"}}},
    ]}
    source = tmp_path / 'scores.json'
    raw = json.dumps(evidence, indent=3).encode()
    source.write_bytes(raw)
    target = write_report(source, tmp_path / 'readout')
    page = target.read_text(encoding='utf-8')
    assert '<svg' in page and 'y = a*x + b' in page
    assert 'extrapolation' in page and 'invalid predictions' in page
    assert '&lt;script&gt;' in page and '<script>alert' not in page
    assert '描画 2 / 保存 3行' in page
    assert '宣言 3' in page
    assert '描画 1 / 保存 2行' in page  # failed attempt remains in each metric-plot denominator
    assert '真値・reference' in page and '未記録。良いfitを真式とみなさない' in page
    assert (target.parent / 'evidence.json').read_bytes() == source.read_bytes() == raw
    with pytest.raises(FileExistsError):
        write_report(source, target.parent)


def test_unknown_schema_is_not_assumed_identification(tmp_path):
    source = tmp_path / 'bad.json'
    source.write_text('{"attempts": 4}')
    with pytest.raises(ValueError, match='results or attempts array'):
        write_report(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_absent_row_predictions_are_explicit_not_synthesized():
    evidence = {"results": [{"case": "case", "equation": "y=x", "test_metrics": {"rmse": 0.1}}]}
    before = deepcopy(evidence)
    page = render(evidence, source_name='source', source_sha256='digest')
    assert '行配列はこの結果に未保存' in page
    assert '実測 vs 予測（破線' not in page
    assert evidence == before


def test_reference_and_observational_truth_are_distinct_and_escaped():
    evidence = {"results": [{"case": "observed", "equation": 'x'}, {"case": "synthetic", "equation": '2*x'}]}
    context = {"observed": {"truth_status": 'No known true formula', "target_unit": 'N'},
               "synthetic": {"truth_status": 'Known generator', "reference_formula": 'y = 2*x',
                             "source": '<img src=x onerror=alert(1)>'}}
    page = render(evidence, source_name='source', source_sha256='digest', context=context)
    assert 'No known true formula' in page and 'Known generator' in page
    assert '生成式 / 公表reference（推定式とは別）' in page
    assert '<img' not in page


def test_plot_handles_invalid_and_extreme_rows_without_changing_data():
    x, y = [-1e308, 0, 1e308, None], [1e308, 0, -1e308, 2]
    before = deepcopy((x, y))
    page = scatter(x, y, title='t', xlabel='x', ylabel='y')
    assert page.count('<circle ') == 3 and 'inf' not in page.lower() and 'nan' not in page.lower()
    assert (x, y) == before
    with pytest.raises(ValueError, match='different lengths'):
        scatter([1], [1, 2], title='t', xlabel='x', ylabel='y')
    page = scatter([1], [None], title='t', xlabel='x', ylabel='y')
    assert '<svg' not in page
