from copy import deepcopy
import pytest
from ml_platform_symbolic.report_data import _display_unit_labels
from ml_platform_symbolic.config import resolve_config
from ml_platform_symbolic.figure_output import build_plot_data
from test_figure_output import saved_view


def schema(unit='1', source='1', scale=1, offset=0):
    return dict(variables=[dict(column='x',unit='1',symbol='x0')],target=dict(column='y',unit=unit),
                unit_coordinates=dict(columns={'y':dict(source_unit=source,problem_unit=unit,scale=scale,offset=offset)}))


def test_metadata_is_optional_and_does_not_mutate():
    data=schema(); before=deepcopy(data)
    assert _display_unit_labels(data,{})=={}
    assert _display_unit_labels(data,{'y':'MW'})['y']=='MW source label; model=1'
    assert data==before


def test_unknown_model_does_not_gain_dimension_proof():
    assert 'model=unknown' in _display_unit_labels(schema(None,None),{'y':'dB'})['y']
    assert _display_unit_labels(schema(None,None),{})=={}


@pytest.mark.parametrize('label',['MW','m','kPa'])
def test_physical_values_cannot_be_relabelled(label):
    with pytest.raises(ValueError):_display_unit_labels(schema('Pa','Pa'),{'y':label})


def test_frozen_public_coordinate_not_working_coordinate_is_authority():
    data=schema('Pa','kPa',1000)
    assert _display_unit_labels(data,{'y':'kPa'})=={'y':'kPa'}
    with pytest.raises(ValueError):_display_unit_labels(data,{'y':'Pa'})
    with pytest.raises(ValueError):_display_unit_labels(schema('1','1',1000),{'y':'MW'})


@pytest.mark.parametrize('display',[{'z':'MW'},{'y':''},{'y':None},{'y':'MW\nmisleading'},[]])
def test_invalid_metadata_rejected(display):
    with pytest.raises(ValueError):_display_unit_labels(schema(),display)


def test_config_preserves_metadata_and_numeric_mapping(synthetic_config):
    raw=deepcopy(synthetic_config); raw['data']['display_units']={'y':'MPa'}
    base=resolve_config(synthetic_config); actual=resolve_config(raw)
    assert actual['data'].pop('display_units')=={'y':'MPa'}
    base['data'].pop('display_units')
    assert actual==base


def test_graph_labels_do_not_change_rows_or_model_units():
    view=saved_view(); view['unit']='1'
    for card in view['candidates']:
        for interp in card['interpretations']:
            for path in interp['paths']:path['unit']='1'
    for pair in view['analysis_comparisons']:pair['unit']='1'
    before=deepcopy(view); baseline=build_plot_data(view)
    view['unit_label']=_display_unit_labels(schema(),{'y':'MW'})['y']
    after=build_plot_data(view)
    assert view['unit']=='1'
    assert [f['rows'] for f in after['figures']]==[f['rows'] for f in baseline['figures']]
    response=next(f for f in after['figures'] if f['kind']=='response')
    assert 'MW source label' in response['y_label']
    assert before['candidates']==view['candidates']


def test_public_export_loader_forwards_explicit_override(tmp_path, monkeypatch):
    import ml_platform_symbolic.reporting as reporting
    captured={}
    def build(directory, **options):
        captured.update(options)
        return {'unit':'1', 'display_units':options['display_units']}
    monkeypatch.setattr(reporting,'build_report_view',build)
    value=reporting.load_report_view(tmp_path,display_units={'y':'MW'})
    assert captured['display_units']=={'y':'MW'}
    assert value['unit']=='1'
