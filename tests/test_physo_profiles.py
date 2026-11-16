from copy import deepcopy
import pytest
from ml_platform_symbolic.engines.physo import profile_options,configure_profile

PRESET={'learning_config':{'batch_size':2000,'n_epochs':10**9,'risk_factor':.05,'entropy_weight':.005},'priors_config':[]}

def test_lightweight_compatible_and_explicit_overrides():
    original=deepcopy(PRESET)
    config,opts=configure_profile({},PRESET)
    assert config['learning_config']['batch_size']==128
    assert opts['epochs']==5 and opts['max_n_evaluations']==2000
    config,opts=configure_profile({'batch_size':64,'epochs':2,'risk_factor':.1},PRESET)
    assert config['learning_config']['batch_size']==64 and config['learning_config']['risk_factor']==.1
    assert opts['epochs']==2 and PRESET==original

@pytest.mark.parametrize('profile',['config1','config2'])
def test_official_profile_uses_preset_with_explicit_budget(profile):
    with pytest.raises(ValueError,match='explicit epochs'):
        profile_options({'profile':profile})
    config,opts=configure_profile({'profile':profile,'epochs':3},PRESET)
    assert config==PRESET and opts['epochs']==3 and opts['max_n_evaluations'] is None
    config,opts=configure_profile({'profile':profile,'max_n_evaluations':100,'batch_size':32},PRESET)
    assert config['learning_config']['batch_size']==32 and opts['epochs'] is None
    assert opts['max_n_evaluations']==100

@pytest.mark.parametrize('bad',[None,[],{},'auto','quality'])
def test_unknown_profile_rejected(bad):
    with pytest.raises(ValueError,match='profile must'):
        profile_options({'profile':bad})

@pytest.mark.parametrize("batch,risk,valid", [
    (19, 0.05, False), (20, 0.05, True), (1, 1.0, True), (1, 0.99, False), (128, 0.05, True),
])
def test_empty_native_elite_is_explicitly_rejected_without_changing_budget(batch, risk, valid):
    options = {"batch_size": batch, "risk_factor": risk}
    original = deepcopy(PRESET)
    if valid:
        config, _ = configure_profile(options, PRESET)
        assert config["learning_config"]["batch_size"] == batch
        assert config["learning_config"]["risk_factor"] == risk
    else:
        with pytest.raises(ValueError, match="zero elite"):
            configure_profile(options, PRESET)
    assert PRESET == original and options == {"batch_size": batch, "risk_factor": risk}


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"batch_size": True}, {"risk_factor": 0},
    {"risk_factor": float("nan")}, {"risk_factor": 1.01},
])
def test_invalid_sampling_options_still_rejected_before_native_import(tmp_path, options):
    import numpy as np
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.physo import validate_request

    r = DiscoveryRequest(
        np.ones((2, 1)), np.ones(2), np.arange(2),
        [{"id": "x", "column": "x", "unit": "1"}], {"column": "y", "unit": "1"},
        ["add"], {"max_nodes": 7}, 17, options,
    )
    with pytest.raises(ValueError):
        validate_request(r, DiscoveryContext(tmp_path))

@pytest.mark.native
@pytest.mark.parametrize("template", [False, True])
def test_zero_elite_guard_runs_before_native_search(tmp_path, monkeypatch, template):
    import physo
    from ml_platform_symbolic.engines.physo import discover
    from test_physo_template import request

    r, c = request(tmp_path)
    if not template:
        r.equation_template = None
    r.options["batch_size"] = 8

    def forbidden(*args, **kwargs):
        pytest.fail("zero-elite settings reached native search")

    monkeypatch.setattr(physo, "SR", forbidden)
    with pytest.raises(ValueError, match="batch_size=8.*risk_factor=0.05.*zero elite"):
        discover(r, c)
    assert r.options["batch_size"] == 8 and "risk_factor" not in r.options
