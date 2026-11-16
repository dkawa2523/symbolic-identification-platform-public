import sys
from types import SimpleNamespace

import pytest
from ml_platform_symbolic.engines import e2e, parfam, rasr, torch_runtime


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2", 3])
def test_thread_limits(monkeypatch, value):
    monkeypatch.setattr(torch_runtime.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    with pytest.raises(ValueError, match="threads"):
        torch_runtime.thread_count({"threads": value})


@pytest.mark.parametrize("engine", [rasr, parfam, e2e])
@pytest.mark.parametrize("fail", [False, True])
def test_adapter_thread_scope_restores_on_success_and_error(monkeypatch, engine, fail):
    state = {"threads": 8}
    torch = SimpleNamespace(get_num_threads=lambda: state["threads"], set_num_threads=lambda n: state.update(threads=n))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(torch_runtime.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    monkeypatch.setattr(engine, "validate_request", lambda *args: None)

    def native(*args):
        assert state["threads"] == 2
        if fail:
            raise RuntimeError("native failure")
        return "result"

    monkeypatch.setattr(engine, "_discover", native)
    request = SimpleNamespace(options={"threads": 2}, equation_template=None)
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            engine.discover(request, SimpleNamespace(assets_dir="unused"))
    else:
        assert engine.discover(request, SimpleNamespace(assets_dir="unused")) == "result"
    assert state["threads"] == 8


def test_default_one():
    assert torch_runtime.thread_count({}) == 1
