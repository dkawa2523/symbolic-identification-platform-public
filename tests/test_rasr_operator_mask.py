"""Pinned sample hook restricts operators, without claiming prefix syntax guarantees."""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.native


@pytest.fixture(autouse=True)
def offline_assets(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/work/pretrained/rasr/cache/xdg")
    monkeypatch.setenv("HF_HOME", "/work/pretrained/rasr/cache/hf")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def fixture():
    import torch
    from ml_platform_symbolic.engines.rasr_decode import OperatorMask

    tokens = {0: "<soe>", 1: "<eoe>", 2: "sin", 3: "x1", 4: "+", 5: "1", 6: "e+", 7: "pow2"}
    fitted = SimpleNamespace(tokenizer=SimpleNamespace(index_to_token=tokens),
                             operator_arity={"sin": 1, "+": 2, "pow2": 1})

    class Decoder:
        soe_token_id, eoe_token_id, max_len = 0, 1, 8
        def _create_memory(self, table, mask):
            return torch.zeros((1, 2, 4)), torch.ones((1, 2), dtype=torch.bool)
        def forward(self, **kwargs):
            prefix = kwargs["decoder_input_ids"]
            logits = torch.full((*prefix.shape, 8), -20.0)
            for row, ids in enumerate(prefix.tolist()):
                if len(ids) == 1:
                    logits[row, -1, 2] = 10.0
                    logits[row, -1, 4] = 9.0
                elif len(ids) == (4 if ids[1] == 4 else 3):
                    logits[row, -1, 1] = 10.0
                else:
                    logits[row, -1, 3] = 10.0
            return logits
    return Decoder(), OperatorMask(fitted, {"add"})


def test_native_sample_applies_mask_before_top_k_and_keeps_raw_scores():
    import torch
    from src.rasr.model.model import RASRTransformer
    model, mask = fixture()
    args = dict(tbl=torch.zeros((1, 3, 2)), temperature=0, top_k=1)
    raw, raw_scores = RASRTransformer.sample(model, **args)
    explicit, explicit_scores = RASRTransformer.sample(model, **args, token_mask=None)
    assert raw == explicit == [[[2, 3]]]
    assert raw_scores == explicit_scores
    constrained, scores = RASRTransformer.sample(model, **args, token_mask=mask)
    assert constrained == [[[4, 3, 3]]]
    # top-k=1 would have eliminated '+' if the restriction occurred afterwards.
    first = model.forward(decoder_input_ids=torch.tensor([[0]]))[0, -1]
    expected = torch.log_softmax(first, -1)[4].item()
    assert scores[0][0] == pytest.approx(expected, abs=1e-5)
    assert mask.steps == 4


def test_operator_only_mask_preserves_digit_exponent_eos_and_fixed_power_meaning():
    import torch
    from ml_platform_symbolic.engines.rasr_decode import OperatorMask
    _, mask = fixture()
    logits = torch.arange(8, dtype=torch.float32)[None, :]
    result = mask(torch.tensor([[0]]), logits, 8)
    assert torch.isneginf(result[0, [2, 7]]).all()
    assert torch.equal(result[0, [0, 1, 3, 4, 5, 6]], logits[0, [0, 1, 3, 4, 5, 6]])
    fitted = SimpleNamespace(tokenizer=SimpleNamespace(index_to_token={0: "pow2", 1: "2"}),
                             operator_arity={"pow2": 1})
    assert OperatorMask(fitted, {"pow_int"}).forbidden == []
    assert OperatorMask(fitted, set()).forbidden == [0]  # Numeric coefficient stays distinct.
    invalid = torch.full((1, 8), -float("inf"))
    invalid[0, 2] = 1.0
    with pytest.raises(ValueError, match="no finite"):
        mask(torch.tensor([[0]]), invalid, 8)
    assert np.isfinite(logits.numpy()).all()  # Callback does not mutate model scores.
