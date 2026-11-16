"""Operator-only decoding does not claim grammar, size or semantic completeness."""

import pytest
from ml_platform_symbolic.engines.decoder_runtime import patch_source


@pytest.mark.parametrize("engine", ["e2e", "flash"])
def test_decoder_patch_is_narrow_idempotent_and_rejects_drift(engine):
    source = (
        "            scores = self.proj(tensor)  # (bs, n_words)\n"
        "            scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)\n"
        if engine == "e2e"
        else "                    original_scores = raw_scores\n"
        "                    if grammar is not None:\n                        # Hard grammar constraint: applied AFTER\n"
        "                    work = active_logits\n"
    )
    updated = patch_source(source, engine)
    assert patch_source(updated, engine) == updated
    assert updated.count('getattr(self, "symbolic_operator_mask", None)') == 2
    if engine == "flash":
        assert updated.index("original_scores = raw_scores") < updated.index("self.symbolic_operator_mask")
    with pytest.raises(ValueError, match="differs"):
        patch_source("unrecognized native revision", engine)


@pytest.mark.native
def test_operator_mask_preserves_numeric_prefix_eos_logits_and_rejects_all_masked():
    import torch
    from ml_platform_symbolic.engines.operator_mask import OperatorMask

    token_ids = {"+": 0, "sin": 1, "<bos>": 2, "<eos>": 3, "<b00>": 4, "x1": 5, "E-3": 6}
    mask = OperatorMask(token_ids, ["+", "sin"], {"+": "add", "sin": "sin"}, ["add"])
    logits = torch.arange(7, dtype=torch.float64)[None, :]
    original = logits.clone()
    result = mask(None, logits)
    assert torch.isneginf(result[0, 1])
    torch.testing.assert_close(result[:, [0, 2, 3, 4, 5, 6]], original[:, [0, 2, 3, 4, 5, 6]], rtol=0, atol=0)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    impossible = torch.full_like(logits, -float("inf"))
    impossible[0, 1] = 0
    with pytest.raises(ValueError, match="no finite"):
        mask(None, impossible)


@pytest.mark.native
def test_e2e_native_operator_mask_keeps_portable_normalization(tmp_path):
    import json
    from pathlib import Path

    import numpy as np

    pytest.importorskip("symbolicregression")
    from ml_platform_symbolic.engines.base import DiscoveryContext, DiscoveryRequest
    from ml_platform_symbolic.engines.e2e import discover
    from ml_platform_symbolic.expressions import inspect_equation

    X = np.linspace(-1, 1, 40)[:, None]
    variables = [{"id": "x0", "column": "x0", "unit": "1", "domain": [-2, 2]}]
    request = DiscoveryRequest(
        X,
        1.3 * X[:, 0] + 0.2,
        np.arange(len(X)),
        variables,
        {"unit": "1"},
        ["add", "mul"],
        {},
        42,
        {"beam_size": 8, "n_trees_to_refine": 8, "refinement_seconds": 3, "operator_mask": True},
    )
    result = discover(request, DiscoveryContext(tmp_path, assets_dir=Path("pretrained/e2e").resolve()))
    ledger = json.loads((tmp_path / "e2e_native.json").read_text())
    assert ledger["candidates"]
    forbidden = set(result.summary["operator_mask"]["forbidden_native_operator_tokens"])
    assert all(not set(c["prefix"].split(",")) & forbidden for c in ledger["candidates"])
    assert result.summary["operator_mask"]["processed_rows"] > 0
    assert any("div" in inspect_equation(c.equation, variables)["operators"] for c in result.candidates)


@pytest.mark.native
@pytest.mark.parametrize("static", [False, True])
def test_flash_native_raw_operator_mask_and_unmasked_score(tmp_path, monkeypatch, static):
    from pathlib import Path

    import numpy as np

    assets = Path("pretrained/flash").resolve()
    monkeypatch.setenv("XDG_CACHE_HOME", str(assets / "cache/xdg"))
    monkeypatch.setenv("HF_HOME", str(assets / "cache/hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    flash = pytest.importorskip("flash_ansr")
    from ml_platform_symbolic.engines.flash_ansr import BINARY, UNARY
    from ml_platform_symbolic.engines.operator_mask import OperatorMask

    model = flash.FlashANSR.load(
        directory=str(assets / "model"),
        generation_config=flash.SoftmaxSamplingConfig(draws=8, static_decode=static),
        compute={"device": "cpu", "workers": 0, "persistent_pool": False},
    )
    decoder = model.flash_ansr_model
    if static and not decoder.supports_static_decode()[0]:
        pytest.skip("Pinned model does not support the static decoder path")
    mapping = {**BINARY, **{op: op for op in UNARY}, "inv": "div", "pow": "pow", "rootn": "rootn"}
    mask = OperatorMask(
        decoder.tokenizer.token2idx,
        [t for t in decoder.tokenizer.token2idx if (model._get_operator_arity(t) or 0) > 0],
        mapping,
        ["add", "mul"],
    )
    decoder.symbolic_operator_mask = mask
    X = np.linspace(-1, 1, 30)[:, None]
    generated = model.generate(X, 1.3 * X[:, 0] + 0.2, seed=42)
    assert generated.raw_beams
    forbidden = set(mask.forbidden)
    assert all(not set(ids) & forbidden for ids in generated.raw_beams)
    assert mask.steps > 0
    decoder.symbolic_operator_mask = None
    # Replay the actual raw spans. The native _score_log_probs_batch helper
    # expects body-only variants and would incorrectly add the prefix twice.
    import torch
    from flash_ansr.flash_ansr import NUMERIC_DTYPE

    prefix, prefix_numbers = decoder._resolve_generation_prefix(prompt_prefix=generated.prompt_prefix)
    eos = int(decoder.tokenizer["<eos>"])
    sequences = []
    for beam in generated.raw_beams:
        sequence = list(beam)
        if eos in sequence[len(prefix) :]:
            sequence = sequence[: sequence.index(eos, len(prefix)) + 1]
        sequences.append(sequence)
    width = max(map(len, sequences))
    tokens = torch.full(
        (len(sequences), width), int(decoder.tokenizer["<pad>"]), dtype=torch.long, device=generated.device
    )
    for index, sequence in enumerate(sequences):
        tokens[index, : len(sequence)] = torch.tensor(sequence, device=generated.device)
    numbers = None
    if decoder._has_numeric_channel():
        numbers = torch.full(tokens.shape, float("nan"), dtype=NUMERIC_DTYPE, device=generated.device)
        if prefix_numbers is not None:
            numbers[:, : len(prefix_numbers)] = torch.tensor(
                prefix_numbers, dtype=NUMERIC_DTYPE, device=generated.device
            )
    with torch.no_grad():
        logits = decoder.forward(input_tokens=tokens, data=None, input_num=numbers, memory=generated.memory)
        raw = torch.log_softmax(logits, dim=-1)
    scores = []
    for index, sequence in enumerate(sequences):
        targets = tokens[index, len(prefix) : len(sequence)]
        values = raw[index, len(prefix) - 1 : len(sequence) - 1]
        scores.append(float(values.gather(1, targets[:, None]).sum()))
    np.testing.assert_allclose(scores, generated.log_probs, atol=2e-4, rtol=2e-5)


def test_e2e_production_decoder_patch_rejects_unrecognized_source_hash(tmp_path):
    from ml_platform_symbolic.engines.decoder_runtime import patch
    path = tmp_path / "transformer.py"
    path.write_text("unrecognized", encoding="utf8")
    with pytest.raises(ValueError, match="SHA256"):
        patch(path, "e2e")
    assert path.read_text() == "unrecognized"


def test_disabled_operator_mask_metadata_does_not_claim_operator_restriction():
    from ml_platform_symbolic.engines.operator_mask import metadata
    report = metadata(None)
    assert report["enabled"] is False
    assert report["processed_rows"] == 0
    assert "not restricted by this mask" in report["distribution"]
