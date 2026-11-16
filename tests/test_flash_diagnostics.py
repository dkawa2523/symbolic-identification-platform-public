import copy
import json
from pathlib import Path

import pytest
from ml_platform_symbolic.engines.flash_diagnostics import classify_failures


def test_failure_diagnosis_never_promotes_success_or_changes_raw_ledger():
    ledger = {"fit_status": [0, 1, 2, 2], "token_lists": [["ok"], ["x1"], ["x1"], ["bad"]]}
    before = copy.deepcopy(ledger)
    result = classify_failures(ledger, lambda raw: raw, lambda expression: expression != ["bad"])
    assert ledger == before
    assert [row["classification"] for row in result["rows"]] == [
        "fit_failed", "not_fitted_valid_syntax", "invalid_expression"
    ]
    assert result["native_invalid_with_valid_syntax"] == 1
    assert not any(row["ledger_index"] == 0 for row in result["rows"])


@pytest.mark.native
def test_real_saved_spans_and_typed_literals_with_pinned_public_decoder(monkeypatch):
    assets = Path("pretrained/flash").resolve()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(assets / "cache/hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(assets / "cache/xdg"))
    from flash_ansr import FlashANSR
    from flash_ansr.refine import refinement_slots
    from ml_platform_symbolic.engines.flash_diagnostics import decode_expression

    model = FlashANSR.load(directory=str(assets / "model"), compute={"device": "cpu", "workers": 0})
    cases = json.loads(Path("tests/fixtures/flash_invalid_spans.json").read_text())["cases"]
    ledger = {"fit_status": [2] * len(cases), "token_lists": [case["raw"] for case in cases]}
    def decode(raw):
        return decode_expression(raw, model.tokenizer, model.simplipy_engine)

    result = classify_failures(ledger, decode, model.simplipy_engine.is_valid)
    assert len(cases) == result["native_invalid_with_valid_syntax"] == 8
    assert all(row["classification"] == "not_fitted_valid_syntax" for row in result["rows"])
    # Public realization keeps a typed exponent a literal, not a free coefficient.
    from flash_ansr.data.serialization import realize_slot_values
    expression = realize_slot_values(["pow", "x1", "<constant>"], [2.0])
    assert refinement_slots(expression, model.simplipy_engine, "fittable") == []
    assert model.simplipy_engine.is_valid(expression)
    # A truncated expression and a malformed numeric span remain invalid.
    invalid = {"fit_status": [2, 1], "token_lists": [[model.tokenizer["+"]], [model.tokenizer["<ieee754>"]]]}
    bad = classify_failures(invalid, decode, model.simplipy_engine.is_valid)
    assert bad["counts"] == {"invalid_expression": 2}


def test_api_errors_and_unknown_status_are_not_syntax_claims():
    def unavailable(raw):
        raise KeyError("vocabulary mismatch")

    result = classify_failures(
        {"fit_status": [2, 77], "token_lists": [[1], [2]]}, unavailable, lambda expression: True
    )
    assert result["counts"] == {"diagnostic_unavailable": 2}
    assert result["native_invalid_with_valid_syntax"] == 0
    assert "Unknown native fit status" == result["rows"][1]["reason"]
