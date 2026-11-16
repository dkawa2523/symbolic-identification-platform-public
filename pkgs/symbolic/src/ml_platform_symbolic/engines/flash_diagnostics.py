"""Read-only Flash failure classification; never changes the native fit pool."""

from collections import Counter


def decode_expression(raw, tokenizer, engine):
    """Realize typed IEEE754 literals before converting the tagged dialect."""
    from flash_ansr.data.serialization import (
        TAGGED_DELIMITER_TOKENS,
        map_ieee754_spans,
        realize_slot_values,
    )
    from flash_ansr.utils.ieee754 import BYTE_TOKENS, IEEE754_END_TOKEN, IEEE754_START_TOKEN

    ids = tokenizer.extract_expression_from_beam(raw)[0]
    start = tokenizer.token2idx.get(IEEE754_START_TOKEN)
    if start is not None and start in ids:
        mapping = map_ieee754_spans(
            ids, start_id=start, end_id=tokenizer[IEEE754_END_TOKEN],
            byte_ids=[tokenizer[token] for token in BYTE_TOKENS],
            constant_id=tokenizer["<constant>"],
        )
        if mapping is None:
            raise ValueError("Malformed IEEE754 literal span")
        mapped, values = mapping
        decoded = realize_slot_values(tokenizer.decode_expression(mapped), values)
    else:
        decoded = tokenizer.decode_expression(ids)
    if any(token in TAGGED_DELIMITER_TOKENS for token in decoded):
        return list(engine.to_prefix(decoded))
    return list(decoded)


def classify_failures(ledger, decode, is_valid):
    """Keep upstream statuses immutable; valid syntax is not a successful fit."""
    rows = []
    for index, status in enumerate(ledger["fit_status"]):
        if status == 0:
            continue
        row = {"ledger_index": index, "native_fit_status": status}
        if status not in {1, 2}:
            row.update(classification="diagnostic_unavailable", reason="Unknown native fit status")
            rows.append(row)
            continue
        try:
            expression = decode(ledger["token_lists"][index])
            row["expression"] = expression
            if not is_valid(expression):
                raise ValueError("Native engine rejects the realized expression")
            row["classification"] = "fit_failed" if status == 1 else "not_fitted_valid_syntax"
            row["reason"] = "Realized expression is syntactically valid; no native survivor recorded. This does not establish whether refinement was attempted or why it failed"
        except ValueError as error:
            row["classification"] = "invalid_expression"
            row["reason"] = str(error)
        except Exception as error:
            row["classification"] = "diagnostic_unavailable"
            row["reason"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return {
        "scope": "Read-only failure diagnosis after exact IEEE754 realization; native search, fits, raw ledger and candidate acceptance unchanged",
        "rows": rows,
        "counts": dict(Counter(row["classification"] for row in rows)),
        "native_invalid_with_valid_syntax": sum(
            row["native_fit_status"] == 2 and row["classification"] == "not_fitted_valid_syntax" for row in rows
        ),
    }
