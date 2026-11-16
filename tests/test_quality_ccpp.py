"""CCPP sheet identity must retain duplicate multiplicity, not merely set equality."""

import importlib.util
from decimal import Decimal
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ccpp_audit", Path(__file__).resolve().parents[1] / "benchmarks/quality/ccpp.py"
)
ccpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ccpp)


def test_multiset_hash_order_independent_but_duplicate_sensitive():
    a = (Decimal("1.00"), Decimal("2.5"))
    b = (Decimal("3"), Decimal("4"))
    assert ccpp.multiset_hash([a, b, a]) == ccpp.multiset_hash([b, a, a])
    assert ccpp.multiset_hash([a, b, a]) != ccpp.multiset_hash([a, b, b])
    assert ccpp.multiset_hash([a]) == ccpp.multiset_hash([(Decimal("1"), Decimal("2.50"))])
