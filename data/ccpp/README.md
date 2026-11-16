# CCPP source data

Official UCI 294, CC BY 4.0, DOI 10.24432/C5002N; attribution: Tfekci & Kaya (2014).
`source.zip` is the original archive; `source_readme.txt` its documentation.
`table.csv` uses Sheet1 only. Five workbook sheets contain the same observations in different orders and must not be concatenated.
`metadata.json` records hashes, schema and duplicate/multiset audit. Same-input duplicate groups are derived leakage guards, not real experiment IDs. Original rows and units are retained.

No split or model evaluation has run. See [the proposed evaluation plan](../../benchmarks/quality/ccpp_plan.md), which is separate from these source artifacts.
