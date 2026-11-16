# NIST StRD supplied-form numerical regression

This is coefficient estimation with the correct formula supplied, not blind symbolic discovery, mechanistic validation, or held-out/generalization evidence. All published observations are fitted. Certified coefficients are used only for grading after each fit; starts are exactly the two published vectors. Parameters are unbounded, no rows are dropped, and no restart, initialization correction or family substitution is performed. Unit labels are dimensionless numerical coordinates, not assertions about physical units.

Cases: Misra1a (14 dental adsorption measurements; two-parameter saturation exponential) and Chwirut2 (54 ultrasonic response measurements; three-parameter exponential/rational model). These modest measured-data cases complement synthetic discovery checks without conflating the two tasks.

## Frozen protocol

Before execution, require BOTH relative L2 prediction disagreement <= 1e-6 and relative RSS error <= 1e-6. Prediction comparison uses the certified prediction vector as denominator with scaling before norm evaluation. RSS is the ordinary unweighted sum of squared residuals, compared to the published certified RSS. These are engineering tolerances, not certified significant-digit claims. Default fitting budget is 1000 function evaluations per start. Failures and disagreements remain in the output, separately from successful agreement. Parameter values are included for interpretation; parameter agreement is not required because predictive agreement does not establish identifiability.

Run offline from the repository with the source package importable:

    python scripts/benchmark_nist.py --output results/nist/calculation/results.json
    python -m pytest tests/test_nist_benchmark.py -q

The script writes every attempt, exact initial/certified/fitted parameters where available, formula, observation count, hashes, source line references, budget, library versions, metrics and failure reasons. Exit success means the report was generated, not that all attempts agreed. Source corruption or missing data fails before fitting rather than silently fetching replacements. There is deliberately no automatic network fetch path.

## Provenance and redistribution

Original collection: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
Files are verbatim NIST/ITL StRD ASCII data retrieved from the Ceres mirror:

- https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/data/nist/Misra1a.dat
- https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/data/nist/Chwirut2.dat

Exact SHA256 values are pinned in scripts/benchmark_nist.py and checked on every benchmark run. URLs point to a mutable mirror, but any replacement must match those bytes. To reproduce acquisition explicitly, download those URLs into data/nist without text newline conversion, then run the hash-checking benchmark. Vendored files allow safe offline use. Preserve source headers and their original references. Parameter/start/certification/data line references are parsed into each output, using one-based original lines.

Ceres explicitly designates data/nist/*.dat NIST-PD in https://github.com/ceres-solver/ceres-solver/blob/master/REUSE.toml; its notice is https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/LICENSES/NIST-PD.txt . This provenance applies to these files, not a blanket relicensing of other benchmark collections. Original NIST webpage fetch returned HTTP 403 during research; the public mirror was used and is identified explicitly.

Fast tests perform no network access or optimization: they check exact source parsing, full row preservation and certified objective reproduction, plus distinct objective/prediction metrics. Real optimization is an explicit benchmark, not a fast-test requirement.
