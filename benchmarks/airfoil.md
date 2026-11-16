# Airfoil condition and velocity generalization audit

This is a reproducible predictive check of the existing common discovery and
evaluation path. It is not a new physical-law discovery claim, an all-engine
leaderboard, or an independent external validation of engineering usefulness.

## Source and availability

[UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise)
is an observed NASA wind-tunnel dataset with 1,503 rows, five inputs, and scaled
sound pressure level in dB as the response. The official dataset license is
CC BY 4.0. Cite Brooks, Pope and Marcolini (1989), DOI
[10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C).

The official ZIP was downloaded and inspected on 2026-11-08: 60,126 bytes,
SHA-256 `5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c`.
It matches the repository's `data/airfoil/source.zip`. The numeric source has
no identical complete rows, 27 distinct attack angles, and four velocities
(31.7, 39.6, 55.5 and 71.3 m/s). The UCI page's Binary type label for attack
angle must not be used to encode its actual numeric values as a binary category.

The 106 groups are exact tuples of attack angle, chord, velocity and displacement
thickness, excluding frequency. They are derived conditions, not original
experiment identifiers; related conditions may still share a blade/test setup.
There is no provided exact ground-truth equation. Public observations and old
Final partitions have been evaluated before. Neither split is guaranteed unseen
by developers, earlier runs, or pretrained models.

## Frozen cases

1. `group_split_seed42`: reuse the existing frozen group split: 937 Train rows
   (67 groups), 246 Selection rows (17 groups), 320 Final rows (22 groups).
2. `velocity_71p3_holdout`: put every 71.3 m/s observation in Final. From the
   remaining groups, shuffle sorted group IDs with Python Random seed
   2026110808 and allocate ceil(20% of groups) to Selection; all other groups
   are Train. This measures transport to a higher observed velocity, not arbitrary
   extrapolation or an isolated causal intervention on velocity.

No exact condition tuple or sample may cross partitions. Split decisions depend
only on inputs/identifiers, not target values. Both cases are fixed before any
discovery. The cases are separate diagnostic conditions: cross-case overlap is
possible and they are not independent replicates.

## Execution, selection and scoring

Use existing `prepare`, `run_discovery`, `create_study`, `select_row` and
`prediction_metrics`. Three deterministic baselines are evaluated: mean, linear,
and dictionary (maximum degree 3, maximum 8 terms). Each has a 60-second wall
budget. Candidate-pool policy is `all`, so all faithfully exported candidates
reach comparison without a selection-pool cap. This is six planned attempts; it
does not measure stochastic solver-seed variability or validate native search.

Original units are retained as CSV source units. The common canonical map
converts attack angle from degrees to radians. External scoring uses that frozen
map and converts predictions to the public output coordinate. dB is retained;
it is not falsely relabeled as a dimensionless physical response. The existing
multiplicative-dimension path does not certify dB relations, so require_units is
false and no dimensional-consistency claim is made.

Solver CSV files and prepared references contain Train/Selection rows only.
Preparation may partition and hash Final data but computes no model metrics on
it. All six candidate selections are written to `choices.json` before a
`test_access.json` marker is created and Final targets are opened for scoring.
Selection uses eligible minimum Selection RMSE, then nodes, free parameters,
and candidate ID. Coefficients are not refitted after Selection. Source, runtime,
input and selected-study hashes guard the phase boundaries. A changed source
requires a new campaign; failed attempts are preserved.

Report native completion, candidate/eligible counts, selected equation and
complexity, Selection RMSE, Final RMSE/MAE/R2, invalid-row coverage, and failures.
An invalid Final row invalidates whole-set metrics rather than being dropped.
The descriptive utility criterion is complete valid coverage and lower Final
RMSE than the model fitted to the training mean. It is not an engineering
acceptance tolerance or a scientific adoption rule. Failed/no-candidate attempts
stay in the denominator.

## Run

From the repository on Windows, with the prebuilt core profile:

```powershell
py -3.12 scripts/run_linux.py --profile core --output-root results/airfoil/calculation -- python scripts/benchmark_airfoil_generalization.py all --root results/airfoil/calculation/campaign
```

The campaign directory must not already exist. `prepare`, `discover`, `select`
and `score` can also be called separately. Never reselect or tune after opening
Final. Results are the saved `plan.json`, `choices.json`, `test_access.json`,
`scores.json` and each engine's native artifacts. No network access is used by
the Docker execution.

Focused checks:

```powershell
py -3.12 scripts/run_linux.py --profile core --output-root results/airfoil/calculation -- python -m ruff check --no-cache scripts/benchmark_airfoil_generalization.py tests/test_airfoil_generalization_benchmark.py
py -3.12 scripts/run_linux.py --profile core --output-root results/airfoil/calculation -- python -m pytest -q tests/test_airfoil_generalization_benchmark.py
```

