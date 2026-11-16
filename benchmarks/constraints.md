# Cases and three independent constraint checks

This document defines the input cases and constraint checks. It does not report completion of a current run. Generate a new plan with `scripts/prepare_constraint_matrix.py`; its resolved configuration and result files are the authority for that run.


Authoritative runnable IDs are the18 rows below:13 prior synthetic generator IDs,3 empirical datasets,2 additions. `polynomial/noisy_extrapolation` and `rational_constraints/rational_unknown` overlap in law families, but retain separate rows/data protocols. Historical WAVE reports are evidence, not extra cases. The manual-only narrow-saturation demonstration, profile oracle, domain-hole and offset tests remain integration fixtures; they are not counted as blind-discovery datasets.

One fixed data/split/units/domain bundle is shared within each case. Arms **A/B/C are independent, not cumulative**. Safe numerical/portable semantics, partition integrity, declared physical input domain and no-target-leakage remain prerequisites in every arm.

- **A — dimensional admissibility:** require established expression dimensions; unknown is not a pass. Use real native dimensional paths where implemented. In ballistic A, the named standard-g constant is additionally supplied as immutable information. B/C intentionally do not request that extra constant; this information difference is explicit, and within an arm all methods get identical information.
- **B — response evidence:** a case-specific finite-domain response request. Synthetic-law-derived facts are tagged **informed**, not blind discovery. Empirical guardrails are calculated only from Train by the fixed rule below, not folklore or full-target inspection. Checks are sampled evidence, not proofs.
- **C — representation regularization:** a common postchecked cap on independent declared free parameters:2when Train has fewer than40rows,6for larger cases with1–2features,8for larger multifeature cases. This is a deliberately declared interpretability/overfit stress, not a claim about the true equation's degree or number of physical parameters. Native node budget remains the common60; no syntax-only grammar tricks.

| Exact case ID | A | B (independent) | C |
|---|---|---|---|
| polynomial | dimensionless units | informed: increasing x, decreasing z on positive operating range | free≤6 |
| noisy_extrapolation | dimensionless units | same informed signs, separately audited ID/OOD | free≤6 |
| rational_constraints | dimensionless units; explicit x≥0 domain repair | informed: nonnegative/decreasing x on[0,4] | free≤6 |
| rational_unknown | dimensionless units | informed: increasing x on[.2,3] | free≤6 |
| sine_unknown | dimensionless units | informed: response in[-2,2]; no exact period supplied | free≤6 |
| nonlinear_noisy_ood | dimensionless units | informed: decreasing x only in z≤0 subregion | free≤6 |
| ambiguous_quadratic | dimensionless units | informed: convex response on the declared operating interval | free≤2 |
| correlated_interaction | dimensionless units | informed: increasing x/decreasing z within[-1,1]^2 | free≤2 |
| implicit_positive | dimensional validity without inventing coefficient units | informed: increasing positive branch; only uniquely projectable predictors may answer point-response questions | free≤6 |
| implicit_ambiguous | same; retain implicit-output semantics | informed: both-sign relation/branch coverage; scalar-only output is not evaluable for this question, not an invalid equation | free≤6 |
| dimensional_ballistic | physical units + immutable named g (extra informed information) | informed: nondecreasing/convex displacement for positive time; g not required in this arm | free≤6 |
| sparse_nuisance | supplied dimensionless input meanings; no hidden active-feature list | informed: drive/gain increasing and offset_input decreasing; nuisance effects unconstrained | free≤8 |
| dimensional_sparse | supplied physical dimensions | informed: speed/duration increasing and offset decreasing; nuisance effects unconstrained | free≤8 |
| yacht | supplied dimensionless schema | Train-derived empirical response guardrail on supported operating conditions; no invented global Froude monotonicity | free≤8 |
| airfoil | dB level semantics currently not supported by multiplicative unit checker: explicit unresolved/unsupported A | Train-derived empirical dB guardrail; no 'dB≥0' physics claim | free≤8 |
| concrete | supplied kg/m³/day/MPa schema | Train-derived empirical response guardrail; no unproven global age/ingredient monotonicity | free≤8 |
| damped_resonance | rad/s,1/s,m; preserve angular-frequency meaning | stated passive-oscillator assumption: positive amplitude, nonincreasing damping response; no exact peak location/count/width imposed | free≤6 |
| activated_kinetics | K,concentration_ratio1,rate1/s | stated positive-activation regime: positive rate and nondecreasing T at fixed concentration; no reaction order or48kJ/mol imposed | free≤6 |

Empirical B guardrail: `[q01(Train)-2*IQR(Train), q99(Train)+2*IQR(Train)]` in the original target units, evaluated only at the predeclared supported conditions. It is an empirical extrapolation guardrail, not a physical law or a claim about sensor limits. For degenerate Train IQR, return an unresolved requirement rather than quietly inventing a bound.

Synthetic informed B requests reveal only their stated property, not a true formula or coefficient. A degree/active-variable/period/peak-count prior is **not** automatically inferred from the hidden law. In particular kinetic n=0/1/2 manual interpretation oracles remain dimensionally valid and observationally identical; neither A, B nor C may eliminate n=0/n=2 as if units established first order. Independent concentration interventions are what discriminate them. Exact peak/width/activation-energy truth metrics are post-selection diagnostics only.

## Ten-method capability/evidence rows

All10appear in every case/arm/seed ledger, including blocked/not-run cells. No surrogate replaces a named library.

| Method | A / native information | B/C and material qualification |
|---|---|---|
| PySR | current adapter records coefficient dimensions unknown; unit metadata question may remain unresolved. Audit real pinned native dimensional API before claiming it enforced | portable point candidates can be postchecked; native operator/argument/nesting configuration is separate |
| PhySO | actual native dimensional prior and named fixed constants, plus common verification | postchecked shape/complexity; arbitrary real-power operator unsupported. dB target unsupported by native unit vocabulary |
| TorchSISSO | actual dimensional path when units are supported; distinct native job when switching that mode. Fixed-constant injection unsupported | descriptor/free-parameter limits and response checked after faithful export |
| Flash-ANSR | supplied units retained; no native unit enforcement claim | prefix/slots/native-domain fidelity preserved; common postchecks; required coefficient dimensions may remain unknown |
| ProGED | generic allowed-operator grammar; no native dimensional enforcement claim | full native coefficients/RHS order; actual DE bounds vs unsupported claims remain distinct; common postchecks |
| ParFam | exact accepted ordinary rational+sin/cos profile; no unverified protected exp/log/sqrt conversion | polynomial/rational approximation is not full elementary-function support. Shape/complexity postchecks |
| RASR | original-space scale transforms retained; no native unit enforcement claim | common point postchecks; full-precision constants and pretrained asset identity retained |
| NASA PySIPS | native SMC structural population/weights separate from later point refits | point equations may be postchecked; do not renormalize/drop posterior mass or turn selected points into a constrained posterior |
| PSE/PSRN | actual enumeration and native coefficient fitting; unit question not silently filled | ordinary/protected semantics and fitted/fixed slots preserved; postchecks; no baseline fallback |
| E2E | released model generates explicit scalar equations; no native unit-guided generation or fixed-constant injection | NoRef and BFGS variants stay separate; common dimension, response and complexity postchecks apply. Unknown coefficient units remain unknown |

Failure of an evaluator/requirement is not automatically numerical invalidity. Preserve native completion, numerical fidelity, requested-constraint result, evaluability and scientific evidence separately in ordinary records. Required unknown/unsupported conditions prevent arm acceptance; they do not become0metrics.

## Workload and staging

- Requested ledger: **18×3×10×3=1,620 cells**, seeds0/1/2. Supplemental baselines/reference rows are labeled separately and do not count toward10.
- Upper native-generation count with perfectly reusable postcheck arms: **540jobs**. Reuse only when actual native data/options/units/constants/search inputs match exactly. Switching dimensional mode, constants or a search-enforced grammar requires another job. Never reuse a full-data fit as leak-free fold discovery.
- Naive upper ceiling at240s:1,620jobs=108h;540jobs=36h. These are caps, **not runtime estimates**. Calibrate real startup/search/export time and resource needs before a prolonged campaign.
- Stage0: save input hashes, exact arm definitions and all cells; validate generators and constraint semantics. Known target/asset blocks stay explicit. No expensive sweep yet.
- Stage1: one seed across the full runnable matrix, after calibration. Do not fill only easy/successful winners. Separate native constrained search from generation followed by constrained evaluation/selection.
- Stage2: remaining two seeds after representative runtime/coverage checkpoint. New constraints/repairs cannot tune against exposed Final. Preserve initial results, rerun affected cells with changed provenance, and identify historical vs independent data use.
- Existing public real-data Final rows have already been exposed in this project; new runs are predeclared audits, not a newly untouched external confirmation. New synthetic points/noise also do not prove pretraining structural novelty.

The shared domain/evaluation now implements explicit central finite-step slope/curvature questions and an independent used-free-parameter cap. Conditions,steps,tolerances,coverage and violations are preserved;these are finite sampled evidence,not global proofs. `create_study(additional_constraints=...)` applies requirements to the immutable source pool without relabeling them as native search enforcement. Original requirements cannot be weakened,and exported bundles retain the effective evaluation conditions. A both-sign implicit branch-coverage evaluator is **not implemented**;its scalar-output cells are explicitly not_evaluable,not false point-prediction successes.


No retrospective historical result has been relabeled as a successful cell in this new matrix.
