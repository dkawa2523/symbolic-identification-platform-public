# Function-identification benchmark protocol (2026-11-08)

This protocol separates three questions. It is not an all-method leaderboard or
proof that an equation is a physical law.

1. **Numerical estimation:** published NIST StRD observations, supplied model,
   published starting values, and certified least-squares solution. This tests
   the common coefficient fitter, not equation discovery or independent prediction.
2. **Observational transport:** SciLaws-Real proton electromagnetic form factor,
   using the public Train and Test partitions. Model selection sees only an inner
   split of Train; Test is opened only after every method/seed choice is frozen.
   No ground-truth structure is available. The ten Test points measure a limited
   change in input range, not general scientific validity.
3. **Identifiability control:** freshly sampled saturation responses with either
   independently varied or perfectly confounded inputs. Three independent data
   draws and three native solver seeds distinguish these two variation sources.
   A new random draw of a documented family is not a hidden structural holdout.

The campaign uses mean, linear, dictionary and ParFam rational search. These are
three deterministic baselines and one optional native method, not a representative
ranking of all supported engines. ParFam's rational grammar is informed by the
documented synthetic family; this is not blind discovery of an unknown class.
Native budgets and constraints are identical across the two synthetic support
arms. No pretrained method is included.

Each run retains native completion status, candidate and eligible counts, selected
equation and complexity, Selection error, held-out error, and failures. Selection
uses eligible minimum RMSE, then nodes, free parameters and candidate ID. We do
not refit after selection. All choices are frozen together before scoring Test.
Runs with no candidate, timeouts, and invalid predictions remain in the denominator. Invalid
rows are never removed to improve a score. Accuracy and scientific response
agreement are recorded separately. No benchmark pass is a deployment approval.

For synthetic cases, report input-range interpolation and extrapolation separately,
and a sampled x-response derivative error on independently varied coordinates.
This is numerical response agreement, not a symbolic equivalence proof or causal
inference from observational data. The confounded-input arm deliberately cannot
identify the two separate effects from Train/Selection alone.

Published datasets and code families can be familiar to developers and models.
The intended separation is executable Train/Selection versus scoring data; it is
not a claim that public benchmark answers are secret. Use a new external study
with predeclared engineering criteria before scientific adoption.
