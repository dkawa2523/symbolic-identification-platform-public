# Published proton form-factor range-transfer audit

This is a custom predictive audit of SciLaws-Bench Real, not an official SciLaws
leaderboard result or a law-recovery test. The target GE/GD is a corrected electric
proton form-factor ratio extracted from scattering observations. Q2 is in GeV^2.
These are not raw instrument readings. There is no ground-truth symbolic law.

The released Train has 37 rows (Q2 0.007 to 1.83), Test has 10 (2.07 to 5.85).
Training rows sorted by Q2 with index modulo four equal to two form inner
Selection; all other training rows fit the methods. Test is used only after all
choices are frozen. Repeated solver seeds are not independent data replications.
The original table's standard uncertainties are omitted from the released CSVs:
unweighted RMSE is not uncertainty-aware physical inference.

## Sources and exact bytes

Hugging Face dataset revision: 15b93258fabc8f1785c5a5ffd4d3e51fc15dd860.
Base URL:
https://huggingface.co/datasets/RealSR/SciLaws-Bench/resolve/15b93258fabc8f1785c5a5ffd4d3e51fc15dd860/tasks/typeI/proton_em_form_factor__GE_over_GD/

- metadata.yaml: 1062 bytes, SHA256 191c2da1d7f9f199b2095dbebe0ce0459e1fbe7d2a9c9f8f5c67a2f12690ac2a
- data/train.csv: 685 bytes, SHA256 6a02432026e54d2ee1d78dcc2b51a1beef3ae201a819b44b94202b3d7057c5bf
- data/test.csv: 199 bytes, SHA256 247d8ed8f867670e99c712021f6a78c8ceac71ea8787257d1515b28c9e08ca33

Explicitly download these three files, preserving bytes, into a local directory
as metadata.yaml, train.csv and test.csv. No automatic fetch occurs in the runner.
Do not download eval/ or simulator/: public SciLaws releases contain reference
laws in multiple files, including serialized simulator state. Code MIT does not
license all task data.

Current task metadata declares CC-BY-4.0. This is the curator's data-license
assertion, not an independent grant of rights to the original article. Preserve
attribution and this distinction. Cite Huang et al. (2026), SciLaws-Bench,
https://arxiv.org/abs/2609.01552 and the primary data analysis below. Raw SciLaws
files remain local data and are excluded from the clean source snapshot.

All 47 released central-value pairs were checked exactly against Table II, PDF
page 12, of J. Arrington, W. Melnitchouk and J. A. Tjon (2007), Global analysis of
proton elastic form factor data with two-photon exchange corrections, Physical
Review C 76, 035205: https://doi.org/10.1103/PhysRevC.76.035205 .
Primary PDF https://arxiv.org/pdf/0707.1861 has SHA256
bf378bb0f3af546bcd973866437c8c01ac910196483df6facbcabe203a80f8ee.
The older RealSR predecessor README points to a different Ye2018 dataset and is
not provenance for these rows. Public data were inspected for provenance before
execution; no fitted model was evaluated during that inspection. This is not a
claim that public test values were never seen by any researcher.

## Run the fixed campaign

Use the supported Linux classic runtime, current source on PYTHONPATH, and a new
writable output directory. Record the exact Docker image ID in
SYMBOLIC_CONTAINER_IMAGE_ID. With local raw files at /raw/proton:

    python scripts/benchmark_identification.py prepare --root /out/campaign --proton /raw/proton
    python scripts/benchmark_identification.py discover --root /out/campaign
    python scripts/benchmark_identification.py select --root /out/campaign
    python scripts/benchmark_identification.py score --root /out/campaign

Render the saved evidence without recomputing predictions or changing choices:

    python scripts/summarize_identification.py /out/campaign/scores.json /out/readout

This creates flattened attempt CSV, nested-draw summary JSON, and a compact
Markdown readout. Missing evidence remains missing. Aggregated accuracy is never
labelled adoption or a calibrated confidence statement.

The runner also includes paired synthetic support controls described in
identification.md. Prepare/discover/select do not read the Test CSV.
Source/runtime identity is enforced before each discovery and selection. Score
marks Test exposure before reading it and refuses repeat scoring or reselection.
An execution failure is retained, never repaired by reducing budgets or choosing
another engine. The split boundary is procedural for trusted adapters, not an OS
security sandbox against malicious code with access to the mounted repository.
