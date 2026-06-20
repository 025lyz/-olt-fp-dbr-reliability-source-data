# OLT Submission Package

Target journal: Optics & Laser Technology.

Manuscript title:
Reliability-oriented inverse design of mid-infrared Ge/SiO2 FP-DBR filters with TMM-validated resonance correction

This repository provides review-stage source data and active scripts for the manuscript. It is intended to support reproducibility of the reported figures, tables, robustness summaries, and TMM-based evaluation workflow.

## Package Contents

- `manuscript/`
  - `main.pdf`: compiled submission PDF.
  - `main.tex`: Elsevier `elsarticle` source.
  - `references.bib`: BibTeX bibliography.
  - `main.bbl`: generated bibliography file.
  - `elsarticle-num.bst`: bibliography style used for local compilation.
  - `figures/`: figures referenced by `main.tex`.
- `source_data/`
  - Source-data tables and JSON summaries for the main benchmark, direct-TMM baseline, threshold sensitivity, fallback accounting, physics-interpretation metrics, external-material replication, and index-sensitivity check.
- `source_code/`
  - Core Python scripts for TMM simulation, inverse-design evaluation, robustness analysis, direct-optimization baselines, and figure/table generation.
- `admin/`
  - Cover letter, CRediT statement, separate highlights file, author vitae draft, generative-AI statement, and submission checklist.
- `code_manifest/`
  - Code and data manifest for review-stage supplementary/source-code preparation.

## Current Submission Stance

- The manuscript is positioned as a practical physics-guided inverse-design workflow for mid-infrared Ge/SiO2 FP-DBR optical filters.
- It does not claim complete device fabrication, FDTD validation, a PINN formulation, universal inverse design, or zero-shot material transfer.
- Data and code availability statements use review-stage supplementary/source-data/source-code wording, with files included in this package, and state that public repository deposition will follow upon acceptance. No DOI or repository URL is claimed before one exists.
- Public OLT/Elsevier checks on 2026-06-15: separate highlights are required; author vitae are requested; generative-AI use must be declared when applicable; Elsevier's research-data Option C requires repository deposit/citation/linking or an explicit availability statement when this is not yet possible; graphical abstract, suggested reviewers, and ORCID were not found as mandatory items in the public OLT Guide for Authors, although the Editorial Manager workflow may request them.

## Large Dataset Note

The complete local synthetic datasets are not included in this lightweight repository:

- `dataset/fp_dbr_data_100000_physics_aware_experiment.npz`
- `dataset/fp_dbr_data_100000_physics_aware_database.npz`

These files are approximately 374--375 MB each. The repository instead includes compact source-data exports and scripts used for manuscript figures and tables. The complete datasets can be deposited separately through a repository service with large-file support if requested during review.

## Items Still Requiring Author Confirmation

- Whether the current single-author submission is final.
- Whether the acknowledged magnetron-sputtering facility wording is acceptable.
- Whether raw ellipsometry/fitting files should be supplied during review.
- Whether a public repository should be prepared before first submission or whether repository DOI/URL will be added after acceptance.
- Whether the journal submission system requests a graphical abstract, suggested reviewers, or ORCID linking during metadata entry. These were not all visible from the public OLT Guide for Authors page checked on 2026-06-15.
- Author passport-type photo for the vitae item, if requested during submission.
