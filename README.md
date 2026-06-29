# OLE Transfer Source Data and Code Package

Target journal: Optics and Lasers in Engineering.

Manuscript title:
TMM-validated reliability-oriented design of mid-infrared Ge/SiO2 FP-DBR filters

This repository provides review-stage source data and active scripts for the manuscript. It is intended to support reproducibility of the reported figures, tables, robustness summaries, and transfer-matrix-method (TMM) based evaluation workflow.

## Repository Contents

- `source_data/`
  - Source-data tables and JSON summaries for the main benchmark, direct-TMM baseline, threshold sensitivity, fallback accounting, physics-interpretation metrics, external-material replication, and index-sensitivity check.
  - Public external optical-constant source text files used for the external-material sensitivity check. In-house thin-film optical-constant tables are not included in this public repository.
- `source_code/`
  - Core Python scripts for dataset generation, TMM simulation, inverse-design evaluation, robustness analysis, direct-optimization baselines, and figure/table generation.
- `code_manifest/`
  - Code and data manifest for review-stage supplementary/source-code preparation.

## Current Submission Stance

- The manuscript is positioned as a practical TMM-validated optical-engineering design workflow for mid-infrared Ge/SiO2 FP-DBR optical filters.
- It does not claim complete device fabrication, FDTD validation, a PINN formulation, universal inverse design, or zero-shot material transfer.
- Data and code availability statements use review-stage supplementary/source-data/source-code wording. This public repository provides a lightweight source-data and source-code record for review-stage reproducibility.
- The manuscript was revised before transfer to Optics and Lasers in Engineering to emphasize optical-method validation, resonance-level acceptance, and engineering tolerance criteria rather than a general AI or parameter-optimization claim.

## Large Dataset Note

The complete local synthetic datasets are not included in this lightweight repository:

- `dataset/fp_dbr_data_100000_physics_aware_experiment.npz`
- `dataset/fp_dbr_data_100000_physics_aware_database.npz`

These files are approximately 374--375 MB each. The repository instead includes compact source-data exports and scripts used for manuscript figures and tables. The complete datasets can be deposited separately through a repository service with large-file support if requested during review.

## Reproducibility Scope

The `source_data/` directory is the primary review-stage reproducibility record for the manuscript tables, benchmark summaries, robustness summaries, sensitivity checks, and figure source data. The `source_code/` directory contains active script copies used in the full local project. Some scripts retain default paths to the full local dataset directory and to in-house optical-constant files that are not included in this lightweight public repository. Full end-to-end reruns therefore require the complete local datasets and in-house optical constants, while the compact tables and JSON files included here support review of the reported numerical results.

## Items Still Requiring Author Confirmation

- Whether the current single-author submission is final.
- Whether the acknowledged magnetron-sputtering facility wording is acceptable.
- Whether raw ellipsometry/fitting files should be supplied during review.
- Whether a repository DOI/archival record should be added before acceptance or only after acceptance.
- Whether the journal submission system requests a graphical abstract, suggested reviewers, or ORCID linking during metadata entry.
- Author passport-type photo for the vitae item, if requested during submission.
