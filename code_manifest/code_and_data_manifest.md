# Code and Data Manifest

This manifest lists the source-code and data artifacts relevant to the manuscript. The full project directory contains additional exploratory scripts and historical outputs; the files below are the active reproducibility set for the Optics Communications submission.

## Local Full-Project Datasets

- `dataset/fp_dbr_data_100000_physics_aware_experiment.npz`
  - Main in-house-material simulated dataset used for the benchmark.
- `dataset/fp_dbr_data_100000_physics_aware_database.npz`
  - External-material database-constant dataset used for the separately trained replication.
- `dataset/Amotchkina-Ge.txt`, `dataset/Kischkat-sio2.txt`
  - Optical-constant tables used for external-material replication.
The public external optical-constant text files are included in `source_data/`. The in-house thin-film optical-constant tables and large `.npz` simulation datasets are not bundled into this lightweight repository. The repository instead includes figure/table source-data exports and the active source-code copies. If the Editorial Manager system or editor requests the complete simulation datasets during review, upload the `.npz` datasets separately or provide them through a repository record with large-file support.

The included `source_code/` files are the active script copies used in the full local project. Some scripts retain default paths to the full local dataset directory and to in-house optical-constant files. The compact public repository is therefore intended to support review of reported source-data tables and script provenance, not to guarantee a full end-to-end rerun without the separately stored datasets and in-house material files.

## Main Source-Data Tables In This Package

- `source_data/olt_50target_source_data_table.csv`
- `source_data/olt_runtime_and_direct_tmm_baseline.csv`
- `source_data/optimization_pycma_20targets_physics_score_320calls.csv`
- `source_data/optimization_pycma_20targets_physics_score_320calls.json`
- `source_data/olt_threshold_sensitivity.csv`
- `source_data/olt_fallback_statistics.csv`
- `source_data/olt_physics_interpretation_metrics.csv`
- `source_data/olt_local_parameter_sensitivity.csv`
- `source_data/olt_fig8_phase_mechanism_source_data.csv`
- `source_data/olt_revision_evidence_summary.json`
- `source_data/olt_index_sensitivity_summary.csv`
- `source_data/strategy12_50targets_unified_r60_comparison.csv`
- `source_data/Amotchkina-Ge.txt`
- `source_data/Kischkat-sio2.txt`
- `source_data/*_summary.csv`
- `source_data/*_summary.json`

## Core Scripts

The active review-stage script copies are included under `source_code/`.

- `real/physics_tmm.py`
  - TMM simulation utilities.
- `real/evaluation_protocol.py`
  - Peak detection and nominal-valid metric logic.
- `real/evaluate_inverse_tmm.py`
  - CVAE candidate evaluation and real-TMM reranking.
- `real/evaluate_fabrication_robustness.py`
  - r60 thickness-perturbation robustness protocol.
- `real/evaluate_optimization_baselines.py`
  - Direct random-search, bounded CMA-ES, and differential-evolution TMM baselines.
- `real/run_inference_time_hybrid_pipeline.py`
  - Hybrid inference-time workflow.
- `real/refine_physics_prior_residual_candidates.py`
  - Physics-prior residual fallback and local refinement.
- `real/summarize_final_robustness.py`
  - Robustness summary generation.
- `real/export_hybrid_final_designs.py`
  - Final selected-design export.
- `real/make_olt_material_replication_figure.py`
- `real/make_olt_mechanism_figure.py`
  - Phase-residual and field-profile mechanism figure generation.
- `real/remake_olt_figures.py`
  - Figure-generation scripts.
- `dataset/generatedata.py`
  - Dataset-generation script used for the main synthetic FP-DBR dataset workflow.
- `dataset/generate_dataset.py`
  - Additional dataset-generation entry point retained for reproducibility.

## Review-Stage Availability

The manuscript currently states that source data and scripts are provided as supplementary/source files for review and will be deposited in a public repository upon acceptance. This GitHub repository provides a lightweight public source-data and source-code record; a formal DOI or archival record can be added later if required by the journal.

## Python Dependency Note

The conventional optimizer baseline uses the official `cma` package (`cma==4.4.4`) for the pycma bounded CMA-ES run. Other core scripts also require the existing project dependencies used for TMM simulation and scientific Python workflows, including `numpy`, `scipy`, `tmm`, and plotting libraries for figure generation.

## Availability Statement Risk

Elsevier research-data instructions generally prefer repository deposit and citation/linking where possible. This package includes source-data and source-code files for review-stage reproducibility; add a formal repository URL/DOI to the manuscript if the Editorial Manager system or editor requires it.
