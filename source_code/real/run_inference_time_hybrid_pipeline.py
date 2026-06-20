import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluation_protocol import nominal_valid


def repo_root():
    return Path(__file__).resolve().parent.parent


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Run the formal inference-time hybrid FP-DBR inverse-design pipeline."
    )
    parser.add_argument(
        "--primary-json",
        type=str,
        default=str(root / "reports" / "inverse_tmm_teacher_replay_strategy12_robust_top24_50x60_r3.json"),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(root / "dataset" / "fp_dbr_data_100000_physics_aware_experiment.npz"),
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(root / "train_data" / "scaler_physics_aware_peaks.pkl"),
    )
    parser.add_argument("--surrogate-path", type=str, default=str(root / "train_data" / "best_surrogate_physics_aware_peaks_v2.pth"))
    parser.add_argument("--surrogate-type", choices=["hybrid", "fourier"], default="hybrid")
    parser.add_argument("--min-peak-count", type=int, default=1)
    parser.add_argument("--strategy-ids", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--num-candidates", type=int, default=180)
    parser.add_argument("--num-branch-seeds", type=int, default=180)
    parser.add_argument("--residual-samples-per-branch", type=int, default=6)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--top-seeds", type=int, default=8)
    parser.add_argument("--near-seeds", type=int, default=4)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--n-radius", type=int, default=1)
    parser.add_argument("--dh-step", type=float, default=20.0)
    parser.add_argument("--dl-step", type=float, default=60.0)
    parser.add_argument("--lc-step", type=float, default=300.0)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--ge-file", type=str, default=str(root / "dataset" / "expriment_Ge.txt"))
    parser.add_argument("--sio2-file", type=str, default=str(root / "dataset" / "expriment_Sio2.txt"))
    parser.add_argument("--output-prefix", type=str, default="formal_hybrid_50targets")
    parser.add_argument("--output-dir", type=str, default=str(root / "reports"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-robustness", action="store_true")
    parser.add_argument("--robustness-trials", type=int, default=60)
    parser.add_argument("--robustness-thickness-sigma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_command(cmd, expected_outputs=None, skip_existing=False):
    expected_outputs = [Path(path) for path in (expected_outputs or [])]
    if skip_existing and expected_outputs and all(path.exists() for path in expected_outputs):
        print(f"Skipping existing: {', '.join(str(path) for path in expected_outputs)}")
        return
    print("Running:")
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=repo_root())


def primary_failure_indices(primary_json):
    payload = load_json(primary_json)
    failures = []
    for summary in payload.get("summaries", []):
        best = summary["best_candidate"]
        if not nominal_valid(best):
            failures.append(int(summary["target_index"]))
    return failures


def write_final_designs_csv(hybrid_csv, output_csv):
    rows = []
    with Path(hybrid_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "target_index": row["target_index"],
                    "d_H": row["final_d_H"],
                    "d_L": row["final_d_L"],
                    "N": row["final_N"],
                    "L_c": row["final_L_c"],
                    "mse": row["final_mse"],
                    "rerank_score": row["final_mse"],
                    "source": row["final_source"],
                }
            )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def write_fallback_designs_csv(hybrid_csv, output_csv):
    rows = []
    with Path(hybrid_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fallback_used"] != "True":
                continue
            rows.append(
                {
                    "target_index": row["target_index"],
                    "d_H": row["final_d_H"],
                    "d_L": row["final_d_L"],
                    "N": row["final_N"],
                    "L_c": row["final_L_c"],
                    "mse": row["final_mse"],
                    "rerank_score": row["final_mse"],
                    "source": "fallback_residual_refine",
                }
            )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return output_csv, len(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = primary_failure_indices(args.primary_json)
    if not failures:
        raise ValueError("Primary run has no nominal-invalid selected targets.")
    failure_text = [str(index) for index in failures]
    print(f"Primary failures: {len(failures)} {failures}")

    fallback_prefix = f"{args.output_prefix}_fallback_candidates"
    fallback_json = output_dir / f"{fallback_prefix}.json"
    fallback_csv = output_dir / f"{fallback_prefix}_candidates.csv"
    run_command(
        [
            sys.executable,
            "real/evaluate_physics_prior_branches.py",
            "--data-path",
            args.data_path,
            "--scaler-path",
            args.scaler_path,
            "--target-indices",
            *failure_text,
            "--exclude-target-indices-from-residual-library",
            "--min-peak-count",
            str(args.min_peak_count),
            "--strategy-ids",
            *[str(item) for item in args.strategy_ids],
            "--num-candidates",
            str(args.num_candidates),
            "--num-branch-seeds",
            str(args.num_branch_seeds),
            "--residual-mode",
            "dataset",
            "--residual-samples-per-branch",
            str(args.residual_samples_per_branch),
            "--residual-scale",
            str(args.residual_scale),
            "--enumerate-n",
            "--include-dataset-target-wavelength",
            "--reference-source-filter",
            "extra_reference",
            "--surrogate-path",
            args.surrogate_path,
            "--surrogate-type",
            args.surrogate_type,
            "--ge-file",
            args.ge_file,
            "--sio2-file",
            args.sio2_file,
            "--output-prefix",
            fallback_prefix,
            "--output-dir",
            args.output_dir,
        ],
        expected_outputs=[fallback_json, fallback_csv],
        skip_existing=args.skip_existing,
    )

    refine_prefix = f"{args.output_prefix}_fallback_refine"
    refine_json = output_dir / f"{refine_prefix}.json"
    refine_csv = output_dir / f"{refine_prefix}_trials.csv"
    run_command(
        [
            sys.executable,
            "real/refine_physics_prior_residual_candidates.py",
            "--candidate-csv",
            str(fallback_csv),
            "--data-path",
            args.data_path,
            "--scaler-path",
            args.scaler_path,
            "--target-indices",
            *failure_text,
            "--top-seeds",
            str(args.top_seeds),
            "--near-seeds",
            str(args.near_seeds),
            "--rounds",
            str(args.refine_rounds),
            "--n-radius",
            str(args.n_radius),
            "--dh-step",
            str(args.dh_step),
            "--dl-step",
            str(args.dl_step),
            "--lc-step",
            str(args.lc_step),
            "--step-decay",
            str(args.step_decay),
            "--ge-file",
            args.ge_file,
            "--sio2-file",
            args.sio2_file,
            "--output-prefix",
            refine_prefix,
            "--output-dir",
            args.output_dir,
        ],
        expected_outputs=[refine_json, refine_csv],
        skip_existing=args.skip_existing,
    )

    hybrid_prefix = f"{args.output_prefix}_summary"
    hybrid_json = output_dir / f"{hybrid_prefix}.json"
    hybrid_csv = output_dir / f"{hybrid_prefix}.csv"
    run_command(
        [
            sys.executable,
            "real/summarize_inference_time_hybrid.py",
            "--primary-json",
            args.primary_json,
            "--fallback-json",
            str(refine_json),
            "--output-prefix",
            hybrid_prefix,
            "--output-dir",
            args.output_dir,
        ],
        expected_outputs=[hybrid_json, hybrid_csv],
        skip_existing=args.skip_existing,
    )

    final_designs_csv = write_final_designs_csv(
        hybrid_csv,
        output_dir / f"{args.output_prefix}_final_designs.csv",
    )
    fallback_designs_csv, fallback_design_count = write_fallback_designs_csv(
        hybrid_csv,
        output_dir / f"{args.output_prefix}_fallback_used_designs.csv",
    )
    print(f"Final designs CSV: {final_designs_csv}")
    print(f"Fallback designs CSV: {fallback_designs_csv} ({fallback_design_count})")

    if args.run_robustness:
        final_robust_prefix = f"{args.output_prefix}_final_robustness_r{args.robustness_trials}"
        run_command(
            [
                sys.executable,
                "real/evaluate_fabrication_robustness.py",
                "--data-path",
                args.data_path,
                "--designs-csv",
                str(final_designs_csv),
                "--design-selection",
                "mse",
                "--num-trials",
                str(args.robustness_trials),
                "--thickness-sigma",
                str(args.robustness_thickness_sigma),
                "--index-sigma-frac",
                "0.0",
                "--seed",
                str(args.seed),
                "--ge-file",
                args.ge_file,
                "--sio2-file",
                args.sio2_file,
                "--output-prefix",
                final_robust_prefix,
                "--output-dir",
                args.output_dir,
            ],
            expected_outputs=[output_dir / f"{final_robust_prefix}.json"],
            skip_existing=args.skip_existing,
        )

    print("Hybrid pipeline complete.")
    print(f"Hybrid summary JSON: {hybrid_json}")


if __name__ == "__main__":
    main()
