import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation_protocol import nominal_valid, protocol_summary
from physics_tmm import (
    FPDBRTMM,
    add_tmm_args,
    detect_peaks,
    dump_json,
    spectral_metrics,
)


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Monte Carlo fabrication robustness evaluation for FP-DBR designs."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(repo_root / "dataset" / "fp_dbr_data_100000_experiment.npz"),
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Use params and target spectrum from a dataset sample.",
    )
    parser.add_argument(
        "--target-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional target-index filter for --designs-json or --designs-csv batch mode.",
    )
    parser.add_argument(
        "--params",
        type=float,
        nargs=4,
        default=None,
        metavar=("D_H", "D_L", "N", "L_C"),
        help="Manual design parameters in nm: d_H d_L N L_c.",
    )
    parser.add_argument(
        "--designs-json",
        type=str,
        default=None,
        help=(
            "Evaluate best candidates from an inverse TMM aggregate JSON "
            "containing summaries[*].best_candidate."
        ),
    )
    parser.add_argument(
        "--designs-csv",
        type=str,
        default=None,
        help=(
            "Evaluate candidates from an inverse TMM candidates CSV. One design "
            "per target is selected by --design-selection."
        ),
    )
    parser.add_argument(
        "--design-selection",
        choices=["mse", "rerank_score", "surrogate_rank"],
        default="mse",
        help="Selection key when --designs-csv contains multiple candidates per target.",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Limit the number of batch designs after loading and filtering.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["dataset", "nominal"],
        default="dataset",
        help="Compare perturbed spectra against dataset target or nominal TMM spectrum.",
    )
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--thickness-sigma", type=float, default=5.0)
    parser.add_argument(
        "--index-sigma-frac",
        type=float,
        default=0.0,
        help="Relative n/k Gaussian sigma. Example: 0.01 means 1 percent.",
    )
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "reports"))
    parser.add_argument("--output-prefix", type=str, default="fabrication_robustness")
    parser.add_argument(
        "--success-mse-threshold",
        type=float,
        default=0.01,
        help="Yield criterion for full-spectrum MSE.",
    )
    parser.add_argument(
        "--success-peak-shift-threshold",
        type=float,
        default=20.0,
        help="Yield criterion for mean peak shift in nm.",
    )
    parser.add_argument(
        "--success-missing-peak-threshold",
        type=int,
        default=0,
        help="Yield criterion for missing target peaks per perturbed spectrum.",
    )
    parser.add_argument(
        "--success-false-peak-threshold",
        type=int,
        default=0,
        help="Yield criterion for extra predicted peaks per perturbed spectrum.",
    )
    parser.add_argument("--plot", action="store_true")
    add_tmm_args(parser)
    return parser.parse_args()


def load_design(args):
    data = np.load(args.data_path)
    wavelengths = data["wavelengths"].astype(np.float32)

    if args.params is not None:
        params = np.array(args.params, dtype=np.float32)
        dataset_target = None
        source = "manual_params"
    elif args.target_index is not None:
        params = data["params"][args.target_index].astype(np.float32)
        dataset_target = data["spectra"][args.target_index].astype(np.float32)
        source = f"dataset_index_{args.target_index}"
    else:
        raise ValueError("Provide either --params D_H D_L N L_C or --target-index INDEX.")

    return wavelengths, params, dataset_target, source


def parse_float(value, default=np.nan):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_dataset(args):
    data = np.load(args.data_path)
    return data, data["wavelengths"].astype(np.float32)


def load_batch_designs(args, data):
    designs = []
    requested = set(args.target_indices) if args.target_indices else None

    if args.designs_json is not None:
        payload = json.loads(Path(args.designs_json).read_text(encoding="utf-8"))
        for summary in payload.get("summaries", []):
            target_index = int(summary["target_index"])
            if requested is not None and target_index not in requested:
                continue
            best = summary["best_candidate"]
            designs.append(
                {
                    "target_index": target_index,
                    "params": np.array(
                        [best["d_H"], best["d_L"], best["N"], best["L_c"]],
                        dtype=np.float32,
                    ),
                    "source": f"json_best_candidate_{target_index}",
                    "nominal_inverse_mse": parse_float(best.get("mse")),
                    "nominal_rerank_score": parse_float(best.get("rerank_score")),
                }
            )

    if args.designs_csv is not None:
        with Path(args.designs_csv).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_target = {}
        for row in rows:
            target_index = int(row["target_index"])
            if requested is not None and target_index not in requested:
                continue
            by_target.setdefault(target_index, []).append(row)

        for target_index, target_rows in sorted(by_target.items()):
            key = args.design_selection
            if key == "rerank_score" and "rerank_score" not in target_rows[0]:
                key = "mse"
            best = min(target_rows, key=lambda row: parse_float(row.get(key), np.inf))
            designs.append(
                {
                    "target_index": target_index,
                    "params": np.array(
                        [
                            parse_float(best["d_H"]),
                            parse_float(best["d_L"]),
                            parse_float(best["N"]),
                            parse_float(best["L_c"]),
                        ],
                        dtype=np.float32,
                    ),
                    "source": f"csv_{args.design_selection}_{target_index}",
                    "nominal_inverse_mse": parse_float(best.get("mse")),
                    "nominal_rerank_score": parse_float(best.get("rerank_score")),
                }
            )

    if args.max_designs is not None:
        designs = designs[: args.max_designs]

    if not designs:
        raise ValueError("No batch designs were loaded.")

    max_index = len(data["params"]) - 1
    bad_indices = [d["target_index"] for d in designs if d["target_index"] > max_index]
    if bad_indices:
        raise ValueError(f"Target indices exceed dataset size: {bad_indices[:10]}")

    return designs


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(args, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    output_dir = Path(args.output_dir)
    mse = [row["mse"] for row in rows]
    peak_shift = [
        row["mean_peak_shift"] for row in rows if np.isfinite(row["mean_peak_shift"])
    ]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(mse, bins=30, color="steelblue", alpha=0.85)
    plt.axvline(args.success_mse_threshold, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Spectral MSE")
    plt.ylabel("Count")
    plt.title("MSE under fabrication perturbation")

    plt.subplot(1, 2, 2)
    if peak_shift:
        plt.hist(peak_shift, bins=30, color="darkorange", alpha=0.85)
        plt.axvline(
            args.success_peak_shift_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
        )
    plt.xlabel("Mean peak shift (nm)")
    plt.title("Peak drift under fabrication perturbation")

    plt.tight_layout()
    plt.savefig(output_dir / f"{args.output_prefix}.png", dpi=180)
    plt.close()


def evaluate_design(
    args,
    tmm_solver,
    wavelengths,
    params,
    dataset_target,
    source,
    target_index=None,
    nominal_inverse_mse=None,
    nominal_rerank_score=None,
):
    nominal_spectrum = tmm_solver.simulate(params)
    if args.target_mode == "dataset" and dataset_target is not None:
        target_spectrum = dataset_target
    else:
        target_spectrum = nominal_spectrum

    nominal_metrics = spectral_metrics(
        target_spectrum,
        nominal_spectrum,
        wavelengths,
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
        peak_distance_threshold_nm=args.peak_distance_threshold,
    )
    nominal_peaks = detect_peaks(
        nominal_spectrum,
        wavelengths,
        height=args.peak_height,
        prominence=args.peak_prominence,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    for trial in range(args.num_trials):
        perturbed_spectrum = tmm_solver.simulate_perturbed(
            params,
            rng,
            thickness_sigma_nm=args.thickness_sigma,
            index_sigma_frac=args.index_sigma_frac,
        )
        metrics = spectral_metrics(
            target_spectrum,
            perturbed_spectrum,
            wavelengths,
            peak_height=args.peak_height,
            peak_prominence=args.peak_prominence,
            peak_distance_threshold_nm=args.peak_distance_threshold,
        )
        perturbed_peaks = detect_peaks(
            perturbed_spectrum,
            wavelengths,
            height=args.peak_height,
            prominence=args.peak_prominence,
        )
        top_peak_drift = np.nan
        if nominal_peaks and perturbed_peaks:
            top_peak_drift = abs(
                nominal_peaks[0]["wavelength"] - perturbed_peaks[0]["wavelength"]
            )

        success = nominal_valid(
            metrics,
            mse_threshold=args.success_mse_threshold,
            peak_shift_threshold=args.success_peak_shift_threshold,
            missing_peak_threshold=args.success_missing_peak_threshold,
            false_peak_threshold=args.success_false_peak_threshold,
        )
        rows.append(
            {
                "target_index": target_index,
                "source": source,
                "trial": int(trial),
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "correlation": metrics["correlation"],
                "spectral_overlap": metrics["spectral_overlap"],
                "mean_peak_shift": metrics["mean_peak_shift"],
                "missing_peak_count": metrics["missing_peak_count"],
                "false_peak_count": metrics["false_peak_count"],
                "mean_fwhm_error": metrics["mean_fwhm_error"],
                "mean_q_error": metrics["mean_q_error"],
                "top_peak_drift": float(top_peak_drift),
                "success": bool(success),
            }
        )

    summary = {
        "source": source,
        "target_index": target_index,
        "nominal_inverse_mse": nominal_inverse_mse,
        "nominal_rerank_score": nominal_rerank_score,
        "params": {
            "d_H": float(params[0]),
            "d_L": float(params[1]),
            "N": int(round(params[2])),
            "L_c": float(params[3]),
        },
        "num_trials": int(args.num_trials),
        "thickness_sigma_nm": float(args.thickness_sigma),
        "index_sigma_frac": float(args.index_sigma_frac),
        "target_mode": args.target_mode,
        "nominal_metrics": nominal_metrics,
        "yield_rate": float(np.mean([row["success"] for row in rows])),
        "mse": finite_summary([row["mse"] for row in rows]),
        "mean_peak_shift": finite_summary([row["mean_peak_shift"] for row in rows]),
        "top_peak_drift": finite_summary([row["top_peak_drift"] for row in rows]),
        "missing_peak_count": finite_summary([row["missing_peak_count"] for row in rows]),
        "false_peak_count": finite_summary([row["false_peak_count"] for row in rows]),
    }
    return summary, rows


def aggregate_batch(design_summaries):
    yield_rates = [item["yield_rate"] for item in design_summaries]
    mse_means = [item["mse"]["mean"] for item in design_summaries]
    peak_shift_means = [item["mean_peak_shift"]["mean"] for item in design_summaries]
    missing_means = [item["missing_peak_count"]["mean"] for item in design_summaries]
    false_means = [item["false_peak_count"]["mean"] for item in design_summaries]

    return {
        "num_designs": int(len(design_summaries)),
        "yield_rate": finite_summary(yield_rates),
        "mse_mean_per_design": finite_summary(mse_means),
        "mean_peak_shift_per_design": finite_summary(peak_shift_means),
        "missing_peak_count_per_design": finite_summary(missing_means),
        "false_peak_count_per_design": finite_summary(false_means),
    }


def flatten_design_summary(summary):
    params = summary["params"]
    return {
        "target_index": summary["target_index"],
        "source": summary["source"],
        "d_H": params["d_H"],
        "d_L": params["d_L"],
        "N": params["N"],
        "L_c": params["L_c"],
        "nominal_inverse_mse": summary["nominal_inverse_mse"],
        "nominal_rerank_score": summary["nominal_rerank_score"],
        "nominal_eval_mse": summary["nominal_metrics"]["mse"],
        "nominal_missing_peak_count": summary["nominal_metrics"]["missing_peak_count"],
        "nominal_false_peak_count": summary["nominal_metrics"]["false_peak_count"],
        "yield_rate": summary["yield_rate"],
        "mse_mean": summary["mse"]["mean"],
        "mse_p90": summary["mse"]["p90"],
        "mean_peak_shift_mean": summary["mean_peak_shift"]["mean"],
        "mean_peak_shift_p90": summary["mean_peak_shift"]["p90"],
        "top_peak_drift_mean": summary["top_peak_drift"]["mean"],
        "top_peak_drift_p90": summary["top_peak_drift"]["p90"],
        "missing_peak_count_mean": summary["missing_peak_count"]["mean"],
        "false_peak_count_mean": summary["false_peak_count"]["mean"],
    }


def run_single(args):
    wavelengths, params, dataset_target, source = load_design(args)
    tmm_solver = FPDBRTMM(
        wavelengths,
        ge_file=args.ge_file,
        sio2_file=args.sio2_file,
        material_wavelength_unit=args.material_wavelength_unit,
        substrate_index=args.substrate_index,
    )
    summary, rows = evaluate_design(
        args,
        tmm_solver,
        wavelengths,
        params,
        dataset_target,
        source,
        target_index=args.target_index,
    )
    summary["evaluation_protocol"] = protocol_summary()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    csv_path = output_dir / f"{args.output_prefix}.csv"
    dump_json(json_path, summary)
    write_csv(csv_path, rows)
    if args.plot:
        maybe_plot(args, rows)

    print("Fabrication robustness evaluation complete.")
    print(
        f"params=[{params[0]:.2f}, {params[1]:.2f}, {int(round(params[2]))}, {params[3]:.2f}]"
    )
    print(f"yield_rate={summary['yield_rate']:.3f}")
    print(f"mse_mean={summary['mse']['mean']}")
    print(f"peak_shift_mean={summary['mean_peak_shift']['mean']}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


def run_batch(args):
    data, wavelengths = load_dataset(args)
    designs = load_batch_designs(args, data)
    tmm_solver = FPDBRTMM(
        wavelengths,
        ge_file=args.ge_file,
        sio2_file=args.sio2_file,
        material_wavelength_unit=args.material_wavelength_unit,
        substrate_index=args.substrate_index,
    )

    design_summaries = []
    trial_rows = []
    print(f"Evaluating fabrication robustness for {len(designs)} designs")
    for design in designs:
        target_index = design["target_index"]
        dataset_target = data["spectra"][target_index].astype(np.float32)
        summary, rows = evaluate_design(
            args,
            tmm_solver,
            wavelengths,
            design["params"],
            dataset_target,
            design["source"],
            target_index=target_index,
            nominal_inverse_mse=design.get("nominal_inverse_mse"),
            nominal_rerank_score=design.get("nominal_rerank_score"),
        )
        design_summaries.append(summary)
        trial_rows.extend(rows)
        print(
            f"target={target_index} yield={summary['yield_rate']:.3f} "
            f"mse_mean={summary['mse']['mean']}"
        )

    aggregate = {
        "data_path": args.data_path,
        "designs_json": args.designs_json,
        "designs_csv": args.designs_csv,
        "design_selection": args.design_selection,
        "num_trials": int(args.num_trials),
        "thickness_sigma_nm": float(args.thickness_sigma),
        "index_sigma_frac": float(args.index_sigma_frac),
        "target_mode": args.target_mode,
        "success_mse_threshold": float(args.success_mse_threshold),
        "success_peak_shift_threshold": float(args.success_peak_shift_threshold),
        "success_missing_peak_threshold": int(args.success_missing_peak_threshold),
        "success_false_peak_threshold": int(args.success_false_peak_threshold),
        "evaluation_protocol": protocol_summary(),
        **aggregate_batch(design_summaries),
        "design_summaries": design_summaries,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    trial_csv_path = output_dir / f"{args.output_prefix}_trials.csv"
    design_csv_path = output_dir / f"{args.output_prefix}_designs.csv"
    dump_json(json_path, aggregate)
    write_csv(trial_csv_path, trial_rows)
    write_csv(design_csv_path, [flatten_design_summary(item) for item in design_summaries])

    print("Batch fabrication robustness evaluation complete.")
    print(f"yield_rate_mean={aggregate['yield_rate']['mean']}")
    print(f"mse_mean_per_design={aggregate['mse_mean_per_design']['mean']}")
    print(f"JSON: {json_path}")
    print(f"Trials CSV:  {trial_csv_path}")
    print(f"Designs CSV: {design_csv_path}")


def main():
    args = parse_args()
    batch_inputs = [args.designs_json is not None, args.designs_csv is not None]
    if any(batch_inputs):
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
