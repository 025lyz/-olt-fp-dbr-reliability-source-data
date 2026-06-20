import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from evaluation_protocol import nominal_valid, protocol_summary
from evaluate_inverse_tmm import boundary_violation
from physics_tmm import FPDBRTMM, add_tmm_args, dump_json, spectral_metrics


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Local TMM refinement around physics-prior residual candidates."
    )
    parser.add_argument(
        "--candidate-csv",
        type=str,
        default=str(
            repo_root
            / "reports"
            / "physics_prior_residual_hardcases_7_v3_targetwl_candidates.csv"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(repo_root / "dataset" / "fp_dbr_data_100000_physics_aware_experiment.npz"),
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(repo_root / "train_data" / "scaler_physics_aware_peaks.pkl"),
    )
    parser.add_argument("--target-indices", type=int, nargs="*", default=None)
    parser.add_argument("--top-seeds", type=int, default=8)
    parser.add_argument("--near-seeds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n-radius", type=int, default=1)
    parser.add_argument("--dh-step", type=float, default=20.0)
    parser.add_argument("--dl-step", type=float, default=60.0)
    parser.add_argument("--lc-step", type=float, default=300.0)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--offset-multipliers", type=float, nargs="*", default=[-1.0, 1.0])
    parser.add_argument("--missing-peak-penalty", type=float, default=0.01)
    parser.add_argument("--false-peak-penalty", type=float, default=0.005)
    parser.add_argument("--peak-shift-penalty", type=float, default=0.005)
    parser.add_argument("--boundary-penalty", type=float, default=0.02)
    parser.add_argument("--boundary-margin-frac", type=float, default=0.03)
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "reports"))
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="physics_prior_residual_v3_local_refine",
    )
    add_tmm_args(parser)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key, default=np.nan):
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row, key, default=0):
    value = as_float(row, key, np.nan)
    if not np.isfinite(value):
        return default
    return int(round(value))


def physics_score(metrics, params, scaler, args):
    peak_shift = metrics["mean_peak_shift"]
    peak_shift_term = (
        peak_shift / max(args.peak_distance_threshold, 1e-6)
        if np.isfinite(peak_shift)
        else 1.0
    )
    return float(
        metrics["mse"]
        + args.missing_peak_penalty * metrics["missing_peak_count"]
        + args.false_peak_penalty * metrics["false_peak_count"]
        + args.peak_shift_penalty * peak_shift_term
        + args.boundary_penalty
        * boundary_violation(params, scaler, margin_frac=args.boundary_margin_frac)
    )


def candidate_params(row):
    return np.array(
        [
            as_float(row, "d_H"),
            as_float(row, "d_L"),
            as_int(row, "N"),
            as_float(row, "L_c"),
        ],
        dtype=float,
    )


def select_seed_rows(rows, args):
    grouped = {}
    for row in rows:
        grouped.setdefault(as_int(row, "target_index"), []).append(row)

    if args.target_indices:
        grouped = {target: grouped[target] for target in args.target_indices if target in grouped}

    selected = {}
    for target, target_rows in grouped.items():
        by_score = sorted(target_rows, key=lambda row: as_float(row, "rerank_score", np.inf))
        near_rows = [
            row
            for row in target_rows
            if as_float(row, "mse", np.inf) <= 0.01
            and as_int(row, "missing_peak_count", 99) <= 1
            and as_int(row, "false_peak_count", 99) <= 1
        ]
        near_rows = sorted(near_rows, key=lambda row: as_float(row, "rerank_score", np.inf))

        seeds = []
        seen = set()
        for row in by_score[: max(args.top_seeds, 0)] + near_rows[: max(args.near_seeds, 0)]:
            params = candidate_params(row)
            key = tuple(np.round(params[[0, 1, 3]], 3).tolist() + [int(params[2])])
            if key in seen:
                continue
            seen.add(key)
            seeds.append(row)
        selected[target] = seeds
    return selected


def clip_params(params, scaler):
    params = np.asarray(params, dtype=float).copy()
    low = np.asarray(scaler.data_min_, dtype=float)
    high = np.asarray(scaler.data_max_, dtype=float)
    params = np.clip(params, low, high)
    params[2] = np.rint(params[2])
    return params


def evaluate_params(
    params,
    target_index,
    target_spectrum,
    wavelengths,
    tmm_solver,
    scaler,
    args,
    seed_rank,
    seed_candidate_index,
    stage,
    trial_index,
):
    spectrum = tmm_solver.simulate(params)
    metrics = spectral_metrics(
        target_spectrum,
        spectrum,
        wavelengths,
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
        peak_distance_threshold_nm=args.peak_distance_threshold,
    )
    score = physics_score(metrics, params, scaler, args)
    return {
        "target_index": int(target_index),
        "seed_rank": int(seed_rank),
        "seed_candidate_index": int(seed_candidate_index),
        "stage": stage,
        "trial_index": int(trial_index),
        "d_H": float(params[0]),
        "d_L": float(params[1]),
        "N": int(round(params[2])),
        "L_c": float(params[3]),
        "rerank_score": score,
        "boundary_violation": boundary_violation(
            params, scaler, margin_frac=args.boundary_margin_frac
        ),
        "nominal_valid": nominal_valid(metrics),
        **metrics,
    }


def row_params(row):
    return np.array([row["d_H"], row["d_L"], row["N"], row["L_c"]], dtype=float)


def better(row, best):
    if best is None:
        return True
    if bool(row["nominal_valid"]) and not bool(best["nominal_valid"]):
        return True
    if bool(row["nominal_valid"]) == bool(best["nominal_valid"]):
        return float(row["rerank_score"]) < float(best["rerank_score"])
    return False


def refine_seed(
    seed_row,
    seed_rank,
    target_index,
    target_spectrum,
    wavelengths,
    tmm_solver,
    scaler,
    args,
):
    seed_params = clip_params(candidate_params(seed_row), scaler)
    low = np.asarray(scaler.data_min_, dtype=float)
    high = np.asarray(scaler.data_max_, dtype=float)
    n_min = int(round(low[2]))
    n_max = int(round(high[2]))

    rows = []
    cache = set()
    trial_index = 0

    def eval_once(params, stage):
        nonlocal trial_index
        params = clip_params(params, scaler)
        key = (
            round(float(params[0]), 6),
            round(float(params[1]), 6),
            int(round(params[2])),
            round(float(params[3]), 6),
        )
        if key in cache:
            return None
        cache.add(key)
        row = evaluate_params(
            params,
            target_index,
            target_spectrum,
            wavelengths,
            tmm_solver,
            scaler,
            args,
            seed_rank,
            as_int(seed_row, "candidate_index"),
            stage,
            trial_index,
        )
        trial_index += 1
        rows.append(row)
        return row

    best = None
    for n_value in range(
        max(n_min, int(round(seed_params[2])) - args.n_radius),
        min(n_max, int(round(seed_params[2])) + args.n_radius) + 1,
    ):
        trial = seed_params.copy()
        trial[2] = n_value
        row = eval_once(trial, "n_scan")
        if row is not None and better(row, best):
            best = row

    steps = np.array([args.dh_step, args.dl_step, 0.0, args.lc_step], dtype=float)
    for round_index in range(args.rounds):
        current = row_params(best)
        round_steps = steps * (args.step_decay ** round_index)
        for dim in [0, 1, 3]:
            for multiplier in args.offset_multipliers:
                trial = current.copy()
                trial[dim] += float(multiplier) * round_steps[dim]
                row = eval_once(trial, f"round{round_index + 1}_dim{dim}")
                if row is not None and better(row, best):
                    best = row
                    current = row_params(best)
    return best, rows


def main():
    args = parse_args()
    data = np.load(args.data_path)
    wavelengths = data["wavelengths"].astype(np.float32)
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    tmm_solver = FPDBRTMM(
        wavelengths,
        ge_file=args.ge_file,
        sio2_file=args.sio2_file,
        material_wavelength_unit=args.material_wavelength_unit,
        substrate_index=args.substrate_index,
    )

    candidates = read_csv(args.candidate_csv)
    selected = select_seed_rows(candidates, args)
    all_trial_rows = []
    summaries = []

    print(f"Refining {len(selected)} targets from {args.candidate_csv}")
    for target_index, seed_rows in selected.items():
        target_spectrum = data["spectra"][target_index].astype(np.float32)
        target_best = None
        target_rows = []
        print(f"target={target_index} seeds={len(seed_rows)}")
        for seed_rank, seed_row in enumerate(seed_rows):
            best, rows = refine_seed(
                seed_row,
                seed_rank,
                target_index,
                target_spectrum,
                wavelengths,
                tmm_solver,
                scaler,
                args,
            )
            target_rows.extend(rows)
            if better(best, target_best):
                target_best = best
        valid_rows = [row for row in target_rows if row["nominal_valid"]]
        best_valid = min(valid_rows, key=lambda row: row["rerank_score"]) if valid_rows else None
        all_trial_rows.extend(target_rows)
        summary = {
            "target_index": int(target_index),
            "seed_count": int(len(seed_rows)),
            "trial_count": int(len(target_rows)),
            "nominal_valid_trial_count": int(len(valid_rows)),
            "candidate_covered": bool(best_valid is not None),
            "selected_nominal_valid": bool(target_best["nominal_valid"]),
            "best_refined": target_best,
            "best_valid_refined": best_valid,
        }
        summaries.append(summary)
        best = target_best
        print(
            f"target={target_index} trials={len(target_rows)} "
            f"valid={len(valid_rows)} selected_valid={best['nominal_valid']} "
            f"mse={best['mse']:.6f} score={best['rerank_score']:.6f} "
            f"params=[{best['d_H']:.1f}, {best['d_L']:.1f}, {best['N']}, {best['L_c']:.1f}]"
        )

    selected_valid = [row["selected_nominal_valid"] for row in summaries]
    covered = [row["candidate_covered"] for row in summaries]
    valid_counts = [row["nominal_valid_trial_count"] for row in summaries]
    best_mse = [row["best_refined"]["mse"] for row in summaries]
    payload = {
        "candidate_csv": args.candidate_csv,
        "data_path": args.data_path,
        "num_targets": int(len(summaries)),
        "top_seeds": int(args.top_seeds),
        "near_seeds": int(args.near_seeds),
        "rounds": int(args.rounds),
        "n_radius": int(args.n_radius),
        "dh_step": float(args.dh_step),
        "dl_step": float(args.dl_step),
        "lc_step": float(args.lc_step),
        "step_decay": float(args.step_decay),
        "offset_multipliers": args.offset_multipliers,
        "missing_peak_penalty": args.missing_peak_penalty,
        "false_peak_penalty": args.false_peak_penalty,
        "peak_shift_penalty": args.peak_shift_penalty,
        "boundary_penalty": args.boundary_penalty,
        "boundary_margin_frac": args.boundary_margin_frac,
        "evaluation_protocol": protocol_summary(),
        "selected_nominal_valid_targets": int(np.sum(selected_valid)),
        "candidate_covered_targets": int(np.sum(covered)),
        "nominal_valid_trial_count_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "nominal_valid_trial_count_median": float(np.median(valid_counts)) if valid_counts else 0.0,
        "best_refined_mse_mean": float(np.mean(best_mse)) if best_mse else np.nan,
        "best_refined_mse_median": float(np.median(best_mse)) if best_mse else np.nan,
        "summaries": summaries,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    csv_path = output_dir / f"{args.output_prefix}_trials.csv"
    dump_json(json_path, payload)
    write_csv(csv_path, all_trial_rows)
    print("Local refinement complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
