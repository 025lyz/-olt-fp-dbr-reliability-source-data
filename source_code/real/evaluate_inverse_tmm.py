import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from models import CVAE, build_surrogate
from evaluation_protocol import nominal_valid, protocol_summary
from physics_tmm import (
    FPDBRTMM,
    add_tmm_args,
    detect_peaks,
    dump_json,
    spectral_metrics,
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Evaluate CVAE inverse designs with real TMM reranking."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(repo_root / "dataset" / "fp_dbr_data_100000_experiment.npz"),
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(repo_root / "train_data" / "scaler_experiment.pkl"),
    )
    parser.add_argument(
        "--cvae-path",
        type=str,
        default=str(
            repo_root
            / "train_data"
            / "best_cvae_pinn_model_experiment_main2.0.pth"
        ),
    )
    parser.add_argument(
        "--surrogate-path",
        type=str,
        default=str(
            repo_root / "train_data" / "best_surrogate_experiment_main2.0.pth"
        ),
    )
    parser.add_argument("--surrogate-type", choices=["hybrid", "fourier"], default="hybrid")
    parser.add_argument("--num-targets", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument(
        "--tmm-top-k",
        type=int,
        default=20,
        help="Only the surrogate-best K candidates are revalidated with TMM.",
    )
    parser.add_argument(
        "--tmm-selection",
        choices=["single_queue", "multi_queue"],
        default="single_queue",
        help="Select TMM candidates from one prefilter order or from multiple queues.",
    )
    parser.add_argument("--mq-mse-quota", type=int, default=0)
    parser.add_argument("--mq-peak-quota", type=int, default=0)
    parser.add_argument("--mq-diverse-quota", type=int, default=0)
    parser.add_argument("--mq-n-diverse-quota", type=int, default=0)
    parser.add_argument(
        "--mq-diverse-pool-multiplier",
        type=int,
        default=4,
        help="Diverse queue chooses from this multiple of its quota after MSE sorting.",
    )
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "reports"))
    parser.add_argument("--output-prefix", type=str, default="inverse_tmm_eval")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--target-indices", type=int, nargs="*", default=None)
    parser.add_argument("--min-peak-count", type=int, default=None)
    parser.add_argument("--strategy-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--rerank-objective",
        choices=["mse", "physics_score", "robustness_score"],
        default="mse",
        help="Select best TMM candidate by raw MSE or physics-aware score.",
    )
    parser.add_argument("--missing-peak-penalty", type=float, default=0.0)
    parser.add_argument("--false-peak-penalty", type=float, default=0.0)
    parser.add_argument("--peak-shift-penalty", type=float, default=0.0)
    parser.add_argument("--boundary-penalty", type=float, default=0.0)
    parser.add_argument(
        "--boundary-margin-frac",
        type=float,
        default=0.03,
        help="Fraction of scaler range treated as near-boundary for penalty.",
    )
    parser.add_argument(
        "--refine-lc",
        action="store_true",
        help="Run local TMM grid search over L_c for each reranked candidate.",
    )
    parser.add_argument("--lc-refine-span", type=float, default=600.0)
    parser.add_argument("--lc-refine-points", type=int, default=21)
    parser.add_argument(
        "--prefilter-objective",
        choices=["surrogate_mse", "surrogate_physics", "surrogate_peak"],
        default="surrogate_mse",
        help=(
            "Use surrogate MSE alone, add feasibility terms, or add surrogate "
            "peak-matching terms before selecting TMM top-K."
        ),
    )
    parser.add_argument("--prefilter-boundary-weight", type=float, default=0.0)
    parser.add_argument("--prefilter-qw-weight", type=float, default=0.0)
    parser.add_argument("--prefilter-missing-peak-weight", type=float, default=0.0)
    parser.add_argument("--prefilter-false-peak-weight", type=float, default=0.0)
    parser.add_argument("--prefilter-peak-shift-weight", type=float, default=0.0)
    parser.add_argument("--max-prefilter-boundary", type=float, default=None)
    parser.add_argument("--max-prefilter-qw", type=float, default=None)
    parser.add_argument("--max-prefilter-missing-peaks", type=int, default=None)
    parser.add_argument("--max-prefilter-false-peaks", type=int, default=None)
    parser.add_argument(
        "--qw-target-source",
        choices=["target_peak", "target_max"],
        default="target_peak",
    )
    parser.add_argument("--num-physics-candidates", type=int, default=0)
    parser.add_argument("--physics-tmm-quota", type=int, default=0)
    parser.add_argument(
        "--physics-enumerate-n",
        action="store_true",
        help="Generate physics-prior candidates for every N in physics-n-min..max.",
    )
    parser.add_argument(
        "--physics-candidates-per-n",
        type=int,
        default=1,
        help="Number of physics-prior candidates generated per N when enumerating N.",
    )
    parser.add_argument("--physics-thickness-jitter", type=float, default=0.08)
    parser.add_argument("--physics-lc-jitter", type=float, default=0.08)
    parser.add_argument("--physics-n-min", type=int, default=2)
    parser.add_argument("--physics-n-max", type=int, default=10)
    parser.add_argument(
        "--robustness-trials",
        type=int,
        default=0,
        help="Monte Carlo fabrication trials per candidate. Disabled when 0.",
    )
    parser.add_argument(
        "--robustness-top-m",
        type=int,
        default=0,
        help=(
            "Evaluate robustness only for the nominal-best M TMM candidates. "
            "0 means all TMM reranked candidates when robustness is enabled."
        ),
    )
    parser.add_argument("--robustness-thickness-sigma", type=float, default=5.0)
    parser.add_argument("--robustness-index-sigma-frac", type=float, default=0.0)
    parser.add_argument("--robustness-seed-offset", type=int, default=100000)
    parser.add_argument("--robustness-yield-penalty", type=float, default=0.02)
    parser.add_argument("--robustness-mse-penalty", type=float, default=1.0)
    parser.add_argument("--robustness-peak-shift-penalty", type=float, default=0.005)
    parser.add_argument("--robustness-missing-peak-penalty", type=float, default=0.01)
    parser.add_argument("--robustness-false-peak-penalty", type=float, default=0.005)
    parser.add_argument("--robustness-success-mse-threshold", type=float, default=0.01)
    parser.add_argument("--robustness-success-peak-shift-threshold", type=float, default=20.0)
    parser.add_argument("--robustness-success-missing-peak-threshold", type=int, default=0)
    parser.add_argument("--robustness-success-false-peak-threshold", type=int, default=0)
    parser.add_argument(
        "--allow-robustness-nominal-invalid",
        action="store_true",
        help=(
            "Allow robustness_score to select nominal candidates with missing/false "
            "peaks or excessive peak shift. Default is to reject them."
        ),
    )
    add_tmm_args(parser)
    return parser.parse_args()


def load_models(args, spectra_dim):
    cvae = CVAE(spectra_dim=spectra_dim).to(device)
    cvae.load_state_dict(torch.load(args.cvae_path, map_location=device), strict=False)
    cvae.eval()

    surrogate = build_surrogate(model_type=args.surrogate_type, output_dim=spectra_dim).to(device)
    surrogate.load_state_dict(torch.load(args.surrogate_path, map_location=device))
    surrogate.eval()
    return cvae, surrogate


def choose_indices(num_items, args, global_indices=None):
    if args.target_indices:
        requested = np.array(args.target_indices, dtype=int)
        if global_indices is None:
            return requested
        global_to_local = {int(global_i): local_i for local_i, global_i in enumerate(global_indices)}
        missing = [int(index) for index in requested if int(index) not in global_to_local]
        if missing:
            raise ValueError(
                "Requested --target-indices are not present after dataset filtering: "
                f"{missing}"
            )
        return np.array([global_to_local[int(index)] for index in requested], dtype=int)
    rng = np.random.default_rng(args.seed)
    return rng.choice(num_items, size=min(args.num_targets, num_items), replace=False)


def build_dataset_mask(data, min_peak_count=None, strategy_ids=None):
    mask = np.ones(len(data["params"]), dtype=bool)
    if min_peak_count is not None:
        if "peak_count" not in data:
            raise KeyError("--min-peak-count requires 'peak_count' in the dataset.")
        mask &= data["peak_count"] >= min_peak_count
    if strategy_ids:
        if "strategy_ids" not in data:
            raise KeyError("--strategy-ids requires 'strategy_ids' in the dataset.")
        mask &= np.isin(data["strategy_ids"], np.array(strategy_ids, dtype=int))
    return mask


def inverse_transform_params(scaler, params_norm):
    params = scaler.inverse_transform(params_norm)
    params[:, 2] = np.rint(params[:, 2])
    return params


def summarize_candidate_diversity(params, scaler=None):
    params = np.asarray(params, dtype=float)
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        boundary_low = np.asarray(scaler.data_min_, dtype=float)
        boundary_high = np.asarray(scaler.data_max_, dtype=float)
    else:
        boundary_low = np.array([100.0, 100.0, 2.0, 500.0], dtype=float)
        boundary_high = np.array([1000.0, 1500.0, 10.0, 5000.0], dtype=float)
    boundary_margin = 0.01 * np.maximum(boundary_high - boundary_low, 1.0)
    if len(boundary_margin) > 2:
        boundary_margin[2] = 0.49
    near_boundary = np.any(
        (params <= boundary_low + boundary_margin)
        | (params >= boundary_high - boundary_margin),
        axis=1,
    )

    if len(params) < 2:
        return {
            "candidate_param_pairwise_distance_mean": 0.0,
            "candidate_param_pairwise_distance_median": 0.0,
            "candidate_param_pairwise_distance_p90": 0.0,
            "candidate_unique_rounded_params": int(len(params)),
            "candidate_unique_N": int(len(np.unique(np.rint(params[:, 2])))),
            "candidate_near_boundary_fraction": float(np.mean(near_boundary))
            if len(params)
            else 0.0,
        }

    diff = params[:, None, :] - params[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    upper = distances[np.triu_indices(len(params), k=1)]
    rounded = np.column_stack(
        [
            np.round(params[:, 0], 1),
            np.round(params[:, 1], 1),
            np.rint(params[:, 2]),
            np.round(params[:, 3], 1),
        ]
    )
    unique_rounded = np.unique(rounded, axis=0)
    return {
        "candidate_param_pairwise_distance_mean": float(np.mean(upper)),
        "candidate_param_pairwise_distance_median": float(np.median(upper)),
        "candidate_param_pairwise_distance_p90": float(np.percentile(upper, 90)),
        "candidate_unique_rounded_params": int(len(unique_rounded)),
        "candidate_unique_N": int(len(np.unique(np.rint(params[:, 2])))),
        "candidate_near_boundary_fraction": float(np.mean(near_boundary)),
    }


def boundary_violation(params, scaler, margin_frac=0.03):
    params = np.asarray(params, dtype=float)
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        low = np.asarray(scaler.data_min_, dtype=float)
        high = np.asarray(scaler.data_max_, dtype=float)
    else:
        low = np.array([100.0, 100.0, 2.0, 500.0], dtype=float)
        high = np.array([1000.0, 1500.0, 10.0, 5000.0], dtype=float)

    width = np.maximum(high - low, 1.0)
    margin = margin_frac * width
    if len(margin) > 2:
        margin[2] = 0.49
    lower_violation = np.maximum((low + margin - params) / margin, 0.0)
    upper_violation = np.maximum((params - (high - margin)) / margin, 0.0)
    return float(np.max(np.maximum(lower_violation, upper_violation)))


def target_reference_wavelength(target_spectrum, wavelengths, args):
    peaks = detect_peaks(
        target_spectrum,
        wavelengths,
        height=args.peak_height,
        prominence=args.peak_prominence,
        max_peaks=1,
    )
    if args.qw_target_source == "target_peak" and peaks:
        return float(peaks[0]["wavelength"])
    return float(wavelengths[int(np.argmax(target_spectrum))])


def quarter_wave_violation(params, reference_wavelength_nm, tmm_solver):
    params = np.asarray(params, dtype=float)
    n_h = float(np.interp(reference_wavelength_nm, tmm_solver.wavelengths_nm, np.real(tmm_solver.n_h)))
    n_l = float(np.interp(reference_wavelength_nm, tmm_solver.wavelengths_nm, np.real(tmm_solver.n_l)))
    if n_h <= 0.0 or n_l <= 0.0:
        return 0.0
    ideal_h = reference_wavelength_nm / (4.0 * n_h)
    ideal_l = reference_wavelength_nm / (4.0 * n_l)
    rel_h = abs(params[0] - ideal_h) / max(ideal_h, 1e-6)
    rel_l = abs(params[1] - ideal_l) / max(ideal_l, 1e-6)
    return float(max(rel_h, rel_l))


def generate_physics_prior_candidates(target_spectrum, wavelengths, tmm_solver, scaler, args):
    if args.num_physics_candidates <= 0 and not args.physics_enumerate_n:
        return np.zeros((0, 4), dtype=np.float32)

    reference_wl = target_reference_wavelength(target_spectrum, wavelengths, args)
    n_h = float(np.interp(reference_wl, tmm_solver.wavelengths_nm, np.real(tmm_solver.n_h)))
    n_l = float(np.interp(reference_wl, tmm_solver.wavelengths_nm, np.real(tmm_solver.n_l)))
    n_c = n_l
    if n_h <= 0.0 or n_l <= 0.0:
        return np.zeros((0, 4), dtype=np.float32)

    ideal_h = reference_wl / (4.0 * n_h)
    ideal_l = reference_wl / (4.0 * n_l)
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        low = np.asarray(scaler.data_min_, dtype=float)
        high = np.asarray(scaler.data_max_, dtype=float)
    else:
        low = np.array([100.0, 100.0, 2.0, 500.0], dtype=float)
        high = np.array([1000.0, 1500.0, 10.0, 5000.0], dtype=float)

    rng = np.random.default_rng(args.seed + int(reference_wl * 10))
    candidates = []
    n_min = max(int(args.physics_n_min), int(low[2]))
    n_max = min(int(args.physics_n_max), int(high[2]))
    m_min = int(np.ceil(2.0 * n_c * low[3] / reference_wl))
    m_max = int(np.floor(2.0 * n_c * high[3] / reference_wl))
    if m_max < m_min:
        m_min = 1
        m_max = max(1, int(np.floor(2.0 * n_c * high[3] / reference_wl)))

    if args.physics_enumerate_n:
        n_sequence = []
        for periods in range(n_min, n_max + 1):
            n_sequence.extend([periods] * max(args.physics_candidates_per_n, 1))
    else:
        n_sequence = [
            int(rng.integers(n_min, n_max + 1))
            for _ in range(args.num_physics_candidates)
        ]

    for periods in n_sequence:
        d_h = rng.normal(ideal_h, max(ideal_h * args.physics_thickness_jitter, 1.0))
        d_l = rng.normal(ideal_l, max(ideal_l * args.physics_thickness_jitter, 1.0))
        order = rng.integers(m_min, m_max + 1)
        lc_ideal = order * reference_wl / (2.0 * n_c)
        lc = rng.normal(lc_ideal, max(lc_ideal * args.physics_lc_jitter, 1.0))
        params = np.array([d_h, d_l, periods, lc], dtype=float)
        params = np.clip(params, low, high)
        params[2] = np.rint(params[2])
        candidates.append(params)

    return np.asarray(candidates, dtype=np.float32)


def surrogate_prefilter_scores(
    surrogate_errors,
    surrogate_spectra,
    generated_params,
    target_spectrum,
    wavelengths,
    tmm_solver,
    scaler,
    args,
):
    zeros = np.zeros_like(surrogate_errors, dtype=float)
    prefilter = {
        "scores": surrogate_errors.copy(),
        "boundary": zeros.copy(),
        "qw": zeros.copy(),
        "missing_peaks": zeros.copy(),
        "false_peaks": zeros.copy(),
        "peak_shift": np.full_like(surrogate_errors, np.nan, dtype=float),
    }
    if args.prefilter_objective == "surrogate_mse":
        return prefilter

    reference_wl = target_reference_wavelength(target_spectrum, wavelengths, args)
    boundary_terms = np.array(
        [
            boundary_violation(params, scaler, margin_frac=args.boundary_margin_frac)
            for params in generated_params
        ],
        dtype=float,
    )
    qw_terms = np.array(
        [
            quarter_wave_violation(params, reference_wl, tmm_solver)
            for params in generated_params
        ],
        dtype=float,
    )
    scores = surrogate_errors.copy()
    scores += args.prefilter_boundary_weight * boundary_terms
    scores += args.prefilter_qw_weight * qw_terms
    missing_terms = zeros.copy()
    false_terms = zeros.copy()
    peak_shift_terms = np.full_like(surrogate_errors, np.nan, dtype=float)
    if args.prefilter_objective == "surrogate_peak":
        for i, pred_spectrum in enumerate(surrogate_spectra):
            metrics = spectral_metrics(
                target_spectrum,
                pred_spectrum,
                wavelengths,
                peak_height=args.peak_height,
                peak_prominence=args.peak_prominence,
                peak_distance_threshold_nm=args.peak_distance_threshold,
            )
            missing_terms[i] = metrics["missing_peak_count"]
            false_terms[i] = metrics["false_peak_count"]
            peak_shift_terms[i] = metrics["mean_peak_shift"]
        peak_shift_score = np.where(
            np.isfinite(peak_shift_terms),
            peak_shift_terms / max(args.peak_distance_threshold, 1e-6),
            1.0,
        )
        scores += args.prefilter_missing_peak_weight * missing_terms
        scores += args.prefilter_false_peak_weight * false_terms
        scores += args.prefilter_peak_shift_weight * peak_shift_score

    invalid = np.zeros_like(scores, dtype=bool)
    if args.max_prefilter_boundary is not None:
        invalid |= boundary_terms > args.max_prefilter_boundary
    if args.max_prefilter_qw is not None:
        invalid |= qw_terms > args.max_prefilter_qw
    if args.max_prefilter_missing_peaks is not None:
        invalid |= missing_terms > args.max_prefilter_missing_peaks
    if args.max_prefilter_false_peaks is not None:
        invalid |= false_terms > args.max_prefilter_false_peaks
    scores = scores.copy()
    scores[invalid] = np.inf
    prefilter.update(
        {
            "scores": scores,
            "boundary": boundary_terms,
            "qw": qw_terms,
            "missing_peaks": missing_terms,
            "false_peaks": false_terms,
            "peak_shift": peak_shift_terms,
        }
    )
    return prefilter


def physics_rerank_score(metrics, params, scaler, args):
    peak_shift = metrics["mean_peak_shift"]
    if not np.isfinite(peak_shift):
        peak_shift_term = 1.0
    else:
        peak_shift_term = peak_shift / max(args.peak_distance_threshold, 1e-6)

    boundary_term = boundary_violation(
        params,
        scaler,
        margin_frac=args.boundary_margin_frac,
    )
    return float(
        metrics["mse"]
        + args.missing_peak_penalty * metrics["missing_peak_count"]
        + args.false_peak_penalty * metrics["false_peak_count"]
        + args.peak_shift_penalty * peak_shift_term
        + args.boundary_penalty * boundary_term
    )


def finite_mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def robustness_success(metrics, args):
    return nominal_valid(
        metrics,
        mse_threshold=args.robustness_success_mse_threshold,
        peak_shift_threshold=args.robustness_success_peak_shift_threshold,
        missing_peak_threshold=args.robustness_success_missing_peak_threshold,
        false_peak_threshold=args.robustness_success_false_peak_threshold,
    )


def robustness_nominal_valid(row, args):
    return nominal_valid(
        row,
        mse_threshold=args.robustness_success_mse_threshold,
        peak_shift_threshold=args.robustness_success_peak_shift_threshold,
        missing_peak_threshold=args.robustness_success_missing_peak_threshold,
        false_peak_threshold=args.robustness_success_false_peak_threshold,
    )


def robustness_summary(params, target_spectrum, wavelengths, tmm_solver, args, rng_seed):
    if args.robustness_trials <= 0:
        return {
            "robustness_trials": 0,
            "robustness_yield": np.nan,
            "robustness_mse_mean": np.nan,
            "robustness_peak_shift_mean": np.nan,
            "robustness_missing_peak_mean": np.nan,
            "robustness_false_peak_mean": np.nan,
        }

    rng = np.random.default_rng(rng_seed)
    trial_metrics = []
    successes = []
    for _ in range(args.robustness_trials):
        perturbed_spectrum = tmm_solver.simulate_perturbed(
            params,
            rng,
            thickness_sigma_nm=args.robustness_thickness_sigma,
            index_sigma_frac=args.robustness_index_sigma_frac,
        )
        metrics = spectral_metrics(
            target_spectrum,
            perturbed_spectrum,
            wavelengths,
            peak_height=args.peak_height,
            peak_prominence=args.peak_prominence,
            peak_distance_threshold_nm=args.peak_distance_threshold,
        )
        trial_metrics.append(metrics)
        successes.append(robustness_success(metrics, args))

    return {
        "robustness_trials": int(args.robustness_trials),
        "robustness_yield": float(np.mean(successes)),
        "robustness_mse_mean": float(np.mean([m["mse"] for m in trial_metrics])),
        "robustness_peak_shift_mean": finite_mean_or_nan(
            [m["mean_peak_shift"] for m in trial_metrics]
        ),
        "robustness_missing_peak_mean": float(
            np.mean([m["missing_peak_count"] for m in trial_metrics])
        ),
        "robustness_false_peak_mean": float(
            np.mean([m["false_peak_count"] for m in trial_metrics])
        ),
    }


def robustness_rerank_score(row, args):
    if row.get("robustness_trials", 0) <= 0 or not np.isfinite(row["robustness_yield"]):
        return np.inf
    if not row.get("robustness_nominal_valid", False) and not args.allow_robustness_nominal_invalid:
        return np.inf
    peak_shift = row["robustness_peak_shift_mean"]
    if not np.isfinite(peak_shift):
        peak_shift_term = 1.0
    else:
        peak_shift_term = peak_shift / max(args.peak_distance_threshold, 1e-6)

    return float(
        row["rerank_score"]
        + args.robustness_yield_penalty * (1.0 - row["robustness_yield"])
        + args.robustness_mse_penalty * row["robustness_mse_mean"]
        + args.robustness_peak_shift_penalty * peak_shift_term
        + args.robustness_missing_peak_penalty * row["robustness_missing_peak_mean"]
        + args.robustness_false_peak_penalty * row["robustness_false_peak_mean"]
    )


def refine_lc_with_tmm(params, target_spectrum, wavelengths, tmm_solver, scaler, args):
    if not args.refine_lc:
        tmm_spectrum = tmm_solver.simulate(params)
        metrics = spectral_metrics(
            target_spectrum,
            tmm_spectrum,
            wavelengths,
            peak_height=args.peak_height,
            peak_prominence=args.peak_prominence,
            peak_distance_threshold_nm=args.peak_distance_threshold,
        )
        return params, metrics, physics_rerank_score(metrics, params, scaler, args)

    params = np.asarray(params, dtype=float)
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        lc_min = float(scaler.data_min_[3])
        lc_max = float(scaler.data_max_[3])
    else:
        lc_min = 500.0
        lc_max = 5000.0

    center_lc = float(params[3])
    candidates = np.linspace(
        max(lc_min, center_lc - args.lc_refine_span),
        min(lc_max, center_lc + args.lc_refine_span),
        max(args.lc_refine_points, 1),
    )
    best_params = None
    best_metrics = None
    best_score = None
    for lc in candidates:
        trial_params = params.copy()
        trial_params[3] = lc
        tmm_spectrum = tmm_solver.simulate(trial_params)
        metrics = spectral_metrics(
            target_spectrum,
            tmm_spectrum,
            wavelengths,
            peak_height=args.peak_height,
            peak_prominence=args.peak_prominence,
            peak_distance_threshold_nm=args.peak_distance_threshold,
        )
        score = physics_rerank_score(metrics, trial_params, scaler, args)
        if best_score is None or score < best_score:
            best_params = trial_params
            best_metrics = metrics
            best_score = score
    return best_params, best_metrics, float(best_score)


def unique_append(selected, candidates, limit):
    seen = set(int(index) for index in selected)
    for index in candidates:
        index = int(index)
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)
        if len(selected) >= limit:
            break
    return selected


def farthest_parameter_indices(params, candidate_indices, quota, seed_indices=None):
    candidate_indices = [int(index) for index in candidate_indices]
    seed_indices = [int(index) for index in (seed_indices or [])]
    candidate_indices = [index for index in candidate_indices if index not in set(seed_indices)]
    if quota <= 0 or not candidate_indices:
        return []
    if len(candidate_indices) <= quota:
        return candidate_indices

    all_indices = candidate_indices + seed_indices
    all_params = np.asarray(params[all_indices], dtype=float)
    scale = np.maximum(np.std(all_params, axis=0), 1.0)
    candidate_params = np.asarray(params[candidate_indices], dtype=float)
    norm_params = candidate_params / scale
    if seed_indices:
        seed_params = np.asarray(params[seed_indices], dtype=float) / scale
        distances = np.sqrt(
            np.sum((norm_params[:, None, :] - seed_params[None, :, :]) ** 2, axis=2)
        )
        min_dist = np.min(distances, axis=1)
        selected_local = []
    else:
        selected_local = [0]
        min_dist = np.full(len(candidate_indices), np.inf, dtype=float)

    while len(selected_local) < quota:
        if selected_local:
            last = norm_params[selected_local[-1]]
            distances = np.sqrt(np.sum((norm_params - last[None, :]) ** 2, axis=1))
            min_dist = np.minimum(min_dist, distances)
            min_dist[selected_local] = -np.inf
        next_local = int(np.argmax(min_dist))
        if min_dist[next_local] == -np.inf:
            break
        selected_local.append(next_local)

    return [candidate_indices[i] for i in selected_local]


def n_diverse_indices(params, ordered_indices, quota):
    if quota <= 0:
        return []
    buckets = {}
    for index in ordered_indices:
        index = int(index)
        n_value = int(round(params[index, 2]))
        buckets.setdefault(n_value, []).append(index)
    selected = []
    while len(selected) < quota and buckets:
        for n_value in sorted(list(buckets.keys())):
            if not buckets[n_value]:
                del buckets[n_value]
                continue
            selected.append(buckets[n_value].pop(0))
            if len(selected) >= quota:
                break
    return selected


def select_tmm_candidate_indices(
    generated_params,
    surrogate_errors,
    prefilter_scores,
    args,
):
    finite_prefilter_order = np.argsort(prefilter_scores)
    finite_prefilter_order = finite_prefilter_order[
        np.isfinite(prefilter_scores[finite_prefilter_order])
    ]
    if len(finite_prefilter_order) == 0:
        finite_prefilter_order = np.argsort(surrogate_errors)

    top_k = min(args.tmm_top_k, len(finite_prefilter_order))
    if args.tmm_selection == "single_queue":
        return finite_prefilter_order[:top_k], {
            "mse": 0,
            "peak": int(top_k),
            "diverse": 0,
            "fill": 0,
        }

    mse_quota = args.mq_mse_quota
    peak_quota = args.mq_peak_quota
    diverse_quota = args.mq_diverse_quota
    n_diverse_quota = args.mq_n_diverse_quota
    if mse_quota <= 0 and peak_quota <= 0 and diverse_quota <= 0 and n_diverse_quota <= 0:
        mse_quota = int(np.ceil(top_k / 4))
        peak_quota = int(np.ceil(top_k / 4))
        n_diverse_quota = int(np.ceil(top_k / 4))
        diverse_quota = max(top_k - mse_quota - peak_quota - n_diverse_quota, 0)

    selected = []
    queue_counts = {"mse": 0, "peak": 0, "n_diverse": 0, "diverse": 0, "fill": 0}

    before = len(selected)
    selected = unique_append(selected, np.argsort(surrogate_errors), min(mse_quota, top_k))
    queue_counts["mse"] = len(selected) - before

    before = len(selected)
    selected = unique_append(selected, finite_prefilter_order, min(len(selected) + peak_quota, top_k))
    queue_counts["peak"] = len(selected) - before

    if len(selected) < top_k and n_diverse_quota > 0:
        n_indices = n_diverse_indices(
            generated_params,
            finite_prefilter_order,
            n_diverse_quota,
        )
        before = len(selected)
        selected = unique_append(
            selected,
            n_indices,
            min(len(selected) + n_diverse_quota, top_k),
        )
        queue_counts["n_diverse"] = len(selected) - before

    if len(selected) < top_k and diverse_quota > 0:
        pool_size = min(
            len(np.argsort(surrogate_errors)),
            max(diverse_quota * max(args.mq_diverse_pool_multiplier, 1), diverse_quota),
        )
        diverse_pool = np.argsort(surrogate_errors)[:pool_size]
        diverse_indices = farthest_parameter_indices(
            generated_params,
            diverse_pool,
            diverse_quota,
            seed_indices=selected,
        )
        before = len(selected)
        selected = unique_append(
            selected,
            diverse_indices,
            min(len(selected) + diverse_quota, top_k),
        )
        queue_counts["diverse"] = len(selected) - before

    if len(selected) < top_k:
        before = len(selected)
        selected = unique_append(selected, finite_prefilter_order, top_k)
        queue_counts["fill"] = len(selected) - before
    if len(selected) < top_k:
        before = len(selected)
        selected = unique_append(selected, np.argsort(surrogate_errors), top_k)
        queue_counts["fill"] += len(selected) - before

    return np.asarray(selected[:top_k], dtype=int), queue_counts


def evaluate_target(
    target_index,
    target_spectrum,
    gt_params,
    wavelengths,
    cvae,
    surrogate,
    scaler,
    tmm_solver,
    args,
):
    cond = torch.tensor(target_spectrum, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        generated_norm = cvae.generate(cond, num_samples=args.num_candidates)
        surrogate_spectra = surrogate(generated_norm).cpu().numpy()
        generated_norm_np = generated_norm.cpu().numpy()

    generated_params = inverse_transform_params(scaler, generated_norm_np)
    physics_params = generate_physics_prior_candidates(
        target_spectrum, wavelengths, tmm_solver, scaler, args
    )
    if len(physics_params):
        generated_params = np.concatenate([generated_params, physics_params], axis=0)
        physics_norm = scaler.transform(physics_params)
        with torch.no_grad():
            physics_spectra = surrogate(
                torch.tensor(physics_norm, dtype=torch.float32, device=device)
            ).cpu().numpy()
        surrogate_spectra = np.concatenate([surrogate_spectra, physics_spectra], axis=0)
    surrogate_errors = np.mean((surrogate_spectra - target_spectrum[None, :]) ** 2, axis=1)
    prefilter = surrogate_prefilter_scores(
        surrogate_errors,
        surrogate_spectra,
        generated_params,
        target_spectrum,
        wavelengths,
        tmm_solver,
        scaler,
        args,
    )
    prefilter_scores = prefilter["scores"]
    surrogate_order = np.argsort(prefilter_scores)
    surrogate_order = surrogate_order[np.isfinite(prefilter_scores[surrogate_order])]
    if len(surrogate_order) == 0:
        surrogate_order = np.argsort(surrogate_errors)
    if args.physics_tmm_quota > 0 and len(physics_params):
        cvae_order = surrogate_order[surrogate_order < args.num_candidates]
        physics_order = surrogate_order[surrogate_order >= args.num_candidates]
        physics_count = min(args.physics_tmm_quota, len(physics_order), args.tmm_top_k)
        cvae_count = max(min(args.tmm_top_k - physics_count, len(cvae_order)), 0)
        top_indices = np.concatenate(
            [cvae_order[:cvae_count], physics_order[:physics_count]]
        )
        queue_counts = {
            "mse": int(cvae_count),
            "peak": 0,
            "diverse": 0,
            "fill": 0,
            "physics_prior": int(physics_count),
        }
    else:
        top_indices, queue_counts = select_tmm_candidate_indices(
            generated_params,
            surrogate_errors,
            prefilter_scores,
            args,
        )
    top_params = generated_params[top_indices]
    diversity_summary = summarize_candidate_diversity(top_params, scaler=scaler)

    candidate_rows = []
    for rank, candidate_index in enumerate(top_indices):
        original_params = generated_params[candidate_index]
        params, metrics, rerank_score = refine_lc_with_tmm(
            original_params,
            target_spectrum,
            wavelengths,
            tmm_solver,
            scaler,
            args,
        )
        boundary_term = boundary_violation(
            params, scaler, margin_frac=args.boundary_margin_frac
        )
        row = {
            "target_index": int(target_index),
            "candidate_index": int(candidate_index),
            "candidate_source": "physics_prior"
            if candidate_index >= args.num_candidates
            else "cvae",
            "surrogate_rank": int(rank),
            "surrogate_mse": float(surrogate_errors[candidate_index]),
            "prefilter_score": float(prefilter_scores[candidate_index]),
            "prefilter_boundary": float(prefilter["boundary"][candidate_index]),
            "prefilter_qw": float(prefilter["qw"][candidate_index]),
            "prefilter_missing_peaks": float(
                prefilter["missing_peaks"][candidate_index]
            ),
            "prefilter_false_peaks": float(
                prefilter["false_peaks"][candidate_index]
            ),
            "prefilter_peak_shift": float(prefilter["peak_shift"][candidate_index])
            if np.isfinite(prefilter["peak_shift"][candidate_index])
            else np.nan,
            "rerank_score": rerank_score,
            "boundary_violation": boundary_term,
            "original_L_c": float(original_params[3]),
            "lc_refined": bool(args.refine_lc),
            "d_H": float(params[0]),
            "d_L": float(params[1]),
            "N": int(params[2]),
            "L_c": float(params[3]),
            **metrics,
            "robustness_trials": 0,
            "robustness_yield": np.nan,
            "robustness_mse_mean": np.nan,
            "robustness_peak_shift_mean": np.nan,
            "robustness_missing_peak_mean": np.nan,
            "robustness_false_peak_mean": np.nan,
            "robustness_nominal_valid": False,
            "robustness_score": np.inf,
        }
        row["robustness_nominal_valid"] = robustness_nominal_valid(row, args)
        candidate_rows.append(row)

    if args.robustness_trials > 0:
        nominal_key = "mse" if args.rerank_objective == "mse" else "rerank_score"
        robust_count = len(candidate_rows)
        if args.robustness_top_m > 0:
            robust_count = min(args.robustness_top_m, robust_count)
        robust_rows = sorted(candidate_rows, key=lambda row: row[nominal_key])[:robust_count]
        for row in robust_rows:
            params = np.array([row["d_H"], row["d_L"], row["N"], row["L_c"]], dtype=float)
            rng_seed = (
                args.seed
                + args.robustness_seed_offset
                + int(target_index) * 1009
                + int(row["candidate_index"]) * 9176
            )
            row.update(
                robustness_summary(
                    params,
                    target_spectrum,
                    wavelengths,
                    tmm_solver,
                    args,
                    rng_seed,
                )
            )
            row["robustness_score"] = robustness_rerank_score(row, args)

    if args.rerank_objective == "mse":
        best_key = "mse"
    elif args.rerank_objective == "physics_score":
        best_key = "rerank_score"
    else:
        best_key = "robustness_score"
    finite_rows = [row for row in candidate_rows if np.isfinite(row[best_key])]
    best_selection_used = args.rerank_objective
    if finite_rows:
        best_row = min(finite_rows, key=lambda row: row[best_key])
    else:
        fallback_key = "rerank_score" if args.rerank_objective == "robustness_score" else "mse"
        best_row = min(candidate_rows, key=lambda row: row[fallback_key])
        best_selection_used = (
            "physics_score_fallback"
            if args.rerank_objective == "robustness_score"
            else "mse_fallback"
        )
    mse_best_row = min(candidate_rows, key=lambda row: row["mse"])
    physics_best_row = min(candidate_rows, key=lambda row: row["rerank_score"])
    nominal_valid_rows = [
        row for row in candidate_rows if robustness_nominal_valid(row, args)
    ]
    robust_evaluated_rows = [
        row for row in candidate_rows if int(row.get("robustness_trials", 0)) > 0
    ]
    robust_evaluated_nominal_valid_rows = [
        row for row in robust_evaluated_rows if row.get("robustness_nominal_valid", False)
    ]

    gt_tmm_spectrum = tmm_solver.simulate(gt_params)
    gt_metrics = spectral_metrics(
        target_spectrum,
        gt_tmm_spectrum,
        wavelengths,
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
        peak_distance_threshold_nm=args.peak_distance_threshold,
    )

    summary = {
        "target_index": int(target_index),
        "gt_d_H": float(gt_params[0]),
        "gt_d_L": float(gt_params[1]),
        "gt_N": int(round(gt_params[2])),
        "gt_L_c": float(gt_params[3]),
        "gt_tmm_mse": gt_metrics["mse"],
        **diversity_summary,
        "nominal_valid_candidate_count": int(len(nominal_valid_rows)),
        "nominal_valid_candidate_fraction": float(len(nominal_valid_rows) / max(len(candidate_rows), 1)),
        "robust_evaluated_candidate_count": int(len(robust_evaluated_rows)),
        "robust_evaluated_nominal_valid_count": int(
            len(robust_evaluated_nominal_valid_rows)
        ),
        "tmm_selection_queue_counts": queue_counts,
        "best_selection_used": best_selection_used,
        "best_candidate": best_row,
        "mse_best_candidate": mse_best_row,
        "physics_best_candidate": physics_best_row,
    }
    return summary, candidate_rows


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(output_dir, prefix, target_spectrum, wavelengths, best_params, tmm_solver, target_index):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    best_spectrum = tmm_solver.simulate(best_params)
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, target_spectrum, "k-", linewidth=2.5, label="Target")
    plt.plot(wavelengths, best_spectrum, "r--", linewidth=2.0, label="Best TMM candidate")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(f"Target {target_index}: CVAE best-of-K with TMM reranking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = Path(output_dir) / f"{prefix}_target_{target_index}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    data = np.load(args.data_path)
    mask = build_dataset_mask(data, args.min_peak_count, args.strategy_ids)
    if not np.any(mask):
        raise ValueError("Dataset filtering removed all samples.")
    global_indices = np.flatnonzero(mask)
    spectra = data["spectra"][mask].astype(np.float32)
    params = data["params"][mask].astype(np.float32)
    wavelengths = data["wavelengths"].astype(np.float32)

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    cvae, surrogate = load_models(args, spectra_dim=spectra.shape[1])
    tmm_solver = FPDBRTMM(
        wavelengths,
        ge_file=args.ge_file,
        sio2_file=args.sio2_file,
        material_wavelength_unit=args.material_wavelength_unit,
        substrate_index=args.substrate_index,
    )

    indices = choose_indices(len(spectra), args, global_indices=global_indices)
    all_summaries = []
    all_candidate_rows = []

    print(f"Evaluating {len(indices)} targets on {device}")
    print(f"Samples after filtering: {len(spectra)}/{len(data['params'])}")
    print(
        f"Candidates per target: {args.num_candidates}; "
        f"TMM rerank top K: {args.tmm_top_k}"
    )

    for local_target_index in indices:
        target_index = int(global_indices[local_target_index])
        summary, candidate_rows = evaluate_target(
            target_index,
            spectra[local_target_index],
            params[local_target_index],
            wavelengths,
            cvae,
            surrogate,
            scaler,
            tmm_solver,
            args,
        )
        all_summaries.append(summary)
        all_candidate_rows.extend(candidate_rows)

        best = summary["best_candidate"]
        print(
            f"target={target_index} best_tmm_mse={best['mse']:.6f} "
            f"peak_shift={best['mean_peak_shift']} params="
            f"[{best['d_H']:.1f}, {best['d_L']:.1f}, {best['N']}, {best['L_c']:.1f}]"
        )

        if args.plot:
            maybe_plot(
                args.output_dir,
                args.output_prefix,
                spectra[local_target_index],
                wavelengths,
                np.array([best["d_H"], best["d_L"], best["N"], best["L_c"]]),
                tmm_solver,
                target_index,
            )

    best_mse_values = [item["best_candidate"]["mse"] for item in all_summaries]
    best_robust_yields = [
        item["best_candidate"]["robustness_yield"]
        for item in all_summaries
        if np.isfinite(item["best_candidate"].get("robustness_yield", np.nan))
    ]
    best_robust_mse = [
        item["best_candidate"]["robustness_mse_mean"]
        for item in all_summaries
        if np.isfinite(item["best_candidate"].get("robustness_mse_mean", np.nan))
    ]
    best_nominal_valid = [
        bool(item["best_candidate"].get("robustness_nominal_valid", False))
        for item in all_summaries
    ]
    nominal_valid_counts = [
        item.get("nominal_valid_candidate_count", 0) for item in all_summaries
    ]
    robust_eval_valid_counts = [
        item.get("robust_evaluated_nominal_valid_count", 0) for item in all_summaries
    ]
    queue_count_keys = ["mse", "peak", "n_diverse", "diverse", "fill", "physics_prior"]
    queue_count_means = {}
    for key in queue_count_keys:
        values = [
            item.get("tmm_selection_queue_counts", {}).get(key, 0)
            for item in all_summaries
        ]
        queue_count_means[key] = float(np.mean(values)) if values else 0.0
    selection_counts = {}
    for item in all_summaries:
        key = item.get("best_selection_used", "unknown")
        selection_counts[key] = selection_counts.get(key, 0) + 1
    aggregate = {
        "num_targets": int(len(all_summaries)),
        "num_candidates": int(args.num_candidates),
        "tmm_top_k": int(args.tmm_top_k),
        "tmm_selection": args.tmm_selection,
        "mq_mse_quota": args.mq_mse_quota,
        "mq_peak_quota": args.mq_peak_quota,
        "mq_diverse_quota": args.mq_diverse_quota,
        "mq_n_diverse_quota": args.mq_n_diverse_quota,
        "mq_diverse_pool_multiplier": args.mq_diverse_pool_multiplier,
        "seed": int(args.seed),
        "data_path": args.data_path,
        "samples_after_filtering": int(len(spectra)),
        "min_peak_count": args.min_peak_count,
        "strategy_ids": args.strategy_ids,
        "rerank_objective": args.rerank_objective,
        "missing_peak_penalty": args.missing_peak_penalty,
        "false_peak_penalty": args.false_peak_penalty,
        "peak_shift_penalty": args.peak_shift_penalty,
        "boundary_penalty": args.boundary_penalty,
        "boundary_margin_frac": args.boundary_margin_frac,
        "refine_lc": bool(args.refine_lc),
        "lc_refine_span": args.lc_refine_span,
        "lc_refine_points": args.lc_refine_points,
        "prefilter_objective": args.prefilter_objective,
        "prefilter_boundary_weight": args.prefilter_boundary_weight,
        "prefilter_qw_weight": args.prefilter_qw_weight,
        "prefilter_missing_peak_weight": args.prefilter_missing_peak_weight,
        "prefilter_false_peak_weight": args.prefilter_false_peak_weight,
        "prefilter_peak_shift_weight": args.prefilter_peak_shift_weight,
        "max_prefilter_boundary": args.max_prefilter_boundary,
        "max_prefilter_qw": args.max_prefilter_qw,
        "max_prefilter_missing_peaks": args.max_prefilter_missing_peaks,
        "max_prefilter_false_peaks": args.max_prefilter_false_peaks,
        "qw_target_source": args.qw_target_source,
        "num_physics_candidates": args.num_physics_candidates,
        "physics_tmm_quota": args.physics_tmm_quota,
        "physics_enumerate_n": bool(args.physics_enumerate_n),
        "physics_candidates_per_n": args.physics_candidates_per_n,
        "physics_thickness_jitter": args.physics_thickness_jitter,
        "physics_lc_jitter": args.physics_lc_jitter,
        "robustness_trials": args.robustness_trials,
        "robustness_top_m": args.robustness_top_m,
        "robustness_thickness_sigma": args.robustness_thickness_sigma,
        "robustness_index_sigma_frac": args.robustness_index_sigma_frac,
        "robustness_yield_penalty": args.robustness_yield_penalty,
        "robustness_mse_penalty": args.robustness_mse_penalty,
        "robustness_peak_shift_penalty": args.robustness_peak_shift_penalty,
        "robustness_missing_peak_penalty": args.robustness_missing_peak_penalty,
        "robustness_false_peak_penalty": args.robustness_false_peak_penalty,
        "robustness_success_mse_threshold": args.robustness_success_mse_threshold,
        "robustness_success_peak_shift_threshold": args.robustness_success_peak_shift_threshold,
        "robustness_success_missing_peak_threshold": args.robustness_success_missing_peak_threshold,
        "robustness_success_false_peak_threshold": args.robustness_success_false_peak_threshold,
        "evaluation_protocol": protocol_summary(),
        "allow_robustness_nominal_invalid": bool(args.allow_robustness_nominal_invalid),
        "best_tmm_mse_mean": float(np.mean(best_mse_values)),
        "best_tmm_mse_median": float(np.median(best_mse_values)),
        "best_tmm_mse_p90": float(np.percentile(best_mse_values, 90)),
        "best_robustness_yield_mean": float(np.mean(best_robust_yields))
        if best_robust_yields
        else None,
        "best_robustness_yield_median": float(np.median(best_robust_yields))
        if best_robust_yields
        else None,
        "best_robustness_mse_mean": float(np.mean(best_robust_mse))
        if best_robust_mse
        else None,
        "best_nominal_valid_rate": float(np.mean(best_nominal_valid))
        if best_nominal_valid
        else None,
        "nominal_valid_candidate_count_mean": float(np.mean(nominal_valid_counts))
        if nominal_valid_counts
        else None,
        "nominal_valid_candidate_count_median": float(np.median(nominal_valid_counts))
        if nominal_valid_counts
        else None,
        "targets_with_nominal_valid_candidate": int(
            np.sum(np.asarray(nominal_valid_counts) > 0)
        ),
        "robust_evaluated_nominal_valid_count_mean": float(
            np.mean(robust_eval_valid_counts)
        )
        if robust_eval_valid_counts
        else None,
        "tmm_selection_queue_count_means": queue_count_means,
        "best_selection_counts": selection_counts,
        "summaries": all_summaries,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / f"{args.output_prefix}.json", aggregate)
    write_csv(output_dir / f"{args.output_prefix}_candidates.csv", all_candidate_rows)

    print("Evaluation complete.")
    print(f"JSON: {output_dir / f'{args.output_prefix}.json'}")
    print(f"CSV:  {output_dir / f'{args.output_prefix}_candidates.csv'}")


if __name__ == "__main__":
    main()
