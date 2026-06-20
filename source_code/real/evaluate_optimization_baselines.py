import argparse
import csv
import time
from pathlib import Path

import numpy as np

from evaluation_protocol import (
    nominal_valid as protocol_nominal_valid,
    protocol_summary,
)
from physics_tmm import FPDBRTMM, add_tmm_args, dump_json, spectral_metrics


DEFAULT_BOUNDS = {
    "d_H": (100.0, 1000.0),
    "d_L": (100.0, 1500.0),
    "N": (2.0, 9.0),
    "L_c": (500.0, 5000.0),
}


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Traditional optimization baselines using real TMM calls."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(repo_root / "dataset" / "fp_dbr_data_100000_experiment.npz"),
    )
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target-indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-targets", type=int, default=None)
    parser.add_argument("--min-peak-count", type=int, default=None)
    parser.add_argument("--strategy-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["random", "de", "cma", "cma_internal"],
        default=["random"],
        help=(
            "random = random search, de = scipy differential evolution, "
            "cma = pycma bounded CMA-ES, cma_internal = dependency-free CMA-style fallback."
        ),
    )
    parser.add_argument("--random-samples", type=int, default=200)
    parser.add_argument("--de-maxiter", type=int, default=20)
    parser.add_argument("--de-popsize", type=int, default=8)
    parser.add_argument(
        "--no-de-polish",
        action="store_true",
        help="Disable scipy's final polish step to keep direct-TMM baselines budgeted.",
    )
    parser.add_argument("--cma-budget", type=int, default=400)
    parser.add_argument("--cma-popsize", type=int, default=8)
    parser.add_argument("--cma-restarts", type=int, default=3)
    parser.add_argument("--cma-sigma", type=float, default=0.28)
    parser.add_argument(
        "--objective",
        choices=["mse", "physics_score"],
        default="mse",
        help="Objective optimized by random search and differential evolution.",
    )
    parser.add_argument("--missing-peak-penalty", type=float, default=0.01)
    parser.add_argument("--false-peak-penalty", type=float, default=0.005)
    parser.add_argument("--peak-shift-penalty", type=float, default=0.005)
    parser.add_argument("--success-mse-threshold", type=float, default=0.01)
    parser.add_argument("--success-peak-shift-threshold", type=float, default=20.0)
    parser.add_argument("--success-missing-peak-threshold", type=int, default=0)
    parser.add_argument("--success-false-peak-threshold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "reports"))
    parser.add_argument("--output-prefix", type=str, default="optimization_baselines")
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--d-h-min", type=float, default=DEFAULT_BOUNDS["d_H"][0])
    parser.add_argument("--d-h-max", type=float, default=DEFAULT_BOUNDS["d_H"][1])
    parser.add_argument("--d-l-min", type=float, default=DEFAULT_BOUNDS["d_L"][0])
    parser.add_argument("--d-l-max", type=float, default=DEFAULT_BOUNDS["d_L"][1])
    parser.add_argument("--n-min", type=float, default=DEFAULT_BOUNDS["N"][0])
    parser.add_argument("--n-max", type=float, default=DEFAULT_BOUNDS["N"][1])
    parser.add_argument("--lc-min", type=float, default=DEFAULT_BOUNDS["L_c"][0])
    parser.add_argument("--lc-max", type=float, default=DEFAULT_BOUNDS["L_c"][1])
    add_tmm_args(parser)
    return parser.parse_args()


def get_bounds(args):
    return np.array(
        [
            [args.d_h_min, args.d_h_max],
            [args.d_l_min, args.d_l_max],
            [args.n_min, args.n_max],
            [args.lc_min, args.lc_max],
        ],
        dtype=float,
    )


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


def choose_target_indices(data, args):
    if args.target_indices:
        return np.array(args.target_indices, dtype=int)
    if args.num_targets is None:
        return np.array([args.target_index], dtype=int)
    mask = build_dataset_mask(data, args.min_peak_count, args.strategy_ids)
    global_indices = np.flatnonzero(mask)
    rng = np.random.default_rng(args.seed)
    return rng.choice(
        global_indices,
        size=min(args.num_targets, len(global_indices)),
        replace=False,
    )


def sanitize_params(params, bounds):
    params = np.asarray(params, dtype=float).copy()
    params = np.clip(params, bounds[:, 0], bounds[:, 1])
    params[2] = np.rint(params[2])
    params[2] = np.clip(params[2], bounds[2, 0], bounds[2, 1])
    return params


def physics_score_from_metrics(metrics, args):
    peak_shift = metrics["mean_peak_shift"]
    if metrics["target_peak_count"] == 0:
        peak_shift_term = 0.0
    elif np.isfinite(peak_shift):
        peak_shift_term = peak_shift / max(args.peak_distance_threshold, 1e-6)
    else:
        peak_shift_term = 1.0
    return float(
        metrics["mse"]
        + args.missing_peak_penalty * metrics["missing_peak_count"]
        + args.false_peak_penalty * metrics["false_peak_count"]
        + args.peak_shift_penalty * peak_shift_term
    )


def objective_value(tmm_solver, target_spectrum, wavelengths, params, args):
    spectrum = tmm_solver.simulate(params)
    if args.objective == "mse":
        mse = float(np.mean((spectrum - target_spectrum) ** 2))
        return mse, {"mse": mse}

    metrics = spectral_metrics(
        target_spectrum,
        spectrum,
        wavelengths,
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
        peak_distance_threshold_nm=args.peak_distance_threshold,
    )
    return physics_score_from_metrics(metrics, args), metrics


def nominal_valid(metrics, args):
    return protocol_nominal_valid(
        metrics,
        mse_threshold=args.success_mse_threshold,
        peak_shift_threshold=args.success_peak_shift_threshold,
        missing_peak_threshold=args.success_missing_peak_threshold,
        false_peak_threshold=args.success_false_peak_threshold,
    )


def objective_from_tmm(tmm_solver, target_spectrum, wavelengths, bounds, args):
    calls = {"count": 0}

    def objective(raw_params):
        params = sanitize_params(raw_params, bounds)
        score, _ = objective_value(tmm_solver, target_spectrum, wavelengths, params, args)
        calls["count"] += 1
        return score

    return objective, calls


def normalized_to_params(x, bounds):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


def run_random_search(tmm_solver, target_spectrum, wavelengths, bounds, args):
    rng = np.random.default_rng(args.seed)
    best_params = None
    best_score = float("inf")
    history = []
    start = time.time()

    for i in range(args.random_samples):
        raw = rng.uniform(bounds[:, 0], bounds[:, 1])
        params = sanitize_params(raw, bounds)
        score, metrics = objective_value(
            tmm_solver, target_spectrum, wavelengths, params, args
        )
        if score < best_score:
            best_score = score
            best_params = params
        history.append(
            {
                "method": "random",
                "objective": args.objective,
                "call": i + 1,
                "objective_score": score,
                "mse": metrics["mse"],
                "best_objective_score": best_score,
                "d_H": float(params[0]),
                "d_L": float(params[1]),
                "N": int(params[2]),
                "L_c": float(params[3]),
            }
        )

    return {
        "method": "random",
        "objective": args.objective,
        "best_params": best_params,
        "best_objective_score": best_score,
        "calls": args.random_samples,
        "elapsed_seconds": time.time() - start,
        "history": history,
    }


def run_random_search_with_seed(tmm_solver, target_spectrum, wavelengths, bounds, args, seed):
    old_seed = args.seed
    args.seed = seed
    try:
        return run_random_search(tmm_solver, target_spectrum, wavelengths, bounds, args)
    finally:
        args.seed = old_seed


def run_differential_evolution(tmm_solver, target_spectrum, wavelengths, bounds, args):
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise ImportError(
            "scipy is required for differential evolution baseline."
        ) from exc

    objective, calls = objective_from_tmm(
        tmm_solver, target_spectrum, wavelengths, bounds, args
    )
    start = time.time()
    result = differential_evolution(
        objective,
        bounds=[tuple(row) for row in bounds],
        maxiter=args.de_maxiter,
        popsize=args.de_popsize,
        seed=args.seed,
        polish=not args.no_de_polish,
        updating="immediate",
        workers=1,
        tol=0.01,
    )
    best_params = sanitize_params(result.x, bounds)
    best_score, _ = objective_value(
        tmm_solver, target_spectrum, wavelengths, best_params, args
    )
    calls["count"] += 1

    return {
        "method": "de",
        "objective": args.objective,
        "best_params": best_params,
        "best_objective_score": best_score,
        "calls": int(calls["count"]),
        "elapsed_seconds": time.time() - start,
        "success": bool(result.success),
        "message": str(result.message),
        "history": [],
    }


def run_internal_cma_es(tmm_solver, target_spectrum, wavelengths, bounds, args, seed):
    """Small dependency-free CMA-style implementation in normalized coordinates.

    The implementation follows the standard full-covariance CMA-ES update and
    clips sampled normalized parameters to [0, 1] before TMM evaluation. The
    integer period count is rounded by sanitize_params after mapping back to
    physical units.
    """
    rng = np.random.default_rng(seed)
    n_dim = 4
    lam = int(args.cma_popsize)
    if lam < 4:
        raise ValueError("--cma-popsize must be at least 4.")
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)

    cc = (4.0 + mu_eff / n_dim) / (n_dim + 4.0 + 2.0 * mu_eff / n_dim)
    cs = (mu_eff + 2.0) / (n_dim + mu_eff + 5.0)
    c1 = 2.0 / ((n_dim + 1.3) ** 2 + mu_eff)
    cmu = min(
        1.0 - c1,
        2.0
        * (mu_eff - 2.0 + 1.0 / mu_eff)
        / ((n_dim + 2.0) ** 2 + 2.0 * mu_eff / 2.0),
    )
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n_dim + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(n_dim) * (1.0 - 1.0 / (4.0 * n_dim) + 1.0 / (21.0 * n_dim**2))

    total_budget = int(args.cma_budget)
    restarts = int(args.cma_restarts)
    per_restart_budget = max(lam, total_budget // max(restarts, 1))
    best_params = None
    best_score = float("inf")
    best_restart = -1
    calls = 0
    start = time.time()
    histories = []

    for restart in range(restarts):
        xmean = rng.uniform(0.15, 0.85, size=n_dim)
        sigma = float(args.cma_sigma) * (0.75**restart)
        pc = np.zeros(n_dim)
        ps = np.zeros(n_dim)
        C = np.eye(n_dim)
        B = np.eye(n_dim)
        D = np.ones(n_dim)
        inv_sqrt_C = np.eye(n_dim)
        restart_calls = 0
        generation = 0

        while calls < total_budget and restart_calls < per_restart_budget:
            generation += 1
            arz = rng.normal(size=(lam, n_dim))
            ary = arz @ (B * D).T
            arx = np.clip(xmean + sigma * ary, 0.0, 1.0)
            fitness = np.zeros(lam, dtype=float)
            params_list = []
            for k in range(lam):
                params = sanitize_params(normalized_to_params(arx[k], bounds), bounds)
                score, _ = objective_value(
                    tmm_solver, target_spectrum, wavelengths, params, args
                )
                fitness[k] = score
                params_list.append(params)
                calls += 1
                restart_calls += 1
                if score < best_score:
                    best_score = float(score)
                    best_params = params
                    best_restart = restart
                if calls >= total_budget or restart_calls >= per_restart_budget:
                    break

            valid_count = len(fitness[: len(params_list)])
            order = np.argsort(fitness[:valid_count])
            selected_x = arx[order[:mu]]
            selected_y = ary[order[:mu]]
            old_xmean = xmean.copy()
            xmean = np.sum(weights[:, None] * selected_x, axis=0)
            y_w = np.sum(weights[:, None] * selected_y, axis=0)

            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * (
                inv_sqrt_C @ y_w
            )
            norm_ps = np.linalg.norm(ps)
            hsig = float(
                norm_ps
                / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * generation))
                / chi_n
                < (1.4 + 2.0 / (n_dim + 1.0))
            )
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mu_eff) * y_w

            rank_mu = np.zeros((n_dim, n_dim))
            for weight, y in zip(weights, selected_y):
                rank_mu += weight * np.outer(y, y)
            C = (
                (1.0 - c1 - cmu) * C
                + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
                + cmu * rank_mu
            )
            C = 0.5 * (C + C.T)
            sigma *= np.exp((cs / damps) * (norm_ps / chi_n - 1.0))

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
                inv_sqrt_C = B @ np.diag(1.0 / D) @ B.T
            except np.linalg.LinAlgError:
                C = np.eye(n_dim)
                B = np.eye(n_dim)
                D = np.ones(n_dim)
                inv_sqrt_C = np.eye(n_dim)

            histories.append(
                {
                    "method": "cma_internal",
                    "objective": args.objective,
                    "restart": restart + 1,
                    "generation": generation,
                    "call": calls,
                    "restart_call": restart_calls,
                    "generation_best_score": float(fitness[order[0]]),
                    "best_objective_score": float(best_score),
                    "sigma": float(sigma),
                    "xmean_0": float(xmean[0]),
                    "xmean_1": float(xmean[1]),
                    "xmean_2": float(xmean[2]),
                    "xmean_3": float(xmean[3]),
                    "old_xmean_0": float(old_xmean[0]),
                    "old_xmean_1": float(old_xmean[1]),
                    "old_xmean_2": float(old_xmean[2]),
                    "old_xmean_3": float(old_xmean[3]),
                }
            )

    return {
        "method": "cma_internal",
        "objective": args.objective,
        "best_params": best_params,
        "best_objective_score": best_score,
        "calls": int(calls),
        "elapsed_seconds": time.time() - start,
        "success": False,
        "message": (
            f"Budgeted bounded CMA-ES completed; restarts={restarts}, "
            f"popsize={lam}, best_restart={best_restart + 1}."
        ),
        "history": histories,
    }


def run_pycma(tmm_solver, target_spectrum, wavelengths, bounds, args, seed):
    try:
        import cma
    except ImportError as exc:
        raise ImportError(
            "The cma package is required for --methods cma. Install it with: pip install cma"
        ) from exc

    n_dim = 4
    lam = int(args.cma_popsize)
    total_budget = int(args.cma_budget)
    restarts = int(args.cma_restarts)
    per_restart_budget = max(lam, total_budget // max(restarts, 1))
    rng = np.random.default_rng(seed)
    best_params = None
    best_score = float("inf")
    best_restart = -1
    calls = 0
    histories = []
    start = time.time()

    for restart in range(restarts):
        x0 = rng.uniform(0.15, 0.85, size=n_dim)
        sigma = float(args.cma_sigma) * (0.75**restart)
        restart_calls = {"count": 0}

        def objective(x):
            nonlocal calls, best_params, best_score, best_restart
            x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
            params = sanitize_params(normalized_to_params(x, bounds), bounds)
            score, _ = objective_value(tmm_solver, target_spectrum, wavelengths, params, args)
            calls += 1
            restart_calls["count"] += 1
            if score < best_score:
                best_score = float(score)
                best_params = params
                best_restart = restart
            return float(score)

        es = cma.CMAEvolutionStrategy(
            x0,
            sigma,
            {
                "bounds": [0.0, 1.0],
                "popsize": lam,
                "maxfevals": per_restart_budget,
                "seed": int(seed + restart * 104729),
                "verbose": -9,
            },
        )
        generation = 0
        while (
            not es.stop()
            and calls < total_budget
            and restart_calls["count"] < per_restart_budget
        ):
            generation += 1
            xs = es.ask()
            remaining = min(
                len(xs),
                total_budget - calls,
                per_restart_budget - restart_calls["count"],
            )
            xs = xs[:remaining]
            scores = [objective(x) for x in xs]
            es.tell(xs, scores)
            histories.append(
                {
                    "method": "cma",
                    "objective": args.objective,
                    "restart": restart + 1,
                    "generation": generation,
                    "call": calls,
                    "restart_call": restart_calls["count"],
                    "generation_best_score": float(np.min(scores)) if scores else np.nan,
                    "best_objective_score": float(best_score),
                    "sigma": float(es.sigma),
                }
            )

    return {
        "method": "cma",
        "objective": args.objective,
        "best_params": best_params,
        "best_objective_score": best_score,
        "calls": int(calls),
        "elapsed_seconds": time.time() - start,
        "success": False,
        "message": (
            f"pycma bounded CMA-ES completed; restarts={restarts}, "
            f"popsize={lam}, best_restart={best_restart + 1}."
        ),
        "history": histories,
    }


def summarize_result(result, target_spectrum, wavelengths, tmm_solver, args):
    spectrum = tmm_solver.simulate(result["best_params"])
    metrics = spectral_metrics(
        target_spectrum,
        spectrum,
        wavelengths,
        peak_height=args.peak_height,
        peak_prominence=args.peak_prominence,
        peak_distance_threshold_nm=args.peak_distance_threshold,
    )
    params = result["best_params"]
    return {
        "method": result["method"],
        "target_index": result.get("target_index"),
        "calls": int(result["calls"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "d_H": float(params[0]),
        "d_L": float(params[1]),
        "N": int(params[2]),
        "L_c": float(params[3]),
        **metrics,
        "nominal_valid": nominal_valid(metrics, args),
        "objective": result.get("objective", args.objective),
        "objective_score": physics_score_from_metrics(metrics, args)
        if result.get("objective", args.objective) == "physics_score"
        else metrics["mse"],
        "optimizer_best_score": float(result["best_objective_score"]),
        "success": result.get("success", None),
        "message": result.get("message", ""),
    }


def aggregate_summaries(summaries):
    aggregate = {}
    for method in sorted(set(row["method"] for row in summaries)):
        rows = [row for row in summaries if row["method"] == method]
        mse_values = [row["mse"] for row in rows]
        valid = [bool(row.get("nominal_valid", False)) for row in rows]
        scores = [row["objective_score"] for row in rows]
        aggregate[method] = {
            "num_targets": int(len(rows)),
            "calls_mean": float(np.mean([row["calls"] for row in rows])),
            "objective": rows[0].get("objective"),
            "objective_score_mean": float(np.mean(scores)),
            "objective_score_median": float(np.median(scores)),
            "objective_score_p90": float(np.percentile(scores, 90)),
            "mse_mean": float(np.mean(mse_values)),
            "mse_median": float(np.median(mse_values)),
            "mse_p90": float(np.percentile(mse_values, 90)),
            "nominal_valid_targets": int(np.sum(valid)),
            "nominal_valid_rate": float(np.mean(valid)) if valid else 0.0,
            "mse_lt_0p01": int(np.sum(np.asarray(mse_values) < 0.01)),
            "mse_lt_0p005": int(np.sum(np.asarray(mse_values) < 0.005)),
        }
    return aggregate


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(args, target_spectrum, wavelengths, summaries, tmm_solver):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, target_spectrum, "k-", linewidth=2.5, label="Target")
    for summary in summaries:
        params = np.array([summary["d_H"], summary["d_L"], summary["N"], summary["L_c"]])
        spectrum = tmm_solver.simulate(params)
        plt.plot(
            wavelengths,
            spectrum,
            linestyle="--",
            linewidth=1.8,
            label=f"{summary['method']} mse={summary['mse']:.4g}",
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(f"Optimization baselines for target {args.target_index}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path = Path(args.output_dir) / f"{args.output_prefix}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    data = np.load(args.data_path)
    wavelengths = data["wavelengths"].astype(np.float32)
    spectra = data["spectra"].astype(np.float32)
    params = data["params"].astype(np.float32)

    target_indices = choose_target_indices(data, args)
    bad_indices = [
        int(index) for index in target_indices if index < 0 or index >= len(spectra)
    ]
    if bad_indices:
        raise IndexError(f"Target indices out of range: {bad_indices[:10]}")

    bounds = get_bounds(args)

    tmm_solver = FPDBRTMM(
        wavelengths,
        ge_file=args.ge_file,
        sio2_file=args.sio2_file,
        material_wavelength_unit=args.material_wavelength_unit,
        substrate_index=args.substrate_index,
    )

    summaries = []
    histories = []
    print(f"Evaluating {len(target_indices)} targets")
    for target_i, target_index in enumerate(target_indices):
        target_index = int(target_index)
        target_spectrum = spectra[target_index]
        gt_params = params[target_index]
        print(
            f"Target {target_index}: gt=[{gt_params[0]:.1f}, {gt_params[1]:.1f}, "
            f"{int(round(gt_params[2]))}, {gt_params[3]:.1f}]"
        )

        results = []
        if "random" in args.methods:
            print(f"  random search with {args.random_samples} TMM calls")
            result = run_random_search_with_seed(
                tmm_solver,
                target_spectrum,
                wavelengths,
                bounds,
                args,
                seed=args.seed + target_index * 1009,
            )
            result["target_index"] = target_index
            results.append(result)
            for row in result["history"]:
                row["target_index"] = target_index
            histories.extend(result["history"])

        if "de" in args.methods:
            print(
                "  differential evolution "
                f"(maxiter={args.de_maxiter}, popsize={args.de_popsize})"
            )
            result = run_differential_evolution(
                tmm_solver, target_spectrum, wavelengths, bounds, args
            )
            result["target_index"] = target_index
            results.append(result)

        if "cma" in args.methods:
            print(
                "  pycma bounded CMA-ES "
                f"(budget={args.cma_budget}, popsize={args.cma_popsize}, "
                f"restarts={args.cma_restarts})"
            )
            result = run_pycma(
                tmm_solver,
                target_spectrum,
                wavelengths,
                bounds,
                args,
                seed=args.seed + target_index * 9173,
            )
            result["target_index"] = target_index
            results.append(result)
            for row in result["history"]:
                row["target_index"] = target_index
            histories.extend(result["history"])

        if "cma_internal" in args.methods:
            print(
                "  internal bounded CMA-style ES "
                f"(budget={args.cma_budget}, popsize={args.cma_popsize}, "
                f"restarts={args.cma_restarts})"
            )
            result = run_internal_cma_es(
                tmm_solver,
                target_spectrum,
                wavelengths,
                bounds,
                args,
                seed=args.seed + target_index * 9173,
            )
            result["target_index"] = target_index
            results.append(result)
            for row in result["history"]:
                row["target_index"] = target_index
            histories.extend(result["history"])

        target_summaries = [
            summarize_result(result, target_spectrum, wavelengths, tmm_solver, args)
            for result in results
        ]
        summaries.extend(target_summaries)
        for summary in target_summaries:
            print(
                f"  {summary['method']}: mse={summary['mse']:.6f}, "
                f"score={summary['objective_score']:.6f}, "
                f"valid={summary['nominal_valid']}, "
                f"peak_shift={summary['mean_peak_shift']}, calls={summary['calls']}"
            )

    payload = {
        "target_indices": [int(index) for index in target_indices],
        "num_targets": int(len(target_indices)),
        "methods": args.methods,
        "objective": args.objective,
        "missing_peak_penalty": args.missing_peak_penalty,
        "false_peak_penalty": args.false_peak_penalty,
        "peak_shift_penalty": args.peak_shift_penalty,
        "success_mse_threshold": args.success_mse_threshold,
        "success_peak_shift_threshold": args.success_peak_shift_threshold,
        "success_missing_peak_threshold": args.success_missing_peak_threshold,
        "success_false_peak_threshold": args.success_false_peak_threshold,
        "evaluation_protocol": protocol_summary(),
        "random_samples": args.random_samples,
        "de_maxiter": args.de_maxiter,
        "de_popsize": args.de_popsize,
        "cma_budget": args.cma_budget,
        "cma_popsize": args.cma_popsize,
        "cma_restarts": args.cma_restarts,
        "cma_sigma": args.cma_sigma,
        "min_peak_count": args.min_peak_count,
        "strategy_ids": args.strategy_ids,
        "bounds": bounds.tolist(),
        "aggregate": aggregate_summaries(summaries),
        "summaries": summaries,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    csv_path = output_dir / f"{args.output_prefix}.csv"
    history_path = output_dir / f"{args.output_prefix}_history.csv"
    dump_json(json_path, payload)
    write_csv(csv_path, summaries)
    write_csv(history_path, histories)
    if args.plot and len(target_indices) == 1:
        target_index = int(target_indices[0])
        target_spectrum = spectra[target_index]
        maybe_plot(args, target_spectrum, wavelengths, summaries, tmm_solver)

    print("Optimization baseline evaluation complete.")
    print(f"Aggregate: {payload['aggregate']}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
