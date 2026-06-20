import csv
import json
from pathlib import Path

import numpy as np

from physics_tmm import FPDBRTMM, spectral_metrics


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATASET = ROOT / "dataset" / "fp_dbr_data_100000_physics_aware_experiment.npz"
SOURCE_TABLE = REPORTS / "olt_50target_source_data_table.csv"
FINAL_DESIGNS = REPORTS / "inference_time_hybrid_50targets_final_designs.csv"
OUTPUT_JSON = REPORTS / "olt_revision_evidence_summary.json"
THRESHOLD_CSV = REPORTS / "olt_threshold_sensitivity.csv"
FALLBACK_CSV = REPORTS / "olt_fallback_statistics.csv"
PHYSICS_CSV = REPORTS / "olt_physics_interpretation_metrics.csv"
SENSITIVITY_CSV = REPORTS / "olt_local_parameter_sensitivity.csv"
RUNTIME_BASELINE_CSV = REPORTS / "olt_runtime_and_direct_tmm_baseline.csv"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=np.nan):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_bool(value):
    return str(value).strip().lower() == "true"


def nominal_under(row, prefix, mse_threshold, peak_shift_threshold):
    mse = to_float(row[f"{prefix}_mse"])
    shift = to_float(row[f"{prefix}_peak_shift_nm"])
    missing = to_float(row[f"{prefix}_missing_peaks"])
    false = to_float(row[f"{prefix}_false_peaks"])
    peak_count = int(round(to_float(row["target_peak_count"], 0)))
    shift_ok = True if peak_count == 0 else np.isfinite(shift) and shift <= peak_shift_threshold
    return bool(
        np.isfinite(mse)
        and mse <= mse_threshold
        and shift_ok
        and np.isfinite(missing)
        and missing <= 0
        and np.isfinite(false)
        and false <= 0
    )


def threshold_sensitivity(rows):
    method_prefixes = [
        ("Baseline CVAE", "baseline"),
        ("Teacher replay", "teacher"),
        ("Hybrid workflow", "hybrid"),
    ]
    mse_thresholds = [0.005, 0.01, 0.02]
    shift_thresholds = [10.0, 20.0, 30.0]
    yield_thresholds = [0.5, 0.7]
    out = []
    for mse_thr in mse_thresholds:
        for shift_thr in shift_thresholds:
            for yield_thr in yield_thresholds:
                for method, prefix in method_prefixes:
                    nominal = [
                        nominal_under(row, prefix, mse_thr, shift_thr) for row in rows
                    ]
                    yields = [to_float(row[f"{prefix}_yield_r60"]) for row in rows]
                    robust = [
                        bool(nom and np.isfinite(yld) and yld >= yield_thr)
                        for nom, yld in zip(nominal, yields)
                    ]
                    out.append(
                        {
                            "method": method,
                            "mse_threshold": mse_thr,
                            "peak_shift_threshold_nm": shift_thr,
                            "yield_threshold": yield_thr,
                            "nominal_valid": int(np.sum(nominal)),
                            "robust_valid": int(np.sum(robust)),
                            "num_targets": len(rows),
                            "nominal_valid_rate": float(np.mean(nominal)),
                            "robust_valid_rate": float(np.mean(robust)),
                        }
                    )
    return out


def fallback_statistics(rows):
    total = len(rows)
    fallback_rows = [row for row in rows if to_bool(row["hybrid_fallback_used"])]
    primary_rows = [row for row in rows if not to_bool(row["hybrid_fallback_used"])]
    baseline_invalid = [row for row in rows if not to_bool(row["baseline_nominal_valid"])]
    repaired_vs_baseline = [
        row
        for row in rows
        if (not to_bool(row["baseline_nominal_valid"])) and to_bool(row["hybrid_nominal_valid"])
    ]
    fallback_repaired_vs_baseline = [
        row
        for row in fallback_rows
        if (not to_bool(row["baseline_nominal_valid"])) and to_bool(row["hybrid_nominal_valid"])
    ]
    fallback_robust = [row for row in fallback_rows if to_bool(row["hybrid_robust_valid"])]
    fallback_nominal = [row for row in fallback_rows if to_bool(row["hybrid_nominal_valid"])]
    unresolved = [row for row in rows if not to_bool(row["hybrid_nominal_valid"])]
    rows_out = [
        {
            "metric": "total_targets",
            "count": total,
            "denominator": total,
            "fraction": 1.0,
        },
        {
            "metric": "fallback_selected_final_designs",
            "count": len(fallback_rows),
            "denominator": total,
            "fraction": len(fallback_rows) / total,
        },
        {
            "metric": "primary_selected_final_designs",
            "count": len(primary_rows),
            "denominator": total,
            "fraction": len(primary_rows) / total,
        },
        {
            "metric": "baseline_nominal_invalid_targets",
            "count": len(baseline_invalid),
            "denominator": total,
            "fraction": len(baseline_invalid) / total,
        },
        {
            "metric": "targets_repaired_vs_baseline_to_hybrid_nominal_valid",
            "count": len(repaired_vs_baseline),
            "denominator": len(baseline_invalid),
            "fraction": len(repaired_vs_baseline) / max(len(baseline_invalid), 1),
        },
        {
            "metric": "fallback_repairs_vs_baseline_to_hybrid_nominal_valid",
            "count": len(fallback_repaired_vs_baseline),
            "denominator": len(baseline_invalid),
            "fraction": len(fallback_repaired_vs_baseline) / max(len(baseline_invalid), 1),
        },
        {
            "metric": "fallback_nominal_valid",
            "count": len(fallback_nominal),
            "denominator": len(fallback_rows),
            "fraction": len(fallback_nominal) / max(len(fallback_rows), 1),
        },
        {
            "metric": "fallback_robust_valid",
            "count": len(fallback_robust),
            "denominator": len(fallback_rows),
            "fraction": len(fallback_robust) / max(len(fallback_rows), 1),
        },
        {
            "metric": "hybrid_unresolved_nominal_invalid",
            "count": len(unresolved),
            "denominator": total,
            "fraction": len(unresolved) / total,
        },
    ]
    return rows_out


def material_real_at(tmm, wavelength_nm):
    n_h = float(np.interp(wavelength_nm, tmm.wavelengths_nm, np.real(tmm.n_h)))
    n_l = float(np.interp(wavelength_nm, tmm.wavelengths_nm, np.real(tmm.n_l)))
    n_c = float(np.interp(wavelength_nm, tmm.wavelengths_nm, np.real(tmm.n_cavity)))
    return n_h, n_l, n_c


def physics_metrics(rows, data, tmm):
    out = []
    for row in rows:
        target_index = int(float(row["target_index"]))
        target_wl = float(data["target_wavelengths"][target_index])
        n_h, n_l, n_c = material_real_at(tmm, target_wl)
        q_h = target_wl / (4.0 * n_h) if n_h > 0 else np.nan
        q_l = target_wl / (4.0 * n_l) if n_l > 0 else np.nan
        peak_count = int(round(to_float(row["target_peak_count"], 0)))
        gt_dh = to_float(row["gt_d_H_nm"])
        gt_dl = to_float(row["gt_d_L_nm"])
        gt_lc = to_float(row["gt_L_c_nm"])
        h_dh = to_float(row["hybrid_d_H_nm"])
        h_dl = to_float(row["hybrid_d_L_nm"])
        h_lc = to_float(row["hybrid_L_c_nm"])
        cavity_order = int(round(2.0 * n_c * h_lc / target_wl)) if target_wl > 0 else 0
        cavity_lc0 = cavity_order * target_wl / (2.0 * n_c) if n_c > 0 else np.nan
        bragg_center_h_nm = 4.0 * n_h * h_dh
        bragg_center_l_nm = 4.0 * n_l * h_dl
        bragg_center_mean_nm = 0.5 * (bragg_center_h_nm + bragg_center_l_nm)
        out.append(
            {
                "target_index": target_index,
                "target_wavelength_nm": target_wl,
                "target_peak_count": peak_count,
                "hybrid_fallback_used": row["hybrid_fallback_used"],
                "hybrid_nominal_valid": row["hybrid_nominal_valid"],
                "hybrid_robust_valid": row["hybrid_robust_valid"],
                "n_H_at_target": n_h,
                "n_L_at_target": n_l,
                "quarter_wave_d_H0_nm": q_h,
                "quarter_wave_d_L0_nm": q_l,
                "hybrid_delta_d_H_vs_quarter_wave_nm": h_dh - q_h,
                "hybrid_delta_d_L_vs_quarter_wave_nm": h_dl - q_l,
                "hybrid_rel_delta_d_H_vs_quarter_wave": (h_dh - q_h) / q_h if q_h else np.nan,
                "hybrid_rel_delta_d_L_vs_quarter_wave": (h_dl - q_l) / q_l if q_l else np.nan,
                "hybrid_cavity_order_nearest": cavity_order,
                "hybrid_cavity_lc0_for_nearest_order_nm": cavity_lc0,
                "hybrid_delta_L_c_vs_nearest_fp_order_nm": h_lc - cavity_lc0,
                "hybrid_cavity_optical_phase_order": 2.0 * n_c * h_lc / target_wl if target_wl else np.nan,
                "hybrid_bragg_center_from_H_nm": bragg_center_h_nm,
                "hybrid_bragg_center_from_L_nm": bragg_center_l_nm,
                "hybrid_bragg_center_mean_nm": bragg_center_mean_nm,
                "hybrid_bragg_center_offset_from_target_nm": bragg_center_mean_nm - target_wl,
                "delta_d_H_vs_dataset_nm": h_dh - gt_dh,
                "delta_d_L_vs_dataset_nm": h_dl - gt_dl,
                "delta_L_c_vs_dataset_nm": h_lc - gt_lc,
                "abs_delta_d_H_vs_dataset_nm": abs(h_dh - gt_dh),
                "abs_delta_d_L_vs_dataset_nm": abs(h_dl - gt_dl),
                "abs_delta_L_c_vs_dataset_nm": abs(h_lc - gt_lc),
                "relative_delta_d_H_vs_dataset": (h_dh - gt_dh) / gt_dh if gt_dh else np.nan,
                "relative_delta_d_L_vs_dataset": (h_dl - gt_dl) / gt_dl if gt_dl else np.nan,
                "relative_delta_L_c_vs_dataset": (h_lc - gt_lc) / gt_lc if gt_lc else np.nan,
                "hybrid_peak_shift_nm": to_float(row["hybrid_peak_shift_nm"]),
                "hybrid_yield_r60": to_float(row["hybrid_yield_r60"]),
                "baseline_peak_shift_nm": to_float(row["baseline_peak_shift_nm"]),
                "baseline_yield_r60": to_float(row["baseline_yield_r60"]),
            }
        )
    return out


def local_parameter_sensitivity(rows, data, tmm, delta_nm=5.0, max_targets=5):
    wavelengths = data["wavelengths"].astype(np.float32)
    spectra = data["spectra"].astype(np.float32)
    out = []
    selected_rows = rows[:max_targets]
    for row in selected_rows:
        target_index = int(float(row["target_index"]))
        target = spectra[target_index]
        base_params = np.array(
            [
                to_float(row["hybrid_d_H_nm"]),
                to_float(row["hybrid_d_L_nm"]),
                to_float(row["hybrid_N"]),
                to_float(row["hybrid_L_c_nm"]),
            ],
            dtype=float,
        )
        base_metrics = spectral_metrics(target, tmm.simulate(base_params), wavelengths)
        perturbations = [
            ("d_H", 0, delta_nm),
            ("d_L", 1, delta_nm),
            ("L_c", 3, delta_nm),
        ]
        n_value = int(round(base_params[2]))
        if n_value < 10:
            perturbations.append(("N", 2, 1.0))
        for parameter, index, delta in perturbations:
            params = base_params.copy()
            params[index] += delta
            if parameter == "N":
                params[index] = int(round(params[index]))
            metrics = spectral_metrics(target, tmm.simulate(params), wavelengths)
            base_shift = to_float(base_metrics["mean_peak_shift"])
            new_shift = to_float(metrics["mean_peak_shift"])
            out.append(
                {
                    "target_index": target_index,
                    "parameter": parameter,
                    "delta": delta,
                    "base_mse": base_metrics["mse"],
                    "perturbed_mse": metrics["mse"],
                    "delta_mse": metrics["mse"] - base_metrics["mse"],
                    "base_peak_shift_nm": base_shift,
                    "perturbed_peak_shift_nm": new_shift,
                    "delta_peak_shift_nm": new_shift - base_shift
                    if np.isfinite(base_shift) and np.isfinite(new_shift)
                    else np.nan,
                    "perturbed_missing_peaks": metrics["missing_peak_count"],
                    "perturbed_false_peaks": metrics["false_peak_count"],
                }
            )
    return out


def summarize_physics(rows):
    summary = {}
    for group_name, group_rows in [
        ("all", rows),
        ("fallback", [r for r in rows if str(r["hybrid_fallback_used"]).lower() == "true"]),
        ("primary", [r for r in rows if str(r["hybrid_fallback_used"]).lower() != "true"]),
    ]:
        if not group_rows:
            continue
        summary[group_name] = {"num_targets": len(group_rows)}
        for key in [
            "abs_delta_d_H_vs_dataset_nm",
            "abs_delta_d_L_vs_dataset_nm",
            "abs_delta_L_c_vs_dataset_nm",
            "hybrid_peak_shift_nm",
            "hybrid_yield_r60",
        ]:
            values = np.array([to_float(r[key]) for r in group_rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[group_name][f"{key}_mean"] = float(np.mean(values)) if values.size else None
            summary[group_name][f"{key}_median"] = float(np.median(values)) if values.size else None
    return summary


def summarize_sensitivity(rows):
    summary = {}
    for parameter in sorted(set(row["parameter"] for row in rows)):
        group = [row for row in rows if row["parameter"] == parameter]
        delta_mse = np.array([abs(to_float(row["delta_mse"])) for row in group], dtype=float)
        delta_shift = np.array(
            [abs(to_float(row["delta_peak_shift_nm"])) for row in group], dtype=float
        )
        delta_mse = delta_mse[np.isfinite(delta_mse)]
        delta_shift = delta_shift[np.isfinite(delta_shift)]
        summary[parameter] = {
            "num_perturbations": len(group),
            "abs_delta_mse_mean": float(np.mean(delta_mse)) if delta_mse.size else None,
            "abs_delta_mse_median": float(np.median(delta_mse)) if delta_mse.size else None,
            "abs_delta_peak_shift_nm_mean": float(np.mean(delta_shift)) if delta_shift.size else None,
            "abs_delta_peak_shift_nm_median": float(np.median(delta_shift)) if delta_shift.size else None,
        }
    return summary


def update_runtime_baseline_table():
    rows = read_csv(RUNTIME_BASELINE_CSV)
    method_names = {
        "Direct bounded CMA-ES TMM optimization, 20-target strong check",
        "Direct differential-evolution TMM optimization, 10-target budgeted check",
        "Direct pycma bounded CMA-ES TMM optimization, 20-target strong check",
    }
    rows = [row for row in rows if row.get("method") not in method_names]
    rows.extend(
        [
            {
                "method": "Direct differential-evolution TMM optimization, 10-target budgeted check",
                "targets": 10,
                "candidate_or_optimization_budget": "physics-score objective; maxiter=4, popsize=4, polish disabled",
                "surrogate_calls_per_target": 0,
                "real_tmm_calls_per_target_nominal_selection": 81,
                "additional_robustness_tmm_calls_per_final_design": "not evaluated",
                "fallback_triggered_targets": "not applicable",
                "wall_time_or_mean_seconds_per_target": "19.1 s per target (recorded run)",
                "nominal_valid": "0/10 = 0%",
                "robust_valid": "not evaluated",
                "mean_yield": "not evaluated",
                "source_files": "reports/optimization_de_10targets_physics_score_maxiter4_pop4_nopolish.json; reports/optimization_de_10targets_physics_score_maxiter4_pop4_nopolish.csv",
            },
            {
                "method": "Direct pycma bounded CMA-ES TMM optimization, 20-target strong check",
                "targets": 20,
                "candidate_or_optimization_budget": "physics-score objective; 4 restarts; population 8; normalized bounded search",
                "surrogate_calls_per_target": 0,
                "real_tmm_calls_per_target_nominal_selection": 320,
                "additional_robustness_tmm_calls_per_final_design": "not evaluated",
                "fallback_triggered_targets": "not applicable",
                "wall_time_or_mean_seconds_per_target": "80.5 s per target (recorded run)",
                "nominal_valid": "6/20 = 30%",
                "robust_valid": "not evaluated",
                "mean_yield": "not evaluated",
                "source_files": "reports/optimization_pycma_20targets_physics_score_320calls.json; reports/optimization_pycma_20targets_physics_score_320calls.csv",
            },
        ]
    )
    write_csv(RUNTIME_BASELINE_CSV, rows)


def main():
    data = np.load(DATASET)
    tmm = FPDBRTMM(data["wavelengths"].astype(np.float32))
    rows = read_csv(SOURCE_TABLE)
    threshold_rows = threshold_sensitivity(rows)
    fallback_rows = fallback_statistics(rows)
    physics_rows = physics_metrics(rows, data, tmm)
    sensitivity_rows = local_parameter_sensitivity(rows, data, tmm)
    physics_summary = summarize_physics(physics_rows)
    sensitivity_summary = summarize_sensitivity(sensitivity_rows)

    write_csv(THRESHOLD_CSV, threshold_rows)
    write_csv(FALLBACK_CSV, fallback_rows)
    write_csv(PHYSICS_CSV, physics_rows)
    write_csv(SENSITIVITY_CSV, sensitivity_rows)
    update_runtime_baseline_table()

    payload = {
        "dataset": str(DATASET),
        "source_table": str(SOURCE_TABLE),
        "threshold_sensitivity_csv": str(THRESHOLD_CSV),
        "fallback_statistics_csv": str(FALLBACK_CSV),
        "physics_interpretation_metrics_csv": str(PHYSICS_CSV),
        "local_parameter_sensitivity_csv": str(SENSITIVITY_CSV),
        "runtime_and_direct_tmm_baseline_csv": str(RUNTIME_BASELINE_CSV),
        "fallback_statistics": fallback_rows,
        "physics_summary": physics_summary,
        "local_parameter_sensitivity_summary": sensitivity_summary,
        "notes": [
            "Threshold sensitivity recomputes validity from stored nominal metrics and r60 yields.",
            "Physics metrics include quarter-wave and nearest Fabry-Perot order deviations evaluated at the dataset target wavelength.",
            "Dataset-design deltas compare final hybrid parameters with the synthetic design that generated each target spectrum; they are not a uniqueness claim.",
            "Local parameter sensitivity uses one-at-a-time perturbations around the final hybrid designs and real TMM reevaluation.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {THRESHOLD_CSV}")
    print(f"Wrote {FALLBACK_CSV}")
    print(f"Wrote {PHYSICS_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
