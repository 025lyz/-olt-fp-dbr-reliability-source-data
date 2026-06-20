import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation_protocol import robust_valid
from evaluation_protocol import nominal_valid as nominal_valid_check
from physics_tmm import dump_json


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Summarize nominal-valid and robust-valid counts from robustness JSON."
    )
    parser.add_argument(
        "--robustness-json",
        type=str,
        default=str(
            repo_root
            / "reports"
            / "inference_time_hybrid_50targets_final_robustness_r60.json"
        ),
    )
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "reports"))
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="inference_time_hybrid_50targets_final_robustness_r60_summary",
    )
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def main():
    args = parse_args()
    payload = json.loads(Path(args.robustness_json).read_text(encoding="utf-8"))
    rows = []
    for summary in payload.get("design_summaries", []):
        metrics = summary["nominal_metrics"]
        nominal_valid = nominal_valid_check(metrics)
        yield_rate = float(summary["yield_rate"])
        robust_is_valid = robust_valid(yield_rate, nominal_is_valid=nominal_valid)
        params = summary["params"]
        rows.append(
            {
                "target_index": int(summary["target_index"]),
                "source": summary["source"],
                "d_H": float(params["d_H"]),
                "d_L": float(params["d_L"]),
                "N": int(round(params["N"])),
                "L_c": float(params["L_c"]),
                "nominal_valid": nominal_valid,
                "robust_valid": robust_is_valid,
                "yield_rate": yield_rate,
                "nominal_mse": float(metrics["mse"]),
                "nominal_peak_shift": metrics["mean_peak_shift"],
                "nominal_missing_peaks": int(metrics["missing_peak_count"]),
                "nominal_false_peaks": int(metrics["false_peak_count"]),
                "robustness_mse_mean": summary["mse"]["mean"],
                "robustness_peak_shift_mean": summary["mean_peak_shift"]["mean"],
                "robustness_missing_peak_mean": summary["missing_peak_count"]["mean"],
                "robustness_false_peak_mean": summary["false_peak_count"]["mean"],
            }
        )

    nominal = [row["nominal_valid"] for row in rows]
    robust = [row["robust_valid"] for row in rows]
    yields = [row["yield_rate"] for row in rows]
    aggregate = {
        "robustness_json": args.robustness_json,
        "num_targets": int(len(rows)),
        "nominal_valid_targets": int(np.sum(nominal)),
        "robust_valid_targets": int(np.sum(robust)),
        "nominal_valid_rate": float(np.mean(nominal)) if rows else None,
        "robust_valid_rate": float(np.mean(robust)) if rows else None,
        "yield_rate": finite_summary(yields),
        "nominal_invalid_targets": [
            int(row["target_index"]) for row in rows if not row["nominal_valid"]
        ],
        "nominal_valid_low_yield_targets": [
            int(row["target_index"])
            for row in rows
            if row["nominal_valid"] and row["yield_rate"] < 0.5
        ],
        "rows": rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    csv_path = output_dir / f"{args.output_prefix}.csv"
    dump_json(json_path, aggregate)
    write_csv(csv_path, rows)
    print(
        f"nominal={aggregate['nominal_valid_targets']}/{aggregate['num_targets']} "
        f"robust={aggregate['robust_valid_targets']}/{aggregate['num_targets']} "
        f"yield_mean={aggregate['yield_rate']['mean']}"
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
