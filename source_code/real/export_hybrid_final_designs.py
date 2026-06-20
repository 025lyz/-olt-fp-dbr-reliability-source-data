import argparse
import csv
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Export final and fallback-used design CSVs from a hybrid summary CSV."
    )
    parser.add_argument(
        "--hybrid-csv",
        type=str,
        default=str(
            repo_root
            / "reports"
            / "inference_time_hybrid_teacher_replay_plus_residual_refine_50targets_failures17.csv"
        ),
    )
    parser.add_argument(
        "--final-designs-csv",
        type=str,
        default=str(repo_root / "reports" / "inference_time_hybrid_50targets_final_designs.csv"),
    )
    parser.add_argument(
        "--fallback-designs-csv",
        type=str,
        default=str(repo_root / "reports" / "inference_time_hybrid_50targets_fallback_used_designs.csv"),
    )
    return parser.parse_args()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def design_row(row, source=None):
    return {
        "target_index": row["target_index"],
        "d_H": row["final_d_H"],
        "d_L": row["final_d_L"],
        "N": row["final_N"],
        "L_c": row["final_L_c"],
        "mse": row["final_mse"],
        "rerank_score": row["final_mse"],
        "source": source or row["final_source"],
    }


def main():
    args = parse_args()
    with Path(args.hybrid_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    final_rows = [design_row(row) for row in rows]
    fallback_rows = [
        design_row(row, source="fallback_residual_refine")
        for row in rows
        if row.get("fallback_used") == "True"
    ]
    write_csv(args.final_designs_csv, final_rows)
    write_csv(args.fallback_designs_csv, fallback_rows)
    print(f"Final designs: {len(final_rows)} -> {args.final_designs_csv}")
    print(f"Fallback designs: {len(fallback_rows)} -> {args.fallback_designs_csv}")


if __name__ == "__main__":
    main()
