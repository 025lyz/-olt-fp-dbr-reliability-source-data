import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "olt_latex" / "figures"


def save_fig(fig, name: str) -> None:
    for directory in (REPORTS, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")


def read_material(path: Path):
    return np.loadtxt(path, comments="#")


def interp_material(path: Path, wavelengths_um):
    data = read_material(path)
    wl = data[:, 0]
    n = np.interp(wavelengths_um, wl, data[:, 1])
    k = np.interp(wavelengths_um, wl, data[:, 2])
    return n, k


def load_audit():
    with (REPORTS / "database_dataset_audit_summary.csv").open(newline="") as f:
        return {row["dataset"]: row for row in csv.DictReader(f)}


def load_summary(path):
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "nominal": payload["nominal_valid_targets"],
        "robust": payload["robust_valid_targets"],
        "yield": payload["yield_rate"]["mean"],
    }


def annotate_delta(ax, x0, x1, y, text, color, text_offset=0.025):
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.0)
    ax.plot([x0, x0], [y - 0.015, y + 0.015], color=color, linewidth=1.0)
    ax.plot([x1, x1], [y - 0.015, y + 0.015], color=color, linewidth=1.0)
    ax.text((x0 + x1) / 2, y + text_offset, text, ha="center", va="bottom", color=color, fontsize=7)


def main():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    wl_um = np.linspace(3.0, 5.0, 240)
    exp_ge_n, exp_ge_k = interp_material(ROOT / "dataset" / "expriment_Ge.txt", wl_um)
    db_ge_n, db_ge_k = interp_material(ROOT / "dataset" / "riinfo_Ge_Amotchkina_nk.txt", wl_um)
    exp_sio2_n, exp_sio2_k = interp_material(ROOT / "dataset" / "expriment_Sio2.txt", wl_um)
    db_sio2_n, db_sio2_k = interp_material(ROOT / "dataset" / "riinfo_SiO2_Kischkat_nk.txt", wl_um)

    audit = load_audit()
    perf = {
        "Sputtered\nbase.": load_summary(REPORTS / "baseline_strategy12_50targets_final_robustness_r60_summary.json"),
        "Sputtered\nhybrid": load_summary(REPORTS / "inference_time_hybrid_50targets_final_robustness_r60_summary.json"),
        "External\nbase.": load_summary(REPORTS / "database_baseline_50targets_final_robustness_r60_summary.json"),
        "External\nhybrid": load_summary(REPORTS / "database_hybrid_50targets_final_robustness_r60_summary.json"),
    }

    fig = plt.figure(figsize=(9.10, 5.35))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.08, 1.34, 1.22],
        height_ratios=[1.0, 0.95],
        wspace=0.66,
        hspace=0.72,
    )
    ax_n = fig.add_subplot(gs[0, 0])
    ax_k = fig.add_subplot(gs[1, 0], sharex=ax_n)
    ax_data = fig.add_subplot(gs[:, 1])
    ax_valid = fig.add_subplot(gs[0, 2])
    ax_yield = fig.add_subplot(gs[1, 2])

    colors = {
        "exp": "#126c77",
        "db": "#9a5c00",
        "ge": "#293241",
        "sio2": "#6b7280",
        "baseline": "#8a9aa3",
        "hybrid": "#1f7a8c",
        "accent": "#a54f3f",
    }

    ax_n.plot(wl_um, exp_ge_n, color=colors["exp"], linewidth=1.7, label="Ge, in-house")
    ax_n.plot(wl_um, db_ge_n, color=colors["db"], linewidth=1.7, linestyle="--", label="Ge, database")
    ax_n.plot(wl_um, exp_sio2_n, color="#5f9ea0", linewidth=1.5, label="SiO$_2$, in-house")
    ax_n.plot(wl_um, db_sio2_n, color="#c58b37", linewidth=1.5, linestyle="--", label="SiO$_2$, database")
    ax_n.set_ylabel("n")
    ax_n.set_title("a  Material constants", loc="left", fontweight="bold", fontsize=9)
    ax_n.text(4.98, db_ge_n[-1] + 0.015, "Ge database", color=colors["db"], fontsize=6.2, ha="right", va="bottom")
    ax_n.text(4.98, exp_ge_n[-1] - 0.015, "Ge in-house", color=colors["exp"], fontsize=6.2, ha="right", va="top")
    ax_n.text(4.98, db_sio2_n[-1] + 0.015, "SiO$_2$ database", color="#c58b37", fontsize=6.0, ha="right", va="bottom")
    ax_n.text(4.98, exp_sio2_n[-1] - 0.015, "SiO$_2$ in-house", color="#5f9ea0", fontsize=6.0, ha="right", va="top")
    ax_n.set_xlim(3.0, 5.0)
    ax_n.grid(axis="y", color="#e8edf0", linewidth=0.7)
    ax_n.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.28),
        ncol=2,
        fontsize=6.0,
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    ax_k.plot(wl_um, exp_ge_k, color=colors["exp"], linewidth=1.7)
    ax_k.plot(wl_um, db_ge_k, color=colors["db"], linewidth=1.7, linestyle="--")
    ax_k.plot(wl_um, exp_sio2_k, color="#5f9ea0", linewidth=1.5)
    ax_k.plot(wl_um, db_sio2_k, color="#c58b37", linewidth=1.5, linestyle="--")
    ax_k.set_xlabel("Wavelength (um)")
    ax_k.set_ylabel("k")
    ax_k.set_yscale("symlog", linthresh=1e-4)
    ax_k.grid(axis="y", color="#e8edf0", linewidth=0.7)

    metrics = [
        ("Mean\npeak count", "mean_peak_count"),
        ("Peak count\n>= 1", "frac_peak_count_ge_1"),
        ("Peak count\n>= 2", "frac_peak_count_ge_2"),
        ("Mean max\nT", "mean_max_transmission"),
    ]
    y = np.arange(len(metrics))
    height = 0.32
    exp_vals = [float(audit["experiment"][key]) for _, key in metrics]
    db_vals = [float(audit["database"][key]) for _, key in metrics]
    ax_data.barh(y + height / 2, exp_vals, height, color=colors["exp"], label="In-house")
    ax_data.barh(y - height / 2, db_vals, height, color=colors["db"], label="Database")
    ax_data.set_yticks(y)
    ax_data.set_yticklabels([label for label, _ in metrics])
    ax_data.invert_yaxis()
    ax_data.set_xlim(0, max(max(exp_vals), max(db_vals)) * 1.18)
    ax_data.set_xlabel("Dataset statistic")
    ax_data.set_title("b  Target distribution shifts", loc="left", fontweight="bold", fontsize=9)
    ax_data.grid(axis="x", color="#e8edf0", linewidth=0.7)
    ax_data.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        ncol=2,
        fontsize=6.4,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    for yi, ev, dv in zip(y, exp_vals, db_vals):
        ax_data.text(ev + 0.025, yi + height / 2, f"{ev:.2f}", va="center", fontsize=6.4, color=colors["exp"])
        ax_data.text(dv + 0.025, yi - height / 2, f"{dv:.2f}", va="center", fontsize=6.4, color=colors["db"])

    groups = list(perf.keys())
    x = np.array([0.0, 0.95, 2.25, 3.20])
    nominal = np.array([perf[group]["nominal"] / 50 for group in groups])
    robust = np.array([perf[group]["robust"] / 50 for group in groups])
    bar_w = 0.34
    ax_valid.bar(x - bar_w / 2, nominal, bar_w, color="#5b8792", label="Nominal-valid")
    ax_valid.bar(x + bar_w / 2, robust, bar_w, color="#b76e48", label="Robust-valid")
    ax_valid.set_ylim(0, 1.02)
    ax_valid.set_xticks(x)
    ax_valid.set_xticklabels(["Base.\n(sput.)", "Hybrid\n(sput.)", "Base.\n(ext.)", "Hybrid\n(ext.)"], fontsize=7.0)
    ax_valid.set_ylabel("Fraction of 50 targets")
    ax_valid.set_title("c  Same strict r60 protocol", loc="left", fontweight="bold", fontsize=9)
    ax_valid.grid(axis="y", color="#e8edf0", linewidth=0.7)
    ax_valid.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.28),
        ncol=2,
        fontsize=6.4,
        handlelength=1.3,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    for xi, n, r in zip(x, nominal, robust):
        ax_valid.text(xi - bar_w / 2, n + 0.025, f"{int(round(n*50))}", ha="center", fontsize=6.5)
        ax_valid.text(xi + bar_w / 2, r + 0.025, f"{int(round(r*50))}", ha="center", fontsize=6.5)

    yields = np.array([perf[group]["yield"] for group in groups])
    ax_yield.bar(
        x,
        yields,
        width=0.55,
        color=[colors["baseline"], colors["hybrid"], colors["baseline"], colors["hybrid"]],
    )
    ax_yield.set_ylim(0, 0.80)
    ax_yield.set_xticks(x)
    ax_yield.set_xticklabels(["Base.\n(sput.)", "Hybrid\n(sput.)", "Base.\n(ext.)", "Hybrid\n(ext.)"], fontsize=7.0)
    ax_yield.set_ylabel("Mean fabrication yield")
    ax_yield.set_title("d  Hybrid gain remains but absolute yield drops", loc="left", fontweight="bold", fontsize=9)
    ax_yield.grid(axis="y", color="#e8edf0", linewidth=0.7)
    for xi, value in zip(x, yields):
        ax_yield.text(xi, value + 0.020, f"{value:.3f}", ha="center", fontsize=6.5)
    annotate_delta(ax_yield, 0, 1, 0.690, "+0.087", colors["hybrid"], text_offset=0.028)
    annotate_delta(ax_yield, 2, 3, 0.43, "+0.113", colors["hybrid"])

    fig.text(
        0.01,
        0.01,
        "Database materials: Ge from Amotchkina et al. and SiO2 from Kischkat et al. via refractiveindex.info; "
        "database replication uses the same 50 target indices, nominal-valid gate, and 60-trial thickness perturbation protocol.",
        fontsize=6.5,
        color="#4b5563",
    )

    save_fig(fig, "olt_fig11_material_replication")
    plt.close(fig)


if __name__ == "__main__":
    main()
