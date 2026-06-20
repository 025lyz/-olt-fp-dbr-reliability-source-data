import json
import shutil
import csv
from pathlib import Path

import numpy as np

from physics_tmm import FPDBRTMM, detect_peaks, match_peaks, spectral_metrics


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "olt_latex" / "figures"

PALETTE = {
    "ink": "#2A2A2A",
    "muted": "#6E6E6E",
    "grid": "#E7E7E7",
    "baseline": "#596DA8",
    "teacher": "#8A7BB8",
    "hybrid": "#C95C58",
    "ge": "#DDE4EE",
    "sio2": "#EEF2F1",
    "cavity": "#F3E8D2",
}


def setup():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, stem):
    setup()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    svg = REPORTS / f"{stem}.svg"
    png = REPORTS / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    shutil.copy2(svg, FIGDIR / svg.name)
    shutil.copy2(png, FIGDIR / png.name)


def write_source_data(stem, rows):
    if not rows:
        return
    path = REPORTS / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax, label, x=-0.08, y=1.02):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["ink"],
        clip_on=False,
    )


def load_json(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def get_row(target_index, name):
    payload = load_json(name)
    return next(r for r in payload["rows"] if int(r["target_index"]) == target_index)


def params_from_row(row):
    return np.array([row["d_H"], row["d_L"], row["N"], row["L_c"]], dtype=float)


def stack_lists(solver, params, wavelength_nm):
    d_h, d_l, periods, lc = np.asarray(params, dtype=float)
    periods = max(1, int(round(periods)))
    i = int(np.argmin(np.abs(solver.wavelengths_nm - wavelength_nm)))
    n_h = solver.n_h[i]
    n_l = solver.n_l[i]
    n_c = solver.n_cavity[i]
    dbr_d = [float(d_h), float(d_l)] * periods
    d_list = [np.inf] + dbr_d + [float(lc)] + dbr_d + [np.inf]
    n_list = [1.0] + [val for pair in ([n_h, n_l] for _ in range(periods)) for val in pair]
    n_list += [n_c]
    n_list += [val for pair in ([n_h, n_l] for _ in range(periods)) for val in pair]
    n_list += [solver.substrate_index]
    return n_list, d_list


def field_profile(solver, params, wavelength_nm, points_per_layer=18):
    from tmm import coh_tmm, position_resolved

    n_list, d_list = stack_lists(solver, params, wavelength_nm)
    coh = coh_tmm("s", n_list, d_list, 0.0, wavelength_nm)
    finite = d_list[1:-1]
    total = float(np.sum(finite))
    z_all, e_all = [], []
    layer_edges = np.concatenate([[0.0], np.cumsum(finite)])
    layer_types = []
    periods = int(round(params[2]))
    names = (["Ge", "SiO2"] * periods) + ["cavity"] + (["Ge", "SiO2"] * periods)

    for layer_i, thickness in enumerate(finite, start=1):
        local = np.linspace(0.0, float(thickness), points_per_layer, endpoint=False)
        for z in local:
            resolved = position_resolved(layer_i, float(z), coh)
            e2 = (
                abs(resolved.get("Ex", 0.0)) ** 2
                + abs(resolved.get("Ey", 0.0)) ** 2
                + abs(resolved.get("Ez", 0.0)) ** 2
            )
            z_all.append(layer_edges[layer_i - 1] + z)
            e_all.append(float(np.real(e2)))
        layer_types.append(names[layer_i - 1])

    z_all.append(total)
    e_all.append(e_all[-1])
    return np.asarray(z_all), np.asarray(e_all), layer_edges, layer_types


def mirror_reflection_phase(solver, params, wavelengths_nm, side):
    from tmm import coh_tmm

    d_h, d_l, periods, _ = np.asarray(params, dtype=float)
    periods = max(1, int(round(periods)))
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    phases = []

    for wl in wavelengths_nm:
        i = int(np.argmin(np.abs(solver.wavelengths_nm - wl)))
        n_h = solver.n_h[i]
        n_l = solver.n_l[i]
        n_c = solver.n_cavity[i]
        if side == "left":
            finite_n = [val for pair in ([n_l, n_h] for _ in range(periods)) for val in pair]
            finite_d = [val for pair in ([float(d_l), float(d_h)] for _ in range(periods)) for val in pair]
            n_list = [n_c] + finite_n + [1.0]
        elif side == "right":
            finite_n = [val for pair in ([n_h, n_l] for _ in range(periods)) for val in pair]
            finite_d = [val for pair in ([float(d_h), float(d_l)] for _ in range(periods)) for val in pair]
            n_list = [n_c] + finite_n + [solver.substrate_index]
        else:
            raise ValueError("side must be 'left' or 'right'")
        d_list = [np.inf] + finite_d + [np.inf]
        phases.append(np.angle(coh_tmm("s", n_list, d_list, 0.0, float(wl))["r"]))

    return np.unwrap(np.asarray(phases, dtype=float))


def phase_residual_trace(solver, params, wavelengths_nm):
    params = np.asarray(params, dtype=float)
    lc = float(params[3])
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    n_c = np.interp(
        wavelengths_nm,
        solver.wavelengths_nm,
        np.real(solver.n_cavity),
    )
    phi_left = mirror_reflection_phase(solver, params, wavelengths_nm, "left")
    phi_right = mirror_reflection_phase(solver, params, wavelengths_nm, "right")
    propagation = 4.0 * np.pi * n_c * lc / wavelengths_nm
    total = np.unwrap(propagation + phi_left + phi_right)
    residual = np.angle(np.exp(1j * total))
    return residual, propagation, phi_left + phi_right


def nearest_zero_crossing(x, y, reference):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    crossings = []
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if y0 == 0:
            crossings.append(float(x[i]))
        elif y0 * y1 < 0 and abs(y1 - y0) > 1e-12:
            frac = -y0 / (y1 - y0)
            crossings.append(float(x[i] + frac * (x[i + 1] - x[i])))
    if not crossings:
        return None
    return min(crossings, key=lambda value: abs(value - reference))


def draw_layer_background(ax, edges_nm, layer_types):
    colors = {"Ge": PALETTE["ge"], "SiO2": PALETTE["sio2"], "cavity": PALETTE["cavity"]}
    edges = np.asarray(edges_nm, dtype=float) / 1000.0
    ymax = ax.get_ylim()[1]
    for i, mat in enumerate(layer_types):
        ax.axvspan(edges[i], edges[i + 1], color=colors[mat], alpha=0.38 if mat == "cavity" else 0.30, lw=0, zorder=0)
    cavity_i = layer_types.index("cavity")
    ax.text(
        (edges[cavity_i] + edges[cavity_i + 1]) / 2,
        ymax * 0.94,
        "SiO2 cavity",
        fontsize=7,
        ha="center",
        va="top",
        color=PALETTE["muted"],
    )


def main(target_index=46328):
    import matplotlib.pyplot as plt

    setup()
    data = np.load(ROOT / "dataset" / "fp_dbr_data_100000_physics_aware_experiment.npz")
    wavelengths = data["wavelengths"].astype(float)
    target = data["spectra"][target_index].astype(float)
    solver = FPDBRTMM(wavelengths)

    configs = [
        (
            "Baseline",
            "baseline_strategy12_50targets_final_robustness_r60_summary.json",
            PALETTE["baseline"],
            (0, (4, 2)),
        ),
        (
            "Teacher",
            "teacher_replay_strategy12_50targets_final_robustness_r60_summary.json",
            PALETTE["teacher"],
            (0, (2, 2)),
        ),
        (
            "Hybrid",
            "inference_time_hybrid_50targets_final_robustness_r60_summary.json",
            PALETTE["hybrid"],
            "solid",
        ),
    ]

    target_peaks = sorted(detect_peaks(target, wavelengths), key=lambda p: p["wavelength"])
    selected_peak = target_peaks[1]["wavelength"]
    rows = []
    for label, filename, color, style in configs:
        row = get_row(target_index, filename)
        params = params_from_row(row)
        spectrum = solver.simulate(params)
        pred_peaks = sorted(detect_peaks(spectrum, wavelengths), key=lambda p: p["wavelength"])
        metrics = spectral_metrics(target, spectrum, wavelengths)
        matched = match_peaks(target_peaks, pred_peaks)["matched"]
        shifts = [item["shift"] for item in matched]
        rows.append(
            {
                "label": label,
                "params": params,
                "spectrum": spectrum,
                "peaks": pred_peaks,
                "metrics": metrics,
                "mean_shift": float(np.mean(shifts)) if shifts else np.nan,
                "color": color,
                "style": style,
            }
        )

    fig = plt.figure(figsize=(7.75, 6.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.16, 1.04], width_ratios=[1.0, 1.0], hspace=0.38, wspace=0.30)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_field = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.set_axis_off()
    panel_label(ax_spec, "a", x=-0.13, y=1.10)
    panel_label(ax_phase, "b", x=-0.08, y=1.10)
    panel_label(ax_field, "c", x=-0.08, y=1.08)
    panel_label(ax_table, "d", x=-0.10, y=1.08)

    ax_spec.plot(wavelengths, target, color=PALETTE["ink"], lw=1.8, label="Target")
    for row in rows:
        lw = 1.75 if row["label"] == "Hybrid" else 1.25
        ax_spec.plot(wavelengths, row["spectrum"], color=row["color"], lw=lw, ls=row["style"], label=row["label"])
    for peak in target_peaks:
        ax_spec.axvline(peak["wavelength"], color="#BFBFBF", lw=0.65, zorder=0)
    ax_spec.axvspan(selected_peak - 20, selected_peak + 20, color=PALETTE["cavity"], alpha=0.22, lw=0)
    ax_spec.set_xlim(3000, 3650)
    ax_spec.set_ylim(-0.02, 0.44)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Transmission")
    ax_spec.set_title("Low MSE can still hide peak-valid failures", fontsize=9, fontweight="bold", pad=10)
    ax_spec.grid(color=PALETTE["grid"], lw=0.7)
    ax_spec.legend(fontsize=7, ncol=2, loc="upper right")

    phase_window = (wavelengths >= 3040) & (wavelengths <= 3520)
    phase_wl = wavelengths[phase_window][::3]
    hybrid_zero = None
    for row in [rows[0], rows[1], rows[2]]:
        residual, propagation, mirror_phase = phase_residual_trace(solver, row["params"], phase_wl)
        row["phase_residual_at_peak"] = float(np.interp(selected_peak, phase_wl, residual))
        row["phase_wl"] = phase_wl
        row["phase_residual"] = residual
        if row["label"] == "Hybrid":
            hybrid_zero = nearest_zero_crossing(phase_wl, residual, selected_peak)
        ax_phase.plot(
            phase_wl,
            residual,
            color=row["color"],
            lw=1.7 if row["label"] == "Hybrid" else 1.15,
            ls=row["style"],
            label=row["label"],
        )
    for peak in target_peaks:
        if 3040 <= peak["wavelength"] <= 3520:
            ax_phase.axvline(peak["wavelength"], color="#BFBFBF", lw=0.65, zorder=0)
    ax_phase.axhline(0, color=PALETTE["ink"], lw=0.85, ls=(0, (3, 3)))
    ax_phase.axvspan(selected_peak - 20, selected_peak + 20, color=PALETTE["cavity"], alpha=0.22, lw=0)
    if hybrid_zero is not None:
        ax_phase.scatter([hybrid_zero], [0], s=24, color=PALETTE["hybrid"], edgecolor="white", linewidth=0.5, zorder=5)
        ax_phase.annotate(
            "target resonance",
            xy=(hybrid_zero, 0),
            xytext=(hybrid_zero + 72, 1.65),
            fontsize=7,
            color=PALETTE["hybrid"],
            arrowprops=dict(arrowstyle="-|>", lw=0.75, color=PALETTE["hybrid"]),
        )
    ax_phase.set_xlim(3040, 3520)
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_yticks([-np.pi, 0, np.pi])
    ax_phase.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax_phase.set_xlabel("Wavelength (nm)")
    ax_phase.set_ylabel(r"wrapped $\Phi(\lambda)$")
    ax_phase.set_title("Round-trip phase residual", fontsize=9, fontweight="bold", pad=10)
    ax_phase.grid(color=PALETTE["grid"], lw=0.7)
    phase_legend = ax_phase.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    phase_legend.get_frame().set_facecolor("white")
    phase_legend.get_frame().set_alpha(0.86)
    phase_legend.get_frame().set_edgecolor("none")

    field_peak = float(selected_peak)
    field_rows = [rows[0], rows[2]]
    max_e = 0.0
    profiles = []
    for row in field_rows:
        z, e2, edges, layer_types = field_profile(solver, row["params"], field_peak)
        profiles.append((row, z, e2, edges, layer_types))
        max_e = max(max_e, float(np.max(e2)))
    ax_field.set_ylim(0, max_e * 1.08)
    draw_layer_background(ax_field, profiles[-1][3], profiles[-1][4])
    for row, z, e2, edges, layer_types in profiles:
        ax_field.plot(z / 1000.0, e2, color=row["color"], lw=1.4 if row["label"] == "Baseline" else 1.8, ls=row["style"], label=row["label"])
    ax_field.set_xlim(0, profiles[-1][3][-1] / 1000.0)
    ax_field.set_xlabel("Depth through multilayer (um)")
    ax_field.set_ylabel(r"1D TMM $|E|^2$")
    ax_field.set_title(f"Field profile at target peak {field_peak:.0f} nm", fontsize=9, fontweight="bold")
    ax_field.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax_field.legend(fontsize=7, loc="upper right")

    ax_table.text(0.02, 0.95, "Mechanism summary", fontsize=9, fontweight="bold", transform=ax_table.transAxes)
    ax_table.text(0.02, 0.82, "method", fontsize=7, color=PALETTE["muted"], transform=ax_table.transAxes)
    ax_table.text(0.66, 0.82, "shift", fontsize=7, color=PALETTE["muted"], ha="right", transform=ax_table.transAxes)
    ax_table.text(0.98, 0.82, "valid / yield", fontsize=7, color=PALETTE["muted"], ha="right", transform=ax_table.transAxes)
    for i, row in enumerate(rows):
        yy = 0.68 - i * 0.18
        nominal = "yes" if row["metrics"]["missing_peak_count"] == 0 and row["metrics"]["false_peak_count"] == 0 and row["metrics"]["mean_peak_shift"] <= 20 else "no"
        source = get_row(
            target_index,
            {
                "Baseline": "baseline_strategy12_50targets_final_robustness_r60_summary.json",
                "Teacher": "teacher_replay_strategy12_50targets_final_robustness_r60_summary.json",
                "Hybrid": "inference_time_hybrid_50targets_final_robustness_r60_summary.json",
            }[row["label"]],
        )
        ax_table.plot([0.02, 0.08], [yy + 0.01, yy + 0.01], color=row["color"], lw=1.8, ls=row["style"], transform=ax_table.transAxes)
        ax_table.text(0.105, yy, row["label"], fontsize=7.3, transform=ax_table.transAxes)
        ax_table.text(0.66, yy, f"{row['metrics']['mean_peak_shift']:.1f} nm", fontsize=7.3, ha="right", transform=ax_table.transAxes)
        ax_table.text(0.98, yy, f"{nominal} / {float(source['yield_rate']):.2f}", fontsize=7.3, ha="right", transform=ax_table.transAxes)
    ax_table.text(
        0.02,
        0.08,
        "Phase residuals use the complex DBR\nreflection coefficients from the same\nnormal-incidence coherent TMM model.",
        fontsize=7,
        color=PALETTE["muted"],
        transform=ax_table.transAxes,
    )

    source_rows = []
    for row in rows:
        source = get_row(
            target_index,
            {
                "Baseline": "baseline_strategy12_50targets_final_robustness_r60_summary.json",
                "Teacher": "teacher_replay_strategy12_50targets_final_robustness_r60_summary.json",
                "Hybrid": "inference_time_hybrid_50targets_final_robustness_r60_summary.json",
            }[row["label"]],
        )
        source_rows.append(
            {
                "target_index": target_index,
                "method": row["label"],
                "selected_peak_nm": field_peak,
                "mean_peak_shift_nm": row["metrics"]["mean_peak_shift"],
                "missing_peak_count": row["metrics"]["missing_peak_count"],
                "false_peak_count": row["metrics"]["false_peak_count"],
                "phase_residual_at_selected_peak_rad": row.get(
                    "phase_residual_at_peak", np.nan
                ),
                "d_H_nm": row["params"][0],
                "d_L_nm": row["params"][1],
                "N": int(round(row["params"][2])),
                "L_c_nm": row["params"][3],
                "yield_rate": float(source["yield_rate"]),
            }
        )
    write_source_data("olt_fig8_phase_mechanism_source_data", source_rows)
    save(fig, "olt_fig8_mechanism_field")
    plt.close(fig)


if __name__ == "__main__":
    main()
