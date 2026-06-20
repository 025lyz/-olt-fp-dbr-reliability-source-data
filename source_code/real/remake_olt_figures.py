import csv
import json
import shutil
from pathlib import Path

import numpy as np
from matplotlib.transforms import blended_transform_factory

from physics_tmm import FPDBRTMM, detect_peaks, match_peaks, spectral_metrics


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "olt_latex" / "figures"


METHODS = [
    ("baseline", "Baseline\nCVAE"),
    ("old_teacher_replay", "Teacher\nreplay"),
    ("inference_time_hybrid", "Hybrid"),
]

PALETTE = {
    "ink": "#2A2A2A",
    "muted": "#6E6E6E",
    "grid": "#E7E7E7",
    "bg": "#FFFFFF",
    "ge": "#5B6D8C",
    "sio2": "#C7D9C3",
    "cavity": "#EACB8B",
    "substrate": "#B8B8B8",
    "baseline": "#596DA8",
    "teacher": "#8A7BB8",
    "hybrid": "#C95C58",
    "hybrid_soft": "#F1D4D2",
    "green": "#3B8F5A",
    "gold": "#C4942F",
    "blue": "#2F6FA5",
    "soft_blue": "#DCE6F4",
    "soft_green": "#DDEBDF",
    "soft_gold": "#F2E4C4",
    "soft_gray": "#F2F2F2",
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


def load_json(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def panel_label(ax, label, x=-0.02, y=1.02):
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


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_round_box(ax, xy, wh, text, face, edge="#777777", fontsize=7.2, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        fc=face,
        ec=edge,
        lw=0.8,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=PALETTE["ink"],
        transform=ax.transAxes,
    )
    return patch


def add_step_box(ax, center_x, y, w, h, number, text, face):
    add_round_box(ax, (center_x - w / 2, y), (w, h), text, face=face, fontsize=7.0)
    ax.text(
        center_x,
        y + h + 0.045,
        number,
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.11", fc=PALETTE["ink"], ec="none"),
    )


def arrow(ax, start, end, color=None, lw=1.0, rad=0.0):
    from matplotlib.patches import FancyArrowPatch

    color = color or PALETTE["ink"]
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def elbow_arrow(ax, points, color=None, lw=1.0):
    color = color or PALETTE["ink"]
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, transform=ax.transAxes, clip_on=False)
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


def add_diamond(ax, center, width, height, text, face, edge="#777777", fontsize=7.0):
    from matplotlib.patches import Polygon

    cx, cy = center
    vertices = [
        (cx, cy + height / 2),
        (cx + width / 2, cy),
        (cx, cy - height / 2),
        (cx - width / 2, cy),
    ]
    patch = Polygon(vertices, closed=True, fc=face, ec=edge, lw=0.9, transform=ax.transAxes, clip_on=False)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=PALETTE["ink"], transform=ax.transAxes)
    return patch


def fig1_device_pipeline():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    setup()
    fig = plt.figure(figsize=(7.5, 6.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.55], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    clean_axis(ax)
    clean_axis(ax2)
    panel_label(ax, "a")
    panel_label(ax2, "b")

    ax.text(0.02, 0.94, "One-dimensional FP-DBR filter", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.84, r"Air / (Ge, SiO$_2$)$^N$ / SiO$_2$ cavity / (Ge, SiO$_2$)$^N$ / substrate", fontsize=8, color=PALETTE["muted"], transform=ax.transAxes)

    y0, h = 0.36, 0.24
    x = 0.08
    ax.text(0.045, y0 + h / 2, "Air", ha="center", va="center", fontsize=7, color=PALETTE["muted"], transform=ax.transAxes)
    dH, dL = 0.021, 0.033
    for side in range(2):
        for _ in range(5):
            ax.add_patch(Rectangle((x, y0), dH, h, fc=PALETTE["ge"], ec="white", lw=0.6, transform=ax.transAxes))
            x += dH
            ax.add_patch(Rectangle((x, y0), dL, h, fc=PALETTE["sio2"], ec="white", lw=0.6, transform=ax.transAxes))
            x += dL
        if side == 0:
            ax.add_patch(Rectangle((x, y0), 0.095, h, fc=PALETTE["cavity"], ec="white", lw=0.6, transform=ax.transAxes))
            ax.annotate(
                r"$L_c$",
                xy=(x + 0.047, y0 + h),
                xytext=(x + 0.047, y0 + h + 0.13),
                ha="center",
                fontsize=8,
                arrowprops=dict(arrowstyle="-", color=PALETTE["muted"], lw=0.8),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
            x += 0.095
    ax.add_patch(Rectangle((x, y0), 0.085, h, fc=PALETTE["substrate"], ec="white", lw=0.6, transform=ax.transAxes))
    ax.text(x + 0.042, y0 - 0.07, "substrate", ha="center", fontsize=7, color=PALETTE["muted"], transform=ax.transAxes)

    ax.annotate(r"$d_H$", xy=(0.092, y0), xytext=(0.092, y0 - 0.12), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["muted"]), xycoords=ax.transAxes, textcoords=ax.transAxes)
    ax.annotate(r"$d_L$", xy=(0.125, y0), xytext=(0.125, y0 - 0.12), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["muted"]), xycoords=ax.transAxes, textcoords=ax.transAxes)
    ax.annotate(r"$N$ periods", xy=(0.23, y0 + h + 0.01), xytext=(0.23, y0 + h + 0.14), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.8, color=PALETTE["muted"]), xycoords=ax.transAxes, textcoords=ax.transAxes)

    legend_x = 0.76
    for i, (label, color) in enumerate([("Ge", PALETTE["ge"]), ("SiO2", PALETTE["sio2"]), ("cavity", PALETTE["cavity"])]):
        xx = legend_x + 0.075 * i
        ax.add_patch(Rectangle((xx, 0.76), 0.030, 0.052, fc=color, ec="0.55", lw=0.5, transform=ax.transAxes))
        ax.text(xx + 0.015, 0.71, label, ha="center", va="top", fontsize=7, transform=ax.transAxes)
    ax.text(0.67, 0.18, r"Design vector: $[d_H,d_L,N,L_c]$", fontsize=8, transform=ax.transAxes)

    ax2.text(0.02, 0.95, "Hybrid inference workflow", fontsize=10, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.02, 0.88, "Real TMM and the peak-valid gate decide whether a candidate is accepted.", fontsize=8, color=PALETTE["muted"], transform=ax2.transAxes)

    ax2.add_patch(
        FancyBboxPatch(
            (0.03, 0.52),
            0.93,
            0.25,
            boxstyle="round,pad=0.014,rounding_size=0.020",
            fc="#F7FAFC",
            ec="#D7E3EC",
            lw=0.7,
            transform=ax2.transAxes,
            zorder=-1,
            clip_on=False,
        )
    )
    ax2.add_patch(
        FancyBboxPatch(
            (0.41, 0.18),
            0.55,
            0.22,
            boxstyle="round,pad=0.014,rounding_size=0.020",
            fc="#FFF9EA",
            ec="#E8D9AF",
            lw=0.7,
            transform=ax2.transAxes,
            zorder=-1,
            clip_on=False,
        )
    )

    def node(cx, cy, text, face, w=0.082, h=0.125, fs=5.85):
        return add_round_box(ax2, (cx - w / 2, cy - h / 2), (w, h), text, face=face, fontsize=fs)

    def line_arrow(start, end, color=None, lw=1.25):
        ax2.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax2.transAxes,
            textcoords=ax2.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=lw,
                color=color or PALETTE["ink"],
                mutation_scale=8.5,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0",
            ),
        )

    def h_connector(x0, x1, y, color=None, lw=1.25, head_len=0.010, head_h=0.020):
        from matplotlib.patches import Polygon

        color = color or PALETTE["ink"]
        if x1 <= x0 + head_len:
            mid = (x0 + x1) / 2
            x0 = mid - head_len * 1.15
            x1 = mid + head_len * 1.15
        base_x = x1 - head_len
        ax2.plot([x0, base_x], [y, y], color=color, lw=lw, transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(
            Polygon(
                [(base_x, y - head_h / 2), (base_x, y + head_h / 2), (x1, y)],
                closed=True,
                fc=color,
                ec=color,
                transform=ax2.transAxes,
                clip_on=False,
            )
        )

    def arrow_glyph(cx, y, color=None, head_len=0.020, head_h=0.030):
        from matplotlib.patches import Polygon

        color = color or PALETTE["ink"]
        ax2.add_patch(
            Polygon(
                [
                    (cx - head_len / 2, y - head_h / 2),
                    (cx - head_len / 2, y + head_h / 2),
                    (cx + head_len / 2, y),
                ],
                closed=True,
                fc=color,
                ec=color,
                transform=ax2.transAxes,
                clip_on=False,
            )
        )

    def mini_connector(cx, y, color=None, total_w=0.038, head_len=0.012, head_h=0.022, lw=1.20):
        from matplotlib.patches import Polygon

        color = color or PALETTE["ink"]
        x0 = cx - total_w / 2
        x1 = cx + total_w / 2
        base_x = x1 - head_len
        ax2.plot([x0, base_x], [y, y], color=color, lw=lw, transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(
            Polygon(
                [(base_x, y - head_h / 2), (base_x, y + head_h / 2), (x1, y)],
                closed=True,
                fc=color,
                ec=color,
                transform=ax2.transAxes,
                clip_on=False,
            )
        )

    def v_connector_down(x, y0, y1, color=None, lw=1.25, head_len=0.020, head_w=0.026):
        from matplotlib.patches import Polygon

        color = color or PALETTE["ink"]
        base_y = y1 + head_len
        ax2.plot([x, x], [y0, base_y], color=color, lw=lw, transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(
            Polygon(
                [(x - head_w / 2, base_y), (x + head_w / 2, base_y), (x, y1)],
                closed=True,
                fc=color,
                ec=color,
                transform=ax2.transAxes,
                clip_on=False,
            )
        )

    y_main = 0.65
    node_w = 0.082
    patch_pad = 0.010
    start_gap = 0.010
    stop_gap = 0.010
    x_main = [0.080, 0.225, 0.370, 0.515]
    texts = ["Target\nspectrum", "Teacher\nreplay\nCVAE", "Surrogate\nprefilter", "Real-TMM\nrerank"]
    faces = [PALETTE["soft_gray"], PALETTE["soft_blue"], PALETTE["soft_blue"], PALETTE["hybrid_soft"]]
    for i, (cx, text, face) in enumerate(zip(x_main, texts, faces), 1):
        node(cx, y_main, text, face, w=node_w)
        ax2.text(cx, y_main + 0.105, str(i), ha="center", va="center", fontsize=6.6, fontweight="bold", color="white", transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.10", fc=PALETTE["ink"], ec="none"))
    for x1, x2 in zip(x_main[:-1], x_main[1:]):
        mini_connector((x1 + x2) / 2, y_main)

    gate_x, gate_y = 0.662, y_main
    gate_w = 0.122
    add_diamond(ax2, (gate_x, gate_y), gate_w, 0.135, "Nominal\nvalid?", face=PALETTE["hybrid_soft"], fontsize=6.6)
    ax2.text(gate_x, gate_y + 0.105, "5", ha="center", va="center", fontsize=6.6, fontweight="bold", color="white", transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.10", fc=PALETTE["ink"], ec="none"))
    h_connector(
        x_main[-1] + node_w / 2 + 0.018,
        gate_x - gate_w / 2 - 0.018,
        y_main,
        lw=1.15,
        head_len=0.010,
        head_h=0.020,
    )

    final_x, final_y = 0.875, 0.485
    final_w, final_h = 0.135, 0.345
    add_round_box(
        ax2,
        (final_x - final_w / 2, final_y - final_h / 2),
        (final_w, final_h),
        "Unified\nr60 robustness\nvalidation",
        face=PALETTE["soft_gray"],
        fontsize=6.8,
    )
    ax2.text(final_x, final_y + final_h / 2 + 0.040, "6", ha="center", va="center", fontsize=6.6, fontweight="bold", color="white", transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.10", fc=PALETTE["ink"], ec="none"))
    final_left = final_x - final_w / 2
    h_connector(
        gate_x + gate_w / 2 + 0.004,
        final_left - 0.014,
        y_main,
        color=PALETTE["green"],
        lw=1.35,
        head_len=0.014,
        head_h=0.024,
    )
    ax2.text(0.752, y_main + 0.052, "yes", fontsize=6.8, color=PALETTE["green"], ha="center", transform=ax2.transAxes)

    ax2.text(0.45, 0.425, "Fallback repair lane", fontsize=7.2, color=PALETTE["gold"], ha="left", transform=ax2.transAxes)
    y_fix = 0.295
    repair_x0, repair_w, repair_h = 0.47, 0.270, 0.135
    add_round_box(
        ax2,
        (repair_x0, y_fix - repair_h / 2),
        (repair_w, repair_h),
        "Physics-prior residual fallback\n+ local real-TMM refinement",
        face=PALETTE["soft_gold"],
        fontsize=6.8,
    )

    # Failure path: vertical into the repair module, then horizontal into the same robustness gate.
    drop_x = gate_x
    repair_top = y_fix + repair_h / 2
    v_connector_down(
        drop_x,
        gate_y - 0.076 - patch_pad - start_gap,
        repair_top + patch_pad + stop_gap,
        color=PALETTE["gold"],
        lw=1.45,
        head_len=0.017,
        head_w=0.020,
    )
    ax2.text(drop_x + 0.018, 0.470, "no", fontsize=6.8, color=PALETTE["gold"], ha="left", transform=ax2.transAxes)

    repair_right = repair_x0 + repair_w
    h_connector(
        repair_right + patch_pad + start_gap,
        final_left - patch_pad - stop_gap,
        y_fix,
        color=PALETTE["gold"],
        lw=1.45,
        head_len=0.012,
        head_h=0.022,
    )

    save(fig, "olt_fig1_device_pipeline")
    plt.close(fig)


def fig2_protocol():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    setup()
    fig = plt.figure(figsize=(7.25, 4.20))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.80, 1.20], width_ratios=[1.10, 0.90], hspace=0.28, wspace=0.24)
    ax = fig.add_subplot(gs[0, :])
    ax_nom = fig.add_subplot(gs[1, 0])
    ax_rob = fig.add_subplot(gs[1, 1])
    for a in [ax, ax_nom, ax_rob]:
        clean_axis(a)
    panel_label(ax, "a")
    panel_label(ax_nom, "b")
    panel_label(ax_rob, "c")

    ax.text(0.02, 0.92, "Peak-valid protocol", fontsize=10.5, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.02,
        0.76,
        "The same peak gate is applied to the nominal design and to every thickness-perturbed trial.",
        fontsize=7.8,
        color=PALETTE["muted"],
        transform=ax.transAxes,
    )

    step_x = [0.105, 0.345, 0.585, 0.825]
    step_labels = ["TMM\nspectrum", "Peak\ndetection", "Peak\nmatching", "Validity\ngate"]
    step_faces = [PALETTE["soft_blue"], PALETTE["soft_gray"], PALETTE["soft_gray"], PALETTE["soft_green"]]
    for idx, (xx, label, face) in enumerate(zip(step_x, step_labels, step_faces), start=1):
        ax.add_patch(
            FancyBboxPatch(
                (xx - 0.080, 0.305),
                0.160,
                0.205,
                boxstyle="round,pad=0.012,rounding_size=0.020",
                fc=face,
                ec="#C6CCD4",
                lw=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(xx - 0.053, 0.468, f"{idx}", fontsize=6.8, fontweight="bold", color=PALETTE["blue"], transform=ax.transAxes)
        ax.text(xx, 0.405, label, fontsize=7.3, ha="center", va="center", transform=ax.transAxes)
    for x0, x1 in zip(step_x[:-1], step_x[1:]):
        ax.plot([x0 + 0.093, x1 - 0.093], [0.407, 0.407], color=PALETTE["ink"], lw=0.85, transform=ax.transAxes)
        arrow(ax, (x1 - 0.108, 0.407), (x1 - 0.093, 0.407), lw=0.85, color=PALETTE["ink"])
    ax.text(
        0.825,
        0.235,
        "peak gate, not MSE-only",
        fontsize=6.5,
        color=PALETTE["green"],
        ha="center",
        transform=ax.transAxes,
    )

    ax_nom.text(0.02, 0.94, "Nominal-valid gate", fontsize=9.0, fontweight="bold", transform=ax_nom.transAxes)
    ax_nom.text(0.02, 0.84, "All four conditions must hold.", fontsize=7.2, color=PALETTE["muted"], transform=ax_nom.transAxes)
    gate_rows = [
        ("Spectral error", r"MSE $\leq$ 0.01"),
        ("Peak count", "missing peaks = 0"),
        ("False resonance", "false peaks = 0"),
        ("Peak position", r"mean shift $\leq$ 20 nm"),
    ]
    x0, y_top, row_h = 0.05, 0.695, 0.135
    for i, (name, value) in enumerate(gate_rows):
        y = y_top - i * row_h
        ax_nom.add_patch(
            FancyBboxPatch(
                (x0, y),
                0.80,
                0.092,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                fc="#FAFAFA",
                ec="#D7DBDF",
                lw=0.65,
                transform=ax_nom.transAxes,
            )
        )
        ax_nom.text(x0 + 0.035, y + 0.046, name, fontsize=6.9, color=PALETTE["muted"], va="center", transform=ax_nom.transAxes)
        ax_nom.text(x0 + 0.765, y + 0.046, value, fontsize=7.0, ha="right", va="center", transform=ax_nom.transAxes)
    gate_centers = [y_top + 0.046 - i * row_h for i in range(len(gate_rows))]
    bracket_x = 0.87
    bracket_pad = 0.020
    ax_nom.text(
        0.915,
        0.5 * (gate_centers[0] + gate_centers[-1]),
        "AND",
        fontsize=7.1,
        fontweight="bold",
        color=PALETTE["green"],
        ha="center",
        va="center",
        transform=ax_nom.transAxes,
    )
    ax_nom.plot(
        [bracket_x, bracket_x],
        [gate_centers[-1] - bracket_pad, gate_centers[0] + bracket_pad],
        color=PALETTE["green"],
        lw=1.0,
        transform=ax_nom.transAxes,
    )
    for y in gate_centers:
        ax_nom.plot([bracket_x - 0.020, bracket_x], [y, y], color=PALETTE["green"], lw=1.0, transform=ax_nom.transAxes)

    ax_rob.text(0.02, 0.94, "Robust-valid gate", fontsize=9.0, fontweight="bold", transform=ax_rob.transAxes)
    ax_rob.text(0.02, 0.84, "Monte Carlo thickness perturbation.", fontsize=7.2, color=PALETTE["muted"], transform=ax_rob.transAxes)
    robust_rows = [
        ("Trials per design", "60"),
        ("Thickness error", r"$\sigma = 5$ nm"),
        ("Index perturbation", "sigma fraction = 0.0"),
        ("Acceptance", r"yield $\geq$ 0.5 + nominal-valid"),
    ]
    for i, (name, value) in enumerate(robust_rows):
        y = y_top - i * row_h
        ax_rob.add_patch(
            FancyBboxPatch(
                (0.05, y),
                0.86,
                0.092,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                fc="#FAFAFA" if i < 3 else PALETTE["soft_green"],
                ec="#D7DBDF" if i < 3 else "#9EC5A5",
                lw=0.65,
                transform=ax_rob.transAxes,
            )
        )
        ax_rob.text(0.085, y + 0.046, name, fontsize=6.9, color=PALETTE["muted"], va="center", transform=ax_rob.transAxes)
        ax_rob.text(0.875, y + 0.046, value, fontsize=7.0, ha="right", va="center", transform=ax_rob.transAxes, color=PALETTE["green"] if i == 3 else PALETTE["ink"])

    save(fig, "olt_fig2_protocol")
    plt.close(fig)


def comparison_rows():
    return load_json("strategy12_50targets_unified_r60_comparison.json")["rows"]


def summaries():
    return {
        "baseline": load_json("baseline_strategy12_50targets_final_robustness_r60_summary.json"),
        "teacher": load_json("teacher_replay_strategy12_50targets_final_robustness_r60_summary.json"),
        "hybrid": load_json("inference_time_hybrid_50targets_final_robustness_r60_summary.json"),
    }


def fig3_valid_rates():
    import matplotlib.pyplot as plt

    setup()
    rows = comparison_rows()
    data = {
        "Nominal-valid": [100 * float(r["nominal_valid_rate"]) for r in rows],
        "Robust-valid": [100 * float(r["robust_valid_rate"]) for r in rows],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55), sharey=True, gridspec_kw={"wspace": 0.10})
    colors = [PALETTE["baseline"], PALETTE["teacher"], PALETTE["hybrid"]]
    names = [m[1].replace("\n", " ") for m in METHODS]
    y = np.arange(len(names))[::-1]
    for ax, (metric, vals), label in zip(axes, data.items(), ["a", "b"]):
        panel_label(ax, label)
        bars = ax.barh(y, vals, color=colors, height=0.52, edgecolor="white", lw=0.8)
        for bar, val in zip(bars, vals):
            ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%", va="center", fontsize=8)
        ax.set_xlim(0, 102)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [])
        ax.set_xlabel("Targets passing gate (%)")
        ax.set_title(metric, fontsize=9, fontweight="bold")
        ax.grid(axis="x", color=PALETTE["grid"], lw=0.8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.08, 0.02, "n = 50 targets, unified r60 protocol", fontsize=7.5, color=PALETTE["muted"])
    fig.subplots_adjust(bottom=0.20)
    save(fig, "olt_fig3_valid_rates_r60")
    plt.close(fig)


def fig4_yield():
    import matplotlib.pyplot as plt

    setup()
    rows = comparison_rows()
    summ = summaries()
    means = [float(r["yield_mean"]) for r in rows]
    dists = [
        [float(r["yield_rate"]) for r in summ["baseline"]["rows"]],
        [float(r["yield_rate"]) for r in summ["teacher"]["rows"]],
        [float(r["yield_rate"]) for r in summ["hybrid"]["rows"]],
    ]
    colors = [PALETTE["baseline"], PALETTE["teacher"], PALETTE["hybrid"]]
    labels = ["Baseline", "Teacher", "Hybrid"]

    fig = plt.figure(figsize=(7.25, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.22], wspace=0.26)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    panel_label(ax0, "a")
    panel_label(ax1, "b")

    y = np.arange(3)[::-1]
    ax0.barh(y, means, color=colors, height=0.50)
    for yy, val in zip(y, means):
        ax0.text(val + 0.015, yy, f"{val:.3f}", va="center", fontsize=8)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_xlim(0, 0.70)
    ax0.set_xlabel("Mean yield")
    ax0.set_title("Central tendency", fontsize=9, fontweight="bold")
    ax0.grid(axis="x", color=PALETTE["grid"])
    ax0.spines["left"].set_visible(False)
    ax0.tick_params(axis="y", length=0)

    pos = np.arange(1, 4)
    parts = ax1.violinplot(dists, positions=pos, widths=0.74, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.32)
    parts["cmedians"].set_color(PALETTE["ink"])
    parts["cmedians"].set_linewidth(1.2)
    rng = np.random.default_rng(7)
    for i, (vals, color) in enumerate(zip(dists, colors), start=1):
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax1.scatter(np.full(len(vals), i) + jitter, vals, s=10, color=color, alpha=0.58, edgecolor="white", linewidth=0.25, zorder=3)
    ax1.axhline(0.5, color=PALETTE["ink"], lw=0.8, ls=(0, (3, 3)))
    ax1.text(
        1.012,
        0.5,
        "0.5 threshold",
        ha="left",
        va="center",
        fontsize=7,
        color=PALETTE["muted"],
        transform=blended_transform_factory(ax1.transAxes, ax1.transData),
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.86},
    )
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_ylabel("Per-target perturbation yield")
    ax1.set_title("Target-level distribution", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", color=PALETTE["grid"])

    save(fig, "olt_fig4_yield_r60")
    plt.close(fig)


def params_from_row(row):
    return np.array([row["d_H"], row["d_L"], row["N"], row["L_c"]], dtype=float)


def fig5_spectra(target_index=46328):
    import matplotlib.pyplot as plt

    setup()
    data = np.load(ROOT / "dataset" / "fp_dbr_data_100000_physics_aware_experiment.npz")
    wavelengths = data["wavelengths"].astype(float)
    target = data["spectra"][target_index].astype(float)
    solver = FPDBRTMM(wavelengths)
    files = {
        "Baseline": "baseline_strategy12_50targets_final_robustness_r60_summary.json",
        "Hybrid": "inference_time_hybrid_50targets_final_robustness_r60_summary.json",
    }
    target_peaks = sorted(detect_peaks(target, wavelengths), key=lambda p: p["transmission"], reverse=True)
    guide_peaks = sorted(target_peaks[:3], key=lambda p: p["wavelength"])
    curves = {}
    for label, file in files.items():
        row = next(r for r in load_json(file)["rows"] if int(r["target_index"]) == target_index)
        spec = solver.simulate(params_from_row(row))
        met = spectral_metrics(target, spec, wavelengths)
        matched = match_peaks(detect_peaks(target, wavelengths), detect_peaks(spec, wavelengths))["matched"]
        curves[label] = {"spectrum": spec, "metrics": met, "matched": matched}

    fig, axes = plt.subplots(1, 2, figsize=(7.85, 3.05), sharey=True, gridspec_kw={"wspace": 0.10})
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    panels = [
        (axes[0], "Baseline", "#8C939D", "Target vs baseline"),
        (axes[1], "Hybrid", PALETTE["hybrid"], "Target vs hybrid"),
    ]
    for ax, label, color, title in panels:
        ax.plot(wavelengths, target, color=PALETTE["ink"], lw=1.55, ls=(0, (4, 2)), label="Target")
        ax.plot(
            wavelengths,
            curves[label]["spectrum"],
            color=color,
            lw=2.15 if label == "Hybrid" else 1.65,
            label=label,
        )
        for peak in guide_peaks:
            ax.axvline(peak["wavelength"], color="#C7C7C7", lw=0.75, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(3040, 3525)
        ax.set_ylim(-0.015, 0.468)
        ax.set_xlabel("Wavelength (nm)", fontsize=8.2)
        ax.set_title(title, fontsize=9.4, fontweight="bold")
        ax.grid(color=PALETTE["grid"], lw=0.65)
        ax.legend(fontsize=7.4, loc="upper right", handlelength=2.2)
        ax.tick_params(labelsize=8)

        if label == "Baseline":
            dominant = min(
                curves[label]["matched"],
                key=lambda item: abs(item["target_wavelength"] - 3178.1782),
            )
            target_wl = dominant["target_wavelength"]
            pred_wl = dominant["pred_wavelength"]
            y_arrow = 0.426
            ax.annotate(
                "",
                xy=(target_wl, y_arrow),
                xytext=(pred_wl, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=color, shrinkA=0, shrinkB=0),
            )
            ax.text(
                (target_wl + pred_wl) / 2,
                0.440,
                rf"mean $\Delta\lambda$ = {curves[label]['metrics']['mean_peak_shift']:.1f} nm",
                fontsize=7.8,
                color=PALETTE["muted"],
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.0},
            )
        else:
            ax.annotate(
                "aligned target peak",
                xy=(3178.2, 0.395),
                xytext=(3236, 0.417),
                fontsize=7.5,
                color=color,
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color=color),
            )
        ax.text(
            0.05,
            0.08,
            rf"mean $\Delta\lambda$ = {curves[label]['metrics']['mean_peak_shift']:.1f} nm"
            + "\n"
            + rf"MSE = {curves[label]['metrics']['mse']:.4f}",
            fontsize=7.8,
            color=color if label == "Hybrid" else PALETTE["muted"],
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.2},
        )
    axes[0].set_ylabel("Transmission", fontsize=8.2)

    save(fig, "olt_fig5_representative_spectra")
    plt.close(fig)


def fig6_hardcase():
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    setup()
    refined_path = REPORTS / "archive_legacy_20260615" / "root_reports" / "physics_prior_residual_v3_local_refine_hardcases_7.json"
    refined = json.loads(refined_path.read_text(encoding="utf-8"))
    stages = [
        ("7 hard cases", 0),
        ("Baseline\nCVAE", 0),
        ("Teacher\nreplay", 2),
        ("Residual\nfallback", 1),
        ("+ local TMM\nrefinement", int(refined["selected_nominal_valid_targets"])),
    ]
    total = 7
    fig, ax = plt.subplots(figsize=(7.85, 3.10))
    clean_axis(ax)
    panel_label(ax, "a")
    ax.text(0.02, 0.93, "Hard-case repair flow", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.84, "Selected nominal-valid targets among seven known hard cases.", fontsize=7.2, color=PALETTE["muted"], transform=ax.transAxes)

    xs = np.linspace(0.08, 0.92, len(stages))
    y0 = 0.20
    scale = 0.070
    gap = 0.010
    box_w = 0.125
    for i, ((name, solved), x) in enumerate(zip(stages, xs)):
        unresolved = total - solved
        h_unsolved = unresolved * scale
        h_solved = solved * scale
        base_y = y0
        if unresolved:
            ax.add_patch(
                plt.Rectangle((x - box_w / 2, base_y), box_w, h_unsolved, fc="#E7E9EC", ec="#C9CDD2", lw=0.65, transform=ax.transAxes)
            )
        if solved:
            ax.add_patch(
                plt.Rectangle((x - box_w / 2, base_y + h_unsolved + gap), box_w, h_solved, fc=PALETTE["soft_green"], ec="#93B99B", lw=0.65, transform=ax.transAxes)
            )
        ax.text(x, 0.075, name, fontsize=7.0, ha="center", va="top", transform=ax.transAxes)
        ax.text(x, 0.76, f"{solved}/7", fontsize=9.0, fontweight="bold", color=PALETTE["green"] if solved else PALETTE["muted"], ha="center", transform=ax.transAxes)

        if i < len(stages) - 1:
            next_solved = stages[i + 1][1]
            x_next = xs[i + 1]
            y_mid = base_y + total * scale * 0.5
            verts = [
                (x + box_w / 2, y_mid + 0.040),
                ((x + x_next) / 2, y_mid + 0.075),
                ((x + x_next) / 2, y_mid - 0.075),
                (x_next - box_w / 2, y_mid - 0.040),
            ]
            path = Path(
                [verts[0], verts[1], verts[2], verts[3]],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO],
            )
            ax.add_patch(PathPatch(path, fc="none", ec="#BFC5CB", lw=1.1, transform=ax.transAxes, capstyle="round"))
            delta = next_solved - solved
            if delta:
                arrow_color = PALETTE["green"] if delta > 0 else PALETTE["gold"]
                ax.annotate(
                    "",
                    xy=(x_next - box_w / 2 - 0.014, y_mid + 0.055),
                    xytext=(x + box_w / 2 + 0.014, y_mid + 0.055),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.1),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )
                ax.text(
                    (x + x_next) / 2,
                    y_mid + 0.095,
                    f"{delta:+d}",
                    fontsize=6.8,
                    color=arrow_color,
                    ha="center",
                    transform=ax.transAxes,
                )
    ax.text(0.74, 0.22, "green = selected valid\ngray = unresolved", fontsize=7, color=PALETTE["muted"], transform=ax.transAxes)
    save(fig, "olt_fig6_hardcase_repair")
    plt.close(fig)


def fig7_failure_categories():
    import matplotlib.pyplot as plt

    setup()
    summ = summaries()
    labels = ["Baseline", "Teacher", "Hybrid"]
    robust, low, invalid = [], [], []
    for key in ["baseline", "teacher", "hybrid"]:
        rows = summ[key]["rows"]
        robust.append(sum(bool(r["robust_valid"]) for r in rows))
        low.append(sum(bool(r["nominal_valid"]) and not bool(r["robust_valid"]) for r in rows))
        invalid.append(sum(not bool(r["nominal_valid"]) for r in rows))
    robust = np.array(robust)
    low = np.array(low)
    invalid = np.array(invalid)
    y = np.arange(3)[::-1]

    fig, ax = plt.subplots(figsize=(7.25, 3.45))
    panel_label(ax, "a")
    ax.barh(y, robust, color=PALETTE["green"], height=0.52, label="Robust-valid")
    ax.barh(y, low, left=robust, color=PALETTE["gold"], height=0.52, label="Nominal-valid, yield < 0.5")
    ax.barh(y, invalid, left=robust + low, color=PALETTE["soft_gray"], edgecolor="#BBBBBB", height=0.52, label="Nominal-invalid")
    for yy, a, b, c in zip(y, robust, low, invalid):
        if a > 0:
            ax.text(a / 2, yy, str(a), color="white", ha="center", va="center", fontsize=8)
        if b > 0:
            ax.text(a + b / 2, yy, str(b), color=PALETTE["ink"], ha="center", va="center", fontsize=8)
        if c > 0:
            ax.text(a + b + c / 2, yy, str(c), color=PALETTE["ink"], ha="center", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Targets")
    ax.set_title("Remaining failures after strict robustness validation", fontsize=9, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=7)
    ax.grid(axis="x", color=PALETTE["grid"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "olt_fig7_failure_categories")
    plt.close(fig)


def main():
    # Fig. 1 is maintained as an author-supplied asset in the manuscript package.
    # Keep the batch redraw from overwriting it unintentionally.
    fig2_protocol()
    fig3_valid_rates()
    fig4_yield()
    fig5_spectra()
    fig6_hardcase()
    fig7_failure_categories()
    print("Redrawn OLT figures written to reports/ and reports/olt_latex/figures/.")


if __name__ == "__main__":
    main()
