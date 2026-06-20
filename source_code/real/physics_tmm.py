import argparse
import json
from pathlib import Path

import numpy as np


N_AIR = 1.0


def resolve_path(path_text, base_dir=None):
    path = Path(path_text).expanduser()
    if path.exists():
        return path

    candidates = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / path_text)

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / path_text,
            here.parent / path_text,
            here.parent / "dataset" / path_text,
            Path.cwd() / path_text,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        f"Could not find file: {path_text}. Put the material txt files in the "
        f"project root or dataset directory, or pass absolute paths with "
        f"--ge-file/--sio2-file. Tried: {tried}"
    )


def load_complex_index(filepath, target_wavelengths_nm, wavelength_unit="auto"):
    data = np.loadtxt(filepath)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{filepath} must contain at least three columns: wavelength, n, k."
        )

    wavelengths = data[:, 0].astype(float)
    n_values = data[:, 1].astype(float)
    k_values = data[:, 2].astype(float)

    if wavelength_unit == "um":
        wavelengths_nm = wavelengths * 1000.0
    elif wavelength_unit == "nm":
        wavelengths_nm = wavelengths
    else:
        wavelengths_nm = wavelengths * 1000.0 if np.nanmax(wavelengths) < 100.0 else wavelengths

    order = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[order]
    n_values = n_values[order]
    k_values = k_values[order]

    if target_wavelengths_nm[0] < wavelengths_nm[0] or target_wavelengths_nm[-1] > wavelengths_nm[-1]:
        raise ValueError(
            f"{filepath} covers {wavelengths_nm[0]:.1f}-{wavelengths_nm[-1]:.1f} nm, "
            f"but target range is {target_wavelengths_nm[0]:.1f}-"
            f"{target_wavelengths_nm[-1]:.1f} nm."
        )

    n_interp = np.interp(target_wavelengths_nm, wavelengths_nm, n_values)
    k_interp = np.interp(target_wavelengths_nm, wavelengths_nm, k_values)
    return n_interp + 1j * k_interp


class FPDBRTMM:
    def __init__(
        self,
        wavelengths_nm,
        ge_file="expriment_Ge.txt",
        sio2_file="expriment_Sio2.txt",
        material_wavelength_unit="auto",
        substrate_index=3.4,
    ):
        self.wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
        self.substrate_index = substrate_index
        self.ge_file = resolve_path(ge_file)
        self.sio2_file = resolve_path(sio2_file)
        self.n_h = load_complex_index(
            self.ge_file, self.wavelengths_nm, material_wavelength_unit
        )
        self.n_l = load_complex_index(
            self.sio2_file, self.wavelengths_nm, material_wavelength_unit
        )
        self.n_cavity = self.n_l

    def simulate(self, params):
        try:
            from tmm import coh_tmm
        except ImportError as exc:
            raise ImportError(
                "The 'tmm' package is required for TMM validation. "
                "Install it with: pip install tmm"
            ) from exc

        d_h, d_l, periods, lc = np.asarray(params, dtype=float)
        periods = int(round(periods))
        periods = max(periods, 1)

        dbr_thicknesses = [d_h, d_l] * periods
        thicknesses = [np.inf] + dbr_thicknesses + [lc] + dbr_thicknesses + [np.inf]
        spectrum = np.zeros(len(self.wavelengths_nm), dtype=np.float32)

        for i, wl in enumerate(self.wavelengths_nm):
            dbr_indices = [self.n_h[i], self.n_l[i]] * periods
            indices = [N_AIR] + dbr_indices + [self.n_cavity[i]] + dbr_indices + [
                self.substrate_index
            ]
            spectrum[i] = np.float32(coh_tmm("s", indices, thicknesses, 0.0, wl)["T"])

        return spectrum

    def simulate_perturbed(
        self,
        params,
        rng,
        thickness_sigma_nm=0.0,
        index_sigma_frac=0.0,
        periods_override=None,
    ):
        try:
            from tmm import coh_tmm
        except ImportError as exc:
            raise ImportError(
                "The 'tmm' package is required for TMM validation. "
                "Install it with: pip install tmm"
            ) from exc

        d_h, d_l, periods, lc = np.asarray(params, dtype=float)
        periods = int(round(periods if periods_override is None else periods_override))
        periods = max(periods, 1)

        if thickness_sigma_nm > 0.0:
            left_h = rng.normal(d_h, thickness_sigma_nm, size=periods)
            left_l = rng.normal(d_l, thickness_sigma_nm, size=periods)
            right_h = rng.normal(d_h, thickness_sigma_nm, size=periods)
            right_l = rng.normal(d_l, thickness_sigma_nm, size=periods)
            cavity_thickness = float(rng.normal(lc, thickness_sigma_nm))
        else:
            left_h = np.full(periods, d_h, dtype=float)
            left_l = np.full(periods, d_l, dtype=float)
            right_h = np.full(periods, d_h, dtype=float)
            right_l = np.full(periods, d_l, dtype=float)
            cavity_thickness = float(lc)

        left_h = np.clip(left_h, 1.0, None)
        left_l = np.clip(left_l, 1.0, None)
        right_h = np.clip(right_h, 1.0, None)
        right_l = np.clip(right_l, 1.0, None)
        cavity_thickness = max(cavity_thickness, 1.0)

        left_dbr_thicknesses = []
        right_dbr_thicknesses = []
        for i in range(periods):
            left_dbr_thicknesses.extend([float(left_h[i]), float(left_l[i])])
            right_dbr_thicknesses.extend([float(right_h[i]), float(right_l[i])])
        thicknesses = (
            [np.inf]
            + left_dbr_thicknesses
            + [cavity_thickness]
            + right_dbr_thicknesses
            + [np.inf]
        )

        if index_sigma_frac > 0.0:
            h_scale = 1.0 + rng.normal(0.0, index_sigma_frac)
            l_scale = 1.0 + rng.normal(0.0, index_sigma_frac)
            c_scale = 1.0 + rng.normal(0.0, index_sigma_frac)
        else:
            h_scale = l_scale = c_scale = 1.0

        spectrum = np.zeros(len(self.wavelengths_nm), dtype=np.float32)
        for i, wl in enumerate(self.wavelengths_nm):
            n_h = complex(np.real(self.n_h[i]) * h_scale, np.imag(self.n_h[i]) * h_scale)
            n_l = complex(np.real(self.n_l[i]) * l_scale, np.imag(self.n_l[i]) * l_scale)
            n_cavity = complex(
                np.real(self.n_cavity[i]) * c_scale,
                np.imag(self.n_cavity[i]) * c_scale,
            )
            dbr_indices = [n_h, n_l] * periods
            indices = [N_AIR] + dbr_indices + [n_cavity] + dbr_indices + [
                self.substrate_index
            ]
            spectrum[i] = np.float32(coh_tmm("s", indices, thicknesses, 0.0, wl)["T"])

        return spectrum

    def simulate_batch(self, params_batch):
        params_batch = np.asarray(params_batch, dtype=float)
        return np.stack([self.simulate(params) for params in params_batch], axis=0)


def detect_peaks(spectrum, wavelengths_nm, height=0.05, prominence=0.02, max_peaks=None):
    try:
        from scipy.signal import find_peaks, peak_widths
    except ImportError as exc:
        raise ImportError(
            "The 'scipy' package is required for peak metrics. Install it with: pip install scipy"
        ) from exc

    spectrum = np.asarray(spectrum, dtype=float)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    peaks, props = find_peaks(spectrum, height=height, prominence=prominence)
    if len(peaks) == 0:
        return []

    order = np.argsort(spectrum[peaks])[::-1]
    peaks = peaks[order]
    widths = peak_widths(spectrum, peaks, rel_height=0.5)[0]
    wl_step = float(np.mean(np.diff(wavelengths_nm)))

    results = []
    for idx, width_samples in zip(peaks, widths):
        fwhm = float(width_samples * wl_step)
        wl = float(wavelengths_nm[idx])
        results.append(
            {
                "index": int(idx),
                "wavelength": wl,
                "transmission": float(spectrum[idx]),
                "fwhm": fwhm,
                "q": float(wl / fwhm) if fwhm > 0 else np.nan,
            }
        )

    if max_peaks is not None:
        results = results[:max_peaks]
    return results


def match_peaks(target_peaks, pred_peaks, distance_threshold_nm=50.0):
    if len(target_peaks) == 0:
        return {
            "matched": [],
            "missing_count": 0,
            "false_count": len(pred_peaks),
            "mean_peak_shift": np.nan,
            "mean_peak_transmission_error": np.nan,
            "mean_fwhm_error": np.nan,
            "mean_q_error": np.nan,
        }

    used_pred = set()
    matched = []
    for target_peak in target_peaks:
        best_j = None
        best_dist = float("inf")
        for j, pred_peak in enumerate(pred_peaks):
            if j in used_pred:
                continue
            dist = abs(pred_peak["wavelength"] - target_peak["wavelength"])
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None and best_dist <= distance_threshold_nm:
            used_pred.add(best_j)
            pred_peak = pred_peaks[best_j]
            matched.append(
                {
                    "target_wavelength": target_peak["wavelength"],
                    "pred_wavelength": pred_peak["wavelength"],
                    "shift": best_dist,
                    "transmission_error": abs(
                        pred_peak["transmission"] - target_peak["transmission"]
                    ),
                    "fwhm_error": abs(pred_peak["fwhm"] - target_peak["fwhm"]),
                    "q_error": abs(pred_peak["q"] - target_peak["q"])
                    if np.isfinite(pred_peak["q"]) and np.isfinite(target_peak["q"])
                    else np.nan,
                }
            )

    missing_count = len(target_peaks) - len(matched)
    false_count = len(pred_peaks) - len(used_pred)

    def mean_or_nan(key):
        values = [item[key] for item in matched if np.isfinite(item[key])]
        return float(np.mean(values)) if values else np.nan

    return {
        "matched": matched,
        "missing_count": int(missing_count),
        "false_count": int(false_count),
        "mean_peak_shift": mean_or_nan("shift"),
        "mean_peak_transmission_error": mean_or_nan("transmission_error"),
        "mean_fwhm_error": mean_or_nan("fwhm_error"),
        "mean_q_error": mean_or_nan("q_error"),
    }


def spectral_metrics(
    target,
    pred,
    wavelengths_nm,
    peak_height=0.05,
    peak_prominence=0.02,
    peak_distance_threshold_nm=50.0,
):
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-12

    target_peaks = detect_peaks(
        target, wavelengths_nm, height=peak_height, prominence=peak_prominence
    )
    pred_peaks = detect_peaks(
        pred, wavelengths_nm, height=peak_height, prominence=peak_prominence
    )
    peak_match = match_peaks(
        target_peaks, pred_peaks, distance_threshold_nm=peak_distance_threshold_nm
    )

    target_centered = target - np.mean(target)
    pred_centered = pred - np.mean(pred)
    denom = np.linalg.norm(target_centered) * np.linalg.norm(pred_centered)
    corr = float(np.dot(target_centered, pred_centered) / (denom + eps))

    overlap = float(
        np.dot(target, pred) / ((np.linalg.norm(target) * np.linalg.norm(pred)) + eps)
    )

    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "max_abs_error": float(np.max(np.abs(pred - target))),
        "correlation": corr,
        "spectral_overlap": overlap,
        "target_peak_count": int(len(target_peaks)),
        "pred_peak_count": int(len(pred_peaks)),
        "missing_peak_count": peak_match["missing_count"],
        "false_peak_count": peak_match["false_count"],
        "mean_peak_shift": peak_match["mean_peak_shift"],
        "mean_peak_transmission_error": peak_match["mean_peak_transmission_error"],
        "mean_fwhm_error": peak_match["mean_fwhm_error"],
        "mean_q_error": peak_match["mean_q_error"],
    }


def add_tmm_args(parser):
    parser.add_argument("--ge-file", type=str, default="expriment_Ge.txt")
    parser.add_argument("--sio2-file", type=str, default="expriment_Sio2.txt")
    parser.add_argument(
        "--material-wavelength-unit", choices=["auto", "um", "nm"], default="auto"
    )
    parser.add_argument("--substrate-index", type=float, default=3.4)
    parser.add_argument("--peak-height", type=float, default=0.05)
    parser.add_argument("--peak-prominence", type=float, default=0.02)
    parser.add_argument("--peak-distance-threshold", type=float, default=50.0)
    return parser


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )
