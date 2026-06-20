import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


"""
Physics-aware FP-DBR dataset generator.

Default structure:

    Air / (Ge, SiO2)^N / SiO2 cavity / (Ge, SiO2)^N / substrate

Compared with pure uniform random sampling, this script mixes:

1. uniform samples for baseline coverage;
2. quarter-wave DBR + FP resonance prior samples;
3. high-reflectivity prior samples with larger N and smaller thickness jitter;
4. off-design cavity samples for negative / broad-spectrum coverage.

Output .npz keeps the old keys:

    params, spectra, wavelengths

and adds physics labels:

    target_wavelengths, resonance_orders, strategy_ids, strategy_names,
    peak_count, peak_wavelengths, peak_transmissions, peak_fwhm,
    peak_q, max_transmission, mean_transmission, stopband_mean_transmission

The script is intended to run on the data-generation machine, not necessarily
on the training machine.
"""


N_AIR = 1.0
WORKER_CONFIG = None
WAVELENGTHS = None
N_H_ARRAY = None
N_L_ARRAY = None
N_CAVITY_ARRAY = None

STRATEGY_NAMES = np.array(
    [
        "uniform_random",
        "quarter_wave_fp",
        "high_reflectivity_fp",
        "quarter_wave_off_design_cavity",
    ]
)


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate a physics-aware FP-DBR TMM dataset."
    )
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--output-dir", type=str, default=str(script_dir))
    parser.add_argument("--output-name", type=str, default=None)

    parser.add_argument("--ge-file", type=str, default="expriment_Ge.txt")
    parser.add_argument("--sio2-file", type=str, default="expriment_Sio2.txt")
    parser.add_argument(
        "--material-wavelength-unit",
        choices=["auto", "um", "nm"],
        default="auto",
        help="Unit of the first column in material txt files.",
    )

    parser.add_argument("--wl-start", type=float, default=3000.0)
    parser.add_argument("--wl-end", type=float, default=5000.0)
    parser.add_argument("--wl-points", type=int, default=1000)

    parser.add_argument("--d-h-min", type=float, default=100.0)
    parser.add_argument("--d-h-max", type=float, default=1000.0)
    parser.add_argument("--d-l-min", type=float, default=100.0)
    parser.add_argument("--d-l-max", type=float, default=1500.0)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=10)
    parser.add_argument("--lc-min", type=float, default=500.0)
    parser.add_argument("--lc-max", type=float, default=5000.0)
    parser.add_argument("--substrate-index", type=float, default=3.4)

    parser.add_argument(
        "--uniform-fraction",
        type=float,
        default=0.20,
        help="Fraction of old-style uniform random samples.",
    )
    parser.add_argument(
        "--off-design-fraction",
        type=float,
        default=0.15,
        help="Fraction of quarter-wave DBR samples with non-resonant cavity length.",
    )
    parser.add_argument(
        "--high-reflectivity-fraction",
        type=float,
        default=0.25,
        help="Fraction of high-N, low-jitter resonant samples.",
    )
    parser.add_argument(
        "--jitter-frac",
        type=float,
        default=0.08,
        help="Relative thickness jitter for physics-prior samples.",
    )
    parser.add_argument(
        "--high-q-jitter-frac",
        type=float,
        default=0.035,
        help="Relative thickness jitter for high-reflectivity samples.",
    )

    parser.add_argument("--peak-height", type=float, default=0.05)
    parser.add_argument("--peak-prominence", type=float, default=0.02)
    parser.add_argument("--max-saved-peaks", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--no-compress", action="store_true")

    return parser.parse_args()


def resolve_input_file(path_text):
    path = Path(path_text).expanduser()
    if path.exists():
        return path

    script_relative = Path(__file__).resolve().parent / path_text
    if script_relative.exists():
        return script_relative

    cwd_relative = Path.cwd() / path_text
    if cwd_relative.exists():
        return cwd_relative

    raise FileNotFoundError(
        f"Material file not found: {path_text}. Tried current working directory "
        f"and script directory."
    )


def load_complex_index(filepath, target_wavelengths_nm, wavelength_unit="auto"):
    data = np.loadtxt(filepath)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{filepath} must contain at least three columns: wavelength, n, k."
        )

    wl_raw = data[:, 0].astype(float)
    n_measured = data[:, 1].astype(float)
    k_measured = data[:, 2].astype(float)

    if wavelength_unit == "um":
        wl_measured_nm = wl_raw * 1000.0
    elif wavelength_unit == "nm":
        wl_measured_nm = wl_raw
    else:
        wl_measured_nm = wl_raw * 1000.0 if np.nanmax(wl_raw) < 100.0 else wl_raw

    order = np.argsort(wl_measured_nm)
    wl_measured_nm = wl_measured_nm[order]
    n_measured = n_measured[order]
    k_measured = k_measured[order]

    if target_wavelengths_nm[0] < wl_measured_nm[0] or target_wavelengths_nm[-1] > wl_measured_nm[-1]:
        raise ValueError(
            f"Material file {filepath} covers {wl_measured_nm[0]:.1f}-"
            f"{wl_measured_nm[-1]:.1f} nm, but target range is "
            f"{target_wavelengths_nm[0]:.1f}-{target_wavelengths_nm[-1]:.1f} nm."
        )

    n_interp = np.interp(target_wavelengths_nm, wl_measured_nm, n_measured)
    k_interp = np.interp(target_wavelengths_nm, wl_measured_nm, k_measured)
    return n_interp + 1j * k_interp


def init_worker(config, wavelengths, n_h_array, n_l_array, n_cavity_array):
    global WORKER_CONFIG, WAVELENGTHS, N_H_ARRAY, N_L_ARRAY, N_CAVITY_ARRAY
    WORKER_CONFIG = config
    WAVELENGTHS = wavelengths
    N_H_ARRAY = n_h_array
    N_L_ARRAY = n_l_array
    N_CAVITY_ARRAY = n_cavity_array


def clip(value, lower, upper):
    return float(np.clip(value, lower, upper))


def index_real_at(array, wavelength_nm):
    return float(np.interp(wavelength_nm, WAVELENGTHS, np.real(array)))


def sample_resonance_order(rng, lambda0, n_cavity, lc_min, lc_max):
    m_min = int(math.ceil(2.0 * n_cavity * lc_min / lambda0))
    m_max = int(math.floor(2.0 * n_cavity * lc_max / lambda0))
    if m_max < m_min:
        return -1, rng.uniform(lc_min, lc_max)

    m = int(rng.integers(m_min, m_max + 1))
    lc = m * lambda0 / (2.0 * n_cavity)
    return m, lc


def jitter_and_clip(rng, base_value, jitter_frac, lower, upper):
    sigma = max(abs(base_value) * jitter_frac, 1.0)
    return clip(rng.normal(base_value, sigma), lower, upper)


def choose_strategy(rng, config):
    r = rng.random()
    uniform_cut = config["uniform_fraction"]
    off_design_cut = uniform_cut + config["off_design_fraction"]
    high_reflectivity_cut = off_design_cut + config["high_reflectivity_fraction"]

    if r < uniform_cut:
        return 0
    if r < off_design_cut:
        return 3
    if r < high_reflectivity_cut:
        return 2
    return 1


def sample_structure(rng):
    config = WORKER_CONFIG
    strategy_id = choose_strategy(rng, config)

    d_h_min, d_h_max = config["d_h_bounds"]
    d_l_min, d_l_max = config["d_l_bounds"]
    lc_min, lc_max = config["lc_bounds"]
    n_min, n_max = config["n_bounds"]
    wl_min, wl_max = config["wavelength_bounds"]

    if strategy_id == 0:
        d_h = rng.uniform(d_h_min, d_h_max)
        d_l = rng.uniform(d_l_min, d_l_max)
        periods = int(rng.integers(n_min, n_max + 1))
        lc = rng.uniform(lc_min, lc_max)
        target_wl = np.nan
        resonance_order = -1
        prior_d_h = np.nan
        prior_d_l = np.nan
        prior_lc = np.nan
        return (
            np.array([d_h, d_l, periods, lc], dtype=float),
            target_wl,
            resonance_order,
            prior_d_h,
            prior_d_l,
            prior_lc,
            strategy_id,
        )

    target_wl = float(rng.uniform(wl_min, wl_max))
    n_h = index_real_at(N_H_ARRAY, target_wl)
    n_l = index_real_at(N_L_ARRAY, target_wl)
    n_c = index_real_at(N_CAVITY_ARRAY, target_wl)

    prior_d_h = target_wl / (4.0 * n_h)
    prior_d_l = target_wl / (4.0 * n_l)
    resonance_order, prior_lc = sample_resonance_order(
        rng, target_wl, n_c, lc_min, lc_max
    )

    if strategy_id == 2:
        jitter_frac = config["high_q_jitter_frac"]
        low_n = max(n_min, int(math.ceil((n_min + n_max) / 2)))
        periods = int(rng.integers(low_n, n_max + 1))
    else:
        jitter_frac = config["jitter_frac"]
        periods = int(rng.integers(n_min, n_max + 1))

    d_h = jitter_and_clip(rng, prior_d_h, jitter_frac, d_h_min, d_h_max)
    d_l = jitter_and_clip(rng, prior_d_l, jitter_frac, d_l_min, d_l_max)

    if strategy_id == 3:
        lc = rng.uniform(lc_min, lc_max)
        resonance_order = -1
    else:
        lc = jitter_and_clip(rng, prior_lc, jitter_frac, lc_min, lc_max)

    return (
        np.array([d_h, d_l, periods, lc], dtype=float),
        target_wl,
        resonance_order,
        prior_d_h,
        prior_d_l,
        prior_lc,
        strategy_id,
    )


def simulate_tmm(params):
    try:
        from tmm import coh_tmm
    except ImportError as exc:
        raise ImportError(
            "The 'tmm' package is required for dataset generation. "
            "Install it on the generation machine with: pip install tmm"
        ) from exc

    config = WORKER_CONFIG
    d_h, d_l, periods_float, lc = params
    periods = int(round(periods_float))

    dbr_thicknesses = [d_h, d_l] * periods
    thicknesses = [np.inf] + dbr_thicknesses + [lc] + dbr_thicknesses + [np.inf]
    spectrum = np.zeros(len(WAVELENGTHS), dtype=np.float32)

    for i, wl in enumerate(WAVELENGTHS):
        n_h = N_H_ARRAY[i]
        n_l = N_L_ARRAY[i]
        n_cavity = N_CAVITY_ARRAY[i]
        dbr_indices = [n_h, n_l] * periods
        indices = [N_AIR] + dbr_indices + [n_cavity] + dbr_indices + [
            config["substrate_index"]
        ]
        result = coh_tmm("s", indices, thicknesses, 0.0, wl)
        spectrum[i] = np.float32(result["T"])

    return spectrum


def extract_spectral_features(spectrum):
    try:
        from scipy.signal import find_peaks, peak_widths
    except ImportError as exc:
        raise ImportError(
            "The 'scipy' package is required for spectral feature extraction. "
            "Install it on the generation machine with: pip install scipy"
        ) from exc

    config = WORKER_CONFIG
    max_peaks = config["max_saved_peaks"]

    peak_wls = np.full(max_peaks, np.nan, dtype=np.float32)
    peak_vals = np.full(max_peaks, np.nan, dtype=np.float32)
    peak_fwhm = np.full(max_peaks, np.nan, dtype=np.float32)
    peak_q = np.full(max_peaks, np.nan, dtype=np.float32)

    peaks, props = find_peaks(
        spectrum,
        height=config["peak_height"],
        prominence=config["peak_prominence"],
    )

    if len(peaks) == 0:
        return {
            "peak_count": 0,
            "peak_wavelengths": peak_wls,
            "peak_transmissions": peak_vals,
            "peak_fwhm": peak_fwhm,
            "peak_q": peak_q,
            "max_transmission": float(np.max(spectrum)),
            "mean_transmission": float(np.mean(spectrum)),
            "stopband_mean_transmission": float(np.mean(spectrum)),
        }

    sort_order = np.argsort(spectrum[peaks])[::-1]
    peaks_sorted = peaks[sort_order]

    widths_result = peak_widths(spectrum, peaks_sorted, rel_height=0.5)
    width_samples = widths_result[0]
    wavelength_step = float(np.mean(np.diff(WAVELENGTHS)))
    fwhm_nm = width_samples * wavelength_step

    saved = min(max_peaks, len(peaks_sorted))
    selected_peaks = peaks_sorted[:saved]
    selected_wls = WAVELENGTHS[selected_peaks]

    peak_wls[:saved] = selected_wls
    peak_vals[:saved] = spectrum[selected_peaks]
    peak_fwhm[:saved] = fwhm_nm[:saved]
    valid_width = peak_fwhm[:saved] > 0.0
    saved_q = peak_q[:saved]
    saved_q[valid_width] = peak_wls[:saved][valid_width] / peak_fwhm[:saved][valid_width]
    peak_q[:saved] = saved_q

    stopband_mask = np.ones(len(WAVELENGTHS), dtype=bool)
    for wl0, width in zip(peak_wls[:saved], peak_fwhm[:saved]):
        if np.isfinite(wl0):
            exclusion = max(float(width), 30.0)
            stopband_mask &= np.abs(WAVELENGTHS - wl0) > exclusion

    if np.any(stopband_mask):
        stopband_mean = float(np.mean(spectrum[stopband_mask]))
    else:
        stopband_mean = float(np.mean(spectrum))

    return {
        "peak_count": int(len(peaks)),
        "peak_wavelengths": peak_wls,
        "peak_transmissions": peak_vals,
        "peak_fwhm": peak_fwhm,
        "peak_q": peak_q,
        "max_transmission": float(np.max(spectrum)),
        "mean_transmission": float(np.mean(spectrum)),
        "stopband_mean_transmission": stopband_mean,
    }


def simulate_single(seed):
    rng = np.random.default_rng(int(seed))
    (
        params,
        target_wl,
        resonance_order,
        prior_d_h,
        prior_d_l,
        prior_lc,
        strategy_id,
    ) = sample_structure(rng)
    spectrum = simulate_tmm(params)
    features = extract_spectral_features(spectrum)

    return {
        "params": params.astype(np.float32),
        "spectrum": spectrum,
        "target_wavelength": np.float32(target_wl),
        "resonance_order": np.int16(resonance_order),
        "prior_d_h": np.float32(prior_d_h),
        "prior_d_l": np.float32(prior_d_l),
        "prior_lc": np.float32(prior_lc),
        "strategy_id": np.int8(strategy_id),
        **features,
    }


def build_config(args):
    total_fraction = (
        args.uniform_fraction + args.off_design_fraction + args.high_reflectivity_fraction
    )
    if total_fraction > 1.0:
        raise ValueError(
            "--uniform-fraction + --off-design-fraction + "
            "--high-reflectivity-fraction must be <= 1.0."
        )

    if args.n_min > args.n_max:
        raise ValueError("--n-min must be <= --n-max.")

    return {
        "d_h_bounds": (args.d_h_min, args.d_h_max),
        "d_l_bounds": (args.d_l_min, args.d_l_max),
        "lc_bounds": (args.lc_min, args.lc_max),
        "n_bounds": (args.n_min, args.n_max),
        "wavelength_bounds": (args.wl_start, args.wl_end),
        "substrate_index": args.substrate_index,
        "uniform_fraction": args.uniform_fraction,
        "off_design_fraction": args.off_design_fraction,
        "high_reflectivity_fraction": args.high_reflectivity_fraction,
        "jitter_frac": args.jitter_frac,
        "high_q_jitter_frac": args.high_q_jitter_frac,
        "peak_height": args.peak_height,
        "peak_prominence": args.peak_prominence,
        "max_saved_peaks": args.max_saved_peaks,
    }


def allocate_outputs(num_samples, num_wavelengths, max_saved_peaks):
    return {
        "params": np.zeros((num_samples, 4), dtype=np.float32),
        "spectra": np.zeros((num_samples, num_wavelengths), dtype=np.float32),
        "target_wavelengths": np.full(num_samples, np.nan, dtype=np.float32),
        "resonance_orders": np.full(num_samples, -1, dtype=np.int16),
        "prior_d_h": np.full(num_samples, np.nan, dtype=np.float32),
        "prior_d_l": np.full(num_samples, np.nan, dtype=np.float32),
        "prior_lc": np.full(num_samples, np.nan, dtype=np.float32),
        "strategy_ids": np.zeros(num_samples, dtype=np.int8),
        "peak_count": np.zeros(num_samples, dtype=np.int16),
        "peak_wavelengths": np.full(
            (num_samples, max_saved_peaks), np.nan, dtype=np.float32
        ),
        "peak_transmissions": np.full(
            (num_samples, max_saved_peaks), np.nan, dtype=np.float32
        ),
        "peak_fwhm": np.full((num_samples, max_saved_peaks), np.nan, dtype=np.float32),
        "peak_q": np.full((num_samples, max_saved_peaks), np.nan, dtype=np.float32),
        "max_transmission": np.zeros(num_samples, dtype=np.float32),
        "mean_transmission": np.zeros(num_samples, dtype=np.float32),
        "stopband_mean_transmission": np.zeros(num_samples, dtype=np.float32),
    }


def store_result(outputs, index, result):
    outputs["params"][index] = result["params"]
    outputs["spectra"][index] = result["spectrum"]
    outputs["target_wavelengths"][index] = result["target_wavelength"]
    outputs["resonance_orders"][index] = result["resonance_order"]
    outputs["prior_d_h"][index] = result["prior_d_h"]
    outputs["prior_d_l"][index] = result["prior_d_l"]
    outputs["prior_lc"][index] = result["prior_lc"]
    outputs["strategy_ids"][index] = result["strategy_id"]
    outputs["peak_count"][index] = result["peak_count"]
    outputs["peak_wavelengths"][index] = result["peak_wavelengths"]
    outputs["peak_transmissions"][index] = result["peak_transmissions"]
    outputs["peak_fwhm"][index] = result["peak_fwhm"]
    outputs["peak_q"][index] = result["peak_q"]
    outputs["max_transmission"][index] = result["max_transmission"]
    outputs["mean_transmission"][index] = result["mean_transmission"]
    outputs["stopband_mean_transmission"][index] = result[
        "stopband_mean_transmission"
    ]


def summarize(outputs):
    strategy_counts = {
        STRATEGY_NAMES[i]: int(np.sum(outputs["strategy_ids"] == i))
        for i in range(len(STRATEGY_NAMES))
    }
    finite_q = outputs["peak_q"][np.isfinite(outputs["peak_q"])]
    summary = {
        "num_samples": int(len(outputs["params"])),
        "strategy_counts": strategy_counts,
        "peak_count_mean": float(np.mean(outputs["peak_count"])),
        "peak_count_median": float(np.median(outputs["peak_count"])),
        "no_peak_fraction": float(np.mean(outputs["peak_count"] == 0)),
        "max_transmission_mean": float(np.mean(outputs["max_transmission"])),
        "max_transmission_p10": float(np.percentile(outputs["max_transmission"], 10)),
        "max_transmission_p90": float(np.percentile(outputs["max_transmission"], 90)),
        "q_p50": float(np.percentile(finite_q, 50)) if finite_q.size else None,
        "q_p90": float(np.percentile(finite_q, 90)) if finite_q.size else None,
    }
    return summary


def main():
    args = parse_args()
    config = build_config(args)

    wavelengths = np.linspace(args.wl_start, args.wl_end, args.wl_points).astype(
        np.float64
    )

    ge_file = resolve_input_file(args.ge_file)
    sio2_file = resolve_input_file(args.sio2_file)

    print("Loading measured complex refractive indices...")
    print(f"  Ge:   {ge_file}")
    print(f"  SiO2: {sio2_file}")

    n_h_array = load_complex_index(
        ge_file, wavelengths, wavelength_unit=args.material_wavelength_unit
    )
    n_l_array = load_complex_index(
        sio2_file, wavelengths, wavelength_unit=args.material_wavelength_unit
    )
    n_cavity_array = n_l_array

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        output_name = f"fp_dbr_data_{args.num_samples}_physics_aware_experiment.npz"
    output_path = output_dir / output_name

    print(f"Generating {args.num_samples} samples")
    print(f"Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_path}")

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=args.num_samples)
    outputs = allocate_outputs(args.num_samples, len(wavelengths), args.max_saved_peaks)

    start_time = time.time()

    if args.workers <= 1:
        init_worker(config, wavelengths, n_h_array, n_l_array, n_cavity_array)
        iterator = (simulate_single(seed) for seed in seeds)
        for i, result in enumerate(tqdm(iterator, total=args.num_samples)):
            store_result(outputs, i, result)
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(config, wavelengths, n_h_array, n_l_array, n_cavity_array),
        ) as pool:
            iterator = pool.imap(simulate_single, seeds, chunksize=args.chunksize)
            for i, result in enumerate(tqdm(iterator, total=args.num_samples)):
                store_result(outputs, i, result)

    elapsed = time.time() - start_time
    summary = summarize(outputs)
    metadata = {
        "generator": "dataset/generatedata.py",
        "sampling": "physics_aware_mixture",
        "structure": "Air/(Ge,SiO2)^N/SiO2 cavity/(Ge,SiO2)^N/substrate",
        "args": vars(args),
        "config": config,
        "ge_file": str(ge_file),
        "sio2_file": str(sio2_file),
        "elapsed_seconds": elapsed,
        "summary": summary,
    }

    save_fn = np.savez if args.no_compress else np.savez_compressed
    save_fn(
        output_path,
        wavelengths=wavelengths.astype(np.float32),
        strategy_names=STRATEGY_NAMES,
        metadata_json=np.array(json.dumps(metadata, indent=2)),
        **outputs,
    )

    print("Dataset generation complete.")
    print(f"Elapsed: {elapsed / 60.0:.2f} min")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
