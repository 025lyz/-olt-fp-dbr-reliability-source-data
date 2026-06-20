import numpy as np


PEAK_HEIGHT = 0.05
PEAK_PROMINENCE = 0.02
PEAK_MATCH_DISTANCE_NM = 50.0

NOMINAL_MSE_THRESHOLD = 0.01
NOMINAL_PEAK_SHIFT_THRESHOLD_NM = 20.0
NOMINAL_MISSING_PEAK_THRESHOLD = 0
NOMINAL_FALSE_PEAK_THRESHOLD = 0

ROBUST_YIELD_THRESHOLD = 0.5
HIGH_ROBUST_YIELD_THRESHOLD = 0.8


def peak_shift_ok(metrics, peak_shift_threshold=NOMINAL_PEAK_SHIFT_THRESHOLD_NM):
    if metrics["target_peak_count"] == 0:
        return True
    return bool(
        np.isfinite(metrics["mean_peak_shift"])
        and metrics["mean_peak_shift"] <= peak_shift_threshold
    )


def nominal_valid(
    metrics,
    mse_threshold=NOMINAL_MSE_THRESHOLD,
    peak_shift_threshold=NOMINAL_PEAK_SHIFT_THRESHOLD_NM,
    missing_peak_threshold=NOMINAL_MISSING_PEAK_THRESHOLD,
    false_peak_threshold=NOMINAL_FALSE_PEAK_THRESHOLD,
):
    return bool(
        metrics["mse"] <= mse_threshold
        and peak_shift_ok(metrics, peak_shift_threshold)
        and metrics["missing_peak_count"] <= missing_peak_threshold
        and metrics["false_peak_count"] <= false_peak_threshold
    )


def nominal_valid_from_args(metrics, args, prefix="success"):
    return nominal_valid(
        metrics,
        mse_threshold=getattr(args, f"{prefix}_mse_threshold"),
        peak_shift_threshold=getattr(args, f"{prefix}_peak_shift_threshold"),
        missing_peak_threshold=getattr(args, f"{prefix}_missing_peak_threshold"),
        false_peak_threshold=getattr(args, f"{prefix}_false_peak_threshold"),
    )


def robust_valid(
    yield_rate,
    nominal_is_valid=True,
    yield_threshold=ROBUST_YIELD_THRESHOLD,
):
    return bool(
        nominal_is_valid and np.isfinite(float(yield_rate)) and float(yield_rate) >= yield_threshold
    )


def high_robust_valid(yield_rate, nominal_is_valid=True):
    return robust_valid(
        yield_rate,
        nominal_is_valid=nominal_is_valid,
        yield_threshold=HIGH_ROBUST_YIELD_THRESHOLD,
    )


def candidate_covered(nominal_valid_candidate_count):
    return int(nominal_valid_candidate_count) > 0


def protocol_summary():
    return {
        "peak_height": PEAK_HEIGHT,
        "peak_prominence": PEAK_PROMINENCE,
        "peak_match_distance_nm": PEAK_MATCH_DISTANCE_NM,
        "nominal_mse_threshold": NOMINAL_MSE_THRESHOLD,
        "nominal_peak_shift_threshold_nm": NOMINAL_PEAK_SHIFT_THRESHOLD_NM,
        "nominal_missing_peak_threshold": NOMINAL_MISSING_PEAK_THRESHOLD,
        "nominal_false_peak_threshold": NOMINAL_FALSE_PEAK_THRESHOLD,
        "robust_yield_threshold": ROBUST_YIELD_THRESHOLD,
        "high_robust_yield_threshold": HIGH_ROBUST_YIELD_THRESHOLD,
    }
