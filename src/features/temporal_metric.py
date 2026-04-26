import numpy as np
from scipy.stats import linregress, skew, kurtosis


def compute_std(feature_curve):
    return np.std(feature_curve)


def compute_mean(feature_curve):
    return np.mean(feature_curve)


def compute_slope(feature_curve):
    x = np.arange(len(feature_curve))
    slope, _, _, _, _ = linregress(x, feature_curve)
    return slope

def compute_short_term_stability(feature_curve, window_size_frames=50):
    """
    Computes average local variance across sliding windows.
    """
    
    local_variances = []
    
    for start in range(0, len(feature_curve) - window_size_frames, window_size_frames // 2):
        window = feature_curve[start:start + window_size_frames]
        local_variances.append(np.var(window))
    
    return np.mean(local_variances)

def compute_half_contrast(feature_curve):
    
    midpoint = len(feature_curve) // 2
    
    first_half = feature_curve[:midpoint]
    second_half = feature_curve[midpoint:]
    
    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)
    
    return abs(mean_second - mean_first)

def compute_percentile(feature_curve, percentile):
    """
    Compute a percentile value for a feature curve.
    """

    return np.percentile(feature_curve, percentile)

def compute_range(feature_curve):
    """
    Compute the range of a feature curve.
    """

    return np.max(feature_curve) - np.min(feature_curve)

def compute_interpercentile_range(curve, low=10, high=90):

    p_low = np.percentile(curve, low)
    p_high = np.percentile(curve, high)

    return p_high - p_low

def compute_skewness(curve):

    return skew(curve)

def compute_kurtosis(curve):

    return kurtosis(curve)

DESCRIPTOR_FUNCTIONS = {
    "mean": compute_mean,
    "std": compute_std,
    "slope": compute_slope,
    "stability": compute_short_term_stability,
    "half_contrast": compute_half_contrast,
    "p5": lambda x: compute_percentile(x, 5),
    "p10": lambda x: compute_percentile(x, 10),
    "p25": lambda x: compute_percentile(x, 25),
    "p50": lambda x: compute_percentile(x, 50),
    "p75": lambda x: compute_percentile(x, 75),
    "p90": lambda x: compute_percentile(x, 90),
    "p95": lambda x: compute_percentile(x, 95),
    "range": compute_range,
    "interpercentile_range": lambda x: compute_interpercentile_range(x, 10, 90),
    "skewness": compute_skewness,
    "kurtosis": compute_kurtosis
}

def apply_selected_descriptors(curve, name, descriptors):

    results = {}

    for d in descriptors:

        func = DESCRIPTOR_FUNCTIONS[d]

        results[f"{name}_{d}"] = func(curve)

    return results