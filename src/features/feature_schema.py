FEATURE_DESCRIPTOR_SCHEMA = {

    "rms": ["std", "p10", "p90", "kurtosis", "interpercentile_range", "slope"],

    "centroid": ["mean", "std", "slope", "stability", "p10", "p90", "kurtosis"],

    "bandwidth": ["mean", "std", "p10", "slope"],

    "rolloff": ["mean", "std", "slope"],

    "flatness": ["mean", "std", "skewness"],

    "contrast": ["mean", "std", "p10", "p90", "slope"],

    "zcr": ["mean", "std", "kurtosis"],

    "flux_positive": ["mean", "std", "p90", "skewness"],
    
    "flux_negative": ["mean", "std", "p90", "skewness"],

    "onset_strength": ["mean", "std", "p90", "p10", "skewness"],

    "tonal_entropy": ["mean", "std", "slope"],

    "chroma": ["mean", "std", "kurtosis"],

    "mfcc": ["mean", "std", "skewness", "kurtosis", "slope"],

    "delta_mfcc": ["mean", "std", "kurtosis"],

    "section_single": ["mean", "std"],
    
    "section_mfcc":   ["mean", "std"],

}