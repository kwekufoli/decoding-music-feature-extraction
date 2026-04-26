from features.feature_schema import FEATURE_DESCRIPTOR_SCHEMA
from features.feature_extractor import N_SEGMENTS

def _stats(schema_key):
    """Return the stat suffixes for a given schema key."""
    return FEATURE_DESCRIPTOR_SCHEMA[schema_key]


def _expand(base_name, labels, schema_key):
    """
    Expand a multi-curve feature into its full list of column names,
    exactly matching what process_multi_curve_feature produces.
    e.g. _expand("chroma", ["C","Cs"], "chroma_pitch")
         -> ["chroma_C_mean", "chroma_C_std", "chroma_Cs_mean", ...]
    """
    return [
        f"{base_name}_{label}_{stat}"
        for label in labels
        for stat in _stats(schema_key)
    ]


def _expand_scalar(base_name, schema_key):
    """
    Expand a single-curve feature into its full list of column names.
    e.g. _expand_scalar("rms", "rms") -> ["rms_mean", "rms_std", ...]
    """
    return [f"{base_name}_{stat}" for stat in _stats(schema_key)]


# --- Label sets (single source of truth) ---

_MFCC_LABELS        = [str(i) for i in range(19)]  # coeff 0 dropped
_DELTA_LABELS       = [str(i) for i in range(19)]
_CHROMA_LABELS      = ["C", "Cs", "D", "Ds", "E", "F",
                        "Fs", "G", "Gs", "A", "As", "B"]
_CONTRAST_LABELS    = ["sub_bass", "bass", "low_mid",
                        "mid", "upper_mid", "presence", "brilliance"]
_SEGMENT_LABELS       = [f"seg_{i}" for i in range(1, N_SEGMENTS + 1)]
_SEGMENT_SINGLE_FEATS = ["rms", "centroid", "onset_strength"]
_SEGMENT_MFCC_LABELS  = [str(i) for i in range(19)]


COMPARISON_FEATURES = [
    # Timbral
    *_expand("mfcc",       _MFCC_LABELS,   "mfcc"),
    *_expand("delta_mfcc", _DELTA_LABELS,  "delta_mfcc"),

    # Rhythmic
    "tempo",
    "onset_density",
    *_expand_scalar("onset_strength", "onset_strength"),

    # Harmonic
    *_expand("chroma", _CHROMA_LABELS, "chroma"),
    # Section-level timbral (MFCCs per segment)
    *[
        f"{seg}_{feat}_{stat}"
        for seg in _SEGMENT_LABELS
        for feat in [f"mfcc_{label}" for label in _SEGMENT_MFCC_LABELS]
        for stat in _stats("section_mfcc")
    ],
]

INTERPRETATION_FEATURES = [
    # Energy
    "integrated_lufs",
    "dynamic_range",
    *_expand_scalar("rms",            "rms"),
    *_expand_scalar("centroid",       "centroid"),
    *_expand_scalar("bandwidth",      "bandwidth"),
    *_expand_scalar("rolloff",        "rolloff"),
    *_expand_scalar("flatness",       "flatness"),
    *_expand_scalar("flux_positive", "flux_positive"),
    *_expand_scalar("flux_negative", "flux_negative"),

    # Spectral contrast per band
    *_expand("contrast", _CONTRAST_LABELS, "contrast"),

    # Tonal
    "tonal_clarity",
    "harmonic_percussive_ratio",
    *_expand_scalar("tonal_entropy", "tonal_entropy"),
    # Section-level energy, brightness, rhythmic intensity
    *[
        f"{seg}_{feat}_{stat}"
        for seg in _SEGMENT_LABELS
        for feat in _SEGMENT_SINGLE_FEATS
        for stat in _stats("section_single")
    ],
    "tempo",
    "onset_density",
    "tonal_clarity",
    "harmonic_percussive_ratio",
    "integrated_lufs",
    "dynamic_range",
    "chord_major_ratio",
    "chord_minor_ratio",
    "chord_dom7_ratio",
    "chord_min7_ratio",
    "chord_change_rate",
    "chord_match_mean",
    "chord_match_std",
]