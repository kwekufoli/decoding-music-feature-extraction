import sys

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from preprocessing.audio_loader import AudioPreprocessor
from features.frame_features import (compute_mfccs, compute_rms, compute_spectral_centroid, compute_spectral_bandwidth, 
                                     compute_spectral_rolloff, compute_zero_crossing_rate, compute_chroma, 
                                     compute_onset_density, compute_harmonic_percussive_ratio, compute_onset_strength, 
                                     compute_spectral_contrast, compute_spectral_flatness, compute_spectral_flux, 
                                     compute_tempo, compute_tonal_clarity, compute_tonal_entropy, compute_chord_features)
from features.temporal_metric import (
    compute_mean,
    compute_std,
    compute_slope,
    compute_short_term_stability,
    compute_half_contrast, compute_percentile, compute_range, apply_selected_descriptors
)
from features.feature_schema import FEATURE_DESCRIPTOR_SCHEMA

def process_multi_curve_feature(feature_matrix, base_name, labels, schema_key):
    """
    Apply temporal descriptors to each curve in a multi-curve feature matrix.
    """
    descriptors = FEATURE_DESCRIPTOR_SCHEMA[schema_key]
    features = {}

    for i, label in enumerate(labels):
        curve = feature_matrix[i]
        name = f"{base_name}_{label}"
        feature_descriptors = apply_selected_descriptors(curve, name, descriptors)
        features.update(feature_descriptors)

    return features

N_SEGMENTS = 4

def extract_segment_features(audio, sr, n_segments=N_SEGMENTS,
                              frame_size=2048, hop_size=512):
    """
    Divide audio into n_segments equal quarters by sample count
    and compute a focused feature set within each segment.
    
    Features computed per segment:
        - RMS energy       (energy character of the section)
        - Spectral centroid (tonal brightness of the section)
        - Onset strength   (rhythmic intensity of the section)
        - MFCCs            (timbral texture of the section)
    
    Returns a flat dict of features keyed as:
        e.g. seg_1_rms_mean, seg_2_mfcc_3_std
    """
    features = {}
    segment_length = len(audio) // n_segments

    for i in range(n_segments):
        seg_label = f"seg_{i + 1}"
        start = i * segment_length
        # Last segment gets any remaining samples from rounding
        end = (i + 1) * segment_length if i < n_segments - 1 else len(audio)
        segment = audio[start:end]

        # --- Single curve features ---
        single_curves = {
            "rms":      compute_rms(segment, sr, frame_size, hop_size),
            "centroid": compute_spectral_centroid(segment, sr, frame_size, hop_size),
            "onset_strength": compute_onset_strength(segment, sr, hop_size),
        }

        descriptors = FEATURE_DESCRIPTOR_SCHEMA["section_single"]

        for feat_name, curve in single_curves.items():
            full_name = f"{seg_label}_{feat_name}"
            features.update(
                apply_selected_descriptors(curve, full_name, descriptors)
            )

        # --- MFCCs ---
        mfccs, _ = compute_mfccs(
            segment, sr,
            n_mfcc=20,
            frame_size=frame_size,
            hop_size=hop_size
        )

        mfcc_descriptors = FEATURE_DESCRIPTOR_SCHEMA["section_mfcc"]
        mfcc_labels = [str(i) for i in range(mfccs.shape[0])]

        for j, label in enumerate(mfcc_labels):
            curve = mfccs[j]
            full_name = f"{seg_label}_mfcc_{label}"
            features.update(
                apply_selected_descriptors(curve, full_name, mfcc_descriptors)
            )

    return features


def extract_features(audio_path, frame_size=2048, hop_size=512):

    Audio = AudioPreprocessor()

    features = {}
    audio, sr, loudness = Audio.load(audio_path)

    features["integrated_lufs"] = loudness["integrated_lufs"]
    features["dynamic_range"] = loudness["dynamic_range"]

    # Frame-level curves
    frame_features = {
        "rms": compute_rms(audio, sr, frame_size, hop_size),
        "centroid": compute_spectral_centroid(audio, sr, frame_size, hop_size),
        "bandwidth": compute_spectral_bandwidth(audio, sr, frame_size, hop_size),
        "rolloff": compute_spectral_rolloff(audio, sr, frame_size, hop_size),
        "flatness": compute_spectral_flatness(audio, sr, frame_size, hop_size),
        "zcr": compute_zero_crossing_rate(audio, sr, frame_size, hop_size),
        "onset_strength": compute_onset_strength(audio, sr, hop_size)
    }

    # Apply temporal descriptors
    for name, curve in frame_features.items():
        descriptors = FEATURE_DESCRIPTOR_SCHEMA[name]
        feature_descriptors = apply_selected_descriptors(curve, name, descriptors)
        features.update(feature_descriptors)

    positive_flux, negative_flux = compute_spectral_flux(audio, sr, frame_size, hop_size)

    features.update(
        apply_selected_descriptors(
            positive_flux, "flux_positive",
            FEATURE_DESCRIPTOR_SCHEMA["flux_positive"]
        )
    )
    features.update(
        apply_selected_descriptors(
            negative_flux, "flux_negative",
            FEATURE_DESCRIPTOR_SCHEMA["flux_negative"]
        )
    )

    # Chroma (multi-curve feature)
    chroma = compute_chroma(audio, sr, frame_size, hop_size)

    chroma_labels = [
        "C", "Cs", "D", "Ds", "E", "F",
        "Fs", "G", "Gs", "A", "As", "B"
    ]

    chroma_features = process_multi_curve_feature(
        chroma,
        "chroma",
        chroma_labels, "chroma"
    )

    features.update(chroma_features)

    chord_features = compute_chord_features(audio, sr, frame_size, hop_size)
    features.update(chord_features)

    contrast = compute_spectral_contrast(audio, sr, 
                                      frame_size, 
                                      hop_size)

    contrast_labels = [
        "sub_bass", "bass", "low_mid", 
        "mid", "upper_mid", "presence", "brilliance"
    ]

    contrast_features = process_multi_curve_feature(
        contrast, "contrast", contrast_labels, "contrast"
    )

    features.update(contrast_features)

    mfccs, delta_mfccs = compute_mfccs(audio, sr, n_mfcc=20, frame_size=frame_size, hop_size=hop_size)

    mfcc_labels = [str(i) for i in range(mfccs.shape[0])]
    delta_labels = [str(i) for i in range(delta_mfccs.shape[0])]

    mfcc_features = process_multi_curve_feature(mfccs, "mfcc", mfcc_labels, "mfcc")
    delta_features = process_multi_curve_feature(delta_mfccs, "delta_mfcc", delta_labels, "delta_mfcc")

    features.update(mfcc_features)
    features.update(delta_features)

    # Harmonic descriptors
    tonal_entropy = compute_tonal_entropy(chroma)
    entropy_descriptors = FEATURE_DESCRIPTOR_SCHEMA["tonal_entropy"]

    entropy_feature_descriptors = apply_selected_descriptors(tonal_entropy, "tonal_entropy", entropy_descriptors)

    features.update(entropy_feature_descriptors)

    features["tonal_clarity"] = compute_tonal_clarity(chroma)

    # Clip-level features
    features["harmonic_percussive_ratio"] = compute_harmonic_percussive_ratio(audio)

    features["onset_density"] = compute_onset_density(audio, sr)

    features["tempo"] = compute_tempo(audio, sr)

    # --- Segment-level features ---
    segment_features = extract_segment_features(audio, sr, frame_size=frame_size, hop_size=hop_size)
    features.update(segment_features)

    return features