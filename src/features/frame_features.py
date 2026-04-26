import numpy as np
import librosa


def compute_rms(audio, sr, frame_size=2048, hop_size=512):
    
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_size,
        hop_length=hop_size
    )
    
    return rms.flatten()

def compute_spectral_centroid(audio, sr, frame_size=2048, hop_size=512):
    
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size
    )
    
    return centroid.flatten()

def compute_spectral_bandwidth(audio, sr, frame_size=2048, hop_size=512):

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size
    )

    return bandwidth.flatten()

def compute_spectral_rolloff(audio, sr, frame_size=2048, hop_size=512):

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size
    )

    return rolloff.flatten()

def compute_zero_crossing_rate(audio, sr, frame_size=2048, hop_size=512):

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_size,
        hop_length=hop_size
    )

    return zcr.flatten()

def compute_spectral_flatness(audio, sr, frame_size=2048, hop_size=512):
    

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=frame_size,
        hop_length=hop_size
    )

    return flatness.flatten()

def compute_spectral_flux(audio, sr, frame_size=2048, hop_size=512):
    
    stft = np.abs(
        librosa.stft(
            audio,
            n_fft=frame_size,
            hop_length=hop_size
        )
    )

    diff = np.diff(stft, axis=1)

    # Positive flux — energy increases, captures onsets and attacks
    positive_flux = np.sum(np.maximum(0, diff), axis=0)

    # Negative flux — energy decreases, captures decays and releases
    negative_flux = np.sum(np.maximum(0, -diff), axis=0)

    return positive_flux, negative_flux


def compute_spectral_contrast(audio, sr, frame_size=2048, hop_size=512):
    

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size
    )

    return contrast

def compute_onset_strength(audio, sr, hop_size=512):
    """
    Compute onset strength over time.
    This measures sudden increases in spectral energy,
    which correspond to musical events such as drum hits or note attacks.
    """


    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_size
    )

    return onset_env

def compute_onset_density(audio, sr, hop_size=512):
    """
    Compute onset density for the audio signal.
    This measures how frequently musical events occur.
    """

    # Detect onset positions
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_size
    )

    # Convert number of onsets into density
    duration = librosa.get_duration(y=audio, sr=sr)

    onset_density = len(onset_frames) / duration

    return onset_density

def compute_tempo(audio, sr, hop_size=512):
    """
    Estimate the tempo (beats per minute) of the audio signal.
    """


    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=sr,
        hop_length=hop_size
    )

    return tempo

def compute_chroma(audio, sr, frame_size=2048, hop_size=512):
    """
    Compute chroma features over time.
    Chroma represents the strength of the twelve pitch classes.
    """


    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size
    )

    return chroma

def compute_tonal_entropy(chroma):
    """
    Compute tonal entropy over time from chroma features.
    Higher entropy means pitch energy is spread across many notes.
    Lower entropy means pitch energy is concentrated in a few notes.
    """

    # Normalize chroma so each frame sums to 1
    chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-10)

    # Compute entropy for each frame
    entropy = -np.sum(chroma_norm * np.log(chroma_norm + 1e-10), axis=0)

    return entropy

def compute_tonal_clarity(chroma):
    """
    Estimate tonal clarity by comparing the average chroma
    distribution to major and minor key templates.
    """

    # Average chroma across time
    chroma_mean = np.mean(chroma, axis=1)

    # Normalize distribution
    chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-10)

    # Major key template (Krumhansl profile)
    major_template = np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
        2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ])

    # Minor key template
    minor_template = np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
        2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])

    # Normalize templates
    major_template = major_template / np.sum(major_template)
    minor_template = minor_template / np.sum(minor_template)

    # Compute correlation
    major_score = np.dot(chroma_mean, major_template)
    minor_score = np.dot(chroma_mean, minor_template)

    tonal_clarity = max(major_score, minor_score)

    return tonal_clarity

def compute_harmonic_percussive_ratio(audio):
    """
    Compute the ratio between harmonic and percussive energy.
    Higher values indicate more percussive content relative to harmonic content.
    """

    # Separate harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(audio)

    # Compute energy of each component
    harmonic_energy = np.sum(harmonic ** 2)
    percussive_energy = np.sum(percussive ** 2)

    # Avoid division by zero
    ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-10)

    return ratio

def compute_mfccs(audio, sr, n_mfcc=20, frame_size=2048, hop_size=512):
    
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=frame_size,
        hop_length=hop_size
    )
    # mfccs shape: (n_mfcc, n_frames)
    
    delta_mfccs = librosa.feature.delta(mfccs)
    # Same shape, captures rate of change
    
    return mfccs, delta_mfccs

def compute_chord_features(audio, sr, frame_size=2048, hop_size=512):
    """
    Detect chord content from chroma using template matching.
    
    Returns a dict of scalar and curve features describing
    the harmonic character of the song.
    """

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr,
        n_fft=frame_size,
        hop_length=hop_size
    )

    # ── Build chord templates (12 keys x 4 qualities) ──────────────────
    # Intervals from root: major=[0,4,7], minor=[0,3,7],
    #                      dom7=[0,4,7,10], min7=[0,3,7,10]
    qualities = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "dom7":  [0, 4, 7, 10],
        "min7":  [0, 3, 7, 10],
    }

    templates = {}
    for root in range(12):
        for quality, intervals in qualities.items():
            template = np.zeros(12)
            for interval in intervals:
                template[(root + interval) % 12] = 1.0
            templates[(root, quality)] = template

    # ── Match each frame to best chord ─────────────────────────────────
    n_frames = chroma.shape[1]
    best_quality = []
    best_score   = np.zeros(n_frames)

    for f in range(n_frames):
        frame = chroma[:, f]
        norm  = np.linalg.norm(frame)

        if norm < 1e-6:
            # Silent frame — no chord
            best_quality.append("none")
            best_score[f] = 0.0
            continue

        frame_norm = frame / norm
        top_score  = -1.0
        top_quality = "none"

        for (root, quality), template in templates.items():
            t_norm = template / (np.linalg.norm(template) + 1e-10)
            score  = np.dot(frame_norm, t_norm)
            if score > top_score:
                top_score   = score
                top_quality = quality

        best_quality.append(top_quality)
        best_score[f] = top_score

    # ── Compute summary features ────────────────────────────────────────
    total_voiced = sum(1 for q in best_quality if q != "none")

    if total_voiced == 0:
        return {
            "chord_major_ratio":    0.0,
            "chord_minor_ratio":    0.0,
            "chord_dom7_ratio":     0.0,
            "chord_min7_ratio":     0.0,
            "chord_change_rate":    0.0,
            "chord_match_mean":     0.0,
            "chord_match_std":      0.0,
        }

    major_count = sum(1 for q in best_quality if q == "major")
    minor_count = sum(1 for q in best_quality if q == "minor")
    dom7_count  = sum(1 for q in best_quality if q == "dom7")
    min7_count  = sum(1 for q in best_quality if q == "min7")

    # Chord change rate — fraction of frames where quality changes
    changes = sum(
        1 for i in range(1, len(best_quality))
        if best_quality[i] != best_quality[i-1]
        and best_quality[i] != "none"
        and best_quality[i-1] != "none"
    )
    change_rate = changes / max(total_voiced - 1, 1)

    voiced_scores = best_score[best_score > 1e-6]

    return {
        "chord_major_ratio":  major_count / total_voiced,
        "chord_minor_ratio":  minor_count / total_voiced,
        "chord_dom7_ratio":   dom7_count  / total_voiced,
        "chord_min7_ratio":   min7_count  / total_voiced,
        "chord_change_rate":  change_rate,
        "chord_match_mean":   float(np.mean(voiced_scores)),
        "chord_match_std":    float(np.std(voiced_scores)),
    }