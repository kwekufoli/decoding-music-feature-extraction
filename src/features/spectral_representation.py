import numpy as np
import librosa


def compute_stft(audio, frame_size=2048, hop_size=512):
    """
    Compute short-time frequency representation.
    
    Returns:
        magnitude_spectrogram: (num_frequency_bins, num_frames)
    """
    
    stft = librosa.stft(
        audio,
        n_fft=frame_size,
        hop_length=hop_size,
        window='hann'
    )
    
    magnitude = np.abs(stft)

    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    return log_magnitude
    