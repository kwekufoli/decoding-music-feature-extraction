import numpy as np
import librosa


def frame_audio(audio, frame_size=2048, hop_size=512):
    """
    Slice waveform into overlapping frames.
    Returns array of shape (num_frames, frame_size)
    """
    frames = librosa.util.frame(
        audio,
        frame_length=frame_size,
        hop_length=hop_size
    )
    
    # librosa returns shape (frame_size, num_frames)
    return frames.T