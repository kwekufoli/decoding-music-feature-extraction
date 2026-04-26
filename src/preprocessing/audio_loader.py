import os
import librosa
import soundfile as sf
import numpy as np
import pyloudnorm as pyln


class AudioPreprocessor:
    def __init__(self, target_sr=22050, mono=True, normalize=True):
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize

    def load(self, filepath):
        """
        Load and resample audio.
        """
        audio, sr = librosa.load(
            filepath,
            sr=self.target_sr,
            mono=self.mono
        )
        loudness_features = self._measure_loudness(audio, sr)

        if self.normalize:
            audio = self._normalize(audio)

        return audio, self.target_sr, loudness_features

    def _normalize(self, audio):
        """
        Peak normalization.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms
        return audio

    def save_processed(self, audio, output_path):
        sf.write(output_path, audio, self.target_sr)

    def _measure_loudness(self, audio, sr):

        meter = pyln.Meter(sr)
        audio_64 = audio.astype(np.float64)

        try:
            integrated_lufs = meter.integrated_loudness(audio_64)
        except Exception:
            integrated_lufs = float("nan")

        # Dynamic range from short-term RMS curve in dB
        hop = sr // 4
        win = sr // 2
        rms_curve = np.array([
            np.sqrt(np.mean(audio[i:i + win] ** 2))
            for i in range(0, len(audio) - win, hop)
        ])
        rms_db = 20 * np.log10(rms_curve + 1e-10)
        dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))

        return {
            "integrated_lufs": float(integrated_lufs),
            "dynamic_range":   dynamic_range
        }