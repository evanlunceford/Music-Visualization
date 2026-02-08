from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt


class StaticSpectogram:
    """
    Simple class to display a spectogram of pre-processesed audio
    """
    def __init__(self, wav_file_dir: Path):
        self.wav_file_dir = wav_file_dir

    def display_spectogram(self):
        waveform, sample_rate = librosa.load(self.wav_file_dir)

        # Short-time fourier transform
        stft = librosa.stft(waveform)

        spectogram = librosa.amplitude_to_db(np.abs(stft))

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(spectogram,
                                sr=sample_rate,
                                x_axis='time',
                                y_axis='log')
        
        plt.colorbar()
        plt.title(f"Spectogram for {self.wav_file_dir}")

        plt.tight_layout()

        plt.show()