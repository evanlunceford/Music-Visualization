from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

def display_spectogram(wav_file_dir: Path):
    waveform, sample_rate = librosa.load(wav_file_dir)

    # Short-time fourier transform
    stft = librosa.stft(waveform)

    spectogram = librosa.amplitude_to_db(np.abs(stft))

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(spectogram,
                             sr=sample_rate,
                             x_axis='time',
                             y_axis='log')
    
    plt.colorbar()
    plt.title(f"Spectogram for {wav_file_dir}")

    plt.tight_layout()

    plt.show()