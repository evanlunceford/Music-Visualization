import librosa
import matplotlib.pyplot as plt

TEST_AUDIO_DIR = "./music_samples/About-You-1975.wav"

def estimate_bpm(wav_file_dir):
    waveform, sampling_rate = librosa.load(wav_file_dir)
    tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate)
    print(f"Tempo: {tempo}")
    print(f"\nBeat times: {beat_times}")

def display_waveform(wav_file_dir):
    y, sr = librosa.load(wav_file_dir)
    plt.figure(figsize=(10,4))
    librosa.display.waveshow(y, sr=sr, color='blue')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


if __name__ == "__main__":
    estimate_bpm(TEST_AUDIO_DIR)
    