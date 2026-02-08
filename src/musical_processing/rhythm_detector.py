import librosa
from pathlib import Path


class RhythmDetector:
    """
    As of 2/7/2026:
    Currently static only detection is present
    """
    def __init__(
        self, 
        wav_file_dir: Path
    ):
        self.wav_file_dir = wav_file_dir


    def estimate_bpm(wav_file_dir) -> float:
        waveform, sampling_rate = librosa.load(wav_file_dir)
        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate)

        print(f"Tempo: {tempo}")
        print(f"\nBeat times: {beat_times}")
        return tempo
