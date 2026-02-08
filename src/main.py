from graphs.live_spectogram import LiveSpectrogram
from graphs.static_spectogram import StaticSpectogram
from musical_processing.chord_detector import ChordDetector
from musical_processing.rhythm_detector import RhythmDetector


TEST_AUDIO_DIR = "./music_samples/About-You-1975.wav"


if __name__ == "__main__":
    # Live objects
    chord_detector = ChordDetector()
    live_spectogram = LiveSpectrogram()


    # To see list of devices, run print(sd.query_devices())
    chord_detector.live_chords()

    # Static objects
    rhythm_detector = RhythmDetector(TEST_AUDIO_DIR)
    static_spectogram = StaticSpectogram(TEST_AUDIO_DIR)