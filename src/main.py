from graphs.live_spectogram import LiveSpectrogram
from graphs.static_spectogram import StaticSpectogram
from musical_processing.chord_detector import ChordDetector
from musical_processing.rhythm_detector import RhythmDetector
from musical_processing.static.static_rhythm_detector import StaticRhythmDetector

# Switch to a wav file you have downloaded
TEST_AUDIO_DIR = "./music_samples/About-You-1975.wav"


if __name__ == "__main__":
    # Live objects
    chord_detector = ChordDetector()
    live_spectogram = LiveSpectrogram()
    rhythm_detector = RhythmDetector()


    # To see list of devices, run print(sd.query_devices())

    # LIVE TEST FUNCTIONS
    # chord_detector.live_chords()
    # live_spectogram.run()
    # rhythm_detector.listen_bpm()


    # Static objects
    static_rhythm_detector = StaticRhythmDetector(TEST_AUDIO_DIR)
    static_spectogram = StaticSpectogram(TEST_AUDIO_DIR)