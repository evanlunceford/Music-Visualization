from music_analysis.cache.sound_cache import SoundCache
from music_analysis.graphs.live_spectogram import LiveSpectrogram
from music_analysis.song_structure.song_structure_analyzer import SongStructureAnalyzer
from sound_processing.input_detection.chord_detector import ChordDetector
from sound_processing.input_detection.rhythm_detector import RhythmDetector



if __name__ == "__main__":
    # Live objects
    chord_detector = ChordDetector()
    live_spectogram = LiveSpectrogram()
    rhythm_detector = RhythmDetector()


    # I still need to add the code to connect the detectors to the sound cache
    sound_cache = SoundCache()
    sound_structure_analyzer = SongStructureAnalyzer()


    # To see list of devices, run print(sd.query_devices())

    # LIVE TEST FUNCTIONS
    # chord_detector.live_chords()
    # live_spectogram.run()
    # rhythm_detector.listen_bpm()
