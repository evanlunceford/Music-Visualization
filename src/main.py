from music_analysis.cache.sound_cache import SoundCache
from music_analysis.graphs.live_spectogram import LiveSpectrogram
from music_analysis.music_analyzer import MusicAnalyzer


if __name__ == "__main__":
    # To see list of devices, run print(sd.query_devices())

    # Initialize cache and analyzer
    cache = SoundCache()
    analyzer = MusicAnalyzer(cache)

    # LIVE TEST FUNCTIONS
    # live_spectogram = LiveSpectrogram()
    # live_spectogram.run()

    # MUSIC ANALYZER EXAMPLES
    # Listen for 30 seconds then analyze structure
    # sections = analyzer.analyze_song_structure_live(listen_seconds=30.0)

    # Listen for 240 seconds then analyze structure
    sections = analyzer.analyze_song_structure_live(listen_seconds=240.0)   
    print(sections)

    # Analyze existing cache data (no listening)
    # sections = analyzer.analyze_song_structure(lookback_seconds=300.0)
