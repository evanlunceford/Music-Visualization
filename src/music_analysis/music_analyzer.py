import time

from music_analysis.utils.audio_stream import AudioStream
from music_analysis.cache.sound_cache import SoundCache
from music_analysis.song_structure.song_structure_analyzer import SongStructureAnalyzer
from music_analysis.input_detection.chord_detector import ChordDetector
from music_analysis.input_detection.rhythm_detector import RhythmDetector


class MusicAnalyzer:
    """
    Main object for providing output about live audio

    Required arguments:
      1. cache: SoundCache
    """

    def __init__(
        self,
        cache: SoundCache,
        sample_rate: int = 44100,
        block_s: float = 0.02,
        device: int | None = None,
    ):
        self.cache = cache
        self.sr = sample_rate
        self.block_s = block_s

        self.audio_stream = AudioStream(sr=sample_rate, block_s=block_s, device=device)

        # Init detectors because they are exclusively used in this object
        self.chord_detector = ChordDetector(sr=sample_rate, update_s=block_s)
        self.rhythm_detector = RhythmDetector(sr=sample_rate, block_s=block_s)

    def analyze_song_structure_live(
        self,
        listen_seconds: float = 60.0,
        lookback_seconds: float = 300.0,
        bin_seconds: float = 2.0,
    ) -> list[dict]:
        """
        Listens to audio input, detects chords and BPM, caches results,
        then analyzes the song structure.

        Args:
            listen_seconds: How long to listen for audio input
            lookback_seconds: How far back to analyze in the cache
            bin_seconds: Bin size for song structure analysis

        Returns:
            List of labeled song sections (verse, chorus, bridge, etc.)
        """
        print(f"Listening for {listen_seconds}s... (analyzing song structure)")
        start_time = time.time()

        with self.audio_stream as stream:
            for block in stream.blocks():
                if (time.time() - start_time) >= listen_seconds:
                    break

                if block is None:
                    continue

                # Process chord detection
                chord = self.chord_detector.process_block(block)

                # Process rhythm detection
                self.rhythm_detector.push_audio(block)
                bpm = self.rhythm_detector.update_bpm()

                # Add to cache when we have new data
                current_chord = chord if chord else self.chord_detector.last_reported
                current_bpm = bpm if bpm else 0.0

                if current_chord and current_chord != "silence":
                    self.cache.add(
                        chord=current_chord,
                        bpm=current_bpm if current_bpm else 0.0,
                    )

        print("Analyzing song structure...")
        return self.analyze_song_structure(lookback_seconds=lookback_seconds, bin_seconds=bin_seconds)

    def analyze_song_structure(
        self,
        lookback_seconds: float = 300.0,
        bin_seconds: float = 2.0,
    ) -> list[dict]:
        """
        Analyzes song structure from existing cache data.

        Args:
            lookback_seconds: How far back to analyze in the cache
            bin_seconds: Bin size for song structure analysis

        Returns:
            List of labeled song sections (verse, chorus, bridge, etc.)
        """
        structure_analyzer = SongStructureAnalyzer(self.cache, bin_seconds=bin_seconds)
        return structure_analyzer.analyze_structure(lookback_seconds=lookback_seconds)
