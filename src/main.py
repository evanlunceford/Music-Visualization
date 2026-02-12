import argparse
import sounddevice as sd

from music_analysis.cache.sound_cache import SoundCache
from music_analysis.graphs.live_spectogram import LiveSpectrogram
from music_analysis.music_analyzer import MusicAnalyzer
from music_analysis.utils.device_config import (
    resolve_device,
    list_devices,
    pick_device_interactive,
)


def parse_args():
    """
    Parses args for the following intents:
      1. Input audio setup
        a. List all audio devices (--list-devices)
        b. Pick an input audio device (--pick-device)
    """
    parser = argparse.ArgumentParser(description="Music Visualization")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--pick-device",
        action="store_true",
        help="Interactively select an audio input device",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_devices:
        print(list_devices())
        exit(0)

    if args.pick_device:
        device = pick_device_interactive()
    else:
        device = resolve_device()

    if device is not None:
        dev_info = sd.query_devices(device)
        print(f"Using audio device: [{device}] {dev_info['name']}")
    else:
        print("Using default audio input device")

    # Initialize cache and analyzer
    cache = SoundCache()
    analyzer = MusicAnalyzer(cache, device=device)

    # LIVE TEST FUNCTIONS
    # live_spectogram = LiveSpectrogram(device=device)
    # live_spectogram.run()

    # MUSIC ANALYZER EXAMPLES
    # Listen for 30 seconds then analyze structure
    # sections = analyzer.analyze_song_structure_live(listen_seconds=30.0)

    # Listen for 240 seconds then analyze structure
    # sections = analyzer.analyze_song_structure_live(listen_seconds=240.0)
    # print(sections)

    # Analyze existing cache data (no listening)
    # sections = analyzer.analyze_song_structure(lookback_seconds=300.0)
